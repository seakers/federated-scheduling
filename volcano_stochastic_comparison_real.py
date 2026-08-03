"""
Volcano Workflow: Stochastic vs Deterministic Comparison
Using real satellite constellations from TLE files and GVP database

RUN MODES
---------
1) Single run (recommended -- one scheduler, one seed, then exit):
     python volcano_comparison.py --scheduler stochastic_log --seed 42 \
            --start 2026-07-22T00:00:00 --results-dir results/campaign_A

   Process exit is the ONLY hard guarantee that memory is reclaimed, so a
   campaign built from these cannot OOM by accumulation. Use run_campaign.py
   to drive them and aggregate at the end.

2) Aggregate an existing results dir (reads run_*.json, writes CSVs + plots):
     python volcano_comparison.py --aggregate --results-dir results/campaign_A

3) Legacy all-in-one-process loop (same behaviour as before, plus cleanup):
     python volcano_comparison.py

IMPORTANT: --start pins SIMULATION_START. Every process in one campaign MUST
share the same value; otherwise the orbital geometry (hence the passes) differs
between runs and the seed-paired comparison is meaningless.

Metrics (see fame_metrics.py):
  PRIMARY   : task/group completion rate, realized quality, net utility
  COST AXIS : total realized cost, submissions, accepted, executed
  REDUNDANCY: TRUE passes-per-task from broker._requests
  SECONDARY : rejection rate, replans (diagnostic only, never a headline)
"""

import argparse
import copy
import datetime as dt
import gc
import glob
import json
import os
import random
from typing import Callable

import numpy as np
import matplotlib

from fame_workflow_stochastic import StochasticTimeline
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import pyorbital
import pyorbital.orbital

# Load environment variables
load_dotenv()

# Import FAME components
from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus
from fame_broker import Broker
from fame_workflow import *
from fame_demand_model import DemandField, DemandFieldConfig

# Import Volcano Domain Machinery
from volcano_utils import (
    load_volcano_locations_from_database,
    create_volcano_workflow,
    register_volcano_phenomena,
)

# demand-based metrics + paired analysis + frontier
from fame_metrics import compute_metrics_v3, paired_summary, plot_cost_frontier

# Shared benchmarking utilities (satellite loading, simulation runner, memory)
from benchmarking_utils import (
    GROUND_STATIONS,
    load_satellites_once as _load_satellites_once_shared,
    run_simulation_forward as _run_simulation_forward_shared,
    rss_gb,
)

# ============================ Configuration =================================
# SIMULATION_START is a module global because the geometry helpers below read
# it. In single-run mode it is overwritten from --start BEFORE anything uses
# it, so all processes in a campaign share identical orbital geometry.
SIMULATION_START = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

lookahead_horizon_h = 12
FOLLOW_UP_INTERVAL_H = 3
MAX_SOLVER_TIME_S = 70
MAX_NUM_INSTANCES = 5
NUM_MC_RUNS = 3

# === COST CONFIGURATION ===
TAX_RATE = 0.0              # Legacy per-booking tax (disabled)
SUBMISSION_COST = 0.05      # Unconditional booking submission overhead
EXEC_COST = 0.2             # Conditional execution cost if accepted

# === PROBABILITY CONFIGURATION ===
# Acceptance probability range: constellation rejects a booking when its demand
# is high, accepts when quiet.  DemandField maps demand → p_accept in [P_ACC_MIN, P_ACC_MAX].
P_ACC_MIN = 0.70            # Minimum acceptance probability (high-demand / congested)
P_ACC_MAX = 1.0           # Maximum acceptance probability (low-demand / quiet)

# Execution probability range: even an accepted pass may fail (cloud cover, sensor issue).
# The function maps look-angle → p_exec in [P_EXEC_MIN, P_EXEC_MAX].
# At nadir (best geometry) → P_EXEC_MAX; at worst geometry → P_EXEC_MIN.
P_EXEC_MIN = 1.0           # Minimum execution probability (worst geometry)
P_EXEC_MAX = 1.0           # Maximum execution probability (best geometry)

SCHEDULERS = ['greedy','stochastic_log', 'deterministic', 'random']
#SCHEDULERS = ['stochastic_log','greedy']

# Set to True to cancel inferior pending passes once a better/sufficient one succeeds.
# Only applies to stochastic_log (redundant scheduling). Saves execution cost at the
# price of reduced quality diversity. Has no effect on greedy/deterministic/random.
ENABLE_CANCELLATIONS = True


def load_satellites_once() -> list:
    """Load the full LEO fleet — delegated to benchmarking_utils for deduplication."""
    return _load_satellites_once_shared(SIMULATION_START, lookahead_horizon_h)


def create_world_and_constellations(cached_satellites: list[Satellite],
                                     volcano_locations: list[Location] = None,
                                     demand_field: DemandField = None,
                                     execution_probability_function=None):
    """
    Creates fresh simulation scopes using copied pre-cached orbital models.
    Registers ground-truth physical Eruption and dynamic Plume phenomena in the world.
    """
    local_satellites = copy.deepcopy(cached_satellites)
    world = World(satellites=local_satellites)
    world.time = SIMULATION_START

    if volcano_locations is not None:
        min_time = SIMULATION_START
        max_time = SIMULATION_START + dt.timedelta(hours=lookahead_horizon_h)
        register_volcano_phenomena(world, volcano_locations, min_time, max_time)

    planet_sats = [s for s in local_satellites if any(x in s.name.upper() for x in ["SKYSAT", "PELICAN", "TANAGER"])]
    umbra_sats = [s for s in local_satellites if "UMBRA" in s.name.upper()]
    capella_sats = [s for s in local_satellites if "CAPELLA" in s.name.upper() or "ACADIA" in s.name.upper()]
    loft_sats = [s for s in local_satellites if "LOFT" in s.name.upper() or "YAM" in s.name.upper()]
    ubotica_sats = [s for s in local_satellites if "UBOTICA" in s.name.upper() or "HAMMER" in s.name.upper() or "ACCENTURE" in s.name.upper()]
    mission_control_sats = [s for s in local_satellites if "PERSISTENCE" in s.name.upper() or "LEMUR" in s.name.upper()]
    aerospace_sats = [s for s in local_satellites if "AEROCUBE" in s.name.upper()]
    iceye_sats = [s for s in local_satellites if "ICEYE" in s.name.upper()]

    if demand_field is not None:
        _sim_acc_fn = demand_field.make_simulator_acceptance_function()
    else:
        _sim_acc_fn = None

    def _make_scheduler(sats, name, legacy_p):
        return ConstellationGroundScheduler(
            satellites=sats,
            ground_stations=GROUND_STATIONS,
            world=world,
            name=name,
            acceptance_probability=legacy_p,
            acceptance_probability_function=_sim_acc_fn,
            execution_probability_function=execution_probability_function,
        )

    scheduler_planet          = _make_scheduler(planet_sats,          "Planet",          0.40)
    scheduler_umbra           = _make_scheduler(umbra_sats,           "Umbra",           0.60)
    scheduler_capella         = _make_scheduler(capella_sats,         "Capella",         0.90)
    scheduler_loft            = _make_scheduler(loft_sats,            "LOFT",            0.71)
    scheduler_ubotica         = _make_scheduler(ubotica_sats,         "Ubotica",         0.50)
    scheduler_mission_control = _make_scheduler(mission_control_sats, "Mission Control", 0.64)
    scheduler_aerospace       = _make_scheduler(aerospace_sats,       "AC",              0.67)
    scheduler_iceye           = _make_scheduler(iceye_sats,           "ICEYE",           0.74)

    all_constellations = [
        scheduler_planet, scheduler_umbra, scheduler_capella, scheduler_loft,
        scheduler_ubotica, scheduler_mission_control, scheduler_aerospace, scheduler_iceye
    ]

    for constellation in all_constellations:
        world.add_constellation(constellation)

    return world, all_constellations


# ======================= shared run-time helpers ============================

def _rss_gb():
    """Peak resident set size in GB — delegates to benchmarking_utils."""
    return rss_gb()


def run_simulation_forward(world, max_safety_limit=40000):
    """Delegates to benchmarking_utils.run_simulation_forward."""
    return _run_simulation_forward_shared(world, max_ticks=max_safety_limit)


def build_demand_field(volcano_db_locations, min_time):
    """Demand field: single source of truth for simulator and planner."""
    _demand_cfg = DemandFieldConfig(use_constant_probability=False,
                                    p_min=P_ACC_MIN, p_max=P_ACC_MAX)
    _horizon_s = lookahead_horizon_h * 3600.0
    demand_field = DemandField(config=_demand_cfg, reference_time=min_time, horizon_s=_horizon_s)

    for vloc in volcano_db_locations:
        demand_field.add_spike(vloc.lat_deg, vloc.lon_deg, min_time)

    _all_constellation_names = ["Planet", "Umbra", "Capella", "LOFT",
                                "Ubotica", "Mission Control", "AC", "ICEYE"]
    print("[DemandField] Precomputing demand trajectories...")
    demand_field.precompute(_all_constellation_names)
    print("[DemandField] Precompute complete.")

    _cost_proxy = {
        "Planet": 0.1, "Ubotica": 0.15, "LOFT": 0.2, "Mission Control": 0.25,
        "ICEYE": 0.3, "AC": 0.35, "Umbra": 0.5, "Capella": 0.6,
    }
    demand_field.check_cost_reliability_tension(_cost_proxy)
    return demand_field


def make_probability_functions(demand_field):
    def acceptance_prob_function(constrained_request, satellite, obs_pass):
        return demand_field.make_acceptance_prob_function()(constrained_request, satellite, obs_pass)

    def execution_prob_function(constrained_request, satellite, obs_opp):
        # obs_opp is an ObservationOpportunity when called from the simulator,
        # and an ObservationPass when called from the MILP solver — handle both.
        opp = obs_opp.highest if hasattr(obs_opp, 'highest') else obs_opp
        look_angle = abs(90.0 - opp.look_angle_dec_deg)
        # Linear interpolation: nadir (look_angle=0) → P_EXEC_MAX, worst (look_angle=90) → P_EXEC_MIN
        t = look_angle / 90.0
        execution_prob = P_EXEC_MAX - t * (P_EXEC_MAX - P_EXEC_MIN)
        return max(P_EXEC_MIN, min(P_EXEC_MAX, execution_prob))

    return acceptance_prob_function, execution_prob_function


def _compute_quality_rank_distribution(workflow_graph, broker, ObservationStatus):
    """
    For each completed task, rank all submitted passes by descending quality and
    find the rank (1-indexed) of the pass that actually succeeded (DATA_RECEIVED).
    Returns a dict {rank: count} for ranks 1..MAX_NUM_INSTANCES, plus 'unranked'.
    """
    tasks = list(workflow_graph.nodes())
    obsreq_to_task = {t.observation_request: t for t in tasks}
    reqs = broker._requests

    rank_counts = {}
    for task in tasks:
        task_rows = reqs[reqs['request'] == task.observation_request].copy()
        if task_rows.empty:
            continue

        # Find the successful pass
        success_rows = task_rows[task_rows['status'] == ObservationStatus.DATA_RECEIVED]
        if success_rows.empty:
            continue

        # Compute quality for each submitted pass and sort descending
        def _q(row):
            rp = row['requested_pass']
            if rp is None:
                return -1.0
            try:
                return task.rewarder(rp.highest)
            except Exception:
                return 0.0

        task_rows = task_rows[task_rows['requested_pass'].notna()].copy()
        task_rows['_q'] = task_rows.apply(_q, axis=1)
        sorted_passes = task_rows.sort_values('_q', ascending=False)['requested_pass'].tolist()

        winning_pass = success_rows.iloc[0]['requested_pass']
        try:
            rank = next(i + 1 for i, rp in enumerate(sorted_passes) if rp is winning_pass)
        except StopIteration:
            rank = 'unranked'

        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    return rank_counts


def run_one_scheduler(scheduler, seed, cached_satellites, volcano_db_locations,
                      demand_field, min_time, max_time, results_dir,
                      plot_schedule=True):
    """
    Build world + workflow + broker for ONE scheduler and ONE seed, run the
    simulation, compute metrics, and write run_seed<seed>_<scheduler>.json.
    """
    plots_dir = os.path.join(results_dir, "plots", scheduler)
    os.makedirs(plots_dir, exist_ok=True)

    acceptance_prob_function, execution_prob_function = make_probability_functions(demand_field)

    print(f"\n  Running {scheduler} (seed {seed})...  [RSS {_rss_gb():.2f} GB]")
    random.seed(seed)
    np.random.seed(seed)

    # ✅ PASS VOLCANO LOCATIONS TO REGISTER ERUPTIONS & PLUMES IN WORLD
    world, constellations = create_world_and_constellations(
        cached_satellites,
        volcano_locations=volcano_db_locations,
        demand_field=demand_field,
        execution_probability_function=execution_prob_function,
    )
    
    # ✅ CREATE VOLCANO WORKFLOW WITH DUAL BRANCHES AND PLUME RETARGETING
    workflow = create_volcano_workflow(volcano_db_locations, min_time, max_time, lookahead_horizon_h=lookahead_horizon_h)
    
    broker = Broker(constellations=constellations, world=world,
                    name=f"Broker-{scheduler}")
    broker.add_workflow(workflow)
    world.add_broker(broker)

    m = None
    try:
        if scheduler == 'stochastic_log':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=True, use_stochastic=True,
                stochastic_formulation="log_linearized",
                acceptance_probability_function=acceptance_prob_function,
                execution_probability_function=execution_prob_function,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_rate=EXEC_COST,
                tax_rate=TAX_RATE,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=False,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                enable_cancellations=ENABLE_CANCELLATIONS,
            )
        elif scheduler == 'deterministic':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=True, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=False, tax_rate=TAX_RATE,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_rate=EXEC_COST
            )
        elif scheduler == 'greedy':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=False, tax_rate=TAX_RATE,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_rate=EXEC_COST,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir
            )
        elif scheduler == 'random':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=False, tax_rate=TAX_RATE,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_rate=EXEC_COST,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                use_random=True,
                random_seed=seed,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        run_simulation_forward(world)

        m = compute_metrics_v3(
            broker._workflow_graph, broker, ObservationStatus,
            submission_cost_rate=SUBMISSION_COST, execution_cost_rate=EXEC_COST,
            verbose=True,
        )
        m['scheduler'] = scheduler
        m['seed'] = seed
        m['tax_rate'] = TAX_RATE
        m['max_num_instances'] = MAX_NUM_INSTANCES
        m['sim_start'] = SIMULATION_START.isoformat()

        # Quality-rank distribution: for each completed task, find what rank (1=best)
        # the pass that actually succeeded was among all submitted passes for that task.
        m['quality_rank_distribution'] = _compute_quality_rank_distribution(
            broker._workflow_graph, broker, ObservationStatus
        )

        run_file = os.path.join(results_dir, f"run_seed{seed:04d}_{scheduler}.json")
        with open(run_file, 'w') as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  [Saved] {run_file}")
    except Exception as e:
        print(f"    Broker Error ({scheduler}, seed {seed}): {e}")

    try:
        del broker, workflow, world, constellations
    except Exception:
        pass
    plt.close('all')
    gc.collect()
    print(f"  [Mem] after {scheduler} seed {seed}: peak RSS {_rss_gb():.2f} GB")
    return m


# ============================== aggregation =================================

def _bootstrap_ci(vals, iters=10000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float('nan'), float('nan'))
    idx = np.random.randint(0, len(vals), size=(iters, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def plot_utility(records, schedulers, out_path, baseline='deterministic'):
    """
    Three-panel utility figure:
      (a) mean net utility per scheduler with bootstrap 95% CI
      (b) paired per-seed utility (each line = one seed across schedulers)
      (c) paired utility differences vs baseline, per seed, with mean +/- CI
    """
    df = pd.DataFrame(records)
    if df.empty or 'utility' not in df.columns:
        print("[Utility plot] no data; skipped.")
        return

    present = [s for s in schedulers if s in set(df['scheduler'])]
    if not present:
        print("[Utility plot] no matching schedulers; skipped.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    # (a) means with bootstrap CI
    means, los, his = [], [], []
    for s in present:
        v = df[df['scheduler'] == s]['utility'].values.astype(float)
        mu = float(np.mean(v)) if len(v) else float('nan')
        lo, hi = _bootstrap_ci(v)
        means.append(mu)
        los.append(mu - lo if np.isfinite(lo) else 0.0)
        his.append(hi - mu if np.isfinite(hi) else 0.0)
    colors = ['#888888', '#4C72B0', '#DD8452', '#55A868'][:len(present)]
    ax1.bar(range(len(present)), means, yerr=[los, his], capsize=5, color=colors)
    ax1.set_xticks(range(len(present)))
    ax1.set_xticklabels(present, rotation=15)
    ax1.set_ylabel("Net utility (realized quality - cost)")
    ax1.set_title("(a) Mean utility, bootstrap 95% CI")
    ax1.grid(alpha=0.3, axis='y')

    # (b) paired per-seed lines
    pivot = pd.DataFrame()
    if 'seed' in df.columns:
        pivot = df.pivot_table(index='seed', columns='scheduler', values='utility')
        pivot = pivot[[c for c in present if c in pivot.columns]]
    if not pivot.empty:
        for _, row in pivot.iterrows():
            ax2.plot(range(len(row)), row.values, '-o', alpha=0.45,
                     linewidth=1, markersize=4)
        ax2.plot(range(pivot.shape[1]), pivot.mean(axis=0).values, '-s',
                 color='k', linewidth=2.5, markersize=8, label='mean')
        ax2.legend()
        ax2.set_xticks(range(pivot.shape[1]))
        ax2.set_xticklabels(list(pivot.columns), rotation=15)
    ax2.set_ylabel("Net utility")
    ax2.set_title("(b) Paired by seed (each line = one seed)")
    ax2.grid(alpha=0.3)

    # (c) paired differences vs baseline
    if not pivot.empty and baseline in pivot.columns:
        data, labels = [], []
        for c in [x for x in pivot.columns if x != baseline]:
            d = (pivot[c] - pivot[baseline]).dropna().values
            if len(d):
                data.append(d)
                labels.append(f"{c}\n- {baseline}")
        if data:
            ax3.boxplot(data, labels=labels)
            for i, d in enumerate(data, start=1):
                ax3.scatter(np.full(len(d), i) + np.random.uniform(-0.06, 0.06, len(d)),
                            d, alpha=0.6, s=22, zorder=3)
                mu = float(np.mean(d))
                lo, hi = _bootstrap_ci(d)
                if np.isfinite(lo):
                    ax3.errorbar([i], [mu], yerr=[[mu - lo], [hi - mu]], fmt='D',
                                 color='crimson', capsize=6, zorder=4)
            ax3.axhline(0, color='k', linestyle='--', linewidth=1)
            ax3.set_ylabel(f"delta utility vs {baseline}")
            ax3.set_title("(c) Paired differences (red = mean, 95% CI)")
            ax3.grid(alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, f"no {baseline} runs to pair against",
                 ha='center', va='center')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Utility plot] saved to {out_path}")


def load_records(results_dir):
    """Read every run_*.json in results_dir into a list of dicts."""
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "run_*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
            # json.dump used default=str, so coerce numeric-looking strings back
            for k, v in list(r.items()):
                if isinstance(v, str) and k not in ('scheduler', 'sim_start'):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            records.append(r)
        except Exception as e:
            print(f"[Warning] could not read {path}: {e}")
    print(f"[Aggregate] loaded {len(records)} run files from {results_dir}")
    return records


def aggregate(results_dir, records=None):
    """Summaries, paired stats, frontier, utility and completion plots."""
    if records is None:
        records = load_records(results_dir)
    if not records:
        print("[Warning] No records found; nothing to summarize.")
        return []

    df = pd.DataFrame(records)
    metrics_csv = os.path.join(results_dir, "metrics_v2.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"\n[Per-Run] Saved all realized metrics to {metrics_csv}")

    print("\n" + "=" * 70)
    print("GVP REAL-WORLD DATABASE BENCHMARK -- SUMMARY")
    print("=" * 70)

    for sched in SCHEDULERS:
        sub = df[df['scheduler'] == sched]
        if sub.empty:
            continue
        print(f"\n{sched.upper()}  (n={len(sub)})")
        print(f"  TASK completion rate : {sub['task_completion_rate'].mean():.3f} "
              f"± {sub['task_completion_rate'].std():.3f}")
        print(f"  GROUP completion rate: {sub['group_completion_rate'].mean():.3f} "
              f"± {sub['group_completion_rate'].std():.3f}")
        print(f"  Realized quality     : {sub['realized_quality'].mean():.1f}")
        print(f"  Net utility          : {sub['utility'].mean():.1f} ± {sub['utility'].std():.1f}")
        print(f"  Total cost           : {sub['total_cost'].mean():.1f}")
        n_exec_fail = sub['n_execution_failed'].mean() if 'n_execution_failed' in sub.columns else float('nan')
        n_cancelled = sub['n_cancelled'].mean() if 'n_cancelled' in sub.columns else float('nan')
        print(f"  Bookings submitted   : {sub['n_submissions'].mean():.1f} "
              f"(accepted {sub['n_accepted'].mean():.1f}, executed ok {sub['n_executed'].mean():.1f}, "
              f"exec-failed {n_exec_fail:.1f}, cancelled {n_cancelled:.1f})")
        print(f"  TRUE passes/task     : {sub['submitted_passes_per_task'].mean():.2f} submitted, "
              f"{sub['exec_passes_per_completed'].mean():.2f} executed/completed")
        print(f"  Rejection rate (diag): {100 * sub['rejection_rate'].mean():.1f}%")
        n_replans = sub['replans'].mean() if 'replans' in sub.columns else float('nan')
        print(f"  Replans / cancels    : {n_replans:.1f} replans, {n_cancelled:.1f} cancellations")

        # Quality-rank distribution (mean counts across runs)
        if 'quality_rank_distribution' in sub.columns:
            all_rank_dicts = sub['quality_rank_distribution'].dropna().tolist()
            merged = {}
            for rd in all_rank_dicts:
                if not isinstance(rd, dict):
                    continue
                for rank, cnt in rd.items():
                    merged[rank] = merged.get(rank, 0) + cnt
            n_runs = max(len(all_rank_dicts), 1)
            total = sum(merged.values())
            if total > 0:
                rank_parts = []
                for rank in sorted((r for r in merged if r != 'unranked'), key=lambda x: int(x) if str(x).isdigit() else 999):
                    rank_parts.append(f"rank {rank}: {merged[rank]/n_runs:.1f} ({100*merged[rank]/total:.0f}%)")
                if 'unranked' in merged:
                    rank_parts.append(f"unranked: {merged['unranked']/n_runs:.1f}")
                print(f"  Quality rank dist    : {', '.join(rank_parts)}")

    summary_cols = [c for c in [
        'task_completion_rate', 'group_completion_rate', 'realized_quality',
        'utility', 'total_cost', 'n_submissions', 'n_accepted', 'n_executed',
        'n_execution_failed', 'n_cancelled', 'n_rejected', 'replans',
        'submitted_passes_per_task', 'exec_passes_per_completed',
        'rejection_rate'] if c in df.columns]
    summary_df = df.groupby('scheduler')[summary_cols].mean().reset_index()
    summary_csv = os.path.join(results_dir, "summary_v2.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[Summary] Saved summary to {summary_csv}")

    if 'deterministic' in set(df['scheduler']):
        paired_summary(records, SCHEDULERS,
                       primary='task_completion_rate', baseline='deterministic')

    # Completion-vs-cost frontier
    try:
        plot_cost_frontier(records, SCHEDULERS,
                           out_path=os.path.join(results_dir, "cost_frontier.png"),
                           y='task_completion_rate', x='total_cost')
    except Exception as e:
        print(f"[Warning] frontier plot failed: {e}")

    # Utility figure (means + paired lines + paired differences)
    try:
        plot_utility(records, SCHEDULERS,
                     out_path=os.path.join(results_dir, "utility.png"),
                     baseline='deterministic')
    except Exception as e:
        print(f"[Warning] utility plot failed: {e}")

    # Completion / utility boxplots
    try:
        present = [s for s in SCHEDULERS if s in set(df['scheduler'])]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.boxplot([df[df['scheduler'] == s]['task_completion_rate'].values for s in present],
                    labels=present)
        ax1.set_title("Task completion rate")
        ax1.set_ylabel("fraction of demand completed")
        ax2.boxplot([df[df['scheduler'] == s]['utility'].values for s in present],
                    labels=present)
        ax2.set_title("Net utility")
        ax2.set_ylabel("realized quality - cost")
        plt.tight_layout()
        box_png = os.path.join(results_dir, "comparison_plots.png")
        plt.savefig(box_png, dpi=150)
        plt.close(fig)
        print(f"[OK] Boxplots saved to {box_png}")
    except Exception as e:
        print(f"[Warning] Boxplot failed: {e}")

    # ---------------------- interpretation ---------------------------------
    print("\n" + "=" * 70)
    print("RESULTS INTERPRETATION")
    print("=" * 70)

    def mean_of(sched, col):
        sub = df[df['scheduler'] == sched]
        return sub[col].mean() if (not sub.empty and col in sub) else float('nan')

    det_comp = mean_of('deterministic', 'task_completion_rate')
    stoch_comp = mean_of('stochastic_log', 'task_completion_rate')
    det_util = mean_of('deterministic', 'utility')
    stoch_util = mean_of('stochastic_log', 'utility')
    det_pass = mean_of('deterministic', 'submitted_passes_per_task')
    stoch_pass = mean_of('stochastic_log', 'submitted_passes_per_task')

    print("\nPrimary (demand completion):")
    if np.isfinite(det_comp) and np.isfinite(stoch_comp):
        print(f"  stochastic {stoch_comp:.3f} vs deterministic {det_comp:.3f} "
              f"(delta = {stoch_comp - det_comp:+.3f} of demand completed)")
    print("\nRedundancy actually exercised (was the mechanism engaged?):")
    print(f"  submitted passes/task: stochastic {stoch_pass:.2f} vs deterministic {det_pass:.2f}")
    print("  (~1.0 for both => little redundancy available; check pass density.")
    print("   Deterministic >> stochastic => the baseline is booking up to its cap")
    print("   because its objective has no diminishing returns and no booking cost.)")
    print("\nUtility (quality net of cost):")
    print(f"  stochastic {stoch_util:.1f} vs deterministic {det_util:.1f}")
    print("\nNote: rejection rate is a diagnostic only. Redundancy books MORE")
    print("passes, so absolute rejections can rise even when the planner works --")
    print("it makes rejections not matter, not rare.")

    return records


# ============================== entry points ================================

def run_comparison(num_monte_carlo_runs=NUM_MC_RUNS, results_dir=None,
                   schedulers=None, plot_schedule=True):
    """
    Legacy all-in-one-process loop. Same scheduling behaviour as before, but
    each run's objects are released before the next starts. For large
    campaigns prefer run_campaign.py (one process per run): process exit is
    the only hard guarantee against OOM.
    """
    schedulers = schedulers or SCHEDULERS
    if results_dir is None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = os.path.join("results", f"volcano_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n[Results] Saving to directory: {results_dir}")
    print(f"[Config] SIMULATION_START = {SIMULATION_START.isoformat()}")

    cached_satellites = load_satellites_once()
    volcano_db_locations = load_volcano_locations_from_database()

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=lookahead_horizon_h)

    demand_field = build_demand_field(volcano_db_locations, min_time)
    try:
        demand_field.plot_heatmaps(os.path.join(results_dir, "demand_heatmaps"))
    except Exception as e:
        print(f"[Warning] demand heatmaps failed: {e}")

    all_records = []
    for run_idx in range(num_monte_carlo_runs):
        run_seed = 42 + run_idx
        print(f"\n--- Monte Carlo Iteration {run_idx + 1}/{num_monte_carlo_runs} "
              f"(seed {run_seed}) ---")
        for scheduler in schedulers:
            m = run_one_scheduler(scheduler, run_seed, cached_satellites,
                                  volcano_db_locations, demand_field,
                                  min_time, max_time, results_dir,
                                  plot_schedule=plot_schedule)
            if m is not None:
                all_records.append(m)

    aggregate(results_dir, records=all_records)
    return all_records


def main():
    global SIMULATION_START

    parser = argparse.ArgumentParser(description="Volcano scheduler comparison")
    parser.add_argument('--scheduler', choices=SCHEDULERS,
                        help="Run exactly one scheduler and exit (campaign mode).")
    parser.add_argument('--seed', type=int, help="Seed for the single run.")
    parser.add_argument('--start', type=str,
                        help="ISO simulation start, e.g. 2026-07-22T00:00:00. "
                             "MUST be identical across every process in a campaign.")
    parser.add_argument('--results-dir', type=str, default=None,
                        help="Directory for run_*.json and aggregate outputs.")
    parser.add_argument('--aggregate', action='store_true',
                        help="Only aggregate an existing --results-dir.")
    parser.add_argument('--runs', type=int, default=NUM_MC_RUNS,
                        help="Number of seeds for the legacy in-process loop.")
    parser.add_argument('--no-schedule-plots', action='store_true',
                        help="Skip per-run schedule plots (saves time and memory).")
    args = parser.parse_args()

    # Aggregate-only mode
    if args.aggregate:
        if not args.results_dir:
            parser.error("--aggregate requires --results-dir")
        aggregate(args.results_dir)
        return

    # Pin the simulation start BEFORE any geometry is computed.
    if args.start:
        SIMULATION_START = dt.datetime.fromisoformat(args.start)

    # Single-run (campaign) mode
    if args.scheduler:
        if args.seed is None:
            parser.error("--scheduler requires --seed")
        if not args.start:
            print("[WARNING] --start not given: this process picked its own start "
                  "time, so its geometry will NOT match other processes. Pass "
                  "--start for any paired campaign.")
        results_dir = args.results_dir or os.path.join(
            "results", f"volcano_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"[Config] scheduler={args.scheduler} seed={args.seed} "
              f"start={SIMULATION_START.isoformat()} dir={results_dir}")

        cached_satellites = load_satellites_once()
        volcano_db_locations = load_volcano_locations_from_database()
        min_time = SIMULATION_START
        max_time = SIMULATION_START + dt.timedelta(hours=lookahead_horizon_h)
        demand_field = build_demand_field(volcano_db_locations, min_time)

        run_one_scheduler(args.scheduler, args.seed, cached_satellites,
                          volcano_db_locations, demand_field, min_time, max_time,
                          results_dir, plot_schedule=not args.no_schedule_plots)
        print(f"[Done] {args.scheduler} seed {args.seed}; peak RSS {_rss_gb():.2f} GB")
        return

    # Legacy in-process loop
    run_comparison(num_monte_carlo_runs=args.runs, results_dir=args.results_dir,
                   plot_schedule=not args.no_schedule_plots)


if __name__ == "__main__":
    main()