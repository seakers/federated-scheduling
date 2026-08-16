"""
Earthquake Damage Assessment Workflow: Stochastic vs Deterministic Comparison
Exercising the GENERAL Logical DAG Formulation (AND / OR / NOT gates)
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Import FAME components
from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus
from fame_broker import Broker
from fame_workflow import *
from fame_demand_model import DemandField, DemandFieldConfig

from earthquake_utils import (
    fetch_usgs_events,
    load_cities,
    select_targets,
    create_earthquake_workflow,
    compute_earthquake_reachability,
    register_earthquake_phenomena,
)
from fame_metrics import compute_metrics_v3, paired_summary, plot_cost_frontier
from benchmarking_utils import (
    GROUND_STATIONS,
    load_satellites_once as _load_satellites_once_shared,
    run_simulation_forward as _run_simulation_forward_shared,
    rss_gb, 
)

# ============================ Configuration =================================
SIMULATION_START = dt.datetime(2026, 8, 1, 0, 0, 0)
LOOKAHEAD_HORIZON_H = 60.0   # Exact workflow lifespan
MAX_SOLVER_TIME_S = 120       # Fast solver timeout cap
MAX_NUM_INSTANCES = 5        # Backup passes per task
NUM_MC_RUNS = 3

# === COST CONFIGURATION ===
TAX_RATE = 0.0              
SUBMISSION_COST = 0.01      
ACCEPT_NOTIFY_MIN_H = 0.25
ACCEPT_NOTIFY_MAX_H = 1.50

PROVIDER_RATES = {
    "Planet":          0.05,
    "Umbra":           0.12,
    "Capella":         0.15,
    "LOFT":            0.08,
    "Ubotica":         0.08,
    "Mission Control": 0.08,
    "AC":              0.08,
    "ICEYE":           0.12,
}

PROVIDER_RATE_DEFAULT = 0.020
LEAD_K = 3.0          
LEAD_T_REF_H = 6.0   

# === PROBABILITY CONFIGURATION ===
P_ACC_MIN = 0.70   
P_ACC_MAX = 0.95  

P_EXEC_MIN = 0.65  
P_EXEC_MAX = 0.85 

# Ablation ladder -- each rung adds exactly one capability, so a gap between
# adjacent rungs is attributable to that one thing:
#
#   random             no intelligence at all
#   greedy             + local quality, ONE pass per task (reactive: a failure is
#                        recovered by replanning, i.e. SERIAL retries)
#   greedy_n           + redundancy budget (N parallel passes), still local and
#                        gate-blind. THE control for "the gain is just backups".
#   deterministic      + cross-task optimisation, but no uncertainty model
#   stochastic_logical + uncertainty model AND the AND/OR/NOT gates  <- the claim
SCHEDULERS = ['stochastic_logical', 'deterministic', 'greedy_n', 'greedy', 'random']
#SCHEDULERS = ['stochastic_logical']
ENABLE_CANCELLATIONS = True  # Immediately releases unneeded backup passes on primary success


def load_satellites_for_earthquake(sim_start: dt.datetime, horizon_h: float) -> list:
    """
    Selects a balanced fleet of ~38 imaging satellites (18 RGB + 20 SAR).
    """
    full_fleet = _load_satellites_once_shared(sim_start, horizon_h)
    
    rgb_sats = [s for s in full_fleet if InstrumentType.RGB in s.instruments][:30]
    sar_sats = [s for s in full_fleet if InstrumentType.SAR in s.instruments][:30]
    
    pruned_fleet = rgb_sats + sar_sats
    print(f"[Fleet Tuning] Selected {len(pruned_fleet)} satellites ({len(rgb_sats)} RGB, {len(sar_sats)} SAR) out of {len(full_fleet)} total.")
    return pruned_fleet


def load_earthquake_targets(min_time: dt.datetime) -> list:
    start_time = dt.datetime(2020, 1, 1)
    end_time = min_time + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
    
    cache_path = os.path.join("cache", "usgs_events_historical.json")
    cities_csv = os.path.join("simplemaps_worldcities_basicv1.901", "worldcities.csv")

    events = fetch_usgs_events(start=start_time, end=end_time, min_magnitude=6.0, cache_path=cache_path)
    
    if os.path.exists(cities_csv):
        cities = load_cities(cities_csv, min_population=50000)
    else:
        cities = [
            {"name": "Tokyo", "lat_deg": 35.6762, "lon_deg": 139.6503, "population": 14000000},
            {"name": "Los Angeles", "lat_deg": 34.0522, "lon_deg": -118.2437, "population": 4000000},
            {"name": "San Francisco", "lat_deg": 37.7749, "lon_deg": -122.4194, "population": 880000},
            {"name": "Jakarta", "lat_deg": -6.2088, "lon_deg": 106.8456, "population": 10500000},
            {"name": "Kahramanmaraş", "lat_deg": 37.5833, "lon_deg": 36.9333, "population": 1100000},
        ]

    targets = select_targets(events, cities, max_targets=4, max_distance_km=None, min_magnitude=6.0)

    if not targets:
        from fame_geometry import Location
        fallback_locs = [
            ("M7.8_Kahramanmaraş", 37.174, 37.032, "Kahramanmaraş", 7.8),
            ("M7.1_SanFrancisco", 37.7749, -122.4194, "San Francisco", 7.1),
        ]
        for name, lat, lon, city, mag in fallback_locs:
            loc = Location(lon, lat, 0.0, name)
            loc.magnitude = mag
            loc.nearest_city = city
            loc.city_distance_km = 15.0
            loc.exposure = 1000000 / 15.0
            loc.event_time = min_time.isoformat()
            targets.append(loc)

    return targets


def create_world_and_constellations(cached_satellites: list[Satellite],
                                     demand_field: DemandField = None,
                                     execution_probability_function=None):
    local_satellites = copy.deepcopy(cached_satellites)
    world = World(satellites=local_satellites)
    world.time = SIMULATION_START

    planet_sats = [s for s in local_satellites if any(x in s.name.upper() for x in ["SKYSAT", "PELICAN", "TANAGER"])]
    umbra_sats = [s for s in local_satellites if "UMBRA" in s.name.upper()]
    capella_sats = [s for s in local_satellites if "CAPELLA" in s.name.upper() or "ACADIA" in s.name.upper()]
    loft_sats = [s for s in local_satellites if "LOFT" in s.name.upper() or "YAM" in s.name.upper()]
    ubotica_sats = [s for s in local_satellites if "UBOTICA" in s.name.upper() or "HAMMER" in s.name.upper() or "ACCENTURE" in s.name.upper()]
    mission_control_sats = [s for s in local_satellites if "PERSISTENCE" in s.name.upper() or "LEMUR" in s.name.upper()]
    aerospace_sats = [s for s in local_satellites if "AEROCUBE" in s.name.upper()]
    iceye_sats = [s for s in local_satellites if "ICEYE" in s.name.upper()]

    _sim_acc_fn = demand_field.make_simulator_acceptance_function() if demand_field is not None else None

    def _make_scheduler(sats, name, legacy_p):
        return ConstellationGroundScheduler(
            satellites=sats,
            ground_stations=GROUND_STATIONS,
            world=world,
            name=name,
            acceptance_probability=legacy_p,
            acceptance_probability_function=_sim_acc_fn,
            execution_probability_function=execution_probability_function,
            acceptance_notification_delay_h=(ACCEPT_NOTIFY_MIN_H, ACCEPT_NOTIFY_MAX_H),
        )

    all_constellations = [
        _make_scheduler(planet_sats,          "Planet",          0.35),
        _make_scheduler(umbra_sats,           "Umbra",           0.45),
        _make_scheduler(capella_sats,         "Capella",         0.50),
        _make_scheduler(loft_sats,            "LOFT",            0.40),
        _make_scheduler(ubotica_sats,         "Ubotica",         0.35),
        _make_scheduler(mission_control_sats, "Mission Control", 0.40),
        _make_scheduler(aerospace_sats,       "AC",              0.35),
        _make_scheduler(iceye_sats,           "ICEYE",           0.50),
    ]

    for constellation in all_constellations:
        world.add_constellation(constellation)

    return world, all_constellations


def build_demand_field(targets, min_time):
    _demand_cfg = DemandFieldConfig(use_constant_probability=False, p_min=P_ACC_MIN, p_max=P_ACC_MAX)
    _horizon_s = LOOKAHEAD_HORIZON_H * 3600.0
    demand_field = DemandField(config=_demand_cfg, reference_time=min_time, horizon_s=_horizon_s)

    for target in targets:
        demand_field.add_spike(target.lat_deg, target.lon_deg, min_time)

    _all_constellation_names = ["Planet", "Umbra", "Capella", "LOFT",
                                "Ubotica", "Mission Control", "AC", "ICEYE"]
    demand_field.precompute(_all_constellation_names)
    demand_field.check_cost_reliability_tension(PROVIDER_RATES)
    return demand_field


def make_probability_functions(demand_field):
    def acceptance_prob_function(constrained_request, satellite, obs_pass):
        return demand_field.make_acceptance_prob_function()(constrained_request, satellite, obs_pass)

    def execution_prob_function(constrained_request, satellite, obs_opp):
        opp = obs_opp.highest if hasattr(obs_opp, 'highest') else obs_opp
        look_angle = abs(90.0 - opp.look_angle_dec_deg)
        t = look_angle / 90.0
        execution_prob = P_EXEC_MAX - t * (P_EXEC_MAX - P_EXEC_MIN)
        return max(P_EXEC_MIN, min(P_EXEC_MAX, execution_prob))

    return acceptance_prob_function, execution_prob_function


def make_execution_cost_fn(all_constellations):
    _sat_to_constellation_name = {
        sat: c.name for c in all_constellations for sat in c.satellites
    }

    def execution_cost_fn(task, satellite, obs_pass, dispatch_time, q_max=None):
        if q_max is None or q_max <= 0:
            q_max = 1.0
        provider_name = _sat_to_constellation_name.get(satellite, "")
        rate = PROVIDER_RATES.get(provider_name, PROVIDER_RATE_DEFAULT)
        try:
            _dt_valid = dispatch_time is not None and dispatch_time == dispatch_time
        except Exception:
            _dt_valid = False
        if _dt_valid and obs_pass is not None:
            rise_time = obs_pass.rise.time if hasattr(obs_pass, 'rise') else obs_pass.highest.time
            lead_h = max(0.0, (rise_time - dispatch_time).total_seconds() / 3600.0)
            multiplier = 1.0 + LEAD_K * max(0.0, 1.0 - lead_h / LEAD_T_REF_H)
        else:
            multiplier = 1.0
        return rate * q_max * multiplier

    return execution_cost_fn


def run_one_scheduler(scheduler, seed, cached_satellites, targets,
                      demand_field, min_time, max_time, results_dir,
                      plot_schedule=True):


    
    plots_dir = os.path.join(results_dir, "plots", scheduler)
    os.makedirs(plots_dir, exist_ok=True)

    acceptance_prob_function, execution_prob_function = make_probability_functions(demand_field)

    print(f"\n  Running {scheduler} (seed {seed})...  [RSS {rss_gb():.2f} GB]")
    random.seed(seed)
    np.random.seed(seed)

    world, constellations = create_world_and_constellations(
        cached_satellites,
        demand_field=demand_field,
        execution_probability_function=execution_prob_function,
    )
    register_earthquake_phenomena(world, targets, min_time, max_time, seed=seed)
    execution_cost_fn = make_execution_cost_fn(constellations)
    
    workflow = create_earthquake_workflow(
        targets=targets,
        min_time=min_time,
        max_time=max_time,
        max_num_instances=MAX_NUM_INSTANCES,
        satellites=world.satellites
    )
    
    broker = Broker(constellations=constellations, world=world, name=f"Broker-{scheduler}")
    broker.add_workflow(workflow)
    world.add_broker(broker)
    for t in workflow.constrained_observation_requests:
        r = t.observation_request
        opps = find_observation_opportunities([r], cached_satellites)
        n = sum(len(v) for v in opps[r].values())
        print(f"{r.name:34s} [{(r.min_time-min_time).total_seconds()/3600:5.1f}h .."
            f"{(r.max_time-min_time).total_seconds()/3600:5.1f}h] {str(r.instrument):18s} {n:3d} passes")
    m = None
    try:
        _horizon = dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
        
        if scheduler in ('stochastic_logical', 'stochastic_log'):
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=True, use_stochastic=True,
                stochastic_formulation="general_logical_dag",
                acceptance_probability_function=acceptance_prob_function,
                execution_probability_function=execution_prob_function,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_fn=execution_cost_fn,
                tax_rate=TAX_RATE,
                max_solver_time_s=MAX_SOLVER_TIME_S, 
                solver_engine="GUROBI",
                update_timelines=False, update_requests=True,
                receding_horizon_duration=_horizon,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                enable_cancellations=ENABLE_CANCELLATIONS,
            )
        elif scheduler == 'deterministic':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=True, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=True, tax_rate=TAX_RATE,
                receding_horizon_duration=_horizon,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_fn=execution_cost_fn,
            )
        elif scheduler == 'greedy':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=True, tax_rate=TAX_RATE,
                receding_horizon_duration=_horizon,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_fn=execution_cost_fn,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir
            )
        elif scheduler == 'greedy_n':
            # REDUNDANCY ABLATION. Same redundancy budget as the stochastic
            # planner (MAX_NUM_INSTANCES) and the same cancellation policy, but
            # selection is purely local: top-N passes by quality per task, no
            # cross-task contention reasoning, no acceptance/execution
            # probabilities, no gates.
            #
            # This is the baseline that separates "books backups" from "sees the
            # DAG". Expected best-of-successes is monotone submodular in the
            # horizontal track, so quality-ordered greedy is NEAR-OPTIMAL there --
            # any remaining gap to the stochastic planner is attributable to the
            # AND/OR/NOT structure rather than to the redundancy itself.
            #
            # enable_cancellations MUST match the stochastic run: without it
            # greedy-N pays for all N passes even after one succeeds, and the
            # cost comparison stops meaning anything.
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=True, tax_rate=TAX_RATE,
                receding_horizon_duration=_horizon,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_fn=execution_cost_fn,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                enable_cancellations=ENABLE_CANCELLATIONS,
                greedy_max_instances=MAX_NUM_INSTANCES,
            )
        elif scheduler == 'random':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=True, tax_rate=TAX_RATE,
                receding_horizon_duration=_horizon,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_fn=execution_cost_fn,
                max_reschedule_depth=10000,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                use_random=True,
                random_seed=seed,
            )
        elif scheduler == 'super_random':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=True, tax_rate=TAX_RATE,
                receding_horizon_duration=_horizon,
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_fn=execution_cost_fn,
                max_reschedule_depth=5,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir,
                use_random=True,
                super_random=True,
                random_seed=seed,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        _run_simulation_forward_shared(world)

        m = compute_metrics_v3(
            broker._workflow_graph, broker, ObservationStatus,
            submission_cost_rate=SUBMISSION_COST,
            execution_cost_fn=execution_cost_fn,
            verbose=True,
            sim_end_time=world.time,
        )

        # --- GATE-AWARE OVERRIDE -----------------------------------------
        # compute_metrics_v3 counts every task equally, but HIRES is
        # unreachable BY CONSTRUCTION whenever URBAN_opt succeeded. Leaving it
        # in the denominator penalises exactly the planner that made the right
        # call, which is the effect this case study exists to measure. An
        # executed pass that returned an EMPTY product is also not a
        # completion: the satellite did its job, the mission did not.
        _tasks = list(broker._workflow_graph.nodes())
        _reqs = broker._requests
        _name_to_task = {t.observation_request.name: t for t in _tasks}

        _executed, _best_q = set(), {}
        for _, _row in _reqs.iterrows():
            if _row['status'] != ObservationStatus.DATA_RECEIVED:
                continue
            if not _row['data_product']:
                continue
            _rp = _row['requested_pass']
            if _rp is None:
                continue
            # Match by NAME: observation_request is mutated in place by the
            # retargeting updater, but a task could still be rebound elsewhere.
            _t = _name_to_task.get(getattr(_row['request'], 'name', None))
            if _t is None:
                continue
            _executed.add(_t)
            _q = _t.rewarder(_rp.highest)
            if _q > _best_q.get(_t, -1e9):
                _best_q[_t] = _q

        reachable = compute_earthquake_reachability(_tasks, _executed)
        _valid = _executed & reachable
        _n_reach = len(reachable)

        for _k in ('realized_quality', 'utility', 'task_completion_rate',
                   'n_tasks_completed', 'n_tasks_reachable'):
            m.pop(_k, None)

        m['n_tasks_reachable']    = _n_reach
        m['n_tasks_completed']    = len(_valid)
        m['task_completion_rate'] = (len(_valid) / _n_reach) if _n_reach else 0.0
        m['realized_quality']     = sum(_best_q.get(t, 0.0) for t in _valid)
        m['utility']              = m['realized_quality'] - m['total_cost']

        print(f"   [Gate-aware] {len(_valid)}/{_n_reach} reachable tasks completed, "
              f"quality {m['realized_quality']:.1f}, utility {m['utility']:.1f}")

        m['scheduler'] = scheduler
        m['seed'] = seed
        m['tax_rate'] = TAX_RATE
        m['max_num_instances'] = MAX_NUM_INSTANCES
        m['sim_start'] = SIMULATION_START.isoformat()

        execution_details = m.pop('execution_details', [])
        exec_file = os.path.join(results_dir, f"executions_seed{seed:04d}_{scheduler}.json")
        with open(exec_file, 'w') as f:
            json.dump(execution_details, f, indent=2, default=str)

        run_file = os.path.join(results_dir, f"run_seed{seed:04d}_{scheduler}.json")
        with open(run_file, 'w') as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  [Saved] {run_file}")

    except Exception as e:
        import traceback
        print(f"    Broker Error ({scheduler}, seed {seed}): {e}")
        traceback.print_exc()

    try:
        del broker, workflow, world, constellations
    except Exception:
        pass
    plt.close('all')
    gc.collect()
    print(f"  [Mem] after {scheduler} seed {seed}: peak RSS {rss_gb():.2f} GB")
    return m


def aggregate(results_dir, records=None):
    if records is None:
        records = []
        for path in sorted(glob.glob(os.path.join(results_dir, "run_*.json"))):
            try:
                with open(path) as f:
                    r = json.load(f)
                for k, v in list(r.items()):
                    if isinstance(v, str) and k not in ('scheduler', 'sim_start'):
                        try:
                            r[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                records.append(r)
            except Exception as e:
                print(f"[Warning] Could not read {path}: {e}")

    if not records:
        print("[Warning] No records found; nothing to summarize.")
        return []

    df = pd.DataFrame(records)
    metrics_csv = os.path.join(results_dir, "metrics_v2.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"\n[Per-Run] Saved all realized metrics to {metrics_csv}")

    print("\n" + "=" * 70)
    print("EARTHQUAKE GENERAL LOGICAL DAG BENCHMARK -- SUMMARY")
    print("=" * 70)

    for sched in SCHEDULERS:
        sub = df[df['scheduler'] == sched]
        if sub.empty:
            continue
        print(f"\n{sched.upper()}  (n={len(sub)})")
        print(f"  TASK completion rate : {sub['task_completion_rate'].mean():.3f} ± {sub['task_completion_rate'].std():.3f}")
        print(f"  GROUP completion rate: {sub['group_completion_rate'].mean():.3f} ± {sub['group_completion_rate'].std():.3f}")
        print(f"  Realized quality     : {sub['realized_quality'].mean():.1f}")
        print(f"  Net utility          : {sub['utility'].mean():.1f} ± {sub['utility'].std():.1f}")
        print(f"  Total cost           : {sub['total_cost'].mean():.1f}")

    try:
        plot_cost_frontier(records, SCHEDULERS,
                           out_path=os.path.join(results_dir, "cost_frontier.png"),
                           y='task_completion_rate', x='total_cost')
    except Exception as e:
        print(f"[Warning] Frontier plot failed: {e}")

    return records


def run_comparison(num_monte_carlo_runs=NUM_MC_RUNS, results_dir=None, plot_schedule=True):
    if results_dir is None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = os.path.join("results", f"earthquake_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    cached_satellites = load_satellites_for_earthquake(SIMULATION_START, horizon_h=LOOKAHEAD_HORIZON_H)
    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)

    targets = load_earthquake_targets(min_time)
    demand_field = build_demand_field(targets, min_time)

    all_records = []
    for run_idx in range(num_monte_carlo_runs):
        run_seed = 42 + run_idx
        for scheduler in SCHEDULERS:
            m = run_one_scheduler(scheduler, run_seed, cached_satellites,
                                  targets, demand_field, min_time, max_time,
                                  results_dir, plot_schedule=plot_schedule)
            if m is not None:
                all_records.append(m)

    aggregate(results_dir, records=all_records)
    return all_records


def main():
    global SIMULATION_START

    parser = argparse.ArgumentParser(description="Earthquake scheduler comparison")
    parser.add_argument('--scheduler', choices=SCHEDULERS,
                        help="Run exactly one scheduler and exit (campaign mode).")
    parser.add_argument('--seed', type=int, help="Seed for the single run.")
    parser.add_argument('--start', type=str,
                        help="ISO simulation start, e.g. 2026-08-01T00:00:00.")
    parser.add_argument('--results-dir', type=str, default=None,
                        help="Directory for run_*.json and aggregate outputs.")
    parser.add_argument('--aggregate', action='store_true',
                        help="Only aggregate an existing --results-dir.")
    parser.add_argument('--runs', type=int, default=NUM_MC_RUNS,
                        help="Number of seeds for the in-process loop.")
    parser.add_argument('--no-schedule-plots', action='store_true',
                        help="Skip per-run schedule plots.")
    args = parser.parse_args()

    if args.aggregate:
        if not args.results_dir:
            parser.error("--aggregate requires --results-dir")
        aggregate(args.results_dir)
        return

    if args.start:
        SIMULATION_START = dt.datetime.fromisoformat(args.start)

    if args.scheduler:
        if args.seed is None:
            parser.error("--scheduler requires --seed")
        results_dir = args.results_dir or os.path.join(
            "results", f"earthquake_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)

        cached_satellites = load_satellites_for_earthquake(SIMULATION_START, LOOKAHEAD_HORIZON_H)
        min_time = SIMULATION_START
        max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)

        targets = load_earthquake_targets(min_time)
        demand_field = build_demand_field(targets, min_time)

        run_one_scheduler(args.scheduler, args.seed, cached_satellites,
                          targets, demand_field, min_time, max_time,
                          results_dir, plot_schedule=not args.no_schedule_plots)
        return

    run_comparison(num_monte_carlo_runs=args.runs, results_dir=args.results_dir,
                   plot_schedule=not args.no_schedule_plots)


if __name__ == "__main__":
    main()