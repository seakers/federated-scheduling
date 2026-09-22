"""River high-flow co-observation: hedging vs greedy / deterministic.

Fan-in of SAR + VIS + TIR (Gorr-style 3-type co-observation) on real USGS
high-flow events (Q >= Q75). Federated twist: DemandField acceptance plus
modality-aware execution (SAR cloud-immune; VIS/TIR degraded by TCC).

Typical band (aligned with volcano/earthquake):
    p_acc ∈ [0.70, 0.95]
    p_exec,SAR ≳ 0.90;  p_exec,VIS/TIR ≈ p_geom · (1 − TCC) ∈ ~[0.55, 0.95]
"""

import argparse
import datetime as dt
import gc
import glob
import json
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import fame_geometry
from fame_geometry import InstrumentType, find_observation_opportunities
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus
from fame_broker import Broker
from fame_workflow import *
from fame_demand_model import DemandField, DemandFieldConfig
from fame_metrics import compute_metrics_v3, plot_cost_frontier
from benchmarking_utils import (
    GROUND_STATIONS,
    load_satellites_once as _load_satellites_once_shared,
    create_world_and_constellations as _create_world_shared,
    run_simulation_forward as _run_simulation_forward_shared,
    rss_gb,
)
from riverflow_data_prep import (
    ensure_cache, load_scenario_day, SCENARIO_DAY, TOY_DAY,
)
from riverflow_for import install as install_for, FUSION_SAT_NAME
from riverflow_utils import (
    CAMPAIGN_HORIZON_H,
    load_events,
    events_as_locations,
    create_riverflow_workflow,
    register_river_phenomena,
    compute_river_reachability,
    make_execution_prob_function,
    is_fusion_task,
    prepare_river_fleet,
    load_extra_tir_satellites,
    make_fusion_satellite,
    opportunity_census,
    print_census_gate,
    P_EXEC_SAR,
)

# ============================ Configuration =================================
# Start before the UTC day so backups can be booked with lead.
DISPATCH_LEAD_H = 4.0
# Scenario day comes from the USGS cache (default 2026-06-17).
SIMULATION_START = SCENARIO_DAY - dt.timedelta(hours=DISPATCH_LEAD_H)
LOOKAHEAD_HORIZON_H = CAMPAIGN_HORIZON_H
MAX_SOLVER_TIME_S = 300
MAX_NUM_INSTANCES = 5
NUM_MC_RUNS = 20

TAX_RATE = 0.0
SUBMISSION_COST = 0.01
# Notify accept/reject uniform(min, max) hours BEFORE the pass (same protocol
# as volcano). (0, 0) would fire at the pass itself — not "immediate on submit".
ACCEPT_NOTIFY_MIN_H = 0.25
ACCEPT_NOTIFY_MAX_H = 1.0

PROVIDER_RATES = {
    "Planet":          0.05,
    "Umbra":           0.12,
    "Capella":         0.15,
    "LOFT":            0.08,
    "Ubotica":         0.08,
    "Mission Control": 0.08,
    "AC":              0.08,
    "ICEYE":           0.12,
    "OroraTech":       0.06,
    "constellr":       0.18,
    "SatVu":           0.20,
    "USGS":            0.07,
    "Copernicus":      0.07,
    "Fusion":          0.0,
}
PROVIDER_RATE_DEFAULT = 0.08
LEAD_K = 3.0
LEAD_T_REF_H = 8.0

P_ACC_MIN = 0.70
P_ACC_MAX = 0.95

SCHEDULERS = ["stochastic_logical", "deterministic", "greedy_n", "greedy", "random"]
ENABLE_CANCELLATIONS = True

CONSTELLATION_NAMES = [
    "Planet", "Umbra", "Capella", "LOFT", "Ubotica", "Mission Control",
    "AC", "ICEYE", "OroraTech", "constellr", "SatVu",
    "USGS", "Copernicus", "Fusion",
]


def _tle_for_scenario(day):
    """Pick tles/all_tles_YYYYMMDD.txt closest to the USGS scenario day."""
    import glob
    import re
    files = sorted(glob.glob(os.path.join("tles", "all_tles_*.txt")))
    if not files:
        return None
    target = day.date() if hasattr(day, "date") else day
    best, best_d = files[-1], None
    for path in files:
        m = re.search(r"all_tles_(\d{8})\.txt$", os.path.basename(path))
        if not m:
            continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        delta = abs((d - target).days)
        if best_d is None or delta < best_d:
            best, best_d = path, delta
    return best


def load_satellites_for_river(sim_start, horizon_h):
    """TIR in full; a trimmed RGB/SAR pool so the MILP stays tractable."""
    ensure_cache(allow_toy=False)
    tle = _tle_for_scenario(load_scenario_day())
    full = _load_satellites_once_shared(sim_start, horizon_h, tle_file=tle)
    full.extend(load_extra_tir_satellites(sim_start, horizon_h, tle_file=tle))
    prepare_river_fleet(full)
    rgb = [s for s in full if InstrumentType.RGB in s.instruments
           and "FOREST" not in s.name.upper()
           and "SKYBEE" not in s.name.upper()
           and "HOTSAT" not in s.name.upper()]
    sar = [s for s in full if InstrumentType.SAR in s.instruments]
    tir = [s for s in full if InstrumentType.TIR in s.instruments]
    fleet = []
    seen = set()
    for s in rgb + sar + tir:
        if id(s) in seen:
            continue
        seen.add(id(s))
        fleet.append(s)
    print(f"[Fleet] {len(fleet)} satellites "
          f"({len(rgb)} RGB, {len(sar)} SAR, {len(tir)} TIR)")
    return fleet


def sync_simulation_clock():
    """Bind SIMULATION_START to the cached USGS scenario day."""
    global SIMULATION_START, DISPATCH_LEAD_H
    day = load_scenario_day()
    SIMULATION_START = day - dt.timedelta(hours=DISPATCH_LEAD_H)
    return SIMULATION_START


def create_world_and_constellations(cached_satellites, demand_field=None,
                                    execution_probability_function=None):
    world, consts = _create_world_shared(
        cached_satellites, SIMULATION_START, demand_field,
        execution_probability_function,
        (ACCEPT_NOTIFY_MIN_H, ACCEPT_NOTIFY_MAX_H),
    )
    extra_groups = (
        ("USGS", lambda n: "LANDSAT" in n.upper()),
        ("Copernicus", lambda n: "SENTINEL 3" in n.upper()),
    )
    sim_acc = demand_field.make_simulator_acceptance_function() if demand_field is not None else None
    for cname, pred in extra_groups:
        sats = [s for s in world.satellites if pred(s.name)]
        if not sats:
            continue
        sched_extra = ConstellationGroundScheduler(
            satellites=sats,
            ground_stations=GROUND_STATIONS,
            world=world,
            name=cname,
            acceptance_probability=0.88,
            acceptance_probability_function=sim_acc,
            execution_probability_function=execution_probability_function,
            acceptance_notification_delay_h=(ACCEPT_NOTIFY_MIN_H, ACCEPT_NOTIFY_MAX_H),
        )
        world.add_constellation(sched_extra)
        consts.append(sched_extra)

    fusion = make_fusion_satellite(world.satellites[0].orbit)
    world.satellites.append(fusion)

    def _fusion_acc(cr, sat, opp):
        return 1.0

    def _fusion_exec(cr, sat, opp):
        return 1.0

    sched = ConstellationGroundScheduler(
        satellites=[fusion],
        ground_stations=GROUND_STATIONS,
        world=world,
        name="Fusion",
        acceptance_probability=1.0,
        acceptance_probability_function=_fusion_acc,
        execution_probability_function=_fusion_exec,
        acceptance_notification_delay_h=(0.0, 0.0),
    )
    world.add_constellation(sched)
    consts.append(sched)
    return world, consts


def build_demand_field(events, min_time):
    cfg = DemandFieldConfig(use_constant_probability=False,
                            p_min=P_ACC_MIN, p_max=P_ACC_MAX)
    cfg.constellation_popularity = {
        **cfg.constellation_popularity,
        "USGS": 0.35,
        "Copernicus": 0.32,
    }
    demand = DemandField(config=cfg, reference_time=min_time,
                         horizon_s=LOOKAHEAD_HORIZON_H * 3600.0)
    for e in events:
        demand.add_spike(e["lat"], e["lon"], dt.datetime.fromisoformat(e["t_start"]))
    demand.precompute(CONSTELLATION_NAMES)
    demand.check_cost_reliability_tension(
        {k: v for k, v in PROVIDER_RATES.items() if k != "Fusion"}
    )
    return demand


def make_probability_functions(demand_field, events):
    demand_acc = demand_field.make_acceptance_prob_function()
    exec_fn = make_execution_prob_function(events)

    def acceptance_prob_function(constrained_request, satellite, obs_pass):
        if is_fusion_task(constrained_request):
            return 1.0
        if satellite is not None and str(getattr(satellite, "name", "")).startswith(FUSION_SAT_NAME):
            return 1.0
        return demand_acc(constrained_request, satellite, obs_pass)

    return acceptance_prob_function, exec_fn


def make_execution_cost_fn(all_constellations):
    sat_to_c = {sat: c.name for c in all_constellations for sat in c.satellites}

    def execution_cost_fn(task, satellite, obs_pass, dispatch_time, q_max=None):
        if is_fusion_task(task) or (
            satellite is not None and str(getattr(satellite, "name", "")).startswith(FUSION_SAT_NAME)
        ):
            return 0.0
        if q_max is None or q_max <= 0:
            q_max = 1.0
        rate = PROVIDER_RATES.get(sat_to_c.get(satellite, ""), PROVIDER_RATE_DEFAULT)
        try:
            valid = dispatch_time is not None and dispatch_time == dispatch_time
        except Exception:
            valid = False
        if valid and obs_pass is not None:
            rise = obs_pass.rise.time if hasattr(obs_pass, "rise") else obs_pass.highest.time
            lead_h = max(0.0, (rise - dispatch_time).total_seconds() / 3600.0)
            multiplier = 1.0 + LEAD_K * max(0.0, 1.0 - lead_h / LEAD_T_REF_H)
        else:
            multiplier = 1.0
        return rate * q_max * multiplier

    return execution_cost_fn


def _gorr_and_set_metrics(tasks, executed, reqs, exec_fn):
    """Primary: complete 3-sensor sets. Also Gorr-style 2/3-type counts."""
    by_gauge = {}
    for t in tasks:
        gid, kind = getattr(t, "rf_meta", (None, None))
        if gid is None:
            continue
        by_gauge.setdefault(gid, {})[kind] = t

    executed_kinds = {}
    for t in executed:
        gid, kind = getattr(t, "rf_meta", (None, None))
        if gid is None:
            continue
        executed_kinds.setdefault(gid, set()).add(kind)

    n_events = len(by_gauge)
    n_sets = sum(1 for gid, ks in executed_kinds.items()
                 if {"sar", "vis", "tir"} <= ks or "fusion" in ks)
    n_co2 = sum(1 for ks in executed_kinds.values()
                if len(ks & {"sar", "vis", "tir"}) >= 2)
    n_co3 = sum(1 for ks in executed_kinds.values()
                if {"sar", "vis", "tir"} <= ks)

    bookings = {"sar": 0, "vis": 0, "tir": 0}
    p_exec_sum = {"sar": 0.0, "vis": 0.0, "tir": 0.0}
    p_exec_n = {"sar": 0, "vis": 0, "tir": 0}
    name_to_task = {t.observation_request.name: t for t in tasks}
    for _, row in reqs.iterrows():
        t = name_to_task.get(getattr(row.get("request"), "name", None))
        if t is None:
            continue
        gid, kind = getattr(t, "rf_meta", (None, None))
        if kind not in bookings:
            continue
        bookings[kind] += 1
        rp = row.get("requested_pass")
        sat = row.get("satellite")
        if rp is None:
            continue
        try:
            p = exec_fn(t, sat, rp)
        except Exception:
            continue
        p_exec_sum[kind] += p
        p_exec_n[kind] += 1

    mean_p = {k: (p_exec_sum[k] / p_exec_n[k] if p_exec_n[k] else float("nan"))
              for k in bookings}
    per_event = {k: bookings[k] / n_events if n_events else 0.0 for k in bookings}
    return {
        "n_events": n_events,
        "n_sets_complete": n_sets,
        "set_completion_rate": n_sets / n_events if n_events else 0.0,
        "n_coobs_2": n_co2,
        "n_coobs_3": n_co3,
        "bookings_sar": bookings["sar"],
        "bookings_vis": bookings["vis"],
        "bookings_tir": bookings["tir"],
        "bookings_per_event_sar": per_event["sar"],
        "bookings_per_event_vis": per_event["vis"],
        "bookings_per_event_tir": per_event["tir"],
        "mean_p_exec_sar": mean_p["sar"],
        "mean_p_exec_vis": mean_p["vis"],
        "mean_p_exec_tir": mean_p["tir"],
    }


def run_one_scheduler(scheduler, seed, cached_satellites, events,
                      demand_field, min_time, max_time, results_dir,
                      plot_schedule=True):
    install_for()
    plots_dir = os.path.join(results_dir, "plots", scheduler)
    os.makedirs(plots_dir, exist_ok=True)

    acc_fn, exec_fn = make_probability_functions(demand_field, events)
    print(f"\n  Running {scheduler} (seed {seed})...  [RSS {rss_gb():.2f} GB]")
    random.seed(seed)
    np.random.seed(seed)

    world, constellations = create_world_and_constellations(
        cached_satellites, demand_field=demand_field,
        execution_probability_function=exec_fn,
    )
    register_river_phenomena(world, events, min_time, max_time)
    execution_cost_fn = make_execution_cost_fn(constellations)
    workflow = create_riverflow_workflow(
        events, min_time, max_time, max_num_instances=MAX_NUM_INSTANCES,
    )
    broker = Broker(constellations=constellations, world=world, name=f"Broker-{scheduler}")
    broker.add_workflow(workflow)
    world.add_broker(broker)

    for t in workflow.constrained_observation_requests:
        r = t.observation_request
        opps = fame_geometry.find_observation_opportunities([r], world.satellites)
        n = sum(len(v) for v in opps.get(r, {}).values())
        print(f"    {r.name:28s} [{(r.min_time - min_time).total_seconds()/3600:5.1f}h .."
              f"{(r.max_time - min_time).total_seconds()/3600:5.1f}h] "
              f"{str(r.instrument):18s} {n:3d} passes")

    m = None
    try:
        _horizon = dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
        common = dict(
            current_time=world.time,
            max_solver_time_s=MAX_SOLVER_TIME_S,
            update_timelines=False, update_requests=True, tax_rate=TAX_RATE,
            receding_horizon_duration=_horizon,
            max_reschedule_depth=10000,
            submission_cost_rate=SUBMISSION_COST,
            execution_cost_fn=execution_cost_fn,
            plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
            results_path=plots_dir,
        )
        if scheduler in ("stochastic_logical", "stochastic_log"):
            broker.schedule_workflow_redundant(
                use_ilp=True, use_stochastic=True,
                stochastic_formulation="general_logical_dag",
                acceptance_probability_function=acc_fn,
                execution_probability_function=exec_fn,
                solver_engine="GUROBI",
                enable_cancellations=ENABLE_CANCELLATIONS,
                **common,
            )
        elif scheduler == "deterministic":
            broker.schedule_workflow_redundant(
                use_ilp=True, use_stochastic=False, solver_engine="GUROBI",
                **common,
            )
        elif scheduler == "greedy":
            broker.schedule_workflow_redundant(
                use_ilp=False, use_stochastic=False, **common,
            )
        elif scheduler == "greedy_n":
            broker.schedule_workflow_redundant(
                use_ilp=False, use_stochastic=False,
                enable_cancellations=ENABLE_CANCELLATIONS,
                greedy_max_instances=MAX_NUM_INSTANCES,
                **common,
            )
        elif scheduler == "random":
            broker.schedule_workflow_redundant(
                use_ilp=False, use_stochastic=False,
                use_random=True, random_seed=seed, **common,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        _run_simulation_forward_shared(world)
        m = compute_metrics_v3(
            broker._workflow_graph, broker, ObservationStatus,
            submission_cost_rate=SUBMISSION_COST,
            execution_cost_fn=execution_cost_fn,
            verbose=True, sim_end_time=world.time,
        )

        tasks = list(broker._workflow_graph.nodes())
        reqs = broker._requests
        name_to_task = {t.observation_request.name: t for t in tasks}
        executed, best_q = set(), {}
        for _, row in reqs.iterrows():
            if row["status"] != ObservationStatus.DATA_RECEIVED:
                continue
            if not row["data_product"]:
                continue
            rp = row["requested_pass"]
            if rp is None:
                continue
            t = name_to_task.get(getattr(row["request"], "name", None))
            if t is None:
                continue
            executed.add(t)
            q = t.rewarder(rp.highest)
            if q > best_q.get(t, -1e9):
                best_q[t] = q

        no_pass = []
        for t in tasks:
            r = t.observation_request
            opps = fame_geometry.find_observation_opportunities([r], world.satellites)
            if sum(len(v) for v in opps.get(r, {}).values()) == 0:
                no_pass.append(r.name)
        reachable = compute_river_reachability(tasks, executed, no_pass)
        valid = executed & reachable
        extra = _gorr_and_set_metrics(tasks, executed, reqs, exec_fn)

        # Same as volcano / earthquake: one credit per completed task, the
        # best successful execution. Set completion stays a side metric.
        n_reach = len(reachable)
        realized = sum(best_q.get(t, 0.0) for t in valid)
        extra["set_quality"] = sum(
            best_q.get(t, 0.0) for t in valid
            if getattr(t, "rf_meta", (None, None))[1] == "fusion"
        )
        for k in ("realized_quality", "utility", "task_completion_rate",
                  "n_tasks_completed", "n_tasks_reachable"):
            m.pop(k, None)
        m["n_tasks_reachable"] = n_reach
        m["n_tasks_completed"] = len(valid)
        m["task_completion_rate"] = (len(valid) / n_reach) if n_reach else 0.0
        m["realized_quality"] = realized
        m["utility"] = realized - m["total_cost"]
        m.update(extra)
        m["scheduler"] = scheduler
        m["seed"] = seed
        m["dispatch_lead_h"] = DISPATCH_LEAD_H
        m["sim_start"] = SIMULATION_START.isoformat()
        m["p_exec_sar_design"] = P_EXEC_SAR

        print(f"   [Sets] {extra['n_sets_complete']}/{extra['n_events']} complete  "
              f"coobs2={extra['n_coobs_2']} coobs3={extra['n_coobs_3']}  "
              f"book/evt SAR={extra['bookings_per_event_sar']:.2f} "
              f"VIS={extra['bookings_per_event_vis']:.2f} "
              f"TIR={extra['bookings_per_event_tir']:.2f}  "
              f"Q={realized:.1f} U={m['utility']:.1f}")

        execution_details = m.pop("execution_details", [])
        with open(os.path.join(results_dir, f"executions_seed{seed:04d}_{scheduler}.json"), "w") as f:
            json.dump(execution_details, f, indent=2, default=str)
        run_file = os.path.join(results_dir, f"run_seed{seed:04d}_{scheduler}.json")
        with open(run_file, "w") as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  [Saved] {run_file}")
        try:
            from riverflow_visualize import dump_bookings
            dump_bookings(
                os.path.join(results_dir, f"bookings_seed{seed:04d}_{scheduler}.json"),
                broker, tasks,
            )
        except Exception as viz_err:
            print(f"  [Viz] booking dump skipped: {viz_err}")
    except Exception as e:
        import traceback
        print(f"    Broker Error ({scheduler}, seed {seed}): {e}")
        traceback.print_exc()

    try:
        del broker, workflow, world, constellations
    except Exception:
        pass
    plt.close("all")
    gc.collect()
    return m


def aggregate(results_dir, records=None):
    if records is None:
        records = []
        for path in sorted(glob.glob(os.path.join(results_dir, "run_*.json"))):
            try:
                with open(path) as f:
                    r = json.load(f)
                for k, v in list(r.items()):
                    if isinstance(v, str) and k not in ("scheduler", "sim_start"):
                        try:
                            r[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                records.append(r)
            except Exception as exc:
                print(f"[Warning] Could not read {path}: {exc}")
    if not records:
        print("[Warning] No records found.")
        return []
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(results_dir, "metrics_v2.csv"), index=False)

    print("\n" + "=" * 70)
    print("RIVER HIGH-FLOW CO-OBSERVATION — SUMMARY")
    print("=" * 70)
    for sched in SCHEDULERS:
        sub = df[df["scheduler"] == sched]
        if sub.empty:
            continue
        print(f"\n{sched.upper()}  (n={len(sub)})")
        if "set_completion_rate" in sub:
            print(f"  Set completion      : {sub['set_completion_rate'].mean():.3f} ± {sub['set_completion_rate'].std():.3f}")
        print(f"  Realized quality    : {sub['realized_quality'].mean():.1f}")
        print(f"  Net utility         : {sub['utility'].mean():.1f} ± {sub['utility'].std():.1f}")
        print(f"  Total cost          : {sub['total_cost'].mean():.1f}")
        if "bookings_per_event_vis" in sub:
            print(f"  Book/evt VIS / TIR / SAR : "
                  f"{sub['bookings_per_event_vis'].mean():.2f} / "
                  f"{sub['bookings_per_event_tir'].mean():.2f} / "
                  f"{sub['bookings_per_event_sar'].mean():.2f}")
        if "n_coobs_3" in sub:
            print(f"  Co-obs 2-type / 3-type : "
                  f"{sub['n_coobs_2'].mean():.1f} / {sub['n_coobs_3'].mean():.1f}")
    try:
        plot_cost_frontier(records, SCHEDULERS,
                           out_path=os.path.join(results_dir, "cost_frontier.png"),
                           y="set_completion_rate", x="total_cost")
    except Exception:
        plot_cost_frontier(records, SCHEDULERS,
                           out_path=os.path.join(results_dir, "cost_frontier.png"),
                           y="task_completion_rate", x="total_cost")
    return records


def run_census(cached_satellites, events):
    install_for()
    rows = opportunity_census(
        events, cached_satellites, fame_geometry.find_observation_opportunities)
    ok = print_census_gate(rows)
    return ok, rows


def run_comparison(num_monte_carlo_runs=NUM_MC_RUNS, results_dir=None, plot_schedule=True):
    install_for()
    sync_simulation_clock()
    if results_dir is None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = os.path.join("results", f"riverflow_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    cached_satellites = load_satellites_for_river(SIMULATION_START, LOOKAHEAD_HORIZON_H)
    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
    events = load_events()
    print(f"[Scenario] day={load_scenario_day().date()}  events={len(events)}  "
          f"sim=[{min_time}, {max_time}]  notify=({ACCEPT_NOTIFY_MIN_H},{ACCEPT_NOTIFY_MAX_H})h before pass")
    ok, rows = run_census(cached_satellites, events)
    with open(os.path.join(results_dir, "census.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if not ok:
        print("[Census] Gate failed. Continuing because the driver was asked "
              "to run; treat results as diagnostic.")

    demand_field = build_demand_field(events, min_time)
    all_records = []
    for run_idx in range(num_monte_carlo_runs):
        run_seed = 42 + run_idx
        for scheduler in SCHEDULERS:
            m = run_one_scheduler(
                scheduler, run_seed, cached_satellites, events, demand_field,
                min_time, max_time, results_dir, plot_schedule=plot_schedule,
            )
            if m is not None:
                all_records.append(m)
    aggregate(results_dir, records=all_records)
    return all_records


def main():
    global SIMULATION_START, DISPATCH_LEAD_H, SCHEDULERS

    parser = argparse.ArgumentParser(description="River high-flow scheduler comparison")
    parser.add_argument("--scheduler", choices=SCHEDULERS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--runs", type=int, default=NUM_MC_RUNS)
    parser.add_argument("--no-schedule-plots", action="store_true")
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--lead-h", type=float, default=None,
                        help="Dispatch lead L_c in hours (moves sim start).")
    parser.add_argument("--viz", action="store_true",
                        help="Write Leaflet animation HTML into the results dir.")
    parser.add_argument("--build-cache", action="store_true",
                        help="Force USGS(+ERA5) cache rebuild then exit.")
    args = parser.parse_args()

    if args.build_cache:
        ensure_cache(force=True, allow_toy=False)
        return

    # Prefer USGS cache; sync clock before anything uses SIMULATION_START.
    ensure_cache(allow_toy=False)
    sync_simulation_clock()

    if args.lead_h is not None:
        DISPATCH_LEAD_H = float(args.lead_h)
        sync_simulation_clock()

    if args.aggregate:
        if not args.results_dir:
            parser.error("--aggregate requires --results-dir")
        aggregate(args.results_dir)
        if args.viz:
            from riverflow_visualize import write_animation_html
            write_animation_html(args.results_dir)
        return

    if args.census_only:
        install_for()
        sats = load_satellites_for_river(SIMULATION_START, LOOKAHEAD_HORIZON_H)
        run_census(sats, load_events())
        return

    if args.scheduler:
        if args.seed is None:
            parser.error("--scheduler requires --seed")
        results_dir = args.results_dir or os.path.join(
            "results", f"riverflow_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)
        install_for()
        cached = load_satellites_for_river(SIMULATION_START, LOOKAHEAD_HORIZON_H)
        min_time = SIMULATION_START
        max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
        events = load_events()
        run_census(cached, events)
        demand = build_demand_field(events, min_time)
        run_one_scheduler(args.scheduler, args.seed, cached, events, demand,
                          min_time, max_time, results_dir,
                          plot_schedule=not args.no_schedule_plots)
        if args.viz:
            from riverflow_visualize import write_animation_html
            write_animation_html(results_dir, seed=args.seed,
                                 scheduler=args.scheduler)
        return

    results_dir = args.results_dir or os.path.join(
        "results", f"riverflow_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    run_comparison(num_monte_carlo_runs=args.runs, results_dir=results_dir,
                   plot_schedule=not args.no_schedule_plots)
    if args.viz:
        from riverflow_visualize import write_animation_html
        write_animation_html(results_dir)


if __name__ == "__main__":
    main()
