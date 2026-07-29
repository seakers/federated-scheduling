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

# demand-based metrics + paired analysis + frontier
from fame_metrics import compute_metrics_v2, paired_summary, plot_cost_frontier

# ============================ Configuration =================================
# SIMULATION_START is a module global because the geometry helpers below read
# it. In single-run mode it is overwritten from --start BEFORE anything uses
# it, so all processes in a campaign share identical orbital geometry.
SIMULATION_START = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

lookahead_horizon_h = 18
FOLLOW_UP_INTERVAL_H = 3
MAX_SOLVER_TIME_S = 70
MAX_NUM_INSTANCES = 5
NUM_MC_RUNS = 2

# === COST CONFIGURATION ===
TAX_RATE = 0.0              # Legacy per-booking tax (disabled)
SUBMISSION_COST = 0.05      # Unconditional booking submission overhead
EXEC_COST = 0.2             # Conditional execution cost if accepted

SCHEDULERS = ['stochastic_log','deterministic', 'greedy']


def load_volcano_locations_from_database() -> list[Location]:
    """
    Reads the global GVP Holocene databases, merges the eruption catalogs,
    and isolates high-priority target positions (VEI > 4, Start Year > 1900).
    """
    print("[Data] Reading GVP Volcano and Eruption database files...")
    volcano_df = pd.read_excel('data/GVP_Volcano_List_Holocene_202606021456.xlsx', header=1)
    eruption_df = pd.read_excel('data/GVP_Eruption_List_Holocene_20260424.xlsx',
                                sheet_name="Eruption List", header=1)

    eruptions = eruption_df.merge(volcano_df, on='Volcano Name', how='left')

    filtered_volcanoes = [
        Location(
            lon_deg=row['Longitude'],
            lat_deg=row['Latitude'],
            alt_km=float(row['Elevation (m)']) / 1e3,
            name=row['Volcano Name'],
        )
        for ix, row in eruptions.iterrows()
        if row['VEI'] > 4 and row['Start Year'] > 1900
    ]

    unique_locations = list(set(filtered_volcanoes))
    print(f"[Data] Loaded {len(unique_locations)} high-VEI target volcanoes into workspace context.")
    return unique_locations


def load_satellites_once() -> list[Satellite]:
    """Dynamically loads all LEO satellites from TLE files using notebook approach."""
    from pyorbital.orbital import Orbital
    import pyorbital

    tle_files = glob.glob("tles/all_tles_*.txt")
    if not tle_files:
        raise FileNotFoundError("No text TLE files found in the tles/ folder context.")
    tle_file_txt = sorted(tle_files)[-1]

    swaths_at_nadir_km = {
        "SKYSAT-A": 8, "SKYSAT-B": 8,
        "SKYSAT-C1": 5.9, "SKYSAT-C2": 5.9, "SKYSAT-C3": 5.9, "SKYSAT-C4": 5.9,
        "SKYSAT-C5": 5.9, "SKYSAT-C6": 5.9, "SKYSAT-C7": 5.9, "SKYSAT-C8": 5.9,
        "SKYSAT-C9": 5.9, "SKYSAT-C10": 5.9, "SKYSAT-C11": 5.9, "SKYSAT-C12": 5.9,
        "SKYSAT-C13": 5.9,
        "PELICAN-1 3001": 8, "PELICAN-2 3009": 8, "PELICAN-3 300A": 8,
        "PELICAN-4 300B": 8, "PELICAN-5 300C": 8, "PELICAN-6 300D": 8,
        "TANAGER-4001": 18,
        "UMBRA-07": 8, "UMBRA-09": 8, "UMBRA-10": 8, "UMBRA-11": 8,
        "CAPELLA-11 (ACADIA)": 10, "CAPELLA-13 (ACADIA)": 10, "CAPELLA-14 (ACADIA)": 10,
        "CAPELLA-15 (ACADIA)": 10, "CAPELLA-16 (ACADIA)": 10, "CAPELLA-17 (ACADIA)": 10,
        "LOFT YAM-6": 19.8,
        "Ubotica CogniSat-6 HAMMER": 20, "Ubotica ACCENTURE-1 SUAC": 20,
        "Mission Control Persistence": 100,
        "AEROCUBE 18A": 80, "AEROCUBE 18B": 80,
    }

    flock_names = []
    iceye_names = []
    with open(tle_file_txt, 'r') as file:
        for line in file:
            if line.startswith("FLOCK"):
                flock_names.append(line.strip())
            elif line.startswith("ICEYE"):
                iceye_names.append(line.strip())

    for dove_name in flock_names:
        swaths_at_nadir_km[dove_name] = 16.4
    for iceye_name in iceye_names:
        swaths_at_nadir_km[iceye_name] = 100

    tle_to_display = {
        "SKYSAT 1": "SKYSAT-A", "SKYSAT 2": "SKYSAT-B",
        "SKYSAT C1": "SKYSAT-C1", "SKYSAT C2": "SKYSAT-C2", "SKYSAT C3": "SKYSAT-C3",
        "SKYSAT C4": "SKYSAT-C4", "SKYSAT C5": "SKYSAT-C5", "SKYSAT C6": "SKYSAT-C6",
        "SKYSAT C7": "SKYSAT-C7", "SKYSAT C8": "SKYSAT-C8", "SKYSAT C9": "SKYSAT-C9",
        "SKYSAT C10": "SKYSAT-C10", "SKYSAT C11": "SKYSAT-C11", "SKYSAT C12": "SKYSAT-C12",
        "SKYSAT C13": "SKYSAT-C13",
    }

    sat_constellation_map = {
        "SKYSAT": ("Planet", InstrumentType.RGB),
        "PELICAN": ("Planet", InstrumentType.RGB),
        "TANAGER": ("Planet", InstrumentType.HYPERSPECTRAL),
        "FLOCK": ("Planet", InstrumentType.RGB),
        "UMBRA": ("Umbra", InstrumentType.SAR),
        "CAPELLA": ("Capella", InstrumentType.SAR),
        "ACADIA": ("Capella", InstrumentType.SAR),
        "YAM": ("LOFT", InstrumentType.HYPERSPECTRAL),
        "LOFT": ("LOFT", InstrumentType.HYPERSPECTRAL),
        "HAMMER": ("Ubotica", InstrumentType.HYPERSPECTRAL),
        "ACCENTURE": ("Ubotica", InstrumentType.HYPERSPECTRAL),
        "LEMUR": ("Mission Control", InstrumentType.RGB),
        "KRISH": ("Mission Control", InstrumentType.RGB),
        "PERSISTENCE": ("Mission Control", InstrumentType.RGB),
        "AEROCUBE": ("Aerospace", InstrumentType.RGB),
        "ICEYE": ("ICEYE", InstrumentType.SAR),
    }

    satellites = []
    skipped_count = 0

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(seconds=3600 * lookahead_horizon_h)
    display_to_tle = {v: k for k, v in tle_to_display.items()}
    for display_name, swath_km in swaths_at_nadir_km.items():
        tle_name = display_to_tle.get(display_name, display_name)
        constellation = "Unknown"
        instrument = InstrumentType.RGB
        for key, (const, inst) in sat_constellation_map.items():
            if key in display_name.upper():
                constellation = const
                instrument = inst
                break

        try:
            orbit = Orbital(tle_name, tle_file=tle_file_txt)
            try:
                _ = orbit.get_lonlatalt(min_time)
                _ = orbit.get_lonlatalt(max_time)
            except (NotImplementedError, Exception):
                print(f"[Skipped] {display_name} failed orbit propagation tests (decayed/deep space).")
                skipped_count += 1
                continue

            sat = Satellite(display_name, orbit, instruments=[instrument],
                            has_continuous_isl_to_ground=True)
            _semi_major = sat.orbit.orbit_elements.semi_major_axis * pyorbital.orbital.A
            _altitude = _semi_major - pyorbital.orbital.A
            _fov = 2 * np.atan2(swath_km / 2, _altitude)
            sat.instrument_fov_rad = {it: _fov for it in sat.instruments}
            satellites.append(sat)
        except Exception as e:
            print(f"[Warning] Could not load {display_name} (TLE: {tle_name}): {e}")
            skipped_count += 1
            continue

    print(f"[Init] Loaded {len(satellites)} satellites (skipped {skipped_count} decayed/invalid).\n")
    return satellites


def create_world_and_constellations(cached_satellites: list[Satellite],
                                    demand_field: DemandField = None):
    """Creates fresh simulation scopes using copied pre-cached orbital models."""
    #local_satellites = copy.deepcopy(cached_satellites)
    local_satellites = load_satellites_once()
    world = World(satellites=local_satellites)
    world.time = SIMULATION_START

    ground_stations = [
        Location(-79.55, 8.9833, 0.028, "KSAT Panama"),
        Location(-51.73363, 64.182789, 0, "KSAT Nuuk"),
        Location(2.53219, -72.01243, 0, "KSAT Troll"),
        Location(142.3689, 43.8, 0, "KSAT Hokkaido"),
        Location(103.9915, 1.3661, 0, "KSAT Singapore"),
        Location(-70.85021, -52.93279, 0, "KSAT Punta Arenas"),
        Location(127.7766, 26.4055, 0, "KSAT Okinawa"),
        Location(57.5565, -20.1142, 0, "KSAT Mauritius"),
        Location(22.62216, 37.84604, 0, "KSAT Nemea"),
        Location(31.12509, 70.36779, 0, "KSAT Vardo"),
        Location(15.39964, 78.22875, 0, "KSAT Svalbard"),
    ]

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
            ground_stations=ground_stations,
            world=world,
            name=name,
            acceptance_probability=legacy_p,
            acceptance_probability_function=_sim_acc_fn,
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


def create_volcano_workflow(volcano_locations, min_time, max_time):
    """Generates detection + follow-up risk chains."""
    constrained_requests = []
    timelines = {}

    def success_declarer_eruption(data_product):
        return len(data_product) > 0

    def rewarder_observation(opportunity: ObservationOpportunity, preferred_zenith_angle_deg=45):
        static_reward = 50
        look_angle_reward = abs(90. - opportunity.look_angle_dec_deg) / 90.
        zenith_angle_reward = abs(preferred_zenith_angle_deg - opportunity.sun_zenith_angle_deg) / 90
        range_reward = 1 / (opportunity.range_km / 1000)
        return look_angle_reward + zenith_angle_reward + range_reward + static_reward

    for volcano in volcano_locations:
        timeline = Timeline(
            name=volcano.name,
            initial_time=min_time,
            initial_value=1.0,
            initial_rate=-1.0 / (3600 * 24),
            min_value=-30,
            max_value=30
        )
        timelines[volcano] = timeline

        detection_request = ObservationRequest(
            lon_deg=volcano.lon_deg, lat_deg=volcano.lat_deg, alt_km=volcano.alt_km,
            min_time=min_time, max_time=min_time + dt.timedelta(hours=12),
            instrument=InstrumentType.RGB, request_name=f"{volcano.name}_detection",
            min_elevation_deg=20.0
        )

        detection_task = ConstrainedObservationRequest(
            name=f"{volcano.name}_detection",
            observation_request=detection_request,
            is_mandatory=True,
            timeline_impacts=[
                TaskTimelineImpact(
                    timeline=timeline, time=TaskImpactTime.POST,
                    type=ImpactType.ADDITION, value=1.0
                )
            ],
            rewarder=rewarder_observation,
            success_declarer=success_declarer_eruption,
            request_group=volcano.name,
            max_num_instances=MAX_NUM_INSTANCES
        )
        constrained_requests.append(detection_task)

        for follow_up_ix in range(0, lookahead_horizon_h, FOLLOW_UP_INTERVAL_H):
            follow_up_request = ObservationRequest(
                lon_deg=volcano.lon_deg, lat_deg=volcano.lat_deg, alt_km=volcano.alt_km,
                min_time=min_time + dt.timedelta(hours=follow_up_ix), max_time=max_time,
                instrument=InstrumentType.RGB,
                request_name=f"{volcano.name}_followup_{follow_up_ix}h",
                min_elevation_deg=20.0
            )

            follow_up_task = ConstrainedObservationRequest(
                name=f"{volcano.name}_followup_{follow_up_ix}h",
                observation_request=follow_up_request,
                is_mandatory=False,
                task_constraints=[
                    Constraint(
                        ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET,
                        detection_task, {'offset': dt.timedelta(hours=follow_up_ix)}
                    ),
                ],
                timeline_constraints=[
                    TaskTimelineConstraint(
                        timeline=timeline, time=TaskImpactTime.PRE,
                        type=TimelineConstraintType.GREATER_OR_EQUAL, value=0.1
                    )
                ],
                rewarder=rewarder_observation,
                success_declarer=success_declarer_eruption,
                request_group=volcano.name,
                max_num_instances=MAX_NUM_INSTANCES
            )
            constrained_requests.append(follow_up_task)

    workflow = Workflow(
        constrained_observation_requests=constrained_requests,
        timelines=list(timelines.values()),
        timeline_updater=lambda c, r, t: t,
        request_updater=lambda c, r, t: None
    )
    return workflow


# ======================= shared run-time helpers ============================

def _rss_gb():
    """Peak resident set size in GB (diagnostic only; no behaviour change)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        return rss / 1e9 if rss > 1e7 else rss / 1e6
    except Exception:
        return float('nan')


def run_simulation_forward(world, max_safety_limit=40000):
    ticks = 0
    last_progress_report = 0
    while True:
        retcode = world.tick(print_forbidden_prefixes=[
            "Downlink", "End of downlink", "Unlock uplink", "Unlock satellite after obs"])
        ticks += 1
        if ticks - last_progress_report >= 5000:
            print(f"  [Sim Progress] {ticks} ticks, Current time: {world.time}")
            last_progress_report = ticks
        if retcode == 0:
            break
        if ticks >= max_safety_limit:
            print(f"  [Warning] Simulation safety cut-off invoked at {max_safety_limit} ticks.")
            break
    print(f"  [Sim] Event-loop finished. Total Ticks: {ticks}, Concluded Simulation Clock: {world.time}")


def build_demand_field(volcano_db_locations, min_time):
    """Demand field: single source of truth for simulator and planner."""
    _demand_cfg = DemandFieldConfig(use_constant_probability=False)
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

    def execution_prob_function(constrained_request, satellite, obs_pass):
        look_angle = abs(90.0 - obs_pass.highest.look_angle_dec_deg)
        execution_prob = 0.95 - (look_angle / 90.0) * 0.20
        return max(0.7, min(0.99, execution_prob))

    return acceptance_prob_function, execution_prob_function


def run_one_scheduler(scheduler, seed, cached_satellites, volcano_db_locations,
                      demand_field, min_time, max_time, results_dir,
                      plot_schedule=True):
    """
    Build world + workflow + broker for ONE scheduler and ONE seed, run the
    simulation, compute metrics, and write run_seed<seed>_<scheduler>.json.

    Scheduling behaviour is identical to the previous in-process version: the
    RNG is reset to `seed` first, so every scheduler faces the same draws.
    Returns the metrics dict (or None on error).
    """
    plots_dir = os.path.join(results_dir, "plots", scheduler)
    os.makedirs(plots_dir, exist_ok=True)

    acceptance_prob_function, execution_prob_function = make_probability_functions(demand_field)

    print(f"\n  Running {scheduler} (seed {seed})...  [RSS {_rss_gb():.2f} GB]")
    random.seed(seed)
    np.random.seed(seed)

    world, constellations = create_world_and_constellations(cached_satellites,
                                                            demand_field=demand_field)
    workflow = create_volcano_workflow(volcano_db_locations, min_time, max_time)
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
                max_reschedule_depth=1,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir
            )
        elif scheduler == 'deterministic':
            broker.schedule_workflow_redundant(
                current_time=world.time, use_ilp=True, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=False, tax_rate=TAX_RATE,
                max_reschedule_depth=1,
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
                max_reschedule_depth=1,
                plot_schedule=plot_schedule, save_schedule_plot=plot_schedule,
                results_path=plots_dir
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        run_simulation_forward(world)

        m = compute_metrics_v2(
            broker._workflow_graph, broker, ObservationStatus,
            submission_cost_rate=SUBMISSION_COST, execution_cost_rate=EXEC_COST,
            verbose=True,
        )
        m['scheduler'] = scheduler
        m['seed'] = seed
        m['tax_rate'] = TAX_RATE
        m['max_num_instances'] = MAX_NUM_INSTANCES
        m['sim_start'] = SIMULATION_START.isoformat()

        run_file = os.path.join(results_dir, f"run_seed{seed:04d}_{scheduler}.json")
        with open(run_file, 'w') as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  [Saved] {run_file}")
    except Exception as e:
        print(f"    Broker Error ({scheduler}, seed {seed}): {e}")

    # Release everything this run allocated. (In single-run mode the process
    # exits right after, which is the real guarantee; this helps the legacy
    # in-process loop.)
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
    Panels (b) and (c) are the ones that support a claim: they remove the
    seed-to-seed variance that otherwise swamps the mean in panel (a).
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
        print(f"  Bookings submitted   : {sub['n_submissions'].mean():.1f} "
              f"(accepted {sub['n_accepted'].mean():.1f}, executed {sub['n_executed'].mean():.1f})")
        print(f"  TRUE passes/task     : {sub['submitted_passes_per_task'].mean():.2f} submitted, "
              f"{sub['exec_passes_per_completed'].mean():.2f} executed/completed")
        print(f"  Rejection rate (diag): {100 * sub['rejection_rate'].mean():.1f}%")

    summary_cols = [c for c in [
        'task_completion_rate', 'group_completion_rate', 'realized_quality',
        'utility', 'total_cost', 'n_submissions', 'n_accepted', 'n_executed',
        'n_rejected', 'submitted_passes_per_task', 'exec_passes_per_completed',
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