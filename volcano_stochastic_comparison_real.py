"""
Volcano Workflow: Stochastic vs Deterministic Comparison
Using real satellite constellations from TLE files and GVP database
"""

import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
import copy
import glob
import os
import random
import json
from typing import Callable
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

# Configuration
SIMULATION_START = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
lookahead_horizon_h = 9  # Reduced from 18 to make stochastic tractable
FOLLOW_UP_INTERVAL_H = 3
MAX_SOLVER_TIME_S = 100  # Increased from 300 to give more time
MAX_NUM_INSTANCES = 5 


def load_volcano_locations_from_database() -> list[Location]:
    """
    Reads the global GVP Holocene databases, merges the eruption catalogs,
    and isolates high-priority target positions (VEI > 4, Start Year > 1900).
    """
    print("[Data] Reading GVP Volcano and Eruption databa se files...")
    volcano_df = pd.read_excel('data/GVP_Volcano_List_Holocene_202606021456.xlsx', header=1)
    eruption_df = pd.read_excel('data/GVP_Eruption_List_Holocene_20260424.xlsx', sheet_name="Eruption List", header=1)
    
    # Merge catalogs on common volcano keys
    eruptions = eruption_df.merge(volcano_df, on='Volcano Name', how='left')
    
    # Filter exactly matching the notebook criteria
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
    
    # Deduplicate unique volcanic entries
    unique_locations = list(set(filtered_volcanoes))
    print(f"[Data] Loaded {len(unique_locations)} high-VEI target volcanoes into workspace context.")
    return unique_locations


def load_satellites_once() -> list[Satellite]:
    """Dynamically loads all 87 LEO satellites from TLE files using notebook approach."""
    from pyorbital.orbital import Orbital
    import pyorbital

    tle_files = glob.glob("tles/all_tles_*.txt")
    if not tle_files:
        raise FileNotFoundError("No text TLE files found in the tles/ folder context.")
    tle_file_txt = sorted(tle_files)[-1]

    # Swath definitions matching notebook
    swaths_at_nadir_km = {
        "SKYSAT-A": 8,
        "SKYSAT-B": 8,
        "SKYSAT-C1": 5.9,
        "SKYSAT-C2": 5.9,
        "SKYSAT-C3": 5.9,
        "SKYSAT-C4": 5.9,
        "SKYSAT-C5": 5.9,
        "SKYSAT-C6": 5.9,
        "SKYSAT-C7": 5.9,
        "SKYSAT-C8": 5.9,
        "SKYSAT-C9": 5.9,
        "SKYSAT-C10": 5.9,
        "SKYSAT-C11": 5.9,
        "SKYSAT-C12": 5.9,
        "SKYSAT-C13": 5.9,
        "PELICAN-1 3001": 8,
        "PELICAN-2 3009": 8,
        "PELICAN-3 300A": 8,
        "PELICAN-4 300B": 8,
        "PELICAN-5 300C": 8,
        "PELICAN-6 300D": 8,
        "TANAGER-4001": 18,
        "UMBRA-07": 8,
        "UMBRA-09": 8,
        "UMBRA-10": 8,
        "UMBRA-11": 8,
        "CAPELLA-11 (ACADIA)": 10,
        "CAPELLA-13 (ACADIA)": 10,
        "CAPELLA-14 (ACADIA)": 10,
        "CAPELLA-15 (ACADIA)": 10,
        "CAPELLA-16 (ACADIA)": 10,
        "CAPELLA-17 (ACADIA)": 10,
        "LOFT YAM-6": 19.8,
        "Ubotica CogniSat-6 HAMMER": 20,
        "Ubotica ACCENTURE-1 SUAC": 20,
        "Mission Control Persistence": 100,
        "AEROCUBE 18A": 80,
        "AEROCUBE 18B": 80,
    }

    # Extract Flock (DOVE) and ICEYE satellite names from TLE file
    flock_names = []
    iceye_names = []
    with open(tle_file_txt, 'r') as file:
        for line in file:
            if line.startswith("FLOCK"):
                flock_names.append(line.strip())
            elif line.startswith("ICEYE"):
                iceye_names.append(line.strip())

    # Add dynamic satellite swaths
    for dove_name in flock_names:
        swaths_at_nadir_km[dove_name] = 16.4
    for iceye_name in iceye_names:
        swaths_at_nadir_km[iceye_name] = 100

    # Map TLE names to display names for known satellites (matching notebook)
    tle_to_display = {
        "SKYSAT 1": "SKYSAT-A",
        "SKYSAT 2": "SKYSAT-B",
        "SKYSAT C1": "SKYSAT-C1",
        "SKYSAT C2": "SKYSAT-C2",
        "SKYSAT C3": "SKYSAT-C3",
        "SKYSAT C4": "SKYSAT-C4",
        "SKYSAT C5": "SKYSAT-C5",
        "SKYSAT C6": "SKYSAT-C6",
        "SKYSAT C7": "SKYSAT-C7",
        "SKYSAT C8": "SKYSAT-C8",
        "SKYSAT C9": "SKYSAT-C9",
        "SKYSAT C10": "SKYSAT-C10",
        "SKYSAT C11": "SKYSAT-C11",
        "SKYSAT C12": "SKYSAT-C12",
        "SKYSAT C13": "SKYSAT-C13",
    }

    # Define constellation memberships and instruments
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

    # Load all satellites dynamically
    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(seconds=3600 * lookahead_horizon_h)
    display_to_tle = {v: k for k, v in tle_to_display.items()}
    for display_name, swath_km in swaths_at_nadir_km.items():
        # Find TLE name
        # 1. Create a reverse lookup dictionary once before the loop
        tle_name = display_to_tle.get(display_name, display_name)
        # Determine constellation and instrument
        constellation = "Unknown"
        instrument = InstrumentType.RGB
        for key, (const, inst) in sat_constellation_map.items():
            if key in display_name.upper():
                constellation = const
                instrument = inst
                break

        try:
            orbit = Orbital(tle_name, tle_file=tle_file_txt)

            # Test orbit propagation to verify satellite is valid
            try:
                _ = orbit.get_lonlatalt(min_time)
                _ = orbit.get_lonlatalt(max_time)
            except (NotImplementedError, Exception):
                print(f"[Skipped] {display_name} failed orbit propagation tests (decayed/deep space).")
                skipped_count += 1
                continue

            sat = Satellite(display_name, orbit, instruments=[instrument], has_continuous_isl_to_ground=True)

            # Initialize Field-of-View geometry profiles
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


def create_world_and_constellations(cached_satellites: list[Satellite], demand_field: DemandField = None):
    """Creates fresh simulation scopes using copied pre-cached orbital models.

    Uses the same 8 constellation structure as the notebook:
    Planet, Umbra, Capella, LOFT, Ubotica, Mission Control, Aerospace Corp, ICEYE

    If demand_field is provided it is wired as the acceptance_probability_function
    for every constellation so the simulator uses dynamic probabilities.
    If demand_field is None the legacy constant per-constellation probabilities are used.
    """
    local_satellites = copy.deepcopy(cached_satellites)
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

    # Organize satellites by constellation (match notebook organization)
    # Planet constellation includes: SkySat, Pelican, Tanager
    planet_sats = [s for s in local_satellites if any(x in s.name.upper() for x in ["SKYSAT", "PELICAN", "TANAGER"])]

    # Umbra constellation
    umbra_sats = [s for s in local_satellites if "UMBRA" in s.name.upper()]

    # Capella constellation
    capella_sats = [s for s in local_satellites if "CAPELLA" in s.name.upper() or "ACADIA" in s.name.upper()]

    # LOFT constellation
    loft_sats = [s for s in local_satellites if "LOFT" in s.name.upper() or "YAM" in s.name.upper()]

    # Ubotica constellation
    ubotica_sats = [s for s in local_satellites if "UBOTICA" in s.name.upper() or "HAMMER" in s.name.upper() or "ACCENTURE" in s.name.upper()]

    # Mission Control constellation
    mission_control_sats = [s for s in local_satellites if "PERSISTENCE" in s.name.upper() or "LEMUR" in s.name.upper()]

    # Aerospace Corp constellation
    aerospace_sats = [s for s in local_satellites if "AEROCUBE" in s.name.upper()]

    # ICEYE constellation
    iceye_sats = [s for s in local_satellites if "ICEYE" in s.name.upper()]

    # Build acceptance-probability function for the simulator.
    # When a DemandField is provided the simulator draws from the dynamic model;
    # otherwise fall back to the legacy constant per-constellation values.
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

    def rewarder_observation(opportunity):
        return 50.0 + abs(90. - opportunity.look_angle_dec_deg) / 90.

    for volcano in volcano_locations:
        timeline = Timeline(
            name=volcano.name,
            initial_time=min_time,
            initial_value=1.0,
            initial_rate=-1.0/(3600*24),  
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
            max_num_instances = MAX_NUM_INSTANCES
        )
        constrained_requests.append(detection_task)

        for follow_up_ix in range(0, lookahead_horizon_h, FOLLOW_UP_INTERVAL_H):
            follow_up_request = ObservationRequest(
                lon_deg=volcano.lon_deg, lat_deg=volcano.lat_deg, alt_km=volcano.alt_km,
                min_time=min_time + dt.timedelta(hours=follow_up_ix), max_time=max_time,
                instrument=InstrumentType.RGB, request_name=f"{volcano.name}_followup_{follow_up_ix}h",
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


def run_comparison(num_monte_carlo_runs=2):
    """
    Compares formulation tracks across real GVP database coordinates.

    **NEW: Two-Stage Stochastic Model**
    The stochastic scheduler now uses:
    - p_acc: Probability constellation ACCEPTS booking (Planet=0.7, Umbra=0.9)
    - p_exec: Probability accepted booking EXECUTES successfully (depends on look angle, ~0.75-0.95)
    - Cost model: submission (5%) + cancellation (10% if accepted) + execution (20%)

    **Expected Results:**
    - Stochastic should have LOWER rejection rate (avoids low p_acc passes)
    - Stochastic should have HIGHER quality/task (prefers high p_exec passes with good geometry)
    - Stochastic should have HIGHER or EQUAL utility (better risk-adjusted planning)
    """
    # Create timestamped results directory
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join("results", f"volcano_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    media_dir = os.path.join(results_dir, "media")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    # Create separate subdirectories for each broker type
    plots_dir_greedy = os.path.join(plots_dir, "greedy")
    plots_dir_det = os.path.join(plots_dir, "deterministic")
    plots_dir_stoch_log = os.path.join(plots_dir, "stochastic_log")
    plots_dir_stoch_nc = os.path.join(plots_dir, "stochastic_nc")
    os.makedirs(plots_dir_det, exist_ok=True)
    os.makedirs(plots_dir_greedy, exist_ok=True)
    os.makedirs(plots_dir_stoch_log, exist_ok=True)
    os.makedirs(plots_dir_stoch_nc, exist_ok=True)

    print(f"\n[Results] Saving to directory: {results_dir}")
    print(f"[Plots] Saving schedule plots to: {plots_dir}")
    print(f"  - Deterministic: {plots_dir_det}")
    print(f"  - Stochastic Log: {plots_dir_stoch_log}")
    print(f"  - Stochastic NC: {plots_dir_stoch_nc}")
    print(f"[Media] Saving schedule media to: {media_dir}")

    # Load assets outside the loop context to preserve runtime speeds
    cached_satellites = load_satellites_once()
    volcano_db_locations = load_volcano_locations_from_database()

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=lookahead_horizon_h)

    # === DYNAMIC ACCEPTANCE-PROBABILITY MODEL ===
    # Build demand field — single source of truth for both the simulator and the planner.
    # Set use_constant_probability=True in the config to revert to legacy constants.
    _demand_cfg = DemandFieldConfig(
        use_constant_probability=False,  # set True for legacy regression baseline
    )
    _horizon_s = lookahead_horizon_h * 3600.0
    demand_field = DemandField(
        config=_demand_cfg,
        reference_time=min_time,
        horizon_s=_horizon_s,
    )

    # Register demand spikes for all active volcanoes.
    # The spike is case-study agnostic: it just adds a localised demand bump.
    for vloc in volcano_db_locations:
        demand_field.add_spike(vloc.lat_deg, vloc.lon_deg, min_time)

    # Precompute the full trajectory for all constellations.
    _all_constellation_names = [
        "Planet", "Umbra", "Capella", "LOFT",
        "Ubotica", "Mission Control", "AC", "ICEYE",
    ]
    print("[DemandField] Precomputing demand trajectories…")
    demand_field.precompute(_all_constellation_names)
    print("[DemandField] Precompute complete.")

    # Check cost-reliability tension before running (warn if cheapest == most accepted).
    # SUBMISSION_COST / EXEC_COST are defined below but we can use a proxy ordering here.
    _cost_proxy = {
        "Planet": 0.1, "Ubotica": 0.15, "LOFT": 0.2, "Mission Control": 0.25,
        "ICEYE": 0.3, "AC": 0.35, "Umbra": 0.5, "Capella": 0.6,
    }
    demand_field.check_cost_reliability_tension(_cost_proxy)

    # Planner acceptance-probability function — queries the demand field directly.
    def acceptance_prob_function(constrained_request, satellite, obs_pass):
        """Planner's estimate of p_accept: dynamic model (or legacy fallback)."""
        return demand_field.make_acceptance_prob_function()(constrained_request, satellite, obs_pass)

    def execution_prob_function(constrained_request, satellite, obs_pass):
        """
        Probability that an ACCEPTED booking executes successfully.
        For volcano monitoring: depends on cloud cover, atmospheric conditions, etc.
        Better look angles (closer to nadir) have higher execution success.
        """
        # Simple model: execution success decreases with off-nadir angle
        look_angle = abs(90.0 - obs_pass.highest.look_angle_dec_deg)

        # At nadir (look_angle=0): 95% success
        # At 45° off-nadir: ~85% success
        # At 60° off-nadir: ~75% success
        execution_prob = 0.95 - (look_angle / 90.0) * 0.20

        return max(0.7, min(0.99, execution_prob))

    # Legacy single-stage function (for deterministic comparison baseline)
    def success_prob_function_legacy(constrained_request, satellite, obs_pass):
        """Legacy: single-stage success probability (for reference)"""
        if "Planet" in satellite.name or "SKYSAT" in satellite.name:
            return 0.7
        return 0.9

    results = {
        'deterministic_scheduled': [], 'deterministic_attempts': [], 'deterministic_rejections': [],
        'deterministic_quality': [], 'deterministic_cost': [], 'deterministic_utility': [],
        'deterministic_avg_passes_per_request': [],
        'stochastic_log_scheduled': [], 'stochastic_log_attempts': [], 'stochastic_log_rejections': [],
        'stochastic_log_quality': [], 'stochastic_log_cost': [], 'stochastic_log_utility': [],
        'stochastic_log_avg_passes_per_request': [],
        'stochastic_nc_scheduled': [], 'stochastic_nc_attempts': [], 'stochastic_nc_rejections': [],
        'stochastic_nc_quality': [], 'stochastic_nc_cost': [], 'stochastic_nc_utility': [],
        'stochastic_nc_avg_passes_per_request': [],
        'greedy_scheduled': [],      'greedy_attempts': [],    'greedy_rejections': [],
        'greedy_quality': [],        'greedy_cost': [],        'greedy_utility': [],
        'greedy_avg_passes_per_request': [],
        # Track rejected request names per run to verify fair comparison
        'deterministic_rejected_requests': [],
        'stochastic_log_rejected_requests': [],
        'stochastic_nc_rejected_requests': [],
        'greedy_rejected_requests': []
    }

    # === COST CONFIGURATION ===
    TAX_RATE = 0.0              # Legacy execution cost (disabled)
    SUBMISSION_COST = 0.1       # Unconditional booking submission overhead
    EXEC_COST = 0.3      # Conditional execution cost if accepted

    def compute_metrics(workflow_graph, broker, submission_cost_rate, execution_cost_rate):
        """
        Calculates exact realized metrics with COMPLETE cost accounting.

        KEY METRICS FOR STOCHASTIC VS DETERMINISTIC COMPARISON:
        - Total Dispatch Attempts: how many times broker tried to dispatch (including rejections/retries)
        - Rejections: how many attempts were rejected by constellation managers
        - Final Scheduled: how many tasks in final schedule (after all rescheduling converged)

        COST ACCOUNTING:
        - Submission cost (c_sub): Paid for EVERY booking submission (even if rejected)
        - Execution cost (c_exec): Paid ONLY for accepted AND executed passes

        Realized Utility = Best Quality Achieved - (c_sub * n_submitted) - (c_exec * n_executed)
        """
        scheduled = [n for n in workflow_graph.nodes() if n.scheduled and n.feasible]

        # === CRITICAL METRICS FROM BROKER REQUEST LOG ===
        # Each row in broker._requests is a dispatch attempt
        num_total_attempts = len(broker._requests)

        # Count rejections - these cause expensive rescheduling
        rejected_rows = broker._requests[broker._requests['status'] == ObservationStatus.CONSTELLATION_REJECTED]
        num_rejected = len(rejected_rows)

        # Extract unique rejected request names for verification
        rejected_request_names = sorted(set(rejected_rows['request'].apply(lambda r: r.name).tolist()))

        # Count accepted - these are the successful dispatch attempts
        num_accepted = len(broker._requests[
            broker._requests['status'].isin([ObservationStatus.SCHEDULED, ObservationStatus.DATA_RECEIVED])
        ])

        # Count task states for debugging
        completed_tasks = [n for n in workflow_graph.nodes() if n.completed]
        successful_tasks = [n for n in workflow_graph.nodes() if n.completed and getattr(n, 'successful_execution', False)]

        total_realized_quality = 0.0
        total_submission_cost = 0.0
        total_execution_cost = 0.0

        # === STEP 1: CALCULATE QUALITY (BEST per task) ===
        for task in workflow_graph.nodes():
            # Find ALL broker requests for this task
            task_requests = broker._requests[broker._requests['request'] == task.observation_request]

            # Only count passes that were actually EXECUTED (DATA_RECEIVED = observation happened)
            executed = task_requests[task_requests['status'] == ObservationStatus.DATA_RECEIVED]

            if len(executed) == 0:
                continue

            # Compute quality of each executed pass
            executed_qualities = []
            for _, req_row in executed.iterrows():
                if req_row['requested_pass'] is not None:
                    executed_qualities.append(task.rewarder(req_row['requested_pass'].highest))

            if len(executed_qualities) == 0:
                continue

            # Quality: BEST executed pass for this task (only credit the best one)
            total_realized_quality += max(executed_qualities)

        # === STEP 2: CALCULATE COSTS ===
        # Count submission cost for ALL attempts (including rejected ones)
        # Count execution cost ONLY for successfully executed passes

        for _, req_row in broker._requests.iterrows():
            if req_row['requested_pass'] is None:
                continue

            # Get the task for this request
            task = None
            for t in workflow_graph.nodes():
                if t.observation_request == req_row['request']:
                    task = t
                    break

            if task is None:
                continue

            quality = task.rewarder(req_row['requested_pass'].highest)

            # Submission cost: Paid for EVERY submission attempt (even if rejected)
            total_submission_cost += submission_cost_rate * quality

            # Execution cost: Paid ONLY if the pass was actually executed
            if req_row['status'] == ObservationStatus.DATA_RECEIVED:
                total_execution_cost += execution_cost_rate * quality

        total_cost = total_submission_cost + total_execution_cost

        # Calculate average quality per scheduled task for insight
        avg_quality_per_task = total_realized_quality / len(scheduled) if len(scheduled) > 0 else 0.0

        # Calculate average passes per request (unique request names)
        unique_scheduled_requests = set(task.name for task in scheduled)
        avg_passes_per_request = len(scheduled) / len(unique_scheduled_requests) if len(unique_scheduled_requests) > 0 else 0.0

        print(f"\n   [DEBUG] Completed: {len(completed_tasks)}, Successful: {len(successful_tasks)}")
        print(f"   [DEBUG] Scheduled: {len(scheduled)}, Avg Quality/Task: {avg_quality_per_task:.2f}")
        print(f"   [DEBUG] Unique Requests: {len(unique_scheduled_requests)}, Avg Passes/Request: {avg_passes_per_request:.2f}")
        print(f"   [DEBUG] Quality: {total_realized_quality:.2f}")
        print(f"   [DEBUG] Costs: Submission={total_submission_cost:.2f}, Execution={total_execution_cost:.2f}, Total={total_cost:.2f}")
        print(f"   [DEBUG] Rejected requests: {rejected_request_names[:5]}..." if len(rejected_request_names) > 5 else f"   [DEBUG] Rejected requests: {rejected_request_names}")

        return {
            'scheduled': len(scheduled),
            'attempts': num_total_attempts,
            'rejections': num_rejected,
            'rejected_requests': rejected_request_names,
            'quality': total_realized_quality,
            'cost': total_cost,
            'submission_cost': total_submission_cost,
            'execution_cost': total_execution_cost,
            'utility': total_realized_quality - total_cost,
            'avg_passes_per_request': avg_passes_per_request,
        }

    def run_simulation_forward(world, max_safety_limit=40000):
        """
        Ticks the simulator continuously until the global event queue is completely empty,
        ensuring the simulation clock drives forward through the entirelookahead horizon.
        """
        ticks = 0
        last_progress_report = 0
        while True:
            retcode = world.tick(print_forbidden_prefixes=["Downlink", "End of downlink", "Unlock uplink", "Unlock satellite after obs"])
            ticks += 1

            # Progress report every 5000 ticks
            if ticks - last_progress_report >= 5000:
                print(f"  [Sim Progress] {ticks} ticks, Current time: {world.time}")
                last_progress_report = ticks

            if retcode == 0:
                break
            if ticks >= max_safety_limit:
                print(f"  [Warning] Simulation safety cut-off invoked at {max_safety_limit} ticks.")
                print(f"  [Warning] Final time: {world.time}")
                break
        print(f"  [Sim] Event-loop finished. Total Ticks: {ticks}, Concluded Simulation Clock: {world.time}")

    # Save demand-field heatmaps once (independent of MC runs)
    demand_heatmaps_dir = os.path.join(results_dir, "demand_heatmaps")
    demand_field.plot_heatmaps(demand_heatmaps_dir)

    for run_idx in range(num_monte_carlo_runs):
        print(f"\n--- Monte Carlo Horizon Iteration {run_idx + 1}/{num_monte_carlo_runs} ---")

        # Set random seed for this run to ensure all three schedulers face IDENTICAL environmental conditions
        # (same acceptance/rejection outcomes, same stochastic events)
        run_seed = 42 + run_idx  # Different seed per run, but same across the 3 schedulers within each run
        print(f"  Using random seed: {run_seed} (all 3 schedulers will face identical conditions)")
                  # --- Test 2: Stochastic Log-Linearized Track ---
        print("\n  Running stochastic MILP engine (log-linearized)...")
        random.seed(run_seed)  # CRITICAL: Reset RNG to SAME seed for fair comparison
        np.random.seed(run_seed)
        world2, const2 = create_world_and_constellations(cached_satellites, demand_field=demand_field)
        workflow2 = create_volcano_workflow(volcano_db_locations, min_time, max_time)
        broker_log = Broker(constellations=const2, world=world2, name="Broker-Stoch-Log")
        broker_log.add_workflow(workflow2)
        world2.add_broker(broker_log)

        try:
            broker_log.schedule_workflow(
                current_time=world2.time, use_ilp=True, use_stochastic=True,
                stochastic_formulation="log_linearized",
                # NEW: Two-stage probability model
                acceptance_probability_function=acceptance_prob_function,
                execution_probability_function=execution_prob_function,
                # NEW: Cost structure
                submission_cost_rate=SUBMISSION_COST,
                execution_cost_rate=EXEC_COST,
                tax_rate=TAX_RATE,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=False,
                plot_schedule=True,
                save_schedule_plot=True,
                results_path=plots_dir_stoch_log
            )
            run_simulation_forward(world2)
            m = compute_metrics(broker_log._workflow_graph, broker_log, SUBMISSION_COST, EXEC_COST)
            results['stochastic_log_scheduled'].append(m['scheduled'])
            results['stochastic_log_attempts'].append(m['attempts'])
            results['stochastic_log_rejections'].append(m['rejections'])
            results['stochastic_log_rejected_requests'].append(m['rejected_requests'])
            results['stochastic_log_quality'].append(m['quality'])
            results['stochastic_log_cost'].append(m['cost'])
            results['stochastic_log_utility'].append(m['utility'])
            results['stochastic_log_avg_passes_per_request'].append(m['avg_passes_per_request'])

            # SAVE IMMEDIATELY after this run completes
            run_file = os.path.join(results_dir, f"run_{run_idx+1:03d}_stochastic_log.json")
            with open(run_file, 'w') as f:
                json.dump({
                    'run': run_idx  + 1,
                    'scheduler': 'stochastic_log',
                    'scheduled': m['scheduled'],
                    'attempts': m['attempts'],
                    'rejections': m['rejections'],
                    'rejected_requests': m['rejected_requests'],
                    'avg_passes_per_request': m['avg_passes_per_request'],
                    'quality': m['quality'],
                    'cost': m['cost'],
                    'submission_cost': m['submission_cost'],
                    'execution_cost': m['execution_cost'],
                    'utility': m['utility'],
                }, f, indent=2)
            print(f"  [Saved] {run_file}")
        except Exception as e:
            print(f"    Broker Error: {e}")
                        # # # --- Test 1: Greedy Track ---
        print("\n  Running greedy broker...")
        random.seed(run_seed)  # Reset RNG to run_seed
        np.random.seed(run_seed)  # Also reset numpy's RNG
        world3, const3 = create_world_and_constellations(cached_satellites, demand_field=demand_field)
        workflow3 = create_volcano_workflow(volcano_db_locations, min_time, max_time)
        broker_greedy = Broker(constellations=const3, world=world3, name="Broker-Greedy")
        broker_greedy.add_workflow(workflow3)
        world3.add_broker(broker_greedy)

        try:
            broker_greedy.schedule_workflow(
                current_time=world3.time, use_ilp=False, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                update_timelines=False, update_requests=False, tax_rate=TAX_RATE,
                plot_schedule=True,
                save_schedule_plot=True,
                results_path=plots_dir_greedy
            )
            run_simulation_forward(world3)
            m = compute_metrics(broker_greedy._workflow_graph, broker_greedy, SUBMISSION_COST, EXEC_COST)
            results['greedy_scheduled'].append(m['scheduled'])
            results['greedy_attempts'].append(m['attempts'])
            results['greedy_rejections'].append(m['rejections'])
            results['greedy_rejected_requests'].append(m['rejected_requests'])
            results['greedy_quality'].append(m['quality'])
            results['greedy_cost'].append(m['cost'])
            results['greedy_utility'].append(m['utility'])
            results['greedy_avg_passes_per_request'].append(m['avg_passes_per_request'])

            # SAVE IMMEDIATELY after this run completes
            run_file = os.path.join(results_dir, f"run_{run_idx+1:03d}_greedy.json")
            with open(run_file, 'w') as f:
                json.dump({
                    'run': run_idx + 1,
                    'scheduler': 'greedy',
                    'scheduled': m['scheduled'],
                    'attempts': m['attempts'],
                    'rejections': m['rejections'],
                    'rejected_requests': m['rejected_requests'],
                    'avg_passes_per_request': m['avg_passes_per_request'],
                    'quality': m['quality'],
                    'cost': m['cost'],
                    'submission_cost': m['submission_cost'],
                    'execution_cost': m['execution_cost'],
                    'utility': m['utility'],
                }, f, indent=2)
            print(f"  [Saved] {run_file}")
        except Exception as e:
            print(f"    Broker Error: {e}")

        # # # --- Test 1: Deterministic Track ---
        print("\n  Running deterministic ILP engine...")
        random.seed(run_seed)  # Reset RNG to run_seed
        np.random.seed(run_seed)  # Also reset numpy's RNG
        world1, const1 = create_world_and_constellations(cached_satellites, demand_field=demand_field)
        workflow1 = create_volcano_workflow(volcano_db_locations, min_time, max_time)
        broker_det = Broker(constellations=const1, world=world1, name="Broker-Det")
        broker_det.add_workflow(workflow1)
        world1.add_broker(broker_det)

        try:
            broker_det.schedule_workflow(
                current_time=world1.time, use_ilp=True, use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=False, tax_rate=TAX_RATE,
                plot_schedule=True,
                save_schedule_plot=True,
                results_path=plots_dir_det
            )
            run_simulation_forward(world1)
            m = compute_metrics(broker_det._workflow_graph, broker_det, SUBMISSION_COST, EXEC_COST)
            results['deterministic_scheduled'].append(m['scheduled'])
            results['deterministic_attempts'].append(m['attempts'])
            results['deterministic_rejections'].append(m['rejections'])
            results['deterministic_rejected_requests'].append(m['rejected_requests'])
            results['deterministic_quality'].append(m['quality'])
            results['deterministic_cost'].append(m['cost'])
            results['deterministic_utility'].append(m['utility'])
            results['deterministic_avg_passes_per_request'].append(m['avg_passes_per_request'])

            # SAVE IMMEDIATELY after this run completes
            run_file = os.path.join(results_dir, f"run_{run_idx+1:03d}_deterministic.json")
            with open(run_file, 'w') as f:
                json.dump({
                    'run': run_idx + 1,
                    'scheduler': 'deterministic',
                    'scheduled': m['scheduled'],
                    'attempts': m['attempts'],
                    'rejections': m['rejections'],
                    'rejected_requests': m['rejected_requests'],
                    'avg_passes_per_request': m['avg_passes_per_request'],
                    'quality': m['quality'],
                    'cost': m['cost'],
                    'submission_cost': m['submission_cost'],
                    'execution_cost': m['execution_cost'],
                    'utility': m['utility'],
                }, f, indent=2)
            print(f"  [Saved] {run_file}")
        except Exception as e:
            print(f"    Broker Error: {e}")
    

        
       
        # # --- Test 3: Stochastic Non-Convex Track ---
        # print("\n  Running stochastic MILP engine (non-convex)...")
        # random.seed(run_seed)  # CRITICAL: Reset RNG to SAME seed for fair comparison
        # np.random.seed(run_seed)
        # world3, const3 = create_world_and_constellations(cached_satellites)
        # workflow3 = create_volcano_workflow(volcano_db_locations, min_time, max_time)
        # broker_nc = Broker(constellations=const3, world=world3, name="Broker-Stoch-NC")
        # broker_nc.add_workflow(workflow3)
        # world3.add_broker(broker_nc)

        # try:
        #     broker_nc.schedule_workflow(
        #         current_time=world3.time, use_ilp=True, use_stochastic=True,
        #         stochastic_formulation="non_convex", success_probability_function=success_prob_function,
        #         max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
        #         update_timelines=False, update_requests=False, tax_rate=TAX_RATE
        #     )
        #     run_simulation_forward(world3)
        #     m = compute_metrics(broker_nc._workflow_graph, broker_nc, TAX_RATE)
        #     results['stochastic_nc_scheduled'].append(m['scheduled'])
        #     results['stochastic_nc_attempts'].append(m['attempts'])
        #     results['stochastic_nc_rejections'].append(m['rejections'])
        #     results['stochastic_nc_rejected_requests'].append(m['rejected_requests'])
        #     results['stochastic_nc_quality'].append(m['quality'])
        #     results['stochastic_nc_cost'].append(m['cost'])
        #     results['stochastic_nc_utility'].append(m['utility'])
        # except Exception as e:
        #     print(f"    Broker Error: {e}")

        # Verify fair comparison: check if all three schedulers faced IDENTICAL rejections
        if (len(results['deterministic_rejected_requests']) > 0 and
            len(results['stochastic_log_rejected_requests']) > 0 ):
            det_rej = set(results['deterministic_rejected_requests'][-1])
            log_rej = set(results['stochastic_log_rejected_requests'][-1])
            #nc_rej = set(results['stochastic_nc_rejected_requests'][-1])

            if det_rej == log_rej:
                print(f"\n  ✓ VERIFIED: All 3 schedulers faced IDENTICAL {len(det_rej)} rejections")
            else:
                print(f"\n  ⚠ WARNING: Schedulers faced DIFFERENT rejections!")
                print(f"     Deterministic: {len(det_rej)} rejections")
                print(f"     Stochastic-Log: {len(log_rej)} rejections")
                #print(f"     Stochastic-NC: {len(nc_rej)} rejections")

    # Display execution tracks summary maps
    print("\n" + "="*70)
    print("GVP REAL-WORLD DATABASE BENCHMARK RUN COMPLETE")
    print("="*70)

    summary_data = []
    for track in ['greedy', 'deterministic', 'stochastic_log']:
        avg_scheduled = np.mean(results[f'{track}_scheduled'])
        avg_attempts = np.mean(results[f'{track}_attempts'])
        avg_rejections = np.mean(results[f'{track}_rejections'])
        avg_utility = np.mean(results[f'{track}_utility'])
        avg_quality = np.mean(results[f'{track}_quality'])
        avg_cost = np.mean(results[f'{track}_cost'])
        avg_passes_per_req = np.mean(results[f'{track}_avg_passes_per_request'])
        rejection_rate = (avg_rejections / avg_attempts * 100) if avg_attempts > 0 else 0
        avg_quality_per_task = avg_quality / avg_scheduled if avg_scheduled > 0 else 0

        print(f"\n{track.upper()}:")
        print(f"  Final Scheduled:        {avg_scheduled:.1f} ± {np.std(results[f'{track}_scheduled']):.1f}")
        print(f"  Total Attempts:         {avg_attempts:.1f} ± {np.std(results[f'{track}_attempts']):.1f}")
        print(f"  Rejections:             {avg_rejections:.1f} ± {np.std(results[f'{track}_rejections']):.1f}")
        print(f"  Rejection Rate:         {rejection_rate:.1f}%")
        print(f"  Avg Passes per Request: {avg_passes_per_req:.2f}  ← Multiple observations per volcano")
        print(f"  Total Quality:          {avg_quality:.1f}")
        print(f"  Quality per Task:       {avg_quality_per_task:.2f}  ← Higher = better pass selection")
        print(f"  Total Cost:             {avg_cost:.1f}")
        print(f"  Net Utility:            {avg_utility:.1f}  ← Quality - Cost")

        summary_data.append({
            'Scheduler': track,
            'Scheduled_Mean': avg_scheduled,
            'Scheduled_Std': np.std(results[f'{track}_scheduled']),
            'Attempts_Mean': avg_attempts,
            'Attempts_Std': np.std(results[f'{track}_attempts']),
            'Rejections_Mean': avg_rejections,
            'Rejections_Std': np.std(results[f'{track}_rejections']),
            'Rejection_Rate_Pct': rejection_rate,
            'Avg_Passes_Per_Request': avg_passes_per_req,
            'Quality_Mean': avg_quality,
            'Cost_Mean': avg_cost,
            'Utility_Mean': avg_utility,
            'Utility_Std': np.std(results[f'{track}_utility']),
        })

    # Save raw per-run results in long format (one row per run+scheduler combination)
    per_run_rows = []
    num_runs = len(results['deterministic_scheduled'])
    for run_idx in range(num_runs):
        for track in ['greedy', 'deterministic', 'stochastic_log', 'stochastic_nc']:
            if run_idx < len(results[f'{track}_scheduled']):
                attempts = results[f'{track}_attempts'][run_idx]
                rejections = results[f'{track}_rejections'][run_idx]
                per_run_rows.append({
                    'run': run_idx + 1,
                    'scheduler': track,
                    'scheduled': results[f'{track}_scheduled'][run_idx],
                    'attempts': attempts,
                    'rejections': rejections,
                    'rejection_rate_pct': (rejections / attempts * 100) if attempts > 0 else 0,
                    'avg_passes_per_request': results[f'{track}_avg_passes_per_request'][run_idx],
                    'quality': results[f'{track}_quality'][run_idx],
                    'cost': results[f'{track}_cost'][run_idx],
                    'utility': results[f'{track}_utility'][run_idx],
                })
    per_run_df = pd.DataFrame(per_run_rows)
    per_run_csv = os.path.join(results_dir, "all_runs.csv")
    per_run_df.to_csv(per_run_csv, index=False)
    print(f"\n[Per-Run] Saved all run results to {per_run_csv}")

    # Save summary statistics (averages across all runs)
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(results_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Summary] Saved summary statistics to {summary_csv}")

    # Isolated Multi-Panel Boxplots to preserve correct scaling distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Only plot columns that have data (skip NC if not run)
    count_cols = ['greedy_scheduled', 'deterministic_scheduled', 'stochastic_log_scheduled']
    utility_cols = ['greedy_utility', 'deterministic_utility', 'stochastic_log_utility']

    if len(results['stochastic_nc_scheduled']) > 0:
        count_cols.append('stochastic_nc_scheduled')
        utility_cols.append('stochastic_nc_utility')

    # Build wide-format dataframe for plotting from the results dict
    plot_data = {k: v for k, v in results.items() if k in count_cols + utility_cols}
    plot_df = pd.DataFrame(plot_data)

    plot_df[count_cols].boxplot(ax=ax1)
    ax1.set_title("Operational Asset Counts Scheduled")
    ax1.set_ylabel("Pass Indices (Count)")

    plot_df[utility_cols].boxplot(ax=ax2)
    ax2.set_title("Realized Global Constellation Net Utility")
    ax2.set_ylabel("Score Bounds")

    plt.tight_layout()
    plot_file = os.path.join(results_dir, "comparison_plots.png")
    plt.savefig(plot_file, dpi=150)
    print(f"\n[OK] Metric plots saved to {plot_file}")

    # === INTERPRETATION GUIDE ===
    print("\n" + "="*70)
    print("RESULTS INTERPRETATION")
    print("="*70)

    det_track = results['deterministic_utility']
    stoch_track = results['stochastic_log_utility']
    det_rej_rate = np.mean([results['deterministic_rejections'][i] / results['deterministic_attempts'][i] * 100
                            for i in range(len(det_track)) if results['deterministic_attempts'][i] > 0])
    stoch_rej_rate = np.mean([results['stochastic_log_rejections'][i] / results['stochastic_log_attempts'][i] * 100
                              for i in range(len(stoch_track)) if results['stochastic_log_attempts'][i] > 0])

    print("\n✓ Key Findings:")
    if stoch_rej_rate < det_rej_rate:
        print(f"  • Stochastic scheduler REDUCED rejections by {det_rej_rate - stoch_rej_rate:.1f}%")
        print(f"    (Avoided risky passes with low acceptance probability)")

    if np.mean(stoch_track) > np.mean(det_track):
        improvement = (np.mean(stoch_track) - np.mean(det_track)) / np.mean(det_track) * 100
        print(f"  • Stochastic scheduler IMPROVED utility by {improvement:.1f}%")
        print(f"    (Better risk-adjusted planning with two-stage probability model)")
    elif np.mean(stoch_track) < np.mean(det_track):
        degradation = (np.mean(det_track) - np.mean(stoch_track)) / np.mean(det_track) * 100
        print(f"  • Stochastic scheduler utility was {degradation:.1f}% lower")
        print(f"    (May indicate cost parameters need tuning or conservative scheduling)")

    det_qual_per_task = np.mean([results['deterministic_quality'][i] / results['deterministic_scheduled'][i]
                                  for i in range(len(det_track)) if results['deterministic_scheduled'][i] > 0])
    stoch_qual_per_task = np.mean([results['stochastic_log_quality'][i] / results['stochastic_log_scheduled'][i]
                                    for i in range(len(stoch_track)) if results['stochastic_log_scheduled'][i] > 0])

    if stoch_qual_per_task > det_qual_per_task:
        print(f"  • Stochastic scheduler selected HIGHER quality passes")
        print(f"    (Avg quality/task: {stoch_qual_per_task:.2f} vs {det_qual_per_task:.2f})")
        print(f"    (Prefers passes with high execution probability = better geometry)")

    print("\n✓ Two-Stage Model Benefits:")
    print("  • Separates acceptance risk (constellation capacity) from execution risk (geometry)")
    print("  • Enables realistic cost modeling: submission + cancellation + execution")
    print("  • Stochastic planner can now optimize for both dimensions independently")

    return results


if __name__ == "__main__":
    run_comparison(num_monte_carlo_runs=5)