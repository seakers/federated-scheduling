"""
MSA Ship Tracking: Greedy vs Deterministic vs Stochastic Comparison
Uses the full 8-constellation fleet from TLE files and a Rotterdam ship scenario.

Usage:
    python msa_stochastic_comparison_real.py

Configuration block is at the top of the file — adjust NUM_MONTE_CARLO_RUNS,
LOOKAHEAD_HORIZON_H, cost rates, and ship kinematics before running.
"""

import datetime as dt
import math
import numpy as np
import matplotlib

from fame_workflow_stochastic import StochasticTimeline
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
import copy
import glob
import os
import random
import json
import time

import pyorbital
import pyorbital.orbital
from pyorbital.orbital import Orbital

load_dotenv()

from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus
from fame_broker import Broker
from fame_workflow import *
from fame_msa_utils import (
    ShipTracker,
    ship_propagator,
    custom_ship_phenomenon_processor,
    propagate_distribution_from_observations,
)
from fame_demand_model import DemandField, DemandFieldConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

SIMULATION_START    = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
LOOKAHEAD_HORIZON_H = 18        # Planning horizon in hours
FOLLOW_UP_INTERVAL_H = 3        # Hours between follow-up observation windows
MAX_SEARCHES_PER_INTERVAL = 10   # SAR search tasks per time window (lower = faster)
MAX_SOLVER_TIME_S   = 60       # Gurobi time limit per solve
NUM_MONTE_CARLO_RUNS = 2        # Number of independent MC runs

# Acceptance-probability model
# True  → DemandField (dynamic, spatiotemporal model — same as volcano script)
# False → hardcoded per-constellation constants (faster, no precompute step)
USE_DYNAMIC_DEMAND_FIELD = True

# Cost model (fractions of quality score)
SUBMISSION_COST = 0.10          # Paid per booking attempt (even if rejected)
EXEC_COST       = 0.30          # Paid only for executed passes
TAX_RATE        = 0.0           # Legacy single-stage cost (disabled)
MAX_NUM_INSTANCES = 5
# Ship kinematics
AVG_SPEED_KPH    = 37.0 / 2.0  # ~18.5 kph ≈ 10 knots
HEADING_VARIANCE = 0.075 / 5.0
SPEED_VARIANCE   = 0.5
R_earth_km = 6378
# Target of interest
Rotterdam = Location(3.8, 52.0, 0.0, "Rotterdam-ish")

# =============================================================================
# SATELLITE LOADING  (mirrors volcano_stochastic_comparison_real.py approach)
# =============================================================================

def load_satellites_once() -> list:
    """Load full 8-constellation LEO fleet from TLE files, filter bad orbits."""
    tle_files = glob.glob("tles/all_tles_*.txt")
    if not tle_files:
        raise FileNotFoundError("No TLE files found in tles/")
    tle_file = sorted(tle_files)[-1]
    print(f"[Init] Using TLE file: {tle_file}")

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)

    # --- Dynamic satellite names from TLE file ---
    flock_names, iceye_names = [], []
    with open(tle_file) as f:
        for line in f:
            if line.startswith("FLOCK"):
                flock_names.append(line.strip())
            elif line.startswith("ICEYE"):
                iceye_names.append(line.strip())

    # --- Swath map (display-name → swath km) ---
    swaths_at_nadir_km = {
        "SKYSAT-A": 8, "SKYSAT-B": 8,
        "SKYSAT-C1": 5.9, "SKYSAT-C2": 5.9, "SKYSAT-C3": 5.9, "SKYSAT-C4": 5.9,
        "SKYSAT-C5": 5.9, "SKYSAT-C6": 5.9, "SKYSAT-C7": 5.9, "SKYSAT-C8": 5.9,
        "SKYSAT-C9": 5.9, "SKYSAT-C10": 5.9, "SKYSAT-C11": 5.9, "SKYSAT-C12": 5.9,
        "SKYSAT-C13": 5.9,
        "PELICAN-3001": 8, "PELICAN-3009": 8, "PELICAN-300A": 8,
        "PELICAN-300B": 8, "PELICAN-5": 8, "PELICAN-6": 8,
        "TANAGER-4001": 18,
        "UMBRA-07": 8, "UMBRA-09": 8, "UMBRA-10": 8, "UMBRA-11": 8,
        "CAPELLA-11 (ACADIA-1)": 10, "CAPELLA-13 (ACADIA-3)": 10,
        "CAPELLA-14 (ACADIA-4)": 10, "CAPELLA-15 (ACADIA-5)": 10,
        "CAPELLA-16 (ACADIA-6)": 10, "CAPELLA-17 (ACADIA-7)": 10,
        "LOFT YAM-6": 19.8,
        "Ubotica CogniSat-6 HAMMER": 20, "Ubotica ACCENTURE-1 SUAC": 20,
        "Mission Control Persistence": 100,
        "AEROCUBE 18A": 80, "AEROCUBE 18B": 80,
    }
    for name in flock_names:
        swaths_at_nadir_km[name] = 16.4
    for name in iceye_names:
        swaths_at_nadir_km[name] = 100

    # TLE name → display name for satellites whose TLE names differ
    tle_to_display = {
        "SKYSAT 1": "SKYSAT-A", "SKYSAT 2": "SKYSAT-B",
        "SKYSAT C1": "SKYSAT-C1", "SKYSAT C2": "SKYSAT-C2", "SKYSAT C3": "SKYSAT-C3",
        "SKYSAT C4": "SKYSAT-C4", "SKYSAT C5": "SKYSAT-C5", "SKYSAT C6": "SKYSAT-C6",
        "SKYSAT C7": "SKYSAT-C7", "SKYSAT C8": "SKYSAT-C8", "SKYSAT C9": "SKYSAT-C9",
        "SKYSAT C10": "SKYSAT-C10", "SKYSAT C11": "SKYSAT-C11", "SKYSAT C12": "SKYSAT-C12",
        "SKYSAT C13": "SKYSAT-C13",
        "PELICAN-1 3001": "PELICAN-3001", "PELICAN-2 3009": "PELICAN-3009",
        "PELICAN-3 300A": "PELICAN-300A", "PELICAN-4 300B": "PELICAN-300B",
        "PELICAN-5 300C": "PELICAN-5",   "PELICAN-6 300D": "PELICAN-6",
        "TANAGER-1 4001": "TANAGER-4001",
        "CAPELLA-11 (ACADIA)": "CAPELLA-11 (ACADIA-1)",
        "CAPELLA-13 (ACADIA)": "CAPELLA-13 (ACADIA-3)",
        "CAPELLA-14 (ACADIA)": "CAPELLA-14 (ACADIA-4)",
        "CAPELLA-15 (ACADIA)": "CAPELLA-15 (ACADIA-5)",
        "CAPELLA-16 (ACADIA)": "CAPELLA-16 (ACADIA-6)",
        "CAPELLA-17 (ACADIA)": "CAPELLA-17 (ACADIA-7)",
        "YAM-6": "LOFT YAM-6",
        "HAMMER": "Ubotica CogniSat-6 HAMMER",
        "ACCENTURE-1": "Ubotica ACCENTURE-1 SUAC",
        "LEMUR 2 KRISH": "Mission Control Persistence",
    }
    display_to_tle = {v: k for k, v in tle_to_display.items()}

    sat_constellation_map = {
        "SKYSAT":      ("Planet",          InstrumentType.RGB),
        "PELICAN":     ("Planet",          InstrumentType.RGB),
        "TANAGER":     ("Planet",          InstrumentType.HYPERSPECTRAL),
        "FLOCK":       ("Planet",          InstrumentType.RGB),
        "UMBRA":       ("Umbra",           InstrumentType.SAR),
        "CAPELLA":     ("Capella",         InstrumentType.SAR),
        "ACADIA":      ("Capella",         InstrumentType.SAR),
        "YAM":         ("LOFT",            InstrumentType.HYPERSPECTRAL),
        "LOFT":        ("LOFT",            InstrumentType.HYPERSPECTRAL),
        "HAMMER":      ("Ubotica",         InstrumentType.HYPERSPECTRAL),
        "ACCENTURE":   ("Ubotica",         InstrumentType.HYPERSPECTRAL),
        "UBOTICA":     ("Ubotica",         InstrumentType.HYPERSPECTRAL),
        "LEMUR":       ("Mission Control", InstrumentType.RGB),
        "PERSISTENCE": ("Mission Control", InstrumentType.RGB),
        "AEROCUBE":    ("Aerospace",       InstrumentType.RGB),
        "ICEYE":       ("ICEYE",           InstrumentType.SAR),
    }

    satellites = []
    skipped = 0
    for display_name, swath_km in swaths_at_nadir_km.items():
        tle_name = display_to_tle.get(display_name, display_name)
        constellation, instrument = "Unknown", InstrumentType.RGB
        for key, (const, inst) in sat_constellation_map.items():
            if key in display_name.upper():
                constellation, instrument = const, inst
                break
        try:
            orbit = Orbital(tle_name, tle_file=tle_file)
            _ = orbit.get_lonlatalt(min_time)
            _ = orbit.get_lonlatalt(max_time)
            sat = Satellite(display_name, orbit, instruments=[instrument], has_continuous_isl_to_ground=True)
            semi_major = sat.orbit.orbit_elements.semi_major_axis * pyorbital.orbital.A
            altitude = semi_major - pyorbital.orbital.A
            fov = 2 * np.atan2(swath_km / 2, altitude)
            sat.instrument_fov_rad = {it: fov for it in sat.instruments}
            satellites.append(sat)
        except Exception as e:
            skipped += 1
            continue

    print(f"[Init] Loaded {len(satellites)} satellites (skipped {skipped} decayed/invalid).")
    return satellites


# =============================================================================
# WORLD + CONSTELLATION FACTORY
# =============================================================================

GROUND_STATIONS = [
    Location(-79.55,   8.9833,   0.028, "KSAT Panama"),
    Location(-51.73363, 64.182789, 0,   "KSAT Nuuk"),
    Location(2.53219,  -72.01243, 0,    "KSAT Troll"),
    Location(142.3689,  43.8,     0,    "KSAT Hokkaido"),
    Location(103.9915,  1.3661,   0,    "KSAT Singapore"),
    Location(-70.85021, -52.93279, 0,   "KSAT Punta Arenas"),
    Location(127.7766,  26.4055,  0,    "KSAT Okinawa"),
    Location(57.5565,  -20.1142,  0,    "KSAT Mauritius"),
    Location(22.62216,  37.84604, 0,    "KSAT Nemea"),
    Location(31.12509,  70.36779, 0,    "KSAT Vardo"),
    Location(15.39964,  78.22875, 0,    "KSAT Svalbard"),
]


def create_world_and_constellations(cached_satellites: list, demand_field: "DemandField | None" = None):
    """
    Spin up a fresh World + 8 ConstellationGroundSchedulers from deep-copied satellites.
    Mirrors the notebook's constellation structure exactly.

    If demand_field is provided it is wired as the simulator's acceptance_probability_function
    (dynamic model).  If None, each scheduler falls back to its static legacy constant.
    """
    local_sats = copy.deepcopy(cached_satellites)
    world = World(satellites=local_sats)
    world.time = SIMULATION_START

    planet_sats   = [s for s in local_sats if any(x in s.name.upper() for x in ["SKYSAT", "PELICAN", "TANAGER", "FLOCK"])]
    umbra_sats    = [s for s in local_sats if "UMBRA" in s.name.upper()]
    capella_sats  = [s for s in local_sats if "CAPELLA" in s.name.upper() or "ACADIA" in s.name.upper()]
    loft_sats     = [s for s in local_sats if "LOFT" in s.name.upper() or "YAM" in s.name.upper()]
    ubotica_sats  = [s for s in local_sats if "UBOTICA" in s.name.upper() or "HAMMER" in s.name.upper() or "ACCENTURE" in s.name.upper()]
    mc_sats       = [s for s in local_sats if "PERSISTENCE" in s.name.upper() or "LEMUR" in s.name.upper()]
    aero_sats     = [s for s in local_sats if "AEROCUBE" in s.name.upper()]
    iceye_sats    = [s for s in local_sats if "ICEYE" in s.name.upper()]

    # Simulator acceptance function: dynamic model or None (falls back to scalar probability)
    sim_acc_fn = demand_field.make_simulator_acceptance_function() if demand_field is not None else None

    def _make(sats, name, legacy_p):
        return ConstellationGroundScheduler(
            satellites=sats, ground_stations=GROUND_STATIONS,
            world=world, name=name,
            acceptance_probability=legacy_p,
            acceptance_probability_function=sim_acc_fn,
        )

    sched_planet  = _make(planet_sats,  "Planet",          0.60)
    sched_umbra   = _make(umbra_sats,   "Umbra",           0.80)
    sched_capella = _make(capella_sats, "Capella",         0.85)
    sched_loft    = _make(loft_sats,    "LOFT",            0.81)
    sched_ubotica = _make(ubotica_sats, "Ubotica",         0.80)
    sched_mc      = _make(mc_sats,      "Mission Control", 0.74)
    sched_aero    = _make(aero_sats,    "AC",              0.77)
    sched_iceye   = _make(iceye_sats,   "ICEYE",           0.74)

    all_constellations = [
        sched_planet, sched_umbra, sched_capella, sched_loft,
        sched_ubotica, sched_mc, sched_aero, sched_iceye,
    ]
    for c in all_constellations:
        world.add_constellation(c)

    return world, all_constellations


# =============================================================================
# SHIP TRAJECTORY
# =============================================================================

def generate_ship_trajectory(min_time: dt.datetime, max_time: dt.datetime, seed: int = 4) -> list:
    """
    Generate ground-truth Rotterdam ship trajectory using the same random walk
    as the notebook — sea-only, 30-min steps.
    """
    phenomenon_update_frequency = dt.timedelta(seconds=1800)
    start_heading_deg = 0.0

    trajectory = [
        Phenomenon(
            lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
            start_time=min_time, end_time=min_time + phenomenon_update_frequency,
            name=Rotterdam.name, heading_deg=start_heading_deg, speed_kph=AVG_SPEED_KPH,
        )
    ]

    random.seed(seed)
    heading_rad = start_heading_deg * math.pi / 180.0

    while trajectory[-1].end_time < max_time:
        next_is_land = True
        guesses = 0
        while next_is_land and guesses < 5000:
            guesses += 1
            heading_rad += random.normalvariate() * HEADING_VARIANCE
            new_speed = AVG_SPEED_KPH + random.normalvariate() * SPEED_VARIANCE
            dt_h = phenomenon_update_frequency.total_seconds() / 3600.0
            dy = new_speed * math.cos(heading_rad) * dt_h
            dx = new_speed * math.sin(heading_rad) * dt_h
            dlat_rad = dy / R_earth_km
            dlon_rad = dx / (R_earth_km * math.cos(trajectory[-1].lat_deg * math.pi / 180.0))
            new_lat = trajectory[-1].lat_deg + dlat_rad * 180.0 / math.pi
            new_lon = trajectory[-1].lon_deg + dlon_rad * 180.0 / math.pi
            next_is_land = is_land(new_lon, new_lat)

        if guesses == 4999:
            print(f"  [Warning] Ship trajectory generator could not escape land at {trajectory[-1].lat_deg:.2f}, {trajectory[-1].lon_deg:.2f} — reusing previous position")
            new_lat = trajectory[-1].lat_deg
            new_lon = trajectory[-1].lon_deg
            new_speed = AVG_SPEED_KPH

        prev_end = trajectory[-1].end_time
        trajectory.append(Phenomenon(
            lon_deg=new_lon, lat_deg=new_lat, alt_km=Rotterdam.alt_km,
            start_time=prev_end, end_time=prev_end + phenomenon_update_frequency,
            heading_deg=heading_rad * 180.0 / math.pi, speed_kph=new_speed,
            name=f"{Rotterdam.name} {prev_end.strftime('%H%M')}",
        ))

    print(f"  [Trajectory] Generated {len(trajectory)} ship positions over {LOOKAHEAD_HORIZON_H}h horizon")
    return trajectory


# =============================================================================
# MSA WORKFLOW BUILDER
# =============================================================================

def rewarder_observation(opportunity):
    """Reward for imaging tasks: static base + look-angle quality + range quality."""
    static_reward = 50.0
    look_angle_reward = abs(90.0 - opportunity.look_angle_dec_deg) / 90.0
    range_reward = 1.0 / (opportunity.range_km / 1000.0)
    return static_reward + look_angle_reward + range_reward


def rewarder_search(opportunity):
    """Reward for SAR search tasks: no static base, just geometry."""
    look_angle_reward = abs(90.0 - opportunity.look_angle_dec_deg) / 90.0
    range_reward = 1.0 / (opportunity.range_km / 1000.0)
    return look_angle_reward + range_reward


def success_declarer(data_product):
    return len(data_product) > 0


def build_msa_workflow(
    min_time: dt.datetime,
    max_time: dt.datetime,
    initial_toi_pose,
) -> "Workflow":
    """
    Builds the MSA conditional tracking workflow:

    - 1 mandatory initial RGB observation at Rotterdam (hours 0–3)
    - For each follow-up window [h, h+dt_h]:
        * 1 imaging task (RGB, GREATER_OR_EQUAL timeline constraint — ship location known)
        * MAX_SEARCHES_PER_INTERVAL search tasks (SAR, LESSER_OR_EQUAL — ship location unknown)
    - Uses StochasticTimeline to cleanly reflect real-time observations without plan-time impact stacking.
    """
    # 1. Instantiate StochasticTimeline
    ship_timeline = StochasticTimeline(
        name="Ship loc. known?",
        initial_time=min_time,
        initial_value=1.0,
        half_life_s=10800.0,  # 3 hours
    )

    init_req = ObservationRequest(
        lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
        min_time=min_time, max_time=min_time + dt.timedelta(hours=3),
        request_name="Initial RGB observation Rotterdam",
        min_elevation_deg=45.0, instrument=InstrumentType.RGB,
    )
    
    # Clean: timeline_impacts=[] because StochasticTimeline updates dynamically
    root_task = ConstrainedObservationRequest(
        name="Initial observation",
        observation_request=init_req,
        is_mandatory=True,
        timeline_constraints=[],
        timeline_impacts=[],
        rewarder=rewarder_observation,
        success_declarer=success_declarer,
    )

    workflow_reqs = [root_task]

    # FIX: Include TIMELINE: True in policy_schedule so conditional search tasks aren't dropped at t=0
    policy_schedule = {
        ConstraintClass.TEMPORAL: True,
        ConstraintClass.SUCCESS: True,
        ConstraintClass.GEOMETRY: True,
    }
    policy_dispatch = {
        ConstraintClass.TEMPORAL: False,
        ConstraintClass.SUCCESS: False,
        ConstraintClass.GEOMETRY: False,
    }

    for h_ix in range(FOLLOW_UP_INTERVAL_H, LOOKAHEAD_HORIZON_H, FOLLOW_UP_INTERVAL_H):
        window_ix = h_ix // FOLLOW_UP_INTERVAL_H

        # FIX: Constrain relative ONLY to window offsets from root_task,
        # avoiding cascading WAIT_FOR_COMPLETION locks on skipped search tasks.
        def _base_constraints(h):
            return [
                Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET,
                           root_task, {'offset': dt.timedelta(hours=h)}),
                Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_BEFORE_OFFSET,
                           root_task, {'offset': dt.timedelta(hours=h + FOLLOW_UP_INTERVAL_H)}),
            ]

        # Imaging task (RGB) — dispatched when ship location IS known (timeline >= 0)
        img_obs_req = ObservationRequest(
            lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
            min_time=min_time + dt.timedelta(hours=h_ix),
            max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H),
            request_name=f"Follow-up imaging {window_ix} Rotterdam",
            min_elevation_deg=45.0, instrument=InstrumentType.RGB,
        )
        img_task = ConstrainedObservationRequest(
            name=f"Follow-up obs. {window_ix}",
            observation_request=img_obs_req,
            is_mandatory=False,
            task_constraints=_base_constraints(h_ix),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            success_declarer=success_declarer,
            timeline_constraints=[TaskTimelineConstraint(
                timeline=ship_timeline, time=TaskImpactTime.PRE,
                type=TimelineConstraintType.GREATER_OR_EQUAL, value=0.0,
            )],
            timeline_impacts=[],
            rewarder=rewarder_observation,
            max_num_instances=MAX_NUM_INSTANCES,
            request_group=f"Follow-up obs. {window_ix}",
        )

        # Search tasks (SAR) — dispatched when ship location is NOT known (timeline <= 0)
        for search_ix in range(MAX_SEARCHES_PER_INTERVAL):
            srch_obs_req = ObservationRequest(
                lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
                min_time=min_time + dt.timedelta(hours=h_ix),
                max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H),
                request_name=f"Follow-up search {window_ix}.{search_ix} Rotterdam",
                min_elevation_deg=45.0, instrument=InstrumentType.SAR,
            )
            srch_task = ConstrainedObservationRequest(
                name=f"Follow-up search {window_ix}.{search_ix}",
                observation_request=srch_obs_req,
                is_mandatory=False,
                task_constraints=_base_constraints(h_ix),
                schedule_policy_if_constraint_unsatisfied=policy_schedule,
                dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
                success_declarer=success_declarer,
                timeline_constraints=[TaskTimelineConstraint(
                    timeline=ship_timeline, time=TaskImpactTime.PRE,
                    type=TimelineConstraintType.LESSER_OR_EQUAL, value=0.0,
                )],
                timeline_impacts=[],
                rewarder=rewarder_search,
                max_num_instances=MAX_NUM_INSTANCES,
                request_group=f"Follow-up search {window_ix}",
            )
            workflow_reqs.append(srch_task)

        workflow_reqs.append(img_task)

    # --- Timeline updater: resets ship-known timeline based on recent successful observations ---
    def timeline_updater_msa(current_time, requests, timelines, max_time_without_obs=dt.timedelta(seconds=10800)):
        for tl in timelines:
            if isinstance(tl, StochasticTimeline):
                active = tl.refresh_if_observed(current_time, requests)
                print(f"    [Timeline] {tl.name} active = {active} at {current_time}")
            else:
                ship_position_known = any(
                    getattr(r, 'completed', False) and getattr(r, 'successful_execution', False)
                    and hasattr(r, 'observation_opportunity') and r.observation_opportunity is not None
                    and (current_time - r.observation_opportunity.time) < max_time_without_obs
                    for r in requests
                )
                new_value = 1.0 if ship_position_known else 0.0
                _, current_rate = tl._get_value_and_rate_at(current_time, print_debug=False)
                tl.reset_timeline(current_time, new_value, current_rate)

        return timelines

    # --- Request updater: re-targets SAR search tasks via particle-filter propagation ---
    def request_updater_msa(current_time, requests, timelines):
        initial_ship_location = Pose(
            time=SIMULATION_START,
            lon_deg=initial_toi_pose.lon_deg, lat_deg=initial_toi_pose.lat_deg,
            alt_km=initial_toi_pose.alt_km,
            heading_deg=initial_toi_pose.heading_deg, speed_kph=initial_toi_pose.speed_kph,
        )

        unfilled = [
            r for r in requests
            if not r.completed and not r.dispatched and r.feasible
            and r.observation_request.max_time > current_time
        ]
        if not unfilled:
            return requests

        unfilled_by_time = {}
        for r in unfilled:
            t = r.observation_request.min_time + (
                r.observation_request.max_time - r.observation_request.min_time
            ) / 2
            unfilled_by_time.setdefault(t, []).append(r)

        pose_prop = lambda prev, delta: ship_propagator(
            previous_pose=prev, deltat=delta,
            heading_variance=HEADING_VARIANCE, speed_variance=SPEED_VARIANCE,
            max_speed_if_unspecified_kph=AVG_SPEED_KPH,
        )

        try:
            hexes_by_time = propagate_distribution_from_observations(
                query_times=list(unfilled_by_time.keys()),
                initial_pose=initial_ship_location,
                observation_requests=requests,
                pose_propagator=pose_prop,
                target_name="Rotterdam",
                h3_resolution=5,
                samples_propagation=100,
                propagate_negative_samples=False,
                verbose=False,
            )
        except Exception as e:
            print(f"    [RequestUpdater] Propagation failed: {e} — keeping original coordinates")
            return requests

        for t, req_list in unfilled_by_time.items():
            hexes = hexes_by_time.get(t)
            if hexes is None:
                continue
            for ix, req in enumerate(req_list):
                if ix >= len(hexes):
                    req.scheduled = True
                    req.dispatched = True
                    req.completed = True
                    req.feasible = False
                    req.successful_execution = False
                else:
                    centroid = hexes.iloc[ix].geometry.centroid
                    req.observation_request = ObservationRequest(
                        lat_deg=centroid.y, lon_deg=centroid.x,
                        min_time=req.observation_request.min_time,
                        max_time=req.observation_request.max_time,
                        alt_km=req.observation_request.alt_km,
                        instrument=req.observation_request.instrument,
                        request_name=req.observation_request.name,
                        min_elevation_deg=req.observation_request.min_elevation_deg,
                    )
        return requests

    n_windows = (LOOKAHEAD_HORIZON_H - FOLLOW_UP_INTERVAL_H) // FOLLOW_UP_INTERVAL_H
    n_tasks = 1 + n_windows * (1 + MAX_SEARCHES_PER_INTERVAL)
    print(f"  [Workflow] Built MSA workflow: {n_tasks} tasks "
          f"({n_windows} follow-up windows × (1 imaging + {MAX_SEARCHES_PER_INTERVAL} search))")

    return Workflow(
        constrained_observation_requests=workflow_reqs,
        timelines=[ship_timeline],
        timeline_updater=timeline_updater_msa,
        request_updater=request_updater_msa,
    )


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(workflow_graph, broker, submission_cost_rate, execution_cost_rate):
    """
    Full cost-accounting metrics from broker request log.
    Mirrors volcano_stochastic_comparison_real.py compute_metrics exactly.
    """
    scheduled = [n for n in workflow_graph.nodes() if n.scheduled and n.feasible]

    num_total_attempts = len(broker._requests)
    rejected_rows = broker._requests[
        broker._requests['status'] == ObservationStatus.CONSTELLATION_REJECTED
    ]
    num_rejected = len(rejected_rows)
    rejected_request_names = sorted(set(rejected_rows['request'].apply(lambda r: r.name).tolist()))

    num_accepted = len(broker._requests[
        broker._requests['status'].isin([ObservationStatus.SCHEDULED, ObservationStatus.DATA_RECEIVED])
    ])
    executed_rows = broker._requests[broker._requests['status'] == ObservationStatus.DATA_RECEIVED]
    num_executed = len(executed_rows)

    total_realized_quality = 0.0
    total_submission_cost = 0.0
    total_execution_cost = 0.0

    for task in workflow_graph.nodes():
        task_requests = broker._requests[broker._requests['request'] == task.observation_request]
        executed = task_requests[task_requests['status'] == ObservationStatus.DATA_RECEIVED]
        if len(executed) == 0:
            continue
        qualities = [
            task.rewarder(row['requested_pass'].highest)
            for _, row in executed.iterrows()
            if row['requested_pass'] is not None
        ]
        if qualities:
            total_realized_quality += max(qualities)

    for _, req_row in broker._requests.iterrows():
        if req_row['requested_pass'] is None:
            continue
        task = next(
            (t for t in workflow_graph.nodes() if t.observation_request == req_row['request']),
            None,
        )
        if task is None:
            continue
        quality = task.rewarder(req_row['requested_pass'].highest)
        total_submission_cost += submission_cost_rate * quality
        if req_row['status'] == ObservationStatus.DATA_RECEIVED:
            total_execution_cost += execution_cost_rate * quality

    total_cost = total_submission_cost + total_execution_cost
    unique_scheduled = set(t.name for t in scheduled)
    avg_passes = len(scheduled) / len(unique_scheduled) if unique_scheduled else 0.0
    rejection_rate = (num_rejected / num_total_attempts * 100) if num_total_attempts > 0 else 0.0

    print(f"   [Metrics] Scheduled={len(scheduled)}, Attempts={num_total_attempts}, "
          f"Rejected={num_rejected} ({rejection_rate:.1f}%), "
          f"Quality={total_realized_quality:.1f}, Cost={total_cost:.1f}, "
          f"Utility={total_realized_quality - total_cost:.1f}")

    return {
        'scheduled': len(scheduled),
        'attempts': num_total_attempts,
        'accepted': num_accepted,
        'executed': num_executed,
        'rejections': num_rejected,
        'rejected_requests': rejected_request_names,
        'rejection_rate': rejection_rate,
        'avg_passes_per_request': avg_passes,
        'quality': total_realized_quality,
        'cost': total_cost,
        'submission_cost': total_submission_cost,
        'execution_cost': total_execution_cost,
        'utility': total_realized_quality - total_cost,
    }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_simulation_forward(world, max_ticks=40000):
    """Tick simulation to completion."""
    ticks = 0
    last_report = 0
    while True:
        retcode = world.tick(print_forbidden_prefixes=[
            "Downlink", "End of downlink", "Unlock uplink",
            "Unlock satellite after obs",
        ])
        ticks += 1
        if ticks - last_report >= 5000:
            print(f"    [Sim] {ticks} ticks, time: {world.time}")
            last_report = ticks
        if retcode == 0:
            break
        if ticks >= max_ticks:
            print(f"    [Sim] Safety cut-off at {max_ticks} ticks.")
            break
    print(f"    [Sim] Done. {ticks} ticks, final time: {world.time}")


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(num_monte_carlo_runs: int = NUM_MONTE_CARLO_RUNS):
    """
    Monte Carlo comparison of greedy, deterministic ILP, and stochastic log-linearized
    schedulers on the MSA Rotterdam ship-tracking scenario.
    """
    print("\n" + "=" * 70)
    print("MSA SHIP TRACKING — SCHEDULER COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Horizon:             {LOOKAHEAD_HORIZON_H}h")
    print(f"  Follow-up windows:   every {FOLLOW_UP_INTERVAL_H}h")
    print(f"  Search tasks/window: {MAX_SEARCHES_PER_INTERVAL}")
    print(f"  Monte Carlo runs:    {num_monte_carlo_runs}")
    print(f"  Submission cost:     {SUBMISSION_COST}")
    print(f"  Execution cost:      {EXEC_COST}")
    print(f"  Solver time limit:   {MAX_SOLVER_TIME_S}s")
    print(f"  Acceptance model:    {'DYNAMIC DemandField' if USE_DYNAMIC_DEMAND_FIELD else 'STATIC per-constellation constants'}")

    # Timestamped results directory
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join("results", f"msa_{timestamp}")
    plots_dir_det   = os.path.join(results_dir, "plots", "deterministic")
    plots_dir_stoch = os.path.join(results_dir, "plots", "stochastic_log")
    plots_dir_greedy = os.path.join(results_dir, "plots", "greedy")
    for d in [results_dir, plots_dir_det, plots_dir_stoch, plots_dir_greedy]:
        os.makedirs(d, exist_ok=True)
    print(f"\n[Results] Saving to: {results_dir}")

    # Load satellites once
    print("\n[Init] Loading satellite fleet...")
    cached_satellites = load_satellites_once()

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)

    initial_toi_pose = Pose(
        lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg,
        time=min_time, alt_km=Rotterdam.alt_km,
        heading_deg=0.0, speed_kph=AVG_SPEED_KPH, name="Rotterdam initial",
    )

    # ── ACCEPTANCE-PROBABILITY MODEL ─────────────────────────────────────────
    # Two modes controlled by USE_DYNAMIC_DEMAND_FIELD (set at top of file):
    #
    #   True  → DemandField: spatiotemporal dynamic model, precomputed once.
    #            Both the SIMULATOR (world/constellations) and the STOCHASTIC
    #            PLANNER query the same DemandField, so planner priors match
    #            the simulated environment. Same approach as the volcano script.
    #
    #   False → Static constants: fixed per-constellation scalars baked in here.
    #            Faster (no precompute), but the planner's priors are decoupled
    #            from the simulation's acceptance draw.

    if USE_DYNAMIC_DEMAND_FIELD:
        print("\n[DemandField] Building dynamic acceptance-probability model...")
        _demand_cfg = DemandFieldConfig(use_constant_probability=False)
        demand_field = DemandField(
            config=_demand_cfg,
            reference_time=min_time,
            horizon_s=LOOKAHEAD_HORIZON_H * 3600.0,
        )
        # Register a demand spike at Rotterdam — the target of interest
        demand_field.add_spike(Rotterdam.lat_deg, Rotterdam.lon_deg, min_time)

        _all_constellation_names = [
            "Planet", "Umbra", "Capella", "LOFT",
            "Ubotica", "Mission Control", "AC", "ICEYE",
        ]
        print("[DemandField] Precomputing demand trajectories...")
        demand_field.precompute(_all_constellation_names)
        print("[DemandField] Precompute complete.")

        # Planner acceptance function: queries the demand field directly
        def acceptance_prob_fn(constrained_request, satellite, obs_pass):
            return demand_field.make_acceptance_prob_function()(constrained_request, satellite, obs_pass)

    else:
        print("\n[AcceptanceModel] Using STATIC per-constellation constants (USE_DYNAMIC_DEMAND_FIELD=False).")
        demand_field = None

        def acceptance_prob_fn(constrained_request, satellite, obs_pass):
            name = satellite.name.upper()
            if any(x in name for x in ["SKYSAT", "PELICAN", "TANAGER", "FLOCK"]):
                return 0.70
            elif "UMBRA" in name:
                return 0.85
            elif "CAPELLA" in name or "ACADIA" in name:
                return 0.90
            elif "LOFT" in name or "YAM" in name:
                return 0.92
            elif "UBOTICA" in name or "HAMMER" in name or "ACCENTURE" in name:
                return 0.93
            elif "PERSISTENCE" in name or "LEMUR" in name:
                return 0.94
            elif "AEROCUBE" in name:
                return 0.95
            elif "ICEYE" in name:
                return 0.96
            return 0.85

    def execution_prob_fn(constrained_request, satellite, obs_pass):
        look_angle = abs(90.0 - obs_pass.highest.look_angle_dec_deg)
        return max(0.70, min(0.99, 0.95 - (look_angle / 90.0) * 0.20))

    # Accumulators
    all_run_rows = []

    track_results = {
        track: {'scheduled': [], 'attempts': [], 'rejections': [], 'rejection_rate': [],
                'avg_passes_per_request': [], 'quality': [], 'cost': [], 'utility': []}
        for track in ['greedy', 'deterministic', 'stochastic_log']
    }

    for run_idx in range(num_monte_carlo_runs):
        print(f"\n{'=' * 70}")
        print(f"MONTE CARLO RUN {run_idx + 1}/{num_monte_carlo_runs}")
        print(f"{'=' * 70}")

        run_seed = 42 + run_idx
        print(f"  Random seed: {run_seed}")

        print("\n  Generating ship trajectory...")
        phenomena = generate_ship_trajectory(min_time, max_time, seed=run_seed)

        print("\n  Building MSA workflow...")
        # One workflow template per run (shared structure, deep-copied per scheduler)
        template_workflow = build_msa_workflow(min_time, max_time, initial_toi_pose)

        for scheduler_name, use_ilp, use_stochastic, plots_dir in [
            ("stochastic_log", True,  True,  plots_dir_stoch),
            ("deterministic",  True,  False, plots_dir_det),
            ("greedy",         False, False, plots_dir_greedy),
        ]:
            print(f"\n  --- {scheduler_name.upper()} ---")
            random.seed(run_seed)
            np.random.seed(run_seed)

            world, constellations = create_world_and_constellations(cached_satellites, demand_field=demand_field)
            # Inject phenomena into this world's copy
            for p in copy.deepcopy(phenomena):
                world.phenomena.append(p)

            broker = Broker(constellations=constellations, world=world, name=f"Broker-{scheduler_name}")
            broker.add_workflow(copy.deepcopy(template_workflow))
            world.add_broker(broker)

            t_start = time.time()
            try:
                schedule_kwargs = dict(
                    current_time=world.time,
                    use_ilp=use_ilp,
                    use_stochastic=use_stochastic,
                    max_solver_time_s=MAX_SOLVER_TIME_S,
                    solver_engine="GUROBI",
                    update_timelines=False,
                    update_requests=False,
                    tax_rate=TAX_RATE,
                    max_reschedule_depth=1,
                    plot_schedule=True,
                    save_schedule_plot=True,
                    results_path=plots_dir,
                )
                if use_stochastic:
                    schedule_kwargs.update(
                        stochastic_formulation="log_linearized",
                        acceptance_probability_function=acceptance_prob_fn,
                        execution_probability_function=execution_prob_fn,
                        submission_cost_rate=SUBMISSION_COST,
                        execution_cost_rate=EXEC_COST,
                    )

                broker.schedule_workflow_redundant(**schedule_kwargs)
                run_simulation_forward(world)
                m = compute_metrics(broker._workflow_graph, broker, SUBMISSION_COST, EXEC_COST)
                elapsed = time.time() - t_start

                # Accumulate per-track
                for k in track_results[scheduler_name]:
                    track_results[scheduler_name][k].append(m[k])

                # Save individual run JSON
                run_file = os.path.join(results_dir, f"run_{run_idx+1:03d}_{scheduler_name}.json")
                with open(run_file, 'w') as f:
                    json.dump({
                        'run': run_idx + 1, 'scheduler': scheduler_name,
                        'acceptance_model': 'dynamic_demand_field' if USE_DYNAMIC_DEMAND_FIELD else 'static_constants',
                        'scheduled': m['scheduled'], 'attempts': m['attempts'],
                        'accepted': m['accepted'], 'executed': m['executed'],
                        'rejections': m['rejections'],
                        'rejected_requests': m['rejected_requests'],
                        'rejection_rate': m['rejection_rate'],
                        'avg_passes_per_request': m['avg_passes_per_request'],
                        'quality': m['quality'], 'cost': m['cost'],
                        'submission_cost': m['submission_cost'],
                        'execution_cost': m['execution_cost'],
                        'utility': m['utility'], 'elapsed_s': elapsed,
                    }, f, indent=2)
                print(f"  [Saved] {run_file}")

                all_run_rows.append({
                    'run': run_idx + 1, 'scheduler': scheduler_name,
                    'acceptance_model': 'dynamic_demand_field' if USE_DYNAMIC_DEMAND_FIELD else 'static_constants',
                    'scheduled': m['scheduled'], 'attempts': m['attempts'],
                    'rejections': m['rejections'],
                    'rejection_rate_pct': m['rejection_rate'],
                    'avg_passes_per_request': m['avg_passes_per_request'],
                    'quality': m['quality'], 'cost': m['cost'],
                    'utility': m['utility'], 'elapsed_s': elapsed,
                })

            except Exception as e:
                print(f"  [Error] {scheduler_name} FAILED on run {run_idx + 1}: {e}")
                import traceback; traceback.print_exc()

        # Verify fairness: did deterministic and stochastic face the same rejections?
        det_rej  = set(track_results['deterministic'].get('rejected_requests', [None])[-1] or []) if track_results['deterministic']['rejections'] else set()
        stoch_rej = set(track_results['stochastic_log'].get('rejected_requests', [None])[-1] or []) if track_results['stochastic_log']['rejections'] else set()
        if det_rej and stoch_rej:
            if det_rej == stoch_rej:
                print(f"\n  [OK] Deterministic and stochastic faced identical {len(det_rej)} rejections")
            else:
                print(f"\n  [Warning] Different rejection sets: Det={len(det_rej)}, Stoch={len(stoch_rej)}")

    # ── SAVE CSVs ─────────────────────────────────────────────────────────────
    per_run_df = pd.DataFrame(all_run_rows)
    all_runs_csv = os.path.join(results_dir, "all_runs.csv")
    per_run_df.to_csv(all_runs_csv, index=False)
    print(f"\n[CSV] Per-run results saved to {all_runs_csv}")

    summary_rows = []
    for track in ['greedy', 'deterministic', 'stochastic_log']:
        r = track_results[track]
        if not r['scheduled']:
            continue
        avg_sched  = np.mean(r['scheduled'])
        avg_att    = np.mean(r['attempts'])
        avg_rej    = np.mean(r['rejections'])
        avg_util   = np.mean(r['utility'])
        avg_qual   = np.mean(r['quality'])
        avg_cost   = np.mean(r['cost'])
        avg_passes = np.mean(r['avg_passes_per_request'])
        rej_rate   = (avg_rej / avg_att * 100) if avg_att > 0 else 0.0
        summary_rows.append({
            'Scheduler': track,
            'Scheduled_Mean': avg_sched, 'Scheduled_Std': np.std(r['scheduled']),
            'Attempts_Mean': avg_att,    'Attempts_Std': np.std(r['attempts']),
            'Rejections_Mean': avg_rej,  'Rejections_Std': np.std(r['rejections']),
            'Rejection_Rate_Pct': rej_rate,
            'Avg_Passes_Per_Request': avg_passes,
            'Quality_Mean': avg_qual,    'Cost_Mean': avg_cost,
            'Utility_Mean': avg_util,    'Utility_Std': np.std(r['utility']),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[CSV] Summary statistics saved to {summary_csv}")

    # ── SUMMARY PRINT ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MSA COMPARISON RESULTS")
    print("=" * 70)
    for row in summary_rows:
        track = row['Scheduler']
        print(f"\n{track.upper()}:")
        print(f"  Scheduled:          {row['Scheduled_Mean']:.1f} ± {row['Scheduled_Std']:.1f}")
        print(f"  Attempts:           {row['Attempts_Mean']:.1f} ± {row['Attempts_Std']:.1f}")
        print(f"  Rejections:         {row['Rejections_Mean']:.1f} ± {row['Rejections_Std']:.1f}")
        print(f"  Rejection Rate:     {row['Rejection_Rate_Pct']:.1f}%")
        print(f"  Avg Passes/Request: {row['Avg_Passes_Per_Request']:.2f}")
        print(f"  Quality:            {row['Quality_Mean']:.1f}")
        print(f"  Cost:               {row['Cost_Mean']:.1f}")
        print(f"  Utility:            {row['Utility_Mean']:.1f} ± {row['Utility_Std']:.1f}")

    # Stochastic vs greedy interpretation
    if len(summary_rows) >= 2:
        stoch_row  = next((r for r in summary_rows if r['Scheduler'] == 'stochastic_log'), None)
        det_row    = next((r for r in summary_rows if r['Scheduler'] == 'deterministic'), None)
        greedy_row = next((r for r in summary_rows if r['Scheduler'] == 'greedy'), None)

        print(f"\n{'=' * 70}")
        print("KEY FINDINGS")
        print("=" * 70)

        if stoch_row and det_row and det_row['Utility_Mean'] != 0:
            delta_util = stoch_row['Utility_Mean'] - det_row['Utility_Mean']
            pct = delta_util / abs(det_row['Utility_Mean']) * 100
            sign = "IMPROVED" if delta_util >= 0 else "lower"
            print(f"\n  Stochastic vs Deterministic utility: {delta_util:+.1f} ({pct:+.1f}%)")
            delta_rej = det_row['Rejection_Rate_Pct'] - stoch_row['Rejection_Rate_Pct']
            if delta_rej > 0:
                print(f"  Stochastic reduced rejection rate by {delta_rej:.1f}pp "
                      f"({det_row['Rejection_Rate_Pct']:.1f}% → {stoch_row['Rejection_Rate_Pct']:.1f}%)")

        if stoch_row and greedy_row and greedy_row['Utility_Mean'] != 0:
            delta_g = stoch_row['Utility_Mean'] - greedy_row['Utility_Mean']
            pct_g = delta_g / abs(greedy_row['Utility_Mean']) * 100
            print(f"  Stochastic vs Greedy utility:        {delta_g:+.1f} ({pct_g:+.1f}%)")

    # ── BOXPLOT ───────────────────────────────────────────────────────────────
    if len(per_run_df) > 0:
        schedulers = per_run_df['scheduler'].unique().tolist()
        count_data   = [per_run_df[per_run_df['scheduler'] == s]['scheduled'].tolist() for s in schedulers]
        utility_data = [per_run_df[per_run_df['scheduler'] == s]['utility'].tolist()   for s in schedulers]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.boxplot(count_data, labels=schedulers)
        ax1.set_title("Tasks Scheduled")
        ax1.set_ylabel("Count")
        ax1.grid(axis='y', alpha=0.4)

        ax2.boxplot(utility_data, labels=schedulers)
        ax2.set_title("Net Utility  (Quality − Cost)")
        ax2.set_ylabel("Utility")
        ax2.grid(axis='y', alpha=0.4)

        plt.suptitle(f"MSA Ship Tracking — {num_monte_carlo_runs} MC Runs", fontsize=12)
        plt.tight_layout()
        plot_file = os.path.join(results_dir, "comparison_plots.png")
        plt.savefig(plot_file, dpi=150)
        print(f"\n[Plot] Saved {plot_file}")

    print(f"\n[Done] All results in {results_dir}")
    return track_results, results_dir


if __name__ == "__main__":
    run_comparison(num_monte_carlo_runs=NUM_MONTE_CARLO_RUNS)
