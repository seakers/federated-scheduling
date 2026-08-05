"""
MSA Ship Tracking: Scheduler Comparison (Stochastic / Deterministic / Greedy / Random)
Uses the full 8-constellation fleet from TLE files and a Rotterdam ship scenario.

RUN MODES
---------
1) Single scheduler, one seed (campaign / HPC mode):
     python msa_stochastic_comparison_real.py --scheduler stochastic --seed 42 \
            --start 2026-08-01T00:00:00 --results-dir results/msa_campaign

2) Aggregate an existing results dir (reads run_*.json, writes CSVs + plots):
     python msa_stochastic_comparison_real.py --aggregate --results-dir results/msa_campaign

3) Legacy all-in-one loop:
     python msa_stochastic_comparison_real.py

IMPORTANT: --start pins SIMULATION_START. Every process in one campaign MUST share
the same value so orbital geometry and the demand field are identical across seeds.

Configuration block is at the top of the file — adjust NUM_MONTE_CARLO_RUNS,
LOOKAHEAD_HORIZON_H, cost rates, and which SCHEDULERS to run.
"""

import argparse
import datetime as dt
import gc
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from dotenv import load_dotenv

load_dotenv()

from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus
from fame_broker import Broker
from fame_workflow import *
from fame_workflow_stochastic import Lit, And, Or, Not, ExclusiveOr, LogicNode
from fame_msa_utils import (
    ShipTracker,
    ship_propagator,
    custom_ship_phenomenon_processor,
    propagate_distribution_from_observations,
)
from fame_demand_model import DemandField, DemandFieldConfig
from fame_metrics import compute_metrics_v3, paired_summary, plot_cost_frontier
from benchmarking_utils import (
    GROUND_STATIONS,
    load_satellites_once as _load_satellites_once,
    create_world_and_constellations as _create_world_and_constellations,
    run_simulation_forward,
    rss_gb,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pinned to a FIXED epoch — wall-clock now() silently re-samples orbital geometry
# and the demand field on every launch, making results non-reproducible.
# Chosen a few days after the TLE epoch (tles/all_tles_20260727.txt → 2026-07-27)
# so SGP4 propagation stays accurate over the 36 h horizon.
SIMULATION_START     = dt.datetime(2026, 8, 1, 0, 0, 0)
LOOKAHEAD_HORIZON_H  = 18         # Planning horizon in hours
FOLLOW_UP_INTERVAL_H = 3          # Hours between follow-up observation windows
MAX_SEARCHES_PER_INTERVAL = 5    # SAR search tasks per window (lower = faster)
MAX_SOLVER_TIME_S    = 70         # Gurobi time limit per solve
NUM_MONTE_CARLO_RUNS = 2          # Seeds for the all-in-one loop

# Which schedulers to run (subset of these four)
SCHEDULERS = ['stochastic', 'deterministic', 'greedy', 'random']
#SCHEDULERS = ['stochastic']

# Redundancy cap: stochastic planner books up to this many passes per task.
# Must be > 1 for the MILP to hedge. Deterministic/greedy/random use MAX_NUM_INSTANCES=1
# (each task is booked once — full quality per booking, no submission cost discount).
MAX_NUM_INSTANCES = 5

# Set True to cancel inferior pending passes once one for the same task succeeds,
# avoiding EXEC_COST for observations we no longer need. No cancellation_cost_rate:
# submission cost is already charged at booking time.
ENABLE_CANCELLATIONS = True

# Acceptance-probability model
# True  → DemandField (dynamic spatiotemporal model — same as volcano script)
# False → hardcoded per-constellation constants (faster, no precompute step)
USE_DYNAMIC_DEMAND_FIELD = True

# Acceptance probability range fed to DemandField (only used when USE_DYNAMIC_DEMAND_FIELD=True).
# DemandField maps demand level → p_accept in [P_ACC_MIN, P_ACC_MAX].
# Set both to 1.0 for deterministic acceptance (isolates scheduling quality from market noise).
P_ACC_MIN = 0.85   # Minimum acceptance probability (high-demand / congested slot)
P_ACC_MAX = 0.95   # Maximum acceptance probability (low-demand / quiet slot)

# Execution probability range: even an accepted pass may fail (cloud cover, sensor issue).
# Maps look-angle → p_exec in [P_EXEC_MIN, P_EXEC_MAX].
# At nadir (best geometry) → P_EXEC_MAX; at worst geometry → P_EXEC_MIN.
# Set both to 1.0 for deterministic execution (isolates scheduling from sensor noise).
P_EXEC_MIN = 0.50  # Minimum execution probability (worst geometry)
P_EXEC_MAX = 0.70  # Maximum execution probability (best geometry)

# Cost model (fractions of quality score)
SUBMISSION_COST = 0.05   # Paid per booking attempt (even if rejected)
EXEC_COST       = 0.20   # Paid only for executed passes
TAX_RATE        = 0.0    # Legacy single-stage cost (disabled)

# Ship kinematics
AVG_SPEED_KPH    = 37.0 / 2.0   # ~18.5 kph ≈ 10 knots
HEADING_VARIANCE = 0.075 / 5.0
SPEED_VARIANCE   = 0.5
R_earth_km = 6378

# Target of interest
Rotterdam = Location(3.8, 52.0, 0.0, "Rotterdam-ish")


# =============================================================================
# SATELLITE + WORLD HELPERS  (thin wrappers that pass SIMULATION_START)
# =============================================================================

def load_satellites_once() -> list:
    return _load_satellites_once(SIMULATION_START, LOOKAHEAD_HORIZON_H)


def create_world_and_constellations(cached_satellites, demand_field=None,
                                     execution_probability_function=None):
    return _create_world_and_constellations(
        cached_satellites, SIMULATION_START, demand_field,
        execution_probability_function,
    )


# =============================================================================
# SHIP TRAJECTORY
# =============================================================================

def generate_ship_trajectory(min_time: dt.datetime, max_time: dt.datetime,
                              seed: int = 4) -> list:
    """Ground-truth Rotterdam ship trajectory: sea-only random walk, 30-min steps."""
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
            print(f"  [Warning] Ship trajectory generator stuck at "
                  f"{trajectory[-1].lat_deg:.2f}, {trajectory[-1].lon_deg:.2f} — reusing last position")
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
# MSA WORKFLOW BUILDERS
# =============================================================================

def rewarder_observation(opportunity, preferred_zenith_angle_deg=45):
    """Imaging reward: static base + look-angle + sun-zenith + range quality."""
    static_reward = 50.0
    look_angle_reward = abs(90.0 - opportunity.look_angle_dec_deg) / 90.0
    zenith_angle_reward = abs(preferred_zenith_angle_deg - opportunity.sun_zenith_angle_deg) / 90.0
    range_reward = 1.0 / (opportunity.range_km / 1000.0)
    return static_reward + look_angle_reward + zenith_angle_reward + range_reward


def rewarder_search(opportunity):
    """SAR search reward: no static base, no sun-zenith term, just geometry."""
    look_angle_reward = abs(90.0 - opportunity.look_angle_dec_deg) / 90.0
    range_reward = 1.0 / (opportunity.range_km / 1000.0)
    return look_angle_reward + range_reward


def success_declarer(data_product):
    # Ship is "detected" if it appears in the data product (FOV check passed in simulator).
    # This drives timeline_updater_msa (det/greedy/random) and K_w resolution on replan
    # (stochastic). compute_metrics uses DATA_RECEIVED rows directly, not successful_execution,
    # so quality accounting is unaffected by this value.
    return len(data_product) > 0


def build_msa_workflow(min_time, max_time, initial_toi_pose):
    """
    MSA workflow for deterministic ILP / greedy / random schedulers.

    Uses a decaying Timeline to encode the ship-known state, with GREATER/LESSER_OR_EQUAL
    timeline constraints and WAIT_FOR_COMPLETION_IF_FEASIBLE SUCCESS edges exactly as in
    the mentor's notebook (FAME-msa-workflow-dispatch.ipynb, cells 29/31/33).
    """
    ship_timeline = Timeline(
        name="Ship loc. known?",
        initial_time=min_time,
        initial_value=1.0,
        initial_rate=-1.0 / 10800.0,
        min_value=-100.0,
        max_value=100.0,
    )

    init_req = ObservationRequest(
        lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
        min_time=min_time, max_time=min_time + dt.timedelta(hours=3),
        request_name="Initial RGB observation Rotterdam",
        min_elevation_deg=45.0, instrument=InstrumentType.RGB,
    )
    root_task = ConstrainedObservationRequest(
        name="Initial observation",
        observation_request=init_req,
        is_mandatory=True,
        timeline_constraints=[],
        timeline_impacts=[TaskTimelineImpact(
            timeline=ship_timeline, time=TaskImpactTime.POST,
            type=ImpactType.ADDITION, value=1.0,
        )],
        rewarder=rewarder_observation,
        success_declarer=success_declarer,
    )

    workflow_reqs = [root_task]
    policy_schedule = {c: True for c in ConstraintClass}
    policy_dispatch  = {c: False for c in ConstraintClass}

    for h_ix in range(FOLLOW_UP_INTERVAL_H, LOOKAHEAD_HORIZON_H, FOLLOW_UP_INTERVAL_H):
        window_ix = h_ix // FOLLOW_UP_INTERVAL_H
        prior_batch_requests = list(workflow_reqs)

        def _base_constraints(h, _prior=prior_batch_requests):
            constraints = [
                Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET,
                           root_task, {'offset': dt.timedelta(hours=h)}),
                Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_BEFORE_OFFSET,
                           root_task, {'offset': dt.timedelta(hours=h + FOLLOW_UP_INTERVAL_H)}),
            ]
            for existing_request in _prior:
                constraints.append(Constraint(
                    ConstraintClass.SUCCESS, SuccessConstraintType.WAIT_FOR_COMPLETION_IF_FEASIBLE,
                    existing_request,
                ))
            return constraints

        img_task = ConstrainedObservationRequest(
            name=f"Follow-up obs. {window_ix}",
            observation_request=ObservationRequest(
                lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
                min_time=min_time + dt.timedelta(hours=h_ix),
                max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H),
                request_name=f"Follow-up imaging {window_ix} Rotterdam",
                min_elevation_deg=45.0, instrument=InstrumentType.RGB,
            ),
            is_mandatory=False,
            task_constraints=_base_constraints(h_ix),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            success_declarer=success_declarer,
            timeline_constraints=[TaskTimelineConstraint(
                timeline=ship_timeline, time=TaskImpactTime.PRE,
                type=TimelineConstraintType.GREATER_OR_EQUAL, value=0.0,
            )],
            timeline_impacts=[TaskTimelineImpact(
                timeline=ship_timeline, time=TaskImpactTime.POST,
                type=ImpactType.ADDITION, value=1.0,
            )],
            rewarder=rewarder_observation,
            max_num_instances=1,
            request_group=f"Follow-up obs. {window_ix}",
        )

        for search_ix in range(MAX_SEARCHES_PER_INTERVAL):
            srch_task = ConstrainedObservationRequest(
                name=f"Follow-up search {window_ix}.{search_ix}",
                observation_request=ObservationRequest(
                    lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
                    min_time=min_time + dt.timedelta(hours=h_ix),
                    max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H),
                    request_name=f"Follow-up search {window_ix}.{search_ix} Rotterdam",
                    min_elevation_deg=45.0, instrument=InstrumentType.SAR,
                ),
                is_mandatory=False,
                task_constraints=_base_constraints(h_ix),
                schedule_policy_if_constraint_unsatisfied=policy_schedule,
                dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
                success_declarer=success_declarer,
                timeline_constraints=[TaskTimelineConstraint(
                    timeline=ship_timeline, time=TaskImpactTime.PRE,
                    type=TimelineConstraintType.LESSER_OR_EQUAL, value=0.0,
                )],
                timeline_impacts=[TaskTimelineImpact(
                    timeline=ship_timeline, time=TaskImpactTime.POST,
                    type=ImpactType.ADDITION, value=1.0,
                )],
                rewarder=rewarder_search,
                max_num_instances=1,
                request_group=f"Follow-up search {window_ix}",
            )
            workflow_reqs.append(srch_task)

        workflow_reqs.append(img_task)

    def timeline_updater_msa(current_time, requests, timelines,
                             max_time_without_observations=dt.timedelta(seconds=10800)):
        ship_position_is_known = any(
            getattr(r, 'completed', False) and getattr(r, 'successful_execution', False)
            and r.observation_opportunity is not None
            and (current_time - r.observation_opportunity.time) < max_time_without_observations
            for r in requests
        )
        tl = timelines[0]
        _, rate = tl._get_value_and_rate_at(current_time, print_debug=False)
        tl.reset_timeline(current_time, 1.0 if ship_position_is_known else 0.0, rate)
        return timelines

    def request_updater_msa(current_time, requests, timelines):
        initial_ship_location = Pose(
            time=SIMULATION_START,
            lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
            heading_deg=0.0, speed_kph=AVG_SPEED_KPH,
        )
        unfilled = [r for r in requests
                    if not r.completed and not r.dispatched and r.feasible
                    and r.observation_request.max_time > current_time]
        if not unfilled:
            return requests
        unfilled_by_time = {}
        for r in unfilled:
            t = r.observation_request.min_time + (
                r.observation_request.max_time - r.observation_request.min_time) / 2
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
                h3_resolution=5, samples_propagation=100,
                propagate_negative_samples=False, verbose=False,
            )
        except Exception as e:
            print(f"    [RequestUpdater] Propagation failed: {e}")
            return requests
        for t, req_list in unfilled_by_time.items():
            hexes = hexes_by_time.get(t)
            if hexes is None:
                continue
            for ix, req in enumerate(req_list):
                if ix >= len(hexes):
                    req.scheduled = req.dispatched = req.completed = True
                    req.feasible = False; req.successful_execution = False
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
    print(f"  [Workflow] Built MSA workflow: {1 + n_windows * (1 + MAX_SEARCHES_PER_INTERVAL)} tasks "
          f"({n_windows} windows × (1 imaging + {MAX_SEARCHES_PER_INTERVAL} search))")
    return Workflow(
        constrained_observation_requests=workflow_reqs,
        timelines=[ship_timeline],
        timeline_updater=timeline_updater_msa,
        request_updater=request_updater_msa,
    )


def build_msa_workflow_logical(min_time, max_time, initial_toi_pose):
    """
    MSA workflow for the STOCHASTIC "general_logical_dag" formulation.

    Replaces the Timeline mechanism with explicit AND/OR/NOT gates and belief-state
    LogicNodes that encode the multiplexer:

        K_1      = LogicNode( Lit(initial_obs) )
        I_w.gate = Lit(K_w)                          (image if tracked)
        S_{w,j}.gate = Not(Lit(K_w))                 (search if lost)
        U_w      = LogicNode( ExclusiveOr(S_{w,1..N}) )
        K_{w+1}  = LogicNode( Or( And(Lit(K_w), Lit(I_w)),
                                  And(Not(Lit(K_w)), Lit(U_w)) ) )

    DISPATCH POLICY: TEMPORAL / SUCCESS constraints are used only by the planner
    (window-pinning) and must NOT block dispatch. The solver pre-assigns all passes
    upfront; all passes are submitted simultaneously at t=0.
    """
    init_req = ObservationRequest(
        lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
        min_time=min_time, max_time=min_time + dt.timedelta(hours=3),
        request_name="Initial RGB observation Rotterdam",
        min_elevation_deg=45.0, instrument=InstrumentType.RGB,
    )
    root_task = ConstrainedObservationRequest(
        name="Initial observation",
        observation_request=init_req,
        is_mandatory=True,
        timeline_constraints=[], timeline_impacts=[],
        rewarder=rewarder_observation,
        success_declarer=success_declarer,
        request_group="Initial observation",
    )
    root_task.gate = None

    workflow_reqs = [root_task]
    logic_nodes = []

    policy_schedule = {c: True for c in ConstraintClass}
    # TEMPORAL + SUCCESS: ignore at dispatch time (gates handle conditional logic in MILP).
    policy_dispatch = {ConstraintClass.TEMPORAL: True, ConstraintClass.SUCCESS: True,
                       ConstraintClass.GEOMETRY: False}

    K_prev = LogicNode(name="K_1 (tracked?)", gate=Lit(root_task), request_group="state")
    logic_nodes.append(K_prev)

    for h_ix in range(FOLLOW_UP_INTERVAL_H, LOOKAHEAD_HORIZON_H, FOLLOW_UP_INTERVAL_H):
        window_ix = h_ix // FOLLOW_UP_INTERVAL_H

        def _temporal_constraints(h):
            return [
                Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET,
                           root_task, {'offset': dt.timedelta(hours=h)}),
                Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_BEFORE_OFFSET,
                           root_task, {'offset': dt.timedelta(hours=h + FOLLOW_UP_INTERVAL_H)}),
            ]

        img_task = ConstrainedObservationRequest(
            name=f"Follow-up obs. {window_ix}",
            observation_request=ObservationRequest(
                lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
                min_time=min_time + dt.timedelta(hours=h_ix),
                max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H),
                request_name=f"Follow-up imaging {window_ix} Rotterdam",
                min_elevation_deg=45.0, instrument=InstrumentType.RGB,
            ),
            is_mandatory=False,
            task_constraints=_temporal_constraints(h_ix),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            success_declarer=success_declarer,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=rewarder_observation,
            max_num_instances=max(2, MAX_NUM_INSTANCES),
            request_group=f"Follow-up obs. {window_ix}",
        )
        img_task.gate = Lit(K_prev)

        search_tasks = []
        for search_ix in range(MAX_SEARCHES_PER_INTERVAL):
            srch_task = ConstrainedObservationRequest(
                name=f"Follow-up search {window_ix}.{search_ix}",
                observation_request=ObservationRequest(
                    lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
                    min_time=min_time + dt.timedelta(hours=h_ix),
                    max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H),
                    request_name=f"Follow-up search {window_ix}.{search_ix} Rotterdam",
                    min_elevation_deg=45.0, instrument=InstrumentType.SAR,
                ),
                is_mandatory=False,
                task_constraints=_temporal_constraints(h_ix),
                schedule_policy_if_constraint_unsatisfied=policy_schedule,
                dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
                success_declarer=success_declarer,
                timeline_constraints=[], timeline_impacts=[],
                rewarder=rewarder_search,
                max_num_instances=1,
                request_group=f"Follow-up search {window_ix}",
            )
            srch_task.gate = Not(Lit(K_prev))
            search_tasks.append(srch_task)
            workflow_reqs.append(srch_task)
        workflow_reqs.append(img_task)

        U_w = LogicNode(name=f"U_{window_ix} (found?)",
                        gate=ExclusiveOr(*[Lit(s) for s in search_tasks]),
                        request_group="state")
        logic_nodes.append(U_w)

        K_next = LogicNode(
            name=f"K_{window_ix + 1} (tracked?)",
            gate=Or(And(Lit(K_prev), Lit(img_task)), And(Not(Lit(K_prev)), Lit(U_w))),
            request_group="state",
        )
        logic_nodes.append(K_next)
        K_prev = K_next

    def request_updater_msa_logical(current_time, requests, timelines):
        initial_ship_location = Pose(
            time=SIMULATION_START,
            lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
            heading_deg=0.0, speed_kph=AVG_SPEED_KPH,
        )
        unfilled = [r for r in requests
                    if not r.completed and not r.dispatched and r.feasible
                    and r.observation_request.max_time > current_time]
        if not unfilled:
            return requests
        unfilled_by_time = {}
        for r in unfilled:
            t = r.observation_request.min_time + (
                r.observation_request.max_time - r.observation_request.min_time) / 2
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
                h3_resolution=5, samples_propagation=100,
                propagate_negative_samples=False, verbose=False,
            )
        except Exception as e:
            print(f"    [RequestUpdater/logical] Propagation failed: {e} — keeping coords, p_det=1")
            return requests

        for t, req_list in unfilled_by_time.items():
            hexes = hexes_by_time.get(t)
            if hexes is None or len(hexes) == 0:
                continue
            total_particles = float(hexes['point_count'].sum())
            if total_particles <= 0:
                continue
            top_mass = float(hexes.iloc[0]['point_count']) / total_particles
            searches = [r for r in req_list if r.observation_request.instrument == InstrumentType.SAR]
            imaging  = [r for r in req_list if r.observation_request.instrument != InstrumentType.SAR]
            for r in imaging:
                r.detection_prob = top_mass
            for ix, req in enumerate(searches):
                if ix >= len(hexes):
                    req.detection_prob = 0.0
                    req.scheduled = req.dispatched = req.completed = True
                    req.feasible = False; req.successful_execution = False
                else:
                    row = hexes.iloc[ix]
                    req.detection_prob = float(row['point_count']) / total_particles
                    centroid = row.geometry.centroid
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
    print(f"  [Workflow] Built MSA LOGICAL workflow: {len(workflow_reqs)} observation tasks "
          f"+ {len(logic_nodes)} belief-state nodes ({n_windows} windows)")
    return Workflow(
        constrained_observation_requests=workflow_reqs,
        timelines=[],
        timeline_updater=lambda t, r, tl: tl,
        request_updater=request_updater_msa_logical,
    )


# =============================================================================
# GATE-AWARE REACHABILITY  (MSA-specific fair completion denominator)
# =============================================================================

def compute_msa_reachability(tasks, completed_tasks):
    """
    Return the set of reachable tasks given realized completions.

    The MSA workflow is a per-window belief-state multiplexer. A task is reachable
    only if the branch leading to it was actually taken:

        K_1        := initial observation succeeded
        imaging_w  reachable  <=>  K_w  (tracked → image)
        search_w.* reachable  <=>  not K_w  (lost → search)
        K_{w+1}    := (K_w and imaging_w succeeded)
                       or (not K_w and any reachable search_w.* succeeded)

    Reconstructed from task names so it applies identically to ALL tracks.
    """
    def succeeded(t):
        return t in completed_tasks and getattr(t, 'successful_execution', True)

    root = None
    imaging_by_w = {}
    searches_by_w = {}
    for t in tasks:
        nm = t.observation_request.name
        if nm.startswith("Initial"):
            root = t
        elif nm.startswith("Follow-up imaging"):
            imaging_by_w[int(nm.split()[2])] = t
        elif nm.startswith("Follow-up search"):
            w = int(nm.split()[2].split('.')[0])
            searches_by_w.setdefault(w, []).append(t)

    reachable = set()
    if root is not None:
        reachable.add(root)
        K_w = succeeded(root)
    else:
        K_w = False

    for w in sorted(set(imaging_by_w) | set(searches_by_w)):
        img     = imaging_by_w.get(w)
        searches = searches_by_w.get(w, [])
        if K_w:
            if img is not None:
                reachable.add(img)
            K_w = img is not None and succeeded(img)
        else:
            for s in searches:
                reachable.add(s)
            K_w = any(succeeded(s) for s in searches)

    return reachable


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(workflow_graph, broker):
    """
    Wrapper around compute_metrics_v3 that adds MSA gate-aware reachability.

    compute_metrics_v3 counts all tasks reachable via graph START_IF_SUCCESSFUL edges.
    The MSA logical workflow has no such edges (gates live in LogicNode, not graph edges),
    so v3 treats every task as reachable. We replace the reachability denominator and
    the realized-quality numerator with gate-aware versions via compute_msa_reachability.

    Completion is determined from broker._requests DATA_RECEIVED rows (ground truth),
    not from task.successful_execution (which could be False if the data product is empty).
    """
    m = compute_metrics_v3(
        workflow_graph, broker, ObservationStatus,
        submission_cost_rate=SUBMISSION_COST,
        execution_cost_rate=EXEC_COST,
        verbose=True,
    )

    # Build the DATA_RECEIVED-based completed set from broker records
    reqs = broker._requests
    tasks = list(workflow_graph.nodes())
    obsreq_to_task = {t.observation_request: t for t in tasks}

    executed_tasks = set()
    best_quality_by_task = {}
    for _, row in reqs.iterrows():
        if row['status'] != ObservationStatus.DATA_RECEIVED:
            continue
        rp = row['requested_pass']
        if rp is None:
            continue
        task = obsreq_to_task.get(row['request'])
        if task is None:
            continue
        executed_tasks.add(task)
        q = task.rewarder(rp.highest)
        if q > best_quality_by_task.get(task, -1e9):
            best_quality_by_task[task] = q

    reachable = compute_msa_reachability(tasks, executed_tasks)
    n_reachable = len(reachable)
    valid_completed = executed_tasks & reachable
    n_valid = len(valid_completed)
    gate_rate = n_valid / n_reachable if n_reachable else 0.0
    total_quality = sum(best_quality_by_task.get(t, 0.0) for t in valid_completed)

    print(f"   [MSA-Reachable] {n_valid}/{n_reachable} ({100*gate_rate:.1f}%) "
          f"gate-reachable tasks completed")

    # Drop v3 fields that don't apply to MSA's gate-conditioned structure —
    # v3 has no START_IF_SUCCESSFUL edges so it treats all 31 tasks as reachable,
    # which makes its reachable/completion counts meaningless for this workflow.
    for key in ('n_tasks_completed_valid', 'n_tasks_reachable',
                'reachable_task_completion_rate', 'realized_quality',
                'utility', 'task_completion_rate', 'raw_task_completion_rate'):
        m.pop(key, None)

    # Gate-aware headline fields
    m['n_tasks_reachable']     = n_reachable   # tasks on the branch actually taken
    m['n_tasks_completed']     = n_valid        # of those, how many got data
    m['task_completion_rate']  = gate_rate      # n_tasks_completed / n_tasks_reachable
    m['realized_quality']      = total_quality
    m['utility']               = total_quality - m['total_cost']

    return m


# =============================================================================
# DEMAND FIELD
# =============================================================================

def build_demand_field(min_time):
    """Build and precompute a dynamic acceptance-probability DemandField."""
    _demand_cfg = DemandFieldConfig(use_constant_probability=False,
                                    p_min=P_ACC_MIN, p_max=P_ACC_MAX)
    demand_field = DemandField(
        config=_demand_cfg,
        reference_time=min_time,
        horizon_s=LOOKAHEAD_HORIZON_H * 3600.0,
    )
    demand_field.add_spike(Rotterdam.lat_deg, Rotterdam.lon_deg, min_time)
    _all_constellation_names = ["Planet", "Umbra", "Capella", "LOFT",
                                 "Ubotica", "Mission Control", "AC", "ICEYE"]
    print("[DemandField] Precomputing demand trajectories...")
    demand_field.precompute(_all_constellation_names)
    print("[DemandField] Precompute complete.")
    return demand_field


def make_probability_functions(demand_field=None):
    """Return (acceptance_prob_fn, execution_prob_fn) for planner + simulator."""
    if demand_field is not None:
        def acceptance_prob_fn(constrained_request, satellite, obs_pass):
            return demand_field.make_acceptance_prob_function()(constrained_request, satellite, obs_pass)
    else:
        def acceptance_prob_fn(constrained_request, satellite, obs_pass):
            name = satellite.name.upper()
            if any(x in name for x in ["SKYSAT", "PELICAN", "TANAGER", "FLOCK"]):
                return 0.70
            elif "UMBRA"   in name: return 0.85
            elif "CAPELLA" in name or "ACADIA" in name: return 0.90
            elif "LOFT"    in name or "YAM"    in name: return 0.92
            elif "UBOTICA" in name or "HAMMER" in name or "ACCENTURE" in name: return 0.93
            elif "PERSISTENCE" in name or "LEMUR" in name: return 0.94
            elif "AEROCUBE" in name: return 0.95
            elif "ICEYE"   in name: return 0.96
            return 0.85

    def execution_prob_fn(constrained_request, satellite, obs_pass):
        opp = obs_pass.highest if hasattr(obs_pass, 'highest') else obs_pass
        look_angle = abs(90.0 - opp.look_angle_dec_deg)
        t = look_angle / 90.0
        return max(P_EXEC_MIN, min(P_EXEC_MAX, P_EXEC_MAX - t * (P_EXEC_MAX - P_EXEC_MIN)))

    return acceptance_prob_fn, execution_prob_fn


# =============================================================================
# SINGLE-SCHEDULER RUN
# =============================================================================

def run_one_scheduler(scheduler, seed, cached_satellites, demand_field,
                      min_time, max_time, results_dir, plot_schedule=True):
    """
    Build world + workflow + broker for ONE scheduler and ONE seed, run the
    simulation, compute metrics, and write run_seed<seed>_<scheduler>.json.
    """
    plots_dir = os.path.join(results_dir, "plots", scheduler)
    os.makedirs(plots_dir, exist_ok=True)

    acceptance_prob_fn, execution_prob_fn = make_probability_functions(
        demand_field if USE_DYNAMIC_DEMAND_FIELD else None
    )

    def detection_prob_fn(constrained_request, satellite, obs_pass):
        return float(getattr(constrained_request, 'detection_prob', 1.0))

    print(f"\n  Running {scheduler} (seed {seed})...  [RSS {rss_gb():.2f} GB]")
    random.seed(seed)
    np.random.seed(seed)

    world, constellations = create_world_and_constellations(
        cached_satellites, demand_field=demand_field if USE_DYNAMIC_DEMAND_FIELD else None,
    )

    # Ship phenomena are per-seed
    phenomena = generate_ship_trajectory(min_time, max_time, seed=seed)
    for p in copy.deepcopy(phenomena):
        world.phenomena.append(p)

    use_stochastic = (scheduler == 'stochastic')
    use_ilp   = scheduler in ('stochastic', 'deterministic')
    use_random = (scheduler == 'random')

    template = (build_msa_workflow_logical(min_time, max_time,
                    Pose(lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg,
                         time=min_time, alt_km=Rotterdam.alt_km,
                         heading_deg=0.0, speed_kph=AVG_SPEED_KPH, name="Rotterdam initial"))
                if use_stochastic
                else build_msa_workflow(min_time, max_time,
                    Pose(lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg,
                         time=min_time, alt_km=Rotterdam.alt_km,
                         heading_deg=0.0, speed_kph=AVG_SPEED_KPH, name="Rotterdam initial")))

    broker = Broker(constellations=constellations, world=world, name=f"Broker-{scheduler}")
    broker.add_workflow(copy.deepcopy(template))
    world.add_broker(broker)

    m = None
    try:
        t_start = time.time()

        schedule_kwargs = dict(
            current_time=world.time,
            use_ilp=use_ilp,
            use_stochastic=use_stochastic,
            use_random=use_random,
            random_seed=seed,
            max_solver_time_s=MAX_SOLVER_TIME_S,
            solver_engine="GUROBI",
            update_timelines=False,
            update_requests=True,   # particle filter retargeting for all schedulers
            tax_rate=TAX_RATE,
            max_reschedule_depth=10000,
            plot_schedule=plot_schedule,
            save_schedule_plot=plot_schedule,
            results_path=plots_dir,
            submission_cost_rate=SUBMISSION_COST,
            execution_cost_rate=EXEC_COST,
        )
        if use_stochastic:
            schedule_kwargs.update(
                stochastic_formulation="general_logical_dag",
                acceptance_probability_function=acceptance_prob_fn,
                execution_probability_function=execution_prob_fn,
                detection_probability_function=detection_prob_fn,
                enable_cancellations=ENABLE_CANCELLATIONS,
            )

        broker.schedule_workflow_redundant(**schedule_kwargs)
        run_simulation_forward(world)
        m = compute_metrics(broker._workflow_graph, broker)
        elapsed = time.time() - t_start

        m['scheduler']         = scheduler
        m['seed']              = seed
        m['tax_rate']          = TAX_RATE
        m['max_num_instances'] = MAX_NUM_INSTANCES
        m['sim_start']         = SIMULATION_START.isoformat()
        m['elapsed_s']         = elapsed

        run_file = os.path.join(results_dir, f"run_seed{seed:04d}_{scheduler}.json")
        with open(run_file, 'w') as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  [Saved] {run_file}")

    except Exception as e:
        import traceback
        print(f"  [Error] {scheduler} seed {seed}: {e}")
        traceback.print_exc()

    try:
        del broker, world, constellations
    except Exception:
        pass
    plt.close('all')
    gc.collect()
    print(f"  [Mem] after {scheduler} seed {seed}: peak RSS {rss_gb():.2f} GB")
    return m


# =============================================================================
# AGGREGATE
# =============================================================================

def load_records(results_dir):
    import glob as _glob
    records = []
    for path in sorted(_glob.glob(os.path.join(results_dir, "run_*.json"))):
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
            print(f"[Warning] could not read {path}: {e}")
    print(f"[Aggregate] loaded {len(records)} run files from {results_dir}")
    return records


def aggregate(results_dir, records=None):
    if records is None:
        records = load_records(results_dir)
    if not records:
        print("[Warning] No records to aggregate.")
        return []

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    print("\n" + "=" * 70)
    print("MSA SHIP TRACKING — SUMMARY")
    print("=" * 70)

    for sched in SCHEDULERS:
        sub = df[df['scheduler'] == sched]
        if sub.empty:
            continue
        print(f"\n{sched.upper()}  (n={len(sub)})")
        for col, label in [
            ('task_completion_rate', 'Task completion (gate-aware)'),
            ('group_completion_rate', 'Group completion rate'),
            ('realized_quality',      'Realized quality'),
            ('utility',               'Net utility'),
            ('total_cost',            'Total cost'),
        ]:
            if col in sub.columns:
                print(f"  {label:35s}: {sub[col].mean():.3f} ± {sub[col].std():.3f}")
        for col, label in [
            ('n_submissions', 'Bookings submitted'),
            ('n_accepted',    '  accepted'),
            ('n_executed',    '  executed ok'),
            ('n_cancelled',   '  cancelled'),
            ('n_rejected',    '  rejected'),
        ]:
            if col in sub.columns:
                print(f"  {label:35s}: {sub[col].mean():.1f}")
        for col, label in [
            ('submitted_passes_per_task',  'Submitted passes/task'),
            ('exec_passes_per_completed',  'Executed passes/completed'),
            ('rejection_rate',             'Rejection rate'),
            ('replans',                    'Replans'),
        ]:
            if col in sub.columns:
                print(f"  {label:35s}: {sub[col].mean():.2f}")

    # Paired comparison vs deterministic
    if 'deterministic' in set(df.get('scheduler', [])):
        paired_summary(records, SCHEDULERS,
                       primary='task_completion_rate', baseline='deterministic')

    # Cost frontier
    try:
        plot_cost_frontier(records, SCHEDULERS,
                           out_path=os.path.join(results_dir, "cost_frontier.png"),
                           y='task_completion_rate', x='total_cost')
    except Exception as e:
        print(f"[Warning] cost frontier: {e}")

    # Boxplot
    try:
        present = [s for s in SCHEDULERS if s in set(df['scheduler'])]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.boxplot([df[df['scheduler'] == s]['task_completion_rate'].values for s in present],
                    labels=present)
        ax1.set_title("Task completion rate (gate-aware)")
        ax1.set_ylabel("Fraction completed")
        ax1.grid(axis='y', alpha=0.4)
        ax2.boxplot([df[df['scheduler'] == s]['utility'].values for s in present],
                    labels=present)
        ax2.set_title("Net utility  (quality − cost)")
        ax2.set_ylabel("Utility")
        ax2.grid(axis='y', alpha=0.4)
        plt.suptitle(f"MSA Ship Tracking — {len(df)} runs", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "comparison_plots.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Warning] boxplot: {e}")

    return records


# =============================================================================
# ALL-IN-ONE LOOP
# =============================================================================

def run_comparison(num_monte_carlo_runs=NUM_MONTE_CARLO_RUNS,
                   schedulers=None, results_dir=None):
    schedulers = schedulers or SCHEDULERS
    if results_dir is None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = os.path.join("results", f"msa_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("MSA SHIP TRACKING — SCHEDULER COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  SIMULATION_START:    {SIMULATION_START.isoformat()}")
    print(f"  Horizon:             {LOOKAHEAD_HORIZON_H}h")
    print(f"  Follow-up windows:   every {FOLLOW_UP_INTERVAL_H}h")
    print(f"  Search tasks/window: {MAX_SEARCHES_PER_INTERVAL}")
    print(f"  MAX_NUM_INSTANCES:   {MAX_NUM_INSTANCES}")
    print(f"  ENABLE_CANCELLATIONS:{ENABLE_CANCELLATIONS}")
    print(f"  Monte Carlo runs:    {num_monte_carlo_runs}")
    print(f"  Schedulers:          {schedulers}")
    print(f"  Submission cost:     {SUBMISSION_COST}")
    print(f"  Execution cost:      {EXEC_COST}")
    print(f"  Solver time limit:   {MAX_SOLVER_TIME_S}s")
    print(f"  Acceptance model:    {'DYNAMIC DemandField' if USE_DYNAMIC_DEMAND_FIELD else 'STATIC constants'}")
    print(f"\n[Results] Saving to: {results_dir}")

    cached_satellites = load_satellites_once()
    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)

    demand_field = build_demand_field(min_time) if USE_DYNAMIC_DEMAND_FIELD else None
    if demand_field is not None:
        try:
            demand_field.plot_heatmaps(os.path.join(results_dir, "demand_heatmaps"))
        except Exception as e:
            print(f"[Warning] demand heatmaps: {e}")

    all_records = []
    for run_idx in range(num_monte_carlo_runs):
        run_seed = 42 + run_idx
        print(f"\n--- Monte Carlo Iteration {run_idx + 1}/{num_monte_carlo_runs} (seed {run_seed}) ---")
        for scheduler in schedulers:
            m = run_one_scheduler(scheduler, run_seed, cached_satellites, demand_field,
                                  min_time, max_time, results_dir)
            if m is not None:
                all_records.append(m)

    aggregate(results_dir, records=all_records)
    return all_records


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    global SIMULATION_START

    parser = argparse.ArgumentParser(description="MSA ship tracking scheduler comparison")
    parser.add_argument('--scheduler', choices=SCHEDULERS,
                        help="Run exactly one scheduler and exit.")
    parser.add_argument('--seed', type=int, help="Seed for the single run.")
    parser.add_argument('--start', type=str,
                        help="ISO simulation start, e.g. 2026-08-01T00:00:00. "
                             "MUST be identical across every process in a campaign.")
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--aggregate', action='store_true',
                        help="Only aggregate an existing --results-dir.")
    parser.add_argument('--runs', type=int, default=NUM_MONTE_CARLO_RUNS,
                        help="Number of seeds for the in-process loop.")
    parser.add_argument('--schedulers', nargs='+', choices=SCHEDULERS,
                        help="Override SCHEDULERS list for the in-process loop.")
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
        if not args.start:
            print("[WARNING] --start not given: geometry may not match other processes in a campaign.")
        results_dir = args.results_dir or os.path.join(
            "results", f"msa_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)
        min_time = SIMULATION_START
        max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
        cached_satellites = load_satellites_once()
        demand_field = build_demand_field(min_time) if USE_DYNAMIC_DEMAND_FIELD else None
        run_one_scheduler(args.scheduler, args.seed, cached_satellites, demand_field,
                          min_time, max_time, results_dir,
                          plot_schedule=not args.no_schedule_plots)
        print(f"[Done] {args.scheduler} seed {args.seed}; peak RSS {rss_gb():.2f} GB")
        return

    # Legacy in-process loop
    run_comparison(
        num_monte_carlo_runs=args.runs,
        schedulers=args.schedulers or SCHEDULERS,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
