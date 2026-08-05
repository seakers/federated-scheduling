"""
volcano_utils.py
----------------
Domain machinery for Volcano Eruption Monitoring in FAME:
1. GVP Database loaders for high-priority target volcanoes.
2. Ground-truth physical Eruption & dynamic Plume phenomenon registration.
3. Workflow DAG builder (Eruption Detection + Parallel Volume & Plume Follow-ups).
4. Dynamic updaters for StochasticTimelines and atmospheric plume retargeting.
"""

import datetime as dt
import pandas as pd
import numpy as np

# FAME imports
from fame_geometry import Location, InstrumentType, ObservationOpportunity
from fame_agents_base import Phenomenon
from fame_workflow import (
    ObservationRequest,
    ConstrainedObservationRequest,
    Constraint,
    ConstraintClass,
    TemporalConstraintType,
    SuccessConstraintType,
    TaskTimelineConstraint,
    TaskImpactTime,
    TimelineConstraintType,
    Workflow,
)
from fame_workflow_stochastic import StochasticTimeline


# =============================================================================
# 1. Ground-Truth Phenomena (Eruption & Drifting Plume)
# =============================================================================

class Eruption(Phenomenon):
    """
    Physical ground-truth eruption phenomenon required by FAME's observation simulator.
    When satellites observe a target location during [start_time, end_time],
    the simulator queries active phenomena to populate valid data products.
    """
    def __init__(
        self,
        lon_deg: float,
        lat_deg: float,
        alt_km: float,
        start_time: dt.datetime,
        end_time: dt.datetime,
        VEI: float = 5.0,
        name: str = ""
    ):
        super().__init__(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            alt_km=alt_km,
            heading_deg=None,
            speed_kph=None,
            start_time=start_time,
            end_time=end_time,
            name=name
        )
        self.VEI = VEI


class Plume(Phenomenon):
    """
    Physical ground-truth dynamic ash plume phenomenon.
    Advects downwind over time for atmospheric dispersion tracking and multispectral imaging.
    """
    def __init__(
        self,
        lon_deg: float,
        lat_deg: float,
        alt_km: float,
        heading_deg: float,
        speed_kph: float,
        start_time: dt.datetime,
        end_time: dt.datetime,
        name: str = ""
    ):
        super().__init__(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            alt_km=alt_km,
            heading_deg=heading_deg,
            speed_kph=speed_kph,
            start_time=start_time,
            end_time=end_time,
            name=name
        )


def register_volcano_phenomena(
    world,
    volcano_locations: list[Location],
    min_time: dt.datetime,
    max_time: dt.datetime,
    wind_speed_kph: float = 40.0,
    wind_heading_deg: float = 45.0,
    plume_alt_km: float = 12.0,
    particle_interval_h: float = 0.5
):
    """
    Registers active Eruption phenomena and Lagrangian plume particle tracers.

    One eruption covers the full planning window. For the plume, one discrete
    tracer is spawned every `particle_interval_h` hours (default 30 min). Each
    tracer is active for exactly one interval and placed at its advected position
    at the midpoint of that interval, approximating a continuous Lagrangian cloud.
    A satellite overpass at time t observes whichever tracers are active at t
    and whose position falls within the instrument FOV.

    FAME's simulator uses static lon/lat from the Phenomenon object, so continuous
    within-window advection is not modelled. 30-min intervals keep position error
    below ~20 km for typical plume speeds.
    """
    particle_interval_td = dt.timedelta(hours=particle_interval_h)
    duration_h = (max_time - min_time).total_seconds() / 3600.0
    n_particles = max(1, int(np.ceil(duration_h / particle_interval_h)))

    for vloc in volcano_locations:
        eruption = Eruption(
            lon_deg=vloc.lon_deg,
            lat_deg=vloc.lat_deg,
            alt_km=vloc.alt_km,
            start_time=min_time,
            end_time=max_time,
            name=vloc.name,
            VEI=5.0
        )
        world.add_phenomenon(eruption)

        for k in range(n_particles):
            t_spawn = min_time + k * particle_interval_td
            t_end = min(t_spawn + particle_interval_td, max_time)
            elapsed_at_midpoint_h = (k + 0.5) * particle_interval_h
            plume_lon, plume_lat = calculate_plume_position(
                vloc.lon_deg, vloc.lat_deg,
                elapsed_at_midpoint_h,
                wind_speed_kph, wind_heading_deg
            )
            particle = Plume(
                lon_deg=plume_lon,
                lat_deg=plume_lat,
                alt_km=plume_alt_km,
                heading_deg=wind_heading_deg,
                speed_kph=wind_speed_kph,
                start_time=t_spawn,
                end_time=t_end,
                name=f"{vloc.name}_plume_p{k:04d}"
            )
            world.add_phenomenon(particle)

    print(f"[VolcanoUtils] Registered {len(volcano_locations)} Eruptions + "
          f"{len(volcano_locations) * n_particles} Plume particles "
          f"(interval={particle_interval_h}h, n={n_particles}) in world.")


# =============================================================================
# 2. Database Loader
# =============================================================================

def load_volcano_locations_from_database(
    volcano_excel_path: str = 'data/GVP_Volcano_List_Holocene_202606021456.xlsx',
    eruption_excel_path: str = 'data/GVP_Eruption_List_Holocene_20260424.xlsx',
    min_vei: float = 4.0,
    min_start_year: int = 1900
) -> list[Location]:
    """
    Reads the global GVP Holocene databases, merges eruption catalogs,
    and isolates high-priority target positions based on VEI and recency.
    """
    print("[VolcanoUtils] Reading GVP Volcano and Eruption database files...")
    volcano_df = pd.read_excel(volcano_excel_path, header=1)
    eruption_df = pd.read_excel(eruption_excel_path, sheet_name="Eruption List", header=1)

    eruptions = eruption_df.merge(volcano_df, on='Volcano Name', how='left')

    filtered_volcanoes = [
        Location(
            lon_deg=row['Longitude'],
            lat_deg=row['Latitude'],
            alt_km=float(row['Elevation (m)']) / 1e3,
            name=row['Volcano Name'],
        )
        for ix, row in eruptions.iterrows()
        if row['VEI'] > min_vei and row['Start Year'] > min_start_year
    ]

    unique_locations = list(set(filtered_volcanoes))
    print(f"[VolcanoUtils] Loaded {len(unique_locations)} target volcanoes into scenario context.")
    return unique_locations


# =============================================================================
# 3. Callbacks & Rewarders
# =============================================================================

def rewarder_observation(opportunity: ObservationOpportunity, preferred_zenith_angle_deg=45):
    """Calculates reward based on look angle, sun zenith angle, and range."""
    static_reward = 50.0
    look_angle_reward = abs(90.0 - opportunity.look_angle_dec_deg) / 90.0
    zenith_angle_reward = abs(preferred_zenith_angle_deg - opportunity.sun_zenith_angle_deg) / 90.0
    range_reward = 1.0 / (opportunity.range_km / 1000.0)
    return (look_angle_reward + zenith_angle_reward + range_reward)*static_reward


def success_declarer_eruption(data_product):
    return len(data_product) > 0


def success_declarer_plume(data_product):
    return len(data_product) > 0


def timeline_updater_volcanoes(current_time: dt.datetime, requests: list, timelines: list):
    """Refreshes StochasticTimelines when satellite observations successfully complete."""
    for tl in timelines:
        if isinstance(tl, StochasticTimeline):
            tl.refresh_if_observed(current_time, requests)
    return timelines


# =============================================================================
# 4. Dynamic Plume Propagation & Retargeting Updaters
# =============================================================================

def calculate_plume_position(
    origin_lon: float,
    origin_lat: float,
    elapsed_hours: float,
    wind_speed_kph: float = 40.0,
    wind_heading_deg: float = 45.0
) -> tuple[float, float]:
    """
    Computes advected plume center coordinates given wind velocity vector and elapsed time.
    """
    distance_km = wind_speed_kph * elapsed_hours
    heading_rad = np.radians(wind_heading_deg)

    # Approximate lat/lon shifts on Earth's surface
    delta_lat = (distance_km * np.cos(heading_rad)) / 111.0
    mean_lat_rad = np.radians(origin_lat + delta_lat / 2.0)
    delta_lon = (distance_km * np.sin(heading_rad)) / (111.0 * np.cos(mean_lat_rad))

    return origin_lon + delta_lon, origin_lat + delta_lat


def request_updater_volcanoes(
    current_time: dt.datetime,
    requests: list[ConstrainedObservationRequest],
    timelines: list,
    volcano_map: dict[str, Location] = None,
    wind_speed_kph: float = 40.0,
    wind_heading_deg: float = 45.0,
    plume_alt_km: float = 12.0,
    n_recent_obs: int = 10
):
    """
    Observation-driven plume retargeting (mirrors the notebook approach).

    For each undispatched plume follow-up request:
    1. Collect all successfully completed plume observations for this volcano.
    2. For each past observation, extract the plume phenomenon position from
       the data product and dead-reckon it forward to the target window midpoint
       using the wind vector (propagate_plume_observation).
    3. Take the median of the N most recent propagated positions and retarget
       the request there.
    4. Fall back to open-loop dead-reckoning from the vent when no observations
       have been made yet.
    """
    volcano_map = volcano_map or {}

    # Index completed plume observations by volcano name for fast lookup.
    # Key: volcano_name (without "_plume" suffix), Value: list of (obs_time, lon, lat)
    plume_obs_by_volcano: dict[str, list] = {}
    for req in requests:
        if not (getattr(req, 'completed', False) and getattr(req, 'successful_execution', False)):
            continue
        if "_plume_followup_" not in req.name:
            continue
        volcano_name = req.request_group.replace("_plume", "")
        obs_time = None
        if req.observation_opportunity is not None:
            obs_time = req.observation_opportunity.time
        if obs_time is None:
            continue
        # Extract plume position from data product (list of Phenomenon objects)
        dp = getattr(req, 'data_product', []) or []
        plume_phenomena = [p for p in dp if hasattr(p, 'lon_deg') and hasattr(p, 'lat_deg')]
        if not plume_phenomena:
            continue
        obs_lon = float(np.mean([p.lon_deg for p in plume_phenomena]))
        obs_lat = float(np.mean([p.lat_deg for p in plume_phenomena]))
        plume_obs_by_volcano.setdefault(volcano_name, []).append((obs_time, obs_lon, obs_lat))

    for req in requests:
        if getattr(req, 'dispatched', False):
            continue
        if "_plume_followup_" not in req.name:
            continue

        volcano_name = req.request_group.replace("_plume", "")

        try:
            offset_str = req.name.split("_plume_followup_")[-1].replace("h", "")
            offset_hours = float(offset_str)
        except ValueError:
            offset_hours = 0.0

        # Target: midpoint of the follow-up observation window
        target_window_mid_h = offset_hours + 1.5
        if volcano_name in volcano_map:
            base_loc = volcano_map[volcano_name]
            base_lon, base_lat = base_loc.lon_deg, base_loc.lat_deg
        else:
            base_lon = req.observation_request.lon_deg
            base_lat = req.observation_request.lat_deg

        past_obs = plume_obs_by_volcano.get(volcano_name, [])

        if past_obs:
            # Sort by observation time, take N most recent
            past_obs_sorted = sorted(past_obs, key=lambda x: x[0])[-n_recent_obs:]

            propagated = []
            for obs_time, obs_lon, obs_lat in past_obs_sorted:
                # Elapsed hours from this observation to the target window midpoint.
                # The target midpoint is min_time + target_window_mid_h.
                # We don't carry min_time here, but we know:
                #   target window start = req.observation_request.min_time
                target_mid_abs = req.observation_request.min_time + dt.timedelta(hours=1.5)
                elapsed_h = (target_mid_abs - obs_time).total_seconds() / 3600.0
                if elapsed_h < 0:
                    # Observation is in the future relative to target window — skip
                    continue
                prop_lon, prop_lat = calculate_plume_position(
                    obs_lon, obs_lat, elapsed_h, wind_speed_kph, wind_heading_deg
                )
                propagated.append((prop_lon, prop_lat))

            if propagated:
                new_lon = float(np.median([p[0] for p in propagated]))
                new_lat = float(np.median([p[1] for p in propagated]))
            else:
                # All observations were future-dated (shouldn't happen); fall back
                new_lon, new_lat = calculate_plume_position(
                    base_lon, base_lat, target_window_mid_h, wind_speed_kph, wind_heading_deg
                )
        else:
            # No observations yet: open-loop dead-reckoning from the vent
            new_lon, new_lat = calculate_plume_position(
                base_lon, base_lat, target_window_mid_h, wind_speed_kph, wind_heading_deg
            )

        req.observation_request.lon_deg = float(new_lon)
        req.observation_request.lat_deg = float(new_lat)
        req.observation_request.alt_km = plume_alt_km

    return requests


# =============================================================================
# 5. Workflow Generator
# =============================================================================

def create_volcano_workflow(
    volcano_locations: list[Location],
    min_time: dt.datetime,
    max_time: dt.datetime,
    lookahead_horizon_h: int = 18,
    follow_up_interval_h: int = 3,
    hours_to_detect: int = 6,
    max_num_instances: int = 5,
    wind_speed_kph: float = 40.0,
    wind_heading_deg: float = 45.0,
    plume_alt_km: float = 12.0
) -> Workflow:
    """
    Generates the complete Volcano Eruption Monitoring workflow:
    1. Eruption Detection Task per volcano (hours 0-6).
    2. Dual parallel follow-up windows [h, h+3h] for Volume & Plume observations.
    3. Mandatory START_IF_SUCCESSFUL gates linking follow-ups to root detection.
    4. Plume retargeting machinery shifting plume observation coordinates downwind.
    """
    constrained_requests = []
    timelines = {}
    volcano_map = {v.name: v for v in volcano_locations}

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

    for volcano in volcano_locations:
        # Stochastic Timeline representing activity probability (24h decay half-life)
        timeline = StochasticTimeline(
            name=f"{volcano.name}_active",
            initial_time=min_time,
            initial_value=1.0,
            half_life_s=86400.0,
            min_value=-30.0,
            max_value=30.0
        )
        timelines[volcano] = timeline
        max_time_detection = min_time + dt.timedelta(hours=hours_to_detect)
        # 1. Root Eruption Detection Task (RGB, 0-6h window)
        detection_request = ObservationRequest(
            lon_deg=volcano.lon_deg, lat_deg=volcano.lat_deg, alt_km=volcano.alt_km,
            min_time=min_time, max_time=max_time_detection,
            instrument=InstrumentType.RGB,
            request_name=f"{volcano.name}_detection",
            min_elevation_deg=20.0
        )

        detection_task = ConstrainedObservationRequest(
            name=f"{volcano.name}_detection",
            observation_request=detection_request,
            is_mandatory=True,
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[],
            timeline_impacts=[],
            rewarder=rewarder_observation,
            success_declarer=success_declarer_eruption,
            request_group=volcano.name,
            max_num_instances=max_num_instances
        )
        constrained_requests.append(detection_task)

        # 2. Follow-up Windows (Volume + Plume tasks per window)
        # Start at follow_up_interval_h, not 0, to avoid colliding with the detection task window.
        for follow_up_ix in range(hours_to_detect, lookahead_horizon_h, follow_up_interval_h):
            obs_task_constraints = [
                Constraint(
                    ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET,
                    detection_task, {'offset': dt.timedelta(hours=follow_up_ix)}
                ),
                Constraint(
                    ConstraintClass.TEMPORAL, TemporalConstraintType.START_BEFORE_OFFSET,
                    detection_task, {'offset': dt.timedelta(hours=follow_up_ix + follow_up_interval_h)}
                ),
                Constraint(
                    ConstraintClass.SUCCESS, SuccessConstraintType.START_IF_SUCCESSFUL,
                    detection_task
                ),
            ]

            # Volume Follow-up Task (Targeted at vent location)
            vol_req = ObservationRequest(
                lon_deg=volcano.lon_deg, lat_deg=volcano.lat_deg, alt_km=volcano.alt_km,
                min_time=min_time + dt.timedelta(hours=follow_up_ix),
                max_time=min_time + dt.timedelta(hours=follow_up_ix + follow_up_interval_h),
                instrument=InstrumentType.RGB,
                request_name=f"{volcano.name}_volume_followup_{follow_up_ix}h",
                min_elevation_deg=20.0
            )

            vol_task = ConstrainedObservationRequest(
                name=f"{volcano.name}_volume_followup_{follow_up_ix}h",
                observation_request=vol_req,
                is_mandatory=False,
                schedule_policy_if_constraint_unsatisfied=policy_schedule,
                dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
                task_constraints=obs_task_constraints,
                timeline_constraints=[
                    TaskTimelineConstraint(
                        timeline=timeline, time=TaskImpactTime.PRE,
                        type=TimelineConstraintType.GREATER_OR_EQUAL, value=0.1
                    )
                ],
                timeline_impacts=[],
                rewarder=rewarder_observation,
                success_declarer=success_declarer_eruption,
                request_group=volcano.name,
                max_num_instances=max_num_instances
            )

            # Plume Follow-up Task (Initial target offset downwind to mid-window)
            mid_window_h = follow_up_ix + (follow_up_interval_h / 2.0)
            initial_plume_lon, initial_plume_lat = calculate_plume_position(
                volcano.lon_deg, volcano.lat_deg, mid_window_h, wind_speed_kph, wind_heading_deg
            )

            plume_req = ObservationRequest(
                lon_deg=initial_plume_lon, lat_deg=initial_plume_lat, alt_km=plume_alt_km,
                min_time=min_time + dt.timedelta(hours=follow_up_ix),
                max_time=min_time + dt.timedelta(hours=follow_up_ix + follow_up_interval_h),
                instrument=InstrumentType.RGB,
                request_name=f"{volcano.name}_plume_followup_{follow_up_ix}h",
                min_elevation_deg=20.0
            )

            plume_task = ConstrainedObservationRequest(
                name=f"{volcano.name}_plume_followup_{follow_up_ix}h",
                observation_request=plume_req,
                is_mandatory=False,
                schedule_policy_if_constraint_unsatisfied=policy_schedule,
                dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
                task_constraints=obs_task_constraints,
                timeline_constraints=[
                    TaskTimelineConstraint(
                        timeline=timeline, time=TaskImpactTime.PRE,
                        type=TimelineConstraintType.GREATER_OR_EQUAL, value=0.1
                    )
                ],
                timeline_impacts=[],
                rewarder=rewarder_observation,
                success_declarer=success_declarer_plume,
                request_group=f"{volcano.name}_plume",
                max_num_instances=max_num_instances
            )

            constrained_requests.append(vol_task)
            constrained_requests.append(plume_task)

    # Dynamic workflow request updater callback closure
    def request_updater(curr_time, reqs, tls):
        return request_updater_volcanoes(
            current_time=curr_time,
            requests=reqs,
            timelines=tls,
            volcano_map=volcano_map,
            wind_speed_kph=wind_speed_kph,
            wind_heading_deg=wind_heading_deg,
            plume_alt_km=plume_alt_km
        )

    return Workflow(
        constrained_observation_requests=constrained_requests,
        timelines=list(timelines.values()),
        timeline_updater=timeline_updater_volcanoes,
        request_updater=request_updater
    )