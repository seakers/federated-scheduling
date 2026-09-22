"""
MSA ship-tracking utilities: particle-filter tracker, propagator, H3 tessellation.
"""

from shapely import Polygon

from fame import *
from fame_geometry import ObservationOpportunity, Pose
import random
import datetime as dt
import math
from fame_workflow import ObservationRequest, InstrumentType, ConstrainedObservationRequest
from fame_agents import Satellite, Phenomenon
from fame_geometry import is_land, is_lla_in_satellite_fov
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

R_earth_km = 6371

# Default propagation step. MUST match the step size of the process that
# generates the ground truth, or the filter's cloud has the wrong shape.
DEFAULT_FILTER_STEP = dt.timedelta(minutes=30)


class ShipTracker():
    def __init__(self, initial_pose: Pose, pose_propagator=None):
        self.last_measurement_time = initial_pose.time
        self.last_update_time = initial_pose.time

        self.estimated_locations = [initial_pose, ]

        if pose_propagator is None:
            # Stationary propagator
            def pose_propagator(_previous_pose: Pose, deltat: dt.timedelta):
                _new_pose = Pose(
                    time=_previous_pose.time + deltat,
                    lon_deg=_previous_pose.lon_deg,
                    lat_deg=_previous_pose.lat_deg,
                    alt_km=_previous_pose.alt_km,
                    heading_deg=_previous_pose.heading_deg,
                    speed_kph=_previous_pose.speed_kph
                )
                return _new_pose

        self.propagator = pose_propagator

    def update_location(self, new_detection: Pose):
        self.estimated_locations = [new_detection, ]

        self.last_update_time = new_detection.time
        self.last_measurement_time = new_detection.time

    def propagate_locations(self, new_time: dt.datetime, samples: int = 100,
                            max_dt: dt.timedelta = DEFAULT_FILTER_STEP):
        """Advance every particle to `new_time`, keeping the population at ~`samples`.

        [F3] Each existing particle is fanned out only as far as needed to reach
        the target population:

            branches = ceil(samples / len(cloud))

        A cloud that already holds `samples` particles therefore propagates 1:1
        instead of exploding to len(cloud) * samples and being pruned straight
        back down.  A 1-particle cloud -- which is what you get immediately
        after re-initialising on a fix -- still fans out to the full target.

        [F2] `max_dt` is the integration step. Each branch takes ceil(dt/max_dt)
        perturbed segments, so the cloud follows a random WALK, matching the
        ground-truth generator, rather than a single straight ray.
        """
        if not self.estimated_locations:
            return

        # [F5] Never move the clock backwards; a query time in the past would
        # otherwise leave the cloud untouched but stamped with the wrong time.
        if new_time <= self.last_update_time:
            self.last_update_time = max(self.last_update_time, new_time)
            return

        n_existing = len(self.estimated_locations)
        branches = max(1, int(math.ceil(samples / float(n_existing))))

        new_estimated_locations = []
        for _location in self.estimated_locations:
            for _ in range(branches):
                _curr_time = self.last_update_time
                _curr_location = _location
                while _curr_time < new_time:
                    _next_time = min(_curr_time + max_dt, new_time)
                    _curr_location = self.propagator(_curr_location, _next_time - _curr_time)
                    _curr_time = _next_time
                new_estimated_locations.append(_curr_location)

        self.estimated_locations = new_estimated_locations
        self.last_update_time = new_time

    def prune_locations(self, samples: int = 100):
        if len(self.estimated_locations) > samples:
            self.estimated_locations = random.sample(self.estimated_locations, samples)

    def plan_next_search_locations(self, current_time, max_search_time, max_search_locations=10,
                                   observation_interval: dt.timedelta = dt.timedelta(seconds=10 * 60),
                                   filter_step: dt.timedelta = DEFAULT_FILTER_STEP):
        # Idea:
        # - Generate a bunch of points sampling the possible current location of the ship
        # - Come up with a tessellation that covers these points
        # - Return those as search areas
        self.propagate_locations(current_time, max_dt=filter_step)
        self.prune_locations()
        all_obs_requests = []
        n_intervals = int(np.ceil(
            (max_search_time - current_time).total_seconds() / observation_interval.total_seconds()))
        for i in range(n_intervals):
            n_pick = min(max_search_locations, len(self.estimated_locations))
            for _pose in random.sample(self.estimated_locations, n_pick):
                _obs_min_time = current_time + observation_interval * i
                _obs_max_time = current_time + observation_interval * (i + 1)
                _obs_center_time = current_time + observation_interval * (i + .5)
                _propagated_pose = self.propagator(_pose, _obs_center_time - _pose.time)
                _request = ObservationRequest(
                    lat_deg=_propagated_pose.lat_deg,
                    lon_deg=_propagated_pose.lon_deg,
                    min_time=_obs_min_time,
                    max_time=_obs_max_time,
                    alt_km=_propagated_pose.alt_km,
                    instrument=InstrumentType.RGB,
                    request_name="Ship_{}-{}".format(
                        _obs_min_time.strftime("%Y-%m-%d %H:%M:%S"),
                        _obs_max_time.strftime("%Y-%m-%d %H:%M:%S")),
                    min_elevation_deg=45,
                )
                all_obs_requests.append(_request)

        return all_obs_requests


def ship_propagator(previous_pose: Pose, deltat: dt.timedelta,
                    heading_variance: float = 1, speed_variance: float = 1,
                    max_speed_if_unspecified_kph=37,
                    max_iters_per_point: int = 5000):
    """One perturbed step of the ship's random walk, rejecting land positions.

    [F1] On exhaustion this HOLDS POSITION rather than returning the last
    (land) candidate.  The previous version's guard tested
    `_it_guesses == max_iters_per_point - 1`, which the loop can never leave it
    at -- the loop exits with `_it_guesses == max_iters_per_point` -- so the
    function fell through and emitted a land particle.  Holding position
    matches what the ground-truth generator does in the same situation, which
    keeps the filter's process model and the truth consistent.
    """
    next_point_is_on_land = True
    _it_guesses = 0

    new_lat_deg = previous_pose.lat_deg
    new_lon_deg = previous_pose.lon_deg
    new_speed_kph = previous_pose.speed_kph if previous_pose.speed_kph is not None else 0.0
    new_heading_deg = previous_pose.heading_deg if previous_pose.heading_deg is not None else 0.0

    while (next_point_is_on_land and _it_guesses < max_iters_per_point):

        _it_guesses += 1

        if (previous_pose.heading_deg is not None):
            new_heading_rad = previous_pose.heading_deg * np.pi / 180. + random.normalvariate() * heading_variance
        else:
            new_heading_rad = random.uniform(-np.pi, np.pi)

        if (previous_pose.speed_kph is not None):
            new_speed_kph = previous_pose.speed_kph + random.normalvariate() * speed_variance
        else:
            new_speed_kph = random.uniform(0, max_speed_if_unspecified_kph)

        dy = new_speed_kph * np.cos(new_heading_rad) * (deltat.total_seconds() / 3600)  # N-S
        dx = new_speed_kph * np.sin(new_heading_rad) * (deltat.total_seconds() / 3600)  # E-W

        dlat_rad = dy / R_earth_km
        dlon_rad = dx / (R_earth_km * np.cos(previous_pose.lat_deg * math.pi / 180.))

        new_lat_deg = previous_pose.lat_deg + dlat_rad * 180. / math.pi
        new_lon_deg = previous_pose.lon_deg + dlon_rad * 180. / math.pi
        new_heading_deg = new_heading_rad * 180. / math.pi

        next_point_is_on_land = is_land(new_lon_deg, new_lat_deg)

    if next_point_is_on_land:
        # [F1] No sea-going heading found. Hold position (speed 0) instead of
        # emitting a land particle.
        return Pose(
            time=previous_pose.time + deltat,
            lon_deg=previous_pose.lon_deg,
            lat_deg=previous_pose.lat_deg,
            alt_km=previous_pose.alt_km,
            speed_kph=0.0,
            heading_deg=previous_pose.heading_deg,
        )

    return Pose(
        time=previous_pose.time + deltat,
        lon_deg=new_lon_deg,
        lat_deg=new_lat_deg,
        alt_km=previous_pose.alt_km,
        speed_kph=new_speed_kph,
        heading_deg=new_heading_deg
    )


def custom_ship_phenomenon_processor(
        observation: ObservationOpportunity,
        spacecraft: Satellite,
        phenomenon: Phenomenon
        ):
    return Pose(
        lon_deg=phenomenon.lon_deg,
        lat_deg=phenomenon.lat_deg,
        time=observation.time,
        alt_km=phenomenon.alt_km,
        heading_deg=phenomenon.heading_deg,
        speed_kph=phenomenon.speed_kph,
        name=phenomenon.name
    )


def h3_to_polygon(h3_id):
    # Get the cell boundary in (lat, lng) format
    boundary_lat_lng = h3.cell_to_boundary(h3_id)

    # Swap to (lng, lat) because Shapely/GeoPandas expect (x, y)
    boundary_lng_lat = [(lng, lat) for lat, lng in boundary_lat_lng]

    # Return as a Shapely Polygon
    return Polygon(boundary_lng_lat)


def _data_product(request):
    """data_product as a list, tolerating None. [F5]"""
    dp = getattr(request, 'data_product', None)
    return dp if dp else []


def _observation_time(request):
    """Realised observation time of a completed request, or None.

    [F4] Prefers `observation_opportunity.time`, which the legacy dispatcher
    sets, and falls back to the data product's own timestamps, which are always
    present on a successful observation regardless of which dispatch path ran.
    Without the fallback, re-initialisation silently never happens on the
    redundant path and the filter propagates from t=0 forever.
    """
    oo = getattr(request, 'observation_opportunity', None)
    if oo is not None and getattr(oo, 'time', None) is not None:
        return oo.time
    times = []
    for p in _data_product(request):
        t = getattr(p, 'time', None) or getattr(p, 'start_time', None)
        if t is not None:
            times.append(t)
    return max(times) if times else None


def _pose_from_detection(phenomenon, obs_time, fallback_alt_km=0.0):
    """Build a Pose from a detected phenomenon without needing the opportunity."""
    return Pose(
        lon_deg=phenomenon.lon_deg,
        lat_deg=phenomenon.lat_deg,
        time=obs_time,
        alt_km=getattr(phenomenon, 'alt_km', fallback_alt_km),
        heading_deg=getattr(phenomenon, 'heading_deg', None),
        speed_kph=getattr(phenomenon, 'speed_kph', None),
        name=getattr(phenomenon, 'name', ''),
    )


def propagate_distribution_from_observations(
        query_times: list[dt.datetime],
        initial_pose: Pose,
        observation_requests: list[ConstrainedObservationRequest],
        pose_propagator=None,
        target_name: str = "",
        h3_resolution: int = 5,          # Res 5: 9.85 km side. Res 6: 3.72 km side
        samples_propagation: int = 200,
        propagate_negative_samples: bool = True,
        verbose: bool = False,
        plot_filter_distribution: bool = False,
        filter_step: dt.timedelta = DEFAULT_FILTER_STEP,   # [F2]
        ):
    """Propagate the ship belief to each query time and tessellate it into H3 cells.

    Returns {query_time: GeoDataFrame} sorted by descending `point_count`, so
    `.iloc[0]` is the modal cell and `point_count / point_count.sum()` is that
    cell's probability mass.

    `filter_step` is the integration step of the process model and MUST match
    the step used to generate the ground truth (see F2 in the module docstring).
    """
    query_times = sorted(query_times)
    if not query_times:
        return {}

    # ---- Re-initialise on the most recent successful detection ---------------
    completed_requests = [
        r for r in observation_requests
        if getattr(r, 'completed', False) and _observation_time(r) is not None
    ]
    completed_requests.sort(key=_observation_time, reverse=True)

    found_latest_request = False
    completed_request_ix = 0
    for completed_request in completed_requests:
        _obs_time = _observation_time(completed_request)
        for ship_location_phenomenon in _data_product(completed_request):
            if ((len(target_name) == 0)
                    or (target_name in (getattr(ship_location_phenomenon, 'name', '') or ''))):
                if verbose:
                    print(f"  [Filter] Re-initialising at {_obs_time} "
                          f"({ship_location_phenomenon.lat_deg:.4f},"
                          f"{ship_location_phenomenon.lon_deg:.4f})")
                found_latest_request = True
                _oo = getattr(completed_request, 'observation_opportunity', None)
                if _oo is not None:
                    initial_pose = custom_ship_phenomenon_processor(
                        observation=_oo,
                        spacecraft=getattr(_oo, 'satellite', None),
                        phenomenon=ship_location_phenomenon,
                    )
                else:
                    # [F4] Redundant dispatch path: no opportunity object, but the
                    # detection itself carries everything the filter needs.
                    initial_pose = _pose_from_detection(
                        ship_location_phenomenon, _obs_time,
                        fallback_alt_km=getattr(initial_pose, 'alt_km', 0.0))
                break
        if found_latest_request:
            break
        completed_request_ix += 1

    if verbose and not found_latest_request:
        print(f"  [Filter] No detection found; anchoring on supplied initial pose "
              f"at {initial_pose.time}")

    ship_tracker = ShipTracker(initial_pose=initial_pose, pose_propagator=pose_propagator)

    # ---- Negative information: particles seen-and-not-found are impossible ---
    if propagate_negative_samples and completed_request_ix > 0:
        for completed_request in reversed(completed_requests[:completed_request_ix]):
            _observation = getattr(completed_request, 'observation_opportunity', None)
            if _observation is None:
                # Cannot run a FOV test without the realised opportunity geometry.
                continue
            if _observation.time > query_times[0]:
                continue
            if _observation.time <= ship_tracker.last_update_time:
                continue
            for ship_location_phenomenon in _data_product(completed_request):
                if ((len(target_name) == 0)
                        or (target_name in (getattr(ship_location_phenomenon, 'name', '') or ''))):
                    raise ValueError(
                        "Found a successful observation after the supposedly latest one; "
                        "the completed-request ordering is inconsistent.")

            ship_tracker.propagate_locations(_observation.time,
                                             samples=samples_propagation,
                                             max_dt=filter_step)
            _particles_out_of_fov = [
                p for p in ship_tracker.estimated_locations
                if (not is_lla_in_satellite_fov(observation=_observation, location=p))
            ]
            # [F5] A negative observation that eliminates every particle means the
            # belief and the truth have diverged. Keep the prior rather than
            # collapsing to an empty cloud, which would crash the tessellation.
            if not _particles_out_of_fov:
                if verbose:
                    print(f"  [Filter] Negative update at {_observation.time} would empty "
                          f"the cloud; keeping prior")
            else:
                ship_tracker.estimated_locations = _particles_out_of_fov
            ship_tracker.prune_locations(samples=samples_propagation)

    # ---- Propagate to each query time and tessellate -------------------------
    tessellations_by_time = {}

    for urtime in query_times:
        ship_tracker.propagate_locations(urtime, samples=samples_propagation,
                                         max_dt=filter_step)
        ship_tracker.prune_locations(samples=samples_propagation)

        if not ship_tracker.estimated_locations:
            tessellations_by_time[urtime] = gpd.GeoDataFrame(
                {'h3_id': [], 'point_count': []}, geometry=[], crs="EPSG:4326")
            continue

        lons = [pos.lon_deg for pos in ship_tracker.estimated_locations]
        lats = [pos.lat_deg for pos in ship_tracker.estimated_locations]

        points_gpd = gpd.GeoDataFrame(
            {'lon': lons, 'lat': lats},
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326"
        )

        h3_cell_ids = list(set([
            h3.latlng_to_cell(est_loc.lat_deg, est_loc.lon_deg, h3_resolution)
            for est_loc in ship_tracker.estimated_locations
        ]))

        points_hex_tessellation = gpd.GeoDataFrame(
            {'h3_id': h3_cell_ids},
            geometry=[h3_to_polygon(cell_id) for cell_id in h3_cell_ids],
            crs="EPSG:4326",
        )

        joined = gpd.sjoin(points_hex_tessellation, points_gpd, how="left", predicate="intersects")
        # [F5] fillna: a hex with no joined point yields NaN, which then poisons
        # the mass normalisation downstream.
        points_hex_tessellation['point_count'] = (
            joined.groupby(joined.index)['index_right'].count().fillna(0).astype(int)
        )

        points_hex_tessellation.sort_values(by='point_count', ascending=False, inplace=True)
        points_hex_tessellation.reset_index(drop=True, inplace=True)

        tessellations_by_time[urtime] = points_hex_tessellation

        if verbose:
            _total = float(points_hex_tessellation['point_count'].sum())
            if _total > 0:
                _top1 = float(points_hex_tessellation.iloc[0]['point_count']) / _total
                _age_h = (urtime - ship_tracker.last_measurement_time).total_seconds() / 3600.0
                print(f"  [Filter] {urtime}: {len(points_hex_tessellation)} cells, "
                      f"top-1 mass {_top1:.3f}, age {_age_h:.1f}h")

    return tessellations_by_time