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

# Let's do everything broker-side for now.
# When we receive an observation, we update the ship tracker.
# Periodically we query the ship tracker with a next set of search locations, which are passed to the broker.
# For the temporal aspect, we just query periodically.
# For the spatial aspect, let's start with "last known location"

class ShipTracker():
    def __init__(self, initial_pose: Pose, pose_propagator=None):
        self.last_measurement_time = initial_pose.time
        self.last_update_time = initial_pose.time

        self.estimated_locations = [initial_pose,]
        
        if pose_propagator is None:
            # Stationary propagator
            def pose_propagator(_previous_pose: Pose, deltat: dt.timedelta):
                _new_pose = Pose(
                    time=_previous_pose.time+deltat,
                    lon_deg=_previous_pose.lon_deg,
                    lat_deg=_previous_pose.lat_deg,
                    alt_km=_previous_pose.alt_km,
                    heading_deg=_previous_pose.heading_deg,
                    speed_kph=_previous_pose.speed_kph
                    )
                return _new_pose
        
        self.propagator = pose_propagator

    def update_location(self, new_detection: Pose):
        self.estimated_locations = [new_detection,]

        self.last_update_time = new_detection.time
        self.last_measurement_time = new_detection.time

    def propagate_locations(self, new_time: dt.datetime, samples: int=10, max_dt: dt.timedelta= dt.timedelta(minutes=60)):
        new_estimated_locations = []
        for _location in self.estimated_locations:
            for _ in range(samples):
                _curr_time = self.last_update_time
                _curr_location = _location
                while _curr_time<new_time:
                    _next_time = min(_curr_time+max_dt, new_time)
                    _curr_location = self.propagator(_curr_location, _next_time-_curr_time)
                    _curr_time = _next_time
                new_estimated_locations.append(_curr_location)
        self.estimated_locations = new_estimated_locations
        self.last_update_time = new_time

    def prune_locations(self, samples: int=100):
        if len(self.estimated_locations)>samples:
            self.estimated_locations = random.sample(self.estimated_locations, samples)

    def plan_next_search_locations(self, current_time, max_search_time, max_search_locations=10, observation_interval: dt.timedelta=dt.timedelta(seconds=10*60)):
        # Idea:
        # - Generate a bunch of points sampling the possible current location of the ship
        # - Come up with a tessellation that covers these points
        # - Return those as search areas
        self.propagate_locations(current_time)
        self.prune_locations()
        all_obs_requests = []
        for i in range(int(np.ceil((max_search_time-current_time).total_seconds()/observation_interval.total_seconds()))):
            for _pose in random.sample(self.estimated_locations, max_search_locations):
                _obs_min_time = current_time+observation_interval*i
                _obs_max_time = current_time+observation_interval*(i+1)
                _obs_center_time = current_time+observation_interval*(i+.5)
                _propagated_pose = self.propagator(_pose, _obs_center_time-_pose.time)
                _request = ObservationRequest(
                    lat_deg=_propagated_pose.lat_deg,
                    lon_deg=_propagated_pose.lon_deg,
                    min_time=_obs_min_time,
                    max_time = _obs_max_time,
                    alt_km=_propagated_pose.alt_km,
                    instrument=InstrumentType.RGB,
                    request_name="Ship_{}-{}".format(_obs_min_time.strftime("%Y-%m-%d %H:%M:%S"), _obs_max_time.strftime("%Y-%m-%d %H:%M:%S")),
                    min_elevation_deg=45,
                )
                all_obs_requests.append(_request)

        return all_obs_requests
    

def ship_propagator(previous_pose: Pose, deltat: dt.timedelta, heading_variance: float = 1, speed_variance: float=1, max_speed_if_unspecified_kph = 37):
    next_point_is_on_land = True
    _it_guesses = 0
    max_iters_per_point = 5000
    while (next_point_is_on_land and _it_guesses<max_iters_per_point):
        
        _it_guesses+=1
        
        # new_heading_rad = random.random()*2*math.pi # 0 at North
        if (previous_pose.heading_deg is not None):
            new_heading_rad = previous_pose.heading_deg*np.pi/180. + random.normalvariate()*heading_variance # 0 at North
        else:
            new_heading_rad = random.uniform(-np.pi, np.pi)

        if (previous_pose.speed_kph is not None):
            new_speed_kph = previous_pose.speed_kph + random.normalvariate()*speed_variance
        else:
            new_speed_kph = random.uniform(0, max_speed_if_unspecified_kph)

        dy = new_speed_kph*np.cos(new_heading_rad)*(deltat.total_seconds()/3600) # E-W
        dx = new_speed_kph*np.sin(new_heading_rad)*(deltat.total_seconds()/3600) # N-S

        dlat_rad = dy/R_earth_km
        dlon_rad = dx/(R_earth_km*np.cos(previous_pose.lat_deg*math.pi/180.))

        new_lat_deg = previous_pose.lat_deg + dlat_rad*180./math.pi
        new_lon_deg = previous_pose.lon_deg + dlon_rad*180./math.pi
        new_heading_deg = new_heading_rad*180./math.pi

        next_point_is_on_land = is_land(new_lon_deg, new_lat_deg)
    if (_it_guesses==max_iters_per_point-1):
        raise ValueError("Could not find a way to stay on land")

    return Pose(
        time=previous_pose.time+deltat,
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
        lat_deg = phenomenon.lat_deg,
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

def propagate_distribution_from_observations(
        query_times: list[dt.datetime],
        initial_pose: Pose,
        observation_requests: list[ConstrainedObservationRequest],
        pose_propagator=None,
        target_name: str="",
        h3_resolution: int=5, # Res 5: 9.85 km side. Res 6: 3.72 km side
        samples_propagation: int=200,
        propagate_negative_samples: bool=True,
        verbose: bool=False,
        plot_filter_distribution: bool=True
        ):

    query_times.sort()
    completed_requests = [r for r in observation_requests if r.completed == True and r.feasible == True]
    # Walk from most recent backward to find the most recent time we saw the ship
    completed_requests.sort(key=lambda x: x.observation_opportunity.time, reverse=True)
    found_latest_request = False
    completed_request_ix = 0
    for completed_request in completed_requests:
        if len(completed_request.data_product):
            for ship_location_phenomenon in completed_request.data_product:
                # If the ship is the right one
                if ((len(target_name)==0) or (target_name in ship_location_phenomenon.name)):
                    if verbose:
                        print(f"Reinitializing the filter at {completed_request.observation_opportunity.time} (phenomenon {ship_location_phenomenon} matches string {target_name})")
                    found_latest_request = True
                    initial_pose = custom_ship_phenomenon_processor(
                        observation=completed_request.observation_opportunity,
                        spacecraft=completed_request.observation_opportunity.satellite,
                        phenomenon=ship_location_phenomenon
                        )
                    break
        if found_latest_request:
            break
        completed_request_ix += 1

    ship_tracker = ShipTracker(initial_pose=initial_pose, pose_propagator=pose_propagator)

    if propagate_negative_samples:
        # # Walk from that point in the past to the present and remove particles incompatible with negative requests
        if completed_request_ix>0: # If the positive observation was not the last request - recall these are reverse-ordered
            for completed_request in reversed(completed_requests[:completed_request_ix-1]):
                # Recall we are going chronologically now, since we reversed
                # This is a completed request AFTER the latest successful one. So it is a negative result.
                _observation = completed_request.observation_opportunity
                if _observation.time > query_times[0]:
                    raise ValueError(f"We have a successful observation in the future, that should never happen. Observation {_observation}, query times {query_times}")
                if len(completed_request.data_product):
                    for ship_location_phenomenon in completed_request.data_product:
                        if ((len(target_name)==0) or (target_name in ship_location_phenomenon.name)):
                            raise ValueError(f"Found successful observation {_observation} after the supposedly last successful observation")
                
                ship_tracker.propagate_locations(_observation.time, samples=samples_propagation, max_dt=dt.timedelta(hours=3))
                _particles_out_of_fov = [p for p in ship_tracker.estimated_locations if (not is_lla_in_satellite_fov(observation=_observation, location=p))]
                ship_tracker.estimated_locations = _particles_out_of_fov
                ship_tracker.prune_locations(samples=samples_propagation)


    # if plot_filter_distribution:
    #     if plot_axis is None:
    #         figglobal = plt.figure(figsize=(10,5))
    #         plot_axis = figglobal.add_subplot(1,1,1, projection=ccrs.Robinson())
    #         plot_axis.coastlines()

    #         fig_distribution, plot_distribution = plt.subplots(figsize=(10,5))
    #         plot_distribution.set_title("Cumulative distribution across hexagons")

    tessellations_by_time = {}

    for urtime in query_times:
        ship_tracker.propagate_locations(urtime, samples=samples_propagation, max_dt=dt.timedelta(hours=3))
        ship_tracker.prune_locations(samples=samples_propagation) # Samples in propagate_location doesn't exactly do what we want...
        
        # TESSELLATE WHAT YOU FOUND AND ADD REQUESTS ACCORDINGLY

        points_gpd = gpd.GeoDataFrame(
            {'lon': [pos.lon_deg for pos in ship_tracker.estimated_locations], 'lat': [pos.lat_deg for pos in ship_tracker.estimated_locations]},
            geometry=gpd.points_from_xy([pos.lon_deg for pos in ship_tracker.estimated_locations], [pos.lat_deg for pos in ship_tracker.estimated_locations]),
            crs="EPSG:4326"
        )
        # Res 5: 9.85 km side. Res 6: 3.72 km side

        # if len(ship_tracker.estimated_locations) == 1:
        h3_cell_ids = list(set([
            h3.latlng_to_cell(est_loc.lat_deg, est_loc.lon_deg, h3_resolution)
            for est_loc in ship_tracker.estimated_locations
            ]))

        points_hex_tessellation = gpd.GeoDataFrame(
            {'h3_id': h3_cell_ids},
            geometry = [h3_to_polygon(cell_id) for cell_id in h3_cell_ids],
            crs="EPSG:4326",
        )

        # Counting points in each hex
        joined_hexagons_and_points = gpd.sjoin(points_hex_tessellation, points_gpd, how="left", predicate="intersects")
        points_hex_tessellation['point_count'] = joined_hexagons_and_points.groupby(joined_hexagons_and_points.index)['index_right'].count()

        points_hex_tessellation.sort_values(by='point_count', ascending=False, inplace=True)

        tessellations_by_time[urtime] = points_hex_tessellation

        # if plot_filter_distribution:
        #     total_points = len(ship_tracker.estimated_locations)
        #     cumulative_points = 0
        #     point_distribution = []
        #     for index, row in points_hex_tessellation.iterrows():
        #         points = row['point_count']
        #         cumulative_points+=points
        #         point_distribution.append(cumulative_points/total_points)
        #     # print("Done, plotting")
        #     points_gpd.plot(ax=plot_axis, transform=ccrs.Geodetic(), markersize=3, label=urtime,alpha=.1)
        #     points_hex_tessellation.plot(
        #         ax=plot_axis,
        #         transform=ccrs.PlateCarree(), 
        #         column='point_count',
        #         cmap='Blues',
        #         alpha=.9,
        #         # vmax=points_hex_tessellation['point_count'].max() + 1
        #         )
            
        #     plot_distribution.plot(list(range(len(point_distribution))), point_distribution, label=urtime)

    return tessellations_by_time