from fame import *
from fame_geometry import Pose
import random
import datetime as dt

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