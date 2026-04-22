import pyorbital
from pyorbital.orbital import Orbital
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import math
from scipy.spatial.transform import Rotation
# from scipy.optimize import bisect as scipy_bisect
from scipy.optimize import newton as scipy_newton
# import pandas as pd
# import bisect
# import uuid
from enum import Enum
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import networkx as nx
import bisect

from fame_geometry import *

import requests
import urllib
import json



# API Endpoints
LOGIN_URL = 'https://www.space-track.org/ajaxauth/login'
# This query gets the "latest" GP TLE for all objects currently in orbit
# QUERY_URL = 'https://www.space-track.org/basicspacedata/query/class/gp/format/tle'
QUERY_URL = 'https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-10/format/json'

def get_gp_tles(tle_file_name_json, space_track_user, space_track_password):
    # Create a session to persist cookies (authentication)
    session = requests.Session()
    
    # Payload for login
    payload = {
        'identity': space_track_user,
        'password': space_track_password
    }
    
    try:
        # 1. Authenticate
        print("Logging in...")
        response = session.post(LOGIN_URL, data=payload)
        
        if response.status_code != 200:
            print(f"Login failed. Status code: {response.status_code}")
            return

        # 2. Query the data
        print("Fetching GP TLEs (this may take a few moments)...")
        # You can add parameters like /orderby/NORAD_CAT_ID/limit/100 to test
        result = session.get(QUERY_URL)
        
        if result.status_code == 200:
            # result.text contains the TLE data in string format
            tles = result.text
            print("Successfully retrieved data.")
            
            # Save to file
            with open(tle_file_name_json, "w") as f:
                f.write(tles)
            print("TLEs saved to {}".format(tle_file_name_json))
        else:
            print(f"Query failed. Status: {result.status_code}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        session.close()

def process_tles_from_json(tles_path_json, output_filepath="all_tles.txt"):
    with open(tles_path_json, 'r') as json_in_file:
        tles_json = json.load(json_in_file)
        with open(output_filepath, 'w') as out_file:
            for entry in tles_json:
                print(entry['OBJECT_NAME'],file=out_file)
                print(entry['TLE_LINE1'],file=out_file)
                print(entry['TLE_LINE2'],file=out_file)



def get_elevation_usgs(lon_deg, lat_deg):
    url = r'https://epqs.nationalmap.gov/v1/json?'
    params = {
        'output': 'json',
        'x': lon_deg,
        'y': lat_deg,
        'units': 'Meters'
    }
    result = requests.get((url + urllib.parse.urlencode(params)))
    return float(result.json()['value'])


def get_elevation_from_open_elevation(longitude_deg, latitude_deg):
    """
    Retrieves elevation data from the Open-Elevation API for a given
    latitude and longitude.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.

    Returns:
        float or None: The elevation in meters if successful, otherwise None.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {
        "locations": f"{latitude_deg},{longitude_deg}"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        if data and "results" in data and len(data["results"]) > 0:
            return data["results"][0]["elevation"]
        else:
            print("No elevation data found in the response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Open-Elevation API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None

# # Example usage:
# lat = 40.7128
# lon = -74.0060

# elevation = get_elevation_from_open_elevation(lat, lon)

# if elevation is not None:
#     print(f"The elevation at ({lat}, {lon}) is: {elevation} meters")
# else:
#     print(f"Could not retrieve elevation for ({lat}, {lon})")

class Location():
    def __init__(self, lon_deg: float, lat_deg: float, alt_km: float=None, name: str=""):
        self.lon_deg = lon_deg
        self.lat_deg = lat_deg
        if alt_km is None:
            alt_km = 1e-3*get_elevation_from_open_elevation(lon_deg, lat_deg)
            print("Queried OpenElevation for elevation at {}, {}: {} km".format(lon_deg, lat_deg, alt_km))
        self.alt_km = alt_km
        self.name = name
    def __str__(self):
        return "Location {} | Lon {}°, lat {}°, alt {} km".format(
            self.name,
            self.lon_deg,
            self.lat_deg,
            self.alt_km,
        )
    def __repr__(self):
        return self.__str__()
    
class Pose(Location):
    def __init__(self, lon_deg: float, lat_deg: float, time: dt.datetime=dt.datetime.fromtimestamp(0), alt_km: float=None, heading_deg: float=None, speed_kph: float=None, name: str=""):
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=alt_km, name=name)
        self.time = time
        self.heading_deg = heading_deg
        self.speed_kph = speed_kph
    def __str__(self):
        return "Pose {} | Lon {}°, lat {}°, alt {} km, heading {}°, speed {} km/h at {}".format(
            self.name,
            self.lon_deg,
            self.lat_deg,
            self.alt_km,
            self.heading_deg,
            self.speed_kph,
            self.time
        )
    def __repr__(self):
        return self.__str__()
    
# Let's search for opportunities. 
# Input: a number of satellites, each with instruments. A number of ground locations we would like to image, with time windows.
# Output: a map from locations to satellite passes.

class InstrumentType(Enum):
    RGB = 0
    SAR = 1
    HYPERSPECTRAL = 2

class ObservationRequest(Location):
    '''
    An observation request asks to observe a given location with a given instrument between a minimum and a maximum time.
    '''
    def __init__(self, lon_deg, lat_deg, min_time, max_time, alt_km=None, instrument=InstrumentType.RGB, request_name="", min_elevation_deg=0):
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=alt_km, name=request_name)
        # self.name = request_name
        # self.lon_deg = lon_deg
        # self.lat_deg = lat_deg
        # self.alt_km = alt_km
        self.min_time = min_time
        self.max_time = max_time
        self.instrument = instrument
        self.min_elevation_deg = min_elevation_deg
    def __str__(self):
        return "Request {}".format(
            self.name,
        )
    def __repr__(self):
        return self.__str__()
        # return "Request {} | Lon {}°, lat {}°, alt {} km from {} to {} with {}, min elevation {} deg".format(
        #     self.name,
        #     self.lon_deg,
        #     self.lat_deg,
        #     self.alt_km,
        #     self.min_time,
        #     self.max_time,
        #     self.instrument,
        #     self.min_elevation_deg
        # )

class ObservationOpportunity(Location):
    '''
    An observation opportunity specifies a time instant when an observation request can be fulfilled 
    '''
    def __init__(self, time, lon_deg, lat_deg, alt_km, look_angle_az_deg, look_angle_dec_deg, sun_zenith_angle_deg, range_km, name="", instrument=InstrumentType.RGB, duration: dt.timedelta=dt.timedelta(seconds=60)):
        self.time = time
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=alt_km, name=name)
        self.instrument = instrument
        self.look_angle_az_deg = look_angle_az_deg
        self.look_angle_dec_deg = look_angle_dec_deg
        self.sun_zenith_angle_deg = sun_zenith_angle_deg
        self.range_km = range_km
        self.duration = duration
    def __str__(self):
        return "Observation {} at {} with {}. Look angle {} | {} az/dec deg, zenith angle {} deg, range {} km, duration {}".format(
            self.name,
            self.time,
            self.instrument,
            self.look_angle_az_deg,
            self.look_angle_dec_deg,
            self.sun_zenith_angle_deg,
            self.range_km,
            self.duration
        )
    def __repr__(self):
        return self.__str__()
        
class ObservationPass:
    ''' An observation pass is a time interval during which a satellite is over the horizon to a given location.
    We model it as observation opportunities for rise time, fall time, and highest time  
    '''
    def __init__(self, rise: ObservationOpportunity, fall: ObservationOpportunity, highest: ObservationOpportunity):
        self.rise = rise
        self.fall = fall
        self.highest = highest
    def __str__(self):
        return "Pass start: {}, highest: {}, fall: {}".format(self.rise.time, self.highest.time, self.fall.time)
    def __repr__(self):
        return self.__str__()
    
class AttitudeController(Enum):
    FREE = 0
    INSTRUMENT = 1
    COMMUNICATION = 2

class Satellite():
    def __init__(self, name: str,  orbit: Orbital, instruments: list = [InstrumentType.RGB], instrument_fov_rad: dict = {InstrumentType.RGB: 15.*np.pi/180.}, isl_links: dict={}, has_continuous_isl_to_ground: bool=False):
        self.name = name
        self.orbit = orbit
        self.instruments = instruments
        self.instrument_fov_rad = instrument_fov_rad
        self.scheduled_observations = []
        self.data_products = {}
        self.known_phenomena = []
        self.attitude_controller_state = AttitudeController.FREE
        self.busy_with = None
        self.isl_links = isl_links # Satellite: range_km
        self.has_continuous_isl_to_ground = has_continuous_isl_to_ground
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()
    
def observation_quality(opportunity: ObservationOpportunity, preferred_zenith_angle_deg=45):
    '''
    A quality function that specifies how good an opportunity is
    '''
    # TODO if opportunity.instrument == InstrumentType.RGB add an entry for local time.
    # First term: static (to make the ILP >0)
    # Second term: look angle. Nadir-ground point-satellite. 90 is "satellite is overhead"
    # Third term: zenith angle. Nadir-ground-sun. An indication of local time.
    # Fourth term: distance.

    return 10+abs(90.-opportunity.look_angle_dec_deg)/90. + abs(preferred_zenith_angle_deg-opportunity.sun_zenith_angle_deg)/90 - opportunity.range_km/1000

def find_observation_opportunities(observation_requests: list, satellites: list, passes_error_s=60):
    """ A function that finds observation opportunities for a given observation request

    Args:
        observation_requests (list[ObservationRequest]): A list of ObservationRequests
        satellites (list[Spacecraft]): a list of spacecraft to search for
        passes_error_s (int, optional): the discretization of the satellite orbit, used when searching for overflights. Defaults to 60.

    Returns:
        dict[Request][Spacecraft]: list[ObservationPass]: A list of passes (an object with a rise time, fall time, and highest time, each with geometric properties) for a given opportunity and spacecraft
    """
    opportunities = {}
    
    for request in observation_requests:
        opportunities[request] = {}
        for satellite in satellites:
            if request.instrument in satellite.instruments:
                orbit = satellite.orbit
                opportunities[request][satellite] = []
                # Pyorbital has some graceless error handling if it can't find a single pass. Let's wrap the thing in a try-except loop.
                try:
                    _passes = orbit.get_next_passes(
                        request.min_time,
                        int(math.ceil((request.max_time-request.min_time).total_seconds()/3600)),
                        request.lon_deg,
                        request.lat_deg,
                        request.alt_km,
                        passes_error_s,
                        request.min_elevation_deg
                    )
                except ValueError as e:
                    if (str(e) == "f(a) and f(b) must have different signs"):
                        opportunities[request].pop(satellite)
                        _passes = []
                        continue
                    else:
                        raise(e)
                # Avoid returning a satellite if there are no passes for it
                if (len(_passes) == 0):
                    opportunities[request].pop(satellite)
                for _pass in _passes:
                    _pass_opportunities_ = []
                    for _time in _pass: # Pass has start time, end time, max alt time
                        if (_time<request.min_time or _time>request.max_time):
                            break
                            # raise ValueError("Observation opportunity {} does not fit in request {}".format(_pass,request))
                        _look_angle_az_el = orbit.get_observer_look(_time, request.lon_deg, request.lat_deg, request.alt_km)
                        _sun_zenith_angle =  pyorbital.astronomy.sun_zenith_angle(_time, request.lon_deg, request.lat_deg)
                        _range = np.linalg.norm(np.array(orbit.get_position(_time, normalize=False)[:3])-np.array(pyorbital.astronomy.observer_position(_time, request.lon_deg, request.lat_deg, request.alt_km)[:3]))
                        # print(_range)
                        _opp = ObservationOpportunity(
                            time=_time,
                            lon_deg = request.lon_deg,
                            lat_deg = request.lat_deg,
                            alt_km = request.alt_km,
                            look_angle_az_deg=_look_angle_az_el[0],
                            look_angle_dec_deg=_look_angle_az_el[1],
                            sun_zenith_angle_deg=_sun_zenith_angle,
                            range_km=_range,
                            instrument=request.instrument,
                        )
                        _pass_opportunities_.append(_opp)
                    if (len(_pass_opportunities_)==3):
                        opportunities[request][satellite].append(ObservationPass(
                            rise=_pass_opportunities_[0],
                            fall=_pass_opportunities_[1],
                            highest=_pass_opportunities_[2],
                        ))
            if (satellite in opportunities[request].keys() and len(opportunities[request][satellite])==0):
                opportunities[request].pop(satellite)
                    
    return opportunities

def find_contact_opportunities(ground_stations: list, satellites: list, min_time: dt.datetime, max_time: dt.datetime, passes_error_s=60, passes_horizon_deg=15):
    
    ground_station_opportunities = [
        ObservationRequest(
            lon_deg=gs.lon_deg,
            lat_deg=gs.lat_deg,
            alt_km=gs.alt_km,
            min_time=min_time,
            max_time=max_time,
            request_name=gs.name,
            min_elevation_deg=passes_horizon_deg
        )
        for gs in ground_stations
    ]

    gs_opportunity_decoder = {
        gs.name: gs for gs in ground_stations
    }

    comm_opportunities_by_ground_station = find_observation_opportunities(
        observation_requests=ground_station_opportunities,
        satellites=satellites,
        passes_error_s=passes_error_s,
    )

    # Now this is a list of opportunity (GS+time)->satellite->pass.
    # What we want is satellite->station->pass, or satellite->(station, pass).
    comm_opportunities_by_satellite_and_station = {}
    comm_opportunities_by_satellite = {}
    
    for gs_req, satpasses in comm_opportunities_by_ground_station.items():
        for sat, satpass in satpasses.items():
            
            if sat not in comm_opportunities_by_satellite_and_station.keys():
                comm_opportunities_by_satellite_and_station[sat] = {}
            if sat not in comm_opportunities_by_satellite.keys():
                comm_opportunities_by_satellite[sat] = []
            gs = gs_opportunity_decoder[gs_req.name]
            
            if gs not in comm_opportunities_by_satellite_and_station[sat]:
                comm_opportunities_by_satellite_and_station[sat][gs] = []    
            # Satpass is a LIST of opportunities for this given ground station and satellite.
            comm_opportunities_by_satellite_and_station[sat][gs].extend(satpass)
            for _pass in satpass:
                comm_opportunities_by_satellite[sat].append((gs, _pass))
            
    for sat, passes in comm_opportunities_by_satellite.items():
        passes.sort(key=lambda x: x[1].highest.time)
        
    for sat, stationpasses in comm_opportunities_by_satellite_and_station.items():
        for station, passes in stationpasses.items():
            passes.sort(key=lambda x: x.highest.time)
    
    return comm_opportunities_by_satellite_and_station, comm_opportunities_by_satellite

# Check if a point is on land

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
_land = prep(land_geom)

def is_land(x, y):
    return _land.contains(sgeom.Point(x, y))

def spacecraft_fov(time: dt.datetime, satellite: Satellite, instrument: InstrumentType, ground_lla: Location, num_samples: int = 12, USE_SPHERICAL_APPROXIMATION=False):
    '''
    Input: a Satellite, a list of the instrument/instruments to show, and a LLA that the satellite is pointing to.
    Output: a Polygon showing the extent of the satellite FOV.
    '''
    # Compute the satellite location in space.
    # Compute the satellite-to-ground (s2g) vector.
    # Rotate that vector by half the FOV along an axis perpendicular to the s2g vector.
    # Rotate _that_ vector around the s2g vector n times.
    # For each, find the intersection with the ground, defined as: e2s+s2g vector modulo is Earth radius.
    # Find the LLA in the J2K frame.
    # Rotate longitude by time.

    # Compute the satellite location in space.
    satellite_position_inertial, _ = satellite.orbit.get_position(time, normalize=False)
    # print("e2s: {}".format(satellite_position_inertial))
    # satellite_position_inertial = satellite_position_velocity[0]
    # Compute the satellite-to-ground (s2g) vector.
    ground_position_inertial, _ = pyorbital.astronomy.observer_position(time, ground_lla.lon_deg, ground_lla.lat_deg, ground_lla.alt_km)
    # print("e2g: {}".format(ground_position_inertial))
    # e2o = e2s+s2o. So s2o = e2o - e2s.
    sat_to_ground_vector_inertial = ground_position_inertial - satellite_position_inertial
    # print("s2g: {}".format(sat_to_ground_vector_inertial))
    sat_to_ground_versor_inertial = sat_to_ground_vector_inertial/np.linalg.norm(sat_to_ground_vector_inertial,2)
    # Rotate that vector by half the FOV along an axis perpendicular to the s2g vector.
    sat_easting_versor = np.cross(sat_to_ground_versor_inertial, [0,0,1]) # This vector is perpendicular to the s2g unit vector. It should also be perpendicular to the N vector, but we care less about that.
    sat_easting_versor/=np.linalg.norm(sat_easting_versor,2)
    
    off_axis_versor_rotation = Rotation.from_rotvec(satellite.instrument_fov_rad[instrument]/2. * sat_easting_versor)
    # Rotate _that_ vector around the s2g vector n times.
    around_axis_versor_rotation = Rotation.from_rotvec(2*np.pi/num_samples * sat_to_ground_versor_inertial)

    # For each, find the intersection with the ground, defined as: e2s+s2g vector modulo is Earth radius.
    llas = []

    for sample_ix in range(num_samples):
        gaze_vector = off_axis_versor_rotation.apply(sat_to_ground_vector_inertial)
        for _s in range(sample_ix):
            gaze_vector = around_axis_versor_rotation.apply(gaze_vector)
        
        if USE_SPHERICAL_APPROXIMATION:
            # Let's start with the circular approximation
            # We want norm(satellite_position_inertial+lambda*gaze_vector) to be 6371. This is just a search, right? There may even be an analytical solution.
            # satellite_position_inertial+lambda*gaze_vector = (e2sx+l*s2gx, e2sy+l*s2gy, e2sz+l*s2gz)
            # (e2sx+l*s2gx)^2 + (e2sy+l*s2gy)^2 + (e2sz+l*s2gz)^2 = R^2
            # e2sx^2 + e2sy^2 + e2sz^2 + l^2*(s2gx^2+s2gy^2+s2gz^2) + 2*l * (e2sx*s2gx + e2sy*s2gy + e2sz*s2gz) = R^2
            # a* l^2 + b * l + c = 0 with
            # a = (s2gx^2+s2gy^2+s2gz^2)=norm(s2g) ; b = 2*(e2sx*s2gx + e2sy*s2gy + e2sz*s2gz) = 2*dot(e2s, s2g); c= e2sx^2 + e2sy^2 + e2sz^2-R^2=norm(e2s) - R^2;
            # l = (-b \pm sqrt (b^2-4*a*c))/2a
            _b = 2*np.dot(satellite_position_inertial, gaze_vector)
            _c = np.dot(satellite_position_inertial,satellite_position_inertial) - (R_earth_km+ground_lla.alt_km)**2
            _a = np.dot(gaze_vector,gaze_vector)
            # The gaze vector intersects the Earth twice (or zero times if it just looks away).
            # We pick the closest intersection.
            gaze_vector_lambda = (-_b-np.sqrt(_b*_b - 4*_a*_c))/(2*_a)
            # print(gaze_vector_lambda)
        else:
            # But in fact the Earth is oblate! We need to explicitly think about oblateness.
            # What we want here is that norm(satellite_position_inertial+lambda*gaze_vector) = R_oblate(lon, lat, alt).
            # Where lon is a function of lambda.
            # Specifically we want:
            # gaze_vector_on_ground = satellite_position_inertial+lambda*gaze_vector
            # latitude_rad = np.atan2(gaze_vector_on_ground[2], np.linalg.norm(gaze_vector_on_ground[:2]))
            # longitude_rad = np.atan2(gaze_vector_on_ground[1], gaze_vector_on_ground[0]) - pyorbital.astronomy.gmst(time)
            # np.linalg.norm(satellite_position_inertial+lambda*gaze_vector)=oblate_altitude(latitude_rad, longitude_rad)+altitude
            # And we can bisect our way home
            def wgs84_oblate_elevation(geocentric_latitude_rad):
                semi_major_axis = pyorbital.astronomy.A
                # f = (major-minor)/major; f*major = major-minor; minor = major(1-f)
                semi_minor_axis = pyorbital.astronomy.A*(1-pyorbital.astronomy.F)
                eccentricity = math.sqrt(2*pyorbital.astronomy.F-pyorbital.astronomy.F**2)
                oblate_elevation = semi_minor_axis/math.sqrt(1-(eccentricity*np.cos(geocentric_latitude_rad))**2)
                return oblate_elevation

            def radius_error_with_oblateness(gaze_vector_lambda):
                gaze_vector_on_ground = satellite_position_inertial+gaze_vector_lambda*gaze_vector
                gaze_radius = np.linalg.norm(gaze_vector_on_ground)
                
                geocentric_latitude_rad = np.atan2(gaze_vector_on_ground[2], np.linalg.norm(gaze_vector_on_ground[:2]))
                # geodetic_latitude_rad = np.atan(np.tan(geocentric_latitude_rad)/(1-pyorbital.astronomy.F)**2)

                # longitude_rad = np.atan2(gaze_vector_on_ground[1], gaze_vector_on_ground[0]) - pyorbital.astronomy.gmst(time)
                oblate_earth_radius = wgs84_oblate_elevation(geocentric_latitude_rad)+ground_lla.alt_km
                return (gaze_radius-oblate_earth_radius)

            try:
                # gaze_vector_lambda = scipy_bisect(radius_error_with_oblateness,0.5, 1.2)
                gaze_vector_lambda = scipy_newton(radius_error_with_oblateness,1.0)
            except ValueError as e:
                print(e)
                # Just skip the sample
                continue


        # Find the LLA in the J2K frame.
        gaze_vector_on_ground = satellite_position_inertial+gaze_vector_lambda*gaze_vector
        # Now we have a position in ECI/J2K. What is the corresponding lon-lat?
        # Geocentric latitude is easy.
        geocentric_latitude_rad = np.atan2(gaze_vector_on_ground[2], np.linalg.norm(gaze_vector_on_ground[:2]))
        # From that, we get the geodetic latitude. https://celestrak.org/columns/v02n03/
        geodetic_latitude_rad = np.atan(np.tan(geocentric_latitude_rad)/(1-pyorbital.astronomy.F)**2)
        
        # For longitude, let's get the angle from the vernal equinox (x).
        local_sidereal_time_rad = np.atan2(gaze_vector_on_ground[1], gaze_vector_on_ground[0])
        # And then we will convert that to longitude by remembering that local sidereal time = GMT sidereal time + longitude
        greenwich_mean_sidereal_time = pyorbital.astronomy.gmst(time)
        longitude_rad = local_sidereal_time_rad-greenwich_mean_sidereal_time
        
        # Finally we pack things together
        llas.append((longitude_rad*180./np.pi, geodetic_latitude_rad*180./np.pi, 0))
        # ground_footprint_poly = Polygon([(_lla[0], _lla[1]) for _lla in ground_footprint_llas])
    return llas
        

class ISLLink():
    def __init__(self, source: Satellite, destination: Satellite, time: dt.datetime):
        self.source = source
        self.destination = destination
        self.time = time
    def __str__(self):
        return "ISL link from {} to {} at {}".format(self.source, self.destination, self.time)
    def __repr__(self):
        return self.__str__()
    
def find_isl_opportunities(satellites: list, min_time: dt.datetime, max_time: dt.datetime, passes_error_s=60, passes_horizon_deg=15):
    # We check at one-minute resolution. Is it close enough? Maybe.
    isl_links = []
    times = min_time + np.array([dt.timedelta(minutes=minutes)
                                for minutes in range( int(math.ceil((max_time-min_time).total_seconds()/60))) ])
    poses = {}
    for satellite in satellites:
        poses[satellite] = {}
        for time in times:
            poses[satellite][time] = np.array(satellite.orbit.get_position(time, normalize=False)[:3])

    for time in times:
        for satellite in satellites:
            # pose = np.array(satellite.orbit.get_position(time, normalize=False)[:3])
            pose = poses[satellite][time]
            for other_satellite, isl_range in satellite.isl_links.items():
                # other_pose = np.array(other_satellite.orbit.get_position(time, normalize=False)[:3])
                other_pose = poses[other_satellite][time]
                _range = np.linalg.norm(other_pose-pose)
                if (_range<isl_range):
                    isl_links.append(
                        ISLLink(
                            source=satellite,
                            destination=other_satellite,
                            time=time
                        )
                    )
    return isl_links

def build_unrolled_temporal_graph(comm_opportunities_by_sat: dict, isl_links: list, ground_stations: list=None):
    # Make a list of GSs we talk to
    if ground_stations is None:
        ground_stations = []
        for _links in comm_opportunities_by_sat.values():
            for _link in _links:
                ground_stations.append(_link[0])
        ground_stations = list(set(ground_stations))

    satellites = list(comm_opportunities_by_sat.keys())
    for isl_link in isl_links:
        satellites.append(isl_link.source)
        satellites.append(isl_link.destination)

    satellites = list(set(satellites))


    # Make a list of times
    times = []
    for isl_link in isl_links:
        times.append(isl_link.time)
    for _links in comm_opportunities_by_sat.values():
        for _link in _links:
            times.append(_link[1].rise.time)
            times.append(_link[1].fall.time)

    times = list(set(times))
    times.sort()

    contact_graph = nx.DiGraph()

    # Create the nodes
    for time in times:
        for satellite in satellites:
            contact_graph.add_node(
                f"{satellite.name}_{time.timestamp()}",
                station=satellite,
                station_type="satellite",
                time=time,
                )
        for gs in ground_stations:
            contact_graph.add_node(
                f"{gs.name}_{time.timestamp()}",
                station=gs,
                station_type="ground_station",
                time=time,
                )

    # Also create some convenience nodes
    for satellite in satellites:
        contact_graph.add_node(f"{satellite.name}_EARLIEST_OUT", station=satellite, station_type="satellite")
        contact_graph.add_node(f"{satellite.name}_LATEST_IN", station=satellite, station_type="satellite")
    for gs in ground_stations:
        contact_graph.add_node(f"{gs.name}_EARLIEST_OUT", station=gs, station_type="ground_station")
        contact_graph.add_node(f"{gs.name}_LATEST_IN", station=gs, station_type="ground_station")

    # Self links
    for time_index in range(len(times)-1):
        for satellite in satellites:
            contact_graph.add_edge(
                f"{satellite.name}_{times[time_index].timestamp()}",
                f"{satellite.name}_{times[time_index+1].timestamp()}",
                time=times[time_index],
                duration=times[time_index+1]-times[time_index],
                source_station=satellite,
                target_station=satellite,
                source_station_type="satellite",
                target_station_type="satellite",
                )
        for gs in ground_stations:
            contact_graph.add_edge(
                f"{gs.name}_{times[time_index].timestamp()}",
                f"{gs.name}_{times[time_index+1].timestamp()}",
                time=times[time_index],
                duration=times[time_index+1]-times[time_index],
                source_station=gs,
                target_station=gs,
                source_station_type="ground_station",
                target_station_type="ground_station",
                )
            
    # Links for earliest, latest
    for time in times:
        for satellite in satellites:
            contact_graph.add_edge(
                f"{satellite.name}_{time.timestamp()}",
                f"{satellite.name}_EARLIEST_OUT",
                time=times[time_index],
                duration=dt.timedelta(seconds=0),
                source_station=satellite,
                target_station=satellite,
                source_station_type="satellite",
                target_station_type="satellite",
                )
            contact_graph.add_edge(
                f"{satellite.name}_LATEST_IN",
                f"{satellite.name}_{time.timestamp()}",
                time=times[time_index],
                duration=dt.timedelta(seconds=0),
                source_station=satellite,
                target_station=satellite,
                source_station_type="satellite",
                target_station_type="satellite",
                )
        for gs in ground_stations:
            contact_graph.add_edge(
                f"{gs.name}_{time.timestamp()}",
                f"{gs.name}_EARLIEST_OUT",
                time=times[time_index],
                duration=dt.timedelta(seconds=0),
                source_station=gs,
                target_station=gs,
                source_station_type="ground_station",
                target_station_type="ground_station",
                )
            contact_graph.add_edge(
                f"{gs.name}_LATEST_IN",
                f"{gs.name}_{time.timestamp()}",
                time=times[time_index],
                duration=dt.timedelta(seconds=0),
                source_station=gs,
                target_station=gs,
                source_station_type="ground_station",
                target_station_type="ground_station",
                )
            
    # The broader Internet connecting ground stations
    for _time in times:
        for gs1 in ground_stations:
            for gs2 in ground_stations:
                if gs2 != gs1:
                    contact_graph.add_edge(
                        f"{gs1.name}_{_time.timestamp()}",
                        f"{gs2.name}_{_time.timestamp()}",
                        time=_time,
                        duration=dt.timedelta(seconds=0),
                        source_station=gs1,
                        target_station=gs2,
                        source_station_type="ground_station",
                        target_station_type="ground_station",
                        )

    # ISL link
    for isl_link in isl_links:
        contact_graph.add_edge(
            f"{isl_link.source.name}_{(isl_link.time.timestamp())}",
            f"{isl_link.destination.name}_{(isl_link.time.timestamp())}",
            time=isl_link.time,
            duration=dt.timedelta(seconds=0),
                source_station=isl_link.source,
                target_station=isl_link.destination,
                source_station_type="satellite",
                target_station_type="satellite",
            )
    
    # GS passes
    for satellite, opportunities in comm_opportunities_by_sat.items():
        for opportunity in opportunities:
            station = opportunity[0]
            obs_pass = opportunity[1]
            
            # IF we have contacts at both rise and fall time, we will be able to route both early and late transmissions.
            contact_graph.add_edge(
                f"{station.name}_{(obs_pass.rise.time.timestamp())}",
                f"{satellite.name}_{(obs_pass.rise.time.timestamp())}",
                time=obs_pass.rise.time,
                duration=dt.timedelta(seconds=0),
                source_station=station,
                target_station=satellite,
                source_station_type="ground_station",
                target_station_type="satellite",
                )
            contact_graph.add_edge(
                f"{satellite.name}_{(obs_pass.rise.time.timestamp())}",
                f"{station.name}_{(obs_pass.rise.time.timestamp())}",
                time=obs_pass.rise.time,
                duration=dt.timedelta(seconds=0),
                source_station=satellite,
                target_station=station,
                source_station_type="satellite",
                target_station_type="ground_station",
                )
            contact_graph.add_edge(
                f"{station.name}_{(obs_pass.rise.time.timestamp())}",
                f"{satellite.name}_{(obs_pass.rise.time.timestamp())}",
                time=obs_pass.fall.time,
                duration=dt.timedelta(seconds=0),
                source_station=station,
                target_station=satellite,
                source_station_type="ground_station",
                target_station_type="satellite",
                )
            contact_graph.add_edge(
                f"{satellite.name}_{(obs_pass.rise.time.timestamp())}",
                f"{station.name}_{(obs_pass.rise.time.timestamp())}",
                time=obs_pass.fall.time,
                duration=dt.timedelta(seconds=0),
                source_station=satellite,
                target_station=station,
                source_station_type="satellite",
                target_station_type="ground_station",
                )


    return contact_graph, times

def find_nodes_closest_to_time(station, desired_time: dt.datetime, contact_graph_times: list):

    insert_index = bisect.bisect(contact_graph_times, desired_time)
    if insert_index == len(contact_graph_times):
        # Last entry, so there is no node after that
        # return ValueError("Node is after end time for contact graph")
        closest_time_post = None
        node_name_post = None
    else: 
        closest_time_post = contact_graph_times[insert_index]
        node_name_post = f"{station.name}_{closest_time_post.timestamp()}"
    if insert_index == 0:
        # First entry, so there is no node before that
        closest_time_pre = None
        node_name_pre = None
        # return ValueError("Node is before start time for contact graph")
    else:
        closest_time_pre = contact_graph_times[insert_index]-1
        node_name_pre = f"{station.name}_{closest_time_pre.timestamp()}"

    return (node_name_pre, node_name_post)

def shortest_path_between_stations(
        source_station_name,
        target_station_name,
        start_time: dt.datetime,
        contact_graph: nx.DiGraph,
        contact_graph_times: list,
        max_time: dt.datetime=None,
        edge_cost=lambda s, t, edge: edge['duration'].total_seconds()
        ):
    
    index_time_closest_to_start_time = bisect.bisect(contact_graph_times, start_time)
    time_closest_to_start_time = contact_graph_times[index_time_closest_to_start_time]
    start_station_name = f"{source_station_name}_{time_closest_to_start_time.timestamp()}"

    if max_time is None:
        end_station_name = f"{target_station_name}_EARLIEST_OUT"
    else:
        index_time_closest_to_end_time = bisect.bisect(contact_graph_times, max_time)
        if index_time_closest_to_end_time>0:
            time_closest_to_end_time = contact_graph_times[index_time_closest_to_end_time]-1
            end_station_name = f"{target_station_name}_{time_closest_to_end_time.timestamp()}"
        else:
            raise nx.NetworkXNoPath("Max end time {} is earlier than earliest link time {}".format(max_time, contact_graph_times[0]))

    if (not (start_station_name in contact_graph.nodes())):
        # print("Start node {} is not reachable!")
        raise nx.NetworkXNoPath("Start node {} is not reachable!")
        # return [], []
    if (not (end_station_name in contact_graph.nodes())):
        raise nx.NetworkXNoPath("End node {} is not reachable!")
        # print("End node {} is not reachable!")
        # return [], []

    _path = nx.shortest_path(
        contact_graph,
        source=start_station_name,
        target=end_station_name,
        weight=edge_cost
        )
    # TODO remove all the self-loops
    abbreviated_path = [contact_graph.nodes[_path[0]]]
    for index in range(1,len(_path)):
        _source = contact_graph.nodes[_path[index-1]]
        _dest = contact_graph.nodes[_path[index]]
        if _source['station']!=_dest['station']:
            if _source != abbreviated_path[-1]:
                abbreviated_path.append(_source)    
            abbreviated_path.append(_dest)
    return abbreviated_path, _path

def plot_shortest_path_between_stations(shortest_path: list):
    
        # comm_opportunities_by_sat

        figglobal = plt.figure(figsize=(24,10))
        axglobal = figglobal.add_subplot(1,2,1, projection=ccrs.Robinson())
        axglobal.set_global()
        axglobal.coastlines()

        stride_s = 5
        _dt = dt.timedelta(seconds=stride_s)

        for hop_index in range(len(shortest_path)-1):
            hop_source = shortest_path[hop_index]
            hop_target = shortest_path[hop_index+1]

            # If both are ground stations, very light dash between them.
            # If both are satellites, and they are different, heavy ISL link between them.
            # If both are satellites, and they are the same, draw the orbit.
            # If station-sat, draw the link.

            if ((hop_source['station_type'] == "satellite") and (hop_target['station_type'] == "satellite")):
                if (hop_source['station'] == hop_target['station']):
                    # Plot trajectory
                    
                    time_steps_for_plotting = [hop_source['time'] + _dt*i for i in range(int(math.ceil((hop_target['time']-hop_source['time']).total_seconds()/_dt.total_seconds())))]
                
                    _orbit = hop_source['station'].orbit
                    _llas = [_orbit.get_lonlatalt(t) for t in time_steps_for_plotting]
                    # ax.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic())
                    axglobal.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic(), label=hop_source['station'].name) #, color=satcolor[satellite.name])

                else:
                    # Plot ISL link
                    if (hop_source['time'] != hop_target['time']):
                        raise ValueError("Times should be the same for a ISL")
                    
                    _llas_source = hop_source['station'].orbit.get_lonlatalt(hop_source['time'])
                    _llas_target = hop_target['station'].orbit.get_lonlatalt(hop_target['time'])

                    # ax.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic())
                    axglobal.plot([_llas_source[0], _llas_target[0]],[_llas_source[1], _llas_target[1]],transform=ccrs.Geodetic(), label=f"ISL {hop_source['station'].name}-{hop_target['station'].name}") #, color=satcolor[satellite.name])


            elif ((hop_source['station_type'] == "ground_station") and (hop_target['station_type'] == "ground_station")):
                if (hop_source['station'] == hop_target['station']):
                    # Do nothing, this is a local loop
                    pass
                else:
                    # Plot inter-GS link
                    if (hop_source['time'] != hop_target['time']):
                        raise ValueError("Times should be the same for a GS-GS link")

                    # ax.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic())
                    axglobal.plot([hop_source['station'].lon_deg, hop_target['station'].lon_deg],[hop_source['station'].lat_deg, hop_target['station'].lat_deg],transform=ccrs.PlateCarree(), label=f"GS {hop_source['station'].name}-{hop_target['station'].name}", linewidth=.2, linestyle=":") #, color=satcolor[satellite.name])

            else:
                # Ground-to-space link link
                if (hop_source['time'] != hop_target['time']):
                    raise ValueError("Times should be the same for a uplink-downlink")
                if (hop_source['station_type'] == "ground_station"):
                    _llas_dest = hop_target['station'].orbit.get_lonlatalt(hop_target['time'])
                    axglobal.plot([hop_source['station'].lon_deg, _llas_dest[0]],[hop_source['station'].lat_deg, _llas_dest[1]],transform=ccrs.Geodetic(), label=f"Uplink {hop_source['station'].name}-{hop_target['station'].name}") #, color=satcolor[satellite.name])
                else:
                    _llas_source = hop_source['station'].orbit.get_lonlatalt(hop_source['time'])
                    axglobal.plot([_llas_source[0], hop_target['station'].lon_deg],[_llas_source[1], hop_target['station'].lat_deg],transform=ccrs.Geodetic(), label=f"Downlink {hop_source['station'].name}-{hop_target['station'].name}") #, color=satcolor[satellite.name])

        if (len(shortest_path)>1 and (shortest_path[-1]['station_type'] == 'satellite') and (shortest_path[-1]['station'] != shortest_path[-2]['station'])):
            final_sat_node = shortest_path[-1]
            # Hack: if the endpoint is a satellite, draw its trajectory for a bit longer
            time_steps_for_plotting = [final_sat_node['time'] + _dt*i for i in range(int(math.ceil(300/_dt.total_seconds())))]
        
            _orbit = final_sat_node['station'].orbit
            _llas = [_orbit.get_lonlatalt(t) for t in time_steps_for_plotting]
            # ax.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic())
            axglobal.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic(), label=final_sat_node['station'].name) #, color=satcolor[satellite.name])


        axglobal.legend()
        # axglobal.set_title("{}-{}".format(min_time.strftime("%Y-%m-%d, %H"), (min_time+duration).strftime("%H")))
        # plt.savefig("{}-{}-{}h.png".format(min_time.strftime("%Y-%m-%d %H"),(min_time+duration).strftime("%H"), duration_h), bbox_inches='tight')