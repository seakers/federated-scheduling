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

class ObservationRequest(Location):
    '''
    An observation request asks to observe a given location with a given instrument between a minimum and a maximum time.
    '''
    def __init__(self, lon_deg, lat_deg, min_time, max_time, alt_km=None, instrument="RGB", request_name="", min_elevation_deg=0):
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
        return "Request {} | Lon {}°, lat {}°, alt {} km from {} to {} with {}, min elevation {} deg".format(
            self.name,
            self.lon_deg,
            self.lat_deg,
            self.alt_km,
            self.min_time,
            self.max_time,
            self.instrument,
            self.min_elevation_deg
        )
    def __repr__(self):
        return self.__str__()

class ObservationOpportunity(Location):
    '''
    An observation opportunity specifies a time instant when an observation request can be fulfilled 
    '''
    def __init__(self, time, lon_deg, lat_deg, alt_km, look_angle_az_deg, look_angle_dec_deg, sun_zenith_angle_deg, range_km, name="", instrument="RGB", duration: dt.timedelta=dt.timedelta(seconds=60)):
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
    def __init__(self, name: str,  orbit: Orbital, instruments: list = ["RGB"], instrument_fov_rad: dict = {"RGB": 15.*np.pi/180.}):
        self.name = name
        self.orbit = orbit
        self.instruments = instruments
        self.instrument_fov_rad = instrument_fov_rad
        self.scheduled_observations = []
        self.data_products = {}
        self.known_phenomena = []
        self.attitude_controller_state = AttitudeController.FREE
        self.busy_with = None
        self.isl_links = {} # Satellite: range_km
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()
    
def observation_quality(opportunity: ObservationOpportunity, preferred_zenith_angle_deg=45):
    '''
    A quality function that specifies how good an opportunity is
    '''
    # TODO if opportunity.instrument == "RGB" add an entry for local time.
    return abs(90.-opportunity.look_angle_dec_deg)/90. + abs(preferred_zenith_angle_deg-opportunity.sun_zenith_angle_deg)/90 - opportunity.range_km/1000

def find_observation_opportunities(observation_requests: list, satellites: list, passes_error_s=60):
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

def spacecraft_fov(time: dt.datetime, satellite: Satellite, instrument: str, ground_lla: Location, num_samples: int = 12, USE_SPHERICAL_APPROXIMATION=False):
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
        