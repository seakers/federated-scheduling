import pyorbital
from pyorbital.orbital import Orbital
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import math
import pandas as pd
import bisect
import uuid
from enum import Enum

from fame import *

import requests
import urllib
import json

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
    
# Let's search for opportunities. 
# Input: a number of satellites, each with instruments. A number of ground locations we would like to image, with time windows.
# Output: a map from locations to satellite passes.

class ObservationRequest(Location):
    '''
    An observation request asks to observe a given location with a given instrument between a minimum and a maximum time.
    '''
    def __init__(self, lon_deg, lat_deg, min_time, max_time, alt_km=None, instrument="RGB", request_name=""):
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=alt_km, name=request_name)
        # self.name = request_name
        # self.lon_deg = lon_deg
        # self.lat_deg = lat_deg
        # self.alt_km = alt_km
        self.min_time = min_time
        self.max_time = max_time
        self.instrument = instrument
    def __str__(self):
        return "Request {} | Lon {}°, lat {}°, alt {} km from {} to {} with {}".format(
            self.name,
            self.lon_deg,
            self.lat_deg,
            self.alt_km,
            self.min_time,
            self.max_time,
            self.instrument,
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
        self.data_products = []
        self.known_phenomena = []
        self.attitude_controller_state = AttitudeController.FREE
        self.busy_with = None
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()
    
def observation_quality(opportunity: ObservationOpportunity, preferred_zenith_angle_deg=45):
    '''
    A quality function that specifies how good an opportunity is
    '''
    return abs(90.-opportunity.look_angle_dec_deg)/90. + abs(preferred_zenith_angle_deg-opportunity.sun_zenith_angle_deg)/90 - opportunity.range_km/1000

def find_observation_opportunities(observation_requests: list, satellites: list, passes_error_s=60, passes_horizon_deg=0):
    opportunities = {}
    
    for request in observation_requests:
        opportunities[request] = {}
        for satellite in satellites:
            if request.instrument in satellite.instruments:
                orbit = satellite.orbit
                opportunities[request][satellite] = []
                _passes = orbit.get_next_passes(
                    request.min_time,
                    int(math.ceil((request.max_time-request.min_time).total_seconds()/3600)),
                    request.lon_deg,
                    request.lat_deg,
                    request.alt_km,
                    passes_error_s,
                    passes_horizon_deg
                )
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
        passes_horizon_deg=passes_horizon_deg
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