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

from fame_geometry import *
import copy

import requests
import urllib
import json

MIN_HORIZON_ANGLE_FOR_PASS_DEG = 15
# MIN_HORIZON_ANGLE_FOR_OBS_DEG = 15

class Event():
    """ An Event has a time and a function that is called at that time.
    """
    def __init__(self, time: dt.datetime, action_callable, name: str="", id: str=None):
        self.time = time
        self.action_callable = action_callable
        self.name = name
        if id is None:
            id = uuid.uuid4()
        self.id = id
    def __str__(self):
        return "Event {} at {}".format(self.name, self.time)
    def __repr__(self):
        return self.__str__()

class ObservationEvent(Event):
    def __init__(self, time: dt.datetime, action_callable, name: str="", id: str=None, satellite: Satellite=None, opportunity: ObservationOpportunity=None):
        """_summary_

        Args:
            time (dt.datetime): The time of the event
            action_callable (_type_): The function that is called at the time of the event.
            name (str, optional): the name of the event. Defaults to "".
            id (str, optional): a unique event ID. Defaults to None (in which case a UUID4 is generated).
            satellite (Satellite, optional): The satellite where the event occurs. Defaults to None.
            opportunity (ObservationOpportunity, optional): The observation opportunity performed by the satellite. Defaults to None.
        """
        super().__init__(time, action_callable, name, id)
        self.satellite = satellite
        self.opportunity = opportunity

class CommunicationEvent(Event):
    def __init__(self, time: dt.datetime, action_callable, name: str="", id: str=None, satellite: Satellite=None, station: Location=None, comm_pass: ObservationPass=None):
        super().__init__(time, action_callable, name, id)
        self.satellite = satellite
        self.station = station
        self.comm_pass = comm_pass

class Phenomenon(Pose):
    def __init__(self, lon_deg: float, lat_deg: float, alt_km: float, start_time: dt.datetime, end_time: dt.datetime, heading_deg: float=None, speed_kph: float=None, name: str=""):
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=alt_km, heading_deg=heading_deg, speed_kph=speed_kph, name=name)
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return "{}: Lon {}°, lat {}°, alt {} km, hdg {}°, speed {} km/h, start {}, end {}".format(self.name, self.lon_deg,self.lat_deg,self.alt_km, self.heading_deg, self.speed_kph, self.start_time, self.end_time)
    def __repr__(self):
        return self.__str__()

   
class World():
    def __init__(self, satellites: list=[], constellations: list = [], brokers: list = [], phenomena: list = [], events: list = []):
        self.satellites = satellites
        self.constellations = constellations
        self.brokers = brokers
        self.phenomena = phenomena
        self.events = events
        self.time = dt.datetime.min
        self.history = []
    
    def tick(self, print_forbidden_prefixes=[]):
        if len(self.events):
            _event = self.events.pop(0)

            _print_event_name = True
            for forbidden_names in print_forbidden_prefixes:
                if _event.name.startswith(forbidden_names):
                    _print_event_name = False
                    break
            if _print_event_name:
                print("Executing {}".format(_event))

            self.time = _event.time
            outcome = _event.action_callable()

            self.history.append({
                'time': _event.time,
                'event': _event,
                'phenomena': [copy.deepcopy(p) for p in self.phenomena],
                'states': {
                    'satellites': [copy.deepcopy(s) for s in self.satellites],
                    # Constellations and brokers have a pointer to World, which has a pointer to constellations, which...recursion!
                    # 'constellations': [copy.deepcopy(c) for c in self.constellations],
                    # 'brokers': [copy.deepcopy(b) for b in self.brokers],
                }
            })
            
        else:
            print("No more events")
            
        return len(self.events)

    def add_satellite(self, satellite):
        self.satellite.append(satellite)

    def add_constellation(self, constellation):
        self.constellations.append(constellation)
    
    def add_broker(self, broker):
        self.brokers.append(broker)
    
    def add_event(self, event):
        # print("Adding {}".format(event))
        if (event.time<self.time):
            raise ValueError("Event {} is earlier than sim time {}".format(event, self.time)) 
        bisect.insort(self.events, event, key=lambda x: x.time)
        
    def do_observation(self, observation: ObservationOpportunity, spacecraft: Satellite, phenomenon_processor=lambda o, s, p: p):
        # Find phenomena close to the observation location in space and at the right time
        # Return a data product and a list of event states
        # print("Obs opp {}".format(observation))
        observed_phenomena = []

        # Now let's see what we observed
        for _phenomenon in self.phenomena:
            if (_phenomenon.start_time <= observation.time and _phenomenon.end_time > observation.time):
                
                # Compute phenomenon location on the planet
                _ph_location_ecf = np.array(pyorbital.astronomy.observer_position(observation.time, _phenomenon.lon_deg, _phenomenon.lat_deg, _phenomenon.alt_km)[0][:3])
                # Compute observation location on the planet
                _obs_location_ecf = np.array(pyorbital.astronomy.observer_position(observation.time, observation.lon_deg, observation.lat_deg, observation.alt_km)[0][:3])
                # Compute SC location
                _sc_location_ecf = np.array(spacecraft.orbit.get_position(observation.time, normalize=False)[0][:3])
                # Compute angle between sc-observation and sc-phenomenon
                _ph_sc_vector = _ph_location_ecf-_sc_location_ecf
                _obs_sc_vector = _obs_location_ecf - _sc_location_ecf
                # print("PhSc {} || ObsSc {}".format(_ph_sc_vector, _obs_sc_vector))
                # print("Dot: {} || N1: {} N2: {}".format(np.dot(_ph_sc_vector,_obs_sc_vector),np.linalg.norm(_ph_sc_vector,2), np.linalg.norm(_obs_sc_vector,2))) 
                # print("Acos: {}, angle: {}".format(np.dot(_ph_sc_vector,_obs_sc_vector)/(np.linalg.norm(_ph_sc_vector,2)*np.linalg.norm(_obs_sc_vector,2)), np.arccos(np.dot(_ph_sc_vector,_obs_sc_vector)/(np.linalg.norm(_ph_sc_vector,2)*np.linalg.norm(_obs_sc_vector,2)))))
                _ph_obs_angle_rad = np.arccos(np.clip(np.dot(_ph_sc_vector,_obs_sc_vector)/(np.linalg.norm(_ph_sc_vector,2)*np.linalg.norm(_obs_sc_vector,2)),-1,1))
                # print("Obs angle (rad) {}".format(_ph_obs_angle_rad))
                # If angle<FOV, return phobservation
                if _ph_obs_angle_rad < spacecraft.instrument_fov_rad[observation.instrument]:
                    # print("Close enough")
                    observed_phenomena.append(_phenomenon)
                else:
                    # print("Too far")
                    pass
        # Store SOMETHING for the completed observation
        if observation not in spacecraft.data_products.keys():
            spacecraft.data_products[observation] = []
        spacecraft.data_products[observation].extend([phenomenon_processor(observation, spacecraft, p) for p in observed_phenomena])
        spacecraft.known_phenomena.append(observed_phenomena)
        
        return True
    
def do_downlink(spacecraft: Satellite, scheduler, comm_pass: ObservationPass): # scheduler is a ConstellationGroundScheduler,defined next 
    # Simple: downlink all. Future: downlink up to x.
    
    # Duration is unused for now
    duration = comm_pass.fall.time - comm_pass.rise.time

    _downlinked = []
    if len(spacecraft.data_products):
        print("Spacecraft {} has {} data products to download".format(spacecraft, len(spacecraft.data_products)))
        
    for _observation, data_product in spacecraft.data_products.items():
        print("  Downlinked {}".format(_observation))
        _downlinked.append(_observation)
        
        # TODO here we assume the DP is for a given observation
        matching_requests = scheduler._requests[scheduler._requests['observation']==_observation]
        # print(scheduler._requests)
        # print(matching_requests)
        if len(matching_requests)>=1:
            # print(data_product)
            # try:
            scheduler._requests.loc[scheduler._requests['observation']==_observation, 'status'] = "OK! Data received"
            # scheduler._requests.loc[scheduler._requests['observation']==_observation, 'data_product'] = data_product
            for _ix, _ready_callback in scheduler._requests.loc[scheduler._requests['observation']==_observation, 'ready_callback'].items():
                scheduler._requests.loc[_ix, 'data_product'] = data_product
                _ready_callback(data_product)
            # except Exception as e:
            #     import pdb; pdb.set_trace()
        else:
            print("   Could not find matching request for observation {}".format(_observation))
        if len(matching_requests)>1:
            print("   I found multiple requests for observation {}!".format(_observation))

        

    for _observation in _downlinked:
        spacecraft.data_products.pop(_observation)

class ConstellationGroundScheduler():
    def __init__(self, satellites: list, ground_stations: list, world: World, name="Constellation"):
        self.name = name
        self.satellites = satellites
        self.ground_stations = ground_stations
        self.world = world
        # self.requests = {}
        self._requests = pd.DataFrame(columns=['request', 'satellite', 'observation', 'uplink', 'downlink', 'status', 'data_product', 'scheduled_callback', 'unscheduled_callback', 'ready_callback'])

    def screen_request_for_feasibility(self, satellite: Satellite, _request: ObservationRequest, screen_against_comm_passes:bool=True):
        # Check if a given request conflicts with existing requests.
        # TODO this is horrifyingly expensive because we do not exploit the fact that
        #  requests are sorted. We should improve this, ideally without rebuilding a full on timeline library.
        conflicting_requests = self._requests.loc[
            self._requests.apply(
            lambda x: 
                (x['status']=="Scheduled") and # We have actually scheduled this
                (x['observation'].time+x['observation'].duration > _request.time) and # The end of the other observation is after we start
                (x['observation'].time < _request.time+_request.duration) and # The start of the other observation is before we end
                (x['satellite'] == satellite) # This request is on the same satellite. Note that we check these are the same OBJECT, not just the same name.
            , axis=1)]
        if len(conflicting_requests):
            print("   [{}:{}]Conflict with another request".format(self.name, satellite.name))
            return False
        if (screen_against_comm_passes):
            conflicting_uplinks = self._requests.loc[
                self._requests.apply(
                lambda x: 
                    (x['status']=="Scheduled") and # We have actually scheduled this
                    (x['uplink'].fall.time > _request.time) and # The end of the comm pass is after we start
                    (x['uplink'].rise.time < _request.time + _request.duration) and # The start of the comm pass is before we end
                    (x['satellite'] == satellite) # This request is on the same satellite
                , axis=1)]
            if len(conflicting_uplinks):
                print("   [{}:{}]Conflict with an uplink".format(self.name, satellite.name))
                return False
            conflicting_downlinks = self._requests.loc[
                self._requests.apply(
                lambda x: 
                    (x['status']=="Scheduled") and # We have actually scheduled this
                    (x['downlink'].fall.time > _request.time) and # The end of the comm pass is after we start
                    (x['downlink'].rise.time < _request.time + _request.duration) and # The start of the comm pass is before we end
                    (x['satellite'] == satellite) # This request is on the same satellite
                , axis=1)]
            if len(conflicting_downlinks):
                print("   [{}:{}]Conflict with a downlink".format(self.name, satellite.name))
                return False
        return True
    
    def screen_pass_for_feasibility(self, satellite: Satellite,  _obs_pass: ObservationPass, screen_against_comm_passes:bool=False):
        # Check if a given pass conflicts with existing requests.
        # TODO this is horrifyingly expensive because we do not exploit the fact that
        #  requests are sorted. We should improve this, ideally without rebuilding a full on timeline library.
        conflicting_requests = self._requests.loc[
            self._requests.apply(
            lambda x: 
                (x['status']=="Scheduled") and # We have actually scheduled this
                (x['observation'].time+x['observation'].duration > _obs_pass.rise.time) and # The end of the other observation is after we start
                (x['observation'].time < _obs_pass.fall.time) and # The start of the other observation is before we end
                (x['satellite'] == satellite) # This request is on the same satellite. Note that we check these are the same OBJECT, not just the same name.
            , axis=1)]
        if len(conflicting_requests):
            return False
        if (screen_against_comm_passes):
            conflicting_uplinks = self._requests.loc[
                self._requests.apply(
                lambda x: 
                    (x['status']=="Scheduled") and # We have actually scheduled this
                    (x['uplink'].fall.time > _obs_pass.rise.time) and # The end of the comm pass is after we start
                    (x['uplink'].rise.time < _obs_pass.fall.time) and # The start of the comm pass is before we end
                    (x['satellite'] == satellite) # This request is on the same satellite
                , axis=1)]
            if len(conflicting_uplinks):
                return False
            conflicting_downlinks = self._requests.loc[
                self._requests.apply(
                lambda x: 
                    (x['status']=="Scheduled") and # We have actually scheduled this
                    (x['downlink'].fall.time > _obs_pass.rise.time) and # The end of the comm pass is after we start
                    (x['downlink'].rise.time < _obs_pass.fall.time) and # The start of the comm pass is before we end
                    (x['satellite'] == satellite) # This request is on the same satellite
                , axis=1)]
            if len(conflicting_downlinks):
                return False
        return True

    def schedule_request(
            self,
            request: ObservationRequest,
            current_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            callback_request_scheduled=lambda req_pass: None,
            callback_request_unscheduled=lambda reason: None,
            callback_request_ready=lambda data_product: None,
            phenomenon_processor=lambda o, s, p: p
            ):
        # Pick the best satellite to fulfill this. This is where we'll need to be smarter. Or not! Just pick something starting the day after.
        print("[{}] Scheduling request {}".format(self.name, request))
        # self.requests[request] = {
        _request_dict = {
            'request': request,
            'satellite': None,
            'observation': None,
            'uplink': None,
            'downlink': None,
            'status': None,
            'data_product': None,
            'scheduled_callback': callback_request_scheduled,
            'unscheduled_callback': callback_request_unscheduled,
            'ready_callback': callback_request_ready,
        }

        try:
            _pdrequest = pd.DataFrame([_request_dict])
        except Exception as e:
            import pdb; pdb.set_trace()

        self._requests = pd.concat([self._requests, _pdrequest], ignore_index=True)
              
        _opportunities = find_observation_opportunities(
            [request,],
            satellites=self.satellites,
            passes_error_s=60,
            # passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_OBS_DEG
        )
        if len(_opportunities):
            # best_request = None
            passes = _opportunities[request]
            # for request, passes in _opportunities.items(): # Only one request, so this just unpacks the opportunities and its OK to reset best_quality below
            if len(passes):
                _best_quality = - np.inf
                _best_satellite = None
                _best_pass = None
                _best_uplink_comm_opportunity = None
                _best_uplink_comm_opportunity_station = None
                _best_downlink_comm_opportunity = None
                _best_downlink_comm_opportunity_station = None

                for satellite, satpasses in passes.items():
                    for satpass in satpasses:
                        # Check if the satellite is free at this time.
                        # Query the table of observations for 1. planned, 2. on the satellite we are examining.
                        # Check by time if there is something nearby.
                        # If there is, back off.
                        _pass_is_feasible = self.screen_request_for_feasibility(satellite, satpass.highest)
                        if (_pass_is_feasible == False):
                            continue

                        _quality = observation_quality(satpass.highest)
                        # Find a feasible uplink for this opportunity
                        _, ul_comm_opportunities = find_contact_opportunities(
                            ground_stations=self.ground_stations,
                            satellites=[satellite, ],
                            min_time=current_time,
                            max_time=satpass.highest.time,
                            passes_error_s=60,
                            passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_PASS_DEG,
                        )

                        if ((satellite in ul_comm_opportunities.keys()) and (len(ul_comm_opportunities[satellite])))==0:
                            # No contacts for this satellite! Maybe we were too greedy
                            print("No contacts for this satellite! Maybe we were too greedy")
                            continue
                        
                        earliest_ul_opportunity = None
                        earliest_ul_opportunity_station = None
                        for comm_opportunity in ul_comm_opportunities[satellite]:
                            if self.screen_pass_for_feasibility(satellite, comm_opportunity[1], screen_against_comm_passes=False):
                                earliest_ul_opportunity = comm_opportunity[1]
                                earliest_ul_opportunity_station = comm_opportunity[0]
                                break

                        if ((earliest_ul_opportunity is None) or (earliest_ul_opportunity_station is None)):
                            # No timely *unconflicted* contact! Maybe we were too greedy
                            print("No timely *unconflicted* contact! Maybe we were too greedy")
                            continue

                        # At this point, satpass contains the satellite pass, earliest_ul_opportunity contains the corresponding uplink

                        # Find a feasible downlink for this opportunity after the event
                        # Find downlink opportunities
                        _, _dl_comm_opportunities = find_contact_opportunities(
                            ground_stations=self.ground_stations,
                            satellites=[satellite, ],
                            min_time=satpass.highest.time+satpass.highest.duration,
                            max_time=satpass.highest.time+satpass.highest.duration+dt.timedelta(hours=48),
                            passes_error_s=60,
                            passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_PASS_DEG,
                        )
                        # # Schedule downlink events for those
                        if satellite not in _dl_comm_opportunities.keys() or len(_dl_comm_opportunities[satellite]) == 0:
                            print("Could not find a suitable downlink")
                            # Could not find a suitable downlink
                            continue
                            # self._requests.loc[self._requests['request']==request, 'status'] = "No timely downlink";
                            # callback_request_unscheduled("No timely downlink")
                            # return -4
                        
                        ## 

                        # Passes are sorted by time. An we checked above that there is at least one pass
                        dl_station = None
                        dl_pass = None
                        for comm_opportunity in _dl_comm_opportunities[satellite]:
                            if self.screen_pass_for_feasibility(satellite, comm_opportunity[1], screen_against_comm_passes=False):
                                dl_pass = comm_opportunity[1]
                                dl_station = comm_opportunity[0]
                                break

                        if ((dl_pass is None) or (dl_station is None)):
                            # Could not find a suitable unconflicted downlink
                            print("Could not find a suitable unconflicted downlink")
                            break
                            # self._requests.loc[self._requests['request']==request, 'status'] = "No timely unconflicted downlink";
                            # callback_request_unscheduled("No timely unconflicted downlink")
                            # return -4.5


                        if _quality >= _best_quality:
                            _best_quality = _quality
                            _best_satellite = satellite
                            _best_pass = satpass
                            _best_uplink_comm_opportunity = earliest_ul_opportunity
                            _best_uplink_comm_opportunity_station = earliest_ul_opportunity_station
                            _best_downlink_comm_opportunity = dl_pass
                            _best_downlink_comm_opportunity_station = dl_station
                        # print("{}: quality {}".format(satpass.highest, observation_quality(satpass.highest)))

                print(" [{}] Best request: {} with {}".format(self.name, _best_pass, _best_satellite))
                if (_best_pass is None):
                    print("   All observation opportunities are conflicting")
                    self._requests.loc[self._requests['request']==request, 'status'] = "All observation opportunities are conflicting"
                    callback_request_unscheduled("All observation opportunities are conflicting")
                    return -5
            else:
                print("No observation opportunities here")
                # print(_opportunities)
                # self.requests[request]['status'] = "No observation opportunities"
                self._requests.loc[self._requests['request']==request, 'status'] = "No observation opportunities"
                callback_request_unscheduled("No observation opportunities")
                return -1
        else:
            print("Something wrong with requests list, did you pass a request?")
        
        _best_sat_object = None
        for sat in self.satellites:
            if sat.name == _best_satellite.name:
                _best_sat_object = sat
        if (_best_sat_object is None):
            print("ERROR! Something wrong with finding the satellite")
            # self.requests[request]['status'] = "Could not find best satellite";
            self._requests.loc[self._requests['request']==request, 'status'] = "Could not find best satellite";

            callback_request_unscheduled("Could not find best satellite")
            return -3
        #

        schedule_observation_uplink(self.world, _best_sat_object, _best_uplink_comm_opportunity, _best_pass.highest, _best_uplink_comm_opportunity_station, phenomenon_processor=phenomenon_processor)
        schedule_sat_downlink(_world=self.world, satellite=_best_sat_object, comm_pass = _best_downlink_comm_opportunity, station = _best_downlink_comm_opportunity_station, constellation_scheduler=self)
        # # Do downlink
        
        # # Schedule an event where we tell the satellite about this. The event calls schedule_observation
        # - pick the earliest opportunity
        # - check it's early enough (if not return)
        # - schedule an Event at the comm opportunity time that, when triggered, calls schedule_observation with best_request
        self._requests.loc[self._requests['request']==request, 'satellite'] = _best_sat_object
        self._requests.loc[self._requests['request']==request, 'observation'] = _best_pass.highest
        self._requests.loc[self._requests['request']==request, 'uplink'] = _best_uplink_comm_opportunity
        self._requests.loc[self._requests['request']==request, 'downlink'] = _best_downlink_comm_opportunity
        self._requests.loc[self._requests['request']==request, 'status'] = "Scheduled";

        callback_request_scheduled(_best_pass.highest)
        return 0

    def unschedule_request(
            self,
            request: ObservationRequest,
            current_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            callback_opportunity_unscheduled=lambda status: None,
    ):
        # We unschedule all opportunities for a given observation request.

        # Check that the request is indeed something in our list
        matching_observations = self._requests[self._requests['request']==request]
        if len(matching_observations) == 0:
            print("[{}]: no matching observations for unschedule request {}".format(self.name, request))
            return None

        raise NotImplemented("TODO")

        # for obs in matching_observations.iterrows:
        #     if obs['status'] == "Scheduled"

        # For every matching request
        # Find the observation
        # Find the uplink for that observation
        # If the uplink has not yet elapsed, remove that event.
        # Optional: If the uplink has already gone out
        # Optional: See if there is a pass between the current time and the observation
        # Optional: If there is no pass, too bad.
        # Optional: If there is a pass, schedule a new uplink to cancel that observation.

    def schedule_downlinks(
        self,
        current_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
        max_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)+dt.timedelta(hours=24)
    ):
        # Find downlink opportunities
        _, comm_opportunities = find_contact_opportunities(
            ground_stations=self.ground_stations,
            satellites=self.satellites,
            min_time=current_time,
            max_time=max_time,
            passes_error_s=60,
            passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_PASS_DEG,
        )
        # Schedule downlink events for those
        for _sat, _comm_passes_and_stations in comm_opportunities.items():
            # print(_comm_pass_and_station)
            # print(_sat)
            for _comm_pass_and_station in _comm_passes_and_stations:
                _station = _comm_pass_and_station[0]
                _comm_pass = _comm_pass_and_station[1]
                schedule_sat_downlink(_world=self.world, satellite=_sat, comm_pass = _comm_pass, station = _station, constellation_scheduler=self)
        # Do downlink
    
    def get_request_status(self, request: ObservationRequest):
        return self._requests[self._requests['request'] == request].status

def schedule_observation(_world, satellite: Satellite, obs_opportunity, phenomenon_processor= lambda o, s, p: p):
    # An observation fires at the time of the observation. It adds known events to the satellite's known_phenomena store.
    # TODO it also adds an observation product to the satellite's 

    # This first bit is quite redundant. What you want is to maintain events for individual agents and then a global copy, right?
    satellite.scheduled_observations.append(obs_opportunity)

    def unlock_satellite(_satellite):
        if _satellite.attitude_controller_state == AttitudeController.INSTRUMENT:
            _satellite.attitude_controller_state = AttitudeController.FREE
            _satellite.busy_with = None
            return True
        else:
            return False

    def lock_satellite_and_observe(_satellite, _obs_opportunity, __world):
        if _satellite.attitude_controller_state != AttitudeController.FREE:
            print("Satellite busy ({})! Sat {} attempted observation {}".format(_satellite.attitude_controller_state, _satellite, _obs_opportunity))
            return False
        _satellite.attitude_controller_state = AttitudeController.INSTRUMENT
        _satellite.busy_with = _obs_opportunity

        _unlock_event = Event(
            name = "Unlock satellite after obs, sat {}".format(_satellite.name),
            time = _obs_opportunity.time+_obs_opportunity.duration,
            action_callable = lambda _sate=satellite: unlock_satellite(_sate)
        )
        _world.add_event(_unlock_event)
        return __world.do_observation(_obs_opportunity, _satellite, phenomenon_processor=phenomenon_processor)
    
    _event = ObservationEvent(
        name = "Obs, sat {}".format(satellite.name),
        time = obs_opportunity.time,
        # Note the kludge of default inputs to make sure the closure works and we capture the variables at the time of creation
        action_callable = lambda _opp=obs_opportunity, _sate=satellite, __world=_world: lock_satellite_and_observe(_sate, _opp, __world),
        satellite=satellite,
        opportunity=obs_opportunity
    )
    _world.add_event(_event)

    return 0

def schedule_observation_uplink(_world: World, satellite: Satellite, comm_opportunity: ObservationPass, obs_opportunity: ObservationOpportunity, station: Location, phenomenon_processor=lambda o, s, p: p):
    # An observation uplink fires at the time of the uplink. It adds an event that will trigger the observation at the appropriate time. 
    if (comm_opportunity.highest.time>obs_opportunity.time):
        raise ValueError("Uplink {} is after related observation {}".format(comm_opportunity, obs_opportunity))
    
    def unlock_satellite(_satellite, verbose=False):
        if (_satellite.attitude_controller_state == AttitudeController.COMMUNICATION 
            and _satellite.busy_with == comm_opportunity):
            _satellite.attitude_controller_state = AttitudeController.FREE
            _satellite.busy_with = None
            return True
        else:
            if verbose:
                print("Could not unlock satellite after ul comm opportunity with station {} at {}!".format(station, comm_opportunity.highest.time))
            return False
        
    def do_uplink_event(__world: World, __satellite: Satellite, __obsopp: ObservationOpportunity):
        if __satellite.attitude_controller_state == AttitudeController.INSTRUMENT:
            print("Satellite busy! Attempted uplink to sat {} from station {}".format(__satellite, station))
            return False
        __satellite.attitude_controller_state == AttitudeController.COMMUNICATION
        __satellite.busy_with = comm_opportunity
        _unlock_event = Event(
            name = "Unlock uplink, station {} to sat {}".format(station.name, __satellite.name),
            time = comm_opportunity.fall.time,
            action_callable = lambda _sate=satellite: unlock_satellite(_sate)
        )
        __world.add_event(_unlock_event)
        return schedule_observation(__world, __satellite, __obsopp, phenomenon_processor=phenomenon_processor)

    _event = CommunicationEvent(
        name="Uplink, station {} to sat {}".format(station.name, satellite.name),
        time = comm_opportunity.highest.time,
        # action_callable = lambda _w=_world, _s=satellite, _o=obs_opportunity: schedule_observation(_w, _s, _o)
        action_callable = lambda _w=_world, _s=satellite, _o=obs_opportunity: do_uplink_event(_w, _s, _o),
        satellite=satellite,
        station=station,
        comm_pass=comm_opportunity,
    )
    _world.add_event(_event)

def schedule_sat_downlink(
    _world: World,
    satellite: Satellite,
    comm_pass: ObservationPass,
    station: Location,
    constellation_scheduler: ConstellationGroundScheduler
):
    
    def unlock_satellite(_satellite, verbose=False):
        if (_satellite.attitude_controller_state == AttitudeController.COMMUNICATION 
            and _satellite.busy_with == comm_pass):
            _satellite.attitude_controller_state = AttitudeController.FREE
            _satellite.busy_with = None
            return True
        else:
            if verbose:
                print("Could not unlock satellite after dl comm opportunity with station {} at {}!".format(station, comm_pass.highest.time))
            return False
        
    def end_downlink_event(_spacecraft: Satellite, _scheduler: ConstellationGroundScheduler, _comm_pass: ObservationPass):
        do_downlink(spacecraft=_spacecraft, scheduler=_scheduler, comm_pass=_comm_pass)
        return unlock_satellite(_spacecraft)

    def start_downlink_event(_spacecraft: Satellite, _scheduler: ConstellationGroundScheduler, _comm_pass: ObservationPass):
        if _spacecraft.attitude_controller_state == AttitudeController.INSTRUMENT:
            print("Satellite busy! Attempted downlink to sat {} from station {} at ".format(_spacecraft, station, _comm_pass.rise.time))
            return False
        _spacecraft.attitude_controller_state = AttitudeController.COMMUNICATION
        _spacecraft.busy_with = _comm_pass

        _end_event = Event(
            name="End of downlink, station {} from sat {}".format(station.name, satellite.name),
            time = _comm_pass.fall.time,
            # action_callable = lambda _cs=_scheduler, _s=_spacecraft, _c=_comm_pass: do_downlink(spacecraft=_s, scheduler=_cs, comm_pass=_c)
            action_callable = lambda _cs=_scheduler, _s=_spacecraft, _c=_comm_pass: end_downlink_event(_spacecraft=_s, _scheduler=_cs, _comm_pass=_c)
        )
        _world.add_event(_end_event)
        return True


    _event = CommunicationEvent(
        name="Downlink, station {} from sat {}".format(station.name, satellite.name),
        time = comm_pass.rise.time,
        # action_callable = lambda _cs=constellation_scheduler, _s=satellite, _c=comm_pass: do_downlink(spacecraft=_s, scheduler=_cs, comm_pass=_c)
        action_callable = lambda _cs=constellation_scheduler, _s=satellite, _c=comm_pass: start_downlink_event(_spacecraft=_s, _scheduler=_cs, _comm_pass=_c),
        satellite=satellite,
        station=station,
        comm_pass=comm_pass
    )
    _world.add_event(_event)

class Broker():
    def __init__(self, constellations: list[ConstellationGroundScheduler], world: World, name="Broker"):
        self.name = name
        self.constellations = constellations
        # self.known_satellites = known_satellites
        self.world = world
        self._requests = pd.DataFrame(columns=['request', 'requested_pass', 'requested_constellation', 'requested_satellite', 'constellation', 'satellite', 'assigned_pass', 'assigned_downlink', 'status', 'data_product', 'scheduled_callback', 'unscheduled_callback', 'ready_callback'])

        # self.requests = {}

    def _screen_pass_for_feasibility(self, satellite: Satellite, _obs_pass: ObservationPass):
        # Check if a given pass conflicts with existing requests.
        # TODO this is horrifyingly expensive because we do not exploit the fact that
        #  requests are sorted. We should improve this, ideally without rebuilding a full on timeline library.
        if len(self._requests):
            conflicting_requests = self._requests.loc[
                self._requests.apply(
                lambda x: 
                    (x['status'] != "OK! Data received") and # We have submitted this, or it's scheduled, OR IT FAILED TO SCHEDULE (which suggests this is a bad time)
                    (x['requested_pass'] is not None) and
                    (x['requested_satellite'] is not None) and
                    (x['requested_pass'].highest.time+x['requested_pass'].highest.duration > _obs_pass.rise.time) and # The end of the other observation is after we start
                    (x['requested_pass'].highest.time < _obs_pass.fall.time) and # The start of the other observation is before we end
                    (x['requested_satellite'] == satellite) # This request is on the same satellite. Note that we check these are the same OBJECT, not just the same name.
                , axis=1)]
            if len(conflicting_requests):
                return False
        return True

    # Broadly, look at the ephemerides, find the best option, find the corresponding constellation, give them a window around that.
    def schedule_request(
            self,
            request: ObservationRequest,
            current_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            number_of_submissions: int=1,
            follow_up_action_success=lambda data_product: None,
            follow_up_action_failure=lambda reason: None,
            phenomenon_processor=lambda o, s, p: p
            ):
        # Pick the best satellite to fulfill this. This is where we'll need to be smarter. Or not! Just pick something starting the day after.
        print("[{}] scheduling request {}".format(self.name, request))
        # self.requests[request] = {


        _known_satellites = []
        _known_satellites_by_constellation = {}
        for constellation in self.constellations:
            _known_satellites += constellation.satellites
            for _sat in constellation.satellites:
                _known_satellites_by_constellation[_sat] = constellation

        _opportunities = find_observation_opportunities(
            [request,],
            satellites=_known_satellites,
            passes_error_s=60,
            # passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_OBS_DEG
        )
        # self.requests[request]['opportunities'] = _opportunities
        # self._requests.loc[self._requests['request']==request, 'opportunities'] = _opportunities

        passes = _opportunities[request]

        if len(passes):
            
            sorted_passes = [(satellite, satpass) for satellite, satpasses in passes.items() for satpass in satpasses if len(satpasses)]
            # Sort by quality
            sorted_passes.sort(key=lambda x: observation_quality(x[1].highest), reverse=True)
            
            # Something strange here. Can passes be of length>0 but sorted_passes be empty?
            _best_satellite = sorted_passes[0][0]

            _best_pass = sorted_passes[0][1]
            _best_quality = observation_quality(_best_pass.highest)

            print("Best request: {} with {}".format(_best_pass, _best_satellite))
        else:
            print("No observation opportunities here")
            # self.requests[request]['status'] = "No observation opportunities";
            _request_dict = {
                'request': request,
                'requested_pass' : None,
                'requested_constellation' : None,
                'requested_satellite' : None,
                'constellation': None,
                'satellite': None,
                'assigned_pass': None,
                'assigned_downlink': None,
                'status': "No observation opportunities",
                'data_product': None,
                'scheduled_callback': lambda x: None,
                'unscheduled_callback': lambda x: None,
                'ready_callback': lambda x: None,
            }
            _pdrequest = pd.DataFrame([_request_dict])
            self._requests = pd.concat([self._requests, _pdrequest], ignore_index=True)

            # self._requests.loc[self._requests['request']==request, 'status'] = "No observation opportunities"
            return -1

        successful_submissions_for_this_request = 0

        for opportunity_ix in range(len(sorted_passes)): # Odd legacy construction, we should probably iterate directly
            if successful_submissions_for_this_request>=number_of_submissions:
                break

            _best_pass = sorted_passes[opportunity_ix][1]
            _best_satellite = sorted_passes[opportunity_ix][0]

            if (_best_pass is not None) and (_best_satellite is not None):

                if not (self._screen_pass_for_feasibility(_best_satellite, _best_pass)):
                    # pass
                    # This pass is not feasible, forget about it
                    print(" Broker skipping a good pass for feasibility")
                    continue
            
                # We are going to register a submission for this
                successful_submissions_for_this_request += 1

                _best_constellation = _known_satellites_by_constellation[_best_satellite]

                def callback_request_scheduled(assigned_pass, _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation, __best_satellite=_best_satellite):
                    print(" [{}] confirmed scheduling of request {} from pass {}, constellation {}".format(self.name, _request, __best_pass, __best_constellation.name))
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'assigned_pass'] = assigned_pass
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = "Scheduled"
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'constellation'] = {}
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'satellite'] = __best_satellite
                    # TODO Mark this satellite/pass as a busy time, keep track for internal rescheduling
                    return
                
                def callback_request_unscheduled(reason, _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation):
                    print(" [{}] received UNscheduling of request {}, pass {}, from {}".format(self.name, request, _best_pass, _best_constellation.name))
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'assigned_pass'] = None
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = reason
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'constellation'] = None
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'satellite'] = None
                    follow_up_action_failure(reason)
                    # Also reschedule

                    return
                
                def callback_request_ready(data_product,  _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation):
                    print(" [{}: ] data ready for request {}, pass {}, from {}".format(self.name, _request, __best_pass, __best_constellation.name))
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = "OK! Data received"
                    for _ix, __dp in self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'data_product'].items():
                        self._requests.loc[_ix, 'data_product'] = data_product
                    follow_up_action_success(data_product)
                    # TODO Attempt to cancel other requests for this observation
                    return

                _constellation_request = ObservationRequest(
                    lon_deg=request.lon_deg,
                    lat_deg=request.lat_deg,
                    min_time=_best_pass.rise.time-dt.timedelta(minutes=1), # This is the magic, we constrain the request to the constellation AND TIME that we like.
                    max_time=_best_pass.fall.time+dt.timedelta(minutes=1),
                    alt_km=request.alt_km,
                    instrument=request.instrument,
                    request_name=request.name,
                    min_elevation_deg=request.min_elevation_deg
                )

                _request_dict = {
                    'request': request,
                    'requested_pass' : _best_pass,
                    'requested_constellation' : _best_constellation,
                    'requested_satellite' :_best_satellite,
                    'constellation': None,
                    'satellite': None,
                    'assigned_pass': None,
                    'assigned_downlink': None,
                    'status': "Submitted",
                    'data_product': None,
                    'scheduled_callback': lambda x: None,
                    'unscheduled_callback': lambda x: None,
                    'ready_callback': lambda x: None,
                }
                _pdrequest = pd.DataFrame([_request_dict])
                self._requests = pd.concat([self._requests, _pdrequest], ignore_index=True)

                # Submit the request to the relevant constellation
                _best_constellation.schedule_request(
                    request=_constellation_request,
                    current_time=current_time,
                    callback_request_scheduled=callback_request_scheduled,
                    callback_request_unscheduled=callback_request_unscheduled,
                    callback_request_ready=callback_request_ready,
                    phenomenon_processor=phenomenon_processor,
                )
        if successful_submissions_for_this_request == 0:
            print("No unconflicted opportunities here")
            # self.requests[request]['status'] = "No observation opportunities";
            _request_dict = {
                'request': request,
                'requested_pass' : None,
                'requested_constellation' : None,
                'requested_satellite' : None,
                'constellation': None,
                'satellite': None,
                'assigned_pass': None,
                'assigned_downlink': None,
                'status': "No unconflicted observation opportunities",
                'data_product': None,
                'scheduled_callback': lambda x: None,
                'unscheduled_callback': lambda x: None,
                'ready_callback': lambda x: None,
            }
            _pdrequest = pd.DataFrame([_request_dict])
            self._requests = pd.concat([self._requests, _pdrequest], ignore_index=True)
                


    # Do the silly thing: decompose the workflow deterministically, then assign ALL those requests...
    def schedule_workflow(
            self,
            workflow,
            current_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
    ):
        pass



def retell_history(world: World):
    for _chronicle in world.history:
        print("Time: {}. Event: {}".format(_chronicle['time'], _chronicle['event']))
        if type(_chronicle['event'])==ObservationEvent:
            print("Observation: sat {} and opportunity {}".format(_chronicle['event'].satellite, _chronicle['event'].opportunity))
        if type(_chronicle['event'])==CommunicationEvent:
            print("Communication: station {} to sat {} during pass {}".format(_chronicle['event'].station, _chronicle['event'].satellite, _chronicle['event'].comm_pass))