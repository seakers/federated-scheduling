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

from fame_agents_base import *

MIN_HORIZON_ANGLE_FOR_PASS_DEG = 15

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