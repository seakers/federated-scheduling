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
import random

from fame_geometry import *
import copy

import requests
import urllib
import json

from fame_workflow import AssignmentTimeline, Impact, ImpactType

from fame_agents_base import *

class ConstellationGroundScheduler():
    def __init__(self, satellites: list, ground_stations: list, world, name="Constellation", ack_probability_if_scheduled: float=1., ack_probability_if_unscheduled: float=1., acceptance_probability: float=1.0, acceptance_probability_function=None, execution_probability_function=None):
        self.name = name
        self.satellites = satellites
        self.ground_stations = ground_stations
        self.world = world
        # self.requests = {}
        self._requests = pd.DataFrame(columns=requests_data_frame_columns)
        self.ack_probability_if_scheduled = ack_probability_if_scheduled
        self.ack_probability_if_unscheduled = ack_probability_if_unscheduled
        self.acceptance_probability = acceptance_probability
        self.acceptance_probability_function = acceptance_probability_function
        self.execution_probability_function = execution_probability_function
        self._satellite_busy_timelines_obs  = {ks: AssignmentTimeline(name=ks.name, initial_time=world.time, initial_value=False) for ks in self.satellites}
        self._satellite_busy_timelines_comm = {ks: AssignmentTimeline(name=ks.name, initial_time=world.time, initial_value=False) for ks in self.satellites}

    def __str__(self):
        return f"Constellation scheduler {self.name} with {len(self.satellites)} satellites"
    def __repr__(self):
        return self.__str__()
    
    def __deepcopy__(self, memo):
        new_constellation = ConstellationGroundScheduler(
            satellites=self.satellites,
            ground_stations=self.ground_stations,
            world=self.world,
            name=self.name,
            ack_probability_if_scheduled=self.ack_probability_if_scheduled,
            ack_probability_if_unscheduled=self.ack_probability_if_unscheduled,
            acceptance_probability=self.acceptance_probability,
            acceptance_probability_function=self.acceptance_probability_function,
            execution_probability_function=self.execution_probability_function,
        )
        new_constellation._requests = copy.deepcopy(self._requests, memo)
        new_constellation._satellite_busy_timelines_obs = copy.deepcopy(self._satellite_busy_timelines_obs, memo)
        new_constellation._satellite_busy_timelines_comm = copy.deepcopy(self._satellite_busy_timelines_comm, memo)
        return new_constellation

    def screen_opportunity_for_feasibility(self, satellite: Satellite, _request: ObservationRequest, screen_against_comm_passes:bool=True):
        # Check that:
        # - The satellite is free at the beginning of the pass
        # - There is nothing between the beginning and the end of the pass
        if (self._satellite_busy_timelines_obs[satellite].get_value_at(_request.time) == True):
            return False
        start_time_index = bisect.bisect(self._satellite_busy_timelines_obs[satellite].impact_container, _request.time, key=lambda x: x.time)
        end_time_index = bisect.bisect(self._satellite_busy_timelines_obs[satellite].impact_container, _request.time+_request.duration, key=lambda x: x.time)
        if (start_time_index != end_time_index): # Something is happening
            return False
        if screen_against_comm_passes:
            if (self._satellite_busy_timelines_comm[satellite].get_value_at(_request.time) == True):
                return False
            start_time_index = bisect.bisect(self._satellite_busy_timelines_comm[satellite].impact_container, _request.time, key=lambda x: x.time)
            end_time_index = bisect.bisect(self._satellite_busy_timelines_comm[satellite].impact_container, _request.time+_request.duration, key=lambda x: x.time)
            if (start_time_index != end_time_index): # Something is happening
                return False
        return True
        
        # return screen_opportunity_for_feasibility(existing_requests=self._requests, satellite=satellite, _request=_request, screen_against_comm_passes=screen_against_comm_passes, log_prefix=self.name)
    
    def screen_pass_for_feasibility(self, satellite: Satellite,  _obs_pass: ObservationPass, screen_against_comm_passes:bool=False):

        # Check that:
        # - The satellite is free at the beginning of the pass
        # - There is nothing between the beginning and the end of the pass
        if (self._satellite_busy_timelines_obs[satellite].get_value_at(_obs_pass.rise.time) == True):
            return False
        start_time_index = bisect.bisect(self._satellite_busy_timelines_obs[satellite].impact_container, _obs_pass.rise.time, key=lambda x: x.time)
        end_time_index = bisect.bisect(self._satellite_busy_timelines_obs[satellite].impact_container,   _obs_pass.fall.time, key=lambda x: x.time)
        if (start_time_index != end_time_index): # Something is happening
            return False
        if screen_against_comm_passes:
            if (self._satellite_busy_timelines_comm[satellite].get_value_at(_obs_pass.rise.time) == True):
                return False
            start_time_index = bisect.bisect(self._satellite_busy_timelines_comm[satellite].impact_container, _obs_pass.rise.time, key=lambda x: x.time)
            end_time_index = bisect.bisect(self._satellite_busy_timelines_comm[satellite].impact_container,   _obs_pass.fall.time, key=lambda x: x.time)
            if (start_time_index != end_time_index): # Something is happening
                return False
        return True

        # return screen_pass_for_feasibility(existing_requests=self._requests, satellite=satellite, _obs_pass=_obs_pass, screen_against_comm_passes=screen_against_comm_passes, log_prefix=self.name)

    def schedule_request_redundant(
            self,
            request: ObservationRequest,
            target_satellite: Satellite,
            target_pass: ObservationPass,
            current_time: dt.datetime = None,
            callback_request_scheduled=lambda req_pass: None,
            callback_request_unscheduled=lambda reason: None,
            callback_request_ready=lambda data_product: None,
            phenomenon_processor=lambda o, s, p: p
            ):
        """
        Submits a specific (target_satellite, target_pass) pair selected by the stochastic MILP planner.
        Bypasses internal greedy selection to guarantee that redundant backup passes are booked on 
        their intended satellites.
        """
        if current_time is None:
            current_time = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

        print(f"[{self.name}] Scheduling target-specific request {request.name} on {target_satellite.name} at {target_pass.highest.time}")

        _request_dict = {
            'request': request,
            'satellite': target_satellite,
            'observation': target_pass.highest,
            'uplink': None,
            'downlink': None,
            'status': ObservationStatus.UNKNOWN,
            'data_product': None,
            'scheduled_callback': callback_request_scheduled,
            'unscheduled_callback': callback_request_unscheduled,
            'ready_callback': callback_request_ready,
        }

        _pdrequest = pd.DataFrame([_request_dict])
        self._requests = pd.concat([self._requests, _pdrequest], ignore_index=True)

        # Step 1: Screen ONLY the target satellite pass for feasibility
        _pass_is_feasible = self.screen_opportunity_for_feasibility(target_satellite, target_pass.highest)
        if not _pass_is_feasible:
            print(f"   [{self.name}] Target pass on {target_satellite.name} is conflicted/busy")
            self._requests.loc[self._requests['request'] == request, 'status'] = ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING
            if random.random() < self.ack_probability_if_unscheduled:
                callback_request_unscheduled(ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING)
            return -5

        # Step 2: Establish Uplink/Downlink opportunities for this target satellite
        if target_satellite.has_continuous_isl_to_ground:
            earliest_ul_opportunity = "ISL"
            earliest_ul_opportunity_station = "ISL"
            dl_pass = "ISL"
            dl_station = "ISL"
        else:
            # Find uplink contact window
            _, ul_comm_opportunities = find_contact_opportunities(
                ground_stations=self.ground_stations,
                satellites=[target_satellite],
                min_time=current_time,
                max_time=target_pass.highest.time,
                passes_error_s=60,
                passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_PASS_DEG,
            )

            earliest_ul_opportunity = None
            earliest_ul_opportunity_station = None
            if target_satellite in ul_comm_opportunities:
                for comm_opportunity in ul_comm_opportunities[target_satellite]:
                    if self.screen_pass_for_feasibility(target_satellite, comm_opportunity[1], screen_against_comm_passes=False):
                        earliest_ul_opportunity = comm_opportunity[1]
                        earliest_ul_opportunity_station = comm_opportunity[0]
                        break

            if earliest_ul_opportunity is None or earliest_ul_opportunity_station is None:
                print(f"   [{self.name}] No unconflicted uplink contact for {target_satellite.name}")
                self._requests.loc[self._requests['request'] == request, 'status'] = ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING
                if random.random() < self.ack_probability_if_unscheduled:
                    callback_request_unscheduled(ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING)
                return -5

            # Find downlink contact window
            _, dl_comm_opportunities = find_contact_opportunities(
                ground_stations=self.ground_stations,
                satellites=[target_satellite],
                min_time=target_pass.highest.time + target_pass.highest.duration,
                max_time=target_pass.highest.time + target_pass.highest.duration + dt.timedelta(hours=48),
                passes_error_s=60,
                passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_PASS_DEG,
            )

            dl_pass = None
            dl_station = None
            if target_satellite in dl_comm_opportunities:
                for comm_opportunity in dl_comm_opportunities[target_satellite]:
                    if self.screen_pass_for_feasibility(target_satellite, comm_opportunity[1], screen_against_comm_passes=False):
                        dl_pass = comm_opportunity[1]
                        dl_station = comm_opportunity[0]
                        break

            if dl_pass is None or dl_station is None:
                print(f"   [{self.name}] No unconflicted downlink contact for {target_satellite.name}")
                self._requests.loc[self._requests['request'] == request, 'status'] = ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING
                if random.random() < self.ack_probability_if_unscheduled:
                    callback_request_unscheduled(ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING)
                return -5

        # Step 3: Stochastic Rejection Simulation
        if self.acceptance_probability_function is not None:
            theta_accept = self.acceptance_probability_function(request, target_satellite, self.world.time)
        else:
            theta_accept = self.acceptance_probability

        if random.random() > theta_accept:
            print(f"   [{self.name}] REJECTED request {request.name} on {target_satellite.name} (acceptance prob={theta_accept:.2f})")
            self._requests.loc[self._requests['request'] == request, 'status'] = ObservationStatus.CONSTELLATION_REJECTED
            if random.random() < self.ack_probability_if_unscheduled:
                callback_request_unscheduled(ObservationStatus.CONSTELLATION_REJECTED)
            return -6  # Rejected: timeline remains free for other requests, but this pass fails

        # Step 4: ACCEPTED - Book pass and lock timeline
        print(f"   [{self.name}] ACCEPTED request {request.name} on {target_satellite.name} (acceptance prob={theta_accept:.2f})")

        if earliest_ul_opportunity == "ISL":
            schedule_observation(self.world, target_pass.highest, phenomenon_processor=phenomenon_processor, execution_probability_function=self.execution_probability_function)
        else:
            schedule_observation_uplink(self.world, earliest_ul_opportunity, target_pass.highest, earliest_ul_opportunity_station, phenomenon_processor=phenomenon_processor, execution_probability_function=self.execution_probability_function)

        if dl_pass == "ISL":
            schedule_isl_downlink(_world=self.world, satellite=target_satellite, time=target_pass.highest.time, constellation_scheduler=self)
        else:
            schedule_sat_downlink(_world=self.world, satellite=target_satellite, comm_pass=dl_pass, station=dl_station, constellation_scheduler=self)

        self._requests.loc[self._requests['request'] == request, 'satellite'] = target_satellite
        self._requests.loc[self._requests['request'] == request, 'observation'] = target_pass.highest
        self._requests.loc[self._requests['request'] == request, 'uplink'] = earliest_ul_opportunity
        self._requests.loc[self._requests['request'] == request, 'downlink'] = dl_pass
        self._requests.loc[self._requests['request'] == request, 'status'] = ObservationStatus.SCHEDULED

        # Lock satellite busy timeline for accepted pass
        self._satellite_busy_timelines_obs[target_satellite].add_impact(Impact(time=target_pass.highest.time, type=ImpactType.ASSIGNMENT, value=True))
        self._satellite_busy_timelines_obs[target_satellite].add_impact(Impact(time=target_pass.highest.time + target_pass.highest.duration, type=ImpactType.ASSIGNMENT, value=False))

        if type(earliest_ul_opportunity) == ObservationPass:
            self._satellite_busy_timelines_comm[target_satellite].add_impact(Impact(time=earliest_ul_opportunity.rise.time, type=ImpactType.ASSIGNMENT, value=True))
            self._satellite_busy_timelines_comm[target_satellite].add_impact(Impact(time=earliest_ul_opportunity.fall.time, type=ImpactType.ASSIGNMENT, value=False))
        if type(dl_pass) == ObservationPass:
            self._satellite_busy_timelines_comm[target_satellite].add_impact(Impact(time=dl_pass.rise.time, type=ImpactType.ASSIGNMENT, value=True))
            self._satellite_busy_timelines_comm[target_satellite].add_impact(Impact(time=dl_pass.fall.time, type=ImpactType.ASSIGNMENT, value=False))

        if random.random() < self.ack_probability_if_scheduled:
            callback_request_scheduled(target_pass.highest)

        return 0
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
            'status': ObservationStatus.UNKNOWN,
            'data_product': None,
            'scheduled_callback': callback_request_scheduled,
            'unscheduled_callback': callback_request_unscheduled,
            'ready_callback': callback_request_ready,
        }

        _pdrequest = pd.DataFrame([_request_dict])


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
            # Only one request, so this just unpacks the opportunities and its OK to reset best_quality below
            if len(passes):
                _best_quality = - np.inf
                _best_satellite = None
                _best_pass = None
                _best_uplink_comm_opportunity = None
                _best_uplink_comm_opportunity_station = None
                _best_downlink_comm_opportunity = None
                _best_downlink_comm_opportunity_station = None

                # To find the best pass, we:
                # - Sort by quality
                # - Check feasibility going down the list
                # - Return the first feasible entry 

                sorted_passes = [(satellite, satpass) for satellite, satpasses in passes.items() for satpass in satpasses if len(satpasses)]
                sorted_passes.sort(key=lambda x: observation_quality(x[1].highest), reverse=True)

                for satellite, satpass in sorted_passes:
                        # Check if the satellite is free at this time.
                        # Query the table of observations for 1. planned, 2. on the satellite we are examining.
                        # Check by time if there is something nearby.
                        # If there is, back off.
                        _pass_is_feasible = self.screen_opportunity_for_feasibility(satellite, satpass.highest)
                        if (_pass_is_feasible == False):
                            continue

                        _quality = observation_quality(satpass.highest)

                        # If satellite has continuous ISL, skip the uplink and downlink search
                        
                        if (satellite.has_continuous_isl_to_ground == True):
                            earliest_ul_opportunity = "ISL"
                            earliest_ul_opportunity_station = "ISL"
                            dl_pass = "ISL"
                            dl_station = "ISL"
                        else:

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
                                print("   No contacts for this satellite! Maybe we were too greedy")
                                continue
                            
                            earliest_ul_opportunity = None
                            earliest_ul_opportunity_station = None
                            for comm_opportunity in ul_comm_opportunities[satellite]:
                                if self.screen_pass_for_feasibility(satellite, comm_opportunity[1], screen_against_comm_passes=False):
                                    earliest_ul_opportunity = comm_opportunity[1]
                                    earliest_ul_opportunity_station = comm_opportunity[0]
                                    break

                            if ((earliest_ul_opportunity is None) or (earliest_ul_opportunity_station is None)):
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
                                continue

                        # if _quality >= _best_quality:
                        _best_quality = _quality
                        _best_satellite = satellite
                        _best_pass = satpass
                        _best_uplink_comm_opportunity = earliest_ul_opportunity
                        _best_uplink_comm_opportunity_station = earliest_ul_opportunity_station
                        _best_downlink_comm_opportunity = dl_pass
                        _best_downlink_comm_opportunity_station = dl_station
                        # If we get here, then we have a complete solution. Since we sorted by quality, we can just stop.
                        break 

                print(" [{}] Best request: {} with {}".format(self.name, _best_pass, _best_satellite))
                if (_best_pass is None):
                    print("   All observation opportunities are conflicting")
                    self._requests.loc[self._requests['request']==request, 'status'] = ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING
                    if (random.random()<self.ack_probability_if_unscheduled):
                        callback_request_unscheduled(ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING)
                    return -5
            else:
                print("No observation opportunities here")
                self._requests.loc[self._requests['request']==request, 'status'] = ObservationStatus.NO_OBSERVATION_OPPORTUNITIES
                if (random.random()<self.ack_probability_if_unscheduled):
                    callback_request_unscheduled(ObservationStatus.NO_OBSERVATION_OPPORTUNITIES)
                return -1
        else:
            print("Something wrong with requests list, did you pass a request?")
        
        _best_sat_object = None
        for sat in self.satellites:
            if sat.name == _best_satellite.name:
                _best_sat_object = sat
        if (_best_sat_object is None):
            print("ERROR! Something wrong with finding the satellite")
            self._requests.loc[self._requests['request']==request, 'status'] = ObservationStatus.COULD_NOT_FIND_BEST_SATELLITE
            if (random.random()<self.ack_probability_if_unscheduled):
                callback_request_unscheduled(ObservationStatus.COULD_NOT_FIND_BEST_SATELLITE)
            return -3

        # === STOCHASTIC REJECTION SIMULATION ===
        # Determine acceptance probability
        if self.acceptance_probability_function is not None:
            # Dynamic acceptance based on custom function (e.g., load-dependent)
            theta_accept = self.acceptance_probability_function(request, _best_sat_object, self.world.time)
        else:
            # Static acceptance probability
            theta_accept = self.acceptance_probability

        # Roll dice: does constellation accept this broker request?
        if random.random() > theta_accept:
            # REJECT: Constellation refuses to schedule (e.g., too busy, internal conflicts)
            print(f"   [{self.name}] REJECTED request {request.name} (acceptance prob={theta_accept:.2f})")
            self._requests.loc[self._requests['request']==request, 'status'] = ObservationStatus.CONSTELLATION_REJECTED
            if (random.random()<self.ack_probability_if_unscheduled):
                callback_request_unscheduled(ObservationStatus.CONSTELLATION_REJECTED)
            return -6  # New error code for constellation rejection

        # ACCEPT: Proceed with scheduling
        print(f"   [{self.name}] ACCEPTED request {request.name} (acceptance prob={theta_accept:.2f})")
        #

        # If we have ISL, just schedule the observation through the magic comm link
        assert _best_sat_object == _best_pass.highest.satellite, "ERROR: something wrong with selecting the best satellite"

        if _best_uplink_comm_opportunity == "ISL":
            schedule_observation(self.world, _best_pass.highest, phenomenon_processor=phenomenon_processor, execution_probability_function=self.execution_probability_function)
        else:
            schedule_observation_uplink(self.world, _best_uplink_comm_opportunity, _best_pass.highest, _best_uplink_comm_opportunity_station, phenomenon_processor=phenomenon_processor, execution_probability_function=self.execution_probability_function)
        
        if _best_downlink_comm_opportunity == "ISL":
            schedule_isl_downlink(_world=self.world, satellite=_best_sat_object, time = _best_pass.highest.time, constellation_scheduler=self)
        else:
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
        self._requests.loc[self._requests['request']==request, 'status'] = ObservationStatus.SCHEDULED

        # Change sat busy timeline accordingly. Add busy around:
        # - Observation
        # - Uplink
        # - Downlink

        self._satellite_busy_timelines_obs[_best_sat_object].add_impact(Impact(time=_best_pass.highest.time, type=ImpactType.ASSIGNMENT, value=True))
        self._satellite_busy_timelines_obs[_best_sat_object].add_impact(Impact(time=_best_pass.highest.time+_best_pass.highest.duration, type=ImpactType.ASSIGNMENT, value=False))
        if type(_best_uplink_comm_opportunity) == ObservationPass:
            self._satellite_busy_timelines_comm[_best_sat_object].add_impact(Impact(time=_best_uplink_comm_opportunity.rise.time, type=ImpactType.ASSIGNMENT, value=True))
            self._satellite_busy_timelines_comm[_best_sat_object].add_impact(Impact(time=_best_uplink_comm_opportunity.fall.time, type=ImpactType.ASSIGNMENT, value=False))
        if type(_best_downlink_comm_opportunity) == ObservationPass:
            self._satellite_busy_timelines_comm[_best_sat_object].add_impact(Impact(time=_best_downlink_comm_opportunity.rise.time, type=ImpactType.ASSIGNMENT, value=True))
            self._satellite_busy_timelines_comm[_best_sat_object].add_impact(Impact(time=_best_downlink_comm_opportunity.fall.time, type=ImpactType.ASSIGNMENT, value=False))

        if (random.random()<self.ack_probability_if_scheduled):
            callback_request_scheduled(_best_pass.highest)
        return 0

    def cancel_request(
            self,
            request: ObservationRequest,
            target_satellite,
            target_pass,
            current_time: dt.datetime,
    ) -> bool:
        """
        Attempt to cancel a specific (request, satellite, pass) booking.

        Cancellation is feasible only if the pass has not yet started AND either:
          - The satellite has continuous ISL (can always be commanded), OR
          - The uplink command window has not yet fired (uplink.highest.time > current_time).

        On success: removes ObservationEvent + uplink CommunicationEvent from world.events,
        releases satellite busy-timeline slots, marks status CANCELLED, returns True.
        On failure: no state change, returns False.
        """
        if target_pass.highest.time <= current_time:
            print(f"   [{self.name}] Cannot cancel {request.name} on {target_satellite.name}: "
                  f"pass at {target_pass.highest.time} already started/past.")
            return False

        mask = self._requests['request'] == request
        sat_match = self._requests['satellite'] == target_satellite
        matching = self._requests[mask & sat_match]
        if len(matching) == 0:
            return False

        row = matching.iloc[0]
        uplink = row['uplink']
        downlink = row['downlink']
        status = row['status']

        if status != ObservationStatus.SCHEDULED:
            return False

        # Feasibility check: can we still send the cancel command?
        if target_satellite.has_continuous_isl_to_ground or uplink == "ISL":
            cancellable = True
        elif uplink is None:
            cancellable = False
        else:
            uplink_cmd_time = uplink.highest.time if hasattr(uplink, 'highest') else uplink.rise.time
            cancellable = uplink_cmd_time > current_time

        if not cancellable:
            print(f"   [{self.name}] Cannot cancel {request.name} on {target_satellite.name}: "
                  f"uplink already sent.")
            return False

        print(f"   [{self.name}] CANCELLING {request.name} on {target_satellite.name} "
              f"(pass at {target_pass.highest.time})")

        # Remove ObservationEvent and its uplink CommunicationEvent from world.events
        events_to_remove = []
        for ev in self.world.events:
            if isinstance(ev, ObservationEvent):
                if ev.satellite == target_satellite and ev.opportunity is target_pass.highest:
                    events_to_remove.append(ev)
            elif isinstance(ev, CommunicationEvent):
                if ev.satellite == target_satellite and uplink not in (None, "ISL"):
                    ev_comm = getattr(ev, 'comm_pass', None)
                    if ev_comm is not None and ev_comm is not None:
                        if hasattr(ev_comm, 'highest') and hasattr(uplink, 'highest'):
                            if ev_comm.highest.time == uplink.highest.time:
                                events_to_remove.append(ev)
        for ev in events_to_remove:
            try:
                self.world.events.remove(ev)
            except ValueError:
                pass

        # Remove from satellite.scheduled_observations
        target_satellite.scheduled_observations = [
            o for o in target_satellite.scheduled_observations
            if o is not target_pass.highest
        ]

        # Release obs busy-timeline
        tl_obs = self._satellite_busy_timelines_obs.get(target_satellite)
        if tl_obs is not None:
            obs_start = target_pass.highest.time
            obs_end = target_pass.highest.time + target_pass.highest.duration
            tl_obs.impact_container = [
                imp for imp in tl_obs.impact_container
                if imp.time != obs_start and imp.time != obs_end
            ]

        # Release comm busy-timeline (uplink + downlink)
        tl_comm = self._satellite_busy_timelines_comm.get(target_satellite)
        if tl_comm is not None:
            times_to_free = set()
            if uplink not in (None, "ISL"):
                if hasattr(uplink, 'rise'):
                    times_to_free.add(uplink.rise.time)
                    times_to_free.add(uplink.fall.time)
            if downlink not in (None, "ISL"):
                if hasattr(downlink, 'rise'):
                    times_to_free.add(downlink.rise.time)
                    times_to_free.add(downlink.fall.time)
            if times_to_free:
                tl_comm.impact_container = [
                    imp for imp in tl_comm.impact_container
                    if imp.time not in times_to_free
                ]

        self._requests.loc[mask & sat_match, 'status'] = ObservationStatus.CANCELLED
        return True

    def unschedule_request(
            self,
            request: ObservationRequest,
            current_time: dt.datetime=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            callback_opportunity_unscheduled=lambda status: None,
    ):
        # Legacy stub — use cancel_request() for targeted cancellation.
        matching_observations = self._requests[self._requests['request']==request]
        if len(matching_observations) == 0:
            print("[{}]: no matching observations for unschedule request {}".format(self.name, request))
            return None
        raise NotImplementedError("Use cancel_request() for targeted cancellation.")

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

def schedule_observation(_world, obs_opportunity: ObservationOpportunity, phenomenon_processor= lambda o, s, p: p, execution_probability_function=None):
    # An observation fires at the time of the observation. It adds known events to the satellite's known_phenomena store.
    # TODO it also adds an observation product to the satellite's

    # This first bit is quite redundant. What you want is to maintain events for individual agents and then a global copy, right?

    satellite = obs_opportunity.satellite
    satellite.scheduled_observations.append(obs_opportunity)

    def unlock_satellite(_satellite: Satellite):
        if _satellite.attitude_controller_state == AttitudeController.INSTRUMENT:
            _satellite.attitude_controller_state = AttitudeController.FREE
            _satellite.busy_with = None
            return True
        else:
            return False

    def lock_satellite_and_observe(_obs_opportunity: ObservationOpportunity, __world: World):
        _satellite = _obs_opportunity.satellite
        if _satellite.attitude_controller_state != AttitudeController.FREE:
            print("Satellite busy ({})! Sat {} attempted observation {}".format(_satellite.attitude_controller_state, _satellite, _obs_opportunity))
            return False
        _satellite.attitude_controller_state = AttitudeController.INSTRUMENT
        _satellite.busy_with = _obs_opportunity

        _unlock_event = Event(
            name = "Unlock satellite after obs, sat {}".format(_satellite.name),
            time = _obs_opportunity.time+_obs_opportunity.duration,
            action_callable = lambda _sate=_satellite: unlock_satellite(_sate)
        )
        _world.add_event(_unlock_event)

        # Execution failure coin flip: if p_exec < 1 and we lose the draw,
        # store an empty data product so do_downlink still fires ready_callback
        # but marks the row EXECUTION_FAILED with zero quality.
        if execution_probability_function is not None:
            p_exec = execution_probability_function(None, _satellite, _obs_opportunity)
            if random.random() > p_exec:
                print(f"   [Execution] FAILED for {_satellite.name} at {_obs_opportunity.time} (p_exec={p_exec:.2f})")
                if _obs_opportunity not in _satellite.data_products:
                    _satellite.data_products[_obs_opportunity] = []
                # Sentinel: None entry signals execution failure to do_downlink
                _satellite.data_products[_obs_opportunity].append(None)
                return False

        return __world.do_observation(_obs_opportunity, phenomenon_processor=phenomenon_processor)
    
    _event = ObservationEvent(
        name = "Obs, sat {}".format(satellite.name),
        time = obs_opportunity.time,
        # Note the kludge of default inputs to make sure the closure works and we capture the variables at the time of creation
        action_callable = lambda _opp=obs_opportunity, __world=_world: lock_satellite_and_observe(_opp, __world),
        satellite=satellite,
        opportunity=obs_opportunity
    )
    _world.add_event(_event)

    return 0

def schedule_observation_uplink(_world: World, comm_opportunity: ObservationPass, obs_opportunity: ObservationOpportunity, station: Location, phenomenon_processor=lambda o, s, p: p, execution_probability_function=None):
    # An observation uplink fires at the time of the uplink. It adds an event that will trigger the observation at the appropriate time. 
    if comm_opportunity.highest.satellite != obs_opportunity.satellite:
        raise ValueError(f"Comm opportunity and obs opportunity refer to different satellites! (Comm: {comm_opportunity.highest.satellite}, obs: {obs_opportunity.satellite})")

    if (comm_opportunity.highest.time>obs_opportunity.time):
        raise ValueError("Uplink {} is after related observation {}".format(comm_opportunity, obs_opportunity))
    
    satellite = obs_opportunity.satellite
    
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
        
    def do_uplink_event(__world: World, __obsopp: ObservationOpportunity):
        __satellite = __obsopp.satellite
        if __satellite.attitude_controller_state == AttitudeController.INSTRUMENT:
            print("Satellite busy! Attempted uplink to sat {} from station {}".format(__satellite, station))
            return False
        __satellite.attitude_controller_state == AttitudeController.COMMUNICATION
        __satellite.busy_with = comm_opportunity
        _unlock_event = Event(
            name = "Unlock uplink, station {} to sat {}".format(station.name, __satellite.name),
            time = comm_opportunity.fall.time,
            action_callable = lambda _sate=__satellite: unlock_satellite(_sate)
        )
        __world.add_event(_unlock_event)
        return schedule_observation(__world, __obsopp, phenomenon_processor=phenomenon_processor, execution_probability_function=execution_probability_function)

    _event = CommunicationEvent(
        name="Uplink, station {} to sat {}".format(station.name, satellite.name),
        time = comm_opportunity.highest.time,
        # action_callable = lambda _w=_world, _s=satellite, _o=obs_opportunity: schedule_observation(_w, _s, _o)
        action_callable = lambda _w=_world, _o=obs_opportunity: do_uplink_event(_w, _o),
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

def schedule_isl_downlink(
    _world: World,
    satellite: Satellite,
    time: dt.datetime,
    constellation_scheduler: ConstellationGroundScheduler
):       
    comm_pass = "ISL"
    _event = CommunicationEvent(
        name="ISL Downlink from sat {}".format(satellite.name),
        time = time,
        action_callable = lambda _cs=constellation_scheduler, _s=satellite, _c=comm_pass: do_downlink(spacecraft=_s, scheduler=_cs, comm_pass=_c),
        satellite=satellite,
        station="ISL",
        comm_pass=comm_pass
    )
    _world.add_event(_event)