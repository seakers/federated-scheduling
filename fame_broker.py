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

from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus

from fame_workflow import *

class Broker():
    def __init__(
            self,
            constellations: list[ConstellationGroundScheduler],
            world: World, name="Broker",
            ):
        self.name = name
        self.constellations = constellations
        # self.known_satellites = known_satellites
        self.world = world
        self._requests = pd.DataFrame(columns=['request', 'requested_pass', 'requested_constellation', 'requested_satellite', 'constellation', 'satellite', 'assigned_pass', 'assigned_downlink', 'status', 'data_product', 'scheduled_callback', 'unscheduled_callback', 'ready_callback'])

        _known_satellites = []
        _known_satellites_by_constellation = {}
        for constellation in self.constellations:
            _known_satellites += constellation.satellites
            for _sat in constellation.satellites:
                _known_satellites_by_constellation[_sat] = constellation
        self._known_satellites = _known_satellites
        self._known_satellites_by_constellation = _known_satellites_by_constellation

    def _screen_pass_for_feasibility(self, satellite: Satellite, _obs_pass: ObservationPass):
        # Check if a given pass conflicts with existing requests.
        # TODO this is horrifyingly expensive because we do not exploit the fact that
        #  requests are sorted. We should improve this, ideally without rebuilding a full on timeline library.
        if len(self._requests):
            conflicting_requests = self._requests.loc[
                self._requests.apply(
                lambda x: 
                    (x['status'] != ObservationStatus.DATA_RECEIVED) and # We have submitted this, or it's scheduled, OR IT FAILED TO SCHEDULE (which suggests this is a bad time)
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

        _opportunities = find_observation_opportunities(
            [request,],
            satellites=self._known_satellites,
            passes_error_s=60,
            # passes_horizon_deg=MIN_HORIZON_ANGLE_FOR_OBS_DEG
        )
        # self.requests[request]['opportunities'] = _opportunities
        # self._requests.loc[self._requests['request']==request, 'opportunities'] = _opportunities

        # TODO what if there are no opportunities?
        passes = _opportunities[request]

        if len(passes):
            
            sorted_passes = [(satellite, satpass) for satellite, satpasses in passes.items() for satpass in satpasses if len(satpasses)]
            # Sort by quality
            sorted_passes.sort(key=lambda x: observation_quality(x[1].highest), reverse=True)
            
            # Something strange here. Can passes be of length>0 but sorted_passes be empty?
            _best_satellite = sorted_passes[0][0]

            _best_pass = sorted_passes[0][1]

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
                'status': ObservationStatus.NO_OBSERVATION_OPPORTUNITIES,
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

                # TODO also check if there is a timely uplink. Specifically, check for a path 
                # try:
                # shortest_path_between_stations(where_the_broker_is_now, _best_satellite, start_time= current_time, max_time=_best_pass.highest.time, contact_graph=contact_graph, contact_graph_times=contact_graph_times)
                # except nx.NetworkXNoPath as e
                # No path, move on
            
                # We are going to register a submission for this
                successful_submissions_for_this_request += 1

                _best_constellation = self._known_satellites_by_constellation[_best_satellite]

                def callback_request_scheduled(assigned_pass, _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation, __best_satellite=_best_satellite):
                    print(" [{}] confirmed scheduling of request {} from pass {}, constellation {}".format(self.name, _request, __best_pass, __best_constellation.name))
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'assigned_pass'] = assigned_pass
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = ObservationStatus.SCHEDULED
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'constellation'] = __best_constellation
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'satellite'] = __best_satellite
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
                    self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = ObservationStatus.DATA_RECEIVED
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
                    'status': ObservationStatus.SUBMITTED,
                    'data_product': None,
                    'scheduled_callback': callback_request_scheduled,
                    'unscheduled_callback': callback_request_unscheduled,
                    'ready_callback': callback_request_ready,
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
                'status': ObservationStatus.ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING,
                'data_product': None,
                'scheduled_callback': lambda x: None,
                'unscheduled_callback': lambda x: None,
                'ready_callback': lambda x: None,
            }
            _pdrequest = pd.DataFrame([_request_dict])
            self._requests = pd.concat([self._requests, _pdrequest], ignore_index=True)
                
    def add_workflow(self, workflow: Workflow):
        self.workflow = workflow
        self._workflow_graph, self._timeline_graph = build_workflow_graph(self.workflow)
        self._workflow_schedule_epoch = 0 # We use this to keep track of whether we rescheduled during dispatch

    def schedule_workflow(
            self,
            current_time: dt.datetime=dt.datetime.now(dt.timezone.utc),
            use_ilp: bool=False,
            update_timelines: bool=True,
            plot_schedule: bool = False
    ):
        # Come up with a schedule that satisfies the workflow
        if use_ilp:
            # TODO:
            # - Empty the timeline graph
            # - Pull the starting value and starting rate (from where?). Add a current starting value and starting rate.
            # - Iterate over in-progress tasks (dispatched not completed)
            # - For each in-progress task add POST impacts
            if update_timelines:
                self.workflow.timeline_updater(self.world.time, self.workflow.constrained_observation_requests, self.workflow.timelines)

            _ = ilp_schedule_workflow(
                workflow_graph=self._workflow_graph,
                timeline_graph=self._timeline_graph,
                satellites=self._known_satellites,
                feasibility_screener=self._screen_pass_for_feasibility,
                current_time=current_time,
                verbose=3
                )
        else:
            _ = greedy_schedule_workflow(
                workflow_graph=self._workflow_graph,
                timeline_graph=self._timeline_graph,
                satellites=self._known_satellites,
                feasibility_screener=self._screen_pass_for_feasibility,
                current_time=current_time,
                verbose=3
                )
                
        self._workflow_schedule_epoch += 1

        if plot_schedule:
            plot_workflow_schedule(workflow_graph=self._workflow_graph, timeline_graph=self._timeline_graph, time=self.world.time)
            plt.savefig(f"Schedule_epoch{self._workflow_schedule_epoch}.pdf", bbox_inches='tight')

        local_workflow_schedule_epoch_when_dispatching_started = self._workflow_schedule_epoch

        dispatchable_task_ids = find_dispatchable_tasks(self._workflow_graph)
        print(f" [Broker] There are {len(dispatchable_task_ids)} dispatchable tasks")

        for dispatchable_task in dispatchable_task_ids:
            # If we rescheduled in the meanwhile, don't keep dispatching stale stuff.
            # The test below will fail when _someone else_ called schedule_workflow elsewhere
            # The end result is that only the last agent to call schedule_workflow gets to dispatch 
            if self._workflow_schedule_epoch != local_workflow_schedule_epoch_when_dispatching_started:
                break
            
            print(f" [Broker]  Attempting to dispatch task {dispatchable_task} ")

            request = dispatchable_task.observation_request
            _best_satellite = dispatchable_task.observation_opportunity_satellite
            _best_constellation = self._known_satellites_by_constellation[_best_satellite]
            follow_up_action_failure = dispatchable_task.follow_up_action_failure
            follow_up_action_success = dispatchable_task.follow_up_action_success
            _best_pass = dispatchable_task.observation_opportunity_pass

            # Let's talk about constraints. The scheduler just checks that constraints are in place before something is scheduled.
            # For data constraints, it's start-after-end.
            # For bool constraints, it's _also_ start-after-end. 
            # Now that we handle rescheduling, we need to distinguish these two.
            # - For data constraint, keep things as is AND update child data from the parent, either when the parent is done or at scheduling time.
            # - For bool constraints, explicitly keep track of whether the task had a positive outcome, and only schedule if that is the case.
            # - For bool constraints, if a task dependency is violated, just don't even try to schedule it. Distinguish "we don't know" and "we know and it's false".

            # Set up callbacks so that, when a request comes in, it status is updated.
            # If a request is unscheduled:
            # - Update the requests table so we won't reuse the same requests (same as above)
            # - Set the request to not dispatched in the graph
            # - Call the scheduler again, which will come up with a new schedule, hopefully
            # If a request is scheduled:
            # - Update the requests table
            # - Update the workflow graph reflecting that the request is dispatched. This won't touch it again in scheduling
            # If a request is done:
            # - Update the requests table
            # - Update the workflow graph reflecting that the request is done. This won't touch it again in scheduling
            # - Update the parameters for the children that depend on that request's data product!

            def callback_request_scheduled(assigned_pass, _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation, __best_satellite=_best_satellite, _dispatchable_task=dispatchable_task):
                print(" [{}] confirmed scheduling of request {} from pass {}, constellation {}".format(self.name, _request, __best_pass, __best_constellation.name))
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'assigned_pass'] = assigned_pass
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = ObservationStatus.SCHEDULED
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'constellation'] = __best_constellation
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'satellite'] = __best_satellite
                _dispatchable_task.scheduled = True
                _dispatchable_task.dispatched = True
                # print(self._workflow_graph.nodes[_dispatchable_task_id])
                return
            
            def callback_request_unscheduled(reason, _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation, _dispatchable_task=dispatchable_task):
                print(" [{}] received UNscheduling of request {}, pass {}, from {}".format(self.name, request, __best_pass, __best_constellation.name))
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'assigned_pass'] = None
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = reason
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'constellation'] = None
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'satellite'] = None
                _dispatchable_task.scheduled = False
                _dispatchable_task.dispatched = False
                follow_up_action_failure(reason)
                # Recurse
                self.schedule_workflow(current_time=self.world.time, use_ilp=use_ilp, plot_schedule=plot_schedule)

                return
            
            def callback_request_ready(data_product,  _request=request, __best_pass=_best_pass, __best_constellation=_best_constellation, _dispatchable_task=dispatchable_task):
                print(" [{}: ] data ready for request {}, pass {}, from {}".format(self.name, _request, __best_pass, __best_constellation.name))
                self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'status'] = ObservationStatus.DATA_RECEIVED

                for _ix, __dp in self._requests.loc[((self._requests['request']==_request) & (self._requests['requested_pass']==__best_pass)), 'data_product'].items():
                    self._requests.loc[_ix, 'data_product'] = data_product
                follow_up_action_success(data_product)

                _dispatchable_task.scheduled = True
                _dispatchable_task.dispatched = True
                _dispatchable_task.completed = True

                
                
                # Update parameters. Phenomenon_processor is called on the observed events to write something to DataProducts
                # We need to pack what we will need (e.g., the location of follow-ups) in there.
                # We also need to compute successful_execution from the data_product
                # Finally, we need to update children parameters based on the data product
                # To that end, having a function that ingests the data product and produces an updated ObsRequest would be good
                _dispatchable_task.successful_execution = _dispatchable_task.success_declarer(data_product)
                
                # print(f" [Broker] Updating task {dispatchable_task_id}: {self._workflow_graph.nodes[dispatchable_task_id]}")

                for child_task_id in self._workflow_graph.successors(_dispatchable_task):
                    # If the constraint type is GEOMETRY
                    # Compute the new geometry for the child from the predecessor data_product
                    # Update the successor's geometry in the graph
                    outedges = self._workflow_graph.get_edge_data(_dispatchable_task, child_task_id)
                    for constraint_key, constraint in outedges.items():
                        if (constraint['constraint_class'] == ConstraintClass.GEOMETRY):
                            child_task_id.observation_request = constraint['parameters']['geometry_generator'](child_task_id.observation_request, data_product)

                # Recurse
                self.schedule_workflow(current_time=self.world.time, use_ilp=use_ilp, plot_schedule=plot_schedule)
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
                'status': ObservationStatus.SUBMITTED,
                'data_product': None,
                'scheduled_callback': callback_request_scheduled,
                'unscheduled_callback': callback_request_unscheduled,
                'ready_callback': callback_request_ready,
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
                phenomenon_processor=dispatchable_task.phenomenon_processor,
            )
            # Note that, if the call above fails, we will immediately receive a reply, via the callback, that will trigger another reschedule - while we are still dispatching things here!


def request_statistics(requests_pd):
    total_requests_no = len(requests_pd)
    all_statuses = set(requests_pd.status.values)    
    for s in all_statuses:
        # matching_statuses = sum([1 if (r['status']==s) else 0 for r in requests.values()])
        matching_statuses = len(requests_pd[requests_pd.status==s])
        print("{}/{} ({}%) of requests are in status {}".format(matching_statuses,total_requests_no, matching_statuses/total_requests_no*100, s))
    # X/Y requests have >1 successful observation
    
    requests_with_detections = requests_pd.apply(lambda x: x['data_product'] is not None and len(x['data_product'])>0, axis=1)

    if len(requests_pd[requests_pd.status=="OK! Data received"])>0:
        print("{}/{} ({}%) successful requests have a phenomenon detection".format(sum(requests_with_detections), len(requests_pd[requests_pd.status=="OK! Data received"]),sum(requests_with_detections)/len(requests_pd[requests_pd.status=="OK! Data received"])*100))
    
    unique_requests = set(requests_pd.request)
    unique_requests_no = len(unique_requests)
    fulfilled_unique_requests_no = 0
    fulfilled_unique_requests_events_found_no = 0
    for ur in unique_requests:
        matching_observation_statuses = requests_pd[(requests_pd['request']==ur) & (requests_pd['status']==ObservationStatus.DATA_RECEIVED)]
        if len(matching_observation_statuses):
            fulfilled_unique_requests_no += 1
        for _dp in matching_observation_statuses['data_product']:
            if _dp is not None and len(_dp):
                fulfilled_unique_requests_events_found_no += 1
                break
    print(" {}/{} ({}%) unique requests have at least one successful observation".format(fulfilled_unique_requests_no, unique_requests_no, fulfilled_unique_requests_no/unique_requests_no*100))
    # for _ix, _dp in requests_pd:
    # print(" of all requests have a phenomenon detection")
    # print(" of all _successful_ requests have a phenomenon detection")

    # print(" of all unique requests have a phenomenon detection")
    if fulfilled_unique_requests_no>0:
        print("{}/{} ({}%) of all successful unique requests have a phenomenon detection".format(fulfilled_unique_requests_events_found_no, fulfilled_unique_requests_no, fulfilled_unique_requests_events_found_no/fulfilled_unique_requests_no*100))