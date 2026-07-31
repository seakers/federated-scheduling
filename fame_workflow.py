
import numpy as np

import networkx as nx

from enum import Enum

from fame_agents_base import *

from matplotlib.pyplot import cm
import matplotlib.dates as mdates

from ortools.linear_solver import pywraplp

import bisect
import gurobipy as gp
import os
from astral import sun, Observer

import random


# # What is a workflow?

# We had described a workflow as:
# - A task or bag of tasks
# - whose completion spawns other tasks (implicitly creating both a START_AFTER_END constraint, and an information dependency).

# We need to handle two cases: (i) the next location is unknown and (ii) the next location is known, but there is a Boolean gate.
# Do we want more complex constraints? It would be nice to have a CONCURRENT constraint. Or START_AFTER_START.
# How about we stsart with the simple stuff?
# We will then implement ~~two~~ three algorithms for scheduling.
# - Priority heuristic
# - ILP
# - Network flow (as above) only for START_AFTER_END

# Let's start defining the format.

# A workflow is (i) a set of ObservationRequests with (ii) dependency constraints of the type START_AFTER_END, CONCURRENT, START_AFTER_START, (iii) Boolean constraints of the type START_IF_SUCCESSFUL, START_IF_FAILED (which imply START_AFTER_END dependency), and (iv) geometry constraints of the form DATA.
# We distinguish the Boolean and geometry constriants because the latter does not allow early scheduling.

# There is a recursive form of this. We will not use it initially, and use an interpreter to translate the unrolled form to the recursive form.

# A set of (ObservationRequest, Dependency (dict with other ObsRequest as key, and obs type as value), BooleanConstraint (dict with other ObsRequest as key, and obs type as value), GeometryConstraint (dict with other ObsRequest as key, and obs type as value))

# Note we are also implicitly defining an output format for observation requests' data products: these need to, at a minimum, return a boolean success, and (if required) geometric data information. That is really an execution constraint.

class ConstraintClass(Enum):
    TEMPORAL = 0
    SUCCESS = 1
    GEOMETRY = 2

class TemporalConstraintType(Enum):
    START_AFTER = 0 # Next task starts after this one
    START_AFTER_OFFSET = 1 # Next task starts after this one with an offset of H hours (which can be positive or negative)
    START_BEFORE = 2 # Next task starts before this one
    START_BEFORE_OFFSET = 3 # Next task starts before this one, with an offset of H hours

class SuccessConstraintType(Enum):
    START_IF_FAILED = 0
    START_IF_SUCCESSFUL = 1
    WAIT_FOR_COMPLETION_IF_FEASIBLE = 2

class GeometryConstraintType(Enum):
    LLA = 0

class Constraint():
    def __init__(self, constraint_class: ConstraintClass, constraint_type, parent: ObservationRequest, parameters: dict={'offset': 0, 'geometry_generator': lambda _obs_req, _data_product: _obs_req}):
        self.constraint_class = constraint_class
        self.constraint_type = constraint_type
        self.parent = parent
        self.parameters = parameters
    def __str__(self):
        return f"Constraint {self.constraint_class}: {self.constraint_type}. Parent {self.parent}. Params: {self.parameters}"
    def __repr__(self):
        return self.__str__()


class ConstrainedObservationRequest():
    def __init__(
            self,
            observation_request: ObservationRequest,
            task_constraints: list = [],
            timeline_constraints: list = [],
            timeline_impacts: list = [],
            is_mandatory: bool=False,
            schedule_policy_if_constraint_unsatisfied: dict={c: True for c in ConstraintClass},
            dispatch_policy_if_constraint_unsatisfied: dict={c: False for c in ConstraintClass},
            success_declarer=lambda data_product: True,
            follow_up_action_failure=lambda reason: None,
            follow_up_action_success=lambda data_product: None,
            phenomenon_processor=lambda o, s, p: p,
            name: str = "",
            max_num_instances: int=1,
            rewarder= lambda _observation: observation_quality(_observation),
            request_group: str = None
            ):
        """_summary_

        Args:
            observation_request (ObservationRequest): An observation request
            task_constraints (list, optional): a list of Constraints linking this task to other tasks. Defaults to [].
            timeline_constraints (list, optional): a list of TaskTimelineConstraint linking this task to other timelines. Defaults to [].
            timeline_impacts (list, optional): a list of TaskTimelineImpact linking this task to other timelines. Defaults to [].
            is_mandatory (bool, optional): Is the task mandatory? Only used by the ILP scheduler. Defaults to False.
            schedule_policy_if_constraint_unsatisfied (_type_, optional): do we schedule this task if some of the constraints are not yet resolved? Defaults to {c: True for c in ConstraintClass}.
            dispatch_policy_if_constraint_unsatisfied (_type_, optional): do we dispatch this task if some of the constraints are not yet resolved? Defaults to {c: False for c in ConstraintClass}.
            success_declarer (function, optional): did the observation succeed (from the perspective of tasks constrained on this). Defaults to lambda(data_product):True.
            follow_up_action_failure (function, optional): What to do when the observation fails to schedule. Defaults to lambda(reason): None.
            follow_up_action_success (function, optional): What to do when the observation returns successfully . Defaults to lambda(data_product:None.
            phenomenon_processor (function, optional): takes as input a list of phenomena and outputs a data product. Defaults to lambda (observation, spacecraft, phenomenon): phenomenon.
            name (str, optional): the name of the COR.
        """
        self.observation_request = observation_request
        self.task_constraints = task_constraints
        self.timeline_constraints = timeline_constraints
        self.timeline_impacts = timeline_impacts
        self.is_mandatory = is_mandatory
        self.schedule_policy = schedule_policy_if_constraint_unsatisfied
        self.dispatch_policy = dispatch_policy_if_constraint_unsatisfied
        self.success_declarer = success_declarer
        self.follow_up_action_failure = follow_up_action_failure
        self.follow_up_action_success= follow_up_action_success
        self.observation_opportunity: ObservationOpportunity = None
        self.observation_opportunity_pass: ObservationPass = None
        self.observation_opportunity_satellite: Satellite = None
        self.uplink_pass: ObservationPass = None
        self.downlink_pass: ObservationPass = None
        self.observation_opportunities = {}
        # All passes selected by the scheduler for this task (satellite, pass) pairs.
        # When max_num_instances > 1 the planner may pick multiple backup passes;
        # the broker dispatches all of them and waits for any one to succeed before
        # declaring the task complete and triggering children.
        self.pending_dispatch_passes: list = []
        self.scheduled: bool = False
        self.feasible: bool = True
        self.dispatched: bool = False
        self.completed: bool = False
        self.data_product: list = []
        self.successful_execution: bool = False
        self.phenomenon_processor = phenomenon_processor
        self.name = name
        if request_group is None:
            request_group = self.name
        self.request_group = request_group
        self.max_num_instances = max_num_instances
        self.rewarder = rewarder

    def __str__(self):
        return f"{self.name}"
    def __repr__(self):
        return self.__str__()
    

class ImpactType(Enum):
    ASSIGNMENT = 0 # Make the timeline this
    ADDITION = 1 # Add this to the initial value
    RATE_ADDITION = 2 # Add this to the rate at this time

class Impact():
    #Impact: It changes the rate of change or produces a suden change in the value of a variable. For example, the rate of change in the battery is different when
    #we are facing the sun and when we are in Eclipse, or when we turn on a transmitter/sensor. We have RATE_ADDITION and ADDITION/ASSIGNMENT
    def __init__(self, time: dt.datetime, type: ImpactType, value, owner: ConstrainedObservationRequest=None):
        self.time = time #When does this imapct happen?
        self.type = type #Is it a rate change, addition or assignment?
        self.value = value #Value of the rate change, addition or assignment
        self.owner = owner #This is the ConstrainedObservationRequest that caused this impact to happen in the first place
    def __str__(self):
        return f"Impact (type {self.type}) at {self.time} with value {self.value} owned by {self.owner}"
    def __repr__(self):
        return self.__str__()

class Timeline():
    #Timeline: It is a function that computes a value for a certain time given its last state and the series of imapcts.
    def __init__(self, name: str, initial_time: dt.datetime, initial_value: float, initial_rate: float, min_value: float=-np.inf, max_value: float=np.inf):
        self.name = name
        self.impact_container = [Impact(initial_time, ImpactType.ASSIGNMENT, initial_value), Impact(initial_time, ImpactType.RATE_ADDITION, initial_rate)]
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self):
        return f"Timeline {self.name} with {len(self.impact_container)} impacts"
    def __repr__(self):
        return self.__str__()

    def add_impact(self, impact: Impact):
        bisect.insort(self.impact_container, impact, key=lambda x: x.time)
    
    def _get_value_and_rate_at(self, time: dt.datetime, print_debug: bool = False):
        
        if print_debug:
            print(f"Getting value and rate through time {time}")
        # Select only the impacts that apply
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)

        if closest_index == 0:
            raise ValueError(f"Attempting to insert at time {time} which is before timeline start at {self.impact_container[0].time}")

        if print_debug:
            
            if closest_index<len(self.impact_container)-1: # There is a next index
                print(f"Insertion index for time {time} is {closest_index}. Time at index is {self.impact_container[closest_index].time} and next time is {self.impact_container[closest_index+1].time}")
            elif closest_index<len(self.impact_container): # There is an index
                print(f"Insertion index for time {time} is {closest_index}. Time at index is {self.impact_container[closest_index].time} and this is the last entry")
            else:
                print(f"Insertion index for time {time} is {closest_index}. This is beyond the last entry at {self.impact_container[-1].time}")

        # Accumulators
        _value = 0
        _rate = 0

        _previous_time = None
        assert (self.impact_container[0].type == ImpactType.ASSIGNMENT), "ERROR: The initial value in a timeline must be an assignment"
        

        for _impact in self.impact_container[:closest_index]:
            if print_debug:
                print(f"Adding impact {_impact} at time {_impact.time}")
            match _impact.type:
                case ImpactType.ASSIGNMENT:
                    _value = _impact.value
                    _previous_time = _impact.time
                case ImpactType.ADDITION:
                    # Propagate the rate
                    _dt = _impact.time-_previous_time
                    _integrated_rate = _rate*_dt.total_seconds()
                    _value += _integrated_rate
                    
                    # Now actually add the impact
                    _value += _impact.value

                    # Reset time
                    _previous_time = _impact.time
                    
                case ImpactType.RATE_ADDITION:
                    # Propagate the rate
                    _dt = _impact.time-_previous_time
                    _integrated_rate = _rate*_dt.total_seconds()
                    _value += _integrated_rate

                    # Now update the rate
                    _rate += _impact.value

                    # Reset time
                    _previous_time = _impact.time
            if print_debug:
                print(f"New value: {_value}, new rate {_rate}")

        # Bring to current time
        # Propagate the rate
        _dt = time-_previous_time
        _integrated_rate = _rate*_dt.total_seconds()
        _value += _integrated_rate

        if print_debug:
            print(f"Final value: {_value}, final rate {_rate}")

        return _value, _rate
    
    def get_value_at(self, time: dt.datetime, print_debug: bool = False):
        value, rate = self._get_value_and_rate_at(time, print_debug=print_debug)
        return value
    
    # def consolidate_impacts(self, time: dt.datetime):
    #     _value_at_time, _rate_at_time = self._get_value_and_rate_at(time)

    #     closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
    #     self.impact_container= self.impact_container[closest_index+1:]
    #     self.add_impact(Impact(time, ImpactType.ASSIGNMENT, _value_at_time))
    #     self.add_impact(Impact(time, ImpactType.RATE_ADDITION, _rate_at_time))

    def remove_impacts_from_owner(self, owner: ConstrainedObservationRequest):
        new_impact_container = [i for i in self.impact_container if i.owner != owner]
        self.impact_container = new_impact_container

    def reset_timeline(self, new_initial_time: dt.datetime, new_initial_value: float, new_initial_rate: float):
        self.impact_container = [
            Impact(new_initial_time, ImpactType.ASSIGNMENT, new_initial_value),
            Impact(new_initial_time, ImpactType.RATE_ADDITION, new_initial_rate)
        ]

class AssignmentTimeline(Timeline):
    def __init__(self, name: str, initial_time: dt.datetime, initial_value: bool):
        super().__init__(name, initial_time=initial_time, initial_value=initial_value, initial_rate=0.)

    def add_impact(self, impact: Impact):
        assert (impact.type == ImpactType.ASSIGNMENT), "ERROR: non-assignment impact on assignment timeline"
        bisect.insort(self.impact_container, impact, key=lambda x: x.time)
    
    def consolidate_impacts(self, time: dt.datetime):
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        if closest_index == 0:
            # Attempting to read at time {time} which is before timeline start
            return
        self.add_impact(Impact(time=time, type=ImpactType.ASSIGNMENT, value=self.impact_container[closest_index-1].value))
        self.impact_container= self.impact_container[closest_index:]
    
    def get_value_at(self, time: dt.datetime):
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        if closest_index == 0:
            raise ValueError(f"Attempting to read at time {time} which is before timeline start at {self.impact_container[0].time}")
        return self.impact_container[closest_index-1].value
    
# class AdditiveTimeline(Timeline):
#     def __init__(self, name: str, initial_time: dt.datetime, initial_value: float):
#         super().__init__(name, initial_time=initial_time, initial_value=initial_value)
    
#     def add_impact(self, impact: Impact):
#         assert (impact.type == ImpactType.ADDITION), "ERROR: non-additive impact on additive timeline"
#         bisect.insort(self.impact_container, impact, key=lambda x: x.time)

#     def get_value_at(self, time):
#         closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
#         return np.sum([_impact.value for _impact in self.impact_container[:closest_index+1]])
    
#     def consolidate_impacts(self, time):
#         _value_at_time = self.get_value_at(time)
#         closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
#         self.impact_container= self.impact_container[closest_index+1:]
#         assert (self.impact_container[0].time>=time), "ERROR: something went horribly wrong consolidating impacts"
#         self.impact_container.insert(0, Impact(time, ImpactType.ASSIGNMENT, _value_at_time))

class TaskImpactTime(Enum):
    PRE = 0 
    POST = 1

class TimelineConstraintType(Enum):
    GREATER_OR_EQUAL = 0 
    LESSER_OR_EQUAL = 1
    EQUAL = 2

class TaskTimelineImpact():
    def __init__(self, timeline: Timeline, time: TaskImpactTime, type: ImpactType, value):
        self.timeline = timeline
        self.time = time
        self.type = type
        self.value = value

class TaskTimelineConstraint():
    def __init__(self, timeline: Timeline, time: TaskImpactTime, type: TimelineConstraintType, value):
        self.timeline = timeline
        self.time = time
        self.type = type
        self.value = value

class Workflow():
    def __init__(
            self,
            constrained_observation_requests: list[ConstrainedObservationRequest],
            timelines: list[Timeline]=[],
            timeline_updater=lambda time, observations, timelines: timelines,
            request_updater=lambda time, observations, timelines: observations,
    ):
        """_summary_

        Args:
            constrained_observation_requests (list[ConstrainedObservationRequest]): a list of ConstrainedObservationRequest
            timelines (list[Timeline], optional): a list of Timelines that the CORs refer to. Defaults to [].
            timeline_updater (function(dt.datetime, list[ConstrainedObservationRequest], list[Timeline]): bool): a function that takes as inputs the current time, observation requests, and the timelines, and updates the timelines when replanning. The change is done in place. Defaults to not touching the timelines.
            observations_updater (function(dt.datetime, list[ConstrainedObservationRequest], list[Timeline]): bool): a function that takes as inputs the current time, observation requests, and the timelines, and updates the observation requests (e.g., retargeting in response to previous observations) when replanning. The change is done in place. Defaults to not touching the observations.
            """
        self.constrained_observation_requests = constrained_observation_requests        
        self.timelines = timelines
        self.timeline_updater = timeline_updater
        self.request_updater = request_updater


def build_workflow_graph(workflow: Workflow):
    # Build a dependency graph
    workflow_graph = nx.MultiDiGraph()
    timeline_graph = nx.MultiDiGraph()

    for constrained_request in workflow.constrained_observation_requests:
        # this_node = request
        workflow_graph.add_node(constrained_request, **constrained_request.__dict__) # Father forgive me for I have sinned against Python
        timeline_graph.add_node(constrained_request, **constrained_request.__dict__)

    for timeline in workflow.timelines:
        timeline_graph.add_node(timeline, **timeline.__dict__) # Father forgive me for I have sinned against Python


    for constrained_request in workflow.constrained_observation_requests:
        # this_node = request
        for constraint in constrained_request.task_constraints:
            workflow_graph.add_edge(
                constraint.parent,
                constrained_request,
                constraint_class=constraint.constraint_class,
                constraint_type=constraint.constraint_type,
                parameters=constraint.parameters,
                )
            
    for constrained_request in workflow.constrained_observation_requests:
        # this_node = request
        for tconstraint in constrained_request.timeline_constraints:
            timeline_graph.add_edge(
                constrained_request,
                tconstraint.timeline,
                edge_type=TaskTimelineConstraint,
                constraint_time=tconstraint.time,
                constraint_type=tconstraint.type,
                constraint_value=tconstraint.value,
                )
        for timpact in constrained_request.timeline_impacts:
            timeline_graph.add_edge(
                constrained_request,
                timpact.timeline,
                edge_type=TaskTimelineImpact,
                impact_time=timpact.time,
                impact_type=timpact.type,
                impact_value=timpact.value,
                )

    return workflow_graph, timeline_graph

# Idea:
# Dependency graph: build who depends on whom and ID the roots
# Then greedily walk down the dependency graph
# Input: a list of constrained observation requests
# Workflow:
# Build a dependency graph
# What if there are loops? <Break them>
# Walk through requests
# For each request, find opportunities with start and end time chosen appropriately
# For each opportunity, filter them according to constraints
# Pick the heuristically best opportunity
# Continue

def greedy_schedule_workflow(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list,
        feasibility_screener = lambda satellite, observation_pass: True,
        current_time: dt.datetime=None,
        verbose: int=99,
        receding_horizon_duration: dt.timedelta=dt.timedelta(weeks=52)
        ):
    
    # Build a dependency graph
    # requests_to_skip_data_not_ready = []

    # We exclude a few tasks. Completed tasks stay in the graph for the constraints, but we do not try to schedule them (see 38 lines below or so).
    # Tasks that have incomplete dependencies, _if_ we specify "don't schedule until dependencies are complete", are skipped.
    # This means we do not schedule them OR evaluate their downstream dependencies.

    # for parent_task, this_task, edge_id, constraint_data in workflow_graph.edges(keys=True, data=True):
    #     _this_task_data = workflow_graph.nodes[this_task]
    #     _constraint_class = constraint_data['constraint_class']
    #     if (
    #         (_this_task_data['schedule_policy'][_constraint_class] == False and
    #             (
    #                 workflow_graph.nodes[parent_task]['completed'] == False or 
    #                 workflow_graph.nodes[parent_task]['successful_execution'] == False
    #             )
    #         )
    #     ):
    #          requests_to_skip_data_not_ready.append(this_task)
    
    # workflow_graph.remove_nodes_from(requests_to_skip_data_not_ready)

    if verbose>2:
        print(f"   [Scheduler] WG: {workflow_graph}")

    # We will reschedule everything that is not dispatched. 
    for node_id in workflow_graph.nodes():
        if ((node_id.dispatched == False) and (node_id.completed == False)):
             node_id.scheduled = False
             node_id.feasible = True
            #  TODO: remove impacts for this task

    # Walk through requests
    # Start with the root nodes
    nodes_to_visit = [node for node, in_degree in workflow_graph.in_degree() if in_degree == 0]
    # TODO also dump the mandatory nodes here? Hmm.
    
    while len(nodes_to_visit):
        if verbose>2:
            print(f"   [Scheduler] Nodes to visit: {nodes_to_visit}")

        constrained_request = nodes_to_visit.pop(0)

        # Add the successors, so we will try to schedule them even if this one fails
        for child_request in workflow_graph.successors(constrained_request):
            if child_request.scheduled == False and child_request not in nodes_to_visit:
            # assert workflow_graph.nodes[child_request]['scheduled']==False, "ERROR: we are traversing the dependency graph in a strange and incorrect way (children)"
                nodes_to_visit.append(child_request)
        if verbose>2:
            print(f"   [Scheduler] Current request: {constrained_request}")
        # If the request has gone off, nothing we can do about it
        if ((constrained_request.dispatched == True) or (constrained_request.completed == True)):
            if verbose>1:
                print(f"   [Scheduler] Request {constrained_request} is already dispatched or completed, skipping")
            continue
        else:
            # Let's flush the impacts for this request.
            if constrained_request in timeline_graph.nodes():
                for _timeline in timeline_graph.successors(constrained_request):
                    _timeline.remove_impacts_from_owner(constrained_request)
            if verbose>4:
                print("   [Scheduler] This request is still in play, let's revisit it")

        # Check if constraints are resolvable. For some constraints (bool, geometry) we need the predecessor task
        # to be completed. If the task is not completed, and the schedule policy is `wait for the information`, we
        # will skip scheduling these tasks (but may still schedule their successors, ignoring the constraints).
        # If the task is completed, or if the shcedule_policy is `do not wait for information`, we just roll through
        # and, crucially, implicitly assume that the constraint will be resolved successfully, and that the default
        # geometric value should be used to identify opportunities.

        _constraints_are_resolvable = True
        for parent_request in workflow_graph.predecessors(constrained_request):
            inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
            for constraint_key, constraint in inedges.items():
                _constraint_class = constraint['constraint_class']
                if (
                    (
                        constrained_request.schedule_policy[_constraint_class] == False and
                        parent_request.completed == False
                    )
                ):
                    _constraints_are_resolvable = False
                    break
        if (_constraints_are_resolvable is False):
            if verbose>1:
                print("   [Scheduler] Request {} has unresolved predecessors and its policy require waiting; skipping.")
            continue


        # Now we start with temporal scheduling. We will identify a single interval that works.
        min_time = constrained_request.observation_request.min_time
        max_time = constrained_request.observation_request.max_time

        if (current_time is not None):
            min_time = max(min_time, current_time)
            max_time = min(max_time, current_time+receding_horizon_duration)

        if ((current_time is not None) and (max_time<current_time)):
            if verbose>1:
                print("   [Scheduler] Request {} max time is in the past; skipping.")
            constrained_request.scheduled=True
            constrained_request.feasible=False
            continue

        # First, we check if any CHILDREN have been scheduled and apply the constraints backwards.
        # If the dependency graph is not degenerate, the check three lines below should fail, and we should do nothing in this cycle. 
        if verbose>3:
            print(f"   [Scheduler] This request has {len(list(workflow_graph.successors(constrained_request)))} children: {list(workflow_graph.successors(constrained_request))}")
        for child_request in workflow_graph.successors(constrained_request):
            if ((child_request.scheduled==True) and (child_request.feasible==True)):
                outedges = workflow_graph.get_edge_data(constrained_request, child_request)
                if verbose>3:
                    print(f"   [Scheduler] Child: {child_request}")
                for constraint_key, constraint in outedges.items():
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                # These are constraints on the CHILD. So START_AFTER means the parent has to start before
                                case TemporalConstraintType.START_AFTER:
                                    max_time = min(max_time, child_request.observation_opportunity.time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    # Note the minus: we want parent_time + offset<child_time
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, child_request.observation_opportunity.time-offset)
                                case TemporalConstraintType.START_BEFORE:
                                    min_time = max(min_time, child_request.observation_opportunity.time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    # Note the - : we want child_time<parent_time+offset
                                    min_time = max(min_time, child_request.observation_opportunity.time-offset)
                        case ConstraintClass.SUCCESS:
                            # The child needs to know if the parent succeeded. So we constrain the parent to finish before the child
                            max_time = min(max_time, child_request.observation_opportunity.time)
                        case ConstraintClass.GEOMETRY:
                            max_time = min(max_time, child_request.observation_opportunity.time)
            else:
                if verbose>3:
                    print(f"Skipping task constraints for child {child_request}, currently unscheduled")

        # Check if any constraints are violated.
        # For temporal constraints that is auto-handled below.
        # Geometry constraints are handled externally. We could also use this to bring the data from the parent to the child.
        # Bool (success) constraints are checked here.
        #  When iterating over parents, if there is a success dependency, and the parent is reporting completed with failure, do not attempt to schedule.

        # Next, we check the PARENTS of a request. This is arguably the more interesting bit.
        if verbose>3:
            print(f"   [Scheduler] This request has {len(list(workflow_graph.predecessors(constrained_request)))} parents: {list(workflow_graph.predecessors(constrained_request))}")
        for parent_request in workflow_graph.predecessors(constrained_request):
            if ((parent_request.scheduled==True) and (parent_request.feasible==True)):
                inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
                if verbose>3:
                    print(f"   [Scheduler] Parent: {parent_request}")
                for constraint_key, constraint in inedges.items():
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                # These are constraints on the CURRENT node. So START_AFTER means the parent has to start before
                                case TemporalConstraintType.START_AFTER:
                                    min_time = max(min_time, parent_request.observation_opportunity.time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    min_time = max(min_time, parent_request.observation_opportunity.time+offset)
                                case TemporalConstraintType.START_BEFORE:
                                    max_time = min(max_time, parent_request.observation_opportunity.time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, parent_request.observation_opportunity.time+offset)
                        case ConstraintClass.SUCCESS:
                            # The current node needs to know if the parent succeeded. So we constrain the current node to start after the parent
                            min_time = max(min_time, parent_request.observation_opportunity.time)
                            if (parent_request.completed is True):
                                if (
                                    (
                                        (constraint['constraint_type'] == SuccessConstraintType.START_IF_FAILED) and 
                                        (parent_request.successful_execution == True)
                                        ) or (
                                        (constraint['constraint_type'] == SuccessConstraintType.START_IF_SUCCESSFUL) and 
                                        (parent_request.successful_execution == False)
                                        )
                                    ):
                                # If incompatible, skip
                                    print(f"   [Scheduler] This request is infeasible due to parent {parent_request}")
                                    constrained_request.scheduled=True
                                    constrained_request.feasible=False
                                    break
                                
                        case ConstraintClass.GEOMETRY:
                            min_time = max(min_time, parent_request.observation_opportunity.time)
            else:
                # REVIEW this is where we can exclude a successor if the parent is infeasible. 
                if (parent_request.feasible==False):
                    if verbose>3:
                        print(f"   [Scheduler] Skipping task constraints for parent {parent_request}, currently infeasible")
                #     constrained_request.scheduled=True
                #     constrained_request.feasible=False
                #     break
                else:
                    if verbose>3:
                        print(f"   [Scheduler] Skipping task constraints for parent {parent_request}, currently unscheduled")
        
        if (constrained_request.feasible==False):
            if verbose>1:
                print(f"   [Scheduler] Request {constrained_request} is infeasible")
            continue

        # Search for opportunities
        trimmed_request = ObservationRequest(
            lon_deg = constrained_request.observation_request.lon_deg,
            lat_deg = constrained_request.observation_request.lat_deg,
            min_time = min_time,
            max_time = max_time,
            alt_km = constrained_request.observation_request.alt_km,
            instrument = constrained_request.observation_request.instrument,
            request_name = constrained_request.observation_request.name+"_trimmed",
            min_elevation_deg = constrained_request.observation_request.min_elevation_deg,
        )
        observation_opportunities = find_observation_opportunities([trimmed_request,], satellites)
        constrained_request.observation_opportunities = observation_opportunities[trimmed_request]
        if verbose>2:
            print(f"Found opportunities from time {min_time} to {max_time}: {observation_opportunities}")
        if trimmed_request not in observation_opportunities.keys():
            raise ValueError("Could not schedule {}".format(constrained_request))
        passes = observation_opportunities[trimmed_request]
        if len(passes)==0:
            # Make a note of this in the graph
            constrained_request.scheduled=True
            constrained_request.feasible=False
            if verbose>0:
                print("   [Scheduler] Could not schedule {} (no passes)".format(constrained_request))
            continue
            # raise ValueError("Could not schedule {} (no passes)".format(constrained_request))
        
        # print(f"Opportunities: {passes}")
        _best_quality = - np.inf
        _best_satellite = None
        _best_pass = None
        allsatpasses = [(satellite, satpass, constrained_request.rewarder(satpass.highest)) for satellite, satpasses in passes.items() for satpass in satpasses]
        allsatpasses.sort(key=lambda x: x[2], reverse=True) # Sort by observation quality
        for (satellite, satpass, _quality) in allsatpasses:
            # Use an external check for feasibility
            if feasibility_screener(satellite, satpass):
                # Now also use an INTERNAL check for feasibility: do not try to clobber existing requests
                there_is_overlap = False
                for existing_request in workflow_graph:
                    if ((existing_request.scheduled == True) and (existing_request.feasible == True) and (existing_request.observation_opportunity is not None)):
                        # If the start time of the other opportunity is before the end of this one
                        # If the end time of the other opportunity is after the start of this one
                        # Then we overlap
                        if verbose>6:
                            print(f"Checking for overlap between {constrained_request} and {existing_request}")
                        if (
                            (existing_request.observation_opportunity.satellite == satellite) and
                            (existing_request.observation_opportunity.time<=satpass.highest.time+satpass.highest.duration) and 
                            (existing_request.observation_opportunity.time+existing_request.observation_opportunity.duration>satpass.highest.time)):
                            # Passes overlap
                            if verbose>6:
                                print("Overlap found! Continuing")
                            there_is_overlap = True
                            break
                        else:
                            if verbose>6:
                                print(f"Pass {satpass.highest} for {constrained_request} does not overlap with scheduled pass {existing_request.observation_opportunity} for {existing_request} ")
                if (not there_is_overlap):
                    _best_quality = _quality
                    _best_satellite = satellite
                    _best_pass = satpass
                    break
            # TODO check timeline constraints here
        # for satellite, satpasses in passes.items():
        #     for satpass in satpasses:
        #         if screen_pass_for_feasibility(existing_requests=None, satellite=satellite, _obs_pass=satpass, screen_against_comm_passes=True):
        #             _quality = observation_quality(satpass.highest)
        #             if _quality >= _best_quality:
        #                 _best_quality = _quality
        #                 _best_satellite = satellite
        #                 _best_pass = satpass
        if _best_pass is None:
            constrained_request.scheduled=True
            constrained_request.feasible=False
            if verbose>0:
                print("   [Scheduler] Could not schedule {} (all conflicts)".format(constrained_request))
            continue
            raise ValueError("Could not schedule {} (all conflicts)".format(constrained_request))

        # Pick the best opportunity
        constrained_request.observation_opportunity_pass = _best_pass
        constrained_request.observation_opportunity = _best_pass.highest
        constrained_request.observation_opportunity_satellite = _best_satellite
        constrained_request.scheduled = True
        # Apply timeline impacts
        if constrained_request in timeline_graph.nodes():
            for _timeline in timeline_graph.successors(constrained_request):
                tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                for impact_key, impact in tl_edges.items():
                    if impact['edge_type'] == TaskTimelineImpact:
                        _time = _best_pass.highest.time
                        if impact['impact_time'] == TaskImpactTime.POST:
                            _time = _best_pass.highest.time + _best_pass.highest.duration
                        tl_impact = Impact(
                            time=_time,
                            type=impact['impact_type'],
                            value=impact['impact_value'], #Now we replace the impact with the actual value
                            owner=constrained_request,
                            )
                        _timeline.add_impact(impact=tl_impact)


        if verbose>0:
            print(f"   [Scheduler] Scheduled request {constrained_request} on {_best_satellite} at {_best_pass.highest}")

    return workflow_graph


def random_schedule_workflow(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list,
        feasibility_screener = lambda satellite, observation_pass: True,
        current_time: dt.datetime=None,
        verbose: int=0,
        receding_horizon_duration: dt.timedelta=dt.timedelta(weeks=52),
        seed: int=None,
        ):
    """Random feasible scheduler — picks a uniformly random feasible pass per task.
    Produces valid (conflict-free, constraint-respecting) solutions with no quality
    optimisation, suitable as a lower-bound baseline."""

    rng = random.Random(seed)

    if verbose>2:
        print(f"   [RandomScheduler] WG: {workflow_graph}")

    for node_id in workflow_graph.nodes():
        if ((node_id.dispatched == False) and (node_id.completed == False)):
            node_id.scheduled = False
            node_id.feasible = True

    nodes_to_visit = [node for node, in_degree in workflow_graph.in_degree() if in_degree == 0]

    while len(nodes_to_visit):
        constrained_request = nodes_to_visit.pop(0)

        for child_request in workflow_graph.successors(constrained_request):
            if child_request.scheduled == False and child_request not in nodes_to_visit:
                nodes_to_visit.append(child_request)

        if ((constrained_request.dispatched == True) or (constrained_request.completed == True)):
            if verbose>1:
                print(f"   [RandomScheduler] Request {constrained_request} already dispatched/completed, skipping")
            continue
        else:
            if constrained_request in timeline_graph.nodes():
                for _timeline in timeline_graph.successors(constrained_request):
                    _timeline.remove_impacts_from_owner(constrained_request)

        _constraints_are_resolvable = True
        for parent_request in workflow_graph.predecessors(constrained_request):
            inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
            for constraint_key, constraint in inedges.items():
                _constraint_class = constraint['constraint_class']
                if (
                    constrained_request.schedule_policy[_constraint_class] == False and
                    parent_request.completed == False
                ):
                    _constraints_are_resolvable = False
                    break
        if not _constraints_are_resolvable:
            if verbose>1:
                print(f"   [RandomScheduler] Request {constrained_request} has unresolved predecessors; skipping")
            continue

        min_time = constrained_request.observation_request.min_time
        max_time = constrained_request.observation_request.max_time

        if current_time is not None:
            min_time = max(min_time, current_time)
            max_time = min(max_time, current_time + receding_horizon_duration)

        if (current_time is not None) and (max_time < current_time):
            constrained_request.scheduled = True
            constrained_request.feasible = False
            if verbose>1:
                print(f"   [RandomScheduler] Request {constrained_request} max time in the past; skipping")
            continue

        for child_request in workflow_graph.successors(constrained_request):
            if (child_request.scheduled == True) and (child_request.feasible == True):
                outedges = workflow_graph.get_edge_data(constrained_request, child_request)
                for constraint_key, constraint in outedges.items():
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                case TemporalConstraintType.START_AFTER:
                                    max_time = min(max_time, child_request.observation_opportunity.time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, child_request.observation_opportunity.time - offset)
                                case TemporalConstraintType.START_BEFORE:
                                    min_time = max(min_time, child_request.observation_opportunity.time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    min_time = max(min_time, child_request.observation_opportunity.time - offset)
                        case ConstraintClass.SUCCESS:
                            max_time = min(max_time, child_request.observation_opportunity.time)
                        case ConstraintClass.GEOMETRY:
                            max_time = min(max_time, child_request.observation_opportunity.time)

        for parent_request in workflow_graph.predecessors(constrained_request):
            if (parent_request.scheduled == True) and (parent_request.feasible == True):
                inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
                for constraint_key, constraint in inedges.items():
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                case TemporalConstraintType.START_AFTER:
                                    min_time = max(min_time, parent_request.observation_opportunity.time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    min_time = max(min_time, parent_request.observation_opportunity.time + offset)
                                case TemporalConstraintType.START_BEFORE:
                                    max_time = min(max_time, parent_request.observation_opportunity.time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, parent_request.observation_opportunity.time + offset)
                        case ConstraintClass.SUCCESS:
                            min_time = max(min_time, parent_request.observation_opportunity.time)
                            if parent_request.completed is True:
                                if (
                                    (constraint['constraint_type'] == SuccessConstraintType.START_IF_FAILED and
                                     parent_request.successful_execution == True) or
                                    (constraint['constraint_type'] == SuccessConstraintType.START_IF_SUCCESSFUL and
                                     parent_request.successful_execution == False)
                                ):
                                    constrained_request.scheduled = True
                                    constrained_request.feasible = False
                                    break
                        case ConstraintClass.GEOMETRY:
                            min_time = max(min_time, parent_request.observation_opportunity.time)

        if constrained_request.feasible == False:
            if verbose>1:
                print(f"   [RandomScheduler] Request {constrained_request} is infeasible")
            continue

        trimmed_request = ObservationRequest(
            lon_deg=constrained_request.observation_request.lon_deg,
            lat_deg=constrained_request.observation_request.lat_deg,
            min_time=min_time,
            max_time=max_time,
            alt_km=constrained_request.observation_request.alt_km,
            instrument=constrained_request.observation_request.instrument,
            request_name=constrained_request.observation_request.name + "_trimmed",
            min_elevation_deg=constrained_request.observation_request.min_elevation_deg,
        )
        observation_opportunities = find_observation_opportunities([trimmed_request,], satellites)
        constrained_request.observation_opportunities = observation_opportunities[trimmed_request]

        if trimmed_request not in observation_opportunities.keys():
            raise ValueError("Could not schedule {}".format(constrained_request))
        passes = observation_opportunities[trimmed_request]

        if len(passes) == 0:
            constrained_request.scheduled = True
            constrained_request.feasible = False
            if verbose>0:
                print(f"   [RandomScheduler] Could not schedule {constrained_request} (no passes)")
            continue

        # Collect all feasible (satellite, pass) pairs then pick one at random
        allsatpasses = [
            (satellite, satpass)
            for satellite, satpasses in passes.items()
            for satpass in satpasses
        ]
        rng.shuffle(allsatpasses)

        chosen_satellite = None
        chosen_pass = None
        for (satellite, satpass) in allsatpasses:
            if not feasibility_screener(satellite, satpass):
                continue
            there_is_overlap = False
            for existing_request in workflow_graph:
                if (existing_request.scheduled == True and
                        existing_request.feasible == True and
                        existing_request.observation_opportunity is not None):
                    if (
                        existing_request.observation_opportunity.satellite == satellite and
                        existing_request.observation_opportunity.time <= satpass.highest.time + satpass.highest.duration and
                        existing_request.observation_opportunity.time + existing_request.observation_opportunity.duration > satpass.highest.time
                    ):
                        there_is_overlap = True
                        break
            if not there_is_overlap:
                chosen_satellite = satellite
                chosen_pass = satpass
                break

        if chosen_pass is None:
            constrained_request.scheduled = True
            constrained_request.feasible = False
            if verbose>0:
                print(f"   [RandomScheduler] Could not schedule {constrained_request} (all conflicts)")
            continue

        constrained_request.observation_opportunity_pass = chosen_pass
        constrained_request.observation_opportunity = chosen_pass.highest
        constrained_request.observation_opportunity_satellite = chosen_satellite
        constrained_request.scheduled = True

        if constrained_request in timeline_graph.nodes():
            for _timeline in timeline_graph.successors(constrained_request):
                tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                for impact_key, impact in tl_edges.items():
                    if impact['edge_type'] == TaskTimelineImpact:
                        _time = chosen_pass.highest.time
                        if impact['impact_time'] == TaskImpactTime.POST:
                            _time = chosen_pass.highest.time + chosen_pass.highest.duration
                        tl_impact = Impact(
                            time=_time,
                            type=impact['impact_type'],
                            value=impact['impact_value'],
                            owner=constrained_request,
                        )
                        _timeline.add_impact(impact=tl_impact)

        if verbose>0:
            print(f"   [RandomScheduler] Scheduled {constrained_request} on {chosen_satellite} at {chosen_pass.highest}")

    return workflow_graph


def ilp_schedule_workflow(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list,
        feasibility_screener = lambda satellite, observation_pass: True,
        current_time: dt.datetime=None,
        verbose: int=0,
        max_solver_time_s: int=1e3,
        receding_horizon_duration: dt.timedelta=dt.timedelta(weeks=52),
        solver_engine: str = "GUROBI",  # <-- ADD THIS (Options: "SCIP" or "GUROBI")
        tax_rate: float = 0.0,  # Cost per scheduled obs as fraction of max quality. Set to 0 to disable.
        submission_cost_rate: float = 0.0,  # c_sub: unconditional per-booking submission overhead (as fraction of quality)
        execution_cost_rate: float = 0.0  # c_canc: conditional cancellation cost if accepted (as fraction of quality)
):
    workflow_graph_request=workflow_graph
    print(f"Function ILP scheduler, the solver engine is {solver_engine}")
    
    if verbose>2:
        print(f"   [Scheduler] WG: {workflow_graph}")

    # We will reschedule everything that is not dispatched. 
    for node in workflow_graph.nodes():
        if ((node.dispatched == False) and (node.completed == False)):
             node.scheduled = False

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('CPLEX_MIXED_INTEGER_PROGRAMMING')
    if not solver:
        print("CPLEX not found, trying SCIP")
        solver = pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
        if not solver:
            raise ValueError("Solver SCIP_MIXED_INTEGER_PROGRAMMING not found")
    
    solver.set_time_limit(int(max_solver_time_s*1e3))
    objective = solver.Objective()

    # for request in graph nodes
    # enumerate the possible opportunities
    # create a variable for each opportunity
    # for each constraint
    # for each pair of opportunities
    # if they do not comply with the constraint, exclude them through sum<1
    # if the predecessor is not scheduled, the successor should ALSO not be scheduled (sum(succ)<sum(pred))
    # if something is already dispatched or executed
    # set the relevant variable to 1
    # 
    solution_holder = {}
    flat_boolean_solution_holder = []
    solution_holder_by_satellite = {}

    # _timeline_holder = {}

    # Go find the overflights and create variables for them
    for constrained_request in workflow_graph.nodes():
        if verbose>3:
            print(f"   [Scheduler] Request: {constrained_request}")

        # If we wanted to not touch already-scheduled requests, this is where we would do it
        # If a request has already been dispatched or completed, skip
        if ((constrained_request.dispatched == True) or (constrained_request.completed == True)):
            if verbose>0:
                print(f"   [Scheduler] Request {constrained_request} is already dispatched or completed, skipping")
            if (constrained_request.completed == False):
                # If dispatched but not completed, we need to update the timeline
                if constrained_request in timeline_graph.nodes():
                    for _timeline in timeline_graph.successors(constrained_request):
                        if (verbose>5):
                            print(f"Adding impacts to timeline {_timeline} from in-progress task {constrained_request}")
                        tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                        for impact_key, impact in tl_edges.items():
                            if impact['edge_type'] == TaskTimelineImpact and impact['impact_time'] == TaskImpactTime.POST:
                                _time = constrained_request.observation_opportunity.time + constrained_request.observation_opportunity.duration
                                tl_impact = Impact(
                                    time=_time,
                                    type=impact['impact_type'],
                                    value=impact['impact_value'],
                                    owner=constrained_request,
                                    )

                                # print(f"   [Scheduler]: In-progress task {constrained_request} applies impact {tl_impact} to timeline {_timeline}")

                                _timeline.add_impact(impact=tl_impact)
                                # if verbose>5:
                                #     print("Added impact to timeline!")
            continue

        solution_holder[constrained_request] = {}

        # Do not look for overflights in the past (if you know the time)
        trimmed_request_min_time = constrained_request.observation_request.min_time
        if current_time is not None and current_time>constrained_request.observation_request.min_time:
            trimmed_request_min_time = current_time
        trimmed_request_max_time = min(constrained_request.observation_request.max_time, trimmed_request_min_time+receding_horizon_duration)
        
        trimmed_request = ObservationRequest(
            lon_deg = constrained_request.observation_request.lon_deg,
            lat_deg = constrained_request.observation_request.lat_deg,
            min_time = trimmed_request_min_time,
            max_time = trimmed_request_max_time,
            alt_km = constrained_request.observation_request.alt_km,
            instrument = constrained_request.observation_request.instrument,
            request_name = constrained_request.observation_request.name+"_trimmed",
            min_elevation_deg = constrained_request.observation_request.min_elevation_deg,
        )
        # Find the overflights
        observation_opportunities = find_observation_opportunities([trimmed_request,], satellites)
        
        # Stuff the overflights in the graph
        constrained_request.observation_opportunities = observation_opportunities[trimmed_request]
        if trimmed_request not in observation_opportunities.keys():
            raise ValueError("Could not schedule {}".format(constrained_request))
        passes = observation_opportunities[trimmed_request]

        if len(passes)==0:
            constrained_request.scheduled=True
            constrained_request.feasible=False
            if verbose>1:
                print("   [Scheduler] Could not schedule {} (no passes)".format(constrained_request))
            continue
        
        # Now go create the decision variables. As you are at it, also add impacts for these decision variables
        _found_a_pass = False
        allsatpasses = [(satellite, satpass, constrained_request.rewarder(satpass.highest)) for satellite, satpasses in passes.items() for satpass in satpasses]
        allsatpasses.sort(key=lambda x: x[2], reverse=True) # Sort by observation quality

        # Compute max quality across all feasible passes for this request, used to compute tax
        _max_quality_for_request = max([q for (_, _, q) in allsatpasses]) if len(allsatpasses) > 0 else 0.0
        _tax = tax_rate * _max_quality_for_request
        _submission_cost = submission_cost_rate* _max_quality_for_request
        _execution_cost = execution_cost_rate*_max_quality_for_request
    

        for (satellite, satpass, _quality) in allsatpasses:
            if feasibility_screener(satellite, satpass):
                _found_a_pass = True

                if satellite not in solution_holder[constrained_request].keys():
                    solution_holder[constrained_request][satellite] = {}
                if satellite not in solution_holder_by_satellite.keys():
                    solution_holder_by_satellite[satellite] = []

                solution_holder[constrained_request][satellite][satpass] = solver.BoolVar(f"{constrained_request}_{satellite}_{satpass}")

                solution_holder_by_satellite[satellite].append((satpass, solution_holder[constrained_request][satellite][satpass]))

                flat_boolean_solution_holder.append(solution_holder[constrained_request][satellite][satpass])

                # Net coefficient = quality - tax (tax=0 disables the cost penalty)
                objective.SetCoefficient(solution_holder[constrained_request][satellite][satpass], _quality - _submission_cost-_execution_cost)

                # TODO for each timeline impacted, add a variable for that timeline value at that time. Add an Impact with that timeline value times "do we do it".
                if constrained_request in timeline_graph.nodes():
                    for _timeline in timeline_graph.successors(constrained_request):
                        if (verbose>5):
                            print(f"Considering timeline {_timeline} ")
                        tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                        for impact_key, impact in tl_edges.items():
                            # if verbose>5:
                            #     print(impact)
                            if impact['edge_type'] == TaskTimelineImpact:
                                _time = satpass.highest.time
                                if impact['impact_time'] == TaskImpactTime.POST:
                                    _time = satpass.highest.time + satpass.highest.duration
                                else:
                                    assert impact['impact_time'] == TaskImpactTime.PRE, "ERROR: Impact time not recognized"
                                tl_impact = Impact(
                                    time=_time,
                                    type=impact['impact_type'],
                                    value=impact['impact_value']*solution_holder[constrained_request][satellite][satpass], # This is the magic: the impact now depends on a ILP variable
                                    owner=constrained_request,
                                    )
                                _timeline.add_impact(impact=tl_impact)

                            
                # For each constraint, we will invoke get_value_at on the timeline and constrain the outcome. Except! You need to do this AFTER all the impacts have 
                #  been tabulated. YOu will need a follow-up pass. See below for that pass
                if (_quality<0):
                    if verbose>0:
                        print(f"   [Scheduler] Negative quality {_quality} for pass {satpass}")
            else:
                if verbose>3:
                    print(f"   [Scheduler] Skipping pass {satpass} as infeasible")

        if _found_a_pass is False:
            constrained_request.scheduled=True
            constrained_request.feasible=False
            if verbose>1:
                print("   [Scheduler] Could not schedule {} (all conflicts)".format(constrained_request))

        # At most max_num_instances observation per request are assigned
        # solver.Add(sum([solution_holder[constrained_request][_satellite][_satpass] for _satellite, _satpasses in solution_holder[constrained_request].items() for _satpass in _satpasses.keys()]) <= constrained_request.max_num_instances)
        solver.Add(sum([solution_holder[constrained_request][_satellite][_satpass] for _satellite, _satpasses in solution_holder[constrained_request].items() for _satpass in _satpasses.keys()]) <= 1) #We hardcoded to 1 because otherwise ILP just books everything

        if verbose>3:
            print(f"   [Scheduler] This request has {len(list(workflow_graph.predecessors(constrained_request)))} parents: {list(workflow_graph.predecessors(constrained_request))}")
        for parent_request in workflow_graph.predecessors(constrained_request):
            inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
            if verbose>3:
                print(f"   [Scheduler] Parent: {parent_request}")
            for constraint_key, constraint in inedges.items():
                match constraint['constraint_class']:
                    case ConstraintClass.TEMPORAL:
                        match constraint['constraint_type']:
                            # These are constraints on the CURRENT node. So START_AFTER means the parent has to start before
                            case TemporalConstraintType.START_AFTER:
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        # If the parent is also being scheduled
                                        if parent_request in solution_holder.keys():
                                            for parent_satellite in solution_holder[parent_request].keys():
                                                for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                    # If the parent request starts after the current request, then they can't be true at the same time
                                                    if parent_pass.highest.time>this_pass.highest.time:
                                                        solver.Add(this_decision_variable + parent_decision_variable <= 1)
                                        elif ((parent_request.dispatched == True) or (parent_request.completed == True)):
                                            # This is already off, so we constrain with respect to what we scheduled.
                                            # Which is held in constrained_request.observation_opportunity
                                            if parent_request.observation_opportunity.time>=this_pass.highest.time:
                                                solver.Add(this_decision_variable == 0)
                                        else:
                                            print(f"    [Scheduler] Ignoring parent {parent_request}, that's odd")

                                # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                            case TemporalConstraintType.START_AFTER_OFFSET:
                                offset = constraint['parameters']['offset']
                                # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        # If the parent is also being scheduled
                                        if parent_request in solution_holder.keys():
                                            for parent_satellite in solution_holder[parent_request].keys():
                                                for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                    # If the parent request starts after the current request, then they can't be true at the same time
                                                    if parent_pass.highest.time+offset>this_pass.highest.time:
                                                        solver.Add(this_decision_variable + parent_decision_variable <= 1)
                                        elif ((parent_request.dispatched == True) or (parent_request.completed == True)):
                                            # This is already off, so we constrain with respect to what we scheduled.
                                            if parent_request.observation_opportunity.time+offset>this_pass.highest.time:
                                                solver.Add(this_decision_variable == 0)
                                        else:
                                            print(f"    [Scheduler] Ignoring parent {parent_request}, that's odd")
                                            
                            case TemporalConstraintType.START_BEFORE:
                                # max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        if parent_request in solution_holder.keys():
                                            for parent_satellite in solution_holder[parent_request].keys():
                                                for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                    # If the parent request starts after the current request, then they can't be true at the same time
                                                    if parent_pass.highest.time<this_pass.highest.time:
                                                        solver.Add(this_decision_variable + parent_decision_variable <= 1)
                                        elif ((parent_request.dispatched == True) or (parent_request.completed == True)):
                                            # This is already off, so we constrain with respect to what we scheduled.
                                            if parent_request.observation_opportunity.time<this_pass.highest.time:
                                                solver.Add(this_decision_variable == 0)
                                        else:
                                            print(f"    [Scheduler] Ignoring parent {parent_request}, that's odd")

                            case TemporalConstraintType.START_BEFORE_OFFSET:
                                offset = constraint['parameters']['offset']
                                # max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        if parent_request in solution_holder.keys():
                                            for parent_satellite in solution_holder[parent_request].keys():
                                                for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                    # If the parent request starts after the current request, then they can't be true at the same time
                                                    if parent_pass.highest.time+offset<this_pass.highest.time:
                                                        solver.Add(this_decision_variable + parent_decision_variable <= 1)
                                        elif ((parent_request.dispatched == True) or (parent_request.completed == True)):
                                            # This is already off, so we constrain with respect to what we scheduled.
                                            if parent_request.observation_opportunity.time+offset<this_pass.highest.time:
                                                solver.Add(this_decision_variable == 0)
                                        else:
                                            print(f"    [Scheduler] Ignoring parent {parent_request}, that's odd")

                    case ConstraintClass.SUCCESS:
                        if (parent_request.completed == True):
                            if (
                                (
                                    (constraint['constraint_type'] == SuccessConstraintType.START_IF_FAILED) and 
                                    (parent_request.successful_execution == True)
                                    ) or (
                                    (constraint['constraint_type'] == SuccessConstraintType.START_IF_SUCCESSFUL) and 
                                    (parent_request.successful_execution == False)
                                    )
                                ):
                            # If incompatible, skip
                                if (verbose>0):
                                    print(f"    [Scheduler] Request {constrained_request} skipped: SUCCESS constraint on parent {parent_request} not met")
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        solver.Add(this_decision_variable == 0)
                                constrained_request.scheduled=True
                                constrained_request.feasible=False
                                break

                        # The current node needs to know if the parent succeeded. So we constrain the current node to start after the parent
                        # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                        for this_satellite in solution_holder[constrained_request].keys():
                            for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                if parent_request in solution_holder.keys():
                                    for parent_satellite in solution_holder[parent_request].keys():
                                        for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                            # If the parent request starts after the current request, then they can't be true at the same time
                                            if parent_pass.highest.time>this_pass.highest.time:
                                                solver.Add(this_decision_variable + parent_decision_variable <= 1)
                                elif (parent_request.dispatched == True or parent_request.completed == True): # Dispatched
                                    if parent_request.observation_opportunity.time>this_pass.highest.time:
                                        solver.Add(this_decision_variable == 0)
                                else:
                                    print(f"    [Scheduler] Ignoring parent {parent_request} (SUCCESS constraint), that's odd")

                            
                    case ConstraintClass.GEOMETRY:
                        # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                        for this_satellite in solution_holder[constrained_request].keys():
                            for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                if parent_request in solution_holder.keys():
                                    for parent_satellite in solution_holder[parent_request].keys():
                                        for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                            # If the parent request starts after the current request, then they can't be true at the same time
                                            if parent_pass.highest.time>this_pass.highest.time:
                                                solver.Add(this_decision_variable + parent_decision_variable <= 1)
                                elif ((parent_request.dispatched == True) or (parent_request.completed == True)):
                                    # This is already off, so we constrain with respect to what we scheduled.
                                    if parent_request.observation_opportunity.time>this_pass.highest.time:
                                        solver.Add(this_decision_variable == 0)
                                else:
                                    print(f"    [Scheduler] Ignoring parent {parent_request} (GEOMETRY), that's odd")

        if constrained_request.is_mandatory is True:
            solver.Add(sum([solution_holder[constrained_request][_satellite][_satpass] for _satellite, _satpasses in solution_holder[constrained_request].items() for _satpass in _satpasses]) >= 1)

    # Second pass for timeline constraints
    for constrained_request in solution_holder.keys():
        for satellite in solution_holder[constrained_request].keys():
            for satpass in solution_holder[constrained_request][satellite].keys():
                this_decision_variable = solution_holder[constrained_request][satellite][satpass]
                if feasibility_screener(satellite, satpass):

                    # For each constraint, invoke get_value_at on the timeline and constrain the outcome. You need to do this AFTER all the impacts have 
                    #  been tabulated. This is the follow-up pass
                    if constrained_request in timeline_graph.nodes():
                        for _timeline in timeline_graph.successors(constrained_request):
                            if (verbose>5):
                                print(f"Considering timeline {_timeline} for constraints")
                            tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                            for _constraint_key, _constraint in tl_edges.items():
                                if _constraint['edge_type'] == TaskTimelineConstraint: #could be an impact, which we ignore
                                    _time = satpass.highest.time
                                    if _constraint['constraint_time'] == TaskImpactTime.POST:
                                        _time = satpass.highest.time + satpass.highest.duration
                                    else:
                                        assert _constraint['constraint_time'] == TaskImpactTime.PRE, "ERROR: Constraint time not recognized" # Either PRE or POST

                                    timeline_value_at = _timeline.get_value_at(_time) # This will be a symbolic expression of the variables

                                    __name = f"{_timeline}_{_time}_{constrained_request}_{satellite}_{satpass}"

                                    # if _timeline not in _timeline_holder.keys():
                                    #     _timeline_holder[_timeline] = {}
                                    # if _time not in _timeline_holder[_timeline].keys():
                                    #     _timeline_holder[_timeline][_time] = {}

                                    # _timeline_holder[_timeline][_time][constrained_request] = solver.NumVar(name=__name, lb=_timeline.min_value, ub=_timeline.max_value)

                                    # solver.Add(timeline_value_at == _timeline_holder[_timeline][_time][constrained_request])

                                    match _constraint['constraint_type']:
                                        case TimelineConstraintType.GREATER_OR_EQUAL:
                                            if not np.isfinite(_timeline.min_value):
                                                raise ValueError(f"You don't want to use a big-M method with no clear bounds. Specify a lower bound on timeline {_timeline}")
                                            solver.Add(timeline_value_at >= _constraint['constraint_value']*this_decision_variable+_timeline.min_value*(1-this_decision_variable)) # Only enforce the constraint if this decision variable is active!
                                        case TimelineConstraintType.LESSER_OR_EQUAL:
                                            if not np.isfinite(_timeline.max_value):
                                                raise ValueError(f"You don't want to use a big-M method with no clear bounds. Specify an upper bound on timeline {_timeline}")
                                            solver.Add(timeline_value_at <= _constraint['constraint_value']*this_decision_variable+_timeline.max_value*(1-this_decision_variable))
                                        case TimelineConstraintType.EQUAL:
                                            # Do both < and >
                                            if not np.isfinite(_timeline.min_value):
                                                raise ValueError(f"You don't want to use a big-M method with no clear bounds. Specify a lower bound on timeline {_timeline}")
                                            if not np.isfinite(_timeline.max_value):
                                                raise ValueError(f"You don't want to use a big-M method with no clear bounds. Specify an upper bound on timeline {_timeline}")
                                            solver.Add(timeline_value_at >= _constraint['constraint_value']*this_decision_variable+_timeline.min_value*(1-this_decision_variable))
                                            solver.Add(timeline_value_at <= _constraint['constraint_value']*this_decision_variable+_timeline.max_value*(1-this_decision_variable))

                                    

                                    if verbose>5:
                                        print("Added constraint to timeline!")

    # Only one assignment per request: see above

    # On the same vehicle, no overlapping requests (not considering comms at this stage)
    # - Conflicts on the same machine: iterate over sat, iterate over tasks
    for _sat in solution_holder_by_satellite.keys():
        solution_holder_by_satellite[_sat].sort(key=lambda x: x[0].highest.time)
        for _this_opportunity_ix, _opportunity_tuple in enumerate(solution_holder_by_satellite[_sat]):
            _this_opportunity = _opportunity_tuple[0]
            _this_decision_variable = _opportunity_tuple[1]
            for _next_opportunity_tuple in solution_holder_by_satellite[_sat][_this_opportunity_ix+1:]:
                _next_opportunity = _next_opportunity_tuple[0]
                _next_decision_variable = _next_opportunity_tuple[1]
                # If the two overlap
                if _next_opportunity.highest.time<_this_opportunity.highest.time+_this_opportunity.highest.duration:
                    solver.Add(_this_decision_variable + _next_decision_variable <= 1)
                else:
                    # The opportunities are ordered so, if we are past the conflict, we can move on to the next set of opportunities
                    break

    # TODO:
    # - Add conflicts with already-dispatched but not-yet-completed requests - this should be taken care of by the check-feasibility thing
    # - Add communication uplinks if ISL is not guaranteed: 
    #   - Add uplinks for sat i
    # If no ISL for sat i and we schedule an observation on sat I, we also schedule an uplink on it
    # Think more about how to represent the MSA workflow accepting that tasks can be scheduled if their parent is not

    objective.SetMaximization()

    # Add a hint
    solver.SetHint(flat_boolean_solution_holder, [0.,]*len(flat_boolean_solution_holder))

    # ==========================================
    # THE SOLVER SWITCH (MPS TRICK)
    # ==========================================
    if solver_engine == "GUROBI":

        if solver.NumVariables() == 0:
            if verbose > 0:
                print("    [Scheduler] No pending tasks to schedule. Skipping Gurobi.")
            # Mimic SCIP returning optimal for an empty problem
            status = pywraplp.Solver.OPTIMAL 
            m = None 
            gurobi_var_map = {}
        else:
            if verbose > 0:
                print("    [Scheduler] Exporting to MPS and solving with Native Gurobi...")
            mps_path = "temp_workflow.mps"
            solver.WriteModelToMpsFile(mps_path, False, False)
            
            # --- THE FIX: Wait for OS to finish writing file ---
            import time
            print("    [Scheduler] Waiting for OS I/O to finish writing file...")
            for _ in range(20): 
                try:
                    with open(mps_path, 'rb') as f:
                        f.seek(-30, os.SEEK_END) 
                        tail = f.read().decode('utf-8', errors='ignore')
                        if "ENDATA" in tail:
                            break 
                except Exception:
                    pass
                time.sleep(0.5) 
            print("    [Scheduler] MPS file flush confirmed!")
            
            # Initialize variables before try block
            env = None
            m = None
            try:
                # --- Authenticate and start Gurobi Env ---
                env = gp.Env(empty=True)
                env.setParam('OutputFlag', 0) 
                env.setParam('MIPGap', 0.01)  
                env.setParam('OutputFlag', 1)
                
                if os.environ.get("WLSACCESSID"):
                    env.setParam("WLSACCESSID", os.environ.get("WLSACCESSID"))
                    env.setParam("WLSSECRET", os.environ.get("WLSSECRET"))
                    env.setParam("LICENSEID", int(os.environ.get("LICENSEID", 0)))
                env.start()
                
                # Load model & solve
                m = gp.read(mps_path, env=env)
                m.optimize()

                # Map Gurobi status back to OR-Tools format
                if m.Status == gp.GRB.OPTIMAL:
                    status = pywraplp.Solver.OPTIMAL
                elif m.Status in [gp.GRB.TIME_LIMIT, gp.GRB.SOLUTION_LIMIT, gp.GRB.INTERRUPTED]:
                    status = pywraplp.Solver.FEASIBLE
                elif m.Status == gp.GRB.INFEASIBLE:
                    if verbose > 0:
                        print(f"    [Scheduler] Model is INFEASIBLE (no valid schedule found)")
                    status = pywraplp.Solver.INFEASIBLE
                elif m.Status == gp.GRB.INF_OR_UNBD:
                    if verbose > 0:
                        print(f"    [Scheduler] Model is INFEASIBLE or UNBOUNDED")
                    status = pywraplp.Solver.ABNORMAL
                else:
                    if verbose > 0:
                        print(f"    [Scheduler] Gurobi status: {m.Status}")
                    status = pywraplp.Solver.NOT_SOLVED

                # Extract variable results
                gurobi_var_map = {}
                if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                    ort_vars = solver.variables()
                    grb_vars = m.getVars()
                    if len(ort_vars) == len(grb_vars):
                        gurobi_var_map = {ort_vars[i].name(): grb_vars[i].X for i in range(len(ort_vars))}
                    else:
                        print("    [Scheduler] WARNING: Variable counts do not match between OR-Tools and Gurobi!")
                                # Safe check: if m is None, it took 0.0 seconds
                gurobi_time = m.Runtime if m is not None else 0.0
                print(f"    [Scheduler] Problem solved with Native Gurobi (RunTime: {gurobi_time:.2f}s)")

            finally:
                # Safely dispose C++ objects only if they were initialized
                if m is not None:
                    m.dispose()
                if env is not None:
                    env.dispose()
    else:
        solver.WriteModelToMpsFile("isolated_benchmark_problem.mps", False, False)
        print(" Isolated problem snapshot saved to file!")
        if verbose > 0:
            print("    [Scheduler] Solving with standard OR-Tools SCIP...")
        status = solver.Solve()
    # ==========================================
    # RECONSTRUCT THE SOLUTION
    # ==========================================
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # if (verbose>0):
        #         if solver_engine == "GUROBI":
        #             # Safe check: if m is None (empty problem), objective is 0.0
        #             obj_val = m.ObjVal if m is not None else 0.0
        #         else:
        #             obj_val = solver.Objective().Value()
                    
        for constrained_request in workflow_graph.nodes():
            if ((constrained_request.dispatched == True) or (constrained_request.completed == True)):
                continue
            else:
                constrained_request.scheduled=True
                constrained_request.pending_dispatch_passes = []  # reset for this scheduling epoch
                _feasible = False
                for this_satellite in solution_holder[constrained_request].keys():
                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():

                        # --- THE VALUE LOOKUP SWITCH ---
                        if solver_engine == "GUROBI":
                            # Look up the value from the Gurobi map we made
                            var_value = gurobi_var_map.get(this_decision_variable.name(), 0.0)
                        else:
                            # Use normal OR-Tools method
                            var_value = this_decision_variable.solution_value()
                        # -------------------------------

                        if var_value > 0.5: # Use 0.5 to be safe with float rounding
                            if verbose>1:
                                print("    [Scheduler] ", this_decision_variable.name(), " = ", var_value)
                            _feasible = True
                            # Store the primary (first/best) pass in the legacy slot for
                            # compatibility with timeline constraints and display code.
                            if not constrained_request.pending_dispatch_passes:
                                constrained_request.observation_opportunity_pass = this_pass
                                constrained_request.observation_opportunity = this_pass.highest
                                constrained_request.observation_opportunity_satellite = this_satellite
                            # Accumulate ALL selected passes so the broker can dispatch
                            # all of them as backups (sorted chronologically below).
                            constrained_request.pending_dispatch_passes.append((this_satellite, this_pass))

                            # Now apply the relevant impacts
                            if constrained_request in timeline_graph.nodes():
                                for _timeline in timeline_graph.successors(constrained_request):
                                    tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                                    for impact_key, impact in tl_edges.items():
                                        if impact['edge_type'] == TaskTimelineImpact:
                                            _time = this_pass.highest.time
                                            if impact['impact_time'] == TaskImpactTime.POST:
                                                _time = this_pass.highest.time + this_pass.highest.duration
                                            tl_impact = Impact(
                                                time=_time,
                                                type=impact['impact_type'],
                                                value=impact['impact_value']*var_value, # Use the dynamic var_value
                                                owner=constrained_request,
                                                )
                                            _timeline.add_impact(impact=tl_impact)

                # Sort backup passes chronologically so the earliest is dispatched first.
                constrained_request.pending_dispatch_passes.sort(key=lambda sp: sp[1].highest.time)
                constrained_request.feasible=_feasible

        # Clean up timeline impacts - remove unpicklable solver objects (OR-Tools and Gurobi)
        for timeline in timeline_graph.nodes():
            if isinstance(timeline, Timeline):
                _new_impact_container = []
                for impact in timeline.impact_container:
                    try:
                        # Keep only realized numerical values (floats/ints)
                        if isinstance(impact.value, (int, float, np.number, bool)):
                            _new_impact_container.append(impact)
                    except Exception:
                        # Dead or freed C++ solver pointer — safely discard it
                        pass
                timeline.impact_container = _new_impact_container

        if (verbose>2):
            print()
            if solver_engine == "SCIP":
                print(f"    [Scheduler] Problem solved in {solver.wall_time():d} milliseconds")
                print(f"    [Scheduler] Problem solved in {solver.iterations():d} iterations")
                print(f"    [Scheduler] Problem solved in {solver.nodes():d} branch-and-bound nodes")
    else:
        # (Keep your existing 'else' print block here for NOT_SOLVED statuses)
        if verbose>0:
            print(f"    [Scheduler] We have not found a feasible solution.")

    return workflow_graph



def plot_workflow_schedule(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph=nx.MultiDiGraph(),
        axes=None,
        time: dt.datetime = None,
        feasibility_screener = lambda satellite, observation_pass: True,
        plot_title = "",
        show_night: bool=False,
        show_night_location: Location= Location(-118,34, 0),
        save_schedule_plot: bool=False,
        save_name: str = "Schedule.pdf",
        ):


    # num_requests = len(workflow_graph)
    # request_names = list(workflow_graph.nodes())

    request_group_names = list(dict.fromkeys([r.request_group for r in workflow_graph.nodes()]))
    num_request_groups = len(request_group_names)
    request_group_sizes = {g: len([r.request_group for r in workflow_graph.nodes() if r.request_group==g]) for g in request_group_names}

    # request_colors_list = cm.rainbow(np.linspace(0, 1, num_requests))
    # request_colors = {task: request_colors_list[task_ix] for task_ix, task in enumerate(request_names)}

    request_group_colors_list = cm.rainbow(np.linspace(0, 1, num_request_groups))
    request_group_colors = {group: request_group_colors_list[group_ix] for group_ix, group in enumerate(request_group_names)}

    num_timelines = len([n for n in timeline_graph.nodes() if type(n) == Timeline])


    if axes is None:
        height_ratios = [num_request_groups]
        height_ratios.extend([1,]*num_timelines)
        fig, axes = plt.subplots(num_timelines+1,1, sharex=True, height_ratios=height_ratios, figsize=(12, int(math.ceil(.2*num_request_groups+num_timelines))))
    else: 
        if ((type(axes)==list) or (type(axes) == np.ndarray)):
            for ax in axes:
                ax.clear()
        else:
            axes.clear()


    if num_timelines>0:
        if ((type(axes)==list) or (type(axes) == np.ndarray)):
            ax_tasks = axes[0]
            if len(axes)>1:
                ax_timelines = axes[1:]
            else:
                ax_timelines = None
        else:
            ax_tasks = axes
            ax_timelines = None

    else:
        if ((type(axes)==list) or (type(axes) == np.ndarray)):
            ax_tasks = axes[0]
        else:
            ax_tasks = axes
        ax_timelines = None

    ax_tasks.set_title(plot_title)

    line_height = .8

    # Find max and min plotting time
    _all_requests_min_time = None 
    _all_requests_max_time = None
    for request in workflow_graph.nodes():
        
        if (request.scheduled and request.feasible):        # if 'observation_opportunities' in request.keys():
            for _sat, _opportunities in request.observation_opportunities.items():
                for _opportunity in _opportunities:
                    if _all_requests_min_time is None:
                        _all_requests_min_time = _opportunity.rise.time
                    else:
                        _all_requests_min_time = min(_all_requests_min_time, _opportunity.rise.time)
                    if _all_requests_max_time is None:
                        _all_requests_max_time = _opportunity.fall.time
                    else:
                        _all_requests_max_time = max(_all_requests_max_time, _opportunity.fall.time)

    # Annotete the plot with constraints: 
    ax_tasks.set_yticks(np.array(range(num_request_groups))+0.5, request_group_names)
    for request_ix, request in enumerate(workflow_graph.nodes()):
        # y_coordinate = request_ix
        y_coordinate = request_group_names.index(request.request_group)
        # Show the request intervals
        min_time = request.observation_request.min_time
        max_time = request.observation_request.max_time
        ax_tasks.add_patch(plt.Rectangle((min_time, y_coordinate), max_time-min_time, line_height, color=request_group_colors[request.request_group], alpha=.03/request_group_sizes[request.request_group]))
        # SHow the constraint intervals
        # For each constraint
        for parent_request in workflow_graph.predecessors(request):
            prequest_data = parent_request
            if (prequest_data.scheduled and prequest_data.feasible):
                parent_time = prequest_data.observation_opportunity.time
                inedges = workflow_graph.get_edge_data(parent_request, request)
                min_time = _all_requests_min_time
                max_time = _all_requests_max_time
                for constraint_key, constraint in inedges.items():
                    # print(constraint)
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                # These are constraints on the CURRENT node. So START_AFTER means the parent has to start before
                                case TemporalConstraintType.START_AFTER:
                                    min_time = max(min_time, parent_request.observation_opportunity.time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    min_time = max(min_time, parent_request.observation_opportunity.time+offset)
                                case TemporalConstraintType.START_BEFORE:
                                    max_time = min(max_time, parent_request.observation_opportunity.time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, parent_request.observation_opportunity.time+offset)
                        case ConstraintClass.SUCCESS:
                            # The current node needs to know if the parent succeeded. So we constrain the current node to start after the parent
                            min_time = max(min_time, parent_request.observation_opportunity.time)
                        case ConstraintClass.GEOMETRY:
                            min_time = max(min_time, parent_request.observation_opportunity.time)
                ax_tasks.add_patch(plt.Rectangle((min_time, y_coordinate), max_time-min_time, line_height, color=request_group_colors[parent_request.request_group], alpha=.1/request_group_sizes[request.request_group]))
        
        # Show where we actually ended up
        if (request.scheduled and request.feasible):
            if request.completed:
                _task_color = 'k'
                _task_width = 9
            elif request.dispatched:
                _task_color = 'm'
                _task_width = 6
            else:
                _task_color = request_group_colors[request.request_group]
                _task_width = 3
            # Plot the time where the request was scheduled.
            ax_tasks.vlines(request.observation_opportunity.time, y_coordinate, y_coordinate+line_height, color=_task_color, linewidth=_task_width)
        # Plot other times where it could have been scheduled.
        for _sat, _opportunities in request.observation_opportunities.items():
            for _opportunity in _opportunities:
                _pass_is_feasible = feasibility_screener(_sat, _opportunity)
                _min_time = _opportunity.rise.time
                _max_time = _opportunity.fall.time
                 
                pass_color = request_group_colors[request.request_group]
                pass_alpha = 0.1/request_group_sizes[request.request_group]
                if not _pass_is_feasible:
                    pass_color = 'red'
                    pass_alpha = 0.8/request_group_sizes[request.request_group]

                ax_tasks.add_patch(plt.Rectangle((_min_time, y_coordinate), _max_time-_min_time, line_height, color=pass_color, alpha=pass_alpha))

    if ax_timelines is not None:
        timeline_ix = 0
        for timeline in timeline_graph.nodes():
            if type(timeline) == Timeline:
                ax_timeline = ax_timelines[timeline_ix]
                _tl_times = []
                _tl_values = []
                if len(timeline.impact_container):
                    _min_time = timeline.impact_container[0].time
                for impact in timeline.impact_container:
                    if impact.time-dt.timedelta(seconds=1)>_min_time: #Skip the first two impacts
                        _just_before_the_impact = impact.time-dt.timedelta(seconds=1)
                        _tl_times.append(_just_before_the_impact)
                        _tl_values.append(timeline.get_value_at(_just_before_the_impact))
                    _tl_times.append(impact.time)
                    _tl_values.append(timeline.get_value_at(impact.time))

                ax_timeline.plot(_tl_times, _tl_values, '-o')
                timeline_ix += 1
                ax_timeline.set_ylabel(timeline.name,rotation=0, ha='right', va='center')
                ax_timeline.grid()

    if ax_timelines is not None:
        ax_timelines[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        ax_timelines[-1].tick_params(axis='x', labelrotation=45)
    else:
        ax_tasks.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        ax_tasks.tick_params(axis='x', labelrotation=45)
    # ax_tasks.set_xlim(_all_requests_min_time, _all_requests_max_time)

    # Show time
    if time is not None:
        ax_tasks.axvline(time, color='k', linewidth=1)

    if show_night:
        # List the days between all_requests_min_time and all_requests_max_time
        # Use sunrise to get rise, fall
        # If both are available
        # Plot a box from fall of day x to rise of day x+1
        # Start from min time
        # If the next object is sunrise, color black
        # until sunset
        #
        initial_date = _all_requests_min_time.date()
        final_date = _all_requests_max_time.date()
        all_dates = [initial_date+dt.timedelta(days=i) for i in range((final_date-initial_date).days+1)]
        
        _times = [(_all_requests_min_time, "I"), (_all_requests_max_time, "A")]
        for _date in all_dates:
            
            _sunrise_time = sun.sunrise(
                observer= Observer(latitude=show_night_location.lat_deg, longitude=show_night_location.lon_deg, elevation=show_night_location.alt_km/1e3),
                date=_date,
                tzinfo = dt.timezone.utc
            ).replace(tzinfo=None)
            _sunset_time = sun.sunset(
                observer= Observer(latitude=show_night_location.lat_deg, longitude=show_night_location.lon_deg, elevation=show_night_location.alt_km/1e3),
                date=_date,
                tzinfo = dt.timezone.utc
            ).replace(tzinfo=None)

            _times.append((_sunrise_time, "R"))
            _times.append((_sunset_time, "S"))
        _times.sort(key=lambda x: x[0])

        for _time_ix, _time in enumerate(_times[:-1]):
            #if the next time is a "R", then we are at night
            # Ignore anything before the min time
            if _time[0]<_all_requests_min_time:
                continue
            # If the next step is the max time, stop - that is a special case
            if _times[_time_ix+1][0]>=_all_requests_max_time:
                break 
            if _times[_time_ix+1][1] == "R":
                #if the next time is a "R", then we are at night
                # Plot black from _time to _times[_time_ix+1][0]
                ax_tasks.axvspan(xmin = _time[0], xmax=_times[_time_ix+1][0], color='gray', alpha=0.2)
        # At this point _time_ix is either len(_times)-2 or the location right before _all_requests_max_time
        
        # if the next time is the last one, then, was the time before the last a "S"? If so, we are at night
        if _times[_time_ix][1] == "S":
            # Plot black from _times[_time_ix][0] to _times[_time_ix+1][0]
            ax_tasks.axvspan(xmin = _times[_time_ix][0], xmax=_times[_time_ix+1][0], color='gray', alpha=0.2)

    if save_schedule_plot:
        plt.savefig(save_name, bbox_inches='tight')
    if axes is None:
        plt.close(fig)


def find_dispatchable_tasks(workflow_graph = nx.MultiDiGraph(), timeline_graph: nx.MultiDiGraph=nx.MultiDiGraph(), verbose: int=1):
    dispatchable_requests = []
    for request in workflow_graph.nodes():
        _dispatchable = True
        if ((request.scheduled == False) or (request.feasible == False) or (request.dispatched == True) or (request.completed == True)):
             if verbose>1:
                print(f"     [Dispatcher] Request {request} not dispatchable (scheduled: {request.scheduled}, feasible: {request.feasible}, dispatched {request.dispatched}, completed {request.completed})")
             _dispatchable = False
             continue
        for parent_request in workflow_graph.predecessors(request):
            constraint_edges = workflow_graph.get_edge_data(parent_request, request)
            for constraint_key, constraint in constraint_edges.items():
                # If we need to check this type of constraint
                if request.dispatch_policy[constraint['constraint_class']] is False:
                    # Add a special case where
                    # If a task is infeasible
                    # and the constraint is specifically "START_IF_FAILED"
                    # then oh yeah we are ready to dispatch
                    if ((parent_request.scheduled is False or (parent_request.scheduled is True and parent_request.feasible is False)) and constraint['constraint_class']==ConstraintClass.SUCCESS and (constraint['constraint_type']==SuccessConstraintType.START_IF_FAILED or constraint['constraint_type']==SuccessConstraintType.WAIT_FOR_COMPLETION_IF_FEASIBLE)):
                        if verbose>2:
                            print(f"     [Dispatcher] Special case for request {request}: parent {parent_request} is infeasible and constraint is {constraint['constraint_type']}, unscheduled/infeasible counts toward this.")
                        continue
                    elif (parent_request.scheduled is False or parent_request.dispatched is False or parent_request.completed is False):
                        if verbose>1:
                            print(f"     [Dispatcher] Request {request} not dispatchable (parent {parent_request} scheduled {parent_request.scheduled}, dispatched {parent_request.dispatched}, completed {parent_request.completed}). Constraint: {constraint['constraint_class']} {constraint['constraint_type']}")
                        _dispatchable = False
                        break
            if _dispatchable == False:
                break
        
        # This is kind of a hack.
        # If a task has all constraints satisfied, _but_ is rejected because of a timeline, 
        # we will mark the task as infeasible. This is because the planner is very aggressive
        # and (improperly?) relies on the dispatcher to mark tasks as not feasible close
        # to dispatch time.
        # But if a task is infeasible that changes the dependencies that rely on WAIT_FOR_COMPLETION_IF_FEASIBLE
        # So, if we mark a task as infeasible, we will invoke the dispatcher again
        should_rerun_dispatcher_again_due_to_infeasible_tasks = False

        if _dispatchable is True:
            # Now check the timeline
            if request in timeline_graph.nodes():
                for _timeline in timeline_graph.successors(request):
                    tl_edges = timeline_graph.get_edge_data(request, _timeline)
                    for _constraint_key, _constraint in tl_edges.items():
                        if _constraint['edge_type'] == TaskTimelineConstraint:
                            _time = request.observation_opportunity.time
                            if _constraint['constraint_time'] == TaskImpactTime.POST:
                                _time = request.observation_opportunity.time + request.observation_opportunity.duration
                            # Check that the constraint is verified 
                            _tl_value_at_time = _timeline.get_value_at(_time)
                            match _constraint['constraint_type']:
                                case TimelineConstraintType.GREATER_OR_EQUAL:
                                    if not (_tl_value_at_time >= _constraint['constraint_value']):
                                        if verbose:
                                            print(f"     [Dispatcher] Request {request} not dispatchable: constraint {_constraint} on timeline {_timeline} at {_time} violated (timeline value is {_tl_value_at_time})")
                                        _dispatchable = False
                                        request.feasible = False
                                        should_rerun_dispatcher_again_due_to_infeasible_tasks = True
                                        break
                                case TimelineConstraintType.LESSER_OR_EQUAL:
                                    if not (_tl_value_at_time <= _constraint['constraint_value']):
                                        if verbose:
                                            print(f"     [Dispatcher] Request {request} not dispatchable: constraint {_constraint} on timeline {_timeline} at {_time} violated (timeline value is {_tl_value_at_time})")
                                        _dispatchable = False
                                        request.feasible = False
                                        should_rerun_dispatcher_again_due_to_infeasible_tasks = True
                                        break
                                case TimelineConstraintType.EQUAL:
                                    if not (_tl_value_at_time == _constraint['constraint_value']):
                                        if verbose:
                                            print(f"     [Dispatcher] Request {request} not dispatchable: constraint {_constraint} on timeline {_timeline} at {_time} violated (timeline value is {_tl_value_at_time})")
                                        _dispatchable = False
                                        request.feasible = False
                                        should_rerun_dispatcher_again_due_to_infeasible_tasks = True
                                        break
                    if _dispatchable is False:
                        break

        if _dispatchable is True:
            dispatchable_requests.append(request)
        if should_rerun_dispatcher_again_due_to_infeasible_tasks:
            if verbose>0:
                print(f"     [Dispatcher] We marked some tasks as infeasible. Rerunning the dispatcher")
            return find_dispatchable_tasks(workflow_graph = workflow_graph, timeline_graph=timeline_graph, verbose=verbose)
    return dispatchable_requests

