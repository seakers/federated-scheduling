
import numpy as np

import networkx as nx

from enum import Enum

from fame_agents_base import *

from matplotlib.pyplot import cm

from ortools.linear_solver import pywraplp

import bisect

# import random


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
            phenomenon_processor=lambda o, s, p: p
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
            phenomenon_processor (function, optional): _description_. Defaults to lambda (observation, spacecraft, phenomenon): phenomenon.
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
        self.scheduled: bool = False
        self.feasible: bool = True
        self.dispatched: bool = False
        self.completed: bool = False
        self.successful_execution: bool = False
        self.phenomenon_processor = phenomenon_processor

    def __str__(self):
        return f"{self.observation_request} with {len(self.task_constraints)} task constraints"
    def __repr__(self):
        return self.__str__()
    

class ImpactType(Enum):
    ASSIGNMENT = 0 # Make the timeline this
    ADDITION = 1 # Add this to the initial value
    RATE_ADDITION = 2 # Add this to the rate at this time

class Impact():
    def __init__(self, time: dt.datetime, type: ImpactType, value, owner: ConstrainedObservationRequest=None):
        self.time = time
        self.type = type
        self.value = value
        self.owner = owner

class Timeline():
    def __init__(self, name: str, initial_time: dt.datetime, initial_value: float, initial_rate: float):
        self.name = name
        self.impact_container = [Impact(initial_time, ImpactType.ASSIGNMENT, initial_value), Impact(initial_time, ImpactType.RATE_ADDITION, initial_rate)]

    def add_impact(self, impact: Impact):
        bisect.insort(self.impact_container, impact, key=lambda x: x.time)
    
    def _get_value_and_rate_at(self, time: dt.datetime):

        # Select only the impacts that apply
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        
        # Accumulators
        _value = 0
        _rate = 0

        _previous_time = dt.datetime.now()
        assert (self.impact_container[0].type == ImpactType.ASSIGNMENT), "ERROR: The initial value in a timeline must be an assignment"
        
        for _impact in self.impact_container[:closest_index+1]:
            match _impact.type:
                case ImpactType.ASSIGNMENT:
                    _value = _impact.value
                    _previous_time = _impact.time
                case ImpactType.ADDITION:
                    # Propagate the rate
                    _dt = _impact.time-_previous_time
                    _integrated_rate = _rate*dt.total_seconds()
                    _value += _integrated_rate
                    
                    # Now actually add the impact
                    _value += _impact.value

                    # Reset time
                    _previous_time = _impact.time
                    
                case ImpactType.RATE_ADDITION:
                    # Propagate the rate
                    _dt = _impact.time-_previous_time
                    _integrated_rate = _rate*dt.total_seconds()
                    _value += _integrated_rate

                    # Now update the rate
                    _rate += _impact.value

                    # Reset time
                    _previous_time = _impact.time

        # Bring to current time
        # Propagate the rate
        _dt = time-_previous_time
        _integrated_rate = _rate*dt.total_seconds()
        _value += _integrated_rate

        return _value, _rate
    
    def get_value_at(self, time: dt.datetime):
        value, rate = self._get_value_and_rate_at(time)
        return value
    
    def consolidate_impacts(self, time: dt.datetime):
        _value_at_time, _rate_at_time = self._get_value_and_rate_at(time)

        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        self.impact_container= self.impact_container[closest_index+1:]
        self.add_impact(Impact(time, ImpactType.ASSIGNMENT, _value_at_time))
        self.add_impact(Impact(time, ImpactType.RATE_ADDITION, _rate_at_time))

    def remove_impacts_from_owner(self, owner: ConstrainedObservationRequest):
        new_impact_container = [i for i in self.impact_container if i.owner != owner]
        self.impact_container = new_impact_container

class AssignmentTimeline(Timeline):
    def __init__(self, name: str, initial_time: dt.datetime, initial_value: bool):
        super().__init__(name, initial_time=initial_time, initial_value=initial_value)

    def add_impact(self, impact: Impact):
        assert (impact.type == ImpactType.ASSIGNMENT), "ERROR: non-assignment impact on assignment timeline"
        bisect.insort(self.impact_container, impact, key=lambda x: x.time)
    
    def consolidate_impacts(self, time: dt.datetime):
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        self.impact_container= self.impact_container[closest_index:]
    
    def get_value_at(self, time: dt.datetime):
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        return self.impact_container[closest_index].value
    
class AdditiveTimeline(Timeline):
    def __init__(self, name: str, initial_time: dt.datetime, initial_value: float):
        super().__init__(name, initial_time=initial_time, initial_value=initial_value)
    
    def add_impact(self, impact: Impact):
        assert (impact.type == ImpactType.ADDITION), "ERROR: non-additive impact on additive timeline"
        bisect.insort(self.impact_container, impact, key=lambda x: x.time)

    def get_value_at(self, time):
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        return np.sum([_impact.value for _impact in self.impact_container[:closest_index+1]])
    
    def consolidate_impacts(self, time):
        _value_at_time = self.get_value_at(time)
        closest_index = bisect.bisect(self.impact_container, time, key=lambda x: x.time)
        self.impact_container= self.impact_container[closest_index+1:]
        assert (self.impact_container[0].time>=time), "ERROR: something went horribly wrong consolidating impacts"
        self.impact_container.insert(0, Impact(time, ImpactType.ASSIGNMENT, _value_at_time))

class TaskImpactTime(Enum):
    PRE = 0 
    POST = 1

class TimelineConstraintType(Enum):
    GREATER_OR_EQUAL = 0 
    LESSER_OR_EQUAL = 1
    EQUAL = 2

class TaskTimelineImpact():
    def __init__(self, timeline: Timeline, time: TaskImpactTime, type: ImpactType, value):
        self.timeline = Timeline
        self.time = time
        self.type = type
        self.value = value

class TaskTimelineConstraint():
    def __init__(self, timeline: Timeline, time: TaskImpactTime, type: TimelineConstraintType, value):
        self.timeline = Timeline
        self.time = time
        self.type = type
        self.value = value

class Workflow():
    def __init__(
            self,
            constrained_observation_requests: list[ConstrainedObservationRequest],
            timelines: list[Timeline]=[],
    ):
        self.constrained_observation_requests = constrained_observation_requests
        self.timelines = timelines


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
                constraint_time=tconstraint.time,
                constraint_type=tconstraint.type,
                constraint_value=tconstraint.value,
                )
        for timpact in constrained_request.timeline_impacts:
            timeline_graph.add_edge(
                constrained_request,
                tconstraint.timeline,
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
        existing_requests: pd.DataFrame= pd.DataFrame(columns=requests_data_frame_columns), current_time: dt.datetime=None,
        verbose: int=99,
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
        if ((workflow_graph.nodes[node_id]['dispatched'] == False) and (workflow_graph.nodes[node_id]['completed'] == False)):
             workflow_graph.nodes[node_id]['scheduled'] = False
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
            if workflow_graph.nodes[child_request]['scheduled']==False:
            # assert workflow_graph.nodes[child_request]['scheduled']==False, "ERROR: we are traversing the dependency graph in a strange and incorrect way (children)"
                nodes_to_visit.append(child_request)
        if verbose>2:
            print(f"   [Scheduler] Current request: {constrained_request}")
        # If the request has gone off, nothing we can do about it
        if ((workflow_graph.nodes[constrained_request]['dispatched'] == True) or (workflow_graph.nodes[constrained_request]['completed'] == True)):
            if verbose>0:
                print(f"   [Scheduler] Request {constrained_request} is already dispatched or completed, skipping")
            continue
        else:
            if verbose>2:
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
                        workflow_graph.nodes[constrained_request]['schedule_policy'][_constraint_class] == False and
                        workflow_graph.nodes[parent_request]['completed'] == False
                    )
                ):
                    _constraints_are_resolvable = False
                    break
        if (_constraints_are_resolvable is False):
            if verbose>1:
                print("   [Scheduler] Request {} has unresolved predecessors and its policy require waiting; skipping.")
            continue


        # Now we start with temporal scheduling. We will identify a single interval that works.
        min_time = workflow_graph.nodes[constrained_request]['observation_request'].min_time
        if (current_time is not None):
            min_time = max(min_time, current_time)
        max_time = workflow_graph.nodes[constrained_request]['observation_request'].max_time


        if ((current_time is not None) and (max_time<current_time)):
            if verbose>1:
                print("   [Scheduler] Request {} max time is in the past; skipping.")
            workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=False
            continue



        # First, we check if any CHILDREN have been scheduled and apply the constraints backwards.
        # If the dependency graph is not degenerate, the check three lines below should fail, and we should do nothing in this cycle. 
        if verbose>3:
            print(f"   [Scheduler] This request has {len(list(workflow_graph.successors(constrained_request)))} children: {list(workflow_graph.successors(constrained_request))}")
        for child_request in workflow_graph.successors(constrained_request):
            if ((workflow_graph.nodes[child_request]['scheduled']==True) and (workflow_graph.nodes[child_request]['feasible']==True)):
                outedges = workflow_graph.get_edge_data(constrained_request, child_request)
                if verbose>3:
                    print(f"   [Scheduler] Child: {child_request}")
                for constraint_key, constraint in outedges.items():
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                # These are constraints on the CHILD. So START_AFTER means the parent has to start before
                                case TemporalConstraintType.START_AFTER:
                                    max_time = min(max_time, workflow_graph.nodes[child_request]['observation_opportunity'].time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    # Note the minus: we want parent_time + offset<child_time
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, workflow_graph.nodes[child_request]['observation_opportunity'].time-offset)
                                case TemporalConstraintType.START_BEFORE:
                                    min_time = max(min_time, workflow_graph.nodes[child_request]['observation_opportunity'].time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    # Note the - : we want child_time<parent_time+offset
                                    min_time = max(min_time, workflow_graph.nodes[child_request]['observation_opportunity'].time-offset)
                        case ConstraintClass.SUCCESS:
                            # The child needs to know if the parent succeeded. So we constrain the parent to finish before the child
                            max_time = min(max_time, workflow_graph.nodes[child_request]['observation_opportunity'].time)
                        case ConstraintClass.GEOMETRY:
                            max_time = min(max_time, workflow_graph.nodes[child_request]['observation_opportunity'].time)
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
            if ((workflow_graph.nodes[parent_request]['scheduled']==True) and (workflow_graph.nodes[parent_request]['feasible']==True)):
                inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
                if verbose>3:
                    print(f"   [Scheduler] Parent: {parent_request}")
                for constraint_key, constraint in inedges.items():
                    match constraint['constraint_class']:
                        case ConstraintClass.TEMPORAL:
                            match constraint['constraint_type']:
                                # These are constraints on the CURRENT node. So START_AFTER means the parent has to start before
                                case TemporalConstraintType.START_AFTER:
                                    min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                                case TemporalConstraintType.START_BEFORE:
                                    max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                        case ConstraintClass.SUCCESS:
                            # The current node needs to know if the parent succeeded. So we constrain the current node to start after the parent
                            min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                            if (workflow_graph.nodes[parent_request]['completed'] is True):
                                if (
                                    (
                                        (constraint['constraint_type'] == SuccessConstraintType.START_IF_FAILED) and 
                                        (workflow_graph.nodes[parent_request]['successful_execution'] == True)
                                        ) or (
                                        (constraint['constraint_type'] == SuccessConstraintType.START_IF_SUCCESSFUL) and 
                                        (workflow_graph.nodes[parent_request]['successful_execution'] == False)
                                        )
                                    ):
                                # If incompatible, skip
                                    workflow_graph.nodes[constrained_request]['scheduled']=True
                                    workflow_graph.nodes[constrained_request]['feasible']=False
                                    break
                                
                        case ConstraintClass.GEOMETRY:
                            min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
            else:
                # REVIEW this is where we can exclude a successor if the parent is infeasible. 
                if (workflow_graph.nodes[parent_request]['feasible']==False):
                    workflow_graph.nodes[constrained_request]['scheduled']=True
                    workflow_graph.nodes[constrained_request]['feasible']=False
                    break
                else:
                    if verbose>2:
                        print(f"   [Scheduler] Skipping task constraints for parent {parent_request}, currently unscheduled")
        
        if (workflow_graph.nodes[constrained_request]['feasible']==False):
            if verbose>1:
                print(f"   [Scheduler] This request is infeasible due to parents")
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
        workflow_graph.nodes[constrained_request]['observation_opportunities'] = observation_opportunities[trimmed_request]
        # print(f"Found opportunities: {observation_opportunities}")
        if trimmed_request not in observation_opportunities.keys():
            raise ValueError("Could not schedule {}".format(constrained_request))
        passes = observation_opportunities[trimmed_request]
        if len(passes)==0:
            # Make a note of this in the graph
            workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=False
            if verbose>0:
                print("   [Scheduler] Could not schedule {} (no passes)".format(constrained_request))
            continue
            # raise ValueError("Could not schedule {} (no passes)".format(constrained_request))
        
        # print(f"Opportunities: {passes}")
        _best_quality = - np.inf
        _best_satellite = None
        _best_pass = None
        allsatpasses = [(satellite, satpass, observation_quality(satpass.highest)) for satellite, satpasses in passes.items() for satpass in satpasses]
        allsatpasses.sort(key=lambda x: x[2], reverse=True) # Sort by observation quality
        for (satellite, satpass, _quality) in allsatpasses:
            # if screen_pass_for_feasibility(existing_requests=existing_requests, satellite=satellite, _obs_pass=satpass, screen_against_comm_passes=True):
            if feasibility_screener(satellite, satpass):
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
            workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=False
            if verbose>0:
                print("   [Scheduler] Could not schedule {} (all conflicts)".format(constrained_request))
            continue
            raise ValueError("Could not schedule {} (all conflicts)".format(constrained_request))

        # Pick the best opportunity
        workflow_graph.nodes[constrained_request]['observation_opportunity_pass'] = _best_pass
        workflow_graph.nodes[constrained_request]['observation_opportunity'] = _best_pass.highest
        workflow_graph.nodes[constrained_request]['observation_opportunity_satellite'] = _best_satellite
        workflow_graph.nodes[constrained_request]['scheduled']=True
        # TODO apply timeline impacts here!
        if verbose>2:
            print(f"   [Scheduler] Scheduled request on {_best_satellite} at {_best_pass.highest}")

    return workflow_graph



def ilp_schedule_workflow(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list,
        feasibility_screener = lambda satellite, observation_pass: True,
        existing_requests: pd.DataFrame= pd.DataFrame(columns=requests_data_frame_columns), current_time: dt.datetime=None,
        verbose: int=99,
        ):
    
    if verbose>2:
        print(f"   [Scheduler] WG: {workflow_graph}")

    # We will reschedule everything that is not dispatched. 
    for node_id in workflow_graph.nodes():
        if ((workflow_graph.nodes[node_id]['dispatched'] == False) and (workflow_graph.nodes[node_id]['completed'] == False)):
             workflow_graph.nodes[node_id]['scheduled'] = False

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
    if not solver:
        raise ValueError("Solver SCIP_MIXED_INTEGER_PROGRAMMING not found")
    
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
    solution_holder_by_satellite = {}
    for constrained_request in workflow_graph.nodes():
        solution_holder[constrained_request] = {}

        observation_opportunities = find_observation_opportunities([constrained_request.observation_request,], satellites)
        workflow_graph.nodes[constrained_request]['observation_opportunities'] = observation_opportunities[constrained_request.observation_request]
        # print(f"Found opportunities: {observation_opportunities}")
        if constrained_request.observation_request not in observation_opportunities.keys():
            raise ValueError("Could not schedule {}".format(constrained_request))
        passes = observation_opportunities[constrained_request.observation_request]

        if len(passes)==0:
            workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=False
            if verbose>0:
                print("   [Scheduler] Could not schedule {} (no passes)".format(constrained_request))
            continue
        
        _found_a_pass = False
        allsatpasses = [(satellite, satpass, observation_quality(satpass.highest)) for satellite, satpasses in passes.items() for satpass in satpasses]
        allsatpasses.sort(key=lambda x: x[2], reverse=True) # Sort by observation quality
        for (satellite, satpass, _quality) in allsatpasses:
            if feasibility_screener(satellite, satpass):
                _found_a_pass = True

                if satellite not in solution_holder[constrained_request].keys():
                    solution_holder[constrained_request][satellite] = {}
                if satellite not in solution_holder_by_satellite.keys():
                    solution_holder_by_satellite[satellite] = []
                solution_holder[constrained_request][satellite][satpass] = solver.BoolVar(f"{constrained_request}_{satellite}_{satpass}")

                solution_holder_by_satellite[satellite].append((satpass, solution_holder[constrained_request][satellite][satpass]))
                
                objective.SetCoefficient(solution_holder[constrained_request][satellite][satpass], _quality)

                # TODO for each timeline impacted, add a variable for that timeline value at that time. Add an Impact with that timeline value times "do we do it".
                # TODO for each constraint, invoke get_value_at on the timeline and constrain the outcome. Except! You need to do this AFTER all the impacts have 
                #  been tabulated. YOu will need a follow-up pass.

        if _found_a_pass is False:
            workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=False
            if verbose>0:
                print("   [Scheduler] Could not schedule {} (all conflicts)".format(constrained_request))

        # At most one observation per request is assigned
        solver.Add(sum([solution_holder[constrained_request][_satellite][_satpass] for _satellite, _satpasses in passes.items() for _satpass in _satpasses]) <= 1)

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
                                        for parent_satellite in solution_holder[parent_request].keys():
                                            for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                # If the parent request starts after the current request, then they can't be true at the same time
                                                if parent_pass.highest.time>this_pass.highest.time:
                                                    solver.Add(this_decision_variable + parent_decision_variable <= 1)


                                # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                            case TemporalConstraintType.START_AFTER_OFFSET:
                                offset = constraint['parameters']['offset']
                                # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        for parent_satellite in solution_holder[parent_request].keys():
                                            for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                # If the parent request starts after the current request, then they can't be true at the same time
                                                if parent_pass.highest.time+offset>this_pass.highest.time:
                                                    solver.Add(this_decision_variable + parent_decision_variable <= 1)
                            case TemporalConstraintType.START_BEFORE:
                                # max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        for parent_satellite in solution_holder[parent_request].keys():
                                            for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                # If the parent request starts after the current request, then they can't be true at the same time
                                                if parent_pass.highest.time<this_pass.highest.time:
                                                    solver.Add(this_decision_variable + parent_decision_variable <= 1)
                            case TemporalConstraintType.START_BEFORE_OFFSET:
                                offset = constraint['parameters']['offset']
                                # max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        for parent_satellite in solution_holder[parent_request].keys():
                                            for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                                # If the parent request starts after the current request, then they can't be true at the same time
                                                if parent_pass.highest.time+offset<this_pass.highest.time:
                                                    solver.Add(this_decision_variable + parent_decision_variable <= 1)
                    case ConstraintClass.SUCCESS:
                        # The current node needs to know if the parent succeeded. So we constrain the current node to start after the parent
                        # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                        for this_satellite in solution_holder[constrained_request].keys():
                            for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                for parent_satellite in solution_holder[parent_request].keys():
                                    for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                        # If the parent request starts after the current request, then they can't be true at the same time
                                        if parent_pass.highest.time>this_pass.highest.time:
                                            solver.Add(this_decision_variable + parent_decision_variable <= 1)

                        if (workflow_graph.nodes[parent_request]['completed'] is True):
                            if (
                                (
                                    (constraint['constraint_type'] == SuccessConstraintType.START_IF_FAILED) and 
                                    (workflow_graph.nodes[parent_request]['successful_execution'] == True)
                                    ) or (
                                    (constraint['constraint_type'] == SuccessConstraintType.START_IF_SUCCESSFUL) and 
                                    (workflow_graph.nodes[parent_request]['successful_execution'] == False)
                                    )
                                ):
                            # If incompatible, skip
                                for this_satellite in solution_holder[constrained_request].keys():
                                    for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                        solver.Add(this_decision_variable == 0)
                                workflow_graph.nodes[constrained_request]['scheduled']=True
                                workflow_graph.nodes[constrained_request]['feasible']=False
                                break
                            
                    case ConstraintClass.GEOMETRY:
                        # min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                        for this_satellite in solution_holder[constrained_request].keys():
                            for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                                for parent_satellite in solution_holder[parent_request].keys():
                                    for parent_pass, parent_decision_variable in solution_holder[parent_request][parent_satellite].items():
                                        # If the parent request starts after the current request, then they can't be true at the same time
                                        if parent_pass.highest.time>this_pass.highest.time:
                                            solver.Add(this_decision_variable + parent_decision_variable <= 1)

        if workflow_graph.nodes[constrained_request]['is_mandatory'] is True:
            solver.Add(sum([solution_holder[constrained_request][_satellite][_satpass] for _satellite, _satpasses in passes.items() for _satpass in _satpasses]) == 1)

    for constrained_request in workflow_graph.nodes():
        passes = observation_opportunities[constrained_request.observation_request]

        if len(passes)==0:
            continue
        
        allsatpasses = [(satellite, satpass, observation_quality(satpass.highest)) for satellite, satpasses in passes.items() for satpass in satpasses]
        allsatpasses.sort(key=lambda x: x[2], reverse=True) # Sort by observation quality
        for (satellite, satpass, _quality) in allsatpasses:
            if feasibility_screener(satellite, satpass):
                # solution_holder[constrained_request][satellite][satpass] = solver.BoolVar(f"{constrained_request}_{satellite}_{satpass}")

                # TODO for each constraint, invoke get_value_at on the timeline and constrain the outcome. You need to do this AFTER all the impacts have 
                #  been tabulated. This is the follow-up pass
                pass

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

    # - Add communication uplinks if ISL is not guaranteed: 
    #   - Add uplinks for sat i
    # If no ISL for sat i and we schedule an observation on sat I, we also schedule an uplink on it
    # Think more about how to represent the MSA workflow accepting that tasks can be scheduled if their parent is not

    objective.SetMaximization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Objective value =", solver.Objective().Value())
        for constrained_request in workflow_graph.nodes():
            _feasible = False
            for this_satellite in solution_holder[constrained_request].keys():
                for this_pass, this_decision_variable in solution_holder[constrained_request][this_satellite].items():
                    
                    if this_decision_variable.solution_value()>0:
                        print(this_decision_variable.name(), " = ", this_decision_variable.solution_value())
                        _feasible = True
                        workflow_graph.nodes[constrained_request]['observation_opportunity_pass'] = this_pass
                        workflow_graph.nodes[constrained_request]['observation_opportunity'] = this_pass.highest
                        workflow_graph.nodes[constrained_request]['observation_opportunity_satellite'] = this_satellite
                        workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=_feasible

        print()
        print(f"Problem solved in {solver.wall_time():d} milliseconds")
        print(f"Problem solved in {solver.iterations():d} iterations")
        print(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")
    else:
        print("The problem does not have an optimal solution.")


    return workflow_graph


def plot_workflow_schedule(workflow_graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    num_requests = len(workflow_graph)
    request_names = list(workflow_graph.nodes())
    request_colors_list = cm.rainbow(np.linspace(0, 1, num_requests))
    request_colors = {task: request_colors_list[task_ix] for task_ix, task in enumerate(request_names)}

    line_height = .8

    _all_requests_min_time = None 
    _all_requests_max_time = None
    for request in workflow_graph.nodes():
        request_data = workflow_graph.nodes[request]
        # Plot other times where it could have been scheduled.
        if (request_data['scheduled'] and request_data['feasible']):        # if 'observation_opportunities' in request_data.keys():
            for _sat, _opportunities in request_data['observation_opportunities'].items():
                for _opportunity in _opportunities:
                    if _all_requests_min_time is None:
                        _all_requests_min_time = _opportunity.rise.time
                    else:
                        _all_requests_min_time = min(_all_requests_min_time, _opportunity.rise.time)
                    if _all_requests_max_time is None:
                        _all_requests_max_time = _opportunity.fall.time
                    else:
                        _all_requests_max_time = max(_all_requests_max_time, _opportunity.fall.time)

    # Annotete the plot
    ax.set_yticks(np.array(range(num_requests))+0.5, request_names)
    for request_ix, request in enumerate(workflow_graph.nodes()):
        request_data = workflow_graph.nodes[request]
        # SHow the constraint intervals
        # For each constraint
        for parent_request in workflow_graph.predecessors(request):
            prequest_data = workflow_graph.nodes[parent_request]
            if (prequest_data['scheduled'] and prequest_data['feasible']):
                parent_time = workflow_graph.nodes[parent_request]['observation_opportunity'].time
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
                                    min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                                case TemporalConstraintType.START_AFTER_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                                case TemporalConstraintType.START_BEFORE:
                                    max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                                case TemporalConstraintType.START_BEFORE_OFFSET:
                                    offset = constraint['parameters']['offset']
                                    max_time = min(max_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time+offset)
                        case ConstraintClass.SUCCESS:
                            # The current node needs to know if the parent succeeded. So we constrain the current node to start after the parent
                            min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                        case ConstraintClass.GEOMETRY:
                            min_time = max(min_time, workflow_graph.nodes[parent_request]['observation_opportunity'].time)
                ax.add_patch(plt.Rectangle((min_time, request_ix), max_time-min_time, line_height, color=request_colors[parent_request], alpha=.1))
        
        # Show where we actually ended up
        if (request_data['scheduled'] and request_data['feasible']):
            # Plot the time where the request was scheduled.
            ax.vlines(request_data['observation_opportunity'].time, request_ix, request_ix+line_height, color=request_colors[request], linewidth=3)
            # Plot other times where it could have been scheduled.
            for _sat, _opportunities in request_data['observation_opportunities'].items():
                for _opportunity in _opportunities:
                    _min_time = _opportunity.rise.time
                    _max_time = _opportunity.fall.time
                    ax.add_patch(plt.Rectangle((_min_time, request_ix), _max_time-_min_time, line_height, color=request_colors[request], alpha=.1))

def find_dispatchable_tasks(workflow_graph = nx.MultiDiGraph()):
    dispatchable_requests = []
    for request in workflow_graph.nodes():
        _dispatchable = True
        request_data = workflow_graph.nodes[request]
        if ((request_data['scheduled'] == False) or (request_data['feasible'] == False) or (request_data['dispatched'] == True) or (request_data['completed'] == True)):
             _dispatchable = False
             continue
        for parent_request in workflow_graph.predecessors(request):
            parent_request_data = workflow_graph.nodes[parent_request]
            constraint_edges = workflow_graph.get_edge_data(parent_request, request)
            for constraint_key, constraint in constraint_edges.items():
                # If we need to check this type of constraint
                if request_data['dispatch_policy'][constraint['constraint_class']] is False:
                    if (parent_request_data['scheduled'] is False or parent_request_data['dispatched'] is False or parent_request_data['completed'] is False):
                        _dispatchable = False
                        break
            if _dispatchable == False:
                break
        if _dispatchable is True:
            dispatchable_requests.append(request)
    return dispatchable_requests

