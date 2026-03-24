
import numpy as np

import networkx as nx

from enum import Enum

from fame_agents_base import *

from matplotlib.pyplot import cm

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
    def __init__(self, constraint_class: ConstraintClass, constraint_type, parent: ObservationRequest, parameters: dict={}):
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
            constraints: list = [],
            schedule_policy_if_constraint_unsatisfied: dict={c: True for c in ConstraintClass},
            dispatch_policy_if_constraint_unsatisfied: dict={c: False for c in ConstraintClass},
            follow_up_action_failure=lambda reason: None,
            follow_up_action_success=lambda data_product: None,
            ):
        self.observation_request = observation_request
        self.constraints = constraints
        self.schedule_policy = schedule_policy_if_constraint_unsatisfied
        self.dispatch_policy = dispatch_policy_if_constraint_unsatisfied
        self.follow_up_action_failure = follow_up_action_failure
        self.follow_up_action_success= follow_up_action_success
        self.observation_opportunity: ObservationOpportunity = None
        self.observation_opportunity_satellite: Satellite = None
        self.scheduled: bool = False
        self.feasible: bool = True
        self.dispatched: bool = False
        self.completed: bool = False
        self.successful_execution: bool = False

    def __str__(self):
        return f"{self.observation_request} with {len(self.constraints)} constraints"
    def __repr__(self):
        return self.__str__()
    
def build_workflow_graph(workflow: list):
    # Build a dependency graph
    workflow_graph = nx.MultiDiGraph()
    for constrained_request in workflow:
        # this_node = request
        workflow_graph.add_node(constrained_request, **constrained_request.__dict__) # Father forgive me for I have sinned against Python

    for constrained_request in workflow:
        # this_node = request
        for constraint in constrained_request.constraints:
            workflow_graph.add_edge(
                constraint.parent,
                constrained_request,
                constraint_class=constraint.constraint_class,
                constraint_type=constraint.constraint_type,
                parameters=constraint.parameters,
                )
    
    return workflow_graph

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

def greedy_schedule_workflow(workflow_graph: nx.MultiDiGraph, satellites: list, existing_requests: pd.DataFrame= pd.DataFrame(columns=requests_data_frame_columns)):
    
    
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

    print(f"WG: {workflow_graph}")
    # Walk through requests
    # Start with the root nodes
    nodes_to_visit = [node for node, in_degree in workflow_graph.in_degree() if in_degree == 0]
    while len(nodes_to_visit):
        print(f"Nodes to visit: {nodes_to_visit}")

        constrained_request = nodes_to_visit.pop(0)

        print(f"Current request: {constrained_request}")
        # If the request has gone off, nothing we can do about it
        if ((constrained_request.dispatched is True) or (constrained_request.completed is True)):
            print("Request {} is already dispatched or completed, skipping")
            continue

        _constraints_are_resolvable = True
        for parent_request in workflow_graph.predecessors(constrained_request):
            inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
            for constraint_key, constraint in inedges.items():
                _constraint_class = constraint['constraint_class']
                if (
                    (
                        constrained_request.schedule_policy[_constraint_class] == False and
                        workflow_graph.nodes[parent_request]['completed'] == False
                    )
                ):
                    _constraints_are_resolvable = False
                    break
        if (_constraints_are_resolvable is False):
            print("Request {} has unresolved predecessors and its policy require waiting; skipping.")
            continue

        # Add the successors, so we will try to schedule them even if this one fails
        for child_request in workflow_graph.successors(constrained_request):
            if workflow_graph.nodes[child_request]['scheduled']==False:
            # assert workflow_graph.nodes[child_request]['scheduled']==False, "ERROR: we are traversing the dependency graph in a strange and incorrect way (children)"
                nodes_to_visit.append(child_request)

        min_time = constrained_request.observation_request.min_time
        max_time = constrained_request.observation_request.max_time

        # TODO check if any constraints are violated.
        # For temporal constraints that is auto-handled below.
        # Data constraints are handled externally? Or we could use this to bring the data from the parent to the child.
        # Bool (success) constraints should be checked here.
        #  When iterating over parents, if there is a success dependency, and the parent is reporting completed with failure, do not attempt to schedule.

        print(f"This request has {len(list(workflow_graph.successors(constrained_request)))} children: {list(workflow_graph.successors(constrained_request))}")
        for child_request in workflow_graph.successors(constrained_request):
            if ((workflow_graph.nodes[child_request]['scheduled']==True) and (workflow_graph.nodes[child_request]['feasible']==True)):
                outedges = workflow_graph.get_edge_data(constrained_request, child_request)
                print(f"Child: {child_request}")
                # print(type(workflow_graph.edges[request]))
                for constraint_key, constraint in outedges.items():
                    # print(constraint)
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
                print(f"Skipping constraints for child {child_request}, currently unscheduled")

        print(f"This request has {len(list(workflow_graph.predecessors(constrained_request)))} parents: {list(workflow_graph.predecessors(constrained_request))}")
        for parent_request in workflow_graph.predecessors(constrained_request):
            if ((workflow_graph.nodes[parent_request]['scheduled']==True) and (workflow_graph.nodes[parent_request]['feasible']==True)):
                inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
                print(f"Parent: {parent_request}")
                # print(type(workflow_graph.edges[request]))
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
                print(f"Skipping constraints for parent {parent_request}, currently unscheduled")
        
        if (workflow_graph.nodes[constrained_request]['feasible']==False):
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
            # TODO make a note of this in the graph
            workflow_graph.nodes[constrained_request]['scheduled']=True
            workflow_graph.nodes[constrained_request]['feasible']=False
            print("Could not schedule {} (no passes)".format(constrained_request))
            continue
            # raise ValueError("Could not schedule {} (no passes)".format(constrained_request))
        
        # print(f"Opportunities: {passes}")
        _best_quality = - np.inf
        _best_satellite = None
        _best_pass = None
        allsatpasses = [(satellite, satpass, observation_quality(satpass.highest)) for satellite, satpasses in passes.items() for satpass in satpasses]
        allsatpasses.sort(key=lambda x: x[2], reverse=True) # Sort by observation quality
        for (satellite, satpass, _quality) in allsatpasses:
            if screen_pass_for_feasibility(existing_requests=existing_requests, satellite=satellite, _obs_pass=satpass, screen_against_comm_passes=True):
                _best_quality = _quality
                _best_satellite = satellite
                _best_pass = satpass
                break
        # for satellite, satpasses in passes.items():
        #     for satpass in satpasses:
        #         if screen_pass_for_feasibility(existing_requests=None, satellite=satellite, _obs_pass=satpass, screen_against_comm_passes=True):
        #             _quality = observation_quality(satpass.highest)
        #             if _quality >= _best_quality:
        #                 _best_quality = _quality
        #                 _best_satellite = satellite
        #                 _best_pass = satpass
        if _best_pass is None:
            raise ValueError("Could not schedule {}".format(constrained_request))

        # Pick the best opportunity
        workflow_graph.nodes[constrained_request]['observation_opportunity'] = _best_pass.highest
        workflow_graph.nodes[constrained_request]['observation_opportunity_satellite'] = _best_satellite
        workflow_graph.nodes[constrained_request]['scheduled']=True

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

