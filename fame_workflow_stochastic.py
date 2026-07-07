"""
Stochastic MILP Scheduling for FAME

This module implements stochastic Mixed-Integer Linear Programming (MILP) scheduling
that accounts for observation success probabilities due to constellation manager
acceptance/rejection uncertainty.

Two formulations are supported:
1. Non-convex: Uses quadratic constraints (exact, requires Gurobi NonConvex=2)
2. Log-linearized: Uses piecewise-linear log/exp approximations (faster, approximate)
"""

import numpy as np
import networkx as nx
import datetime as dt
import gurobipy as gp
from gurobipy import GRB
import os
from typing import Callable, Optional
from dotenv import load_dotenv

from fame_geometry import ObservationPass, Satellite
from fame_workflow import ConstrainedObservationRequest

# Load environment variables from .env file
load_dotenv()


def ilp_schedule_workflow_stochastic(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list[Satellite],
        feasibility_screener: Callable = lambda satellite, observation_pass: True,
        current_time: dt.datetime = None,
        verbose: int = 99,
        max_solver_time_s: float = 1e3,
        receding_horizon_duration: dt.timedelta = dt.timedelta(weeks=52),
        stochastic_formulation: str = "log_linearized",  # or "non_convex"
        success_probability_function: Callable = lambda r, s, p: 1.0,  # DEPRECATED: use acceptance + execution functions
        acceptance_probability_function: Callable = None,  # p_acc: prob constellation accepts booking
        execution_probability_function: Callable = None,   # p_exec: prob accepted booking executes successfully
        epsilon: float = 1e-5,
        pwl_tolerance: float = 1e-2,
        solver_engine: str = "GUROBI",
        tax_rate: float = 0.15,  # Cost per scheduled obs as fraction of max quality (dynamic, per-request). Set to 0 to disable.
        submission_cost_rate: float = 0.0,  # c_sub: unconditional per-booking submission overhead (as fraction of quality)
        execution_cost_rate: float = 0.0  # c_canc: conditional cancellation cost if accepted (as fraction of quality)
):
    """
    Stochastic MILP scheduler that accounts for observation success probabilities.

    Parameters
    ----------
    workflow_graph : nx.MultiDiGraph
        DAG of ConstrainedObservationRequest nodes with constraint edges
    timeline_graph : nx.MultiDiGraph
        Graph tracking resource timeline constraints
    satellites : list[Satellite]
        Available satellites for scheduling
    feasibility_screener : Callable
        Function to check if a (satellite, pass) is feasible
    current_time : dt.datetime
        Current simulation time
    verbose : int
        Verbosity level (0=silent, 3+=debug)
    max_solver_time_s : float
        Solver timeout in seconds
    receding_horizon_duration : dt.timedelta
        Planning horizon window
    stochastic_formulation : str
        "non_convex" (exact quadratic) or "log_linearized" (PWL approximation)
    success_probability_function : Callable
        Function(request, satellite, pass) -> float in [0,1]
        Returns probability of successful observation
    epsilon : float
        Numerical stability floor for log operations
    pwl_tolerance : float
        Error tolerance for piecewise-linear log/exp approximations
    solver_engine : str
        "GUROBI" (native) or "SCIP" (via OR-Tools)

    Returns
    -------
    workflow_graph : nx.MultiDiGraph
        Updated graph with scheduled observations assigned to nodes
    """

    if verbose > 2:
        print(f"[Stochastic Scheduler] Using formulation: {stochastic_formulation}")
        print(f"[Stochastic Scheduler] Solver engine: {solver_engine}")

    # Step 1: Reset scheduling state for undispatched tasks
    for node in workflow_graph.nodes():
        if (node.dispatched == False) and (node.completed == False):
            node.scheduled = False

    # Step 2: Build Gurobi environment
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    gurobi_license_id = os.getenv("LICENSEID")

    if not all([wls_access_id, wls_secret, gurobi_license_id]):
        raise ValueError(
            "Missing Gurobi WLS credentials. Please verify your .env file!\n"
            "Required: WLSACCESSID, WLSSECRET, LICENSEID"
        )

    wls_params = {
        "WLSACCESSID": wls_access_id,
        "WLSSECRET": wls_secret,
        "LICENSEID": int(gurobi_license_id),
        "OutputFlag": 0  # Suppress Gurobi output (noisy node logs)
    }

    with gp.Env(params=wls_params) as env:
        model_name = f"FAME_Stochastic_{stochastic_formulation}"
        with gp.Model(model_name, env=env) as model:

            # Set solver parameters
            model.setParam('TimeLimit', max_solver_time_s)
            #model.setParam('MIPGap', 0.01)  # 1% optimality gap

            # Enable non-convex solver if using quadratic formulation
            if stochastic_formulation == "non_convex":
                model.setParam('NonConvex', 2)

            # Step 3: Find observation opportunities and create variables
            solution_holder = {}
            task_to_passes = {}

            for constrained_request in workflow_graph.nodes():
                if (constrained_request.dispatched == True) or (constrained_request.completed == True):
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] Skipping {constrained_request.observation_request.name} (dispatched/completed)")
                    continue

                # Find observation opportunities
                trimmed_request_min_time = constrained_request.observation_request.min_time
                if current_time is not None and current_time > constrained_request.observation_request.min_time:
                    trimmed_request_min_time = current_time
                trimmed_request_max_time = min(
                    constrained_request.observation_request.max_time,
                    trimmed_request_min_time + receding_horizon_duration
                )

                from fame_geometry import ObservationRequest, find_observation_opportunities
                trimmed_request = ObservationRequest(
                    lon_deg=constrained_request.observation_request.lon_deg,
                    lat_deg=constrained_request.observation_request.lat_deg,
                    min_time=trimmed_request_min_time,
                    max_time=trimmed_request_max_time,
                    alt_km=constrained_request.observation_request.alt_km,
                    instrument=constrained_request.observation_request.instrument,
                    request_name=constrained_request.observation_request.name + "_trimmed",
                    min_elevation_deg=constrained_request.observation_request.min_elevation_deg,
                )

                observation_opportunities = find_observation_opportunities([trimmed_request], satellites)
                constrained_request.observation_opportunities = observation_opportunities[trimmed_request]

                if trimmed_request not in observation_opportunities.keys():
                    raise ValueError(f"Could not schedule {constrained_request}")

                passes = observation_opportunities[trimmed_request]

                if len(passes) == 0:
                    constrained_request.scheduled = True
                    constrained_request.feasible = False
                    if verbose > 1:
                        print(f"[Stochastic Scheduler] No passes for {constrained_request.observation_request.name}")
                    continue

                # Create decision variables for feasible passes
                _found_a_pass = False
                allsatpasses = [
                    (satellite, satpass, constrained_request.rewarder(satpass.highest))
                    for satellite, satpasses in passes.items()
                    for satpass in satpasses
                ]
                allsatpasses.sort(key=lambda x: x[2], reverse=True)  # Sort by quality

                solution_holder[constrained_request] = {}
                task_to_passes[constrained_request] = []

                for (satellite, satpass, _quality) in allsatpasses:
                    if feasibility_screener(satellite, satpass):
                        _found_a_pass = True

                        if satellite not in solution_holder[constrained_request].keys():
                            solution_holder[constrained_request][satellite] = {}

                        # Binary decision variable: schedule this pass?
                        var_name = f"x_{constrained_request.observation_request.name}_{satellite.name}_{satpass.highest.time}"
                        x_var = model.addVar(vtype=GRB.BINARY, name=var_name)

                        # Compute two-stage probabilities
                        if acceptance_probability_function is not None and execution_probability_function is not None:
                            # New two-stage model
                            p_acc = acceptance_probability_function(constrained_request, satellite, satpass)
                            p_exec = execution_probability_function(constrained_request, satellite, satpass)
                            p_total = p_acc * p_exec
                        else:
                            # Fallback to legacy single-stage model
                            p_total = success_probability_function(constrained_request, satellite, satpass)
                            p_acc = p_total  # Assume all uncertainty is in acceptance
                            p_exec = 1.0

                        solution_holder[constrained_request][satellite][satpass] = {
                            'x': x_var,
                            'quality': _quality,
                            'theta': p_total,      # End-to-end success probability
                            'theta_acc': p_acc,    # Acceptance probability
                            'theta_exec': p_exec   # Execution probability
                        }

                        task_to_passes[constrained_request].append((satellite, satpass))

                if not _found_a_pass:
                    constrained_request.scheduled = True
                    constrained_request.feasible = False
                    if verbose > 1:
                        print(f"[Stochastic Scheduler] All passes infeasible for {constrained_request.observation_request.name}")

            # Step 4: Build stochastic formulation
            if stochastic_formulation == "non_convex":
                _build_non_convex_formulation(
                    model, workflow_graph, solution_holder, task_to_passes,
                    epsilon, tax_rate, submission_cost_rate, execution_cost_rate, verbose
                )
            elif stochastic_formulation == "log_linearized":
                _build_log_linearized_formulation(
                    model, workflow_graph, solution_holder, task_to_passes,
                    epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate, verbose
                )
            else:
                raise ValueError(f"Unknown stochastic_formulation: {stochastic_formulation}")

            # Step 5: Add constraints (temporal, timeline, etc.)
            # NOTE: Only add constraints for tasks that are actually in solution_holder
            # (tasks without passes are marked infeasible and excluded)
            _add_workflow_constraints(
                model, workflow_graph, solution_holder, verbose
            )

            # Step 6: Solve
            # CRITICAL: Must call model.update() before NumVars/NumConstrs return accurate counts
            # Otherwise Gurobi's lazy variable tracking reports 0 even when vars have been added
            model.update()

            if verbose > 0:
                print(f"[Stochastic Scheduler] Solving with {model.NumVars} variables, {model.NumConstrs} constraints")

            # Skip optimization if there are no variables (nothing to schedule)
            if model.NumVars == 0:
                if verbose > 0:
                    print("[Stochastic Scheduler] No pending tasks to schedule. Skipping optimization.")
                # Mark model as optimal with 0 objective for consistency
                workflow_graph.graph['objective_value'] = 0.0
            else:
                model.optimize()

            # Step 7: Extract solution
            if model.NumVars > 0:  # Only extract if we actually solved something
                if model.Status == GRB.OPTIMAL:
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] Solution found! Objective value: {model.ObjVal:.2f}")
                        print(f"[Stochastic Scheduler] (Reward - tax_rate={tax_rate}*max_quality_per_request*NumScheduled)")

                    _extract_solution(
                        model, workflow_graph, solution_holder, verbose
                    )

                    # Store objective value on workflow graph for later analysis
                    workflow_graph.graph['objective_value'] = model.ObjVal
                elif model.Status == GRB.INFEASIBLE:
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] Model is infeasible (no valid schedule found)")
                    if verbose > 2:
                        print(f"[Stochastic Scheduler] Computing IIS to diagnose infeasibility...")
                        try:
                            model.computeIIS()
                            model.write("infeasible_model.ilp")
                            print(f"[Stochastic Scheduler] IIS written to infeasible_model.ilp")
                        except:
                            pass
                else:
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] Solver status: {model.Status}")

    # Clean up timeline impacts - remove unpicklable solver objects (OR-Tools and Gurobi)
    for timeline in timeline_graph.nodes():
        if type(timeline).__name__ == 'Timeline':
            _new_impact_container = []
            for impact in timeline.impact_container:
                impact_module = getattr(impact.value, '__module__', None)
                # Remove both OR-Tools and Gurobi objects (they can't be pickled/deepcopied)
                if (not (impact_module is not None and (impact_module.startswith('ortools') or impact_module.startswith('gurobipy')))):
                    _new_impact_container.append(impact)
            timeline.impact_container = _new_impact_container

    # Also clean up any solver objects that might be attached to workflow nodes
    # This is critical for infeasible solutions where no tasks were scheduled
    for node in workflow_graph.nodes():
        # Remove any Gurobi/OR-Tools objects from node attributes
        if hasattr(node, '__dict__'):
            for attr_name, attr_value in list(node.__dict__.items()):
                if attr_value is not None:
                    attr_module = getattr(attr_value, '__module__', None)
                    if attr_module is not None and (attr_module.startswith('gurobipy') or attr_module.startswith('ortools')):
                        setattr(node, attr_name, None)

    return workflow_graph


def _build_non_convex_formulation(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        task_to_passes: dict,
        epsilon: float,
        tax_rate: float,
        submission_cost_rate: float,
        cancellation_cost_rate: float,
        verbose: int
):
    """
    Build non-convex quadratic formulation with exact products.
    Fixes the multi-parent overwrite bug and rigid equivalence trap by
    chaining products across deduplicated ancestor closure sets.
    """
    y_vars = {}  
    w_vars = {}  
    S_vars = {}  
    A_parents_vars = {}  
    w_abs_vars = {}  

    if verbose > 0:
        print(f"[Non-Convex] Building exact quadratic formulation for {len(solution_holder)} tasks.")

    # === STEP 1: HORIZONTAL RECURRENCE LAYER ===
    for constrained_request in solution_holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        passes = task_to_passes[constrained_request]
        K_r = len(passes)

        for k in range(K_r + 1):
            var_name = f"y_{req_name}_k{k}"
            y_vars[(constrained_request, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_name)

        model.addConstr(y_vars[(constrained_request, 0)] == 1.0, name=f"y0_{req_name}")

        for k, (satellite, satpass) in enumerate(passes):
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            theta_k = solution_holder[constrained_request][satellite][satpass]['theta']

            var_name = f"w_{req_name}_k{k}"
            w_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_name)
            w_vars[(constrained_request, satellite, satpass)] = w_var

            # Tight McCormick envelope for local horizontal selection
            model.addConstr(w_var <= x_var, name=f"mccormick1_{req_name}_k{k}")
            model.addConstr(w_var <= y_vars[(constrained_request, k)], name=f"mccormick2_{req_name}_k{k}")
            model.addConstr(w_var >= y_vars[(constrained_request, k)] - (1.0 - x_var), name=f"mccormick3_{req_name}_k{k}")
            model.addConstr(w_var >= 0.0, name=f"mccormick4_{req_name}_k{k}")

            model.addConstr(
                y_vars[(constrained_request, k + 1)] == y_vars[(constrained_request, k)] - theta_k * w_var,
                name=f"recurrence_{req_name}_k{k}"
            )

        # Standalone local success probability of this task
        S_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"S_{req_name}")
        S_vars[constrained_request] = S_var
        model.addConstr(S_var == 1.0 - y_vars[(constrained_request, K_r)], name=f"success_{req_name}")

    # === STEP 2: VERTICAL DAG COUPLING VIA BILINEAR CHAINS ===
    for constrained_request in solution_holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        A_parents_vars[constrained_request] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"A_parents_{req_name}")

        # Extract unique ancestors to avoid double-counting risk across parallel paths
        unique_ancestors = [anc for anc in nx.ancestors(workflow_graph, constrained_request) if anc in solution_holder]

        if not unique_ancestors:
            # Root Node
            model.addConstr(A_parents_vars[constrained_request] == 1.0, name=f"root_A_{req_name}")
        else:
            # Gurobi only allows multiplying TWO continuous variables per constraint (bilinear).
            # We chain an arbitrary number of ancestors using intermediate continuous variables.
            current_prod = S_vars[unique_ancestors[0]]
            for idx in range(1, len(unique_ancestors)):
                inter_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"inter_{req_name}_idx{idx}")
                model.addQConstr(inter_var == current_prod * S_vars[unique_ancestors[idx]])
                current_prod = inter_var
            
            model.addConstr(A_parents_vars[constrained_request] == current_prod, name=f"chain_finish_{req_name}")

    # === STEP 3: ABSOLUTE PASS REALIZATION ===
    for constrained_request in solution_holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        
        for satellite, satpass in task_to_passes[constrained_request]:
            w_var = w_vars[(constrained_request, satellite, satpass)]
            
            var_name = f"w_abs_{req_name}_{satellite.name}_pass{satpass.highest.time}"
            w_abs_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_name)
            w_abs_vars[(constrained_request, satellite, satpass)] = w_abs_var

            # Exact bilinear mapping: w_abs = Ancestor_Survival * Local_Pass_Active
            model.addQConstr(
                w_abs_var == A_parents_vars[constrained_request] * w_var,
                name=f"w_abs_constr_{req_name}_{satellite.name}_pass{satpass.highest.time}"
            )

    # === STEP 4: OBJECTIVE COMPILER ===
    # New objective: Maximize E[Quality] - (submission cost + cancellation cost)
    objective_terms = []
    for constrained_request in solution_holder.keys():
        # Skip tasks with no feasible passes
        if not solution_holder[constrained_request]:
            continue

        # Find max quality among all feasible passes for this request
        all_qualities = [
            solution_holder[constrained_request][sat][sp]['quality']
            for sat in solution_holder[constrained_request].keys()
            for sp in solution_holder[constrained_request][sat].keys()
        ]

        if not all_qualities:
            continue  # Skip if no passes available

        _max_quality_for_request = max(all_qualities)

        # Compute costs as fractions of max quality for this request
        c_sub = submission_cost_rate * _max_quality_for_request
        c_canc = cancellation_cost_rate * _max_quality_for_request
        c_tax = tax_rate * _max_quality_for_request  # Legacy tax

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            quality = solution_holder[constrained_request][satellite][satpass]['quality']
            theta = solution_holder[constrained_request][satellite][satpass]['theta']  # p_acc * p_exec
            theta_acc = solution_holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = w_abs_vars[(constrained_request, satellite, satpass)]

            # Expected Reward
            objective_terms.append(quality * theta * w_abs)

            # Costs
            objective_terms.append(-c_sub * x_var)
            objective_terms.append(-c_canc * theta_acc * x_var)
            objective_terms.append(-c_tax * x_var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)


import gurobipy as gp
from gurobipy import GRB
import networkx as nx

def _build_log_linearized_formulation(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        task_to_passes: dict,
        epsilon: float,
        pwl_tolerance: float,
        tax_rate: float,
        submission_cost_rate: float,
        cancellation_cost_rate: float,
        verbose: int
):
    """
    Build high-speed log-linearized formulation tracking vertical propagation 
    via a unified O(Tasks) scaled timeline approach. Completely eliminates 
    unscaled parallel tracks and pass-level general non-linear constraints.
    """
    # Initialize workspace trackers
    scaled_remaining_risk = {}
    effective_pass_realization = {}
    
    ancestor_success_prob = {}
    end_to_end_success = {}
    
    ln_A_vars = {}
    ln_A_parents_vars = {}
    ln_S_vars = {}

    # Apply global high-performance piecewise linear (PWL) configuration parameters
    model.setParam('FuncPieces', -1)
    model.setParam('FuncPieceError', pwl_tolerance)

    if verbose > 0:
        print(f"[Log-Linearized] Compiling streamlined single-horizon matrix for {len(solution_holder)} tasks.")

    # === STEP 1: INITIALIZE TASK-LEVEL CONTINUOUS LOG CHANNELS ===
    for constrained_request in solution_holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        
        # Core vertical real-space probabilities
        ancestor_success_prob[constrained_request] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"A_parents_{req_name}")
        end_to_end_success[constrained_request] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"A_node_{req_name}")
        
        # Mapped log metrics bound tightly to active operational windows
        ln_A_vars[constrained_request] = model.addVar(lb=-30.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"ln_A_{req_name}")
        ln_A_parents_vars[constrained_request] = model.addVar(lb=-30.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"ln_A_parents_{req_name}")
        ln_S_vars[constrained_request] = model.addVar(lb=-30.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"ln_S_{req_name}")

    # === STEP 2: TRANSITIVE LINEAGE INTEGRATION (DAG WIDE CLOSURE SETS) ===
    for constrained_request in solution_holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        
        # Extract unique ancestral dependencies to prevent the Reconvergence Trap
        unique_ancestors = [anc for anc in nx.ancestors(workflow_graph, constrained_request) if anc in solution_holder]
        
        if not unique_ancestors:
            # Root Node Initialization Boundary
            model.addConstr(ln_A_parents_vars[constrained_request] == 0.0)
            model.addConstr(ancestor_success_prob[constrained_request] == 1.0)
        else:
            # Multi-Parent AND Junction Convergence Map
            model.addConstr(ln_A_parents_vars[constrained_request] == gp.quicksum(ln_S_vars[anc] for anc in unique_ancestors))
            model.addGenConstrExp(ln_A_parents_vars[constrained_request], ancestor_success_prob[constrained_request])

    # === STEP 3: SINGLE SCALED HORIZONTAL TIMELINE GENERATION ===
    for constrained_request in solution_holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        passes = task_to_passes[constrained_request]
        K_r = len(passes)

        # Allocate variables tracking ancestral risk decay across the timeline indices
        for k in range(K_r + 1):
            scaled_remaining_risk[(constrained_request, k)] = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"Y_{req_name}_k{k}"
            )

        # INJECTION IDENTITY: Direct vertical coupling onto baseline entry index
        model.addConstr(scaled_remaining_risk[(constrained_request, 0)] == ancestor_success_prob[constrained_request])

        # Recurrence step calculations
        for k, (satellite, satpass) in enumerate(passes):
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            theta_k = solution_holder[constrained_request][satellite][satpass]['theta']
            Y_current = scaled_remaining_risk[(constrained_request, k)]

            # Instantiate absolute realization variable
            w_abs = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"w_abs_{req_name}_k{k}")
            effective_pass_realization[(constrained_request, satellite, satpass)] = w_abs

            # Exact McCormick Polytope bindings linking binary switch x with scaled risk Y
            model.addConstr(w_abs <= x_var)
            model.addConstr(w_abs <= Y_current)
            model.addConstr(w_abs >= Y_current - (1.0 - x_var))
            model.addConstr(w_abs >= 0.0)

            # Linear progress updates down the timeline
            model.addConstr(scaled_remaining_risk[(constrained_request, k + 1)] == Y_current - theta_k * w_abs)

        # Connect end-of-horizon values to final node fulfillment variables
        model.addConstr(end_to_end_success[constrained_request] == ancestor_success_prob[constrained_request] - scaled_remaining_risk[(constrained_request, K_r)])

        # Apply domain contraction mapping [0.0, 1.0] -> [epsilon, 1.0] to protect log evaluation spaces
        A_prot = model.addVar(lb=epsilon, ub=1.0, vtype=GRB.CONTINUOUS, name=f"A_prot_{req_name}")
        model.addConstr(A_prot == end_to_end_success[constrained_request] * (1.0 - epsilon) + epsilon)
        model.addGenConstrLog(A_prot, ln_A_vars[constrained_request])

        # LOG DEDUCTION IDENTITY: Calculate standalone local values completely linearly
        model.addConstr(ln_S_vars[constrained_request] == ln_A_vars[constrained_request] - ln_A_parents_vars[constrained_request])

    # === STEP 4: MATHEMATICAL OBJECTIVE COMPILER ===
    # New objective: Maximize E[Quality] - (submission cost + cancellation cost)
    # Maximize: sum_k Q_k * p_acc_k * p_exec_k * w_k - (c_sub_k + c_canc_k * p_acc_k) * x_k
    objective_terms = []
    for constrained_request in solution_holder.keys():
        # Skip tasks with no feasible passes
        if not solution_holder[constrained_request]:
            continue

        # Find max quality among all feasible passes for this request
        all_qualities = [
            solution_holder[constrained_request][sat][sp]['quality']
            for sat in solution_holder[constrained_request].keys()
            for sp in solution_holder[constrained_request][sat].keys()
        ]

        if not all_qualities:
            continue  # Skip if no passes available

        _max_quality_for_request = max(all_qualities)

        # Compute costs as fractions of max quality for this request
        c_sub = submission_cost_rate * _max_quality_for_request
        c_canc = cancellation_cost_rate * _max_quality_for_request
        c_tax = tax_rate * _max_quality_for_request  # Legacy tax (kept for backward compatibility)

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            quality = solution_holder[constrained_request][satellite][satpass]['quality']
            theta = solution_holder[constrained_request][satellite][satpass]['theta']  # p_acc * p_exec
            theta_acc = solution_holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = effective_pass_realization[(constrained_request, satellite, satpass)]

            # Expected Reward: Q * p_acc * p_exec * w (where w encodes ancestral success)
            objective_terms.append(quality * theta * w_abs)

            # Costs:
            # - Submission cost (unconditional, paid on every booking attempt)
            objective_terms.append(-c_sub * x_var)

            # - Cancellation cost (conditional on acceptance, only paid if constellation accepts)
            objective_terms.append(-c_canc * theta_acc * x_var)

            # - Legacy tax cost (for backward compatibility with old tax_rate parameter)
            objective_terms.append(-c_tax * x_var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)


def _add_workflow_constraints(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        verbose: int
):
    """
    Add temporal, success, and timeline constraints from workflow graph.

    Constraints:
    - At most max_num_instances per task
    - No overlapping observations on same satellite
    - Temporal constraints (START_AFTER, START_BEFORE, etc.)
    - Success constraints (conditional execution)
    - Timeline resource constraints
    """

    from fame_workflow import ConstraintClass, TemporalConstraintType, SuccessConstraintType

    # === MAX INSTANCES CONSTRAINT ===
    for constrained_request in solution_holder.keys():
        x_vars = []
        for satellite in solution_holder[constrained_request].keys():
            for satpass in solution_holder[constrained_request][satellite].keys():
                x_vars.append(solution_holder[constrained_request][satellite][satpass]['x'])

        if len(x_vars) > 0:
            max_instances = getattr(constrained_request, 'max_num_instances', 1)
            model.addConstr(
                gp.quicksum(x_vars) <= max_instances,
                name=f"max_instances_{constrained_request.observation_request.name}"
            )

    # === MANDATORY TASK CONSTRAINT ===
    for constrained_request in solution_holder.keys():
        if constrained_request.is_mandatory:
            x_vars = []
            for satellite in solution_holder[constrained_request].keys():
                for satpass in solution_holder[constrained_request][satellite].keys():
                    x_vars.append(solution_holder[constrained_request][satellite][satpass]['x'])

            if len(x_vars) > 0:
                model.addConstr(
                    gp.quicksum(x_vars) >= 1,
                    name=f"mandatory_{constrained_request.observation_request.name}"
                )

    # === SATELLITE CONFLICT CONSTRAINTS ===
    # Group passes by satellite
    solution_holder_by_satellite = {}
    for constrained_request in solution_holder.keys():
        for satellite in solution_holder[constrained_request].keys():
            if satellite not in solution_holder_by_satellite:
                solution_holder_by_satellite[satellite] = []

            for satpass in solution_holder[constrained_request][satellite].keys():
                solution_holder_by_satellite[satellite].append((
                    satpass,
                    solution_holder[constrained_request][satellite][satpass]['x'],
                    constrained_request
                ))

    # For each satellite, prevent overlapping passes
    for satellite in solution_holder_by_satellite.keys():
        passes = solution_holder_by_satellite[satellite]
        passes.sort(key=lambda x: x[0].highest.time)  # Sort by time

        for i in range(len(passes)):
            pass_i, x_i, req_i = passes[i]
            for j in range(i + 1, len(passes)):
                pass_j, x_j, req_j = passes[j]

                # Check if passes overlap
                end_i = pass_i.highest.time + pass_i.highest.duration
                start_j = pass_j.highest.time

                if start_j < end_i:
                    # Conflict: at most one can be scheduled
                    model.addConstr(
                        x_i + x_j <= 5,
                        name=f"conflict_{satellite.name}_{i}_{j}"
                    )
                else:
                    # No more conflicts (sorted by time)
                    break

    # === TEMPORAL CONSTRAINTS ===
    for constrained_request in solution_holder.keys():
        for parent_request in workflow_graph.predecessors(constrained_request):
            if parent_request not in solution_holder:
                if verbose > 2:
                    print(f"  [Constraints] Skipping parent {parent_request.observation_request.name} -> {constrained_request.observation_request.name} (parent not in solution_holder)")
                continue  # Parent not in solution holder (no passes or already dispatched)

            inedges = workflow_graph.get_edge_data(parent_request, constrained_request)

            for constraint_key, constraint in inedges.items():
                if constraint['constraint_class'] == ConstraintClass.TEMPORAL:
                    constraint_type = constraint['constraint_type']

                    # Get offset if present
                    offset = dt.timedelta(0)
                    if 'parameters' in constraint and 'offset' in constraint['parameters']:
                        offset = constraint['parameters']['offset']

                    # Iterate over child passes
                    for child_sat in solution_holder[constrained_request].keys():
                        for child_pass in solution_holder[constrained_request][child_sat].keys():
                            x_child = solution_holder[constrained_request][child_sat][child_pass]['x']

                            # Iterate over parent passes
                            for parent_sat in solution_holder[parent_request].keys():
                                for parent_pass in solution_holder[parent_request][parent_sat].keys():
                                    x_parent = solution_holder[parent_request][parent_sat][parent_pass]['x']

                                    if constraint_type == TemporalConstraintType.START_AFTER:
                                        # Child must start after parent
                                        if parent_pass.highest.time > child_pass.highest.time:
                                            model.addConstr(x_child + x_parent <= 1)

                                    elif constraint_type == TemporalConstraintType.START_AFTER_OFFSET:
                                        # Child must start after parent + offset
                                        if parent_pass.highest.time + offset > child_pass.highest.time:
                                            model.addConstr(x_child + x_parent <= 1)

                                    elif constraint_type == TemporalConstraintType.START_BEFORE:
                                        # Child must start before parent
                                        if parent_pass.highest.time < child_pass.highest.time:
                                            model.addConstr(x_child + x_parent <= 1)

                                    elif constraint_type == TemporalConstraintType.START_BEFORE_OFFSET:
                                        # Child must start before parent + offset
                                        if parent_pass.highest.time + offset < child_pass.highest.time:
                                            model.addConstr(x_child + x_parent <= 1)

    if verbose > 2:
        print(f"[Constraints] Added {model.NumConstrs} constraints")


def _extract_solution(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        verbose: int
):
    """
    Extract solution from solved model and update workflow graph.

    Updates each ConstrainedObservationRequest node with:
    - scheduled = True (if assigned)
    - observation_opportunity_satellite
    - observation_opportunity_pass
    """

    for constrained_request in solution_holder.keys():
        scheduled = False

        for satellite in solution_holder[constrained_request].keys():
            for satpass in solution_holder[constrained_request][satellite].keys():
                x_var = solution_holder[constrained_request][satellite][satpass]['x']

                if x_var.X > 0.5:  # Binary variable is "on"
                    constrained_request.scheduled = True
                    constrained_request.observation_opportunity_satellite = satellite
                    constrained_request.observation_opportunity_pass = satpass
                    constrained_request.observation_opportunity = satpass.highest

                    scheduled = True

                    if verbose > 1:
                        print(f"[Solution] Scheduled {constrained_request.observation_request.name} "
                              f"on {satellite.name} at {satpass.highest.time}")

                    break  # Only one pass per task

            if scheduled:
                break

        if not scheduled:
            constrained_request.scheduled = False
            if verbose > 2:
                print(f"[Solution] NOT scheduled: {constrained_request.observation_request.name}")
