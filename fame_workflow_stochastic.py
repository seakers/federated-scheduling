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
from fame_workflow import ConstrainedObservationRequest, TaskTimelineImpact, TaskImpactTime, Impact, Timeline

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

    # Step 2: Branch based on solver engine
    if solver_engine == "GUROBI":
        return _solve_with_gurobi(
            workflow_graph, timeline_graph, satellites, feasibility_screener,
            current_time, verbose, max_solver_time_s, receding_horizon_duration,
            stochastic_formulation, success_probability_function,
            acceptance_probability_function, execution_probability_function,
            epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate
        )
    elif solver_engine == "SCIP":
        return _solve_with_scip(
            workflow_graph, timeline_graph, satellites, feasibility_screener,
            current_time, verbose, max_solver_time_s, receding_horizon_duration,
            stochastic_formulation, success_probability_function,
            acceptance_probability_function, execution_probability_function,
            epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate
        )
    else:
        raise ValueError(f"Unknown solver_engine: {solver_engine}. Use 'GUROBI' or 'SCIP'.")


def _solve_with_gurobi(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list[Satellite],
        feasibility_screener: Callable,
        current_time: dt.datetime,
        verbose: int,
        max_solver_time_s: float,
        receding_horizon_duration: dt.timedelta,
        stochastic_formulation: str,
        success_probability_function: Callable,
        acceptance_probability_function: Callable,
        execution_probability_function: Callable,
        epsilon: float,
        pwl_tolerance: float,
        tax_rate: float,
        submission_cost_rate: float,
        execution_cost_rate: float
):
    """Solve stochastic scheduling problem using native Gurobi."""
    # Build Gurobi environment
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
            model.setParam('MIPGap', 0.05)  # 5% optimality gap (relaxed for faster feasible solutions)
            model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
            model.setParam('Heuristics', 0.2)  # Spend 20% of time on heuristics
            model.setParam('Presolve', 2)  # Aggressive presolve

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
                        print(f"[Stochastic Scheduler] Optimal solution found! Objective value: {model.ObjVal:.2f}")
                        print(f"[Stochastic Scheduler] (Reward - tax_rate={tax_rate}*max_quality_per_request*NumScheduled)")

                    _extract_solution(
                        model, workflow_graph, solution_holder, verbose, timeline_graph
                    )

                    # Store objective value on workflow graph for later analysis
                    workflow_graph.graph['objective_value'] = model.ObjVal
                elif model.Status in [GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED]:
                    # Solver hit time limit but may have found a feasible solution
                    if model.SolCount > 0:  # At least one feasible solution found
                        if verbose > 0:
                            print(f"[Stochastic Scheduler] Time limit reached, but feasible solution found! Objective: {model.ObjVal:.2f}")
                            print(f"[Stochastic Scheduler] (Not proven optimal, but using best solution found)")

                        _extract_solution(
                            model, workflow_graph, solution_holder, verbose, timeline_graph
                        )

                        workflow_graph.graph['objective_value'] = model.ObjVal
                    else:
                        if verbose > 0:
                            print(f"[Stochastic Scheduler] Time limit reached with no feasible solution found")
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

    # Clean up and return
    _cleanup_solver_objects(workflow_graph, timeline_graph)
    return workflow_graph


def _solve_with_scip(
        workflow_graph: nx.MultiDiGraph,
        timeline_graph: nx.MultiDiGraph,
        satellites: list[Satellite],
        feasibility_screener: Callable,
        current_time: dt.datetime,
        verbose: int,
        max_solver_time_s: float,
        receding_horizon_duration: dt.timedelta,
        stochastic_formulation: str,
        success_probability_function: Callable,
        acceptance_probability_function: Callable,
        execution_probability_function: Callable,
        epsilon: float,
        pwl_tolerance: float,
        tax_rate: float,
        submission_cost_rate: float,
        execution_cost_rate: float
):
    """
    Solve stochastic scheduling problem using OR-Tools SCIP with manual PWL approximations.

    NOTE: Only log_linearized formulation is supported with SCIP.
    Non-convex formulation requires Gurobi's quadratic solver.
    """
    from ortools.linear_solver import pywraplp

    if stochastic_formulation == "non_convex":
        raise ValueError(
            "Non-convex formulation requires Gurobi (quadratic constraints). "
            "Use stochastic_formulation='log_linearized' with SCIP."
        )

    if verbose > 0:
        print("[SCIP Stochastic] Building log-linearized formulation with manual PWL approximations...")

    # Create SCIP solver
# Create SCIP solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise ValueError("SCIP solver not available")

    solver.set_time_limit(int(max_solver_time_s * 1000))  # milliseconds
    
    # Configure SCIP to find high-quality feasible solutions quickly
    # Configure SCIP with valid key-value parameters
    # Configure SCIP with valid key-value parameters
    scip_params = (
        "limits/gap = 0.05\n"
        "display/verblevel = 0\n"
        "heuristics/rounding/freq = 1\n"
        "heuristics/shifting/freq = 1\n"
        "heuristics/rens/freq = 1\n"    
        "presolving/maxrounds = 3\n"     
        "separating/maxroundsroot = 3\n"
        "heuristics/rins/freq = 1\n"
        "heuristics/alns/freq = 1\n"
        "heuristics/feaspump/freq = 1"  # Aggressively forces feasibility checks early
    )
    solver.SetSolverSpecificParametersAsString(scip_params)

    objective = solver.Objective()

    # Step 3: Find observation opportunities and create variables
    solution_holder = {}
    task_to_passes = {}

    for constrained_request in workflow_graph.nodes():
        if (constrained_request.dispatched == True) or (constrained_request.completed == True):
            if verbose > 0:
                print(f"[SCIP Stochastic] Skipping {constrained_request.observation_request.name} (dispatched/completed)")
            continue

        # Find observation opportunities (same as Gurobi path)
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
                print(f"[SCIP Stochastic] No passes for {constrained_request.observation_request.name}")
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
                x_var = solver.BoolVar(var_name)

                # Compute two-stage probabilities
                if acceptance_probability_function is not None and execution_probability_function is not None:
                    p_acc = acceptance_probability_function(constrained_request, satellite, satpass)
                    p_exec = execution_probability_function(constrained_request, satellite, satpass)
                    p_total = p_acc * p_exec
                else:
                    p_total = success_probability_function(constrained_request, satellite, satpass)
                    p_acc = p_total
                    p_exec = 1.0

                solution_holder[constrained_request][satellite][satpass] = {
                    'x': x_var,
                    'quality': _quality,
                    'theta': p_total,
                    'theta_acc': p_acc,
                    'theta_exec': p_exec
                }

                task_to_passes[constrained_request].append((satellite, satpass))

        if not _found_a_pass:
            constrained_request.scheduled = True
            constrained_request.feasible = False
            if verbose > 1:
                print(f"[SCIP Stochastic] All passes infeasible for {constrained_request.observation_request.name}")

    # Step 4: Build log-linearized formulation with manual PWL
    # === ADVANCED FEASIBLE WARM-START GENERATION ENGINE ===
    import math

    greedy_chosen_pass = {}
    scheduled_intervals = {sat: [] for sat in satellites}
    
    # Sort requests prioritizing mandatory items to satisfy structural equality constraints
    sorted_reqs = list(solution_holder.keys())
    def _greedy_priority(r):
        if not solution_holder[r] or r not in task_to_passes:
            return (False, 0.0)
        max_q = max(solution_holder[r][sat][sp]['quality'] for sat, sp in task_to_passes[r])
        return (getattr(r, 'is_mandatory', False), max_q)
    sorted_reqs.sort(key=_greedy_priority, reverse=True)

    for req in sorted_reqs:
        if not solution_holder[req]:
            continue
        
        # Pull candidate intervals sorted by target quality
        candidates = []
        for sat, satpass in task_to_passes[req]:
            candidates.append((sat, satpass, solution_holder[req][sat][satpass]['quality']))
        candidates.sort(key=lambda x: x[2], reverse=True)

        placed = False
        for sat, satpass, _ in candidates:
            start = satpass.highest.time
            end = satpass.highest.time + satpass.highest.duration
            
            overlap = False
            for s_start, s_end in scheduled_intervals[sat]:
                if not (end <= s_start or start >= s_end):
                    overlap = True
                    break
            
            if not overlap:
                greedy_chosen_pass[req] = (sat, satpass)
                scheduled_intervals[sat].append((start, end))
                placed = True
                break
        
        # Force placement on mandatory nodes to protect constraint validity
        if getattr(req, 'is_mandatory', False) and not placed and candidates:
            greedy_chosen_pass[req] = (candidates[0][0], candidates[0][1])
            scheduled_intervals[candidates[0][0]].append((candidates[0][1].highest.time, candidates[0][1].highest.time + candidates[0][1].highest.duration))

    # Calculate exact matching network probability structures for the hint array
    hint_values_map = {}
    for req in solution_holder.keys():
        for sat in solution_holder[req].keys():
            for satpass in solution_holder[req][sat].keys():
                hint_values_map[solution_holder[req][sat][satpass]['x']] = 0.0
    for req, (sat, satpass) in greedy_chosen_pass.items():
        hint_values_map[solution_holder[req][sat][satpass]['x']] = 1.0

    # Downstream DAG real-space expectation propagation pipeline
    val_ln_S = {}
    for req in nx.topological_sort(workflow_graph):
        if req not in solution_holder or not solution_holder[req]:
            continue
        unique_ancestors = [anc for anc in nx.ancestors(workflow_graph, req) if anc in solution_holder]
        
        if not unique_ancestors:
            ln_A_parents = 0.0
            A_parents = 1.0
        else:
            ln_A_parents = max(-12.0, min(0.0, sum(val_ln_S[anc] for anc in unique_ancestors if anc in val_ln_S)))
            A_parents = math.exp(ln_A_parents)
            
        current_Y = A_parents
        for sat, satpass in task_to_passes[req]:
            x_val = hint_values_map[solution_holder[req][sat][satpass]['x']]
            w_abs_val = x_val * current_Y
            current_Y = current_Y - solution_holder[req][sat][satpass]['theta'] * w_abs_val
            
        end_to_end = A_parents - current_Y
        A_prot_val = end_to_end * (1.0 - epsilon) + epsilon
        ln_A = max(-12.0, min(0.0, math.log(A_prot_val)))
        val_ln_S[req] = max(-12.0, min(0.0, ln_A - ln_A_parents))

        # Store calculated continuous values into dictionary mapping for instantiation time
        hint_values_map[f"A_parents_{req.observation_request.name}"] = A_parents
        hint_values_map[f"A_node_{req.observation_request.name}"] = end_to_end
        hint_values_map[f"ln_A_{req.observation_request.name}"] = ln_A
        hint_values_map[f"ln_A_parents_{req.observation_request.name}"] = ln_A_parents
        hint_values_map[f"ln_S_{req.observation_request.name}"] = val_ln_S[req]

        current_Y_track = A_parents
        for k, (sat, satpass) in enumerate(task_to_passes[req]):
            hint_values_map[f"Y_{req.observation_request.name}_k{k}"] = current_Y_track
            x_val = hint_values_map[solution_holder[req][sat][satpass]['x']]
            w_abs_val = x_val * current_Y_track
            hint_values_map[f"w_abs_{req.observation_request.name}_k{k}"] = w_abs_val
            current_Y_track -= solution_holder[req][sat][satpass]['theta'] * w_abs_val
        hint_values_map[f"Y_{req.observation_request.name}_k{len(task_to_passes[req])}"] = current_Y_track
# Step 4: Build log-linearized formulation with manual PWL
        _build_scip_log_linearized_formulation(
            solver, workflow_graph, solution_holder, task_to_passes,
            epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate, verbose,
            hint_values_map  # <-- Add this to pass the generated hints!
        )

    # Step 5: Add constraints
    _add_scip_workflow_constraints(
        solver, workflow_graph, solution_holder, task_to_passes, verbose
    )

    # Step 6: Solve
    objective.SetMaximization()

    if verbose > 0:
        print(f"[SCIP Stochastic] Solving with {solver.NumVariables()} variables, {solver.NumConstraints()} constraints")

    if solver.NumVariables() == 0:
        if verbose > 0:
            print("[SCIP Stochastic] No pending tasks to schedule. Skipping optimization.")
        workflow_graph.graph['objective_value'] = 0.0
    else:
        # === INJECT WARM-START BASELINE ===
        all_binary_vars = []
        hint_values = []
        for constrained_request in solution_holder.keys():
            for satellite in solution_holder[constrained_request].keys():
                for satpass in solution_holder[constrained_request][satellite].keys():
                    all_binary_vars.append(solution_holder[constrained_request][satellite][satpass]['x'])
                    hint_values.append(0.0)

        if all_binary_vars:
            # Give SCIP a completely valid 'do nothing' baseline to guarantee a FEASIBLE status on timeout
            solver.SetHint(all_binary_vars, hint_values)

        # Step 6: Solve
        objective.SetMaximization()
        status = solver.Solve()

        # Enhanced diagnostics
        status_names = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
        if verbose > 0:
            print(f"[SCIP Stochastic] Solver status: {status} ({status_names[status] if status < len(status_names) else 'UNKNOWN'})")
            if status == pywraplp.Solver.NOT_SOLVED:
                print(f"[SCIP Stochastic] NOT_SOLVED means: hit time/iteration limit or couldn't solve")
                print(f"[SCIP Stochastic] Try: (1) increase max_solver_time_s, (2) reduce pwl_tolerance, or (3) use GUROBI")

        # Step 7: Extract solution
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            if verbose > 0:
                status_str = "Optimal" if status == pywraplp.Solver.OPTIMAL else "Feasible"
                print(f"[SCIP Stochastic] {status_str} solution found! Objective: {solver.Objective().Value():.2f}")

            _extract_scip_solution(
                solver, workflow_graph, solution_holder, verbose, timeline_graph
            )

            workflow_graph.graph['objective_value'] = solver.Objective().Value()
        elif status == pywraplp.Solver.INFEASIBLE:
            if verbose > 0:
                print("[SCIP Stochastic] Model is infeasible (no valid schedule found)")
        else:
            if verbose > 0:
                print(f"[SCIP Stochastic] Could not find solution (status: {status_names[status] if status < len(status_names) else status})")

    # Clean up and return
    _cleanup_solver_objects(workflow_graph, timeline_graph)
    return workflow_graph


def _manual_pwl_log(solver, x_var, result_var, epsilon=1e-5, num_segments=5):
    """
    Manual piecewise-linear approximation of ln(x) for OR-Tools with SOS2 constraints.

    Approximates: result_var ≈ ln(x_var) for x_var in [epsilon, 1.0]
    Uses num_segments linear pieces with proper SOS2 enforcement.
    """
    # Generate breakpoints
    x_min = epsilon
    x_max = 1.0
    breakpoints = np.linspace(x_min, x_max, num_segments + 1)
    n_breakpoints = len(breakpoints)

    # Lambda variables (one per breakpoint)
    lambda_vars = []
    for i in range(n_breakpoints):
        lambda_i = solver.NumVar(0, 1, f"{result_var.name()}_lambda_{i}")
        lambda_vars.append(lambda_i)

    # Binary variables for segment selection (one per segment)
    z_vars = []
    for i in range(num_segments):
        z_i = solver.BoolVar(f"{result_var.name()}_z_{i}")
        z_vars.append(z_i)

    # === BASIC PWL CONSTRAINTS ===
    # Sum of lambdas = 1
    solver.Add(sum(lambda_vars) == 1)

    # Exactly one segment is active
    solver.Add(sum(z_vars) == 1)

    # === SOS2 CONSTRAINTS ===
    # At most 2 adjacent lambdas can be non-zero (endpoints of the active segment)
    # lambda_0 <= z_0 (first breakpoint only active in first segment)
    solver.Add(lambda_vars[0] <= z_vars[0])

    # lambda_i <= z_{i-1} + z_i (middle breakpoints active in adjacent segments)
    for i in range(1, num_segments):
        solver.Add(lambda_vars[i] <= z_vars[i-1] + z_vars[i])

    # lambda_n <= z_{n-1} (last breakpoint only active in last segment)
    solver.Add(lambda_vars[n_breakpoints - 1] <= z_vars[num_segments - 1])

    # === PWL FUNCTION MAPPING ===
    # x = sum(lambda_i * breakpoint_i)
    solver.Add(x_var == sum(lambda_vars[i] * breakpoints[i] for i in range(n_breakpoints)))

    # result = sum(lambda_i * ln(breakpoint_i))
    log_values = [np.log(bp) for bp in breakpoints]
    solver.Add(result_var == sum(lambda_vars[i] * log_values[i] for i in range(n_breakpoints)))


def _manual_pwl_exp(solver, x_var, result_var, num_segments=10):
    """
    Manual piecewise-linear approximation of exp(x) for OR-Tools with SOS2 constraints.

    Approximates: result_var ≈ exp(x_var) for x_var in [-30, 0]
    Uses num_segments linear pieces with proper SOS2 enforcement.
    """
    x_min = -12.0
    x_max = 0.0
    breakpoints = np.linspace(x_min, x_max, num_segments + 1)
    n_breakpoints = len(breakpoints)

    # Lambda variables (one per breakpoint)
    lambda_vars = []
    for i in range(n_breakpoints):
        lambda_i = solver.NumVar(0, 1, f"{result_var.name()}_lambda_{i}")
        lambda_vars.append(lambda_i)

    # Binary variables for segment selection (one per segment)
    z_vars = []
    for i in range(num_segments):
        z_i = solver.BoolVar(f"{result_var.name()}_z_{i}")
        z_vars.append(z_i)

    # === BASIC PWL CONSTRAINTS ===
    # Sum of lambdas = 1
    solver.Add(sum(lambda_vars) == 1)

    # Exactly one segment is active
    solver.Add(sum(z_vars) == 1)

    # === SOS2 CONSTRAINTS ===
    # At most 2 adjacent lambdas can be non-zero (endpoints of the active segment)
    # lambda_0 <= z_0 (first breakpoint only active in first segment)
    solver.Add(lambda_vars[0] <= z_vars[0])

    # lambda_i <= z_{i-1} + z_i (middle breakpoints active in adjacent segments)
    for i in range(1, num_segments):
        solver.Add(lambda_vars[i] <= z_vars[i-1] + z_vars[i])

    # lambda_n <= z_{n-1} (last breakpoint only active in last segment)
    solver.Add(lambda_vars[n_breakpoints - 1] <= z_vars[num_segments - 1])

    # === PWL FUNCTION MAPPING ===
    # x = sum(lambda_i * breakpoint_i)
    solver.Add(x_var == sum(lambda_vars[i] * breakpoints[i] for i in range(n_breakpoints)))

    # result = sum(lambda_i * exp(breakpoint_i))
    exp_values = [np.exp(bp) for bp in breakpoints]
    solver.Add(result_var == sum(lambda_vars[i] * exp_values[i] for i in range(n_breakpoints)))


def _build_scip_log_linearized_formulation(
        solver, graph, holder, task_to_passes, eps, tol, tax, sub, canc, verbose, hint_values_map
):
    """
    Build log-linearized stochastic formulation using OR-Tools with manual PWL approximations.
    """
    scaled_remaining_risk = {}
    effective_pass_realization = {}
    ancestor_success_prob = {}
    end_to_end_success = {}
    ln_A_vars = {}
    ln_A_parents_vars = {}
    ln_S_vars = {}

    if verbose > 0:
        print(f"[SCIP Log-Linearized] Building formulation for {len(holder)} tasks with manual PWL.")

    # === STEP 1: INITIALIZE TASK-LEVEL CONTINUOUS LOG CHANNELS ===
    for constrained_request in holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))

        ancestor_success_prob[constrained_request] = solver.NumVar(0.0, 1.0, f"A_parents_{req_name}")
        end_to_end_success[constrained_request] = solver.NumVar(0.0, 1.0, f"A_node_{req_name}")
        
        # Optimization: Tight floor bounds [-12.0, 0.0] instead of [-30.0, 0.0]
        # eliminates degenerate numerical zones where exp(x) ≈ 0
        ln_A_vars[constrained_request] = solver.NumVar(-12.0, 0.0, f"ln_A_{req_name}")
        ln_A_parents_vars[constrained_request] = solver.NumVar(-12.0, 0.0, f"ln_A_parents_{req_name}")
        ln_S_vars[constrained_request] = solver.NumVar(-12.0, 0.0, f"ln_S_{req_name}")

        # Inject continuous hints directly into compile block
        if f"A_parents_{req_name}" in hint_values_map:
            ancestor_success_prob[constrained_request].SetHint(hint_values_map[f"A_parents_{req_name}"])
        if f"A_node_{req_name}" in hint_values_map:
            end_to_end_success[constrained_request].SetHint(hint_values_map[f"A_node_{req_name}"])
        if f"ln_A_{req_name}" in hint_values_map:
            ln_A_vars[constrained_request].SetHint(hint_values_map[f"ln_A_{req_name}"])
        if f"ln_A_parents_{req_name}" in hint_values_map:
            ln_A_parents_vars[constrained_request].SetHint(hint_values_map[f"ln_A_parents_{req_name}"])
        if f"ln_S_{req_name}" in hint_values_map:
            ln_S_vars[constrained_request].SetHint(hint_values_map[f"ln_S_{req_name}"])

    # === STEP 2: TRANSITIVE LINEAGE INTEGRATION ===
    for constrained_request in holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        unique_ancestors = [anc for anc in nx.ancestors(graph, constrained_request) if anc in holder]

        if not unique_ancestors:
            solver.Add(ln_A_parents_vars[constrained_request] == 0.0)
            solver.Add(ancestor_success_prob[constrained_request] == 1.0)
        else:
            solver.Add(ln_A_parents_vars[constrained_request] == sum(ln_S_vars[anc] for anc in unique_ancestors))
            _manual_pwl_exp(solver, ln_A_parents_vars[constrained_request], ancestor_success_prob[constrained_request])

    # === STEP 3: SINGLE SCALED HORIZONTAL TIMELINE GENERATION ===
    for constrained_request in holder.keys():
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        passes = task_to_passes[constrained_request]
        K_r = len(passes)

        for k in range(K_r + 1):
            risk_var_name = f"Y_{req_name}_k{k}"
            scaled_remaining_risk[(constrained_request, k)] = solver.NumVar(0.0, 1.0, risk_var_name)
            if risk_var_name in hint_values_map:
                scaled_remaining_risk[(constrained_request, k)].SetHint(hint_values_map[risk_var_name])

        solver.Add(scaled_remaining_risk[(constrained_request, 0)] == ancestor_success_prob[constrained_request])

        for k, (satellite, satpass) in enumerate(passes):
            x_var = holder[constrained_request][satellite][satpass]['x']
            theta_k = holder[constrained_request][satellite][satpass]['theta']
            Y_current = scaled_remaining_risk[(constrained_request, k)]

            w_abs_name = f"w_abs_{req_name}_k{k}"
            w_abs = solver.NumVar(0.0, 1.0, w_abs_name)
            effective_pass_realization[(constrained_request, satellite, satpass)] = w_abs

            # Apply warm-start hints to horizontal timeline variables
            if w_abs_name in hint_values_map:
                w_abs.SetHint(hint_values_map[w_abs_name])
            if x_var in hint_values_map:
                x_var.SetHint(hint_values_map[x_var])

            # McCormick envelope (linearized product approximation)
            solver.Add(w_abs <= x_var)
            solver.Add(w_abs <= Y_current)
            solver.Add(w_abs >= Y_current - (1.0 - x_var))
            solver.Add(w_abs >= 0.0)

            solver.Add(scaled_remaining_risk[(constrained_request, k + 1)] == Y_current - theta_k * w_abs)

        solver.Add(end_to_end_success[constrained_request] == ancestor_success_prob[constrained_request] - scaled_remaining_risk[(constrained_request, K_r)])

        # Log domain mapping with epsilon protection
        A_prot = solver.NumVar(eps, 1.0, f"A_prot_{req_name}")
        solver.Add(A_prot == end_to_end_success[constrained_request] * (1.0 - eps) + eps)
        _manual_pwl_log(solver, A_prot, ln_A_vars[constrained_request], eps)

        solver.Add(ln_S_vars[constrained_request] == ln_A_vars[constrained_request] - ln_A_parents_vars[constrained_request])

    # === STEP 4: OBJECTIVE COMPILER ===
    objective = solver.Objective()
    for constrained_request in holder.keys():
        if not holder[constrained_request]:
            continue

        all_qualities = [
            holder[constrained_request][sat][sp]['quality']
            for sat in holder[constrained_request].keys()
            for sp in holder[constrained_request][sat].keys()
        ]

        if not all_qualities:
            continue

        _max_quality_for_request = max(all_qualities)
        c_sub = sub * _max_quality_for_request
        c_canc = canc * _max_quality_for_request
        c_tax = tax * _max_quality_for_request

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = holder[constrained_request][satellite][satpass]['x']
            quality = holder[constrained_request][satellite][satpass]['quality']
            theta = holder[constrained_request][satellite][satpass]['theta']
            theta_acc = holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = effective_pass_realization[(constrained_request, satellite, satpass)]

            # Expected reward
            objective.SetCoefficient(w_abs, quality * theta)
            # Costs
            objective.SetCoefficient(x_var, -c_sub - c_canc * theta_acc - c_tax)

def _add_scip_workflow_constraints(
        solver,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        task_to_passes: dict,
        verbose: int
):
    """Add workflow constraints using OR-Tools."""
    from fame_workflow import ConstraintClass, TemporalConstraintType

    # Max instances
    for constrained_request in solution_holder.keys():
        x_vars = [
            solution_holder[constrained_request][sat][sp]['x']
            for sat in solution_holder[constrained_request].keys()
            for sp in solution_holder[constrained_request][sat].keys()
        ]
        if x_vars:
            max_instances = getattr(constrained_request, 'max_num_instances', 1)
            solver.Add(sum(x_vars) <= max_instances)

    # Mandatory tasks
    for constrained_request in solution_holder.keys():
        if constrained_request.is_mandatory:
            x_vars = [
                solution_holder[constrained_request][sat][sp]['x']
                for sat in solution_holder[constrained_request].keys()
                for sp in solution_holder[constrained_request][sat].keys()
            ]
            if x_vars:
                solver.Add(sum(x_vars) >= 1)

    # Satellite conflicts
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

    for satellite in solution_holder_by_satellite.keys():
        passes = solution_holder_by_satellite[satellite]
        passes.sort(key=lambda x: x[0].highest.time)

        for i in range(len(passes)):
            pass_i, x_i, req_i = passes[i]
            for j in range(i + 1, len(passes)):
                pass_j, x_j, req_j = passes[j]
                end_i = pass_i.highest.time + pass_i.highest.duration
                start_j = pass_j.highest.time
                if start_j < end_i:
                    solver.Add(x_i + x_j <= 1)
                else:
                    break

    # Temporal constraints
    for constrained_request in solution_holder.keys():
        for parent_request in workflow_graph.predecessors(constrained_request):
            if parent_request not in solution_holder:
                continue

            inedges = workflow_graph.get_edge_data(parent_request, constrained_request)
            for constraint_key, constraint in inedges.items():
                if constraint['constraint_class'] == ConstraintClass.TEMPORAL:
                    constraint_type = constraint['constraint_type']
                    offset = dt.timedelta(0)
                    if 'parameters' in constraint and 'offset' in constraint['parameters']:
                        offset = constraint['parameters']['offset']

                    for child_sat in solution_holder[constrained_request].keys():
                        for child_pass in solution_holder[constrained_request][child_sat].keys():
                            x_child = solution_holder[constrained_request][child_sat][child_pass]['x']

                            for parent_sat in solution_holder[parent_request].keys():
                                for parent_pass in solution_holder[parent_request][parent_sat].keys():
                                    x_parent = solution_holder[parent_request][parent_sat][parent_pass]['x']

                                    if constraint_type == TemporalConstraintType.START_AFTER:
                                        if parent_pass.highest.time > child_pass.highest.time:
                                            solver.Add(x_child + x_parent <= 1)
                                    elif constraint_type == TemporalConstraintType.START_AFTER_OFFSET:
                                        if parent_pass.highest.time + offset > child_pass.highest.time:
                                            solver.Add(x_child + x_parent <= 1)
                                    elif constraint_type == TemporalConstraintType.START_BEFORE:
                                        if parent_pass.highest.time < child_pass.highest.time:
                                            solver.Add(x_child + x_parent <= 1)
                                    elif constraint_type == TemporalConstraintType.START_BEFORE_OFFSET:
                                        if parent_pass.highest.time + offset < child_pass.highest.time:
                                            solver.Add(x_child + x_parent <= 1)


def _extract_scip_solution(
        solver,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        verbose: int,
        timeline_graph: nx.MultiDiGraph = None
):
    """Extract solution from OR-Tools solver."""
    from fame_workflow import TaskTimelineImpact, TaskImpactTime, Impact

    for constrained_request in solution_holder.keys():
        scheduled = False

        for satellite in solution_holder[constrained_request].keys():
            for satpass in solution_holder[constrained_request][satellite].keys():
                x_var = solution_holder[constrained_request][satellite][satpass]['x']

                if x_var.solution_value() > 0.5:
                    constrained_request.scheduled = True
                    constrained_request.observation_opportunity_satellite = satellite
                    constrained_request.observation_opportunity_pass = satpass
                    constrained_request.observation_opportunity = satpass.highest

                    scheduled = True

                    if verbose > 1:
                        print(f"[SCIP Solution] Scheduled {constrained_request.observation_request.name} "
                              f"on {satellite.name} at {satpass.highest.time}")

                    # Apply timeline impacts
                    if timeline_graph is not None and constrained_request in timeline_graph.nodes():
                        for _timeline in timeline_graph.successors(constrained_request):
                            tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                            for impact_key, impact in tl_edges.items():
                                if impact['edge_type'] == TaskTimelineImpact:
                                    _time = satpass.highest.time
                                    if impact['impact_time'] == TaskImpactTime.POST:
                                        _time = satpass.highest.time + satpass.highest.duration
                                    tl_impact = Impact(
                                        time=_time,
                                        type=impact['impact_type'],
                                        value=impact['impact_value'] * x_var.solution_value(),
                                        owner=constrained_request,
                                    )
                                    _timeline.add_impact(impact=tl_impact)

                    break

            if scheduled:
                break

        if not scheduled:
            constrained_request.scheduled = False
            if verbose > 2:
                print(f"[SCIP Solution] NOT scheduled: {constrained_request.observation_request.name}")


def _cleanup_solver_objects(workflow_graph: nx.MultiDiGraph, timeline_graph: nx.MultiDiGraph):
    """Remove unpicklable solver objects from graph structures."""
    # Clean up timeline impacts
    for timeline in timeline_graph.nodes():
        if type(timeline).__name__ == 'Timeline':
            _new_impact_container = []
            for impact in timeline.impact_container:
                impact_module = getattr(impact.value, '__module__', None)
                if (not (impact_module is not None and (impact_module.startswith('ortools') or impact_module.startswith('gurobipy')))):
                    _new_impact_container.append(impact)
            timeline.impact_container = _new_impact_container

    # Clean up workflow node attributes
    for node in workflow_graph.nodes():
        if hasattr(node, '__dict__'):
            for attr_name, attr_value in list(node.__dict__.items()):
                if attr_value is not None:
                    attr_module = getattr(attr_value, '__module__', None)
                    if attr_module is not None and (attr_module.startswith('gurobipy') or attr_module.startswith('ortools')):
                        setattr(node, attr_name, None)


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
                        x_i + x_j <= 1,
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


def _extract_solution(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        verbose: int,
        timeline_graph: nx.MultiDiGraph = None
):
    """
    Extract solution from solved model and update workflow graph.

    Updates each ConstrainedObservationRequest node with:
    - scheduled = True (if assigned)
    - observation_opportunity_satellite
    - observation_opportunity_pass
    - Applies timeline impacts for scheduled tasks
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

                    # Apply timeline impacts (matching deterministic scheduler behavior)
                    if timeline_graph is not None and constrained_request in timeline_graph.nodes():
                        for _timeline in timeline_graph.successors(constrained_request):
                            tl_edges = timeline_graph.get_edge_data(constrained_request, _timeline)
                            for impact_key, impact in tl_edges.items():
                                if impact['edge_type'] == TaskTimelineImpact:
                                    _time = satpass.highest.time
                                    if impact['impact_time'] == TaskImpactTime.POST:
                                        _time = satpass.highest.time + satpass.highest.duration
                                    tl_impact = Impact(
                                        time=_time,
                                        type=impact['impact_type'],
                                        value=impact['impact_value'] * x_var.X,
                                        owner=constrained_request,
                                    )
                                    _timeline.add_impact(impact=tl_impact)

                    break  # Only one pass per task

            if scheduled:
                break

        if not scheduled:
            constrained_request.scheduled = False
            if verbose > 2:
                print(f"[Solution] NOT scheduled: {constrained_request.observation_request.name}")
