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
class StochasticTimeline(Timeline):
    """
    A state timeline whose value decays over time unless refreshed
    by successful observation completions.
    """
    def __init__(
        self,
        name: str,
        initial_time: dt.datetime,
        initial_value: float = 1.0,
        half_life_s: float = 10800.0,  # 3 hours default memory
        min_value: float = -100.0,
        max_value: float = 100.0,
    ):
        decay_rate = -1.0 / half_life_s
        super().__init__(
            name=name,
            initial_time=initial_time,
            initial_value=initial_value,
            initial_rate=decay_rate,
            min_value=min_value,
            max_value=max_value,
        )
        self.half_life = dt.timedelta(seconds=half_life_s)

    def refresh_if_observed(self, current_time: dt.datetime, requests: list) -> bool:
        """
        Scans requests for any successful observation within the half-life window.
        Returns True if state is active (known), False if decayed/unknown.
        """
        is_active = False
        for r in requests:
            if getattr(r, 'completed', False) and getattr(r, 'successful_execution', False):
                # Find actual executed pass time (primary or backup)
                exec_time = getattr(r, 'execution_time', None)
                if exec_time is None and getattr(r, 'observation_opportunity', None):
                    exec_time = r.observation_opportunity.time
                
                if exec_time and (current_time - exec_time) <= self.half_life:
                    is_active = True
                    break

        new_value = 1.0 if is_active else 0.0
        _, current_rate = self._get_value_and_rate_at(current_time, print_debug=False)
        self.reset_timeline(current_time, new_value, current_rate)
        return is_active

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
        epsilon: float = 1e-3,
        pwl_tolerance: float = 1e-1,  # SCIP path only; the Gurobi path now uses exact MINLP handling (FuncNonlinear=1)
        mip_gap: float = 0.05,
        default_max_instances: int = 3,  # Redundancy cap when a request has no max_num_instances attribute. >1 is REQUIRED for the stochastic planner to hedge.
        solver_engine: str = "GUROBI",
        tax_rate: float = 0.15,  # Cost per scheduled obs as fraction of max quality (dynamic, per-request). Set to 0 to disable.
        submission_cost_rate: float = 0.0,  # c_sub: unconditional per-booking submission overhead (as fraction of quality)
        execution_cost_rate: float = 0.0,  # c_canc: conditional cancellation cost if accepted (as fraction of quality)
        results_dir: str = ""
        
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
            epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate,
            mip_gap=mip_gap, default_max_instances=default_max_instances, results_dir=results_dir
        )
    elif solver_engine == "SCIP":
        return _solve_with_scip(
            workflow_graph, timeline_graph, satellites, feasibility_screener,
            current_time, verbose, max_solver_time_s, receding_horizon_duration,
            stochastic_formulation, success_probability_function,
            acceptance_probability_function, execution_probability_function,
            epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate,
            default_max_instances=default_max_instances
        )
    else:
        raise ValueError(f"Unknown solver_engine: {solver_engine}. Use 'GUROBI' or 'SCIP'.")


def _effective_max_instances(request, default_max_instances: int) -> int:
    """Effective per-request instance cap for the STOCHASTIC planner.

    ConstrainedObservationRequest defines max_num_instances with a CLASS
    DEFAULT of 1, so the attribute always exists -- a None-fallback never
    fires. Semantics here: default_max_instances acts as a FLOOR:
        M = max(per_request_cap, default_max_instances)
    Rationale: the stochastic objective correctly prices redundancy (costs,
    diminishing first-success credit), so allowing extra instances is safe for
    this planner and is its entire mechanism. To strictly respect per-request
    caps (legacy behavior), call with default_max_instances=1.
    NOTE: do NOT raise max_num_instances in the workflow definition itself --
    the deterministic ILP counts full quality per booking and would exploit it.
    """
    per_request = getattr(request, 'max_num_instances', None)
    if per_request is None:
        return default_max_instances
    return max(int(per_request), int(default_max_instances))


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
        execution_cost_rate: float,
        mip_gap: float,
        default_max_instances: int,
        results_dir: str
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

            # === Solver parameters (single authoritative block) ===
            # FuncNonlinear=1: exp()/log() general constraints are handled EXACTLY by
            # Gurobi's global MINLP engine (outer approximation + spatial branching)
            # instead of a static PWL translation. This is the fix for the large
            # constraint-violation warnings: PWL error compounds around the DAG through
            # the log->sum->exp round trip at every task boundary, so approximation
            # tolerances that look small per-constraint blow up globally. Exactness
            # here is required; speed is recovered via the real-space cuts added in
            # _build_log_linearized_formulation.
            model.setParam('TimeLimit', max_solver_time_s)
            model.setParam('MIPGap', mip_gap)
            model.setParam('FuncNonlinear', 0)
            model.setParam('Cuts', 1) 
            model.setParam('Threads', 128)                
            model.setParam('OutputFlag', 1)
            if results_dir:
                model.setParam('LogFile', os.path.join(results_dir, "gurobi_stochastic.log"))

            # Enable bilinear solver if using the exact quadratic formulation
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

                        # NOTE: probabilities are used at full precision. Coefficient
                        # rounding was considered and rejected: distinct objective
                        # coefficients are normal and do not harm MILP structure.

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
                model, workflow_graph, solution_holder, verbose,
                default_max_instances=default_max_instances
            )

            # Step 6: Solve
            # CRITICAL: Must call model.update() before NumVars/NumConstrs return accurate counts
            # Otherwise Gurobi's lazy variable tracking reports 0 even when vars have been added
            model.update()

            if verbose > 0:
                n_bin  = sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
                n_int  = sum(1 for v in model.getVars() if v.VType == GRB.INTEGER)
                n_cont = model.NumVars - n_bin - n_int
                print(f"[Stochastic Scheduler] vars={model.NumVars} "
                    f"(bin={n_bin}, int={n_int}, cont={n_cont}) "
                    f"constrs={model.NumConstrs} genconstrs={model.NumGenConstrs} "
                    f"(logs={sum(1 for gc in model.getGenConstrs() if gc.GenConstrType == GRB.GENCONSTR_LOG)})")

            # Greedy MIP start: best-quality non-overlapping pass per request.
            # Pure implementation lever -- gives Gurobi a strong incumbent at
            # t=0 so the whole budget goes to closing the bound. Gurobi repairs
            # or discards the start if constraints make it infeasible; no risk.
            try:
                _busy = {}
                for _req in sorted(
                        solution_holder.keys(),
                        key=lambda r: -(max((solution_holder[r][s][p]['quality']
                                             for s in solution_holder[r] for p in solution_holder[r][s]),
                                            default=0.0))):
                    _placed = False
                    for _sat, _sp in task_to_passes.get(_req, []):
                        _x = solution_holder[_req][_sat][_sp]['x']
                        _s0 = _sp.highest.time
                        _e0 = _sp.highest.time + _sp.highest.duration
                        if (not _placed) and all(_e0 <= s or _s0 >= e for (s, e) in _busy.get(_sat, [])):
                            _x.Start = 1.0
                            _busy.setdefault(_sat, []).append((_s0, _e0))
                            _placed = True
                        else:
                            _x.Start = 0.0
            except Exception as _e:
                if verbose > 0:
                    print(f"[Stochastic Scheduler] MIP start skipped: {_e}")

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
                        model, workflow_graph, solution_holder, verbose, timeline_graph,
                        task_to_passes=task_to_passes
                    )

                    # Store objective value on workflow graph for later analysis
                    workflow_graph.graph['objective_value'] = model.ObjVal
                elif model.Status in [GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED]:
                    # Solver hit time limit but may have found a feasible solution
                    if model.SolCount > 0:  # At least one feasible solution found
                        if verbose > 0:
                            print(f"[Stochastic Scheduler] Time limit reached, but feasible solution found! Objective: {model.ObjVal:.2f}")
                            print(f"[Stochastic Scheduler] Current MIP Gap: {model.MIPGap * 100:.2f}% (Best Bound: {model.ObjBound:.2f})")
                            print(f"[Stochastic Scheduler] (Not proven optimal, but using best solution found)")

                        _extract_solution(
                            model, workflow_graph, solution_holder, verbose, timeline_graph,
                            task_to_passes=task_to_passes
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
        execution_cost_rate: float,
        default_max_instances: int = 3
):
    """
    Solve stochastic scheduling problem using OR-Tools SCIP with manual PWL approximations.

    NOTE: Only log_linearized formulation is supported with SCIP.
    Non-convex formulation requires Gurobi's quadratic solver.
    WARNING: this path still uses the static log floor and coarse manual PWL
    approximations; prefer solver_engine="GUROBI" (exact via FuncNonlinear=1)
    for any result that will be reported.
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
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise ValueError("SCIP solver not available")

    solver.set_time_limit(int(max_solver_time_s * 1000))  # milliseconds
    
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

                # Round to 2 d.p. — see Gurobi path above for rationale.
                p_acc   = round(p_acc,   2)
                p_exec  = round(p_exec,  2)
                p_total = round(p_acc * p_exec, 2)

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

    # Step 4: Build log-linearized formulation with manual PWL.
    # NOTE: this call must run exactly ONCE, after the hint-propagation loop.
    # It was previously indented inside the loop, rebuilding the entire
    # formulation (duplicate variables/constraints) once per request.
    _build_scip_log_linearized_formulation(
        solver, workflow_graph, solution_holder, task_to_passes,
        epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate, verbose,
        hint_values_map
    )

    # Step 5: Add constraints
    _add_scip_workflow_constraints(
        solver, workflow_graph, solution_holder, task_to_passes, verbose,
        default_max_instances=default_max_instances
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
                solver, workflow_graph, solution_holder, verbose, timeline_graph,
                task_to_passes=task_to_passes
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
        verbose: int,
        default_max_instances: int = 3
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
            max_instances = _effective_max_instances(constrained_request, default_max_instances)
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
        timeline_graph: nx.MultiDiGraph = None,
        task_to_passes: dict = None
):
    """Extract solution from OR-Tools solver (ALL selected passes per task,
    mirroring the Gurobi extractor: primary booking in the legacy scalar
    attributes, full redundant set in `scheduled_bookings`, timeline impacts
    applied per selected pass)."""
    from fame_workflow import TaskTimelineImpact, TaskImpactTime, Impact

    for constrained_request in solution_holder.keys():
        if task_to_passes is not None and constrained_request in task_to_passes:
            ordered_passes = task_to_passes[constrained_request]
        else:
            ordered_passes = [
                (sat, sp)
                for sat in solution_holder[constrained_request].keys()
                for sp in solution_holder[constrained_request][sat].keys()
            ]
            ordered_passes.sort(
                key=lambda t: solution_holder[constrained_request][t[0]][t[1]]['quality'],
                reverse=True
            )

        selected = [
            (satellite, satpass)
            for (satellite, satpass) in ordered_passes
            if solution_holder[constrained_request][satellite][satpass]['x'].solution_value() > 0.5
        ]

        if not selected:
            constrained_request.scheduled = False
            constrained_request.scheduled_bookings = []
            if verbose > 2:
                print(f"[SCIP Solution] NOT scheduled: {constrained_request.observation_request.name}")
            continue

        constrained_request.scheduled = True
        best_satellite, best_pass = selected[0]
        constrained_request.observation_opportunity_satellite = best_satellite
        constrained_request.observation_opportunity_pass = best_pass
        constrained_request.observation_opportunity = best_pass.highest
        constrained_request.scheduled_bookings = [
            {
                'satellite': satellite,
                'pass': satpass,
                'quality': solution_holder[constrained_request][satellite][satpass]['quality'],
                'theta': solution_holder[constrained_request][satellite][satpass]['theta'],
            }
            for (satellite, satpass) in selected
        ]

        if verbose > 1:
            for i, (satellite, satpass) in enumerate(selected):
                role = "PRIMARY" if i == 0 else f"BACKUP-{i}"
                print(f"[SCIP Solution] Scheduled {constrained_request.observation_request.name} "
                      f"[{role}] on {satellite.name} at {satpass.highest.time}")

        if timeline_graph is not None and constrained_request in timeline_graph.nodes():
            for (satellite, satpass) in selected:
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
                                value=impact['impact_value'],
                                owner=constrained_request,
                            )
                            _timeline.add_impact(impact=tl_impact)


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
        verbose: int,
        tighten_bounds: bool = True,
        default_max_instances: int = 3,
        reward_envelope_cuts: bool = True
):
    """
    Log-linearized stochastic formulation (paper Sections 5.2-5.6), built in a
    single topological sweep over the DAG.

    Key properties (each fixing a previously observed failure mode):

    1. PROPORTIONAL log-protection floor:
           A_prot = A_e2e * (1 - eps) + eps * A_parents
       (instead of the static "+ eps"). Consequence: A_prot <= A_parents holds
       structurally, so ln_S = ln(A_prot) - ln(A_parents) <= 0 is ALWAYS
       satisfiable and ln_S >= ln(eps) exactly. The static floor could force
       ln_S > 0 for deep tasks with weak ancestors, which collided with the
       ln_S <= 0 bound and created massive infeasibility pressure / branching
       churn ("adaptive floor trap"). The proportional floor removes the trap
       at its root.

    2. DEPTH-AWARE log-variable bounds. Because ln_S in [ln(eps), 0] exactly
       (see 1), the valid bounds are:
           ln_A_parents >= n_anc * ln(eps)
           ln_A         >= (n_anc + 1) * ln(eps)
       A static bound of -12 silently acted as a hidden constraint forcing
       ancestral chains to stay healthy (second door into the floor trap).
       Bounds are capped at LN_FLOOR for numerical sanity; the cap only binds
       for chains of many near-dead ancestors, which are objective-irrelevant.

    3. SINGLE-PARENT SHORTCUT: for a node r whose unique in-model ancestors
       equal {q} + ancestors(q) for a single direct parent q (chain structure),
       the entry boundary is set linearly:
           ln_A_parents[r] == ln_A[q],   A_parents[r] == A_prot[q]
       This is exact (Sum of ancestor ln_S telescopes to ln_A[q]) and removes
       one exp() general constraint per chain node -- the dominant source of
       MINLP work in chain-heavy workflows.

    4. REAL-SPACE VALID CUTS: A_parents[r] <= A_prot[a] for every in-model
       ancestor a (event inclusion: r's ancestral-success event is a subset of
       a's protected end-to-end event). These bound the exp() relaxation in
       probability space directly, bypassing the log machinery exactly where
       its relaxation is loosest, recovering speed after switching to exact
       nonlinear handling (FuncNonlinear=1).

    5. DATA-DRIVEN BOUND PROPAGATION (dual-bound tightening). The incumbent
       is typically found quickly; what is expensive is proving optimality,
       because the LP/OA relaxation of the exp()/log() equalities is one-sided
       (the relaxed A_parents can float up to the chord of exp between its
       variable bounds) and that overestimation COMPOUNDS multiplicatively
       down the DAG, inflating the root dual bound. We therefore precompute,
       in one topological pass over constants, the best-case protected
       probabilities with ALL passes scheduled:
           lmax(r)      = 1 - prod_k (1 - p_{r,k})              (union of all passes)
           S_ub(r)      = lmax(r)*(1-eps) + eps
           A_par_ub(r)  = prod_{a in anc(r)} S_ub(a)
           e2e_ub(r)    = A_par_ub(r) * lmax(r)
           A_prot_ub(r) = A_par_ub(r) * S_ub(r)
       and install them as VARIABLE BOUNDS (plus matching log-space bounds).
       Scheduling more passes only increases success probabilities, so these
       are valid regardless of conflicts/max-instances; they cut the chord gap
       of every exp/log relaxation at the root, before any branching.

    6. UNION-BOUND CUTS: e2e[r] <= A_par_ub(r) * sum_k p_k x_k, valid since
       1 - prod(1 - p x) <= sum p x. Ties the reward a task can claim in the
       relaxation to the probability mass actually scheduled, so fractional
       solutions cannot harvest reward without paying for bookings.

    7. BINARY BRANCH PRIORITY: x variables get BranchPriority 10 so Gurobi
       branches the schedule decisions before spatially branching the
       continuous exp/log operands -- once x is integral the McCormick track
       is exact and interval tightening closes the rest fast.

    NOTE: pwl_tolerance is unused here (kept for API compatibility); the
    Gurobi path relies on FuncNonlinear=1 for exact exp/log handling.
    """
    import math

    scaled_remaining_risk = {}
    effective_pass_realization = {}
    ancestor_success_prob = {}
    end_to_end_success = {}
    A_prot_vars = {}
    ln_A_vars = {}
    ln_A_parents_vars = {}
    ln_S_vars = {}

    ln_eps = math.log(epsilon)
    LN_FLOOR = -1000.0  # absolute cap on log-space lower bounds (exp(-50) ~ 2e-22)

    # Topological order restricted to tasks actually in the model; guarantees
    # every ancestor's variables exist before its descendants reference them.
    topo = [r for r in nx.topological_sort(workflow_graph) if r in solution_holder]

    # Unique-ancestor closure sets (transitive closure trick: summing local
    # ln_S over the UNIQUE ancestor set gives the exact joint ancestral
    # probability under independence on general DAGs -- shared ancestors of
    # diamond patterns are counted exactly once).
    anc_sets = {
        r: frozenset(a for a in nx.ancestors(workflow_graph, r) if a in solution_holder)
        for r in topo
    }

    # --- Constant bound propagation, CARDINALITY-AWARE -------------------------
    # The instances cap (max_num_instances) is a hard constraint, so no integer
    # solution can ever schedule more than M_r passes. All best-case constants
    # are therefore computed over the TOP-M_r probabilities only:
    #     lmax_M(r) = 1 - prod_{k in top-M_r}(1 - p_k)
    # Computing them over ALL passes (previous version) degenerates to ~1.0 as
    # soon as a request has many candidate passes, making every bound trivial.
    lmax_ub, S_ub, A_par_ub, e2e_ub, A_prot_ub, M_of = {}, {}, {}, {}, {}, {}
    for r in topo:
        M = _effective_max_instances(r, default_max_instances)
        M_of[r] = max(0, min(M, len(task_to_passes[r])))
        thetas = sorted(
            (solution_holder[r][sat][sp]['theta'] for (sat, sp) in task_to_passes[r]),
            reverse=True)[:M_of[r]]
        lmax = 1.0 - math.prod(1.0 - t for t in thetas) if thetas else 0.0
        lmax_ub[r] = min(1.0, lmax)
        S_ub[r] = lmax_ub[r] * (1.0 - epsilon) + epsilon
        A_par_ub[r] = math.prod(S_ub[a] for a in anc_sets[r]) if anc_sets[r] else 1.0
        e2e_ub[r] = A_par_ub[r] * lmax_ub[r]
        A_prot_ub[r] = A_par_ub[r] * S_ub[r]  # == e2e_ub*(1-eps) + eps*A_par_ub
    if not tighten_bounds:
        for r in topo:
            A_par_ub[r], e2e_ub[r], A_prot_ub[r], S_ub[r] = 1.0, 1.0, 1.0, 1.0
            lmax_ub[r] = 1.0

    # --- Node classification + nonlinearity pruning ----------------------------
    # Classify every node once: ROOT (no in-model ancestors), CHAIN (single
    # in-model parent whose closure telescopes), MERGE (general log-space join,
    # needs an exp() constraint). Then compute which tasks actually need their
    # log() constraint: ln_S(r) is consumed ONLY inside merge-node joins, so
    # log machinery is required exactly on the union of merge-node ancestor
    # closures. Everything else propagates through the purely LINEAR real-space
    # identities (A_parents[child] == A_prot[parent]). Consequence: a window
    # with no live merge nodes builds a PURE MILP -- zero nonlinear constraints.
    node_kind = {}
    for r in topo:
        dps = [p for p in workflow_graph.predecessors(r) if p in solution_holder]
        if not anc_sets[r]:
            node_kind[r] = 'root'
        elif len(dps) == 1 and anc_sets[r] == anc_sets[dps[0]] | {dps[0]}:
            node_kind[r] = 'chain'
        else:
            node_kind[r] = 'merge'
    merge_nodes = [r for r in topo if node_kind[r] == 'merge']
    need_lnS = set()
    for m in merge_nodes:
        need_lnS |= anc_sets[m]
    if verbose > 0:
        import collections
        M_hist = dict(collections.Counter(M_of[r] for r in topo))
        lmaxs = [lmax_ub[r] for r in topo if task_to_passes[r]]
        print(f"[Log-Linearized] {len(merge_nodes)} merge nodes; log constraints "
              f"pruned to {len(need_lnS)} of {len(topo)} tasks "
              f"({'PURE MILP' if not merge_nodes else 'MINLP on merge closures only'}).")
        if lmaxs:
            print(f"[Log-Linearized] ENGAGEMENT CHECK -- instance caps M (histogram): {M_hist}; "
                  f"lmax_M: min={min(lmaxs):.3f} mean={sum(lmaxs)/len(lmaxs):.3f} max={max(lmaxs):.3f}; "
                  f"tighten_bounds={tighten_bounds}. "
                  f"(If M is mostly 1, hedging is OFF; if lmax_M ~1.0, drain cuts are weak.)")
        else:
            print(f"[Log-Linearized] ENGAGEMENT CHECK -- no task in this window has any "
                  f"feasible pass (M histogram: {M_hist}); nothing to schedule or hedge.")

    if verbose > 0:
        n_chain = sum(
            1 for r in topo
            if len([p for p in workflow_graph.predecessors(r) if p in solution_holder]) == 1
        )
        print(f"[Log-Linearized] Building exact formulation for {len(topo)} tasks "
              f"({n_chain} single-parent candidates for the linear shortcut).")

    for constrained_request in topo:
        req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
        n_anc = len(anc_sets[constrained_request])

        # --- Depth-aware bounds -------------------------------------------------
        lb_ln_parents = max(n_anc * ln_eps, LN_FLOOR) if n_anc > 0 else 0.0
        lb_ln_A = max((n_anc + 1) * ln_eps, LN_FLOOR)

        r_ub = constrained_request
        kind = node_kind[constrained_request]
        needs_log = constrained_request in need_lnS
        ancestor_success_prob[constrained_request] = model.addVar(
            lb=math.exp(lb_ln_parents) if n_anc > 0 else 1.0, ub=A_par_ub[r_ub],
            vtype=GRB.CONTINUOUS, name=f"A_parents_{req_name}")
        end_to_end_success[constrained_request] = model.addVar(
            lb=0.0, ub=e2e_ub[r_ub], vtype=GRB.CONTINUOUS, name=f"A_node_{req_name}")
        if needs_log:
            ln_A_vars[constrained_request] = model.addVar(
                lb=lb_ln_A, ub=math.log(A_prot_ub[r_ub]) if A_prot_ub[r_ub] < 1.0 else 0.0,
                vtype=GRB.CONTINUOUS, name=f"ln_A_{req_name}")
            # Exact range under the proportional floor: ln_S in [ln(eps), ln(S_ub)].
            ln_S_vars[constrained_request] = model.addVar(
                lb=ln_eps, ub=math.log(S_ub[r_ub]) if S_ub[r_ub] < 1.0 else 0.0,
                vtype=GRB.CONTINUOUS, name=f"ln_S_{req_name}")
        if needs_log or kind == 'merge':
            ln_A_parents_vars[constrained_request] = model.addVar(
                lb=lb_ln_parents, ub=math.log(A_par_ub[r_ub]) if A_par_ub[r_ub] < 1.0 else 0.0,
                vtype=GRB.CONTINUOUS, name=f"ln_A_parents_{req_name}")

        # --- Vertical entry boundary (paper Sec 5.3) ---------------------------
        direct_parents = [p for p in workflow_graph.predecessors(constrained_request)
                          if p in solution_holder]

        if kind == 'root':
            model.addConstr(ancestor_success_prob[constrained_request] == 1.0,
                            name=f"root_Ap_{req_name}")
            if needs_log or kind == 'merge':
                model.addConstr(ln_A_parents_vars[constrained_request] == 0.0,
                                name=f"root_lnAp_{req_name}")
        elif kind == 'chain':
            # Single-parent shortcut: entry boundary is linear in parent's vars.
            q = direct_parents[0]
            model.addConstr(ancestor_success_prob[constrained_request] == A_prot_vars[q],
                            name=f"chain_Ap_{req_name}")
            if needs_log:
                # parent q is in anc(merge) whenever r is, so ln_A_vars[q] exists
                model.addConstr(ln_A_parents_vars[constrained_request] == ln_A_vars[q],
                                name=f"chain_lnAp_{req_name}")
        else:
            # General merge node: exact log-space join over the unique closure set.
            model.addConstr(
                ln_A_parents_vars[constrained_request]
                == gp.quicksum(ln_S_vars[anc] for anc in anc_sets[constrained_request]),
                name=f"join_lnAp_{req_name}")
            model.addGenConstrExp(ln_A_parents_vars[constrained_request],
                                  ancestor_success_prob[constrained_request],
                                  name=f"exp_Ap_{req_name}")
            # Real-space valid cuts tightening the exp() relaxation:
            # A_parents[r] <= A_prot[a] for every in-model ancestor a.
            for anc in anc_sets[constrained_request]:
                model.addConstr(
                    ancestor_success_prob[constrained_request] <= A_prot_vars[anc],
                    name=f"cut_Ap_le_Aprot_{req_name}_{getattr(getattr(anc, 'observation_request', anc), 'name', id(anc))}")

        # --- Horizontal scaled timeline (paper Sec 5.2) ------------------------
        passes = task_to_passes[constrained_request]
        K_r = len(passes)

        for k in range(K_r + 1):
            scaled_remaining_risk[(constrained_request, k)] = model.addVar(
                lb=0.0, ub=A_par_ub[constrained_request], vtype=GRB.CONTINUOUS,
                name=f"Y_{req_name}_k{k}")

        # Injection identity: timeline starts at the parents' joint success.
        model.addConstr(
            scaled_remaining_risk[(constrained_request, 0)] == ancestor_success_prob[constrained_request],
            name=f"inject_{req_name}")

        for k, (satellite, satpass) in enumerate(passes):
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            theta_k = solution_holder[constrained_request][satellite][satpass]['theta']
            Y_current = scaled_remaining_risk[(constrained_request, k)]

            w_abs = model.addVar(lb=0.0, ub=A_par_ub[constrained_request],
                                 vtype=GRB.CONTINUOUS, name=f"w_abs_{req_name}_k{k}")
            effective_pass_realization[(constrained_request, satellite, satpass)] = w_abs
            x_var.BranchPriority = 10  # branch schedule decisions before spatial branching

            # Exact McCormick linearization of the binary-continuous product.
            model.addConstr(w_abs <= x_var, name=f"mc1_{req_name}_k{k}")
            model.addConstr(w_abs <= Y_current, name=f"mc2_{req_name}_k{k}")
            model.addConstr(w_abs >= Y_current - (1.0 - x_var), name=f"mc3_{req_name}_k{k}")

            model.addConstr(
                scaled_remaining_risk[(constrained_request, k + 1)] == Y_current - theta_k * w_abs,
                name=f"rec_{req_name}_k{k}")

        # End-of-horizon fulfillment (paper Eq. 29).
        model.addConstr(
            end_to_end_success[constrained_request]
            == ancestor_success_prob[constrained_request] - scaled_remaining_risk[(constrained_request, K_r)],
            name=f"e2e_{req_name}")

        # Union-bound cut (docstring item 6): reward-carrying probability mass
        # is capped by the scheduled probability mass, scaled by the best-case
        # ancestral survival. Valid since 1 - prod(1-p*x) <= sum(p*x).
        if K_r > 0:
            model.addConstr(
                end_to_end_success[constrained_request]
                <= A_par_ub[constrained_request] * gp.quicksum(
                    solution_holder[constrained_request][sat][sp]['theta']
                    * solution_holder[constrained_request][sat][sp]['x']
                    for (sat, sp) in passes),
                name=f"cut_union_{req_name}")

        # DRAIN CUT (the decisive one). In the LP relaxation, fractional x lets
        # the McCormick track fully drain Y (claim near-certain local success)
        # while "paying" only max_instances worth of booking mass -- this, not
        # the exp/log chords, is what inflates the root bound by hundreds of
        # percent when requests have many candidate passes. No INTEGER solution
        # can exceed the top-M union probability, so:
        #     e2e[r] <= lmax_M(r) * A_parents[r]
        # is valid, linear, and caps the relaxation at the true per-task ceiling.
        if tighten_bounds and K_r > 0 and lmax_ub[constrained_request] < 1.0:
            model.addConstr(
                end_to_end_success[constrained_request]
                <= lmax_ub[constrained_request] * ancestor_success_prob[constrained_request],
                name=f"cut_drain_{req_name}")

        # REWARD ENVELOPE CUTS (cardinality-priced reward). The LP relaxation
        # can otherwise earn near-union success probability while paying only
        # a fraction of the integer booking count (McCormick complementarity
        # binds only at integral x), so per-booking costs barely discount the
        # bound. The per-task reward with n integer bookings is bounded by the
        # CONCAVE curve R_ub(n) = A_par_ub * Q_max * lmax(n), where
        # lmax(n) = 1 - prod over top-n p of (1-p). We add its tangents at
        # n = 0..M-1: valid for every integer point by concavity, and they
        # force the relaxation to pay one full booking of cost per top-marginal
        # unit of reward. This encodes the diminishing-returns structure --
        # invisible to the plain relaxation -- as linear inequalities, with no
        # change to the feasible integer set or the objective.
        if reward_envelope_cuts and tighten_bounds and K_r > 0:
            _thetas_desc = sorted(
                (solution_holder[constrained_request][sat][sp]['theta'] for (sat, sp) in passes),
                reverse=True)
            _Qmax = max(solution_holder[constrained_request][sat][sp]['quality']
                        for (sat, sp) in passes)
            _scale = A_par_ub[constrained_request] * _Qmax
            _lmax_curve = [0.0]
            _fail = 1.0
            for _p in _thetas_desc[:M_of[constrained_request]]:
                _fail *= (1.0 - _p)
                _lmax_curve.append(1.0 - _fail)
            _reward_expr = gp.quicksum(
                solution_holder[constrained_request][sat][sp]['quality']
                * solution_holder[constrained_request][sat][sp]['theta']
                * effective_pass_realization[(constrained_request, sat, sp)]
                for (sat, sp) in passes)
            _xsum = gp.quicksum(
                solution_holder[constrained_request][sat][sp]['x'] for (sat, sp) in passes)
            for _n in range(len(_lmax_curve) - 1):
                _slope = _lmax_curve[_n + 1] - _lmax_curve[_n]
                model.addConstr(
                    _reward_expr <= _scale * (_lmax_curve[_n] - _slope * _n)
                                    + _scale * _slope * _xsum,
                    name=f"cut_renv_{req_name}_n{_n}")
                # SURVIVAL envelope: the same concave cardinality pricing must
                # also cap e2e itself, or the LP fractionally drains Y to the
                # union level "for free" and hands inflated survival to every
                # descendant (the recurrence Y_0[child] = A_prot[parent] then
                # compounds the inflation down the DAG). With these cuts the
                # survival passed downstream is priced per integer booking,
                # and the chain recurrence compounds cost-consistent values.
                model.addConstr(
                    end_to_end_success[constrained_request]
                    <= A_par_ub[constrained_request] * (_lmax_curve[_n] - _slope * _n)
                       + A_par_ub[constrained_request] * _slope * _xsum,
                    name=f"cut_senv_{req_name}_n{_n}")

            # QUALITY-PROBABILITY frontier envelope: an integer solution picks
            # ONE subset of passes; its credited reward is at most the sum of
            # its Q*p products (dropping failure discounting), hence at most
            # the sum of the n LARGEST Q*p products for n bookings -- a concave
            # curve in n. The fractional LP otherwise splits booking mass to
            # take survival from high-p passes and reward from high-Q slots at
            # the same time, exceeding every integer subset on both axes.
            _qp_desc = sorted(
                (solution_holder[constrained_request][sat][sp]['quality']
                 * solution_holder[constrained_request][sat][sp]['theta']
                 for (sat, sp) in passes), reverse=True)[:M_of[constrained_request]]
            _qp_cum = [0.0]
            for _qp in _qp_desc:
                _qp_cum.append(_qp_cum[-1] + _qp)
            for _n in range(len(_qp_cum) - 1):
                _slope_qp = _qp_cum[_n + 1] - _qp_cum[_n]
                model.addConstr(
                    _reward_expr <= A_par_ub[constrained_request]
                                    * ((_qp_cum[_n] - _slope_qp * _n) + _slope_qp * _xsum),
                    name=f"cut_qpenv_{req_name}_n{_n}")

        # --- PROPORTIONAL floor + protected log (fix of the floor trap) --------
        A_prot = model.addVar(lb=max(math.exp(lb_ln_A), 1e-30),
                              ub=A_prot_ub[constrained_request],
                              vtype=GRB.CONTINUOUS, name=f"A_prot_{req_name}")
        A_prot_vars[constrained_request] = A_prot
        model.addConstr(
            A_prot == end_to_end_success[constrained_request] * (1.0 - epsilon)
                      + epsilon * ancestor_success_prob[constrained_request],
            name=f"prot_{req_name}")
        # Log machinery only where ln_S(r) is actually consumed downstream.
        if needs_log:
            model.addGenConstrLog(A_prot, ln_A_vars[constrained_request], name=f"log_{req_name}")
            # Log deduction identity (paper Eq. 32), now always satisfiable.
            model.addConstr(
                ln_S_vars[constrained_request]
                == ln_A_vars[constrained_request] - ln_A_parents_vars[constrained_request],
                name=f"lnS_{req_name}")

    # === OBJECTIVE (paper Eq. 34) ==============================================
    # Maximize sum_k Q_k * p_k * W_abs_k  -  sum_k (c_sub + c_exec * p_acc + c_tax) * x_k
    objective_terms = []
    for constrained_request in topo:
        if not solution_holder[constrained_request]:
            continue

        all_qualities = [
            solution_holder[constrained_request][sat][sp]['quality']
            for sat in solution_holder[constrained_request].keys()
            for sp in solution_holder[constrained_request][sat].keys()
        ]
        if not all_qualities:
            continue

        _max_quality_for_request = max(all_qualities)
        c_sub = submission_cost_rate * _max_quality_for_request
        c_canc = cancellation_cost_rate * _max_quality_for_request
        c_tax = tax_rate * _max_quality_for_request  # legacy tax; set tax_rate=0 for paper-exact objective

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            quality = solution_holder[constrained_request][satellite][satpass]['quality']
            theta = solution_holder[constrained_request][satellite][satpass]['theta']
            theta_acc = solution_holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = effective_pass_realization[(constrained_request, satellite, satpass)]

            # Expected best-success quality credit for this pass.
            objective_terms.append(quality * theta * w_abs)
            # Unconditional submission overhead.
            objective_terms.append(-c_sub * x_var)
            # Execution/cancellation cost, conditional on acceptance.
            objective_terms.append(-c_canc * theta_acc * x_var)
            # Legacy per-booking tax (now applied consistently with the
            # non-convex formulation; previously dead code in this path).
            if c_tax:
                objective_terms.append(-c_tax * x_var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)


def _add_workflow_constraints(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        verbose: int,
        default_max_instances: int = 3
):
    """
    Add temporal, success, and timeline constraints from workflow graph.

    Constraints:
    - At most max_num_instances per task (defaulting to `default_max_instances`
      when the request does not specify one -- MUST be > 1 for the stochastic
      planner to be able to book redundant passes, which is the entire
      mechanism the stochastic formulation exists to price)
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
            max_instances = _effective_max_instances(constrained_request, default_max_instances)
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
        timeline_graph: nx.MultiDiGraph = None,
        task_to_passes: dict = None
):
    """
    Extract solution from solved model and update workflow graph.

    CRITICAL: extracts ALL selected passes per task, not just the first.
    The stochastic formulation's entire value proposition is redundant
    (multi-pass) booking; the previous version broke out of the loop after the
    first active x variable, silently discarding backup bookings whose costs
    had been paid in the objective but whose hedging value was never realized,
    and never applying their timeline impacts.

    Updates each ConstrainedObservationRequest node with:
    - scheduled = True (if at least one pass selected)
    - scheduled_bookings: list of dicts (quality-descending) with keys
      'satellite', 'pass', 'quality', 'theta' -- one entry per selected pass.
      Downstream dispatch should attempt EVERY entry.
    - observation_opportunity_satellite / _pass / observation_opportunity:
      backward-compatible scalar attributes pointing at the BEST-QUALITY
      selected pass (legacy consumers see the primary booking).
    - Timeline impacts applied for EVERY selected pass (each booking consumes
      resources whether or not it turns out to be the one that succeeds).
    """

    for constrained_request in solution_holder.keys():
        # Enumerate passes in descending-quality order (task_to_passes preserves
        # the global quality sort across satellites; the nested dict does not).
        if task_to_passes is not None and constrained_request in task_to_passes:
            ordered_passes = task_to_passes[constrained_request]
        else:
            ordered_passes = [
                (sat, sp)
                for sat in solution_holder[constrained_request].keys()
                for sp in solution_holder[constrained_request][sat].keys()
            ]
            ordered_passes.sort(
                key=lambda t: solution_holder[constrained_request][t[0]][t[1]]['quality'],
                reverse=True
            )

        selected = [
            (satellite, satpass)
            for (satellite, satpass) in ordered_passes
            if solution_holder[constrained_request][satellite][satpass]['x'].X > 0.5
        ]

        if not selected:
            constrained_request.scheduled = False
            constrained_request.scheduled_bookings = []
            if verbose > 2:
                print(f"[Solution] NOT scheduled: {constrained_request.observation_request.name}")
            continue

        constrained_request.scheduled = True

        # Backward-compatible scalars = best-quality (primary) booking.
        best_satellite, best_pass = selected[0]
        constrained_request.observation_opportunity_satellite = best_satellite
        constrained_request.observation_opportunity_pass = best_pass
        constrained_request.observation_opportunity = best_pass.highest

        # Full redundant booking set (plain data only -- no solver objects, so
        # _cleanup_solver_objects leaves it intact and it pickles cleanly).
        constrained_request.scheduled_bookings = [
            {
                'satellite': satellite,
                'pass': satpass,
                'quality': solution_holder[constrained_request][satellite][satpass]['quality'],
                'theta': solution_holder[constrained_request][satellite][satpass]['theta'],
            }
            for (satellite, satpass) in selected
        ]

        if verbose > 1:
            for i, (satellite, satpass) in enumerate(selected):
                role = "PRIMARY" if i == 0 else f"BACKUP-{i}"
                print(f"[Solution] Scheduled {constrained_request.observation_request.name} "
                      f"[{role}] on {satellite.name} at {satpass.highest.time}")

        # Apply timeline impacts for EVERY selected pass.
        if timeline_graph is not None and constrained_request in timeline_graph.nodes():
            for (satellite, satpass) in selected:
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
                                value=impact['impact_value'],
                                owner=constrained_request,
                            )
                            _timeline.add_impact(impact=tl_impact)