"""
Stochastic MILP Scheduling for FAME

This module implements Hedging Mixed-Integer Linear Programming (MILP) scheduling
that accounts for observation success probabilities due to constellation manager
acceptance/rejection uncertainty and execution failures.

Two formulations are supported:
1. Non-convex: Uses quadratic constraints (exact, requires Gurobi NonConvex=2)
2. Log-linearized: Uses piecewise-linear log/exp approximations (faster, approximate)
"""
N_THREADS=10
import numpy as np
import networkx as nx
import datetime as dt
import gurobipy as gp
from gurobipy import GRB
import os
from typing import Callable, Optional
from dotenv import load_dotenv

from fame_geometry import ObservationPass, Satellite
from fame_workflow import (
    ConstrainedObservationRequest, TaskTimelineImpact, TaskImpactTime, Impact,
    Timeline, opportunity_satisfies_task,
)

# Booking lead time is a property of the MARKET, so the planner must respect the
# same rule the constellation enforces: a pass it cannot legally book must not
# become a decision variable, or the model assigns expected reward to passes
# that will be refused at submission and ObjVal stops bounding anything
# realisable.
try:
    from fame_constellation_scheduler import BOOKING_LEAD_TIME_H
except ImportError:
    BOOKING_LEAD_TIME_H = 0.0

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# GENERAL AND/OR GATE ALGEBRA  (stochastic_milp_formulation.tex, Sec. "Generalization
# to Arbitrary AND/OR Dependencies")
# =============================================================================
#
# A task's prerequisite is a Boolean function G_r over the *local successes* T_i
# of its ancestors. The pure-AND base formulation hardcodes
# A_parents_r = prod_{i in ancestors} S_i, which cannot express OR/NOT gates
# (e.g. the MSA multiplexer "image if tracked ELSE search"). Here we represent
# the gate explicitly as a small expression tree and compile it to a Disjoint
# Sum Of Products (DSOP) via Shannon expansion, so that
#     A_parents_r = P(G_r) = sum_k P(D_k)          (linear, by disjointness)
#     P(D_k)      = prod_{i in pos} S_i * prod_{j in neg} (1 - S_j)
# with each path an independent product of literals. See _compile_gate_to_dsop.
#
# Literals reference NODES: either an observation ConstrainedObservationRequest
# (its local success S_r) or a LogicNode (a derived belief-state variable, e.g.
# the MSA "ship tracked?" state K_w, which has a gate but no observation passes).


class Gate:
    """Base class for the Boolean prerequisite expression tree."""
    def literals(self) -> set:
        """Return the set of leaf nodes (literals) referenced by this gate."""
        raise NotImplementedError


class Lit(Gate):
    """A single literal: the local success T_node (probability S_node)."""
    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node

    def literals(self) -> set:
        return {self.node}

    def __repr__(self):
        return f"Lit({_node_name(self.node)})"


class And(Gate):
    """Conjunction of sub-gates (all must be true)."""
    __slots__ = ("operands",)

    def __init__(self, *operands):
        self.operands = tuple(operands)

    def literals(self) -> set:
        out = set()
        for g in self.operands:
            out |= g.literals()
        return out

    def __repr__(self):
        return "And(" + ", ".join(repr(g) for g in self.operands) + ")"


class Or(Gate):
    """Disjunction of sub-gates (at least one true)."""
    __slots__ = ("operands",)

    def __init__(self, *operands):
        self.operands = tuple(operands)

    def literals(self) -> set:
        out = set()
        for g in self.operands:
            out |= g.literals()
        return out

    def __repr__(self):
        return "Or(" + ", ".join(repr(g) for g in self.operands) + ")"


class Not(Gate):
    """Negation of a sub-gate."""
    __slots__ = ("operand",)

    def __init__(self, operand):
        self.operand = operand

    def literals(self) -> set:
        return self.operand.literals()

    def __repr__(self):
        return f"Not({self.operand!r})"


class ExclusiveOr(Gate):
    """A group of MUTUALLY EXCLUSIVE literals (at most one can be true).

    Semantics differ from a general OR: because the events are disjoint by
    construction (e.g. the ship is in at most one H3 search hex), the gate
    probability is the plain linear SUM of the member local successes,
        P = sum_j S_j    (automatically <= 1),
    with NO DSOP expansion and NO exp() constraint. This is the aggregation
    trick from the design discussion: it turns N search tasks into a single
    literal U_w and removes an exp constraint rather than adding paths.
    """
    __slots__ = ("members",)

    def __init__(self, *members):
        # members are Lit gates (or raw nodes, normalized to Lit)
        self.members = tuple(m if isinstance(m, Gate) else Lit(m) for m in members)

    def literals(self) -> set:
        out = set()
        for m in self.members:
            out |= m.literals()
        return out

    def __repr__(self):
        return "ExclusiveOr(" + ", ".join(repr(m) for m in self.members) + ")"


class LogicNode:
    """A derived belief-state node (e.g. the MSA "ship tracked?" state K_w).

    It carries a prerequisite gate but has NO observation passes: it is never
    dispatched and never appears in the broker's workflow_graph. Its local
    success S equals the probability that its gate is satisfied,
        S_{LogicNode} = P(gate) = A_parents_{LogicNode}.
    The solver discovers LogicNodes by walking the gates of the observation
    tasks (and recursively the gates of referenced LogicNodes).
    """
    __slots__ = ("name", "gate", "request_group")

    def __init__(self, name: str, gate: Gate, request_group: str = None):
        self.name = name
        self.gate = gate
        self.request_group = request_group if request_group is not None else name

    def __repr__(self):
        return f"LogicNode({self.name})"


def _node_name(node) -> str:
    """Human-readable name for a literal node (task or LogicNode)."""
    if isinstance(node, LogicNode):
        return node.name
    obs = getattr(node, "observation_request", None)
    if obs is not None:
        return getattr(obs, "name", str(id(node)))
    return getattr(node, "name", str(id(node)))


# --- Shannon expansion -> Disjoint Sum Of Products --------------------------
#
# We evaluate the gate under a partial assignment {node: True/False}. Splitting
# recursively on an unassigned literal produces disjoint branches (one assumes
# the literal true, the other false), so the leaves are mutually exclusive
# scenarios D_k. Each D_k is recorded as (frozenset positives, frozenset
# negatives). See the worked example G_5 in the .tex.

def _gate_eval(gate: Gate, assignment: dict):
    """Evaluate a gate under a partial assignment.

    Returns True, False, or None (undetermined). An ExclusiveOr is treated as a
    plain OR for the purpose of Boolean satisfaction (its special probability
    handling happens elsewhere); this only decides *whether* the gate can still
    be satisfied under the partial assignment.
    """
    if isinstance(gate, Lit):
        return assignment.get(gate.node, None)
    if isinstance(gate, Not):
        v = _gate_eval(gate.operand, assignment)
        return None if v is None else (not v)
    if isinstance(gate, And):
        result = True
        for g in gate.operands:
            v = _gate_eval(g, assignment)
            if v is False:
                return False
            if v is None:
                result = None
        return result
    if isinstance(gate, (Or, ExclusiveOr)):
        operands = gate.operands if isinstance(gate, Or) else gate.members
        result = False
        for g in operands:
            v = _gate_eval(g, assignment)
            if v is True:
                return True
            if v is None:
                result = None
        return result
    raise TypeError(f"Unknown gate type: {type(gate)}")


def _compile_gate_to_dsop(gate: Gate):
    """Compile a Boolean gate into a Disjoint Sum Of Products.

    Returns a list of paths, each a (positives, negatives) pair of frozensets of
    literal nodes: the scenario "all positives succeeded AND all negatives
    failed". Paths are pairwise mutually exclusive, so
        P(gate) = sum_paths ( prod_{i in pos} S_i * prod_{j in neg} (1 - S_j) ).

    ExclusiveOr sub-gates are NOT expanded here: they are handled upstream as a
    single aggregated literal (their U node), so by the time a gate reaches this
    compiler every leaf is an ordinary independent Lit. Implemented via Shannon
    expansion (see .tex Sec. "Shannon Expansion").
    """
    variables = list(gate.literals())
    paths = []

    def recurse(assignment, remaining):
        val = _gate_eval(gate, assignment)
        if val is True:
            pos = frozenset(n for n, b in assignment.items() if b)
            neg = frozenset(n for n, b in assignment.items() if not b)
            paths.append((pos, neg))
            return
        if val is False:
            return
        if not remaining:
            # Undetermined with nothing left to assign: guard against infinite
            # recursion on a malformed gate.
            return
        pivot = remaining[0]
        rest = remaining[1:]
        a_true = dict(assignment); a_true[pivot] = True
        recurse(a_true, rest)
        a_false = dict(assignment); a_false[pivot] = False
        recurse(a_false, rest)

    recurse({}, variables)
    return paths


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
        half_life_s: float = 86400.0,  # 24 hours
        min_value: float = -30.0,
        max_value: float = 30.0,
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
        Scans requests for observations matching this timeline's volcano group.
        Maintains initial active state (True) while detection is pending/in-flight.
        """
        volcano_prefix = self.name.replace("_active", "")
        group_requests = [
            r for r in requests 
            if getattr(r, 'request_group', '').startswith(volcano_prefix)
        ]

        # Check if any observation for this volcano succeeded recently
        recent_success = any(
            getattr(r, 'completed', False) and getattr(r, 'successful_execution', False)
            and (current_time - (getattr(r, 'execution_time', None) or r.observation_opportunity.time)) <= self.half_life
            for r in group_requests
            if hasattr(r, 'observation_opportunity') and r.observation_opportunity is not None
        )

        if recent_success:
            is_active = True
        else:
            # Check if root detection has completed and failed
            detection_failed = any(
                getattr(r, 'completed', False) and not getattr(r, 'successful_execution', False)
                and 'detection' in getattr(r, 'name', '')
                for r in group_requests
            )
            # If detection completed & failed -> inactive. Otherwise keep active while pending.
            is_active = not detection_failed

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
        detection_probability_function: Callable = None,   # p_det: prob target present / in footprint (entry-boundary factor). None => 1.0. Only used by "general_logical_dag".
        epsilon: float = 1e-3,
        pwl_tolerance: float = 1e-1,  # SCIP path only; the Gurobi path now uses exact MINLP handling (FuncNonlinear=1)
        mip_gap: float = 0.05,
        default_max_instances: int = 3,  # Redundancy cap when a request has no max_num_instances attribute. >1 is REQUIRED for the stochastic planner to hedge.
        solver_engine: str = "GUROBI",
        tax_rate: float = 0.0,  # Cost per scheduled obs as fraction of max quality (dynamic, per-request). Set to 0 to disable.
        submission_cost_rate: float = 0.0,  # c_sub: unconditional per-booking submission overhead (as fraction of Q_MAX_task)
        execution_cost_rate: float = 0.0,   # fallback if execution_cost_fn is None
        execution_cost_fn = None,           # callable(task, satellite, obs_pass, dispatch_time, q_max) -> float
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
        "non_convex" (exact quadratic), "log_linearized" (PWL approximation), or
        "general_logical_dag" (arbitrary AND/OR/NOT prerequisite gates via DSOP,
        with a p_det entry-boundary factor -- see _build_general_logical_formulation)
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
            execution_cost_fn=execution_cost_fn,
            mip_gap=mip_gap, default_max_instances=default_max_instances, results_dir=results_dir,
            detection_probability_function=detection_probability_function
        )
    elif solver_engine == "SCIP":
        return _solve_with_scip(
            workflow_graph, timeline_graph, satellites, feasibility_screener,
            current_time, verbose, max_solver_time_s, receding_horizon_duration,
            stochastic_formulation, success_probability_function,
            acceptance_probability_function, execution_probability_function,
            epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate,
            execution_cost_fn=execution_cost_fn,
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
        execution_cost_fn,
        mip_gap: float,
        default_max_instances: int,
        results_dir: str,
        detection_probability_function: Callable = None
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
            model.setParam('TimeLimit', max_solver_time_s)
            model.setParam('MIPGap', mip_gap)
            # general_logical_dag needs exact MINLP handling (outer approximation
            # + spatial branching): its DSOP chains stack ~45 exp constraints
            # deep, and PWL error compounds through the log->sum->exp round trip
            # at every task boundary.
            #
            # log_linearized relies on cuts/tightening that work best with PWL,
            # AND -- with single-parent (chain) workflows -- the node classifier
            # prunes the log machinery to zero, so most windows build as a PURE
            # MILP with no nonlinear constraints at all.
            if stochastic_formulation == "general_logical_dag":
                model.setParam('FuncNonlinear', 0)
                model.setParam('FuncPieceError', 1e-1)  # Tighten tolerance (default: 1e-3)
                model.setParam('FuncPieces', -2)
            else:
                model.setParam('FuncNonlinear', 0)
                model.setParam('FuncPieceError', 1e-2)  # Tighten tolerance (default: 1e-3)
                model.setParam('FuncPieces', -2)
            model.setParam('Cuts', 1)
            model.setParam('Threads', N_THREADS)
            model.setParam('OutputFlag', 1)
            # Favour finding good incumbents over proving optimality: a schedule
            # that is 2% off is worth far more here than a proof.
            model.setParam('MIPFocus', 1)
            model.setParam('Heuristics', 0.5)
            if results_dir:
                model.setParam('LogFile', os.path.join(results_dir, "gurobi_stochastic.log"))

            # Enable bilinear solver if using the exact quadratic formulation
            if stochastic_formulation == "non_convex":
                model.setParam('NonConvex', 2)

            # Step 3: Find observation opportunities and create variables
            solution_holder = {}
            task_to_passes = {}
            # Bookings already dispatched in an earlier solve. They are excluded
            # from the model (no decision left to make) but still consume the
            # satellite, so their windows must block new variables -- otherwise
            # this solve re-books an occupied slot and the constellation rejects
            # it at submission time.
            committed_bookings = []   # (satellite, obs_start, obs_end)

            for constrained_request in workflow_graph.nodes():
                if (constrained_request.dispatched == True) or (constrained_request.completed == True):
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] Skipping {constrained_request.observation_request.name} (dispatched/completed)")
                    _bookings = getattr(constrained_request, 'scheduled_bookings', None) or []
                    if (not _bookings) and getattr(constrained_request, 'observation_opportunity_pass', None) is not None:
                        _bookings = [{
                            'satellite': getattr(constrained_request, 'observation_opportunity_satellite', None),
                            'pass': constrained_request.observation_opportunity_pass,
                        }]
                    for _b in _bookings:
                        _sp = _b.get('pass')
                        _bsat = _b.get('satellite')
                        if _sp is None or _bsat is None:
                            continue
                        committed_bookings.append((
                            _bsat, _sp.highest.time,
                            _sp.highest.time + _sp.highest.duration))
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
                # Sort by quality DESCENDING. This is load-bearing: the
                # diminishing recurrence credits the FIRST success in this order,
                # so descending quality == best-of-N crediting rather than
                # first-in-time.
                allsatpasses.sort(key=lambda x: x[2], reverse=True)

                solution_holder[constrained_request] = {}
                task_to_passes[constrained_request] = []

                for (satellite, satpass, _quality) in allsatpasses:
                    # Inside the lead-time window this pass cannot be booked, so
                    # it must not become a variable.
                    if BOOKING_LEAD_TIME_H > 0 and current_time is not None:
                        if ((satpass.highest.time - current_time).total_seconds() / 3600.0
                                < BOOKING_LEAD_TIME_H):
                            continue
                    if (feasibility_screener(satellite, satpass)
                            and opportunity_satisfies_task(
                                constrained_request, satellite, satpass)):
                        _found_a_pass = True

                        if satellite not in solution_holder[constrained_request].keys():
                            solution_holder[constrained_request][satellite] = {}

                        var_name = f"x_{constrained_request.observation_request.name}_{satellite.name}_{satpass.highest.time}"
                        x_var = model.addVar(vtype=GRB.BINARY, name=var_name)

                        # Compute two-stage probabilities
                        if acceptance_probability_function is not None and execution_probability_function is not None:
                            p_acc = acceptance_probability_function(constrained_request, satellite, satpass)
                            p_exec = execution_probability_function(constrained_request, satellite, satpass)
                            p_total = p_acc * p_exec
                        else:
                            p_total = success_probability_function(constrained_request, satellite, satpass)
                            p_acc = p_total
                            p_exec = 1.0

                        # NOTE: p_det is deliberately NOT folded into theta. It is
                        # SHARED across a task's redundant passes (they all aim at
                        # the same point, so if the target is absent they ALL
                        # miss), whereas acceptance and execution fail
                        # independently. p_det enters at the entry boundary
                        # Y_0 = p_det * A_parents inside the formulation builder.
                        solution_holder[constrained_request][satellite][satpass] = {
                            'x': x_var,
                            'quality': _quality,
                            'theta': p_total,      # p_acc * p_exec
                            'theta_acc': p_acc,
                            'theta_exec': p_exec
                        }

                        task_to_passes[constrained_request].append((satellite, satpass))

                if not _found_a_pass:
                    constrained_request.scheduled = True
                    constrained_request.feasible = False
                    if verbose > 1:
                        print(f"[Stochastic Scheduler] All passes infeasible for {constrained_request.observation_request.name}")

            # Step 4: Build stochastic formulation
            _general_logical_warm_start_fn = None
            if stochastic_formulation == "non_convex":
                _build_non_convex_formulation(
                    model, workflow_graph, solution_holder, task_to_passes,
                    epsilon, tax_rate, submission_cost_rate, execution_cost_rate, verbose,
                    execution_cost_fn=execution_cost_fn, current_time=current_time
                )
            elif stochastic_formulation == "log_linearized":
                _build_log_linearized_formulation(
                    model, workflow_graph, solution_holder, task_to_passes,
                    epsilon, pwl_tolerance, tax_rate, submission_cost_rate, execution_cost_rate, verbose,
                    execution_cost_fn=execution_cost_fn, current_time=current_time,
                    default_max_instances=default_max_instances,
                    # p_det entry boundary. Without this the function defaults to
                    # None -> 1.0 and the whole detection model is silently inert
                    # (the log then reads "p_det inactive (all 1.0)" even though
                    # the tasks carry real detection_prob values).
                    detection_probability_function=detection_probability_function
                )
            elif stochastic_formulation == "general_logical_dag":
                _general_logical_warm_start_fn = _build_general_logical_formulation(
                    model, workflow_graph, solution_holder, task_to_passes,
                    epsilon, tax_rate, submission_cost_rate, execution_cost_rate, verbose,
                    detection_probability_function=detection_probability_function,
                    default_max_instances=default_max_instances,
                    execution_cost_fn=execution_cost_fn, current_time=current_time
                )
            else:
                raise ValueError(f"Unknown stochastic_formulation: {stochastic_formulation}")

            # Step 5: Add constraints (temporal, timeline, branch exclusivity, ...)
            _add_workflow_constraints(
                model, workflow_graph, solution_holder, verbose,
                default_max_instances=default_max_instances,
                committed_bookings=committed_bookings
            )

            # Step 6: Solve
            # CRITICAL: model.update() before NumVars/NumConstrs are accurate.
            model.update()

            if verbose > 0:
                n_bin  = sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
                n_int  = sum(1 for v in model.getVars() if v.VType == GRB.INTEGER)
                n_cont = model.NumVars - n_bin - n_int
                print(f"[Stochastic Scheduler] vars={model.NumVars} "
                    f"(bin={n_bin}, int={n_int}, cont={n_cont}) "
                    f"constrs={model.NumConstrs} genconstrs={model.NumGenConstrs} "
                    f"(logs={sum(1 for gc in model.getGenConstrs() if gc.GenConstrType == GRB.GENCONSTR_LOG)})")

            # === MIP STARTS ===
            # Gurobi keeps whichever start produces an incumbent.
            #   0: greedy-N -- same policy as the greedy scheduler: up to
            #      min(task.max_num_instances, default_max_instances) highest-
            #      quality non-overlapping passes, DAG order, TEMPORAL + SUCCESS.
            #   1: expected-value greedy-N -- same filters; pick by
            #      Q * theta * remaining first-success risk (diminishing).
            #   2: mandatory-only -- one pass per mandatory task (safety net).
            from fame_workflow import ConstraintClass, TemporalConstraintType

            def _temporal_ok(_child, _t_child, _placed_slots):
                if workflow_graph is None or _child not in workflow_graph:
                    return True
                for _parent in workflow_graph.predecessors(_child):
                    _slots = _placed_slots.get(_parent)
                    if not _slots:
                        continue
                    _rel = []
                    for _k, _c in (workflow_graph.get_edge_data(_parent, _child) or {}).items():
                        if _c.get('constraint_class') != ConstraintClass.TEMPORAL:
                            continue
                        _rel.append((
                            _c.get('constraint_type'),
                            (_c.get('parameters') or {}).get('offset', dt.timedelta(0)),
                        ))
                    if not _rel:
                        continue

                    def _ok_one(_t_parent, _rel=_rel):
                        for _ct, _off in _rel:
                            if _ct == TemporalConstraintType.START_AFTER and _t_parent > _t_child:
                                return False
                            if _ct == TemporalConstraintType.START_AFTER_OFFSET and _t_parent + _off > _t_child:
                                return False
                            if _ct == TemporalConstraintType.START_BEFORE and _t_parent < _t_child:
                                return False
                            if _ct == TemporalConstraintType.START_BEFORE_OFFSET and _t_parent + _off < _t_child:
                                return False
                        return True

                    if not any(_ok_one(_s) for _s, _e in _slots):
                        return False
                return True

            def _success_ok(_child, _t_child, _placed_slots):
                if not getattr(_child, 'enforce_success_precedence', False):
                    return True
                if workflow_graph is None or _child not in workflow_graph:
                    return True
                _parents = []
                for _parent in workflow_graph.predecessors(_child):
                    _edges = workflow_graph.get_edge_data(_parent, _child) or {}
                    if any(_c.get('constraint_class') == ConstraintClass.SUCCESS
                           for _c in _edges.values()):
                        _parents.append(_parent)
                if not _parents:
                    return True

                def _precedes(_parent):
                    for _s, _e in _placed_slots.get(_parent, []):
                        if _e <= _t_child:
                            return True
                    if _parent in solution_holder:
                        return False
                    _oo = getattr(_parent, 'observation_opportunity', None)
                    _t = getattr(_oo, 'time', None)
                    if _t is not None:
                        _dur = getattr(_oo, 'duration', dt.timedelta(0))
                        return _t + _dur <= _t_child
                    for _b in getattr(_parent, 'scheduled_bookings', None) or []:
                        _p = _b.get('pass')
                        if _p is None:
                            continue
                        if _p.highest.time + _p.highest.duration <= _t_child:
                            return True
                    return False

                _mode = getattr(_child, 'success_constraint_mode', 'all')
                if _mode == 'any':
                    return any(_precedes(_p) for _p in _parents)
                return all(_precedes(_p) for _p in _parents)

            def _free_on_sat(_busy, _sat, _s0, _e0):
                return all(_e0 <= s or _s0 >= e for (s, e) in _busy.get(_sat, []))

            def _task_k(_req, mandatory_only):
                if mandatory_only:
                    return 1
                _per = int(getattr(_req, 'max_num_instances', 1) or 1)
                return max(1, min(_per, int(default_max_instances)))

            def _build_start(mandatory_only: bool, score="quality"):
                _busy, _placed_slots, _x_vals, _ok = {}, {}, {}, True
                try:
                    _order = [r for r in nx.topological_sort(workflow_graph) if r in solution_holder]
                    _order += [r for r in solution_holder if r not in set(_order)]
                except Exception:
                    _order = list(solution_holder.keys())

                for _req in _order:
                    _passes = task_to_passes.get(_req, [])
                    for _sat, _sp in _passes:
                        _x_vals[(_req, _sat, _sp)] = 0.0
                    if mandatory_only and not getattr(_req, 'is_mandatory', False):
                        continue
                    _k_want = _task_k(_req, mandatory_only)
                    _remaining = list(_passes)
                    _y = 1.0
                    _n_placed = 0
                    while _n_placed < _k_want and _remaining:
                        _best_i = None
                        _best_sc = None
                        _best = None
                        for _i, (_sat, _sp) in enumerate(_remaining):
                            _s0 = _sp.highest.time
                            _e0 = _s0 + _sp.highest.duration
                            if not _free_on_sat(_busy, _sat, _s0, _e0):
                                continue
                            if not _temporal_ok(_req, _s0, _placed_slots):
                                continue
                            if not _success_ok(_req, _s0, _placed_slots):
                                continue
                            _h = solution_holder[_req][_sat][_sp]
                            if score == "ev":
                                _sc = float(_h['quality']) * float(_h.get('theta', 1.0)) * _y
                            else:
                                _sc = float(_h['quality'])
                            if _best is None or _sc > _best_sc:
                                _best = (_sat, _sp, _s0, _e0, _h)
                                _best_sc = _sc
                                _best_i = _i
                        if _best is None:
                            break
                        _sat, _sp, _s0, _e0, _h = _best
                        _x_vals[(_req, _sat, _sp)] = 1.0
                        _busy.setdefault(_sat, []).append((_s0, _e0))
                        _placed_slots.setdefault(_req, []).append((_s0, _e0))
                        _y *= (1.0 - float(_h.get('theta', 1.0)))
                        _remaining.pop(_best_i)
                        _n_placed += 1
                    if _n_placed == 0 and getattr(_req, 'is_mandatory', False) and _passes:
                        _ok = False
                return _x_vals, _ok

            try:
                _starts = []
                _gn_x, _gn_ok = _build_start(mandatory_only=False, score="quality")
                if _gn_ok:
                    _starts.append(("greedy-n", _gn_x))
                _ev_x, _ev_ok = _build_start(mandatory_only=False, score="ev")
                if _ev_ok:
                    _starts.append(("ev-greedy-n", _ev_x))
                _fallback_x, _fallback_ok = _build_start(mandatory_only=True, score="quality")
                if _fallback_ok:
                    _starts.append(("mandatory-only", _fallback_x))

                if _starts:
                    model.NumStart = len(_starts)
                    for _i, (_label, _xv) in enumerate(_starts):
                        model.params.StartNumber = _i
                        for (_req, _sat, _sp), _v in _xv.items():
                            solution_holder[_req][_sat][_sp]['x'].Start = _v
                        # general_logical_dag is PWL (FuncNonlinear=0). A binary-
                        # only start is incomplete against the log/exp pieces and
                        # Gurobi drops it ("did not produce a new incumbent").
                        if _general_logical_warm_start_fn is not None:
                            _general_logical_warm_start_fn(_xv)
                    model.params.StartNumber = -1
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] MIP starts: "
                              f"{', '.join(l for l, _ in _starts)}")
                elif verbose > 0:
                    print("[Stochastic Scheduler] No feasible MIP start could be built "
                          "(a mandatory task has no temporally compatible pass)")
            except Exception as _e:
                if verbose > 0:
                    print(f"[Stochastic Scheduler] MIP start skipped: {_e}")

            # Skip optimization if there are no variables (nothing to schedule)
            if model.NumVars == 0:
                if verbose > 0:
                    print("[Stochastic Scheduler] No pending tasks to schedule. Skipping optimization.")
                workflow_graph.graph['objective_value'] = 0.0
            else:
                model.optimize()
                try:
                    workflow_graph.graph['solve_time_s'] = model.Runtime
                except Exception:
                    workflow_graph.graph['solve_time_s'] = float('nan')
                try:
                    workflow_graph.graph['mip_gap'] = model.MIPGap if model.SolCount > 0 else float('nan')
                except Exception:
                    workflow_graph.graph['mip_gap'] = float('nan')

            # Step 7: Extract solution
            if model.NumVars > 0:
                if model.Status == GRB.OPTIMAL:
                    if verbose > 0:
                        print(f"[Stochastic Scheduler] Optimal solution found! Objective value: {model.ObjVal:.2f}")

                    _extract_solution(
                        model, workflow_graph, solution_holder, verbose, timeline_graph,
                        task_to_passes=task_to_passes
                    )
                    workflow_graph.graph['objective_value'] = model.ObjVal

                elif model.Status in [GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED]:
                    if model.SolCount > 0:
                        if verbose > 0:
                            print(f"[Stochastic Scheduler] Time limit reached, but feasible solution found! Objective: {model.ObjVal:.2f}")
                            print(f"[Stochastic Scheduler] Current MIP Gap: {model.MIPGap * 100:.2f}% (Best Bound: {model.ObjBound:.2f})")
                            if model.MIPGap > 0.20:
                                print("[Stochastic Scheduler] WARNING: gap > 20% -- ObjVal is NOT a "
                                      "trustworthy planning value; do not compare it against realized utility.")

                        _extract_solution(
                            model, workflow_graph, solution_holder, verbose, timeline_graph,
                            task_to_passes=task_to_passes
                        )
                        workflow_graph.graph['objective_value'] = model.ObjVal
                    else:
                        if verbose > 0:
                            print("[Stochastic Scheduler] Time limit reached with no feasible solution found")
                        # SolCount == 0 after a full time limit usually means
                        # infeasibility, not difficulty: booking one pass per
                        # mandatory task and nothing else satisfies the recurrence,
                        # so if Gurobi found NO incumbent something is structurally
                        # blocking it. Probe with a bounded re-solve.
                        if verbose > 1:
                            print("[Stochastic Scheduler] Probing feasibility (bounded)...")
                            model.setParam('TimeLimit', 120)
                            model.setParam('SolutionLimit', 1)
                            model.optimize()
                            if model.Status == GRB.INFEASIBLE:
                                print("[Stochastic Scheduler] INFEASIBLE. Computing IIS...")
                                try:
                                    model.setParam('IISMethod', 1)   # heuristic: far faster
                                    model.computeIIS()
                                    _ilp = os.path.join(results_dir, "infeasible_model.ilp") if results_dir else "infeasible_model.ilp"
                                    model.write(_ilp)
                                    print(f"[Stochastic Scheduler] IIS written to {_ilp}")
                                    for c in model.getConstrs():
                                        if c.IISConstr:
                                            print(f"  IIS constr: {c.ConstrName}")
                                    for v in model.getVars():
                                        if v.IISLB or v.IISUB:
                                            print(f"  IIS bound: {v.VarName} [{v.LB}, {v.UB}]")
                                except Exception as _e:
                                    print(f"[Stochastic Scheduler] IIS failed: {_e}")
                            elif model.SolCount > 0:
                                print("[Stochastic Scheduler] Feasible after all -- the time limit "
                                      "was simply too short to find an incumbent.")
                            else:
                                print(f"[Stochastic Scheduler] Still undetermined (status {model.Status}).")

                elif model.Status == GRB.INFEASIBLE:
                    if verbose > 0:
                        print("[Stochastic Scheduler] Model is infeasible (no valid schedule found)")
                    if verbose > 2:
                        try:
                            model.setParam('IISMethod', 1)
                            model.computeIIS()
                            model.write("infeasible_model.ilp")
                            print("[Stochastic Scheduler] IIS written to infeasible_model.ilp")
                        except Exception:
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
        execution_cost_fn = None,
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
            # See the note in _solve_with_gurobi: the planner respects the same
            # lead-time rule the constellation enforces.
            if BOOKING_LEAD_TIME_H > 0 and current_time is not None:
                if ((satpass.highest.time - current_time).total_seconds() / 3600.0
                        < BOOKING_LEAD_TIME_H):
                    continue
            if (feasibility_screener(satellite, satpass)
                    and opportunity_satisfies_task(
                        constrained_request, satellite, satpass)):
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
        hint_values_map,
        execution_cost_fn=execution_cost_fn, current_time=current_time
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
        workflow_graph.graph['solve_time_s'] = 0.0
        workflow_graph.graph['mip_gap'] = 0.0
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
        # Store solver diagnostics for planning-session metrics (ortools WallTime is in ms)
        workflow_graph.graph['solve_time_s'] = solver.WallTime() / 1000.0
        workflow_graph.graph['mip_gap'] = float('nan')  # ortools doesn't expose MIP gap

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

    # Exactly one segment is
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
        solver, graph, holder, task_to_passes, eps, tol, tax, sub, canc, verbose, hint_values_map,
        execution_cost_fn=None, current_time=None
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

        _q_max_task = max(all_qualities)
        c_sub = sub * _q_max_task
        c_tax = tax * _q_max_task

        # Determine assumed dispatch time for follow-up tasks vs root tasks.
        _has_success_parent = any(
            'SUCCESS' in str(e.get('constraint_class', ''))
            for p in graph.predecessors(constrained_request)
            for e in graph.get_edge_data(p, constrained_request).values()
        )
        if _has_success_parent:
            # Assume dispatched at earliest parent pass time (conservative but linear).
            _parent_passes = [
                sp.highest.time
                for p in graph.predecessors(constrained_request)
                for sat, sp in task_to_passes.get(p, [])
            ]
            _t_dispatch = min(_parent_passes) if _parent_passes else current_time
        else:
            _t_dispatch = current_time

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = holder[constrained_request][satellite][satpass]['x']
            quality = holder[constrained_request][satellite][satpass]['quality']
            theta = holder[constrained_request][satellite][satpass]['theta']
            theta_acc = holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = effective_pass_realization[(constrained_request, satellite, satpass)]

            if execution_cost_fn is not None:
                try:
                    c_exec_k = execution_cost_fn(constrained_request, satellite, satpass, _t_dispatch, q_max=_q_max_task)
                except Exception:
                    c_exec_k = canc * _q_max_task
            else:
                c_exec_k = canc * _q_max_task

            # Expected reward; costs use Q_MAX_task for provider-rate billing.
            objective.SetCoefficient(w_abs, quality * theta)
            objective.SetCoefficient(x_var, -c_sub - c_exec_k * theta_acc - c_tax)

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

        # Track the max end time seen up to index i so the inner loop can break
        # correctly even when passes from different tasks have non-monotonic end times.
        max_end_so_far = {}
        running_max = passes[0][0].highest.time + passes[0][0].highest.duration if passes else None
        for i, (p, _, _r) in enumerate(passes):
            running_max = max(running_max, p.highest.time + p.highest.duration)
            max_end_so_far[i] = running_max

        for i in range(len(passes)):
            pass_i, x_i, req_i = passes[i]
            for j in range(i + 1, len(passes)):
                pass_j, x_j, req_j = passes[j]
                if pass_j.highest.time >= max_end_so_far[i]:
                    break
                end_i = pass_i.highest.time + pass_i.highest.duration
                start_j = pass_j.highest.time
                if start_j < end_i:
                    solver.Add(x_i + x_j <= 1)

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
        verbose: int,
        execution_cost_fn=None,
        current_time=None
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
    # Costs use Q_MAX_task (max quality over all passes for the task) for provider-rate
    # billing, decoupled from per-pass geometry. Matches realized metrics billing.
    objective_terms = []
    for constrained_request in solution_holder.keys():
        if not solution_holder[constrained_request]:
            continue

        _q_max_task = max(
            solution_holder[constrained_request][s][p]['quality']
            for s, p in task_to_passes[constrained_request]
        )
        _has_success_parent = any(
            'SUCCESS' in str(e.get('constraint_class', ''))
            for p in workflow_graph.predecessors(constrained_request)
            for e in workflow_graph.get_edge_data(p, constrained_request).values()
        )
        if _has_success_parent:
            _parent_passes = [sp.highest.time for p in workflow_graph.predecessors(constrained_request) for sat, sp in task_to_passes.get(p, [])]
            _t_dispatch = min(_parent_passes) if _parent_passes else current_time
        else:
            _t_dispatch = current_time

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            quality = solution_holder[constrained_request][satellite][satpass]['quality']
            theta = solution_holder[constrained_request][satellite][satpass]['theta']
            theta_acc = solution_holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = w_abs_vars[(constrained_request, satellite, satpass)]

            c_sub = submission_cost_rate * _q_max_task
            if execution_cost_fn is not None:
                try:
                    c_exec_k = execution_cost_fn(constrained_request, satellite, satpass, _t_dispatch, q_max=_q_max_task)
                except Exception:
                    c_exec_k = cancellation_cost_rate * _q_max_task
            else:
                c_exec_k = cancellation_cost_rate * _q_max_task
            c_tax = tax_rate * _q_max_task

            objective_terms.append(quality * theta * w_abs)
            objective_terms.append(-c_sub * x_var)
            objective_terms.append(-c_exec_k * theta_acc * x_var)
            objective_terms.append(-c_tax * x_var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)


def _get_gate(node):
    """Return the prerequisite Gate attached to a node, or None.

    Observation tasks (ConstrainedObservationRequest) may carry an optional
    `.gate` attribute set by a workflow builder; LogicNodes always carry `.gate`.
    """
    return getattr(node, "gate", None)


def build_logical_dag(observation_tasks, workflow_graph=None):
    """Collect the full node universe for the general logical formulation.

    Starting from the observation tasks (the nodes that have candidate passes),
    transitively discover every LogicNode referenced by their gates (and by the
    gates of those LogicNodes, recursively). Returns

        (ordered_nodes, dsop_of, is_logic)

    where `ordered_nodes` is a dependency-topological order (every literal a
    node's gate references appears before that node), `dsop_of[node]` is the
    compiled DSOP path list for nodes that have a gate (None otherwise), and
    `is_logic[node]` marks LogicNodes (no passes, no reward).

    Backward compatibility: if NO node carries a gate, this returns the pure-AND
    interpretation implicitly (dsop_of all None), and the caller falls back to
    the AND-over-graph-ancestors entry boundary -- matching the base formulation.
    """
    # 1. Transitive discovery of every node reachable through gates: LogicNodes
    #    AND observation tasks (the latter may be gate literals that are no longer
    #    schedulable -- already dispatched/completed -- and hence absent from the
    #    caller's observation_tasks list; they still need a local-success value so
    #    downstream gates can reference them, handled as constants by the builder).
    universe = list(observation_tasks)
    seen = set(id(n) for n in universe)
    frontier = list(observation_tasks)
    while frontier:
        node = frontier.pop()
        gate = _get_gate(node)
        if gate is None:
            continue
        for lit_node in gate.literals():
            if id(lit_node) not in seen:
                seen.add(id(lit_node))
                universe.append(lit_node)
                frontier.append(lit_node)

    # 2. Dependency edges: literal -> node (literal must be built first).
    #    Build a dependency graph over the universe and topologically sort it.
    dep = nx.DiGraph()
    for node in universe:
        dep.add_node(id(node))
    id_to_node = {id(n): n for n in universe}
    for node in universe:
        gate = _get_gate(node)
        if gate is None:
            continue
        for lit_node in gate.literals():
            if id(lit_node) in id_to_node:  # only intra-universe deps
                dep.add_edge(id(lit_node), id(node))
    if not nx.is_directed_acyclic_graph(dep):
        cycle = nx.find_cycle(dep)
        raise ValueError(f"Logical-DAG gate dependencies contain a cycle: {cycle}")
    ordered_nodes = [id_to_node[i] for i in nx.topological_sort(dep)]

    # 3. Compile each gate to DSOP once. ExclusiveOr gates are NOT compiled --
    #    they are disjoint groups handled as a linear sum by the builder (the
    #    DSOP compiler would wrongly treat them as an independent OR). A node
    #    with an ExclusiveOr gate gets dsop_of == None but is still gated (the
    #    builder distinguishes it by inspecting the gate type).
    dsop_of = {}
    is_logic = {}
    for node in ordered_nodes:
        is_logic[node] = isinstance(node, LogicNode)
        gate = _get_gate(node)
        if gate is None or isinstance(gate, ExclusiveOr):
            dsop_of[node] = None
        else:
            dsop_of[node] = _compile_gate_to_dsop(gate)
    return ordered_nodes, dsop_of, is_logic


def _gate_eval_bool(gate: Gate, realized: dict) -> bool:
    """Evaluate a gate to a definite bool under a TOTAL success assignment.

    `realized[node]` gives each literal node's realized success (True/False).
    Missing literals are treated as False (never succeeded). Unlike `_gate_eval`
    (which returns None for partial assignments during DSOP compilation), this
    assumes every literal is decided, so it always returns True/False.
    """
    if isinstance(gate, Lit):
        return bool(realized.get(gate.node, False))
    if isinstance(gate, Not):
        return not _gate_eval_bool(gate.operand, realized)
    if isinstance(gate, And):
        return all(_gate_eval_bool(g, realized) for g in gate.operands)
    if isinstance(gate, Or):
        return any(_gate_eval_bool(g, realized) for g in gate.operands)
    if isinstance(gate, ExclusiveOr):
        return any(_gate_eval_bool(m, realized) for m in gate.members)
    raise TypeError(f"Unknown gate type: {type(gate)}")


def compute_gate_reachability(observation_tasks, success_of):
    """Gate-aware reachability for a logical (gate-based) workflow.

    A task is REACHABLE iff the branch that leads to it was actually taken in
    simulation -- i.e. its prerequisite gate evaluates True under the realized
    per-task success outcomes. This is the fair denominator for the MSA workflow,
    whose windows are mutually exclusive (image XOR search) and whose search
    hexes are disjoint (ExclusiveOr): only ~one branch per window is ever live,
    so counting all 1 + W*(1+N) tasks as the denominator is misleading.

    Args:
        observation_tasks: the schedulable observation tasks (workflow_graph
            nodes). Their gates transitively reference LogicNodes and each other.
        success_of: callable(task) -> bool, the realized successful execution of
            an OBSERVATION task (LogicNodes are derived from their gates).

    Returns:
        (reachable_obs_tasks, realized) where reachable_obs_tasks is the set of
        observation tasks whose gate fired, and realized maps every node
        (observation tasks AND LogicNodes) to its realized boolean success.
        A gate-free (root) task is always reachable.
    """
    ordered_nodes, _dsop, is_logic = build_logical_dag(list(observation_tasks))

    # Forward pass in dependency order: each node's realized success is known
    # before any node whose gate references it (topological guarantee).
    realized = {}
    for node in ordered_nodes:
        gate = _get_gate(node)
        if is_logic[node]:
            # Belief-state node: its "success" is the truth of its gate.
            realized[node] = _gate_eval_bool(gate, realized) if gate is not None else False
        else:
            # Observation task: realized success comes from the simulation.
            realized[node] = bool(success_of(node))

    obs_set = set(observation_tasks)
    reachable = set()
    for node in ordered_nodes:
        if is_logic[node] or node not in obs_set:
            continue
        gate = _get_gate(node)
        if gate is None or _gate_eval_bool(gate, realized):
            reachable.add(node)
    return reachable, realized


def _build_general_logical_formulation(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        task_to_passes: dict,
        epsilon: float,
        tax_rate: float,
        submission_cost_rate: float,
        cancellation_cost_rate: float,
        verbose: int,
        detection_probability_function: Callable = None,
        default_max_instances: int = 3,
        execution_cost_fn=None,
        current_time=None
):
    """General AND/OR/NOT stochastic formulation (paper Sec. "Generalization to
    Arbitrary AND/OR Dependencies").

    Differences from `_build_log_linearized_formulation` (the pure-AND base):

    1. ENTRY BOUNDARY FROM AN EXPLICIT GATE. Instead of A_parents = prod over
       graph-ancestors, each node's prerequisite is a Boolean gate compiled to a
       Disjoint Sum Of Products. With ln S_i and ln F_i = ln(1 - S_i) available
       for every ancestor literal,
           ln P(D_k) = sum_{i in pos} ln S_i + sum_{j in neg} ln F_j
           P(D_k)    = exp(ln P(D_k))                    (one exp per path)
           A_parents = sum_k P(D_k)                      (linear, disjoint paths)
       Nodes with NO gate fall back to the AND-over-graph-ancestors boundary, so
       gate-free workflows (e.g. volcano) reproduce the base formulation.

    2. TWO-SIDED LOG PROTECTION. Negative literals need ln(1 - S), so S is
       contracted into the OPEN interval: S_prot = S*(1 - 2 eps) + eps in
       [eps, 1 - eps], giving finite ln S and ln F for any feasible schedule.

    3. p_det ENTRY FACTOR. The horizontal timeline starts at
           Y_0 = A_parents * p_det
       where p_det (target present / in footprint) is a per-task CONSTANT from
       detection_probability_function (default 1.0). This is the correct place
       for a factor shared across a task's passes -- moving it out of the per-pass
       recurrence stops the redundancy machinery from over-crediting backup
       passes for a target that may simply be absent. p_det = 1 reproduces the
       base behavior exactly.

    4. LOGIC NODES. Belief-state nodes (e.g. MSA K_w) carry a gate but no passes.
       Their local success S equals their gate probability A_parents; they emit
       no reward and no bookings. They exist only to be referenced as literals by
       downstream gates.

    Everything else (McCormick horizontal recurrence, instance caps, conflicts,
    temporal constraints via _add_workflow_constraints, extraction) is shared
    with the base path.
    """
    import math

    ln_eps = math.log(epsilon)
    LN_FLOOR = -20.0

    observation_tasks = list(solution_holder.keys())
    ordered_nodes, dsop_of, is_logic = build_logical_dag(observation_tasks, workflow_graph)

    any_gate = any(dsop_of[n] is not None for n in ordered_nodes)
    if verbose > 0:
        n_logic = sum(1 for n in ordered_nodes if is_logic[n])
        n_paths = sum(len(dsop_of[n]) for n in ordered_nodes if dsop_of[n] is not None)
        print(f"[General-Logical] {len(ordered_nodes)} nodes "
              f"({n_logic} logic/state, {len(observation_tasks)} observation); "
              f"{n_paths} DSOP paths; "
              f"{'GATES present' if any_gate else 'no gates -> pure-AND fallback'}.")

    # Per-node continuous variables.
    A_parents = {}   # gate probability P(G_r)  (entry boundary, pre-p_det)
    S_local = {}     # LOCAL success S_r = P(r fulfilled | parents reached).
                     #   This is the literal probability used by downstream gates.
                     #   For observation nodes S_r = p_det * (1 - y_{K_r}) from the
                     #   UNSCALED local track (linear, exact -- see .tex footnote).
                     #   For logic nodes S_r == A_parents (== gate probability).
    ln_S = {}        # ln S_prot (positive literal)
    ln_F = {}        # ln(1 - S_prot) (negative literal)
    Y = {}           # SCALED remaining risk (for reward): Y_0 = A_parents * p_det
    W_abs = {}       # linearized x*Y per pass (reward-carrying mass)

    # Ancestor closure over the workflow graph (for the pure-AND fallback only).
    def _graph_ancestors(node):
        if workflow_graph is not None and node in workflow_graph:
            return [a for a in nx.ancestors(workflow_graph, node) if a in solution_holder]
        return []

    def _p_det_for(node):
        if detection_probability_function is None:
            return 1.0
        if task_to_passes.get(node):
            _sat0, _sp0 = task_to_passes[node][0]
            v = float(detection_probability_function(node, _sat0, _sp0))
        else:
            v = float(detection_probability_function(node, None, None))
        return min(1.0, max(0.0, v))

    # Helper: two-sided protected logs of a local-success source (var or const).
    def _emit_logs(node, name, s_source):
        # A CONSTANT source (an already-resolved task) pins S_prot to an ENDPOINT
        # of the contracted domain: realized=1 -> S_prot = 1-eps, realized=0 ->
        # S_prot = eps. At those points ln_S and ln_F sit EXACTLY on their own
        # variable bounds -- zero slack -- while addGenConstrLog is satisfied only
        # to within FuncPieceError. Presolve fixes the variables, evaluates the
        # curve, lands outside the bounds, and the model is infeasible before a
        # single simplex iteration. Every resolved literal is a landmine, and they
        # accumulate as the run proceeds, which is why this fires mid-simulation
        # rather than at the first solve.
        #
        # The two logs are KNOWN NUMBERS here. Fix them; emit no curve. This also
        # removes two gen-constraints per settled node, which is most of them late
        # in a run.
        if isinstance(s_source, (int, float)):
            sv = min(1.0 - epsilon,
                     max(epsilon, float(s_source) * (1.0 - 2.0 * epsilon) + epsilon))
            ln_s_v, ln_f_v = math.log(sv), math.log(1.0 - sv)
            ln_S[node] = model.addVar(lb=ln_s_v, ub=ln_s_v,
                                      vtype=GRB.CONTINUOUS, name=f"ln_S_{name}")
            ln_F[node] = model.addVar(lb=ln_f_v, ub=ln_f_v,
                                      vtype=GRB.CONTINUOUS, name=f"ln_F_{name}")
            return

        s_prot = model.addVar(lb=epsilon, ub=1.0 - epsilon,
                              vtype=GRB.CONTINUOUS, name=f"S_prot_{name}")
        model.addConstr(s_prot == s_source * (1.0 - 2.0 * epsilon) + epsilon,
                        name=f"prot_{name}")
        lnS = model.addVar(lb=ln_eps, ub=math.log(1.0 - epsilon),
                           vtype=GRB.CONTINUOUS, name=f"ln_S_{name}")
        model.addGenConstrLog(s_prot, lnS, name=f"log_S_{name}")
        one_minus = model.addVar(lb=epsilon, ub=1.0 - epsilon,
                                 vtype=GRB.CONTINUOUS, name=f"F_prot_{name}")
        model.addConstr(one_minus == 1.0 - s_prot, name=f"Fprot_{name}")
        lnF = model.addVar(lb=ln_eps, ub=math.log(1.0 - epsilon),
                           vtype=GRB.CONTINUOUS, name=f"ln_F_{name}")
        model.addGenConstrLog(one_minus, lnF, name=f"log_F_{name}")
        ln_S[node] = lnS
        ln_F[node] = lnF

    for node in ordered_nodes:
        name = _node_name(node)
        gate = _get_gate(node)
        gate_paths = dsop_of[node]

        # === Resolved observation tasks (constant literal) ====================
        # A gate may reference an observation task that is already dispatched or
        # completed and hence NOT in solution_holder (it has no decision vars this
        # solve). Its local success is a CONSTANT read from the realized outcome:
        # 1.0 if it executed successfully, else 0.0. It contributes no reward and
        # no A_parents machinery -- only ln_S / ln_F for downstream gate literals.
        if not is_logic[node] and node not in solution_holder:
            realized = 1.0 if getattr(node, 'successful_execution', False) else 0.0
            S_local[node] = realized
            _emit_logs(node, name, realized)
            continue

        # === Entry boundary A_parents = P(gate) ===============================
        A_parents[node] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                       name=f"A_parents_{name}")
        if isinstance(gate, ExclusiveOr):
            # Disjoint group: P = sum_j S_j (linear, <= 1 automatically). No DSOP,
            # no exp. Members are Lit gates over already-built literal nodes.
            member_nodes = [m.node for m in gate.members]
            model.addConstr(
                A_parents[node] == gp.quicksum(S_local[m] for m in member_nodes),
                name=f"Aparents_xor_{name}")
        elif gate_paths is not None:
            # DSOP: A_parents = sum_k P(D_k), each path an exp of a linear log sum.
            path_prob_vars = []
            for k, (pos, neg) in enumerate(gate_paths):
                if not pos and not neg:
                    pk = model.addVar(lb=1.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                      name=f"D_{name}_{k}")
                    path_prob_vars.append(pk)
                    continue
                ln_pk = model.addVar(lb=LN_FLOOR, ub=0.0, vtype=GRB.CONTINUOUS,
                                     name=f"lnD_{name}_{k}")
                model.addConstr(
                    ln_pk == gp.quicksum(ln_S[i] for i in pos)
                             + gp.quicksum(ln_F[j] for j in neg),
                    name=f"lnD_def_{name}_{k}")
                pk = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                  name=f"D_{name}_{k}")
                model.addGenConstrExp(ln_pk, pk, name=f"exp_D_{name}_{k}")
                path_prob_vars.append(pk)
            model.addConstr(A_parents[node] == gp.quicksum(path_prob_vars),
                            name=f"Aparents_dsop_{name}")
        else:
            # No gate: pure-AND fallback over graph ancestors (base-formulation
            # parity for gate-free workflows such as volcano).
            ancs = _graph_ancestors(node)
            if not ancs:
                model.addConstr(A_parents[node] == 1.0, name=f"root_Ap_{name}")
            else:
                ln_ap = model.addVar(lb=LN_FLOOR, ub=0.0, vtype=GRB.CONTINUOUS,
                                     name=f"ln_A_parents_{name}")
                model.addConstr(ln_ap == gp.quicksum(ln_S[a] for a in ancs),
                                name=f"join_lnAp_{name}")
                model.addGenConstrExp(ln_ap, A_parents[node], name=f"exp_Ap_{name}")

        # === Local success S_r ================================================
        if is_logic[node]:
            # Logic/state node: no passes, no reward. Local success == gate prob.
            S_local[node] = A_parents[node]
        else:
            p_det = _p_det_for(node)
            passes = task_to_passes[node]
            K_r = len(passes)

            # --- UNSCALED local track: y_0 = 1, y_{k+1} = y_k - theta_k*(x_k*y_k)
            #     recovers S_r = p_det * (1 - y_{K_r}), the LOCAL success used as a
            #     downstream literal. Linear/exact via McCormick; independent of
            #     A_parents so it never double-counts ancestor probability.
            y = {0: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                 name=f"y_{name}_k0")}
            model.addConstr(y[0] == 1.0, name=f"y0_{name}")
            for k in range(1, K_r + 1):
                y[k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                    name=f"y_{name}_k{k}")

            # --- SCALED reward track: Y_0 = A_parents * p_det (entry factor). ---
            Y[(node, 0)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                        name=f"Y_{name}_k0")
            model.addConstr(Y[(node, 0)] == p_det * A_parents[node],
                            name=f"inject_{name}")
            for k in range(1, K_r + 1):
                Y[(node, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                            name=f"Y_{name}_k{k}")

            for k, (satellite, satpass) in enumerate(passes):
                x_var = solution_holder[node][satellite][satpass]['x']
                theta_k = solution_holder[node][satellite][satpass]['theta']
                x_var.BranchPriority = 10

                # unscaled McCormick w_loc = x * y_k
                w_loc = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                     name=f"wloc_{name}_k{k}")
                model.addConstr(w_loc <= x_var, name=f"mcl1_{name}_k{k}")
                model.addConstr(w_loc <= y[k], name=f"mcl2_{name}_k{k}")
                model.addConstr(w_loc >= y[k] - (1.0 - x_var), name=f"mcl3_{name}_k{k}")
                model.addConstr(y[k + 1] == y[k] - theta_k * w_loc,
                                name=f"recl_{name}_k{k}")

                # scaled McCormick W_abs = x * Y_k  (reward-carrying)
                w = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                 name=f"w_abs_{name}_k{k}")
                W_abs[(node, satellite, satpass)] = w
                Y_cur = Y[(node, k)]
                model.addConstr(w <= x_var, name=f"mc1_{name}_k{k}")
                model.addConstr(w <= Y_cur, name=f"mc2_{name}_k{k}")
                model.addConstr(w >= Y_cur - (1.0 - x_var), name=f"mc3_{name}_k{k}")
                model.addConstr(Y[(node, k + 1)] == Y_cur - theta_k * w,
                                name=f"rec_{name}_k{k}")

            # Local success S_r = p_det * (1 - y_{K_r}).
            S_local[node] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                         name=f"S_{name}")
            model.addConstr(S_local[node] == p_det * (1.0 - y[K_r]),
                            name=f"Sdef_{name}")

        # === Two-sided protected logs of the LOCAL success ====================
        # Needed for downstream literals: ln S (positive) and ln F = ln(1 - S)
        # (negative). Contract S into [eps, 1 - eps] so both logs stay finite.
        _emit_logs(node, name, S_local[node])

    # === OBJECTIVE — only observation nodes contribute reward. =================
    # Costs use Q_MAX_task for provider-rate billing; decoupled from pass geometry.
    objective_terms = []
    for node in observation_tasks:
        if not solution_holder[node]:
            continue

        _q_max_task = max(
            solution_holder[node][s][p]['quality']
            for s, p in task_to_passes[node]
        )
        _has_success_parent = any(
            'SUCCESS' in str(e.get('constraint_class', ''))
            for p in workflow_graph.predecessors(node)
            for e in workflow_graph.get_edge_data(p, node).values()
        )
        if _has_success_parent:
            _parent_passes = [sp.highest.time for p in workflow_graph.predecessors(node) for sat, sp in task_to_passes.get(p, [])]
            _t_dispatch = min(_parent_passes) if _parent_passes else current_time
        else:
            _t_dispatch = current_time

        for satellite, satpass in task_to_passes[node]:
            x_var = solution_holder[node][satellite][satpass]['x']
            quality = solution_holder[node][satellite][satpass]['quality']
            theta = solution_holder[node][satellite][satpass]['theta']
            theta_acc = solution_holder[node][satellite][satpass]['theta_acc']
            w = W_abs[(node, satellite, satpass)]

            c_sub = submission_cost_rate * _q_max_task
            if execution_cost_fn is not None:
                try:
                    c_exec_k = execution_cost_fn(node, satellite, satpass, _t_dispatch, q_max=_q_max_task)
                except Exception:
                    c_exec_k = cancellation_cost_rate * _q_max_task
            else:
                c_exec_k = cancellation_cost_rate * _q_max_task
            c_tax = tax_rate * _q_max_task

            objective_terms.append(quality * theta * w)
            objective_terms.append(-c_sub * x_var)
            objective_terms.append(-c_exec_k * theta_acc * x_var)
            if c_tax:
                objective_terms.append(-c_tax * x_var)
    model.update()
    for node in ordered_nodes:
        if is_logic[node]:
            continue
        _nm = _node_name(node)
        _pd = _p_det_for(node)
        _apub = A_parents[node].UB if node in A_parents else None
        _np = len(task_to_passes.get(node, []))
        print(f"[GL-DBG] {_nm:34s} p_det={_pd:.3f} passes={_np:2d} "
              f"A_parents.ub={_apub}")
    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)

    # === Return a warm-start closure ==========================================
    # Captures all variable handles so the caller can propagate a binary x
    # assignment through the entire continuous chain before calling optimize().
    # All per-node state dicts (A_parents, S_local, Y, W_abs, ln_S, ln_F) are
    # captured by closure; the caller passes x_vals: {(req,sat,sp)->0|1}.
    _ws_ordered_nodes = ordered_nodes
    _ws_is_logic = is_logic
    _ws_dsop_of = dsop_of
    _ws_A_parents = A_parents
    _ws_S_local = S_local
    _ws_ln_S = ln_S
    _ws_ln_F = ln_F
    _ws_Y = Y
    _ws_W_abs = W_abs
    _ws_solution_holder = solution_holder
    _ws_task_to_passes = task_to_passes
    _ws_epsilon = epsilon

    def _warm_start_fn(x_vals: dict) -> None:
        """Set .Start on all continuous vars given a binary x assignment.

        x_vals: {(req, sat, sp) -> 0.0 or 1.0}
        Works by a forward numerical pass in topological order.
        """
        import math as _math

        # Numerical S_local values (same meaning as the Gurobi S_local vars).
        _S_num: dict = {}

        for node in _ws_ordered_nodes:
            # Resolved (dispatched/completed) obs task: constant.
            if not _ws_is_logic[node] and node not in _ws_solution_holder:
                realized = 1.0 if getattr(node, 'successful_execution', False) else 0.0
                _S_num[node] = realized
                # Set S_prot, ln_S, ln_F .Start for this constant node.
                _s_prot_v = realized * (1.0 - 2.0 * _ws_epsilon) + _ws_epsilon
                _s_prot_v = max(_ws_epsilon, min(1.0 - _ws_epsilon, _s_prot_v))
                _ln_s_v = _math.log(_s_prot_v)
                _ln_f_v = _math.log(1.0 - _s_prot_v)
                if _ws_ln_S.get(node) is not None:
                    _ws_ln_S[node].Start = _ln_s_v
                    _ws_ln_F[node].Start = _ln_f_v
                continue

            gate = _get_gate(node)
            gate_paths = _ws_dsop_of[node]

            # --- Numerical A_parents ---
            if isinstance(gate, ExclusiveOr):
                _ap_num = sum(_S_num.get(m.node, 0.0) for m in gate.members)
            elif gate_paths is not None:
                _ap_num = 0.0
                for pos, neg in gate_paths:
                    _lnpk = sum(_math.log(max(_ws_epsilon, _S_num.get(i, _ws_epsilon)))
                                for i in pos)
                    _lnpk += sum(_math.log(max(_ws_epsilon, 1.0 - _S_num.get(j, 1.0 - _ws_epsilon)))
                                 for j in neg)
                    _ap_num += _math.exp(_lnpk)
            else:
                # Pure-AND fallback: product over graph-ancestor S_num.
                _ap_num = 1.0
                if workflow_graph is not None and node in workflow_graph:
                    for anc in nx.ancestors(workflow_graph, node):
                        if anc in _ws_solution_holder and anc in _S_num:
                            _ap_num *= max(_ws_epsilon, _S_num[anc])

            _ap_num = max(0.0, min(1.0, _ap_num))
            if _ws_A_parents.get(node) is not None:
                _ws_A_parents[node].Start = _ap_num

            if _ws_is_logic[node]:
                _S_num[node] = _ap_num
            else:
                p_det = 1.0
                if detection_probability_function is not None:
                    _passes = _ws_task_to_passes.get(node, [])
                    if _passes:
                        _sat0, _sp0 = _passes[0]
                        p_det = float(detection_probability_function(node, _sat0, _sp0))
                        p_det = max(0.0, min(1.0, p_det))

                passes = _ws_task_to_passes.get(node, [])
                K_r = len(passes)

                # Forward pass through unscaled y[] track (y[0] = 1).
                _y_num = {0: 1.0}
                for k, (sat, sp) in enumerate(passes):
                    xv = x_vals.get((node, sat, sp), 0.0)
                    theta_k = _ws_solution_holder[node][sat][sp]['theta']
                    w_loc_num = xv * _y_num[k]
                    _y_num[k + 1] = _y_num[k] - theta_k * w_loc_num

                s_num = p_det * (1.0 - _y_num[K_r])
                _S_num[node] = max(0.0, min(1.0, s_num))

                # Forward pass through scaled Y[] track (Y[0] = A_parents * p_det).
                _Y_num = {0: _ap_num * p_det}
                for k, (sat, sp) in enumerate(passes):
                    xv = x_vals.get((node, sat, sp), 0.0)
                    theta_k = _ws_solution_holder[node][sat][sp]['theta']
                    w_abs_num = xv * _Y_num[k]
                    _Y_num[k + 1] = _Y_num[k] - theta_k * w_abs_num
                    # Set .Start on Y vars and W_abs vars if accessible by name.
                    _yvar_k = model.getVarByName(f"Y_{_node_name(node)}_k{k}")
                    if _yvar_k is not None:
                        _yvar_k.Start = max(0.0, min(1.0, _Y_num[k]))
                    _wabs_var = _ws_W_abs.get((node, sat, sp))
                    if _wabs_var is not None:
                        _wabs_var.Start = max(0.0, min(1.0, w_abs_num))
                    _y0var = model.getVarByName(f"y_{_node_name(node)}_k{k}")
                    if _y0var is not None:
                        _y0var.Start = max(0.0, min(1.0, _y_num[k]))
                _yvar_Kr = model.getVarByName(f"Y_{_node_name(node)}_k{K_r}")
                if _yvar_Kr is not None:
                    _yvar_Kr.Start = max(0.0, min(1.0, _Y_num[K_r]))
                _y0var_Kr = model.getVarByName(f"y_{_node_name(node)}_k{K_r}")
                if _y0var_Kr is not None:
                    _y0var_Kr.Start = max(0.0, min(1.0, _y_num[K_r]))
                _y0var_0 = model.getVarByName(f"y_{_node_name(node)}_k0")
                if _y0var_0 is not None:
                    _y0var_0.Start = 1.0
                _Y0var = model.getVarByName(f"Y_{_node_name(node)}_k0")
                if _Y0var is not None:
                    _Y0var.Start = max(0.0, min(1.0, _ap_num * p_det))
                _svar = model.getVarByName(f"S_{_node_name(node)}")
                if _svar is not None:
                    _svar.Start = _S_num[node]

            # Set .Start on ln_S / ln_F vars.
            _s_v = max(_ws_epsilon, min(1.0 - _ws_epsilon, _S_num[node]))
            _s_prot_v = _s_v * (1.0 - 2.0 * _ws_epsilon) + _ws_epsilon
            _s_prot_v = max(_ws_epsilon, min(1.0 - _ws_epsilon, _s_prot_v))
            if _ws_ln_S.get(node) is not None:
                _ws_ln_S[node].Start = _math.log(_s_prot_v)
                _ws_ln_F[node].Start = _math.log(1.0 - _s_prot_v)

    return _warm_start_fn
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
        reward_envelope_cuts: bool = True,
        execution_cost_fn=None,
        current_time=None,
        detection_probability_function=None
):
    """
    Log-linearized stochastic formulation (paper Sections 5.2-5.6), built in a
    single topological sweep over the DAG.

    ===========================================================================
    p_det ENTRY BOUNDARY  (new; None => 1.0 => exactly the previous behaviour)
    ===========================================================================
    A per-task detection probability pi_r (target present / inside the footprint)
    now enters at the START of the horizontal timeline:

        Y_{r,0} = pi_r * A_parents_r                      (was: A_parents_r)
        e2e_r   = Y_{r,0} - Y_{r,K_r}                      (was: A_parents_r - Y_K)
        S_r     = pi_r * (1 - y_{r,K_r})                   (implicit, via e2e)

    WHY THE ENTRY BOUNDARY AND NOT theta.  pi_r is SHARED across a task's
    redundant passes: they all aim at the same point, so if the target is not
    there they ALL miss -- perfectly correlated.  Folding pi_r into theta_k
    would make the recurrence treat them as independent detection draws and
    systematically over-credit redundancy.  Acceptance and execution ARE
    independent per pass and stay in theta_k.

    The identity chain is unchanged:
        Y_{r,K} = pi_r * A_parents_r * prod_k (1 - theta_k x_k)
              => e2e_r = pi_r * A_parents_r * (1 - y_{r,K}) = A_parents_r * S_r
    so A_prot / ln_A / ln_S and the log-space join all carry through verbatim.

    BEST-OF-N IS PRESERVED.  Passes are sorted by quality DESCENDING, so the
    diminishing recurrence credits the FIRST success in quality order, i.e. the
    HIGHEST-QUALITY success -- not the first in time.  Verified by enumeration
    against E[Q of best success] for pi_r = 1 and pi_r < 1.

    All best-case constants, variable bounds and valid cuts are scaled by pi_r
    so the relaxation stays as tight as before.

    ---------------------------------------------------------------------------
    Existing properties (each fixing a previously observed failure mode):

    1. PROPORTIONAL log-protection floor:
           A_prot = A_e2e * (1 - eps) + eps * A_parents
       so A_prot <= A_parents holds structurally and ln_S = ln(A_prot) -
       ln(A_parents) <= 0 is ALWAYS satisfiable ("adaptive floor trap" fix).

    2. DEPTH-AWARE log-variable bounds (ln_S in [ln eps, 0] exactly).

    3. SINGLE-PARENT SHORTCUT: chain nodes get a linear entry boundary,
       removing one exp() per chain node.

    4. REAL-SPACE VALID CUTS: A_parents[r] <= A_prot[a] for every ancestor a.

    5. DATA-DRIVEN BOUND PROPAGATION (cardinality-aware best-case constants).

    6. UNION-BOUND CUTS tying claimable reward to scheduled probability mass.

    7. BINARY BRANCH PRIORITY on x.

    NOTE: pwl_tolerance is unused here (kept for API compatibility).
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
    LN_FLOOR = -20  # absolute cap on log-space lower bounds

    topo = [r for r in nx.topological_sort(workflow_graph) if r in solution_holder]

    anc_sets = {
        r: frozenset(a for a in nx.ancestors(workflow_graph, r) if a in solution_holder)
        for r in topo
    }

    # --- p_det per task -------------------------------------------------------
    # A CONSTANT per task, evaluated once. None => 1.0, which reduces every
    # expression below to the pre-p_det formulation exactly (volcano parity).
    def _p_det_for(node):
        if detection_probability_function is None:
            return 1.0
        try:
            if task_to_passes.get(node):
                _sat0, _sp0 = task_to_passes[node][0]
                v = float(detection_probability_function(node, _sat0, _sp0))
            else:
                v = float(detection_probability_function(node, None, None))
        except Exception:
            return 1.0
        if v != v:          # NaN guard
            return 1.0
        return min(1.0, max(0.0, v))

    p_det = {r: _p_det_for(r) for r in topo}

    # --- Constant bound propagation, CARDINALITY- AND p_det-AWARE -------------
    lmax_ub, S_ub, A_par_ub, e2e_ub, A_prot_ub, M_of = {}, {}, {}, {}, {}, {}
    for r in topo:
        M = _effective_max_instances(r, default_max_instances)
        M_of[r] = max(0, min(M, len(task_to_passes[r])))
        thetas = sorted(
            (solution_holder[r][sat][sp]['theta'] for (sat, sp) in task_to_passes[r]),
            reverse=True)[:M_of[r]]
        lmax = 1.0 - math.prod(1.0 - t for t in thetas) if thetas else 0.0
        lmax_ub[r] = min(1.0, lmax)
        # S_r <= pi_r * lmax_M(r); protected the same proportional way.
        S_ub[r] = p_det[r] * lmax_ub[r] * (1.0 - epsilon) + epsilon
        A_par_ub[r] = math.prod(S_ub[a] for a in anc_sets[r]) if anc_sets[r] else 1.0
        e2e_ub[r] = A_par_ub[r] * p_det[r] * lmax_ub[r]
        A_prot_ub[r] = A_par_ub[r] * S_ub[r]   # == e2e_ub*(1-eps) + eps*A_par_ub
    if not tighten_bounds:
        for r in topo:
            A_par_ub[r], A_prot_ub[r], S_ub[r] = 1.0, 1.0, 1.0
            lmax_ub[r] = 1.0
            e2e_ub[r] = p_det[r]

    # --- Node classification + nonlinearity pruning ---------------------------
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
        _pds = [p_det[r] for r in topo]
        if _pds and min(_pds) < 1.0:
            print(f"[Log-Linearized] p_det ACTIVE: min={min(_pds):.3f} "
                  f"mean={sum(_pds)/len(_pds):.3f} max={max(_pds):.3f} "
                  f"({sum(1 for v in _pds if v < 1.0)}/{len(_pds)} tasks < 1.0)")
        else:
            print("[Log-Linearized] p_det inactive (all 1.0) -- reduces to the base formulation.")
        if lmaxs:
            print(f"[Log-Linearized] ENGAGEMENT CHECK -- instance caps M (histogram): {M_hist}; "
                  f"lmax_M: min={min(lmaxs):.3f} mean={sum(lmaxs)/len(lmaxs):.3f} max={max(lmaxs):.3f}; "
                  f"tighten_bounds={tighten_bounds}. "
                  f"(If M is mostly 1, hedging is OFF; if lmax_M ~1.0, drain cuts are weak.)")
        else:
            print("[Log-Linearized] ENGAGEMENT CHECK -- no task in this window has any "
                  "feasible pass; nothing to schedule or hedge.")

    for constrained_request in topo:
        req_name = getattr(getattr(constrained_request, 'observation_request',
                                   constrained_request), 'name', str(id(constrained_request)))
        n_anc = len(anc_sets[constrained_request])
        pi_r = p_det[constrained_request]

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
            ln_S_vars[constrained_request] = model.addVar(
                lb=ln_eps, ub=math.log(S_ub[r_ub]) if S_ub[r_ub] < 1.0 else 0.0,
                vtype=GRB.CONTINUOUS, name=f"ln_S_{req_name}")
        if needs_log or kind == 'merge':
            ln_A_parents_vars[constrained_request] = model.addVar(
                lb=lb_ln_parents, ub=math.log(A_par_ub[r_ub]) if A_par_ub[r_ub] < 1.0 else 0.0,
                vtype=GRB.CONTINUOUS, name=f"ln_A_parents_{req_name}")

        # --- Vertical entry boundary (paper Sec 5.3) --------------------------
        direct_parents = [p for p in workflow_graph.predecessors(constrained_request)
                          if p in solution_holder]

        if kind == 'root':
            model.addConstr(ancestor_success_prob[constrained_request] == 1.0,
                            name=f"root_Ap_{req_name}")
            if needs_log or kind == 'merge':
                model.addConstr(ln_A_parents_vars[constrained_request] == 0.0,
                                name=f"root_lnAp_{req_name}")
        elif kind == 'chain':
            q = direct_parents[0]
            model.addConstr(ancestor_success_prob[constrained_request] == A_prot_vars[q],
                            name=f"chain_Ap_{req_name}")
            if needs_log:
                model.addConstr(ln_A_parents_vars[constrained_request] == ln_A_vars[q],
                                name=f"chain_lnAp_{req_name}")
        else:
            model.addConstr(
                ln_A_parents_vars[constrained_request]
                == gp.quicksum(ln_S_vars[anc] for anc in anc_sets[constrained_request]),
                name=f"join_lnAp_{req_name}")
            model.addGenConstrExp(ln_A_parents_vars[constrained_request],
                                  ancestor_success_prob[constrained_request],
                                  name=f"exp_Ap_{req_name}")
            for anc in anc_sets[constrained_request]:
                model.addConstr(
                    ancestor_success_prob[constrained_request] <= A_prot_vars[anc],
                    name=f"cut_Ap_le_Aprot_{req_name}_"
                         f"{getattr(getattr(anc, 'observation_request', anc), 'name', id(anc))}")

        # --- Horizontal scaled timeline (paper Sec 5.2) -----------------------
        passes = task_to_passes[constrained_request]
        K_r = len(passes)

        # Y is scaled by pi_r, so its upper bound is too.
        _Y_ub = p_det[constrained_request] * A_par_ub[constrained_request]
        for k in range(K_r + 1):
            scaled_remaining_risk[(constrained_request, k)] = model.addVar(
                lb=0.0, ub=_Y_ub, vtype=GRB.CONTINUOUS, name=f"Y_{req_name}_k{k}")

        # *** p_det ENTRY BOUNDARY ***
        # The timeline starts at (parents' joint success) x (target detectable).
        # pi_r is shared across this task's redundant passes -- they all aim at
        # the same point, so their detection outcomes are perfectly correlated.
        # Putting pi_r here rather than in theta_k is what stops the recurrence
        # from treating them as independent detection draws.
        model.addConstr(
            scaled_remaining_risk[(constrained_request, 0)]
            == pi_r * ancestor_success_prob[constrained_request],
            name=f"inject_{req_name}")

        for k, (satellite, satpass) in enumerate(passes):
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            theta_k = solution_holder[constrained_request][satellite][satpass]['theta']
            Y_current = scaled_remaining_risk[(constrained_request, k)]

            w_abs = model.addVar(lb=0.0, ub=_Y_ub,
                                 vtype=GRB.CONTINUOUS, name=f"w_abs_{req_name}_k{k}")
            effective_pass_realization[(constrained_request, satellite, satpass)] = w_abs
            x_var.BranchPriority = 10

            model.addConstr(w_abs <= x_var, name=f"mc1_{req_name}_k{k}")
            model.addConstr(w_abs <= Y_current, name=f"mc2_{req_name}_k{k}")
            model.addConstr(w_abs >= Y_current - (1.0 - x_var), name=f"mc3_{req_name}_k{k}")

            model.addConstr(
                scaled_remaining_risk[(constrained_request, k + 1)] == Y_current - theta_k * w_abs,
                name=f"rec_{req_name}_k{k}")

        # End-of-horizon fulfillment (paper Eq. 29), now measured from Y_0 so it
        # is A_parents * S_r with S_r = pi_r * (1 - y_K).  With pi_r = 1 this is
        # identical to the previous "A_parents - Y_K".
        model.addConstr(
            end_to_end_success[constrained_request]
            == scaled_remaining_risk[(constrained_request, 0)]
               - scaled_remaining_risk[(constrained_request, K_r)],
            name=f"e2e_{req_name}")

        # Union-bound cut, scaled by pi_r.
        if K_r > 0 and (pi_r * A_par_ub[constrained_request]) >= 1e-9:
            model.addConstr(
                end_to_end_success[constrained_request]
                <= pi_r * A_par_ub[constrained_request] * gp.quicksum(
                    solution_holder[constrained_request][sat][sp]['theta']
                    * solution_holder[constrained_request][sat][sp]['x']
                    for (sat, sp) in passes),
                name=f"cut_union_{req_name}")

        # DRAIN CUT: no integer solution can exceed pi_r * (top-M union prob).
        if (tighten_bounds and K_r > 0
                and 1e-9 <= (pi_r * lmax_ub[constrained_request]) < 1.0):
            model.addConstr(
                end_to_end_success[constrained_request]
                <= pi_r * lmax_ub[constrained_request]
                   * ancestor_success_prob[constrained_request],
                name=f"cut_drain_{req_name}")

        # REWARD / SURVIVAL / QUALITY-PROBABILITY ENVELOPE CUTS, all scaled by pi_r.
        # Envelope cuts are SKIPPED when their scale is numerically meaningless.
        # _base = A_par_ub * pi_r can reach ~1e-11 when a small p_det multiplies a
        # small ancestor product; coefficients that size pollute the constraint
        # matrix (observed range [5e-11, 1e+02]) and Gurobi treats those rows as
        # noise -- which is also why the IIS came back "non-minimal" with
        # "Numerical troubles encountered".  Dropping them costs only bound
        # tightness on tasks that are worth ~nothing anyway.
        _base_scale = A_par_ub[constrained_request] * pi_r
        _cuts_ok = _base_scale >= 1e-6
        if reward_envelope_cuts and tighten_bounds and K_r > 0 and not _cuts_ok and verbose > 1:
            print(f"  [Log-Linearized] envelope cuts skipped for {req_name} "
                  f"(scale {_base_scale:.2e} below 1e-6)")

        if reward_envelope_cuts and tighten_bounds and K_r > 0 and _cuts_ok:
            _thetas_desc = sorted(
                (solution_holder[constrained_request][sat][sp]['theta'] for (sat, sp) in passes),
                reverse=True)
            _Qmax = max(solution_holder[constrained_request][sat][sp]['quality']
                        for (sat, sp) in passes)
            _base = A_par_ub[constrained_request] * pi_r
            _scale = _base * _Qmax
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
                model.addConstr(
                    end_to_end_success[constrained_request]
                    <= _base * (_lmax_curve[_n] - _slope * _n)
                       + _base * _slope * _xsum,
                    name=f"cut_senv_{req_name}_n{_n}")

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
                    _reward_expr <= _base
                                    * ((_qp_cum[_n] - _slope_qp * _n) + _slope_qp * _xsum),
                    name=f"cut_qpenv_{req_name}_n{_n}")

        # --- PROPORTIONAL floor + protected log -------------------------------
        A_prot = model.addVar(lb=max(math.exp(lb_ln_A), 1e-30),
                              ub=A_prot_ub[constrained_request],
                              vtype=GRB.CONTINUOUS, name=f"A_prot_{req_name}")
        A_prot_vars[constrained_request] = A_prot
        model.addConstr(
            A_prot == end_to_end_success[constrained_request] * (1.0 - epsilon)
                      + epsilon * ancestor_success_prob[constrained_request],
            name=f"prot_{req_name}")
        if needs_log:
            model.addGenConstrLog(A_prot, ln_A_vars[constrained_request], name=f"log_{req_name}")
            model.addConstr(
                ln_S_vars[constrained_request]
                == ln_A_vars[constrained_request] - ln_A_parents_vars[constrained_request],
                name=f"lnS_{req_name}")

    # === OBJECTIVE ============================================================
    # Maximize sum_k Q_k * theta_k * W_abs_k
    #         - sum_k (c_sub_k + c_exec_k * theta_acc_k + c_tax_k) * x_k
    #
    # p_det needs NO separate objective term: W_abs rides on Y, which already
    # starts at pi_r * A_parents, so the reward is discounted automatically.
    # Costs are NOT discounted by p_det -- you pay to book the pass whether or
    # not the target turns out to be in the footprint, which is exactly the
    # asymmetry that makes low-p_det tasks correctly unattractive.
    objective_terms = []
    for constrained_request in topo:
        if not solution_holder[constrained_request]:
            continue

        _q_max_task = max(
            solution_holder[constrained_request][s][p]['quality']
            for s, p in task_to_passes[constrained_request]
        )
        _has_success_parent = any(
            'SUCCESS' in str(e.get('constraint_class', ''))
            for p in workflow_graph.predecessors(constrained_request)
            for e in workflow_graph.get_edge_data(p, constrained_request).values()
        )
        if _has_success_parent:
            _parent_passes = [sp.highest.time
                              for p in workflow_graph.predecessors(constrained_request)
                              for sat, sp in task_to_passes.get(p, [])]
            _t_dispatch = min(_parent_passes) if _parent_passes else current_time
        else:
            _t_dispatch = current_time

        for satellite, satpass in task_to_passes[constrained_request]:
            x_var = solution_holder[constrained_request][satellite][satpass]['x']
            quality = solution_holder[constrained_request][satellite][satpass]['quality']
            theta = solution_holder[constrained_request][satellite][satpass]['theta']
            theta_acc = solution_holder[constrained_request][satellite][satpass]['theta_acc']
            w_abs = effective_pass_realization[(constrained_request, satellite, satpass)]

            c_sub = submission_cost_rate * _q_max_task
            if execution_cost_fn is not None:
                try:
                    c_exec_k = execution_cost_fn(constrained_request, satellite, satpass,
                                                 _t_dispatch, q_max=_q_max_task)
                except Exception:
                    c_exec_k = cancellation_cost_rate * _q_max_task
            else:
                c_exec_k = cancellation_cost_rate * _q_max_task
            c_tax = tax_rate * _q_max_task

            objective_terms.append(quality * theta * w_abs)
            objective_terms.append(-c_sub * x_var)
            objective_terms.append(-c_exec_k * theta_acc * x_var)
            if c_tax:
                objective_terms.append(-c_tax * x_var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)

# def _build_log_linearized_formulation(
#         model: gp.Model,
#         workflow_graph: nx.MultiDiGraph,
#         solution_holder: dict,
#         task_to_passes: dict,
#         epsilon: float,
#         pwl_tolerance: float,
#         tax_rate: float,
#         submission_cost_rate: float,
#         cancellation_cost_rate: float,
#         verbose: int,
#         tighten_bounds: bool = True,
#         default_max_instances: int = 3,
#         reward_envelope_cuts: bool = True,
#         execution_cost_fn=None,
#         current_time=None
# ):
#     """
#     Log-linearized stochastic formulation (paper Sections 5.2-5.6), built in a
#     single topological sweep over the DAG.

#     Key properties (each fixing a previously observed failure mode):

#     1. PROPORTIONAL log-protection floor:
#            A_prot = A_e2e * (1 - eps) + eps * A_parents
#        (instead of the static "+ eps"). Consequence: A_prot <= A_parents holds
#        structurally, so ln_S = ln(A_prot) - ln(A_parents) <= 0 is ALWAYS
#        satisfiable and ln_S >= ln(eps) exactly. The static floor could force
#        ln_S > 0 for deep tasks with weak ancestors, which collided with the
#        ln_S <= 0 bound and created massive infeasibility pressure / branching
#        churn ("adaptive floor trap"). The proportional floor removes the trap
#        at its root.

#     2. DEPTH-AWARE log-variable bounds. Because ln_S in [ln(eps), 0] exactly
#        (see 1), the valid bounds are:
#            ln_A_parents >= n_anc * ln(eps)
#            ln_A         >= (n_anc + 1) * ln(eps)
#        A static bound of -12 silently acted as a hidden constraint forcing
#        ancestral chains to stay healthy (second door into the floor trap).
#        Bounds are capped at LN_FLOOR for numerical sanity; the cap only binds
#        for chains of many near-dead ancestors, which are objective-irrelevant.

#     3. SINGLE-PARENT SHORTCUT: for a node r whose unique in-model ancestors
#        equal {q} + ancestors(q) for a single direct parent q (chain structure),
#        the entry boundary is set linearly:
#            ln_A_parents[r] == ln_A[q],   A_parents[r] == A_prot[q]
#        This is exact (Sum of ancestor ln_S telescopes to ln_A[q]) and removes
#        one exp() general constraint per chain node -- the dominant source of
#        MINLP work in chain-heavy workflows.

#     4. REAL-SPACE VALID CUTS: A_parents[r] <= A_prot[a] for every in-model
#        ancestor a (event inclusion: r's ancestral-success event is a subset of
#        a's protected end-to-end event). These bound the exp() relaxation in
#        probability space directly, bypassing the log machinery exactly where
#        its relaxation is loosest, recovering speed after switching to exact
#        nonlinear handling (FuncNonlinear=1).

#     5. DATA-DRIVEN BOUND PROPAGATION (dual-bound tightening). The incumbent
#        is typically found quickly; what is expensive is proving optimality,
#        because the LP/OA relaxation of the exp()/log() equalities is one-sided
#        (the relaxed A_parents can float up to the chord of exp between its
#        variable bounds) and that overestimation COMPOUNDS multiplicatively
#        down the DAG, inflating the root dual bound. We therefore precompute,
#        in one topological pass over constants, the best-case protected
#        probabilities with ALL passes scheduled:
#            lmax(r)      = 1 - prod_k (1 - p_{r,k})              (union of all passes)
#            S_ub(r)      = lmax(r)*(1-eps) + eps
#            A_par_ub(r)  = prod_{a in anc(r)} S_ub(a)
#            e2e_ub(r)    = A_par_ub(r) * lmax(r)
#            A_prot_ub(r) = A_par_ub(r) * S_ub(r)
#        and install them as VARIABLE BOUNDS (plus matching log-space bounds).
#        Scheduling more passes only increases success probabilities, so these
#        are valid regardless of conflicts/max-instances; they cut the chord gap
#        of every exp/log relaxation at the root, before any branching.

#     6. UNION-BOUND CUTS: e2e[r] <= A_par_ub(r) * sum_k p_k x_k, valid since
#        1 - prod(1 - p x) <= sum p x. Ties the reward a task can claim in the
#        relaxation to the probability mass actually scheduled, so fractional
#        solutions cannot harvest reward without paying for bookings.

#     7. BINARY BRANCH PRIORITY: x variables get BranchPriority 10 so Gurobi
#        branches the schedule decisions before spatially branching the
#        continuous exp/log operands -- once x is integral the McCormick track
#        is exact and interval tightening closes the rest fast.

#     NOTE: pwl_tolerance is unused here (kept for API compatibility); the
#     Gurobi path relies on FuncNonlinear=1 for exact exp/log handling.
#     """
#     import math

#     scaled_remaining_risk = {}
#     effective_pass_realization = {}
#     ancestor_success_prob = {}
#     end_to_end_success = {}
#     A_prot_vars = {}
#     ln_A_vars = {}
#     ln_A_parents_vars = {}
#     ln_S_vars = {}

#     ln_eps = math.log(epsilon)
#     LN_FLOOR = -20  # absolute cap on log-space lower bounds (exp(-50) ~ 2e-22)

#     # Topological order restricted to tasks actually in the model; guarantees
#     # every ancestor's variables exist before its descendants reference them.
#     topo = [r for r in nx.topological_sort(workflow_graph) if r in solution_holder]

#     # Unique-ancestor closure sets (transitive closure trick: summing local
#     # ln_S over the UNIQUE ancestor set gives the exact joint ancestral
#     # probability under independence on general DAGs -- shared ancestors of
#     # diamond patterns are counted exactly once).
#     anc_sets = {
#         r: frozenset(a for a in nx.ancestors(workflow_graph, r) if a in solution_holder)
#         for r in topo
#     }

#     # --- Constant bound propagation, CARDINALITY-AWARE -------------------------
#     # The instances cap (max_num_instances) is a hard constraint, so no integer
#     # solution can ever schedule more than M_r passes. All best-case constants
#     # are therefore computed over the TOP-M_r probabilities only:
#     #     lmax_M(r) = 1 - prod_{k in top-M_r}(1 - p_k)
#     # Computing them over ALL passes (previous version) degenerates to ~1.0 as
#     # soon as a request has many candidate passes, making every bound trivial.
#     lmax_ub, S_ub, A_par_ub, e2e_ub, A_prot_ub, M_of = {}, {}, {}, {}, {}, {}
#     for r in topo:
#         M = _effective_max_instances(r, default_max_instances)
#         M_of[r] = max(0, min(M, len(task_to_passes[r])))
#         thetas = sorted(
#             (solution_holder[r][sat][sp]['theta'] for (sat, sp) in task_to_passes[r]),
#             reverse=True)[:M_of[r]]
#         lmax = 1.0 - math.prod(1.0 - t for t in thetas) if thetas else 0.0
#         lmax_ub[r] = min(1.0, lmax)
#         S_ub[r] = lmax_ub[r] * (1.0 - epsilon) + epsilon
#         A_par_ub[r] = math.prod(S_ub[a] for a in anc_sets[r]) if anc_sets[r] else 1.0
#         e2e_ub[r] = A_par_ub[r] * lmax_ub[r]
#         A_prot_ub[r] = A_par_ub[r] * S_ub[r]  # == e2e_ub*(1-eps) + eps*A_par_ub
#     if not tighten_bounds:
#         for r in topo:
#             A_par_ub[r], e2e_ub[r], A_prot_ub[r], S_ub[r] = 1.0, 1.0, 1.0, 1.0
#             lmax_ub[r] = 1.0

#     # --- Node classification + nonlinearity pruning ----------------------------
#     # Classify every node once: ROOT (no in-model ancestors), CHAIN (single
#     # in-model parent whose closure telescopes), MERGE (general log-space join,
#     # needs an exp() constraint). Then compute which tasks actually need their
#     # log() constraint: ln_S(r) is consumed ONLY inside merge-node joins, so
#     # log machinery is required exactly on the union of merge-node ancestor
#     # closures. Everything else propagates through the purely LINEAR real-space
#     # identities (A_parents[child] == A_prot[parent]). Consequence: a window
#     # with no live merge nodes builds a PURE MILP -- zero nonlinear constraints.
#     node_kind = {}
#     for r in topo:
#         dps = [p for p in workflow_graph.predecessors(r) if p in solution_holder]
#         if not anc_sets[r]:
#             node_kind[r] = 'root'
#         elif len(dps) == 1 and anc_sets[r] == anc_sets[dps[0]] | {dps[0]}:
#             node_kind[r] = 'chain'
#         else:
#             node_kind[r] = 'merge'
#     merge_nodes = [r for r in topo if node_kind[r] == 'merge']
#     need_lnS = set()
#     for m in merge_nodes:
#         need_lnS |= anc_sets[m]
#     if verbose > 0:
#         import collections
#         M_hist = dict(collections.Counter(M_of[r] for r in topo))
#         lmaxs = [lmax_ub[r] for r in topo if task_to_passes[r]]
#         print(f"[Log-Linearized] {len(merge_nodes)} merge nodes; log constraints "
#               f"pruned to {len(need_lnS)} of {len(topo)} tasks "
#               f"({'PURE MILP' if not merge_nodes else 'MINLP on merge closures only'}).")
#         if lmaxs:
#             print(f"[Log-Linearized] ENGAGEMENT CHECK -- instance caps M (histogram): {M_hist}; "
#                   f"lmax_M: min={min(lmaxs):.3f} mean={sum(lmaxs)/len(lmaxs):.3f} max={max(lmaxs):.3f}; "
#                   f"tighten_bounds={tighten_bounds}. "
#                   f"(If M is mostly 1, hedging is OFF; if lmax_M ~1.0, drain cuts are weak.)")
#         else:
#             print(f"[Log-Linearized] ENGAGEMENT CHECK -- no task in this window has any "
#                   f"feasible pass (M histogram: {M_hist}); nothing to schedule or hedge.")

#     if verbose > 0:
#         n_chain = sum(
#             1 for r in topo
#             if len([p for p in workflow_graph.predecessors(r) if p in solution_holder]) == 1
#         )
#         print(f"[Log-Linearized] Building exact formulation for {len(topo)} tasks "
#               f"({n_chain} single-parent candidates for the linear shortcut).")

#     for constrained_request in topo:
#         req_name = getattr(getattr(constrained_request, 'observation_request', constrained_request), 'name', str(id(constrained_request)))
#         n_anc = len(anc_sets[constrained_request])

#         # --- Depth-aware bounds -------------------------------------------------
#         lb_ln_parents = max(n_anc * ln_eps, LN_FLOOR) if n_anc > 0 else 0.0
#         lb_ln_A = max((n_anc + 1) * ln_eps, LN_FLOOR)

#         r_ub = constrained_request
#         kind = node_kind[constrained_request]
#         needs_log = constrained_request in need_lnS
#         ancestor_success_prob[constrained_request] = model.addVar(
#             lb=math.exp(lb_ln_parents) if n_anc > 0 else 1.0, ub=A_par_ub[r_ub],
#             vtype=GRB.CONTINUOUS, name=f"A_parents_{req_name}")
#         end_to_end_success[constrained_request] = model.addVar(
#             lb=0.0, ub=e2e_ub[r_ub], vtype=GRB.CONTINUOUS, name=f"A_node_{req_name}")
#         if needs_log:
#             ln_A_vars[constrained_request] = model.addVar(
#                 lb=lb_ln_A, ub=math.log(A_prot_ub[r_ub]) if A_prot_ub[r_ub] < 1.0 else 0.0,
#                 vtype=GRB.CONTINUOUS, name=f"ln_A_{req_name}")
#             # Exact range under the proportional floor: ln_S in [ln(eps), ln(S_ub)].
#             ln_S_vars[constrained_request] = model.addVar(
#                 lb=ln_eps, ub=math.log(S_ub[r_ub]) if S_ub[r_ub] < 1.0 else 0.0,
#                 vtype=GRB.CONTINUOUS, name=f"ln_S_{req_name}")
#         if needs_log or kind == 'merge':
#             ln_A_parents_vars[constrained_request] = model.addVar(
#                 lb=lb_ln_parents, ub=math.log(A_par_ub[r_ub]) if A_par_ub[r_ub] < 1.0 else 0.0,
#                 vtype=GRB.CONTINUOUS, name=f"ln_A_parents_{req_name}")

#         # --- Vertical entry boundary (paper Sec 5.3) ---------------------------
#         direct_parents = [p for p in workflow_graph.predecessors(constrained_request)
#                           if p in solution_holder]

#         if kind == 'root':
#             model.addConstr(ancestor_success_prob[constrained_request] == 1.0,
#                             name=f"root_Ap_{req_name}")
#             if needs_log or kind == 'merge':
#                 model.addConstr(ln_A_parents_vars[constrained_request] == 0.0,
#                                 name=f"root_lnAp_{req_name}")
#         elif kind == 'chain':
#             # Single-parent shortcut: entry boundary is linear in parent's vars.
#             q = direct_parents[0]
#             model.addConstr(ancestor_success_prob[constrained_request] == A_prot_vars[q],
#                             name=f"chain_Ap_{req_name}")
#             if needs_log:
#                 # parent q is in anc(merge) whenever r is, so ln_A_vars[q] exists
#                 model.addConstr(ln_A_parents_vars[constrained_request] == ln_A_vars[q],
#                                 name=f"chain_lnAp_{req_name}")
#         else:
#             # General merge node: exact log-space join over the unique closure set.
#             model.addConstr(
#                 ln_A_parents_vars[constrained_request]
#                 == gp.quicksum(ln_S_vars[anc] for anc in anc_sets[constrained_request]),
#                 name=f"join_lnAp_{req_name}")
#             model.addGenConstrExp(ln_A_parents_vars[constrained_request],
#                                   ancestor_success_prob[constrained_request],
#                                   name=f"exp_Ap_{req_name}")
#             # Real-space valid cuts tightening the exp() relaxation:
#             # A_parents[r] <= A_prot[a] for every in-model ancestor a.
#             for anc in anc_sets[constrained_request]:
#                 model.addConstr(
#                     ancestor_success_prob[constrained_request] <= A_prot_vars[anc],
#                     name=f"cut_Ap_le_Aprot_{req_name}_{getattr(getattr(anc, 'observation_request', anc), 'name', id(anc))}")

#         # --- Horizontal scaled timeline (paper Sec 5.2) ------------------------
#         passes = task_to_passes[constrained_request]
#         K_r = len(passes)

#         for k in range(K_r + 1):
#             scaled_remaining_risk[(constrained_request, k)] = model.addVar(
#                 lb=0.0, ub=A_par_ub[constrained_request], vtype=GRB.CONTINUOUS,
#                 name=f"Y_{req_name}_k{k}")

#         # Injection identity: timeline starts at the parents' joint success.
#         model.addConstr(
#             scaled_remaining_risk[(constrained_request, 0)] == ancestor_success_prob[constrained_request],
#             name=f"inject_{req_name}")

#         for k, (satellite, satpass) in enumerate(passes):
#             x_var = solution_holder[constrained_request][satellite][satpass]['x']
#             theta_k = solution_holder[constrained_request][satellite][satpass]['theta']
#             Y_current = scaled_remaining_risk[(constrained_request, k)]

#             w_abs = model.addVar(lb=0.0, ub=A_par_ub[constrained_request],
#                                  vtype=GRB.CONTINUOUS, name=f"w_abs_{req_name}_k{k}")
#             effective_pass_realization[(constrained_request, satellite, satpass)] = w_abs
#             x_var.BranchPriority = 10  # branch schedule decisions before spatial branching

#             # Exact McCormick linearization of the binary-continuous product.
#             model.addConstr(w_abs <= x_var, name=f"mc1_{req_name}_k{k}")
#             model.addConstr(w_abs <= Y_current, name=f"mc2_{req_name}_k{k}")
#             model.addConstr(w_abs >= Y_current - (1.0 - x_var), name=f"mc3_{req_name}_k{k}")

#             model.addConstr(
#                 scaled_remaining_risk[(constrained_request, k + 1)] == Y_current - theta_k * w_abs,
#                 name=f"rec_{req_name}_k{k}")

#         # End-of-horizon fulfillment (paper Eq. 29).
#         model.addConstr(
#             end_to_end_success[constrained_request]
#             == ancestor_success_prob[constrained_request] - scaled_remaining_risk[(constrained_request, K_r)],
#             name=f"e2e_{req_name}")

#         # Union-bound cut (docstring item 6): reward-carrying probability mass
#         # is capped by the scheduled probability mass, scaled by the best-case
#         # ancestral survival. Valid since 1 - prod(1-p*x) <= sum(p*x).
#         if K_r > 0:
#             model.addConstr(
#                 end_to_end_success[constrained_request]
#                 <= A_par_ub[constrained_request] * gp.quicksum(
#                     solution_holder[constrained_request][sat][sp]['theta']
#                     * solution_holder[constrained_request][sat][sp]['x']
#                     for (sat, sp) in passes),
#                 name=f"cut_union_{req_name}")

#         # DRAIN CUT (the decisive one). In the LP relaxation, fractional x lets
#         # the McCormick track fully drain Y (claim near-certain local success)
#         # while "paying" only max_instances worth of booking mass -- this, not
#         # the exp/log chords, is what inflates the root bound by hundreds of
#         # percent when requests have many candidate passes. No INTEGER solution
#         # can exceed the top-M union probability, so:
#         #     e2e[r] <= lmax_M(r) * A_parents[r]
#         # is valid, linear, and caps the relaxation at the true per-task ceiling.
#         if tighten_bounds and K_r > 0 and lmax_ub[constrained_request] < 1.0:
#             model.addConstr(
#                 end_to_end_success[constrained_request]
#                 <= lmax_ub[constrained_request] * ancestor_success_prob[constrained_request],
#                 name=f"cut_drain_{req_name}")

#         # REWARD ENVELOPE CUTS (cardinality-priced reward). The LP relaxation
#         # can otherwise earn near-union success probability while paying only
#         # a fraction of the integer booking count (McCormick complementarity
#         # binds only at integral x), so per-booking costs barely discount the
#         # bound. The per-task reward with n integer bookings is bounded by the
#         # CONCAVE curve R_ub(n) = A_par_ub * Q_max * lmax(n), where
#         # lmax(n) = 1 - prod over top-n p of (1-p). We add its tangents at
#         # n = 0..M-1: valid for every integer point by concavity, and they
#         # force the relaxation to pay one full booking of cost per top-marginal
#         # unit of reward. This encodes the diminishing-returns structure --
#         # invisible to the plain relaxation -- as linear inequalities, with no
#         # change to the feasible integer set or the objective.
#         if reward_envelope_cuts and tighten_bounds and K_r > 0:
#             _thetas_desc = sorted(
#                 (solution_holder[constrained_request][sat][sp]['theta'] for (sat, sp) in passes),
#                 reverse=True)
#             _Qmax = max(solution_holder[constrained_request][sat][sp]['quality']
#                         for (sat, sp) in passes)
#             _scale = A_par_ub[constrained_request] * _Qmax
#             _lmax_curve = [0.0]
#             _fail = 1.0
#             for _p in _thetas_desc[:M_of[constrained_request]]:
#                 _fail *= (1.0 - _p)
#                 _lmax_curve.append(1.0 - _fail)
#             _reward_expr = gp.quicksum(
#                 solution_holder[constrained_request][sat][sp]['quality']
#                 * solution_holder[constrained_request][sat][sp]['theta']
#                 * effective_pass_realization[(constrained_request, sat, sp)]
#                 for (sat, sp) in passes)
#             _xsum = gp.quicksum(
#                 solution_holder[constrained_request][sat][sp]['x'] for (sat, sp) in passes)
#             for _n in range(len(_lmax_curve) - 1):
#                 _slope = _lmax_curve[_n + 1] - _lmax_curve[_n]
#                 model.addConstr(
#                     _reward_expr <= _scale * (_lmax_curve[_n] - _slope * _n)
#                                     + _scale * _slope * _xsum,
#                     name=f"cut_renv_{req_name}_n{_n}")
#                 # SURVIVAL envelope: the same concave cardinality pricing must
#                 # also cap e2e itself, or the LP fractionally drains Y to the
#                 # union level "for free" and hands inflated survival to every
#                 # descendant (the recurrence Y_0[child] = A_prot[parent] then
#                 # compounds the inflation down the DAG). With these cuts the
#                 # survival passed downstream is priced per integer booking,
#                 # and the chain recurrence compounds cost-consistent values.
#                 model.addConstr(
#                     end_to_end_success[constrained_request]
#                     <= A_par_ub[constrained_request] * (_lmax_curve[_n] - _slope * _n)
#                        + A_par_ub[constrained_request] * _slope * _xsum,
#                     name=f"cut_senv_{req_name}_n{_n}")

#             # QUALITY-PROBABILITY frontier envelope: an integer solution picks
#             # ONE subset of passes; its credited reward is at most the sum of
#             # its Q*p products (dropping failure discounting), hence at most
#             # the sum of the n LARGEST Q*p products for n bookings -- a concave
#             # curve in n. The fractional LP otherwise splits booking mass to
#             # take survival from high-p passes and reward from high-Q slots at
#             # the same time, exceeding every integer subset on both axes.
#             _qp_desc = sorted(
#                 (solution_holder[constrained_request][sat][sp]['quality']
#                  * solution_holder[constrained_request][sat][sp]['theta']
#                  for (sat, sp) in passes), reverse=True)[:M_of[constrained_request]]
#             _qp_cum = [0.0]
#             for _qp in _qp_desc:
#                 _qp_cum.append(_qp_cum[-1] + _qp)
#             for _n in range(len(_qp_cum) - 1):
#                 _slope_qp = _qp_cum[_n + 1] - _qp_cum[_n]
#                 model.addConstr(
#                     _reward_expr <= A_par_ub[constrained_request]
#                                     * ((_qp_cum[_n] - _slope_qp * _n) + _slope_qp * _xsum),
#                     name=f"cut_qpenv_{req_name}_n{_n}")

#         # --- PROPORTIONAL floor + protected log (fix of the floor trap) --------
#         A_prot = model.addVar(lb=max(math.exp(lb_ln_A), 1e-30),
#                               ub=A_prot_ub[constrained_request],
#                               vtype=GRB.CONTINUOUS, name=f"A_prot_{req_name}")
#         A_prot_vars[constrained_request] = A_prot
#         model.addConstr(
#             A_prot == end_to_end_success[constrained_request] * (1.0 - epsilon)
#                       + epsilon * ancestor_success_prob[constrained_request],
#             name=f"prot_{req_name}")
#         # Log machinery only where ln_S(r) is actually consumed downstream.
#         if needs_log:
#             model.addGenConstrLog(A_prot, ln_A_vars[constrained_request], name=f"log_{req_name}")
#             # Log deduction identity (paper Eq. 32), now always satisfiable.
#             model.addConstr(
#                 ln_S_vars[constrained_request]
#                 == ln_A_vars[constrained_request] - ln_A_parents_vars[constrained_request],
#                 name=f"lnS_{req_name}")

#     # === OBJECTIVE =============================================================
#     # Maximize sum_k Q_k * theta_k * W_abs_k
#     #         - sum_k (c_sub_k + c_exec_k * theta_acc_k + c_tax_k) * x_k
#     #
#     # Costs use Q_MAX_task (max quality over all passes for the task) for
#     # provider-rate billing, decoupled from per-pass geometry. Matches
#     # realized metrics billing in compute_metrics_v3.
#     #
#     # Execution cost uses theta_acc (= p_acc), NOT theta (= p_acc * p_exec):
#     # the simulator bills execution cost for both DATA_RECEIVED and
#     # EXECUTION_FAILED — both require acceptance, so expected cost = p_acc.
#     objective_terms = []
#     for constrained_request in topo:
#         if not solution_holder[constrained_request]:
#             continue

#         _q_max_task = max(
#             solution_holder[constrained_request][s][p]['quality']
#             for s, p in task_to_passes[constrained_request]
#         )
#         _has_success_parent = any(
#             'SUCCESS' in str(e.get('constraint_class', ''))
#             for p in workflow_graph.predecessors(constrained_request)
#             for e in workflow_graph.get_edge_data(p, constrained_request).values()
#         )
#         if _has_success_parent:
#             _parent_passes = [sp.highest.time for p in workflow_graph.predecessors(constrained_request) for sat, sp in task_to_passes.get(p, [])]
#             _t_dispatch = min(_parent_passes) if _parent_passes else current_time
#         else:
#             _t_dispatch = current_time

#         for satellite, satpass in task_to_passes[constrained_request]:
#             x_var = solution_holder[constrained_request][satellite][satpass]['x']
#             quality = solution_holder[constrained_request][satellite][satpass]['quality']
#             theta = solution_holder[constrained_request][satellite][satpass]['theta']
#             theta_acc = solution_holder[constrained_request][satellite][satpass]['theta_acc']
#             w_abs = effective_pass_realization[(constrained_request, satellite, satpass)]

#             c_sub = submission_cost_rate * _q_max_task
#             if execution_cost_fn is not None:
#                 try:
#                     c_exec_k = execution_cost_fn(constrained_request, satellite, satpass, _t_dispatch, q_max=_q_max_task)
#                 except Exception:
#                     c_exec_k = cancellation_cost_rate * _q_max_task
#             else:
#                 c_exec_k = cancellation_cost_rate * _q_max_task
#             c_tax = tax_rate * _q_max_task

#             # Expected best-success quality credit for this pass.
#             objective_terms.append(quality * theta * w_abs)
#             # Unconditional submission overhead (paid regardless of acceptance).
#             objective_terms.append(-c_sub * x_var)
#             # Execution cost, conditional on acceptance (theta_acc = p_acc).
#             objective_terms.append(-c_exec_k * theta_acc * x_var)
#             if c_tax:
#                 objective_terms.append(-c_tax * x_var)

#     model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)


def _add_workflow_constraints(
        model: gp.Model,
        workflow_graph: nx.MultiDiGraph,
        solution_holder: dict,
        verbose: int,
        default_max_instances: int = 3,
        committed_bookings=None,
        slew_margin: dt.timedelta = dt.timedelta(0)
):
    """
    Add instance caps, mandatory coverage, satellite conflicts, temporal
    relations, committed-booking exclusion, and branch exclusivity.

    Constraints:
    - At most max_num_instances per task (defaulting to `default_max_instances`
      when the request does not specify one -- MUST be > 1 for the stochastic
      planner to book redundant passes, which is the entire mechanism the
      stochastic formulation exists to price)
    - Mandatory tasks book at least one pass
    - No overlapping observations on the same satellite
    - Committed bookings (already dispatched) block colliding variables
    - Temporal relations, EXISTENTIALLY (see below)
    - BRANCH EXCLUSIVITY between the imaging and search arms of a window
    """

    from fame_workflow import ConstraintClass, TemporalConstraintType, SuccessConstraintType
    import re

    committed_bookings = committed_bookings or []

    # === MAX INSTANCES CONSTRAINT ===
    for constrained_request in solution_holder.keys():
        x_vars = [solution_holder[constrained_request][sat][sp]['x']
                  for sat in solution_holder[constrained_request]
                  for sp in solution_holder[constrained_request][sat]]
        if len(x_vars) > 0:
            max_instances = _effective_max_instances(constrained_request, default_max_instances)
            model.addConstr(
                gp.quicksum(x_vars) <= max_instances,
                name=f"max_instances_{constrained_request.observation_request.name}"
            )

    # === MANDATORY TASK CONSTRAINT ===
    for constrained_request in solution_holder.keys():
        if constrained_request.is_mandatory:
            x_vars = [solution_holder[constrained_request][sat][sp]['x']
                      for sat in solution_holder[constrained_request]
                      for sp in solution_holder[constrained_request][sat]]
            if len(x_vars) > 0:
                model.addConstr(
                    gp.quicksum(x_vars) >= 1,
                    name=f"mandatory_{constrained_request.observation_request.name}"
                )

    # === SATELLITE CONFLICT CONSTRAINTS ===
    solution_holder_by_satellite = {}
    for constrained_request in solution_holder.keys():
        for satellite in solution_holder[constrained_request].keys():
            solution_holder_by_satellite.setdefault(satellite, [])
            for satpass in solution_holder[constrained_request][satellite].keys():
                solution_holder_by_satellite[satellite].append((
                    satpass,
                    solution_holder[constrained_request][satellite][satpass]['x'],
                    constrained_request
                ))

    for satellite in solution_holder_by_satellite.keys():
        passes = solution_holder_by_satellite[satellite]
        passes.sort(key=lambda x: x[0].highest.time)

        # Max end time seen up to index i, so the inner loop breaks correctly
        # even when passes from different tasks have non-monotonic end times.
        max_end_so_far = {}
        running_max = passes[0][0].highest.time + passes[0][0].highest.duration if passes else None
        for i, (p, _, _r) in enumerate(passes):
            running_max = max(running_max, p.highest.time + p.highest.duration)
            max_end_so_far[i] = running_max

        for i in range(len(passes)):
            pass_i, x_i, req_i = passes[i]
            for j in range(i + 1, len(passes)):
                pass_j, x_j, req_j = passes[j]
                if pass_j.highest.time >= max_end_so_far[i] + slew_margin:
                    break
                end_i = pass_i.highest.time + pass_i.highest.duration
                start_j = pass_j.highest.time
                if start_j < end_i + slew_margin:
                    model.addConstr(x_i + x_j <= 1,
                                    name=f"conflict_{satellite.name}_{i}_{j}")

    # === COMMITTED-BOOKING EXCLUSION ===
    # Variables colliding with an already-dispatched booking on the same
    # satellite are fixed to zero. ub=0 rather than deletion keeps
    # solution_holder / task_to_passes structurally intact for _extract_solution
    # and the MIP starts; presolve removes them anyway. Without this the model
    # assigns expected reward to passes that can never be flown, so ObjVal stops
    # being an upper bound on anything realizable.
    if committed_bookings:
        _n_blocked = 0
        for constrained_request in solution_holder.keys():
            for satellite in solution_holder[constrained_request].keys():
                for satpass, entry in solution_holder[constrained_request][satellite].items():
                    s = satpass.highest.time
                    e = s + satpass.highest.duration
                    for (c_sat, c_s, c_e) in committed_bookings:
                        if c_sat is not satellite:
                            continue
                        if s < c_e + slew_margin and c_s < e + slew_margin:
                            entry['x'].ub = 0.0
                            _n_blocked += 1
                            break
        if verbose > 0 and _n_blocked:
            print(f"  [Constraints] Blocked {_n_blocked} pass variable(s) colliding with "
                  f"{len(committed_bookings)} committed booking(s)")

    # === TEMPORAL CONSTRAINTS (EXISTENTIAL) ===
    # The old pairwise form (x_child + x_parent <= 1 for each violating pair)
    # required EVERY selected parent pass to satisfy the offset, so with K
    # redundant parent passes the child's admissible window shrank to
    # [max_p t_p + h, min_p t_p + h + delta] -- the offset window MINUS the
    # parent spread -- and could go empty. That charges redundancy a downstream
    # feasibility cost the objective never sees, and only the stochastic planner
    # books redundantly, so only it pays.
    #
    # The child needs ONE compatible parent, not all of them:
    #     x_child <= sum over temporally-compatible parent passes of x_parent
    #
    # All relations on an edge are tested JOINTLY -- a parent satisfying
    # START_AFTER while a different parent satisfies START_BEFORE is not a
    # feasible anchor.
    #
    # This is the planning-time relaxation; the exact anchor is resolved after
    # the fact by the GEOMETRY re-anchor collapsing the child window onto the
    # realised parent time, followed by a replan.
    for constrained_request in solution_holder.keys():
        for parent_request in workflow_graph.predecessors(constrained_request):

            inedges = workflow_graph.get_edge_data(parent_request, constrained_request) or {}
            temporal = []
            for _key, constraint in inedges.items():
                if constraint.get('constraint_class') != ConstraintClass.TEMPORAL:
                    continue
                temporal.append((
                    constraint['constraint_type'],
                    (constraint.get('parameters') or {}).get('offset', dt.timedelta(0)),
                ))
            if not temporal:
                continue

            def _compatible(t_parent, t_child, _temporal=temporal):
                for ctype, off in _temporal:
                    if ctype == TemporalConstraintType.START_AFTER and t_parent > t_child:
                        return False
                    if ctype == TemporalConstraintType.START_AFTER_OFFSET and t_parent + off > t_child:
                        return False
                    if ctype == TemporalConstraintType.START_BEFORE and t_parent < t_child:
                        return False
                    if ctype == TemporalConstraintType.START_BEFORE_OFFSET and t_parent + off < t_child:
                        return False
                return True

            # --- Parent already dispatched: its pass time is REALISED, so this
            # is a constant test rather than a pairwise one. Previously the whole
            # edge was skipped here, which left follow-up timing completely
            # unconstrained on every solve after the first.
            if parent_request not in solution_holder:
                _oo = getattr(parent_request, 'observation_opportunity', None)
                _t_parent = getattr(_oo, 'time', None)
                if _t_parent is None:
                    _op = getattr(parent_request, 'observation_opportunity_pass', None)
                    _t_parent = getattr(getattr(_op, 'highest', None), 'time', None)
                if _t_parent is None:
                    if verbose > 2:
                        print(f"  [Constraints] {parent_request.observation_request.name} -> "
                              f"{constrained_request.observation_request.name}: dispatched parent "
                              f"with no realised time; edge left unconstrained")
                    continue
                _n_fixed = 0
                for _sat in solution_holder[constrained_request]:
                    for _sp, _entry in solution_holder[constrained_request][_sat].items():
                        if not _compatible(_t_parent, _sp.highest.time):
                            _entry['x'].ub = 0.0
                            _n_fixed += 1
                if verbose > 2 and _n_fixed:
                    print(f"  [Constraints] {constrained_request.observation_request.name}: "
                          f"{_n_fixed} pass(es) excluded by realised parent time {_t_parent}")
                continue

            # --- Parent still schedulable: existential over its candidate passes.
            _n_blocked = 0
            for child_sat in solution_holder[constrained_request]:
                for child_pass, child_entry in solution_holder[constrained_request][child_sat].items():
                    x_child = child_entry['x']
                    t_child = child_pass.highest.time
                    compatible_parents = [
                        solution_holder[parent_request][psat][ppass]['x']
                        for psat in solution_holder[parent_request]
                        for ppass in solution_holder[parent_request][psat]
                        if _compatible(ppass.highest.time, t_child)
                    ]
                    if not compatible_parents:
                        x_child.ub = 0.0
                        _n_blocked += 1
                    else:
                        model.addConstr(
                            x_child <= gp.quicksum(compatible_parents),
                            name=f"temporal_{constrained_request.observation_request.name}"
                                 f"_{child_sat.name}_{t_child:%H%M%S}"
                        )
            if verbose > 2 and _n_blocked:
                print(f"  [Constraints] {constrained_request.observation_request.name}: "
                      f"{_n_blocked} pass(es) have no compatible parent pass")

    # === SUCCESS PRECEDENCE (ZERO DELAY) ===
    # A SUCCESS edge also means that the child cannot start before the parent
    # observation has ended.  Keep this separate from the TEMPORAL block above:
    # TEMPORAL edges define campaign windows, while SUCCESS edges identify the
    # immediate operational predecessor.
    #
    # For the default "all" mode, every SUCCESS parent supplies one compatible
    # predecessor.  For "any" (e.g. earthquake TRIAGE), one compatible pass from
    # any SUCCESS parent is sufficient, preserving the optical-OR-SAR semantics.
    for constrained_request in solution_holder.keys():
        if not getattr(constrained_request, 'enforce_success_precedence', False):
            continue
        success_parents = []
        for parent_request in workflow_graph.predecessors(constrained_request):
            inedges = workflow_graph.get_edge_data(parent_request, constrained_request) or {}
            if any(c.get('constraint_class') == ConstraintClass.SUCCESS
                   for c in inedges.values()):
                success_parents.append(parent_request)

        if not success_parents:
            continue

        success_mode = getattr(constrained_request, 'success_constraint_mode', 'all')

        def _compatible_success_passes(parent_request, t_child):
            """Decision variables for parent passes ending no later than child."""
            if parent_request not in solution_holder:
                return []
            return [
                solution_holder[parent_request][psat][ppass]['x']
                for psat in solution_holder[parent_request]
                for ppass in solution_holder[parent_request][psat]
                if ppass.highest.time + ppass.highest.duration <= t_child
            ]

        def _fixed_parent_precedes(parent_request, t_child):
            """Whether an already-dispatched/completed parent can precede child."""
            if parent_request in solution_holder:
                return False
            if (getattr(parent_request, 'completed', False)
                    and not getattr(parent_request, 'successful_execution', False)):
                return False

            bookings = getattr(parent_request, 'scheduled_bookings', None) or []
            fixed_passes = [b.get('pass') for b in bookings if b.get('pass') is not None]
            if not fixed_passes:
                fixed_pass = getattr(parent_request, 'observation_opportunity_pass', None)
                if fixed_pass is not None:
                    fixed_passes = [fixed_pass]

            return any(
                p.highest.time + p.highest.duration <= t_child
                for p in fixed_passes
            )

        for child_sat in solution_holder[constrained_request]:
            for child_pass, child_entry in solution_holder[constrained_request][child_sat].items():
                x_child = child_entry['x']
                t_child = child_pass.highest.time

                if success_mode == 'any':
                    if any(_fixed_parent_precedes(p, t_child) for p in success_parents):
                        continue
                    compatible = [
                        x_parent
                        for parent_request in success_parents
                        for x_parent in _compatible_success_passes(parent_request, t_child)
                    ]
                    if compatible:
                        model.addConstr(
                            x_child <= gp.quicksum(compatible),
                            name=f"success_precedence_any_"
                                 f"{constrained_request.observation_request.name}_"
                                 f"{child_sat.name}_{t_child:%H%M%S}"
                        )
                    else:
                        x_child.ub = 0.0
                else:
                    for parent_request in success_parents:
                        if _fixed_parent_precedes(parent_request, t_child):
                            continue
                        compatible = _compatible_success_passes(parent_request, t_child)
                        if compatible:
                            model.addConstr(
                                x_child <= gp.quicksum(compatible),
                                name=f"success_precedence_all_"
                                     f"{constrained_request.observation_request.name}_"
                                     f"{parent_request.observation_request.name}_"
                                     f"{child_sat.name}_{t_child:%H%M%S}"
                            )
                        else:
                            x_child.ub = 0.0
                            break
    # === NO BRANCH EXCLUSIVITY UNDER general_logical_dag ===
    # The gates already encode the branch: imaging is gated on Lit(K_w), search
    # on Not(Lit(K_w)), so the DSOP compilation zeroes whichever arm the belief
    # state rules out. Forcing sum(x_img) <= M*z and sum(x_srch) <= M*(1-z) on
    # top of that ALSO forbids booking a search while tracked -- which is exactly
    # the option-value play the general formulation exists to price. A successful
    # search restores K, unlocking imaging in every later window, and that is
    # worth an order of magnitude more than a search's direct reward.
    # _by_window = {}
    # for _req in solution_holder:
    #     _nm = getattr(getattr(_req, 'observation_request', _req), 'name', '')
    #     _m = re.search(r'Follow-up (imaging|search) (\d+)', _nm)
    #     if _m:
    #         _by_window.setdefault(int(_m.group(2)),
    #                               {'imaging': [], 'search': []})[_m.group(1)].append(_req)

    # _n_branch = 0
    # for _w in sorted(_by_window):
    #     _arms = _by_window[_w]
    #     _img_x = [solution_holder[r][s][p]['x'] for r in _arms['imaging']
    #               for s in solution_holder[r] for p in solution_holder[r][s]]
    #     _srch_x = [solution_holder[r][s][p]['x'] for r in _arms['search']
    #                for s in solution_holder[r] for p in solution_holder[r][s]]
    #     if not _img_x or not _srch_x:
    #         continue   # only one arm is live this solve -- nothing to exclude
    #     _z = model.addVar(vtype=GRB.BINARY, name=f"branch_w{_w}")
    #     model.addConstr(gp.quicksum(_img_x) <= len(_img_x) * _z,
    #                     name=f"branch_img_w{_w}")
    #     model.addConstr(gp.quicksum(_srch_x) <= len(_srch_x) * (1 - _z),
    #                     name=f"branch_srch_w{_w}")
    #     _n_branch += 1
    # if verbose > 0 and _n_branch:
    #     print(f"  [Constraints] Branch exclusivity on {_n_branch} window(s) "
    #           f"(imaging XOR search)")


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
