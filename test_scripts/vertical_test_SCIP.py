import os
import time
import math
import gurobipy as gp
from gurobipy import GRB
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv
import networkx as nx
import numpy as np

# Load parameters from local .env file
load_dotenv()

def add_custom_pwl_constraint(solver, x_var, y_var, func, lb, ub, num_segments=16, spacing="linear", name_prefix=""):
    """
    Adds a 100% solver-independent Piecewise-Linear constraint (y = func(x)) 
    using the Lambda/Convex-Combination method. Safe for SCIP / OR-Tools.
    """
    if spacing == "logarithmic":
        X = np.geomspace(lb, ub, num_segments + 1)
    else:
        X = np.linspace(lb, ub, num_segments + 1)
        
    Y = [func(val) for val in X]
    
    # Binary tracking variables for active intervals
    b = [solver.BoolVar(f"{name_prefix}_interval_b_{i}") for i in range(num_segments)]
    solver.Add(sum(b) == 1)
    
    # Continuous vertex coordinate weights
    lambdas = [solver.NumVar(0.0, 1.0, f"{name_prefix}_vertex_L_{j}") for j in range(num_segments + 1)]
    solver.Add(sum(lambdas) == 1)
    
    # Adjacency rules (Forces only the vertices bounding the active interval to be non-zero)
    solver.Add(lambdas[0] <= b[0])
    for i in range(1, num_segments):
        solver.Add(lambdas[i] <= b[i-1] + b[i])
    solver.Add(lambdas[num_segments] <= b[num_segments-1])
    
    # Link input and output values to coordinates
    solver.Add(x_var == sum(lambdas[j] * X[j] for j in range(num_segments + 1)))
    solver.Add(y_var == sum(lambdas[j] * Y[j] for j in range(num_segments + 1)))

def run_cross_solver_benchmark(num_tasks=6, passes_per_task=4, gamma=0.15):
    print(f"=======================================================")
    print(f"--- DUAL-SOLVER CASCADING BENCHMARK (Depth = {num_tasks}) ---")
    print(f"=======================================================")
    print(f"-> Model 1: NATIVE GUROBI (Raw Exact Non-Convex Quadratic)")
    print(f"-> Model 2: OR-TOOLS SCIP (Streamlined Log-Linearized MILP)")
    print(f"-------------------------------------------------------")
    
    epsilon = 10 ** (-4)
    
    # --- Step 1: Generate Data Structures ---
    tasks = [f"task_{i}" for i in range(num_tasks)]
    roots = [tasks[0]]
    edges = [(tasks[i], tasks[i+1]) for i in range(num_tasks - 1)]
    
    dag = nx.DiGraph()
    dag.add_nodes_from(tasks)
    dag.add_edges_from(edges)
    
    task_data = {}
    for i, r in enumerate(tasks):
        raw_passes = list(range(passes_per_task))
        qualities = {k: float(110.0 - i * 4.0 - k * 5.0) for k in raw_passes}
        probabilities = {k: round(0.70 + 0.04 * ((k + i) % 4), 2) for k in raw_passes}
        task_data[r] = {
            'passes': sorted(raw_passes, key=lambda k: qualities[k], reverse=True),
            'qualities': qualities,
            'probabilities': probabilities
        }

    dynamic_taxes = {r: gamma * max(task_data[r]['qualities'].values()) for r in tasks}

    obj_m1, obj_m2 = None, None
    runtime_m1, runtime_m2 = 0.0, 0.0

    # =====================================================================
    # SOLVER 1: NATIVE GUROBI (RAW NON-CONVEX REFERENCE TRUTH)
    # =====================================================================
    print("\n⚡ Initializing Model 1: Native Gurobi [NonConvex=2]...")
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    gurobi_license_id = os.getenv("LICENSEID")

    gurobi_env = gp.Env(params={
        "WLSACCESSID": wls_access_id, "WLSSECRET": wls_secret,
        "LICENSEID": int(gurobi_license_id) if gurobi_license_id else 0, "OutputFlag": 0
    }) if wls_access_id else gp.Env()
    gurobi_env.setParam("OutputFlag", 0)

    x_idx = [(r, k) for r in tasks for k in task_data[r]['passes']]
    y_idx = [(r, k) for r in tasks for k in range(len(task_data[r]['passes']) + 1)]

    with gp.Model("Gurobi_NonConvex", env=gurobi_env) as m1:
        m1.setParam('NonConvex', 2)
        
        pass_scheduled_1 = m1.addVars(x_idx, vtype=GRB.BINARY, name="x1")
        task_remaining_risk_1 = m1.addVars(y_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y1")
        active_local_risk_1 = m1.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w1")
        
        S_vars_1 = m1.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="S1")
        A_parents_1 = m1.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="A_parents1")
        w_abs_vars_1 = m1.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w_abs1")
        
        for r in tasks:
            m1.addConstr(task_remaining_risk_1[r, 0] == 1.0)
            for k in task_data[r]['passes']:
                m1.addConstr(active_local_risk_1[r, k] <= pass_scheduled_1[r, k])
                m1.addConstr(active_local_risk_1[r, k] <= task_remaining_risk_1[r, k])
                m1.addConstr(active_local_risk_1[r, k] >= task_remaining_risk_1[r, k] - (1.0 - pass_scheduled_1[r, k]))
                m1.addConstr(active_local_risk_1[r, k] >= 0.0)
                m1.addConstr(task_remaining_risk_1[r, k+1] == task_remaining_risk_1[r, k] - task_data[r]['probabilities'][k] * active_local_risk_1[r, k])
            
            m1.addConstr(S_vars_1[r] == 1.0 - task_remaining_risk_1[r, len(task_data[r]['passes'])])
        
        for r in tasks:
            unique_ancestors = list(nx.ancestors(dag, r))
            if not unique_ancestors:
                m1.addConstr(A_parents_1[r] == 1.0)
            else:
                current_prod = S_vars_1[unique_ancestors[0]]
                for idx in range(1, len(unique_ancestors)):
                    inter_var = m1.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"inter_{r}_{idx}")
                    m1.addQConstr(inter_var == current_prod * S_vars_1[unique_ancestors[idx]])
                    current_prod = inter_var
                m1.addConstr(A_parents_1[r] == current_prod)
        
        for r in tasks:
            for k in task_data[r]['passes']:
                m1.addQConstr(w_abs_vars_1[r, k] == A_parents_1[r] * active_local_risk_1[r, k])
        
        rev_m1 = gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * w_abs_vars_1[r, k] for r in tasks for k in task_data[r]['passes'])
        pen_m1 = gp.quicksum(dynamic_taxes[r] * pass_scheduled_1[r, k] for r in tasks for k in task_data[r]['passes'])
        m1.setObjective(rev_m1 - pen_m1, GRB.MAXIMIZE)
        
        t1_start = time.perf_counter()
        m1.optimize()
        runtime_m1 = time.perf_counter() - t1_start
        if m1.SolCount > 0:
            obj_m1 = m1.ObjVal

    # =====================================================================
    # SOLVER 2: OR-TOOLS SCIP (LOG-LINEARIZED MILP)
    # =====================================================================
    print("🚀 Initializing Model 2: OR-Tools SCIP Engine...")
    scip_solver = pywraplp.Solver.CreateSolver('SCIP')
    if not scip_solver:
        print("SCIP driver missing from installation target. Check OR-Tools configuration.")
        return

    pass_scheduled_2 = {}
    effective_pass_realization_2 = {}
    scaled_remaining_risk = {}
    
    ancestor_success_prob = {}
    end_to_end_success_2 = {}
    
    ln_A = {}
    ln_A_parents = {}
    ln_S = {}

    # Initialize Core SCIP continuous layers
    for r in tasks:
        ancestor_success_prob[r] = scip_solver.NumVar(0.0, 1.0, f"A_parents_{r}")
        end_to_end_success_2[r] = scip_solver.NumVar(0.0, 1.0, f"A_node_{r}")
        ln_A[r] = scip_solver.NumVar(-15.0, 0.0, f"ln_A_{r}")
        ln_A_parents[r] = scip_solver.NumVar(-15.0, 0.0, f"ln_A_parents_{r}")
        ln_S[r] = scip_solver.NumVar(-15.0, 0.0, f"ln_S_{r}")
        
        for k in task_data[r]['passes']:
            pass_scheduled_2[(r, k)] = scip_solver.BoolVar(f"x_{r}_k{k}")
            effective_pass_realization_2[(r, k)] = scip_solver.NumVar(0.0, 1.0, f"w_abs_{r}_k{k}")
        for k in range(len(task_data[r]['passes']) + 1):
            scaled_remaining_risk[(r, k)] = scip_solver.NumVar(0.0, 1.0, f"Y_{r}_k{k}")

    # Vertical Lineage Map (SCIP Addition)
    for r in tasks:
        unique_ancestors = list(nx.ancestors(dag, r))
        if not unique_ancestors:
            scip_solver.Add(ln_A_parents[r] == 0.0)
            scip_solver.Add(ancestor_success_prob[r] == 1.0)
        else:
            scip_solver.Add(ln_A_parents[r] == sum(ln_S[anc] for anc in unique_ancestors))
            add_custom_pwl_constraint(scip_solver, ln_A_parents[r], ancestor_success_prob[r], math.exp, -15.0, 0.0, num_segments=32, name_prefix=f"exp_{r}")

    # Single-Track Horizon Math (SCIP Polytope)
    for r in tasks:
        K_r = len(task_data[r]['passes'])
        scip_solver.Add(scaled_remaining_risk[(r, 0)] == ancestor_success_prob[r])
        
        for k in task_data[r]['passes']:
            w_abs = effective_pass_realization_2[(r, k)]
            x_var = pass_scheduled_2[(r, k)]
            Y_var = scaled_remaining_risk[(r, k)]
            theta = task_data[r]['probabilities'][k]
            
            # Linear McCormick boundaries mapping binary tracking
            scip_solver.Add(w_abs <= x_var)
            scip_solver.Add(w_abs <= Y_var)
            scip_solver.Add(w_abs >= Y_var - (1.0 - x_var))
            scip_solver.Add(w_abs >= 0.0)
            scip_solver.Add(scaled_remaining_risk[(r, k+1)] == Y_var - theta * w_abs)
            
        scip_solver.Add(end_to_end_success_2[r] == ancestor_success_prob[r] - scaled_remaining_risk[(r, K_r)])
        
        A_prot = scip_solver.NumVar(epsilon, 1.0, f"A_prot_{r}")
        scip_solver.Add(A_prot == end_to_end_success_2[r] * (1.0 - epsilon) + epsilon)
        
        # Segmented log mapping via geometric spacing for clean slope resolution near zero bounds
        add_custom_pwl_constraint(scip_solver, A_prot, ln_A[r], math.log, epsilon, 1.0, num_segments=48, spacing="logarithmic", name_prefix=f"log_{r}")
        scip_solver.Add(ln_S[r] == ln_A[r] - ln_A_parents[r])

    rev_m2 = sum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * effective_pass_realization_2[(r, k)] for r in tasks for k in task_data[r]['passes'])
    pen_m2 = sum(dynamic_taxes[r] * pass_scheduled_2[(r, k)] for r in tasks for k in task_data[r]['passes'])
    scip_solver.Maximize(rev_m2 - pen_m2)
    
    t2_start = time.perf_counter()
    scip_status = scip_solver.Solve()
    runtime_m2 = time.perf_counter() - t2_start
    
    if scip_status == pywraplp.Solver.OPTIMAL:
        obj_m2 = scip_solver.Objective().Value()

    # --- Cross-Platform Metrics Matrix ---
    print("\n⏱️  ================ CROSS-SOLVER PERFORMANCE BENCHMARK ================")
    if obj_m1 is not None and obj_m2 is not None:
        print(f"Model 1 (Gurobi Non-Convex Quadratic) Objective: {obj_m1:.6f}")
        print(f"Model 2 (OR-Tools SCIP Log-Linearized) Objective: {obj_m2:.6f}")
        print(f"Absolute Precision Mathematical Discrepancy:      {abs(obj_m1 - obj_m2):.5e}")
        print("----------------------------------------------------------------")
        print(f"Model 1 Gurobi Internal Runtime (Exact Solver):   {runtime_m1:.6f} seconds")
        print(f"Model 2 SCIP Internal Runtime (PWL MILP Solver):  {runtime_m2:.6f} seconds")
        print("----------------------------------------------------------------")
        speedup = runtime_m1 / max(runtime_m2, 1e-6)
        print(f"📈 NET ARCHITECTURAL CHANGE SPEEDUP FACTORS:       {speedup:.2f}x Faster")
    else:
        print(f"Benchmark failure. Gurobi Status Available: {obj_m1 is not None} | SCIP Status Available: {obj_m2 is not None}")
    print("================================================================\n")

if __name__ == "__main__":
    run_cross_solver_benchmark(num_tasks=4, passes_per_task=3, gamma=0.15)