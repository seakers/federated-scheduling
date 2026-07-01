import os
import time
import gurobipy as gp
from gurobipy import GRB
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv

# Load local environment parameters
load_dotenv()

def run_gurobi_vs_scip_horizontal_test(num_tasks=15, passes_per_task=10, gamma=0.15):
    print(f"=======================================================")
    print(f"--- BENCHMARKING NATIVE GUROBI NC VS SCIP MCCORMICK ---")
    print(f"=======================================================")
    print(f"Target Scale: {num_tasks} Tasks x {passes_per_task} Passes")
    
    # --- Step 1: Generate Identical Problem Data ---
    tasks = [f"task_{i}" for i in range(num_tasks)]
    task_data = {}
    
    for i, r in enumerate(tasks):
        passes = list(range(passes_per_task))
        base_quality = 80.0 + (i % 3) * 10.0
        qualities = {k: float(base_quality - k * 1.5) for k in passes}
        probabilities = {k: round(0.40 + 0.07 * ((k + i) % 8), 2) for k in passes}
        
        task_data[r] = {
            'passes': passes,
            'qualities': qualities,
            'probabilities': probabilities
        }

    dynamic_taxes = {}
    for r in tasks:
        max_q = max(task_data[r]['qualities'].values())
        dynamic_taxes[r] = gamma * max_q

    # =====================================================================
    # TRACK 1: NATIVE GUROBI NON-CONVEX QUADRATIC
    # =====================================================================
    print("\n⚡ Solving Track 1: Gurobi Non-Convex (Smooth Curves)...")
    
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    gurobi_license_id = os.getenv("LICENSEID")

    # Use WLS context if available, fallback to default local environment if not
    if all([wls_access_id, wls_secret, gurobi_license_id]):
        wls_params = {
            "WLSACCESSID": wls_access_id,
            "WLSSECRET": wls_secret,
            "LICENSEID": int(gurobi_license_id),
            "OutputFlag": 0  
        }
        env = gp.Env(params=wls_params)
    else:
        env = gp.Env()
        env.setParam("OutputFlag", 0)

    obj_gurobi = None
    gurobi_decisions = {}
    
    with gp.Model("Gurobi_NonConvex", env=env) as m1:
        m1.setParam('NonConvex', 2)
        
        x_idx = [(r, k) for r in tasks for k in task_data[r]['passes']]
        p_idx = [(r, k) for r in tasks for k in range(len(task_data[r]['passes']) + 1)]
        
        x1 = m1.addVars(x_idx, vtype=GRB.BINARY, name="x1")
        p = m1.addVars(p_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="p")
        
        for r in tasks:
            m1.addConstr(p[r, 0] == 1.0)
            for k in task_data[r]['passes']:
                theta = task_data[r]['probabilities'][k]
                m1.addQConstr(p[r, k+1] == p[r, k] - theta * p[r, k] * x1[r, k])
        
        rev_m1 = gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * x1[r, k] * p[r, k] for r in tasks for k in task_data[r]['passes'])
        pen_m1 = gp.quicksum(dynamic_taxes[r] * x1[r, k] for r in tasks for k in task_data[r]['passes'])
        m1.setObjective(rev_m1 - pen_m1, GRB.MAXIMIZE)
        
        t_start = time.perf_counter()
        m1.optimize()
        runtime_gurobi = time.perf_counter() - t_start
        
        if m1.Status == GRB.OPTIMAL:
            obj_gurobi = m1.ObjVal
            gurobi_decisions = {idx: x1[idx].X for idx in x_idx}

    # =====================================================================
    # TRACK 2: OPEN-SOURCE SCIP LINEAR MIP (MCCORMICK POLYTOPE)
    # =====================================================================
    print("🚀 Solving Track 2: SCIP Linear MILP (McCormick Envelopes)...")
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver backend could not be initialized in OR-Tools.")
        
    obj_scip = None
    scip_decisions = {}
    
    x2 = {}
    y = {}
    w = {}
    
    # Declare Linear Variables
    for r in tasks:
        y[(r, 0)] = solver.NumVar(1.0, 1.0, f"y_{r}_0")
        for k in task_data[r]['passes']:
            x2[(r, k)] = solver.BoolVar(f"x2_{r}_{k}")
            y[(r, k+1)] = solver.NumVar(0.0, 1.0, f"y_{r}_{k+1}")
            w[(r, k)] = solver.NumVar(0.0, 1.0, f"w_{r}_{k}")

    # Build Linear Horizontal Dynamics and McCormick Bounding Constraints
    for r in tasks:
        for k in task_data[r]['passes']:
            theta = task_data[r]['probabilities'][k]
            
            # State recurrence update
            solver.Add(y[(r, k+1)] == y[(r, k)] - theta * w[(r, k)])
            
            # Exact Continuous-Binary McCormick Envelopes
            solver.Add(w[(r, k)] <= x2[(r, k)])
            solver.Add(w[(r, k)] <= y[(r, k)])
            solver.Add(w[(r, k)] >= y[(r, k)] - (1.0 - x2[(r, k)]))
            solver.Add(w[(r, k)] >= 0.0)

    # Compile Objective Function Bounds
    objective = solver.Objective()
    for r in tasks:
        for k in task_data[r]['passes']:
            q = task_data[r]['qualities'][k]
            t = task_data[r]['probabilities'][k]
            tax = dynamic_taxes[r]
            
            # Note that x * p in non-convex space translates directly to w in McCormick space
            objective.SetCoefficient(w[(r, k)], q * t)
            objective.SetCoefficient(x2[(r, k)], -tax)
            
    objective.SetMaximization()
    
    t_start = time.perf_counter()
    status = solver.Solve()
    runtime_scip = time.perf_counter() - t_start
    
    if status == pywraplp.Solver.OPTIMAL:
        obj_scip = objective.Value()
        scip_decisions = {(r, k): x2[(r, k)].solution_value() for r in tasks for k in task_data[r]['passes']}

    # =====================================================================
    # COMPARISON REPORTING
    # =====================================================================
    print("\n⏱️  ================ FRAMEWORK CONVERGENCE REPORT ================")
    print(f"Gurobi Native Non-Convex Objective:  {obj_gurobi:.5f}")
    print(f"SCIP Transformed McCormick Objective: {obj_scip:.5f}")
    
    precision_delta = abs(obj_gurobi - obj_scip)
    print(f"Absolute Precision Variance Delta:   {precision_delta:.5e}")
    
    # Verify decision space alignment matches perfectly
    mismatched_decisions = 0
    for idx in gurobi_decisions:
        g_val = round(gurobi_decisions[idx])
        s_val = round(scip_decisions[idx])
        if g_val != s_val:
            mismatched_decisions += 1
            
    print(f"Total Mismatched Schedule Decisions: {mismatched_decisions} / {len(gurobi_decisions)}")
    print("------------------------------------------------------------------")
    print(f"Gurobi RunTime (Non-Convex Smooth):  {runtime_gurobi:.4f} seconds")
    print(f"SCIP RunTime (Transformed Matrix):   {runtime_scip:.4f} seconds")
    print("==================================================================\n")


if __name__ == "__main__":
    run_gurobi_vs_scip_horizontal_test(num_tasks=15, passes_per_task=10, gamma=0.15)