import os
import time
import gurobipy as gp
from gurobipy import GRB
from dotenv import load_dotenv

# 1. Load parameters from local .env file
load_dotenv()

def run_large_scale_speed_test(num_tasks=60, passes_per_task=80, gamma=0.15):
    print(f"\n=======================================================")
    print(f"--- BATCH SPEED TEST: HORIZONTAL SCALING BENCHMARK ---")
    print(f"=======================================================")
    print(f"-> Scaling Footprint: {num_tasks} Tasks x {passes_per_task} Passes = {num_tasks * passes_per_task} Total Assets")
    
    # --- Step 1: Procedurally Generate Large Dataset ---
    tasks = [f"target_{i}" for i in range(num_tasks)]
    task_data = {}
    
    for i, r in enumerate(tasks):
        passes = list(range(passes_per_task))
        # Ensure qualities are cleanly sorted descending per task
        base_quality = 80.0 + (i % 3) * 10.0
        qualities = {k: float(base_quality - k * 4.0) for k in passes}
        # Dynamic probabilities oscillating between 0.35 and 0.95
        probabilities = {k: round(0.40 + 0.07 * ((k + i) % 8), 2) for k in passes}
        
        task_data[r] = {
            'passes': passes,
            'qualities': qualities,
            'probabilities': probabilities
        }

    # Precalculate Task-Specific Dynamic Taxes
    dynamic_taxes = {}
    for r in tasks:
        max_q = max(task_data[r]['qualities'].values())
        dynamic_taxes[r] = gamma * max_q

    # --- Step 2: Extract WLS Credentials ---
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    gurobi_license_id = os.getenv("LICENSEID")

    if not all([wls_access_id, wls_secret, gurobi_license_id]):
        raise ValueError("Missing Gurobi WLS credentials. Please verify your .env file layout!")

    wls_params = {
        "WLSACCESSID": wls_access_id,
        "WLSSECRET": wls_secret,
        "LICENSEID": int(gurobi_license_id),
        "OutputFlag": 0  # Suppress internal prints to keep terminal focused on metrics
    }

    x_idx = [(r, k) for r in tasks for k in task_data[r]['passes']]
    p_idx = [(r, k) for r in tasks for k in range(len(task_data[r]['passes']) + 1)]

    obj_m1, obj_m2 = None, None
    runtime_m1, runtime_m2 = 0.0, 0.0

    with gp.Env(params=wls_params) as env:
        
        # =====================================================================
        # MODEL 1: RAW NON-CONVEX BENCHMARK
        # =====================================================================
        print("⚡ Executing Model 1: Raw Non-Convex (Spatial Branch-and-Bound)...")
        start_time = time.time()
        
        with gp.Model("Large_NonConvex", env=env) as m1:
            m1.setParam('NonConvex', 2)
            m1.setParam('Presolve', 0)

            
            x1 = m1.addVars(x_idx, vtype=GRB.CONTINUOUS, name="x1")
            p = m1.addVars(p_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="p")
            
            for r in tasks:
                m1.addConstr(p[r, 0] == 1.0)
                for k in task_data[r]['passes']:
                    theta = task_data[r]['probabilities'][k]
                    m1.addQConstr(p[r, k+1] == p[r, k] - theta * p[r, k] * x1[r, k])
            
            rev_m1 = gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * x1[r, k] * p[r, k] for r in tasks for k in task_data[r]['passes'])
            pen_m1 = gp.quicksum(dynamic_taxes[r] * x1[r, k] for r in tasks for k in task_data[r]['passes'])
            m1.setObjective(rev_m1 - pen_m1, GRB.MAXIMIZE)
            
            m1.optimize()
            runtime_m1 = time.time() - start_time
            if m1.Status == GRB.OPTIMAL:
                obj_m1 = m1.ObjVal

        # =====================================================================
        # MODEL 2: TRANSFORMED LINEAR BENCHMARK
        # =====================================================================
        print("🚀 Executing Model 2: Transformed Linear MIP (Convex Polytope)...")
        start_time = time.time()
        
        with gp.Model("Large_Transformed_Linear", env=env) as m2:
            m2.setParam('Presolve', 0)

            x2 = m2.addVars(x_idx, vtype=GRB.BINARY, name="x2")
            y = m2.addVars(p_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
            w = m2.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")
            
            for r in tasks:
                m2.addConstr(y[r, 0] == 1.0)
                for k in task_data[r]['passes']:
                    theta = task_data[r]['probabilities'][k]
                    m2.addConstr(y[r, k+1] == y[r, k] - theta * w[r, k])
                    m2.addConstr(w[r, k] <= x2[r, k])
                    m2.addConstr(w[r, k] <= y[r, k])
                    m2.addConstr(w[r, k] >= y[r, k] - (1.0 - x2[r, k]))
                    #m2.addConstr(w[r, k] >= 0.0)
            
            rev_m2 = gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * w[r, k] for r in tasks for k in task_data[r]['passes'])
            pen_m2 = gp.quicksum(dynamic_taxes[r] * x2[r, k] for r in tasks for k in task_data[r]['passes'])
            m2.setObjective(rev_m2 - pen_m2, GRB.MAXIMIZE)
            
            m2.optimize()
            runtime_m2 = time.time() - start_time
            if m2.Status == GRB.OPTIMAL:
                obj_m2 = m2.ObjVal

    # --- Print Benchmark Analytics ---
    print("\n⏱️  ================ SPEED BENCHMARK REPORT ================")
    print(f"Model 1 (Raw Non-Convex) Optimal Objective:   {obj_m1:.4f}")
    print(f"Model 2 (Transformed Linear) Optimal Objective:{obj_m2:.4f}")
    print(f"Mathematical Precision Check Delta:            {abs(obj_m1 - obj_m2):.5e}")
    print("-------------------------------------------------------")
    print(f"Model 1 Execution Time (Non-Convex):           {runtime_m1:.4f} seconds")
    print(f"Model 2 Execution Time (Linear MIP):           {runtime_m2:.4f} seconds")
    print("-------------------------------------------------------")
    
    speedup = runtime_m1 / max(runtime_m2, 1e-6)
    print(f"📈 NET FRAMEWORK PERFORMANCE SPEEDUP:          {speedup:.2f}x Faster")
    print("=======================================================\n")

if __name__ == "__main__":
    # Feel free to turn num_tasks up to 100 or 150 to watch Model 1's runtime scale exponentially
    run_large_scale_speed_test(num_tasks=10, passes_per_task=12, gamma=0.12)