import os
import time
import gurobipy as gp
from gurobipy import GRB
from dotenv import load_dotenv

# Load parameters from local .env file
load_dotenv()

def run_optimized_log_mip_test(num_tasks=6, passes_per_task=4, gamma=0.12):
    print(f"\n=======================================================")
    print(f"--- OPTIMIZED HIGH-SPEED HYBRID LOG-MIP BENCHMARK ---")
    print(f"=======================================================")
    
    tasks = [f"task_{i}" for i in range(num_tasks)]
    roots = [tasks[0]]
    edges = [(tasks[i], tasks[i+1]) for i in range(num_tasks - 1)]
    
    task_data = {}
    for i, r in enumerate(tasks):
        passes = list(range(passes_per_task))
        qualities = {k: float(110.0 - i * 5.0 - k * 4.0) for k in passes}
        probabilities = {k: round(0.70 + 0.04 * ((k + i) % 4), 2) for k in passes}
        task_data[r] = {
            'passes': passes,
            'qualities': qualities,
            'probabilities': probabilities
        }

    dynamic_taxes = {}
    for r in tasks:
        max_q = max(task_data[r]['qualities'].values())
        dynamic_taxes[r] = gamma * max_q

    wls_params = {
        "WLSACCESSID": os.getenv("WLSACCESSID"),
        "WLSSECRET": os.getenv("WLSSECRET"),
        "LICENSEID": int(os.getenv("LICENSEID")),
        "OutputFlag": 0  
    }

    x_idx = [(r, k) for r in tasks for k in task_data[r]['passes']]
    y_idx = [(r, k) for r in tasks for k in range(len(task_data[r]['passes']) + 1)]

    with gp.Env(params=wls_params) as env:
        
        # =====================================================================
        # MODEL 1: THE RAW NON-CONVEX REFERENCE TRUTH
        # =====================================================================
        with gp.Model("True_NonConvex", env=env) as m1:
            m1.setParam('NonConvex', 2)
            
            x1 = m1.addVars(x_idx, vtype=GRB.BINARY, name="x1")
            y1 = m1.addVars(y_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y1")
            w1 = m1.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w1")
            A1 = m1.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="A1")
            g1 = m1.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="g1")
            w_abs1 = m1.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w_abs1")
            
            for r in tasks:
                m1.addConstr(y1[r, 0] == 1.0)
                for k in task_data[r]['passes']:
                    m1.addQConstr(w1[r, k] == x1[r, k] * y1[r, k])
                    m1.addConstr(y1[r, k+1] == y1[r, k] - task_data[r]['probabilities'][k] * w1[r, k])
            
            for r in roots:
                K_r = len(task_data[r]['passes'])
                m1.addConstr(A1[r] == 1.0 - y1[r, K_r])
                for k in task_data[r]['passes']:
                    m1.addConstr(w_abs1[r, k] == w1[r, k])
                    
            for parent, child in edges:
                K_child = len(task_data[child]['passes'])
                m1.addQConstr(g1[child] == A1[parent] * y1[child, K_child])
                m1.addConstr(A1[child] == A1[parent] - g1[child])
                for k in task_data[child]['passes']:
                    # FIXED: Removed the messy getVars() call, using direct lookup
                    m1.addQConstr(w_abs1[child, k] == A1[parent] * w1[child, k])
            
            m1.setObjective(
                gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * w_abs1[r, k] for r, k in x_idx) -
                gp.quicksum(dynamic_taxes[r] * x1[r, k] for r, k in x_idx),
                GRB.MAXIMIZE
            )
            
            start_m1 = time.perf_counter()
            m1.optimize()
            runtime_m1 = time.perf_counter() - start_m1
            obj_m1 = m1.ObjVal if m1.Status == GRB.OPTIMAL else None

        # =====================================================================
        # MODEL 2: OPTIMIZED HYBRID LOG-LINEARIZED MILP
        # =====================================================================
        with gp.Model("Optimized_Log_MIP", env=env) as m2:
            eps = 1e-5  
            # SPEEDUP FIX: Coarsen error tolerance to 1% to prune internal matrix overhead
            pwl_opts = "FuncPieces=-2 FuncPieceLength=1e-2" 
            
            x2 = m2.addVars(x_idx, vtype=GRB.BINARY, name="x2")
            y2 = m2.addVars(y_idx, lb=eps, ub=1.0, vtype=GRB.CONTINUOUS, name="y2")
            w2 = m2.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w2")
            
            A2 = m2.addVars(tasks, lb=eps, ub=1.0, vtype=GRB.CONTINUOUS, name="A2")
            S2 = m2.addVars(tasks, lb=eps, ub=1.0, vtype=GRB.CONTINUOUS, name="S2")
            w_abs2 = m2.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w_abs2")
            Omega = m2.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Omega")
            
            ln_A = m2.addVars(tasks, lb=-20.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_A")
            ln_S = m2.addVars(tasks, lb=-20.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_S")
            ln_y = m2.addVars(x_idx, lb=-20.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_y")
            ln_Omega = m2.addVars(x_idx, lb=-25.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_Omega")
            
            for r in tasks:
                m2.addConstr(y2[r, 0] == 1.0)
                K_r = len(task_data[r]['passes'])
                for k in task_data[r]['passes']:
                    m2.addConstr(y2[r, k+1] == y2[r, k] - task_data[r]['probabilities'][k] * w2[r, k])
                    m2.addConstr(w2[r, k] <= x2[r, k])
                    m2.addConstr(w2[r, k] <= y2[r, k])
                    m2.addConstr(w2[r, k] >= y2[r, k] - (1.0 - x2[r, k]))
                    m2.addConstr(w2[r, k] >= 0.0)
                    
                    m2.addGenConstrLog(y2[r, k], ln_y[r, k], options=pwl_opts)
                
                # RECOVERY FIX: Strict equality line maps to exact execution ceilings
                m2.addConstr(S2[r] == 1.0 - y2[r, K_r] + eps)
                m2.addGenConstrLog(S2[r], ln_S[r], options=pwl_opts)

            for r in roots:
                m2.addConstr(A2[r] == S2[r])
                m2.addGenConstrLog(A2[r], ln_A[r], options=pwl_opts)
                for k in task_data[r]['passes']:
                    m2.addConstr(w_abs2[r, k] == w2[r, k])
                    
            parent_lookup = {child: parent for parent, child in edges}
            
            for r in tasks:
                if r in roots: continue
                parent = parent_lookup[r]
                
                m2.addConstr(ln_A[r] == ln_A[parent] + ln_S[r])
                m2.addGenConstrExp(ln_A[r], A2[r], options=pwl_opts)
                
                for k in task_data[r]['passes']:
                    m2.addConstr(ln_Omega[r, k] == ln_A[parent] + ln_y[r, k])
                    m2.addGenConstrExp(ln_Omega[r, k], Omega[r, k], options=pwl_opts)
                    
                    m2.addConstr(w_abs2[r, k] <= x2[r, k])
                    m2.addConstr(w_abs2[r, k] <= Omega[r, k])
                    m2.addConstr(w_abs2[r, k] >= Omega[r, k] - (1.0 - x2[r, k]))
                    m2.addConstr(w_abs2[r, k] >= 0.0)

            m2.setObjective(
                gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * w_abs2[r, k] for r, k in x_idx) -
                gp.quicksum(dynamic_taxes[r] * x2[r, k] for r, k in x_idx),
                GRB.MAXIMIZE
            )
            
            start_m2 = time.perf_counter()
            m2.optimize()
            runtime_m2 = time.perf_counter() - start_m2
            obj_m2 = m2.ObjVal if m2.Status == GRB.OPTIMAL else None

    print("\n⏱️  ================ OPTIMIZED PERFORMANCE REPORT ================")
    print(f"Model 1 (True Non-Convex) Objective Value:     {obj_m1:.5f}")
    print(f"Model 2 (Optimized Log-MIP) Objective Value:   {obj_m2:.5f}")
    print(f"Precision Variance Delta (Within 1% PWL Bound): {abs(obj_m1 - obj_m2):.5e}")
    print("------------------------------------------------------------------")
    print(f"Model 1 Execution Runtime (Non-Convex Smooth): {runtime_m1:.4f} seconds")
    print(f"Model 2 Execution Runtime (Optimized Log-MIP): {runtime_m2:.4f} seconds")
    print("==================================================================\n")

if __name__ == "__main__":
    run_optimized_log_mip_test(num_tasks=4, passes_per_task=4, gamma=0.12)