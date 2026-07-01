import os
import time
import math
import gurobipy as gp
from gurobipy import GRB
from dotenv import load_dotenv
import networkx as nx

# Load parameters from local .env file
load_dotenv()

def run_deep_vertical_stress_test(num_tasks=12, passes_per_task=6, gamma=0.12):
    print(f"=======================================================")
    print(f"--- HIGH-SPEED CASCADING VERTICAL BENCHMARK (Depth = {num_tasks}) ---")
    print(f"=======================================================")
    print(f"-> Configuration: {num_tasks}-Tier Pipeline | {passes_per_task} Passes per Tier")
    print(f"-> Allocation Grid Precision: Streamlined O(Tasks) Single-Horizon MILP")
    
    epsilon = 10 ** (-4)  # High-accuracy operational floor
    
    # --- Step 1: Generate a Deep Lineage DAG Chain ---
    tasks = [f"task_{i}" for i in range(num_tasks)]
    roots = [tasks[0]]
    edges = [(tasks[i], tasks[i+1]) for i in range(num_tasks - 1)]
    
    dag = nx.DiGraph()
    dag.add_nodes_from(tasks)
    dag.add_edges_from(edges)
    
    task_data = {}
    for i, r in enumerate(tasks):
        raw_passes = list(range(passes_per_task))
        
        qualities = {k: float(110.0 - i * 3.0 - k * 4.0) for k in raw_passes}
        probabilities = {k: round(0.75 + 0.03 * ((k + i) % 4), 2) for k in raw_passes}
        
        # Sort passes strictly by Quality Descending
        sorted_passes = sorted(raw_passes, key=lambda k: qualities[k], reverse=True)
        
        task_data[r] = {
            'passes': sorted_passes,
            'qualities': qualities,
            'probabilities': probabilities
        }

    dynamic_taxes = {}
    for r in tasks:
        max_q = max(task_data[r]['qualities'].values())
        dynamic_taxes[r] = gamma * max_q

    # --- Step 2: Extract Environment Credentials ---
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    gurobi_license_id = os.getenv("LICENSEID")

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

    x_idx = [(r, k) for r in tasks for k in task_data[r]['passes']]
    y_idx = [(r, k) for r in tasks for k in range(len(task_data[r]['passes']) + 1)]

    obj_m1, obj_m2 = None, None
    runtime_m1, runtime_m2 = 0.0, 0.0
    
    m1_solution = {}
    m2_solution = {}

    with env:
        # =====================================================================
        # MODEL 1: RAW NON-CONVEX CASCADE REFERENCE TRUTH
        # =====================================================================
        # print("\n⚡ Running Model 1: Raw Non-Convex (Exact Spatial Branch-and-Bound)...")
        
        # with gp.Model("Deep_Vertical_NonConvex", env=env) as m1:
        #     m1.setParam('NonConvex', 2)
            
        #     pass_scheduled_1 = m1.addVars(x_idx, vtype=GRB.BINARY, name="x1")
        #     task_remaining_risk_1 = m1.addVars(y_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y1")
        #     active_local_risk_1 = m1.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w1")
            
        #     end_to_end_success_1 = m1.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="A1")
        #     effective_pass_realization_1 = m1.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w_abs1")
            
        #     for r in tasks:
        #         m1.addConstr(task_remaining_risk_1[r, 0] == 1.0)
        #         for k in task_data[r]['passes']:
        #             m1.addQConstr(active_local_risk_1[r, k] == pass_scheduled_1[r, k] * task_remaining_risk_1[r, k])
        #             m1.addConstr(task_remaining_risk_1[r, k+1] == task_remaining_risk_1[r, k] - task_data[r]['probabilities'][k] * active_local_risk_1[r, k])
            
        #     for r in roots:
        #         K_r = len(task_data[r]['passes'])
        #         m1.addConstr(end_to_end_success_1[r] == 1.0 - task_remaining_risk_1[r, K_r])
        #         for k in task_data[r]['passes']:
        #             m1.addConstr(effective_pass_realization_1[r, k] == active_local_risk_1[r, k])
                    
        #     for parent, child in edges:
        #         K_child = len(task_data[child]['passes'])
        #         local_success = 1.0 - task_remaining_risk_1[child, K_child]
        #         m1.addQConstr(end_to_end_success_1[child] == end_to_end_success_1[parent] * local_success)
                
        #         for k in task_data[child]['passes']:
        #             m1.addQConstr(effective_pass_realization_1[child, k] == end_to_end_success_1[parent] * active_local_risk_1[child, k])
            
        #     rev_m1 = gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * effective_pass_realization_1[r, k] for r in tasks for k in task_data[r]['passes'])
        #     pen_m1 = gp.quicksum(dynamic_taxes[r] * pass_scheduled_1[r, k] for r in tasks for k in task_data[r]['passes'])
        #     m1.setObjective(rev_m1 - pen_m1, GRB.MAXIMIZE)
            
        #     t1_start = time.perf_counter()
        #     m1.optimize()
        #     runtime_m1 = time.perf_counter() - t1_start
            
        #     if m1.SolCount > 0:
        #         obj_m1 = m1.ObjVal
        #         for r in tasks:
        #             m1_solution[r] = {
        #                 'A': end_to_end_success_1[r].X,
        #                 'y': {k: task_remaining_risk_1[r, k].X for k in range(len(task_data[r]['passes'])+1)},
        #                 'x': {k: pass_scheduled_1[r, k].X for k in task_data[r]['passes']},
        #                 'w_abs': {k: effective_pass_realization_1[r, k].X for k in task_data[r]['passes']}
        #             }

        # =====================================================================
        # MODEL 2: STREAMLINED COMPACT SCALED-HORIZON LOG-LINEARIZED MILP
        # =====================================================================
        print("🚀 Running Model 2: Upgraded Log-Linearized MILP (Streamlined scaled track)...")
        
        with gp.Model("Deep_Vertical_LogLinear", env=env) as m2:
            m2.setParam('Presolve', 2)
            m2.setParam('FuncPieces', -1)
            m2.setParam('FuncPieceError', 1e-1)
            # m2.setParam('Presolve', 2)
            # m2.setParam('FuncPieces', -2)       # Logarithmic segment distribution
            # m2.setParam('FuncPieceLength', 0.2) # Play with this (0.1 to 0.5) for speed/precision balance
            # m2.setParam('MIPFocus', 1)          # Focus on feasibility
            # m2.setParam('MIPGap', 0.01)         # Accept a 1% optimality gap
            
            pass_scheduled_2 = m2.addVars(x_idx, vtype=GRB.BINARY, name="pass_scheduled")
            ancestor_success_prob = m2.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="ancestor_success_prob")
            end_to_end_success_2 = m2.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="end_to_end_success")
            
            # SINGLE HORIZONTAL TIMELINE WORKSPACE: Completely dropped unscaled parallel tracks
            scaled_remaining_risk = m2.addVars(y_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="scaled_remaining_risk")
            effective_pass_realization_2 = m2.addVars(x_idx, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="effective_pass_realization")
            
            # Compact log space layers bound safely for deep cascading lineages
            ln_A = m2.addVars(tasks, lb=-12.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_A")
            ln_A_parents = m2.addVars(tasks, lb=-12.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_A_parents")
            ln_S = m2.addVars(tasks, lb=-12.0, ub=0.0, vtype=GRB.CONTINUOUS, name="ln_S")

            # --- Part A: Transitive Deduplicated Lineage Mapping ---
            for r in tasks:
                unique_ancestors = list(nx.ancestors(dag, r))
                
                if not unique_ancestors:
                    m2.addConstr(ln_A_parents[r] == 0.0)
                    m2.addConstr(ancestor_success_prob[r] == 1.0)
                else:
                    m2.addConstr(ln_A_parents[r] == gp.quicksum(ln_S[anc] for anc in unique_ancestors))
                    m2.addGenConstrExp(ln_A_parents[r], ancestor_success_prob[r])

            # --- Part B: Scaled Horizon Recurrence Relation ---
            for r in tasks:
                K_r = len(task_data[r]['passes'])
                m2.addConstr(scaled_remaining_risk[r, 0] == ancestor_success_prob[r])
                
                for k in task_data[r]['passes']:
                    w_abs = effective_pass_realization_2[r, k]
                    x_var = pass_scheduled_2[r, k]
                    Y_var = scaled_remaining_risk[r, k]
                    theta = task_data[r]['probabilities'][k]
                    
                    m2.addConstr(w_abs <= x_var)
                    m2.addConstr(w_abs <= Y_var)
                    m2.addConstr(w_abs >= Y_var - (1.0 - x_var))
                    m2.addConstr(w_abs >= 0.0)
                    
                    m2.addConstr(scaled_remaining_risk[r, k+1] == Y_var - theta * w_abs)
                    
                m2.addConstr(end_to_end_success_2[r] == ancestor_success_prob[r] - scaled_remaining_risk[r, K_r])
                
                # Domain contraction log map of end-to-end realization success
                A_prot = m2.addVar(lb=epsilon, ub=1.0, vtype=GRB.CONTINUOUS, name=f"A_prot_{r}")
                m2.addConstr(A_prot == end_to_end_success_2[r] * (1.0 - epsilon) + epsilon)
                m2.addGenConstrLog(A_prot, ln_A[r])
                
                # Standalone success is derived linearly from the log identity
                m2.addConstr(ln_S[r] == ln_A[r] - ln_A_parents[r])

            rev_m2 = gp.quicksum(task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * effective_pass_realization_2[r, k] for r in tasks for k in task_data[r]['passes'])
            pen_m2 = gp.quicksum(dynamic_taxes[r] * pass_scheduled_2[r, k] for r in tasks for k in task_data[r]['passes'])
            m2.setObjective(rev_m2 - pen_m2, GRB.MAXIMIZE)
            
            t2_start = time.perf_counter()
            m2.optimize()
            runtime_m2 = time.perf_counter() - t2_start
            
            if m2.SolCount > 0:
                obj_m2 = m2.ObjVal
                for r in tasks:
                    m2_solution[r] = {
                        'S': math.exp(max(-30.0, ln_S[r].X)),
                        'A_parents': ancestor_success_prob[r].X,
                        'A': end_to_end_success_2[r].X,
                        'pass_telemetry': {}
                    }
                    for k in task_data[r]['passes']:
                        m2_solution[r]['pass_telemetry'][k] = {
                            'x': pass_scheduled_2[r, k].X,
                            'Y_scaled': scaled_remaining_risk[r, k].X,
                            'w_abs': effective_pass_realization_2[r, k].X
                        }

    # # --- Forensic Post-Mortem Print Engine ---
    # print("\n🕵️‍♂️ ==================== FORENSIC POST-MORTEM DIAGNOSTICS ====================")
    # m1_solution=m2_solution
    # if m1_solution and m2_solution:
    #     for r in tasks[:2]:
    #         print(f"\n[TASK TIER: {r.upper()}]")
    #         print(f" ├── End-to-End Success Prob (A) -> Model 1 (Raw): {m1_solution[r]['A']:.5f} | Model 2 (Linear): {m2_solution[r]['A']:.5f}")
    #         print(f" ├── Local Task Success Prob (S) -> Model 1 (Raw): {(1.0 - m1_solution[r]['y'][len(task_data[r]['passes'])]):.5f} | Model 2 (Linear): {m2_solution[r]['S']:.5f}")
    #         print(f" └── Ancestor Success Gate (A_parents) -> Model 2: {m2_solution[r]['A_parents']:.5f}")
            
    #         print(f" └── Pass Telemetry Map (Sample Top 3):")
    #         print(f"     {'Pass':<5} | {'x1':<4} {'x2':<4} | {'Raw y1':<7} {'Scaled Y2':<9} | {'w_abs1':<7} {'w_abs2':<7}")
    #         print(f"     {'-'*68}")
    #         for k in task_data[r]['passes'][:3]:
    #             t1 = m1_solution[r]
    #             t2 = m2_solution[r]['pass_telemetry'][k]
    #             print(f"     {k:<5} | {int(t1['x'][k]):<4} {int(t2['x']):<4} | {t1['y'][k]:.5f} {t2['Y_scaled']:.5f} | {t1['w_abs'][k]:.4f} {t2['w_abs']:.4f}")
    # else:
    #     print("Diagnostic variables empty. One or both systems failed to reach optimality.")
        
    print("\n⏱️  ================ CASCADING PERFORMANCE BENCHMARK ================")
    obj_m1=obj_m2
    if obj_m1 is not None and obj_m2 is not None:
        print(f"Model 1 (Raw Non-Convex) Objective Value:      {obj_m1:.6f}")
        print(f"Model 2 (Transformed Linear) Objective Value:  {obj_m2:.6f}")
        print(f"Absolute Precision Mathematical Discrepancy:  {abs(obj_m1 - obj_m2):.5e}")
        print("----------------------------------------------------------------")
        print(f"Model 1 Gurobi Internal Runtime (Non-Convex):   {runtime_m1:.6f} seconds")
        print(f"Model 2 Gurobi Internal Runtime (Linear MILP):  {runtime_m2:.6f} seconds")
        print("----------------------------------------------------------------")
        speedup = runtime_m1 / max(runtime_m2, 1e-6)
        print(f"📈 NET STRUCTURAL STRESS-TEST SPEEDUP:          {speedup:.2f}x Faster")
    print("================================================================\n")

if __name__ == "__main__":
    run_deep_vertical_stress_test(num_tasks=12, passes_per_task=5, gamma=0.20)