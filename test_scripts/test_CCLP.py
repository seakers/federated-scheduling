import os
import gurobipy as gp
from gurobipy import GRB
from dotenv import load_dotenv

# 1. Load the parameters from your local .env file
load_dotenv()

def run_workflow_stochastic_test(gamma):
    print(f"\n=======================================================")
    print(f"--- Running Workflow WLS Optimization (Gamma = {gamma}) ---")
    print(f"=======================================================")
    
    # --- Step 1: Define Workflow DAG Structure & Pass Data ---
    tasks = ['volcano_detection', 'volcano_followup']
    roots = ['volcano_detection']
    edges = [('volcano_detection', 'volcano_followup')] # (Parent, Child)
    
    # Independent pass metrics per task (Each list pre-sorted by Quality descending)
    task_data = {
        'volcano_detection': {
            'passes': [0, 1, 2],
            'qualities': {0: 100.0, 1: 90.0, 2: 60.0},
            'probabilities': {0: 0.70, 1: 0.82, 2: 0.98}
        },
        'volcano_followup': {
            'passes': [0, 1],
            'qualities': {0: 90.0, 1: 70.0},
            'probabilities': {0: 0.50, 1: 0.80}
        }
    }

    # Precalculate Task-Specific Dynamic Taxes
    dynamic_taxes = {}
    for r in tasks:
        max_q = max(task_data[r]['qualities'].values())
        dynamic_taxes[r] = gamma * max_q
        print(f"-> Task '{r}': Max Quality = {max_q:.1f} | Dynamic Tax (c_{r}) = {dynamic_taxes[r]:.2f}")

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
        "OutputFlag": 0  
    }

    # --- Step 3: Initialize Environment & Model ---
    with gp.Env(params=wls_params) as env:
        with gp.Model("Stochastic_Workflow_WLS", env=env) as model:

            # --- Step 4: Declare Multidimensional Variables ---
            x_indices = [(r, k) for r in tasks for k in task_data[r]['passes']]
            w_indices = x_indices
            
            y_indices = []
            for r in tasks:
                K_r = len(task_data[r]['passes'])
                for k in range(K_r + 1):
                    y_indices.append((r, k))

            # Core Local Timeline Variables
            x = model.addVars(x_indices, vtype=GRB.BINARY, name="x")
            y = model.addVars(y_indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
            w = model.addVars(w_indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")
            
            # High-Level Path Tracking Variables
            A = model.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="A")
            g = model.addVars(tasks, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="g")
            w_abs = model.addVars(x_indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w_abs")

            # --- Step 5: Inject Matrix Constraints ---
            
            # 1. Sanity Check: Every single task starts 100% locally unfulfilled
            for r in tasks:
                model.addConstr(y[r, 0] == 1.0, name=f"Horiz_Init_{r}")

            # 2. Define Absolute Path Success Bounds
            # For Root Tasks: Absolute success is exactly its local success rate
            for r in roots:
                K_r = len(task_data[r]['passes'])
                model.addConstr(A[r] == 1.0 - y[r, K_r], name=f"Root_Abs_Success_{r}")

            # Lookup mapping to easily trace dependencies downstream
            parent_lookup = {child: parent for parent, child in edges}

            for r in tasks:
                K_r = len(task_data[r]['passes'])
                
                if r in roots:
                    # Root has no parent, absolute pass probability is just local pass probability
                    for k in task_data[r]['passes']:
                        model.addConstr(w_abs[r, k] == w[r, k], name=f"Root_W_Abs_{r}_{k}")
                else:
                    parent = parent_lookup[r]
                    # Linearized Vertical Bridge: A[child] == A[parent] - g[child]
                    model.addConstr(A[r] == A[parent] - g[r], name=f"Vertical_Link_{r}")
                    
                    # Continuous-Continuous McCormick bounds for g[child] = A[parent] * y[child, K_child]
                    model.addConstr(g[r] >= 0.0, name=f"McCormick_g_L1_{r}")
                    model.addConstr(g[r] >= A[parent] + y[r, K_r] - 1.0, name=f"McCormick_g_L2_{r}")
                    model.addConstr(g[r] <= A[parent], name=f"McCormick_g_U1_{r}")
                    model.addConstr(g[r] <= y[r, K_r], name=f"McCormick_g_U2_{r}")
                    
                    # Continuous-Continuous McCormick bounds for w_abs[child, k] = A[parent] * w[child, k]
                    for k in task_data[r]['passes']:
                        model.addConstr(w_abs[r, k] >= 0.0, name=f"McCormick_wabs_L1_{r}_{k}")
                        model.addConstr(w_abs[r, k] >= A[parent] + w[r, k] - 1.0, name=f"McCormick_wabs_L2_{r}_{k}")
                        model.addConstr(w_abs[r, k] <= A[parent], name=f"McCormick_wabs_U1_{r}_{k}")
                        model.addConstr(w_abs[r, k] <= w[r, k], name=f"McCormick_wabs_U2_{r}_{k}")

            # 3. Horizontal Local Timeline Tracking Loops
            for r in tasks:
                passes = task_data[r]['passes']
                probs = task_data[r]['probabilities']
                
                for k in passes:
                    model.addConstr(y[r, k+1] == y[r, k] - probs[k] * w[r, k], name=f"Horiz_Prop_{r}_{k}")
                    model.addConstr(w[r, k] <= x[r, k], name=f"McCormick_U1_{r}_{k}")
                    model.addConstr(w[r, k] <= y[r, k], name=f"McCormick_U2_{r}_{k}")
                    model.addConstr(w[r, k] >= y[r, k] - (1.0 - x[r, k]), name=f"McCormick_L1_{r}_{k}")
                    model.addConstr(w[r, k] >= 0.0, name=f"McCormick_L2_{r}_{k}")

            # --- Step 6: Construct Absolute Joint Objective ---
            # Reward is scaled dynamically by the absolute path weight variable
            reward_term = gp.quicksum(
                task_data[r]['qualities'][k] * task_data[r]['probabilities'][k] * w_abs[r, k] 
                for r in tasks for k in task_data[r]['passes']
            )
            penalty_term = gp.quicksum(
                dynamic_taxes[r] * x[r, k] 
                for r in tasks for k in task_data[r]['passes']
            )
            model.setObjective(reward_term - penalty_term, GRB.MAXIMIZE)

            # --- Step 7: Execute Optimization Engine ---
            model.optimize()

            # --- Step 8: Parse Global Pipeline Output ---
            if model.Status == GRB.OPTIMAL:
                print(f"\n🟢 Optimization Successful. Global Score: {model.ObjVal:.2f}")
                
                for r in tasks:
                    K_r = len(task_data[r]['passes'])
                    print(f"\n📋 Schedule Profile for Task: '{r}'")
                    print(f"  Initial Local Reservoir (y_0): {y[r, 0].X:.3f} (Locally 100% Unfulfilled)")
                    print(f"  ✨ Absolute Path Joint Success (A_r): {A[r].X * 100:.1f}%")
                    
                    for k in task_data[r]['passes']:
                        status = "Scheduled" if x[r, k].X > 0.5 else "Skipped  "
                        print(f"    🛰️ Pass {k+1} -> {status} (x={int(x[r, k].X)}), "
                              f"Pre-Pass Risk (y={y[r, k].X:.3f}), "
                              f"Conditional Proxy (w={w[r, k].X:.3f}), "
                              f"Absolute Proxy (w_abs={w_abs[r, k].X:.3f})")
                              
                    print(f"  💥 Conditional Failure Rate (y_final): {y[r, K_r].X * 100:.1f}%")
            else:
                print("\n🔴 WLS Optimization failed or token validation timed out.")

if __name__ == "__main__":
    try:
        run_workflow_stochastic_test(gamma=0.10)
    except Exception as e:
        print(f"Execution Error: {e}")