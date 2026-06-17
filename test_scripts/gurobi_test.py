import os
import sys
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv

def test_bridge():
    print("\n" + "="*60)
    print("🚀 TESTING OR-TOOLS -> GUROBI BRIDGE (WITH PATH PATCH)")
    print("="*60)
    
    load_dotenv()
    
    # 1. Dynamically locate the Gurobi DLL inside your virtual environment
    venv_base = sys.prefix
    potential_dll_path = os.path.join(venv_base, "Lib", "site-packages", "gurobipy", "gurobi110.dll")
    
    print(f"Checking for DLL at: {potential_dll_path}")
    if os.path.exists(potential_dll_path):
        print("✅ Found the Gurobi DLL inside the virtual environment!")
        # Force the environment variable OR-Tools relies on for custom paths
        os.environ["GUROBI_HOME"] = os.path.dirname(potential_dll_path)
    else:
        print("❌ Could not locate gurobi110.dll in venv. Double-check your path.")

    # Explicitly map tokens for the wrapper layer
    os.environ["WLSACCESSID"] = os.environ.get("WLSACCESSID", "")
    os.environ["WLSSECRET"] = os.environ.get("WLSSECRET", "")
    os.environ["LICENSEID"] = os.environ.get("LICENSEID", "")

    print("Attempting to initialize Gurobi via OR-Tools wrapper...")
    solver = pywraplp.Solver.CreateSolver("GUROBI_MIXED_INTEGER_PROGRAMMING")
    
    if not solver:
        print("❌ CRITICAL: OR-Tools still could not link with the backend.")
        return

    print("🎉 Solver backend created successfully! Building test model...")
    
    x = solver.IntVar(0, 1, 'x')
    y = solver.IntVar(0, 1, 'y')
    
    solver.Maximize(2 * x + y)
    solver.Add(x + y <= 1)
    
    print("📥 Invoking solver.Solve()...")
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print("\n🏆 SUCCESS! The OR-Tools Gurobi bridge is fully functional.")
        print(f"📊 Objective Value: {solver.Objective().Value()}")
    else:
        print(f"\n❌ Solver failed with status code: {status}")

if __name__ == "__main__":
    test_bridge()