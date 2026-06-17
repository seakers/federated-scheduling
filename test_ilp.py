import os
import cProfile
import pstats
import datetime as dt
import copy
import gurobipy as gp
from pyorbital.orbital import Orbital
from dotenv import load_dotenv

# Import your FAME ecosystem
from fame_agents_base import Satellite, ObservationRequest, InstrumentType
from fame_workflow import (
    Constraint, ConstraintClass, TemporalConstraintType, 
    ConstrainedObservationRequest, Workflow, build_workflow_graph, ilp_schedule_workflow
)

# ---------------------------------------------------------
# 2. INITIALIZE SATELLITES & TARGETS
# ---------------------------------------------------------
test_satellites = [
    Satellite("LOFT YAM-3", Orbital("YAM-3", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
    Satellite("LOFT YAM-5", Orbital("YAM-5", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
    Satellite("LOFT YAM-6", Orbital("YAM-6", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
    Satellite("LOFT YAM-7", Orbital("YAM-7", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
    Satellite("LOFT YAM-8", Orbital("YAM-8", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
    Satellite("LOFT YAM-10", Orbital("YAM-10", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
    Satellite("Ubotica CogniSat-6 HAMMER", Orbital("HAMMER", tle_file="TLEs.txt"), instruments=[InstrumentType.RGB]),
]

# Time window for the test (24 hours)
start_time = dt.datetime(2025, 10, 1, 12, 0, 0)
end_time = start_time + dt.timedelta(hours=24)

# Los Angeles Coordinates
LA_lon, LA_lat, LA_alt = -118.24, 34.05, 0.0

# ---------------------------------------------------------
# 3. BUILD THE WORKFLOW (Task 1 -> Task 2)
# ---------------------------------------------------------
print("⚙️ Building Workflow Graph...")

# In your target definition block, generate a dense conflict cluster
test_requests = []
for i in range(100):
    req = ObservationRequest(
        lon_deg=LA_lon + (i * 0.001), # Tiny offsets so they are unique targets
        lat_deg=LA_lat + (i * 0.001),
        alt_km=LA_alt,
        min_time=start_time,
        max_time=end_time,
        instrument=InstrumentType.RGB,
        request_name=f"LA_Stress_{i}",
        min_elevation_deg=20
    )
    cor = ConstrainedObservationRequest(observation_request=req, name=f"COR_{i}")
    test_requests.append(cor)

test_workflow = Workflow(constrained_observation_requests=test_requests)
wf_graph, tl_graph = build_workflow_graph(test_workflow)

print(f"📊 Graph built: {len(wf_graph.nodes())} nodes, {len(wf_graph.edges())} constraint edges.")

# ---------------------------------------------------------
# 4. PROFILE BOTH SOLVERS (SCIP vs Native Gurobi)
# ---------------------------------------------------------
dummy_feasibility_screener = lambda satellite, obs_pass: True
solver_profiles = {}

# We will run the original function (which uses SCIP)
# And then the "Gurobi Native" run using the MPS trick
run_configs = ["Gurobi (Native)","SCIP (Original)"]

for config in run_configs:
    print("\n" + "="*60)
    print(f"🚀 Benchmarking: {config}")
    print("="*60)
    
    run_wf_graph = copy.deepcopy(wf_graph)
    run_tl_graph = copy.deepcopy(tl_graph)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    if config == "SCIP (Original)":
        ilp_schedule_workflow(
            workflow_graph=run_wf_graph,
            timeline_graph=run_tl_graph,
            satellites=test_satellites,
            feasibility_screener=dummy_feasibility_screener,
            current_time=start_time,
            verbose=1,
            max_solver_time_s=30.0,
            receding_horizon_duration=dt.timedelta(hours=24),
            solver_engine="SCIP"  # Trigger the standard SCIP solve
        )
    else:
        # Load your WLS tokens so the gurobipy instance inside your workflow can authenticate
        load_dotenv()
        
        # Optional: Quick check to ensure the env loaded properly before we dive into the heavy math
        if not os.environ.get("WLSACCESSID"):
            print("⚠️ Warning: WLSACCESSID not found in environment. Gurobi may fail to authenticate.")

        # Trigger the MPS trick by passing solver_engine="GUROBI"
        ilp_schedule_workflow(
            workflow_graph=run_wf_graph,
            timeline_graph=run_tl_graph,
            satellites=test_satellites,
            feasibility_screener=dummy_feasibility_screener,
            current_time=start_time,
            verbose=1,
            max_solver_time_s=30.0,
            receding_horizon_duration=dt.timedelta(hours=24),
            solver_engine="GUROBI"  # Trigger the Gurobi export-and-solve
        )

    profiler.disable()
    solver_profiles[config] = profiler

# ---------------------------------------------------------
# 5. PRINT COMPARATIVE PROFILER RESULTS
# ---------------------------------------------------------
for name, prof_data in solver_profiles.items():
    print("\n" + "#"*60)
    print(f"⏱️ TOP 15 SLOWEST FUNCTIONS: {name}")
    print("#"*60)
    stats = pstats.Stats(prof_data).sort_stats('cumtime')
    stats.print_stats(15)