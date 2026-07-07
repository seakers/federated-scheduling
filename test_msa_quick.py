"""Quick MSA test to debug hanging issue"""
import datetime as dt
import sys

print("Step 1: Basic imports...")
from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler
from fame_broker import Broker
from fame_workflow import *
print("[OK] Basic imports complete")

print("\nStep 2: Loading TLE satellites...")
from test_msa_stochastic_vs_deterministic import load_fleet_satellites_once
satellites = load_fleet_satellites_once()
print(f"[OK] Loaded {len(satellites)} satellites")

print("\nStep 3: Creating world and constellations...")
import copy
min_time = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
world = World(satellites=copy.deepcopy(satellites))
world.time = min_time
print(f"[OK] World created at {world.time}")

print("\nStep 4: Creating constellation schedulers...")
gs_planet = ConstellationGroundScheduler(
    satellites=world.satellites[:len(world.satellites)//2],
    ground_stations=[],
    world=world,
    name="Planet",
    acceptance_probability=0.7
)
gs_umbra = ConstellationGroundScheduler(
    satellites=world.satellites[len(world.satellites)//2:],
    ground_stations=[],
    world=world,
    name="Umbra",
    acceptance_probability=0.9
)
world.add_constellation(gs_planet)
world.add_constellation(gs_umbra)
print(f"[OK] Created 2 constellations: Planet ({len(gs_planet.satellites)} sats), Umbra ({len(gs_umbra.satellites)} sats)")

print("\nStep 5: Building MSA workflow...")
from test_msa_stochastic_vs_deterministic import build_msa_workflow
max_time = min_time + dt.timedelta(hours=12)
workflow = build_msa_workflow(min_time, max_time)
print(f"[OK] Workflow built with {len(workflow.constrained_observation_requests)} requests")

print("\nStep 6: Creating broker...")
broker = Broker(constellations=[gs_planet, gs_umbra], world=world, name="Test-Broker")
broker.add_workflow(workflow)
world.add_broker(broker)
print("[OK] Broker created and workflow added")

print("\nStep 7: Scheduling workflow (DETERMINISTIC)...")
sys.stdout.flush()
try:
    broker.schedule_workflow(
        current_time=world.time,
        use_ilp=True,
        use_stochastic=False,
        solver_engine="GUROBI",
        max_solver_time_s=60,  # Short timeout for testing
        tax_rate=0.05,
        update_timelines=False,
        update_requests=False
    )
    print("[OK] Scheduling complete!")
except Exception as e:
    print(f"[ERROR] Scheduling failed: {e}")
    import traceback
    traceback.print_exc()

print("\nAll steps completed successfully!")
