"""Quick debug test for stochastic scheduler infeasibility"""

import datetime as dt
import copy
from dotenv import load_dotenv

load_dotenv()

from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler
from fame_broker import Broker
from fame_workflow import *

# Use the real setup
from volcano_stochastic_comparison_real import (
    load_satellites_from_tle,
    create_volcano_workflow,
    VOLCANO_LOCATIONS,
    SIMULATION_START,
    LOOKAHEAD_HORIZON_H,
    MAX_SOLVER_TIME_S
)

print("Creating world with real satellites...")
world = World(SIMULATION_START)
satellites = load_satellites_from_tle()

ground_stations = [
    Location(lon_deg=-79.55, lat_deg=8.9833, alt_km=0, name="KSAT Panama"),
    Location(lon_deg=22.62216, lat_deg=37.84604, alt_km=0, name="KSAT Nemea"),
]

# Create single constellation
constellation = ConstellationGroundScheduler(
    satellites=satellites,
    ground_stations=ground_stations,
    world=world,
    name="TestConst",
    acceptance_probability=0.9
)

min_time = SIMULATION_START
max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)

workflow = create_volcano_workflow(VOLCANO_LOCATIONS, min_time, max_time)

def success_prob_function(constrained_request, satellite, obs_pass):
    return 0.9

print(f"\nWorkflow has {len(workflow.constrained_observation_requests)} tasks")
print(f"Workflow has {len(workflow.timelines)} timelines")

# Test stochastic scheduler
print("\n" + "="*70)
print("TESTING STOCHASTIC NON-CONVEX")
print("="*70)

broker = Broker(constellations=[constellation], world=world, name="TestBroker")
broker.add_workflow(copy.deepcopy(workflow))

broker.schedule_workflow(
    current_time=world.time,
    use_ilp=True,
    use_stochastic=True,
    stochastic_formulation="non_convex",
    success_probability_function=success_prob_function,
    max_solver_time_s=MAX_SOLVER_TIME_S,
    solver_engine="GUROBI",
    update_timelines=False,
    update_requests=False
)

scheduled = [n for n in broker._workflow_graph.nodes() if n.scheduled]
print(f"\nFinal: {len(scheduled)} tasks scheduled")
for task in scheduled:
    print(f"  - {task.observation_request.name}: scheduled={task.scheduled}, feasible={task.feasible}")
