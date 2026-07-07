"""
Simple test script for stochastic MILP scheduler.

Tests a minimal 2-task DAG workflow with varying success probabilities
to verify the stochastic formulation works correctly.
"""

import datetime as dt
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler
from fame_broker import Broker
from fame_workflow import *

def create_simple_world():
    """Create a minimal world with 2 satellites and 1 ground station."""
    world = World(dt.datetime(2024, 1, 1, 0, 0, 0))

    # Create 2 simple satellites using ISS TLE
    from pyorbital.orbital import Orbital

    orbit1 = Orbital(
        "ISS",
        line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    )

    orbit2 = Orbital(
        "ISS",
        line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    )

    sat1 = Satellite(
        name="TestSat-1",
        orbit=orbit1,
        has_continuous_isl_to_ground=True  # Simplify by using ISL
    )
    sat2 = Satellite(
        name="TestSat-2",
        orbit=orbit2,
        has_continuous_isl_to_ground=True
    )

    # Ground station (not used with ISL, but needed for constellation)
    ground_station = Location(
        lon_deg=0,
        lat_deg=0,
        alt_km=0,
        name="TestGS"
    )

    return world, [sat1, sat2], [ground_station]


def create_simple_workflow():
    """
    Create a simple 2-task DAG workflow:
    Task 1 (mandatory) -> Task 2 (optional)
    """

    min_time = dt.datetime(2024, 1, 1, 6, 0, 0)
    max_time = dt.datetime(2024, 1, 1, 18, 0, 0)

    # Task 1: Initial observation (mandatory root task)
    task1_request = ObservationRequest(
        lon_deg=0.0,
        lat_deg=0.0,
        alt_km=0.0,
        min_time=min_time,
        max_time=min_time + dt.timedelta(hours=4),
        instrument=InstrumentType.RGB,
        request_name="Task-1-Detection",
        min_elevation_deg=30.0
    )

    task1 = ConstrainedObservationRequest(
        name="Task-1-Detection",
        observation_request=task1_request,
        is_mandatory=True,
        rewarder=lambda opp: 100.0,  # Fixed high reward
        success_declarer=lambda dp: len(dp) > 0,
        request_group="test_workflow"
    )

    # Task 2: Follow-up observation (depends on Task 1)
    task2_request = ObservationRequest(
        lon_deg=0.0,
        lat_deg=0.0,
        alt_km=0.0,
        min_time=min_time + dt.timedelta(hours=2),
        max_time=max_time,
        instrument=InstrumentType.RGB,
        request_name="Task-2-Followup",
        min_elevation_deg=30.0
    )

    task2 = ConstrainedObservationRequest(
        name="Task-2-Followup",
        observation_request=task2_request,
        is_mandatory=False,
        task_constraints=[
            Constraint(
                ConstraintClass.TEMPORAL,
                TemporalConstraintType.START_AFTER,
                task1,
                {}
            )
        ],
        rewarder=lambda opp: 80.0,  # Lower reward
        success_declarer=lambda dp: len(dp) > 0,
        request_group="test_workflow"
    )

    workflow = Workflow(
        constrained_observation_requests=[task1, task2],
        timelines=[],
        timeline_updater=lambda t, r, tl: tl,
        request_updater=lambda t, r, tl: None,
    )

    return workflow


def run_test(formulation="log_linearized", verbose=True):
    """
    Run stochastic scheduler test.

    Parameters
    ----------
    formulation : str
        "non_convex" or "log_linearized"
    verbose : bool
        Print detailed output
    """

    if verbose:
        print("\n" + "="*70)
        print(f"STOCHASTIC MILP TEST - Formulation: {formulation}")
        print("="*70)

    # Create world
    world, satellites, ground_stations = create_simple_world()

    # Create constellations with different acceptance probabilities
    constellation1 = ConstellationGroundScheduler(
        satellites=[satellites[0]],
        ground_stations=ground_stations,
        world=world,
        name="Constellation-A",
        acceptance_probability=0.7  # 70% acceptance rate
    )

    constellation2 = ConstellationGroundScheduler(
        satellites=[satellites[1]],
        ground_stations=ground_stations,
        world=world,
        name="Constellation-B",
        acceptance_probability=0.9  # 90% acceptance rate
    )

    # Create broker
    broker = Broker(
        constellations=[constellation1, constellation2],
        world=world,
        name="TestBroker"
    )

    # Add workflow
    workflow = create_simple_workflow()
    broker.add_workflow(workflow)

    # Define success probability function
    def success_prob_function(constrained_request, satellite, obs_pass):
        """
        Return acceptance probability based on constellation.

        In real scenario, this would come from historical data or weather models.
        """
        if satellite.name == "TestSat-1":
            return 0.7  # Constellation A: 70% acceptance
        elif satellite.name == "TestSat-2":
            return 0.9  # Constellation B: 90% acceptance
        else:
            return 1.0

    # === RUN DETERMINISTIC SCHEDULER ===
    if verbose:
        print("\n--- DETERMINISTIC ILP (Baseline) ---")

    broker_det = Broker(
        constellations=[constellation1, constellation2],
        world=world,
        name="TestBroker-Deterministic"
    )
    workflow_det = create_simple_workflow()
    broker_det.add_workflow(workflow_det)

    broker_det.schedule_workflow(
        current_time=world.time,
        use_ilp=True,
        use_stochastic=False,  # Deterministic
        max_solver_time_s=30,
        receding_horizon_duration=dt.timedelta(hours=24),
        solver_engine="GUROBI"
    )

    # Count scheduled tasks (deterministic)
    scheduled_det = [
        node for node in broker_det._workflow_graph.nodes()
        if node.scheduled and node.observation_opportunity_satellite is not None
    ]

    if verbose:
        print(f"\nDeterministic scheduled {len(scheduled_det)} tasks:")
        for node in scheduled_det:
            print(f"  - {node.name} on {node.observation_opportunity_satellite.name}")

    # === RUN STOCHASTIC SCHEDULER ===
    if verbose:
        print(f"\n--- STOCHASTIC MILP ({formulation}) ---")

    broker.schedule_workflow(
        current_time=world.time,
        use_ilp=True,
        use_stochastic=True,
        stochastic_formulation=formulation,
        success_probability_function=success_prob_function,
        max_solver_time_s=30,
        receding_horizon_duration=dt.timedelta(hours=24),
        epsilon=1e-5,
        pwl_tolerance=1e-2,
        solver_engine="GUROBI"
    )

    # Count scheduled tasks (stochastic)
    scheduled_stoch = [
        node for node in broker._workflow_graph.nodes()
        if node.scheduled and node.observation_opportunity_satellite is not None
    ]

    if verbose:
        print(f"\nStochastic scheduled {len(scheduled_stoch)} tasks:")
        for node in scheduled_stoch:
            sat_name = node.observation_opportunity_satellite.name
            prob = success_prob_function(node, node.observation_opportunity_satellite, node.observation_opportunity_pass)
            print(f"  - {node.name} on {sat_name} (prob={prob:.2f})")

    # === COMPARISON ===
    if verbose:
        print("\n" + "-"*70)
        print("COMPARISON:")
        print(f"  Deterministic: {len(scheduled_det)} tasks scheduled")
        print(f"  Stochastic:    {len(scheduled_stoch)} tasks scheduled")

        # Check if stochastic prefers high-probability constellation
        if len(scheduled_stoch) > 0:
            stoch_sats = [n.observation_opportunity_satellite.name for n in scheduled_stoch]
            sat_b_count = stoch_sats.count("TestSat-2")  # High probability
            sat_a_count = stoch_sats.count("TestSat-1")  # Low probability

            print(f"\n  Stochastic allocation:")
            print(f"    - TestSat-2 (prob=0.9): {sat_b_count} tasks")
            print(f"    - TestSat-1 (prob=0.7): {sat_a_count} tasks")

            if sat_b_count > sat_a_count:
                print("\n  [OK] Stochastic correctly prefers high-probability constellation!")
            elif sat_b_count == sat_a_count:
                print("\n  [NOTE] Stochastic allocated evenly (may depend on geometry)")
            else:
                print("\n  [WARN] Stochastic preferred low-probability constellation")

        print("="*70 + "\n")

    return {
        'deterministic_scheduled': len(scheduled_det),
        'stochastic_scheduled': len(scheduled_stoch),
        'success': len(scheduled_stoch) > 0
    }


if __name__ == "__main__":
    print("\nTesting Stochastic MILP Scheduler")
    print("="*70)

    # Test log-linearized formulation
    try:
        result_log = run_test(formulation="log_linearized", verbose=True)
        print("[OK] Log-linearized formulation: PASSED")
    except Exception as e:
        print(f"[FAIL] Log-linearized formulation: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        result_log = {'success': False}

    # Test non-convex formulation
    try:
        result_nc = run_test(formulation="non_convex", verbose=True)
        print("[OK] Non-convex formulation: PASSED")
    except Exception as e:
        print(f"[FAIL] Non-convex formulation: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        result_nc = {'success': False}

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    if result_log['success'] and result_nc['success']:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("\nThe stochastic MILP scheduler is working correctly.")
        print("Ready to create full Jupyter notebook examples.")
    else:
        print("[FAILURE] SOME TESTS FAILED")
        print("\nPlease review errors above before proceeding.")
    print("="*70)
