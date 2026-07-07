"""
MSA Ship Tracking: Stochastic vs Deterministic Benchmarking Suite
Isolates robust scheduling yield under constellation manager drop-out behaviors
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
import copy
import glob
import os
import random
from pathlib import Path

# Load environment parameters
load_dotenv()

# Import FAME components
from fame import *
from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler, ObservationStatus
from fame_broker import Broker
from fame_workflow import *
from fame_msa_utils import ShipTracker, ship_propagator, custom_ship_phenomenon_processor, propagate_distribution_from_observations

# Configuration Parameters
SIMULATION_START = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
LOOKAHEAD_HORIZON_H = 12  # REDUCED from 36 to speed up testing
FOLLOW_UP_INTERVAL_H = 3
MAX_SEARCHES_PER_INTERVAL = 3  # REDUCED from 10 to speed up testing
MAX_SOLVER_TIME_S = 300
TAX_RATE = 0.05  # Execution cost (5% of quality)
SUBMISSION_COST = 0.02  # Booking submission overhead (2% of quality)
CANCELLATION_COST = 0.03  # Cancellation penalty (3% of quality)
R_EARTH_KM = 6371

# Ship Kinematics
AVG_SPEED_KPH = 37.0 / 2.0  # ~10 knots
HEADING_VARIANCE = 0.075 / 5.0
speed_variance = 0.5
num_monte_carlo_runs = 5

# Observation Coordinates
Rotterdam = Location(3.8, 52.0, 0.0, "Rotterdam-ish")
INITIAL_TOI_POSE = Pose(
    lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, time=SIMULATION_START, 
    alt_km=Rotterdam.alt_km, heading_deg=0.0, speed_kph=AVG_SPEED_KPH, name="Rotterdam Initial"
)


def load_fleet_satellites_once() -> list[Satellite]:
    """Parses real LEO orbital parameters from local text catalogs once to clear loop friction."""
    from pyorbital.orbital import Orbital

    tle_files = glob.glob("tles/all_tles_*.txt")
    if not tle_files:
        raise FileNotFoundError("No text TLE files found in the tles/ folder context.")
    tle_file_txt = sorted(tle_files)[-1]
    
    skysat_names = [f"SKYSAT C{i}" for i in range(1, 13)] + ["UMBRA-07", "UMBRA-09", "UMBRA-10", "UMBRA-11"]
    swaths_at_nadir = {name: 5.9 if "SKYSAT" in name else 8.0 for name in skysat_names}
    
    satellites = []
    for name in skysat_names:
        try:
            orbit = Orbital(name, tle_file=tle_file_txt)
            sat = Satellite(name, orbit, instruments=[InstrumentType.RGB, InstrumentType.SAR], has_continuous_isl_to_ground=True)
            
            # Initialize Field-of-View geometry profiles
            _semi_major = sat.orbit.orbit_elements.semi_major_axis * pyorbital.orbital.A
            _altitude = _semi_major - pyorbital.orbital.A
            _fov = 2 * np.atan2(swaths_at_nadir[name] / 2, _altitude)
            sat.instrument_fov_rad = {it: _fov for it in sat.instruments}
            
            satellites.append(sat)
        except Exception:
            continue
            
    print(f"[Init] Successfully cached {len(satellites)} operational platforms for comparison.\n")
    return satellites


def generate_real_ship_trajectory(max_time: dt.datetime) -> list[Phenomenon]:
    """Generates the true underlying maritime trajectory (Ground Truth physics target)."""
    random.seed(4)  # Lock tracking context seed
    phenomenon_update_frequency = dt.timedelta(seconds=1800)
    
    trajectory = [
        Phenomenon(
            lon_deg=Rotterdam.lon_deg, lat_deg=Rotterdam.lat_deg, alt_km=Rotterdam.alt_km,
            start_time=SIMULATION_START, end_time=SIMULATION_START + phenomenon_update_frequency,
            name=Rotterdam.name, heading_deg=0.0, speed_kph=AVG_SPEED_KPH
        )
    ]
    
    while trajectory[-1].end_time < max_time:
        new_heading = trajectory[-1].heading_deg * np.pi / 180.0 + random.normalvariate() * HEADING_VARIANCE
        new_speed = AVG_SPEED_KPH + random.normalvariate() * speed_variance
        
        dy = new_speed * np.cos(new_heading) * (phenomenon_update_frequency.total_seconds() / 3600.0)
        dx = new_speed * np.sin(new_heading) * (phenomenon_update_frequency.total_seconds() / 3600.0)
        
        dlat = dy / R_earth_km
        dlon = dx / (R_earth_km * np.cos(trajectory[-1].lat_deg * np.pi / 180.0))
        
        trajectory.append(
            Phenomenon(
                lon_deg=trajectory[-1].lon_deg + dlon * 180.0 / np.pi,
                lat_deg=trajectory[-1].lat_deg + dlat * 180.0 / np.pi,
                alt_km=Rotterdam.alt_km,
                start_time=trajectory[-1].end_time,
                end_time=trajectory[-1].end_time + phenomenon_update_frequency,
                heading_deg=new_heading * 180.0 / np.pi,
                speed_kph=new_speed,
                name=f"Ship_{trajectory[-1].end_time.strftime('%H%M')}"
            )
        )
    return trajectory


def build_msa_workflow(min_time: dt.datetime, max_time: dt.datetime) -> Workflow:
    """Assembles the conditional maritime tracking dependency pipeline."""
    
    ship_location_timeline = Timeline(
        name="Ship loc. known?", initial_time=min_time, initial_value=1.0, 
        initial_rate=-1.0 / 10800, min_value=-100.0, max_value=100.0
    )
    
    init_request = ObservationRequest(
        Rotterdam.lon_deg, Rotterdam.lat_deg, min_time=min_time, max_time=min_time + dt.timedelta(hours=3),
        alt_km=Rotterdam.alt_km, request_name="Initial RGB request in Rotterdam", min_elevation_deg=45.0, instrument=InstrumentType.RGB
    )
    
    root_task = ConstrainedObservationRequest(
        name="Initial observation", observation_request=init_request, is_mandatory=True,
        timeline_impacts=[TaskTimelineImpact(timeline=ship_location_timeline, time=TaskImpactTime.POST, type=ImpactType.ADDITION, value=1.0)],
        rewarder=lambda opt: 50.0 + abs(90.0 - opt.look_angle_dec_deg) / 90.0,
        success_declarer=lambda dp: len(dp) > 0
    )
    
    workflow_reqs = [root_task]
    
    for h_ix in range(FOLLOW_UP_INTERVAL_H, LOOKAHEAD_HORIZON_H, FOLLOW_UP_INTERVAL_H):
        num_ancestors = len(workflow_reqs)
        
        base_constraints = [
            Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET, root_task, {'offset': dt.timedelta(hours=h_ix)}),
            Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_BEFORE_OFFSET, root_task, {'offset': dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H)})
        ]
        for ancestor in workflow_reqs[:num_ancestors]:
            base_constraints.append(Constraint(ConstraintClass.SUCCESS, SuccessConstraintType.WAIT_FOR_COMPLETION_IF_FEASIBLE, ancestor))
            
        img_request = ObservationRequest(
            Rotterdam.lon_deg, Rotterdam.lat_deg, min_time=min_time + dt.timedelta(hours=h_ix),
            max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H), alt_km=Rotterdam.alt_km,
            request_name=f"Follow-up imaging {h_ix}h", min_elevation_deg=45.0, instrument=InstrumentType.RGB
        )
        
        img_task = ConstrainedObservationRequest(
            name=f"Follow-up obs. {h_ix}h", observation_request=img_request, is_mandatory=False,
            task_constraints=copy.deepcopy(base_constraints),
            timeline_constraints=[TaskTimelineConstraint(timeline=ship_location_timeline, time=TaskImpactTime.PRE, type=TimelineConstraintType.GREATER_OR_EQUAL, value=0.0)],
            timeline_impacts=[TaskTimelineImpact(timeline=ship_location_timeline, time=TaskImpactTime.POST, type=ImpactType.ADDITION, value=1.0)],
            rewarder=lambda opt: 50.0 + abs(90.0 - opt.look_angle_dec_deg) / 90.0, max_num_instances=1, request_group=f"Follow-up obs. {h_ix}h"
        )
        
        for search_ix in range(MAX_SEARCHES_PER_INTERVAL):
            search_request = ObservationRequest(
                Rotterdam.lon_deg, Rotterdam.lat_deg, min_time=min_time + dt.timedelta(hours=h_ix),
                max_time=min_time + dt.timedelta(hours=h_ix + FOLLOW_UP_INTERVAL_H), alt_km=Rotterdam.alt_km,
                request_name=f"Follow-up search {h_ix}h.{search_ix}", min_elevation_deg=45.0, instrument=InstrumentType.SAR
            )
            
            search_task = ConstrainedObservationRequest(
                name=f"Follow-up search {h_ix}h.{search_ix}", observation_request=search_request, is_mandatory=False,
                task_constraints=copy.deepcopy(base_constraints),
                timeline_constraints=[TaskTimelineConstraint(timeline=ship_location_timeline, time=TaskImpactTime.PRE, type=TimelineConstraintType.LESSER_OR_EQUAL, value=0.0)],
                timeline_impacts=[TaskTimelineImpact(timeline=ship_location_timeline, time=TaskImpactTime.POST, type=ImpactType.ADDITION, value=1.0)],
                rewarder=lambda opt: abs(90.0 - opt.look_angle_dec_deg) / 90.0, max_num_instances=1, request_group=f"Follow-up search {h_ix}h"
            )
            workflow_reqs.append(search_task)
            
        workflow_reqs.append(img_task)
        
    # Timeline and request updaters - using simple pass-through since MSA-specific updaters aren't defined
    def internal_timeline_wrapper(t, reqs, tls):
        # Simple timeline updater - timelines evolve automatically via their rates
        return tls

    def internal_request_wrapper(t, reqs, tls):
        # Simple request updater - requests don't need dynamic updating for this test
        return None

    return Workflow(
        constrained_observation_requests=workflow_reqs, timelines=[ship_location_timeline],
        timeline_updater=internal_timeline_wrapper, request_updater=internal_request_wrapper
    )


def run_msa_comparison_suite(num_runs=2):
    """Executes Monte Carlo evaluation runs across parallel world configurations."""
    cached_satellites = load_fleet_satellites_once()
    
    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=LOOKAHEAD_HORIZON_H)
    
    results = {
        'deterministic_scheduled': [], 'deterministic_attempts': [], 'deterministic_rejections': [], 'deterministic_utility': [],
        'stochastic_log_scheduled': [], 'stochastic_log_attempts': [], 'stochastic_log_rejections': [], 'stochastic_log_utility': [],
        'stochastic_nc_scheduled': [], 'stochastic_nc_attempts': [], 'stochastic_nc_rejections': [], 'stochastic_nc_utility': []
    }
    
    # === TWO-STAGE PROBABILITY MODEL ===
    def acceptance_prob_function(constrained_request, satellite, obs_pass):
        """Probability that constellation ACCEPTS the booking request."""
        if any(p in satellite.name for p in ["Planet", "SKYSAT"]):
            return 0.7
        return 0.9

    def execution_prob_function(constrained_request, satellite, obs_pass):
        """Probability that an ACCEPTED booking executes successfully."""
        # Execution success depends on look angle (better geometry = higher success)
        look_angle = abs(90.0 - obs_pass.highest.look_angle_dec_deg)
        execution_prob = 0.95 - (look_angle / 90.0) * 0.20
        return max(0.7, min(0.99, execution_prob))

    def evaluate_executed_yield(workflow_graph, broker, tax_rate, run_id, pipeline_label):
        """
        Evaluates exact realized metrics with rigorous text profiling.
        KEY METRICS FOR STOCHASTIC VS DETERMINISTIC COMPARISON:
        - Total Dispatch Attempts: how many times broker tried to dispatch (including rejections/retries)
        - Rejections: how many attempts were rejected by constellation managers
        - Final Scheduled: how many tasks in final schedule (after all rescheduling converged)
        """
        # Count from the final workflow graph state
        scheduled = [n for n in workflow_graph.nodes() if n.scheduled and n.feasible]
        completed = [n for n in workflow_graph.nodes() if n.completed]
        successful = [n for n in workflow_graph.nodes() if n.completed and getattr(n, 'successful_execution', False)]

        # === CRITICAL METRICS FROM BROKER REQUEST LOG ===
        # Each row in broker._requests is a dispatch attempt
        num_total_attempts = len(broker._requests)

        # Count rejections - these cause expensive rescheduling
        num_rejected = len(broker._requests[broker._requests['status'] == ObservationStatus.CONSTELLATION_REJECTED])

        # Count accepted - these are the successful dispatch attempts
        num_accepted = len(broker._requests[
            broker._requests['status'].isin([ObservationStatus.SCHEDULED, ObservationStatus.DATA_RECEIVED])
        ])

        # Unique tasks attempted (some tasks may be tried multiple times after rejection)
        num_unique_attempted = broker._requests['request'].nunique() if len(broker._requests) > 0 else 0

        print(f"\n   ╔══════════════════════════════════════════════════════════════")
        print(f"   ║ {pipeline_label} | Run {run_id+1}")
        print(f"   ╠══════════════════════════════════════════════════════════════")
        print(f"   ║ DISPATCH EFFICIENCY METRICS:")
        print(f"   ║   Total Dispatch Attempts:  {num_total_attempts:3d}  (includes retries after rejection)")
        print(f"   ║   └─ Accepted by Constellations: {num_accepted:3d}")
        print(f"   ║   └─ REJECTED by Constellations: {num_rejected:3d}  ← KEY: Stochastic should reduce this!")
        print(f"   ║   Unique Tasks Attempted:   {num_unique_attempted:3d}")
        print(f"   ║   Final Schedule Size:      {len(scheduled):3d}  (after all rescheduling)")
        print(f"   ║")
        print(f"   ║ EXECUTION METRICS:")
        print(f"   ║   Completed:    {len(completed):3d}")
        print(f"   ║   Successful:   {len(successful):3d}")

        # Compute financial metrics
        total_realized_quality = 0.0
        total_execution_cost = 0.0

        for task in workflow_graph.nodes():
            # Cost: Pay tax for ONLY the pass that was actually dispatched and executed
            # task.observation_opportunity (singular) = the specific pass that was dispatched
            if task.dispatched and task.observation_opportunity is not None:
                executed_quality = task.rewarder(task.observation_opportunity)
                task_cost = tax_rate * executed_quality
                total_execution_cost += task_cost

            # Quality: Earn credit for successfully completed observations
            if task.completed and getattr(task, 'successful_execution', False):
                if hasattr(task, 'observation_opportunity') and task.observation_opportunity is not None:
                    earned_q = task.rewarder(task.observation_opportunity)
                    total_realized_quality += earned_q

        net_utility = total_realized_quality - total_execution_cost
        rejection_rate = (num_rejected / num_total_attempts * 100) if num_total_attempts > 0 else 0

        print(f"   ║")
        print(f"   ║ FINANCIAL PERFORMANCE:")
        print(f"   ║   Quality Earned:  {total_realized_quality:7.2f}")
        print(f"   ║   Costs Paid:     {total_execution_cost:7.2f}")
        print(f"   ║   Net Utility:    {net_utility:7.2f}")
        print(f"   ║")
        print(f"   ║ REJECTION RATE: {rejection_rate:5.1f}%  ← Stochastic should be lower!")
        print(f"   ╚══════════════════════════════════════════════════════════════\n")

        # Return: scheduled, total attempts (shows rescheduling overhead), rejections, net utility
        return len(scheduled), num_total_attempts, num_rejected, net_utility

    def run_unbounded_simulation(world, max_safety_limit=40000):
        """Ticks the discrete event loop until the queue dries out to capture every window."""
        ticks = 0
        while True:
            retcode = world.tick(print_forbidden_prefixes=["Downlink", "End of downlink", "Unlock uplink", "Unlock satellite after obs"])
            ticks += 1
            if retcode == 0:
                break
            if ticks >= max_safety_limit:
                print(f"      [Warning] Simulation safety cut-off invoked at {max_safety_limit} ticks.")
                break
        print(f"      [Sim Loop Concluded] Total Ticks: {ticks}, Final Clock State: {world.time}")

    for run in range(num_monte_carlo_runs):
        print(f"\n" + "="*70)
        print(f"LAUNCHING MONTE CARLO HORIZON RUN {run + 1}/{num_monte_carlo_runs}")
        print("="*70)

        print("  [1/10] Generating ship trajectory...")
        true_phenomena = generate_real_ship_trajectory(max_time)
        print(f"  [OK] Generated {len(true_phenomena)} trajectory points")

        print("  [2/10] Building MSA workflow...")
        master_workflow = build_msa_workflow(min_time, max_time)
        print(f"  [OK] Workflow built with {len(master_workflow.constrained_observation_requests)} requests")
        
        # -----------------------------------------------------------------
        # EVALUATION 1: STANDARD DETERMINISTIC ILP PIPELINE
        # -----------------------------------------------------------------
        print("\n  [3/10] Running Deterministic ILP Pipeline...")
        print("      Creating world with satellites and phenomena...")
        world_det = World(satellites=copy.deepcopy(cached_satellites), phenomena=copy.deepcopy(true_phenomena))
        world_det.time = min_time
        print(f"      World created with {len(world_det.satellites)} satellites, {len(world_det.phenomena)} phenomena")
        
        print("      Creating constellation schedulers...")
        gs_planet = ConstellationGroundScheduler(satellites=world_det.satellites[:len(world_det.satellites)//2], ground_stations=[], world=world_det, name="Planet", acceptance_probability=0.7)
        print(f"      Planet constellation: {len(gs_planet.satellites)} satellites")
        gs_umbra = ConstellationGroundScheduler(satellites=world_det.satellites[len(world_det.satellites)//2:], ground_stations=[], world=world_det, name="Umbra", acceptance_probability=0.9)
        print(f"      Umbra constellation: {len(gs_umbra.satellites)} satellites")
        world_det.add_constellation(gs_planet); world_det.add_constellation(gs_umbra)
        print("      Constellations added to world")
        
        print("      Creating broker and adding workflow...")
        broker_det = Broker(constellations=[gs_planet, gs_umbra], world=world_det, name="Broker-Det")
        print("      Broker created, adding workflow...")
        broker_det.add_workflow(copy.deepcopy(master_workflow))
        print("      Workflow added, adding broker to world...")
        world_det.add_broker(broker_det)
        print("      [OK] Setup complete, starting scheduling...")
        
        try:
            print(f"      [Det] Starting ILP scheduling at {world_det.time}...")
            broker_det.schedule_workflow(
                current_time=world_det.time, use_ilp=True, use_stochastic=False,
                solver_engine="GUROBI", max_solver_time_s=MAX_SOLVER_TIME_S,
                tax_rate=TAX_RATE, update_timelines=False, update_requests=False
            )
            print(f"      [Det] ILP scheduling complete, running simulation...")
            run_unbounded_simulation(world_det)
            print(f"      [Det] Simulation complete, computing metrics...")
            s, attempts, rejections, u = evaluate_executed_yield(broker_det._workflow_graph, broker_det, TAX_RATE, run, "Deterministic")
            results['deterministic_scheduled'].append(s)
            results['deterministic_attempts'].append(attempts)
            results['deterministic_rejections'].append(rejections)
            results['deterministic_utility'].append(u)
        except Exception as e:
            print(f"    [Error Det]: {e}")
            import traceback
            traceback.print_exc()

        # -----------------------------------------------------------------
        # EVALUATION 2: ROBUST STOCHASTIC MILP (LOG-LINEARIZED)
        # -----------------------------------------------------------------
        print("\n🛰️  Running stochastic log-linearized optimization track...")
        world_log = World(satellites=copy.deepcopy(cached_satellites), phenomena=copy.deepcopy(true_phenomena))
        world_log.time = min_time
        gs_planet2 = ConstellationGroundScheduler(satellites=world_log.satellites[:len(world_log.satellites)//2], ground_stations=[], world=world_log, name="Planet", acceptance_probability=0.7)
        gs_umbra2 = ConstellationGroundScheduler(satellites=world_log.satellites[len(world_log.satellites)//2:], ground_stations=[], world=world_log, name="Umbra", acceptance_probability=0.9)
        world_log.add_constellation(gs_planet2); world_log.add_constellation(gs_umbra2)
        
        broker_log = Broker(constellations=[gs_planet2, gs_umbra2], world=world_log, name="Broker-Log")
        broker_log.add_workflow(copy.deepcopy(master_workflow))
        world_log.add_broker(broker_log)
        
        try:
            print(f"      [Stoch-Log] Starting stochastic log-linearized MILP scheduling at {world_log.time}...")
            broker_log.schedule_workflow(
                current_time=world_log.time, use_ilp=True, use_stochastic=True,
                stochastic_formulation="log_linearized",
                # NEW: Two-stage probability model
                acceptance_probability_function=acceptance_prob_function,
                execution_probability_function=execution_prob_function,
                # NEW: Cost structure
                submission_cost_rate=SUBMISSION_COST,
                cancellation_cost_rate=CANCELLATION_COST,
                tax_rate=TAX_RATE,
                max_solver_time_s=MAX_SOLVER_TIME_S, solver_engine="GUROBI",
                update_timelines=False, update_requests=False
            )
            print(f"      [Stoch-Log] MILP scheduling complete, running simulation...")
            run_unbounded_simulation(world_log)
            print(f"      [Stoch-Log] Simulation complete, computing metrics...")
            s, attempts, rejections, u = evaluate_executed_yield(broker_log._workflow_graph, broker_log, TAX_RATE, run, "Stochastic_Log")
            results['stochastic_log_scheduled'].append(s)
            results['stochastic_log_attempts'].append(attempts)
            results['stochastic_log_rejections'].append(rejections)
            results['stochastic_log_utility'].append(u)
        except Exception as e:
            print(f"    [Error Log-MIP]: {e}")
            import traceback
            traceback.print_exc()

        # -----------------------------------------------------------------
        # EVALUATION 3: ROBUST STOCHASTIC MILP (NON-CONVEX SMOOTH)
        # -----------------------------------------------------------------
        # DISABLED: Non-convex formulation has pickle issues with Gurobi objects during simulation
        # The log-linearized formulation is the preferred method anyway
        print("\n  [SKIP] Non-convex formulation disabled (Gurobi pickle issues)")

    # =====================================================================
    # PERFORMANCE METRICS EXPORT LAYER
    # =====================================================================
    print("\n" + "="*70)
    print("MARITIME DOMAIN AWARENESS FLIGHT PERFORMANCE REPORT")
    print("="*70)

    summary_data = []
    for track in ['deterministic', 'stochastic_log', 'stochastic_nc']:
        avg_scheduled = np.mean(results[f'{track}_scheduled'])
        avg_attempts = np.mean(results[f'{track}_attempts'])
        avg_rejections = np.mean(results[f'{track}_rejections'])
        avg_utility = np.mean(results[f'{track}_utility'])
        rejection_rate = (avg_rejections / avg_attempts * 100) if avg_attempts > 0 else 0

        print(f"\n{track.upper()}:")
        print(f"  Final Scheduled:     {avg_scheduled:.1f} ± {np.std(results[f'{track}_scheduled']):.1f}")
        print(f"  Total Attempts:      {avg_attempts:.1f} ± {np.std(results[f'{track}_attempts']):.1f}")
        print(f"  Rejections:          {avg_rejections:.1f} ± {np.std(results[f'{track}_rejections']):.1f}")
        print(f"  Rejection Rate:      {rejection_rate:.1f}%  ← KEY METRIC")
        print(f"  Net Utility:         {avg_utility:.2f}")

        summary_data.append({
            'Scheduler': track,
            'Scheduled_Mean': avg_scheduled,
            'Scheduled_Std': np.std(results[f'{track}_scheduled']),
            'Attempts_Mean': avg_attempts,
            'Attempts_Std': np.std(results[f'{track}_attempts']),
            'Rejections_Mean': avg_rejections,
            'Rejections_Std': np.std(results[f'{track}_rejections']),
            'Rejection_Rate_Pct': rejection_rate,
            'Utility_Mean': avg_utility,
            'Utility_Std': np.std(results[f'{track}_utility']),
        })

    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv("msa_stochastic_comparison_metrics.csv", index=False)

    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("msa_stochastic_comparison_summary.csv", index=False)
    print("\n[Summary] Saved to msa_stochastic_comparison_summary.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    df[['deterministic_scheduled', 'stochastic_log_scheduled', 'stochastic_nc_scheduled']].boxplot(ax=ax1)
    ax1.set_title("Total Task Commits Scheduled")
    ax1.set_ylabel("Count")
    
    df[['deterministic_utility', 'stochastic_log_utility', 'stochastic_nc_utility']].boxplot(ax=ax2)
    ax2.set_title("True Realized Mission Yield (Utility)")
    ax2.set_ylabel("Yield Performance Bounds")
    
    plt.tight_layout()
    plt.savefig("msa_stochastic_comparison.png", dpi=150)
    print("\n[OK] Benchmarking metrics successfully compiled to project data directories.")


if __name__ == "__main__":
    run_msa_comparison_suite(num_runs=num_monte_carlo_runs)