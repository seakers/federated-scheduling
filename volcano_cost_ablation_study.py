"""
Cost Ablation Study for Stochastic vs Deterministic Scheduler Comparison

This script runs the volcano workflow comparison across a grid of cost parameter values
to understand how submission_cost and execution_cost affect the relative performance
of stochastic vs deterministic schedulers.

Key Questions:
1. How sensitive is stochastic scheduler performance to cost parameters?
2. What cost values make stochastic outperform deterministic?
3. Is there a sweet spot where stochastic beats deterministic?
"""

import datetime as dt
import numpy as np
import pandas as pd
import os
import json
from itertools import product

# Import the main comparison function
from volcano_stochastic_comparison_real import (
    load_volcano_locations_from_database,
    load_satellites_once,
    create_world_and_constellations,
    create_volcano_workflow,
    # acceptance_prob_function,
    # execution_prob_function,
    SIMULATION_START,
    lookahead_horizon_h,
    MAX_SOLVER_TIME_S,
) 
from fame_constellation_scheduler import ObservationStatus
import random
def acceptance_prob_function(constrained_request, satellite, obs_pass):
    """
    Probability that constellation ACCEPTS the booking request.
    This reflects constellation's internal capacity, conflicts, and scheduling flexibility.

    Uses acceptance probabilities matching the 8 constellation structure from the notebook.
    """
    # Planet constellation (busiest - largest constellation)
    if any(x in satellite.name.upper() for x in ["SKYSAT", "PELICAN", "TANAGER"]):
        return 0.70
    # Umbra
    elif "UMBRA" in satellite.name.upper():
        return 0.85
    # Capella
    elif "CAPELLA" in satellite.name.upper() or "ACADIA" in satellite.name.upper():
        return 0.90
    # LOFT
    elif "LOFT" in satellite.name.upper() or "YAM" in satellite.name.upper():
        return 0.92
    # Ubotica
    elif "UBOTICA" in satellite.name.upper() or "HAMMER" in satellite.name.upper() or "ACCENTURE" in satellite.name.upper():
        return 0.93
    # Mission Control
    elif "PERSISTENCE" in satellite.name.upper() or "LEMUR" in satellite.name.upper():
        return 0.94
    # Aerospace Corp
    elif "AEROCUBE" in satellite.name.upper():
        return 0.95
    # ICEYE (least busy)
    elif "ICEYE" in satellite.name.upper():
        return 0.96
    # Default fallback
    return 0.85

def execution_prob_function(constrained_request, satellite, obs_pass):
    """
    Probability that an ACCEPTED booking executes successfully.
    For volcano monitoring: depends on cloud cover, atmospheric conditions, etc.
    Better look angles (closer to nadir) have higher execution success.
    """
    # Simple model: execution success decreases with off-nadir angle
    look_angle = abs(90.0 - obs_pass.highest.look_angle_dec_deg)

    # At nadir (look_angle=0): 95% success
    # At 45° off-nadir: ~85% success
    # At 60° off-nadir: ~75% success
    execution_prob = 0.95 - (look_angle / 90.0) * 0.20

    return max(0.7, min(0.99, execution_prob))


def run_single_comparison(submission_cost, exec_cost, run_seed=42, results_dir=None, plots_dir=None):
    """
    Run a single deterministic vs stochastic comparison with given cost parameters.

    Returns metrics dictionary with keys:
    - det_scheduled, det_quality, det_cost, det_utility
    - sto_scheduled, sto_quality, sto_cost, sto_utility

    Args:
        submission_cost: Cost rate for submission
        exec_cost: Cost rate for execution
        run_seed: Random seed for reproducibility
        results_dir: Optional directory to save individual run results
        plots_dir: Optional directory to save schedule plots
    """
    TAX_RATE = 0.0

    # Load shared resources
    cached_satellites = load_satellites_once()
    volcano_db_locations = load_volcano_locations_from_database()

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=lookahead_horizon_h)

    def compute_metrics(workflow_graph, broker, submission_cost_rate, execution_cost_rate):
        """Simplified metrics computation."""
        scheduled = [n for n in workflow_graph.nodes() if n.scheduled and n.feasible]

        num_total_attempts = len(broker._requests)
        rejected_rows = broker._requests[broker._requests['status'] == ObservationStatus.CONSTELLATION_REJECTED]
        num_rejected = len(rejected_rows)

        # Count accepted requests (SCHEDULED or DATA_RECEIVED)
        accepted_rows = broker._requests[
            broker._requests['status'].isin([ObservationStatus.SCHEDULED, ObservationStatus.DATA_RECEIVED])
        ]
        num_accepted = len(accepted_rows)

        # Count successfully executed requests (DATA_RECEIVED only)
        executed_rows = broker._requests[broker._requests['status'] == ObservationStatus.DATA_RECEIVED]
        num_executed = len(executed_rows)

        total_realized_quality = 0.0
        total_submission_cost = 0.0
        total_execution_cost = 0.0

        # Calculate quality (BEST per task)
        for task in workflow_graph.nodes():
            task_requests = broker._requests[broker._requests['request'] == task.observation_request]
            executed = task_requests[task_requests['status'] == ObservationStatus.DATA_RECEIVED]

            if len(executed) == 0:
                continue

            executed_qualities = []
            for _, req_row in executed.iterrows():
                if req_row['requested_pass'] is not None:
                    executed_qualities.append(task.rewarder(req_row['requested_pass'].highest))

            if len(executed_qualities) == 0:
                continue

            total_realized_quality += max(executed_qualities)

        # Calculate costs
        for _, req_row in broker._requests.iterrows():
            if req_row['requested_pass'] is None:
                continue

            task = None
            for t in workflow_graph.nodes():
                if t.observation_request == req_row['request']:
                    task = t
                    break

            if task is None:
                continue

            quality = task.rewarder(req_row['requested_pass'].highest)

            # Submission cost: ALL attempts
            total_submission_cost += submission_cost_rate * quality

            # Execution cost: ONLY executed
            if req_row['status'] == ObservationStatus.DATA_RECEIVED:
                total_execution_cost += execution_cost_rate * quality

        total_cost = total_submission_cost + total_execution_cost

        # Calculate rates
        rejection_rate = (num_rejected / num_total_attempts * 100) if num_total_attempts > 0 else 0.0
        acceptance_rate = (num_accepted / num_total_attempts * 100) if num_total_attempts > 0 else 0.0
        execution_rate = (num_executed / num_accepted * 100) if num_accepted > 0 else 0.0

        return {
            'scheduled': len(scheduled),
            'attempts': num_total_attempts,
            'accepted': num_accepted,
            'executed': num_executed,
            'rejections': num_rejected,
            'rejection_rate': rejection_rate,
            'acceptance_rate': acceptance_rate,
            'execution_rate': execution_rate,
            'quality': total_realized_quality,
            'cost': total_cost,
            'submission_cost': total_submission_cost,
            'execution_cost': total_execution_cost,
            'utility': total_realized_quality - total_cost,
        }

    def run_simulation_forward(world, max_safety_limit=40000, wall_clock_timeout_s=300):
        """
        Run simulation to completion with multiple safety mechanisms.

        Safety mechanisms:
        1. Max tick count (prevent infinite loops)
        2. Wall-clock timeout (prevent hung simulations)
        3. Progress detection (detect stuck states)
        """
        import time

        ticks = 0
        start_time = time.time()
        last_sim_time = world.time
        stuck_count = 0

        while True:
            retcode = world.tick(print_forbidden_prefixes=["Downlink", "End of downlink", "Unlock uplink", "Unlock satellite after obs", "Check timeout", "Executing Event"])
            ticks += 1

            # Check if simulation is progressing
            if world.time == last_sim_time:
                stuck_count += 1
                if stuck_count > 100:  # Simulation stuck for 100 ticks
                    print(f"      WARNING: Simulation stuck at time {world.time} for {stuck_count} ticks. Terminating.")
                    break
            else:
                stuck_count = 0
                last_sim_time = world.time

            # Normal termination
            if retcode == 0:
                break

            # Safety limit on tick count
            if ticks >= max_safety_limit:
                print(f"      WARNING: Reached max tick limit ({max_safety_limit}). Terminating.")
                break

            # Wall-clock timeout
            elapsed = time.time() - start_time
            if elapsed > wall_clock_timeout_s:
                print(f"      WARNING: Wall-clock timeout ({wall_clock_timeout_s}s) reached. Terminating.")
                break

        return ticks

    results = {}

    # === STOCHASTIC ===
    print(f"    Running stochastic (sub={submission_cost:.2f}, exec={exec_cost:.2f})...")
    import time
    sto_start = time.time()

    random.seed(run_seed)
    np.random.seed(run_seed)

    from fame_broker import Broker

    world_sto, const_sto = create_world_and_constellations(cached_satellites)
    workflow_sto = create_volcano_workflow(volcano_db_locations, min_time, max_time)
    broker_sto = Broker(constellations=const_sto, world=world_sto, name="Broker-Sto")
    broker_sto.add_workflow(workflow_sto)
    world_sto.add_broker(broker_sto)

    try:
        # Save schedule plot to plots directory
        if plots_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(plots_dir)
                broker_sto.schedule_workflow(
                    current_time=world_sto.time,
                    use_ilp=True,
                    use_stochastic=True,
                    stochastic_formulation="log_linearized",
                    acceptance_probability_function=acceptance_prob_function,
                    execution_probability_function=execution_prob_function,
                    submission_cost_rate=submission_cost,
                    execution_cost_rate=exec_cost,
                    tax_rate=TAX_RATE,
                    max_solver_time_s=MAX_SOLVER_TIME_S,
                    solver_engine="GUROBI",
                    update_timelines=False,
                    update_requests=False,
                    plot_schedule=True,
                    save_schedule_plot=True
                )
            finally:
                os.chdir(original_cwd)
        else:
            broker_sto.schedule_workflow(
                current_time=world_sto.time,
                use_ilp=True,
                use_stochastic=True,
                stochastic_formulation="log_linearized",
                acceptance_probability_function=acceptance_prob_function,
                execution_probability_function=execution_prob_function,
                submission_cost_rate=submission_cost,
                execution_cost_rate=exec_cost,
                tax_rate=TAX_RATE,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                solver_engine="GUROBI",
                update_timelines=False,
                update_requests=False
            )
        ticks = run_simulation_forward(world_sto)
        m = compute_metrics(broker_sto._workflow_graph, broker_sto, submission_cost, exec_cost)
        sto_elapsed = time.time() - sto_start

        results['sto_scheduled'] = m['scheduled']
        results['sto_attempts'] = m['attempts']
        results['sto_accepted'] = m['accepted']
        results['sto_executed'] = m['executed']
        results['sto_rejections'] = m['rejections']
        results['sto_rejection_rate'] = m['rejection_rate']
        results['sto_acceptance_rate'] = m['acceptance_rate']
        results['sto_execution_rate'] = m['execution_rate']
        results['sto_quality'] = m['quality']
        results['sto_cost'] = m['cost']
        results['sto_submission_cost'] = m['submission_cost']
        results['sto_execution_cost'] = m['execution_cost']
        results['sto_utility'] = m['utility']
        results['sto_ticks'] = ticks
        results['sto_elapsed_s'] = sto_elapsed
        print(f"      Stochastic: sched={m['scheduled']}, qual={m['quality']:.1f}, cost={m['cost']:.1f}, util={m['utility']:.1f}, ticks={ticks}, time={sto_elapsed:.1f}s")

        # Save immediately after this run completes
        if results_dir:
            run_file = os.path.join(results_dir, f"run_sub{submission_cost:.2f}_exec{exec_cost:.2f}_stochastic.json")
            with open(run_file, 'w') as f:
                json.dump({
                    'scheduler': 'stochastic',
                    'submission_cost': submission_cost,
                    'execution_cost': exec_cost,
                    'scheduled': m['scheduled'],
                    'attempts': m['attempts'],
                    'accepted': m['accepted'],
                    'executed': m['executed'],
                    'rejections': m['rejections'],
                    'rejection_rate': m['rejection_rate'],
                    'acceptance_rate': m['acceptance_rate'],
                    'execution_rate': m['execution_rate'],
                    'quality': m['quality'],
                    'cost': m['cost'],
                    'submission_cost_value': m['submission_cost'],
                    'execution_cost_value': m['execution_cost'],
                    'utility': m['utility'],
                    'ticks': ticks,
                    'elapsed_s': sto_elapsed,
                }, f, indent=2)
            print(f"      [Saved] {run_file}")
    except Exception as e:
        print(f"      Stochastic FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['sto_scheduled'] = 0
        results['sto_attempts'] = 0
        results['sto_accepted'] = 0
        results['sto_executed'] = 0
        results['sto_rejections'] = 0
        results['sto_rejection_rate'] = 0
        results['sto_acceptance_rate'] = 0
        results['sto_execution_rate'] = 0
        results['sto_quality'] = 0
        results['sto_cost'] = 0
        results['sto_submission_cost'] = 0
        results['sto_execution_cost'] = 0
        results['sto_utility'] = 0
        results['sto_ticks'] = 0
        results['sto_elapsed_s'] = 0

    # === DETERMINISTIC ===
    print(f"    Running deterministic (sub={submission_cost:.2f}, exec={exec_cost:.2f})...")
    det_start = time.time()

    random.seed(run_seed)
    np.random.seed(run_seed)

    world_det, const_det = create_world_and_constellations(cached_satellites)
    workflow_det = create_volcano_workflow(volcano_db_locations, min_time, max_time)
    broker_det = Broker(constellations=const_det, world=world_det, name="Broker-Det")
    broker_det.add_workflow(workflow_det)
    world_det.add_broker(broker_det)

    try:
        # Save schedule plot to plots directory
        if plots_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(plots_dir)
                broker_det.schedule_workflow(
                    current_time=world_det.time,
                    use_ilp=True,
                    use_stochastic=False,
                    max_solver_time_s=MAX_SOLVER_TIME_S,
                    solver_engine="GUROBI",
                    update_timelines=False,
                    update_requests=False,
                    tax_rate=TAX_RATE,
                    plot_schedule=True,
                    save_schedule_plot=True
                )
            finally:
                os.chdir(original_cwd)
        else:
            broker_det.schedule_workflow(
                current_time=world_det.time,
                use_ilp=True,
                use_stochastic=False,
                max_solver_time_s=MAX_SOLVER_TIME_S,
                solver_engine="GUROBI",
                update_timelines=False,
                update_requests=False,
                tax_rate=TAX_RATE
            )
        ticks = run_simulation_forward(world_det)
        m = compute_metrics(broker_det._workflow_graph, broker_det, submission_cost, exec_cost)
        det_elapsed = time.time() - det_start

        results['det_scheduled'] = m['scheduled']
        results['det_attempts'] = m['attempts']
        results['det_accepted'] = m['accepted']
        results['det_executed'] = m['executed']
        results['det_rejections'] = m['rejections']
        results['det_rejection_rate'] = m['rejection_rate']
        results['det_acceptance_rate'] = m['acceptance_rate']
        results['det_execution_rate'] = m['execution_rate']
        results['det_quality'] = m['quality']
        results['det_cost'] = m['cost']
        results['det_submission_cost'] = m['submission_cost']
        results['det_execution_cost'] = m['execution_cost']
        results['det_utility'] = m['utility']
        results['det_ticks'] = ticks
        results['det_elapsed_s'] = det_elapsed
        print(f"      Deterministic: sched={m['scheduled']}, qual={m['quality']:.1f}, cost={m['cost']:.1f}, util={m['utility']:.1f}, ticks={ticks}, time={det_elapsed:.1f}s")

        # Save immediately after this run completes
        if results_dir:
            run_file = os.path.join(results_dir, f"run_sub{submission_cost:.2f}_exec{exec_cost:.2f}_deterministic.json")
            with open(run_file, 'w') as f:
                json.dump({
                    'scheduler': 'deterministic',
                    'submission_cost': submission_cost,
                    'execution_cost': exec_cost,
                    'scheduled': m['scheduled'],
                    'attempts': m['attempts'],
                    'accepted': m['accepted'],
                    'executed': m['executed'],
                    'rejections': m['rejections'],
                    'rejection_rate': m['rejection_rate'],
                    'acceptance_rate': m['acceptance_rate'],
                    'execution_rate': m['execution_rate'],
                    'quality': m['quality'],
                    'cost': m['cost'],
                    'submission_cost_value': m['submission_cost'],
                    'execution_cost_value': m['execution_cost'],
                    'utility': m['utility'],
                    'ticks': ticks,
                    'elapsed_s': det_elapsed,
                }, f, indent=2)
            print(f"      [Saved] {run_file}")
    except Exception as e:
        print(f"      Deterministic FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['det_scheduled'] = 0
        results['det_attempts'] = 0
        results['det_accepted'] = 0
        results['det_executed'] = 0
        results['det_rejections'] = 0
        results['det_rejection_rate'] = 0
        results['det_acceptance_rate'] = 0
        results['det_execution_rate'] = 0
        results['det_quality'] = 0
        results['det_cost'] = 0
        results['det_submission_cost'] = 0
        results['det_execution_cost'] = 0
        results['det_utility'] = 0
        results['det_ticks'] = 0
        results['det_elapsed_s'] = 0

    return results


def run_ablation_study():
    """
    Run ablation study across a grid of cost parameter values.
    """
    print("\n" + "="*70)
    print("COST ABLATION STUDY: Stochastic vs Deterministic")
    print("="*70)

    # Create results directory
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join("results", f"cost_ablation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n[Results] Saving to directory: {results_dir}")
    print(f"[Plots] Saving schedule plots to: {plots_dir}")

    # Define parameter grid
    submission_costs = [0.00, 0.10, 0.15, 0.25]
    execution_costs = [0.05, 0.20, 0.40, 0.50]

    print(f"\n[Config] Testing {len(submission_costs)} x {len(execution_costs)} = {len(submission_costs) * len(execution_costs)} combinations")
    print(f"  Submission costs: {submission_costs}")
    print(f"  Execution costs: {execution_costs}")

    all_results = []

    for sub_cost, exec_cost in product(submission_costs, execution_costs):
        print(f"\n--- Testing submission_cost={sub_cost:.2f}, execution_cost={exec_cost:.2f} ---")

        # Run comparison (pass results_dir and plots_dir for immediate saves)
        result = run_single_comparison(sub_cost, exec_cost, run_seed=42, results_dir=results_dir, plots_dir=plots_dir)

        # Add cost parameters to result
        result['submission_cost_param'] = sub_cost
        result['execution_cost_param'] = exec_cost

        # Calculate relative performance
        if result['det_utility'] > 0:
            result['sto_vs_det_utility_ratio'] = result['sto_utility'] / result['det_utility']
            result['sto_vs_det_utility_diff'] = result['sto_utility'] - result['det_utility']
        else:
            result['sto_vs_det_utility_ratio'] = 0
            result['sto_vs_det_utility_diff'] = 0

        if result['det_scheduled'] > 0:
            result['sto_quality_per_task'] = result['sto_quality'] / result['sto_scheduled'] if result['sto_scheduled'] > 0 else 0
            result['det_quality_per_task'] = result['det_quality'] / result['det_scheduled']
        else:
            result['sto_quality_per_task'] = 0
            result['det_quality_per_task'] = 0

        all_results.append(result)

        # Save individual result
        result_file = os.path.join(results_dir, f"result_sub{sub_cost:.2f}_exec{exec_cost:.2f}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Save summary CSV (one row per parameter combination with aggregated metrics)
    df = pd.DataFrame(all_results)
    summary_csv = os.path.join(results_dir, "ablation_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"\n[Summary] Saved ablation results to {summary_csv}")

    # Save detailed per-run results in long format (one row per scheduler+parameter combination)
    detailed_rows = []
    for result in all_results:
        sub_cost = result['submission_cost_param']
        exec_cost = result['execution_cost_param']

        # Stochastic row
        detailed_rows.append({
            'submission_cost': sub_cost,
            'execution_cost': exec_cost,
            'scheduler': 'stochastic',
            'scheduled': result['sto_scheduled'],
            'attempts': result['sto_attempts'],
            'accepted': result.get('sto_accepted', 0),
            'executed': result.get('sto_executed', 0),
            'rejections': result['sto_rejections'],
            'rejection_rate': result.get('sto_rejection_rate', 0),
            'acceptance_rate': result.get('sto_acceptance_rate', 0),
            'execution_rate': result.get('sto_execution_rate', 0),
            'quality': result['sto_quality'],
            'cost': result['sto_cost'],
            'submission_cost_value': result['sto_submission_cost'],
            'execution_cost_value': result['sto_execution_cost'],
            'utility': result['sto_utility'],
            'ticks': result.get('sto_ticks', 0),
            'elapsed_s': result.get('sto_elapsed_s', 0),
        })

        # Deterministic row
        detailed_rows.append({
            'submission_cost': sub_cost,
            'execution_cost': exec_cost,
            'scheduler': 'deterministic',
            'scheduled': result['det_scheduled'],
            'attempts': result['det_attempts'],
            'accepted': result.get('det_accepted', 0),
            'executed': result.get('det_executed', 0),
            'rejections': result['det_rejections'],
            'rejection_rate': result.get('det_rejection_rate', 0),
            'acceptance_rate': result.get('det_acceptance_rate', 0),
            'execution_rate': result.get('det_execution_rate', 0),
            'quality': result['det_quality'],
            'cost': result['det_cost'],
            'submission_cost_value': result['det_submission_cost'],
            'execution_cost_value': result['det_execution_cost'],
            'utility': result['det_utility'],
            'ticks': result.get('det_ticks', 0),
            'elapsed_s': result.get('det_elapsed_s', 0),
        })

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = os.path.join(results_dir, "all_runs.csv")
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"[Detailed] Saved detailed per-run results to {detailed_csv}")

    # Print summary analysis
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)

    # Find best configuration for stochastic
    best_sto_idx = df['sto_utility'].idxmax()
    best_sto = df.loc[best_sto_idx]

    print(f"\n✓ Best Stochastic Configuration:")
    print(f"  Submission cost: {best_sto['submission_cost_param']:.2f}")
    print(f"  Execution cost: {best_sto['execution_cost_param']:.2f}")
    print(f"  Utility: {best_sto['sto_utility']:.2f}")
    print(f"  Scheduled: {best_sto['sto_scheduled']:.0f}")
    print(f"  Quality/task: {best_sto['sto_quality_per_task']:.2f}")

    # Find configuration where stochastic beats deterministic by most
    df['sto_advantage'] = df['sto_utility'] - df['det_utility']
    best_advantage_idx = df['sto_advantage'].idxmax()
    best_advantage = df.loc[best_advantage_idx]

    print(f"\n✓ Largest Stochastic Advantage:")
    print(f"  Submission cost: {best_advantage['submission_cost_param']:.2f}")
    print(f"  Execution cost: {best_advantage['execution_cost_param']:.2f}")
    print(f"  Stochastic utility: {best_advantage['sto_utility']:.2f}")
    print(f"  Deterministic utility: {best_advantage['det_utility']:.2f}")
    print(f"  Advantage: {best_advantage['sto_advantage']:.2f} ({best_advantage['sto_advantage']/best_advantage['det_utility']*100:.1f}%)")

    # Find where deterministic wins by most
    worst_advantage_idx = df['sto_advantage'].idxmin()
    worst_advantage = df.loc[worst_advantage_idx]

    print(f"\n✓ Largest Deterministic Advantage:")
    print(f"  Submission cost: {worst_advantage['submission_cost_param']:.2f}")
    print(f"  Execution cost: {worst_advantage['execution_cost_param']:.2f}")
    print(f"  Stochastic utility: {worst_advantage['sto_utility']:.2f}")
    print(f"  Deterministic utility: {worst_advantage['det_utility']:.2f}")
    print(f"  Disadvantage: {worst_advantage['sto_advantage']:.2f} ({worst_advantage['sto_advantage']/worst_advantage['det_utility']*100:.1f}%)")

    # Analyze sensitivity
    print(f"\n✓ Sensitivity Analysis:")
    print(f"  Stochastic utility range: {df['sto_utility'].min():.2f} to {df['sto_utility'].max():.2f}")
    print(f"  Deterministic utility range: {df['det_utility'].min():.2f} to {df['det_utility'].max():.2f}")
    print(f"  Stochastic advantage range: {df['sto_advantage'].min():.2f} to {df['sto_advantage'].max():.2f}")

    # Count wins
    sto_wins = (df['sto_utility'] > df['det_utility']).sum()
    det_wins = (df['det_utility'] > df['sto_utility']).sum()
    ties = (df['det_utility'] == df['sto_utility']).sum()

    print(f"\n✓ Win/Loss Record:")
    print(f"  Stochastic wins: {sto_wins}/{len(df)} ({sto_wins/len(df)*100:.1f}%)")
    print(f"  Deterministic wins: {det_wins}/{len(df)} ({det_wins/len(df)*100:.1f}%)")
    print(f"  Ties: {ties}/{len(df)}")

    return df, results_dir


if __name__ == "__main__":
    df, results_dir = run_ablation_study()
    print(f"\n[Done] All results saved to {results_dir}")
