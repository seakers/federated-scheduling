"""
Cost Ablation Study: Greedy vs Deterministic vs Stochastic Scheduler Comparison

Sweeps a grid of submission_cost × execution_cost values to understand how cost
parameters affect the relative performance and trade-offs of all three schedulers:
1. Stochastic MILP (Log-linearized with PWL / exact MINLP handling)
2. Deterministic ILP (Single-pass primary booking without risk modeling)
3. Greedy Baseline (Myopic priority-based booking)

Key Metrics Analyzed (via fame_metrics.compute_metrics_v2):
- Task Completion Rate: Fraction of unique DAG requests delivering DATA_RECEIVED
- Group Completion Rate: Fraction of workflow request groups fully completed
- Realized Quality: Best delivered data quality per request
- Total Cost: Unconditional submission overhead + conditional execution fees
- Net Utility: Realized Quality - Total Realized Cost
- True Redundancy: Average submitted passes per task
"""

import datetime as dt
import numpy as np
import pandas as pd
import os
import json
import time
import random
from itertools import product

# Import simulation components from the primary volcano comparison module
from volcano_stochastic_comparison_real import (
    load_volcano_locations_from_database,
    load_satellites_once,
    create_world_and_constellations,
    create_volcano_workflow,
    SIMULATION_START,
    lookahead_horizon_h,
    MAX_SOLVER_TIME_S,
)
from fame_constellation_scheduler import ObservationStatus
from fame_broker import Broker
from fame_demand_model import DemandField, DemandFieldConfig

# Import authoritative demand-aware metric compiler from fame_metrics
from fame_metrics import compute_metrics_v2


def _execution_prob_function(constrained_request, satellite, obs_pass):
    """
    Computes P(execute | accepted) conditional execution probability based on satellite look angle.
    Returns a probability bounded in [0.70, 0.99].
    """
    look_angle = abs(90.0 - obs_pass.highest.look_angle_dec_deg)
    return max(0.7, min(0.99, 0.95 - (look_angle / 90.0) * 0.20))


def run_simulation_forward(world, max_safety_limit=40000, wall_clock_timeout_s=300):
    """
    Tick discrete-event simulation to completion with tick-count, wall-clock, 
    and stuck-state detection safety limits.
    """
    ticks = 0
    start_time = time.time()
    last_sim_time = world.time
    stuck_count = 0
    last_progress_report = 0

    while True:
        retcode = world.tick(print_forbidden_prefixes=[
            "Downlink", "End of downlink", "Unlock uplink",
            "Unlock satellite after obs", "Check timeout", "Executing Event"
        ])
        ticks += 1

        if ticks - last_progress_report >= 5000:
            print(f"      [Sim] {ticks} ticks, current sim time: {world.time}")
            last_progress_report = ticks

        if world.time == last_sim_time:
            stuck_count += 1
            if stuck_count > 100:
                print(f"      WARNING: Simulation clock stuck at {world.time} for {stuck_count} ticks. Terminating.")
                break
        else:
            stuck_count = 0
            last_sim_time = world.time

        if retcode == 0:
            break
        if ticks >= max_safety_limit:
            print(f"      WARNING: Max tick safety limit ({max_safety_limit}) reached. Terminating.")
            break
        if time.time() - start_time > wall_clock_timeout_s:
            print(f"      WARNING: Wall-clock timeout ({wall_clock_timeout_s}s) reached. Terminating.")
            break

    return ticks


def _save_run_json(path, scheduler, sub_cost, exec_cost, m, ticks, elapsed_s):
    """Serializes per-run metrics dictionary to a JSON file."""
    record = {
        'scheduler': scheduler,
        'submission_cost_param': sub_cost,
        'execution_cost_param': exec_cost,
        'task_completion_rate': m.get('task_completion_rate', 0.0),
        'group_completion_rate': m.get('group_completion_rate', 0.0),
        'realized_quality': m.get('realized_quality', 0.0),
        'total_cost': m.get('total_cost', 0.0),
        'submission_cost': m.get('submission_cost', 0.0),
        'execution_cost': m.get('execution_cost', 0.0),
        'utility': m.get('utility', 0.0),
        'n_submissions': m.get('n_submissions', 0),
        'n_accepted': m.get('n_accepted', 0),
        'n_executed': m.get('n_executed', 0),
        'n_rejected': m.get('n_rejected', 0),
        'rejection_rate': m.get('rejection_rate', 0.0),
        'submitted_passes_per_task': m.get('submitted_passes_per_task', 0.0),
        'exec_passes_per_completed': m.get('exec_passes_per_completed', 0.0),
        'ticks': ticks,
        'elapsed_s': elapsed_s,
    }
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)


def _zero_metrics():
    """Returns a zeroed metrics dictionary for graceful error recovery on run failure."""
    return {
        'task_completion_rate': 0.0,
        'group_completion_rate': 0.0,
        'realized_quality': 0.0,
        'total_cost': 0.0,
        'submission_cost': 0.0,
        'execution_cost': 0.0,
        'utility': 0.0,
        'n_submissions': 0,
        'n_accepted': 0,
        'n_executed': 0,
        'n_rejected': 0,
        'rejection_rate': 0.0,
        'submitted_passes_per_task': 0.0,
        'exec_passes_per_completed': 0.0,
        'ticks': 0,
        'elapsed_s': 0.0,
    }


def run_single_comparison(
    submission_cost,
    exec_cost,
    run_seed=42,
    results_dir=None,
    plots_dir=None,
    cached_satellites=None,
    volcano_db_locations=None,
    demand_field=None,
):
    """
    Executes all three schedulers (Stochastic Log-Linearized, Deterministic ILP, and Greedy)
    under an identical random seed and cost parameter pair (submission_cost, exec_cost).

    Returns a unified flat dictionary with 'sto_*', 'det_*', and 'grd_*' prefixed metrics.
    """
    TAX_RATE = 0.0  # Legacy tax disabled; costs are directly managed via submission_cost and exec_cost

    if cached_satellites is None:
        cached_satellites = load_satellites_once()
    if volcano_db_locations is None:
        volcano_db_locations = load_volcano_locations_from_database()

    min_time = SIMULATION_START
    max_time = SIMULATION_START + dt.timedelta(hours=lookahead_horizon_h)

    # Use dynamic spatio-temporal acceptance probability function from DemandField if present
    if demand_field is not None:
        acceptance_prob_fn = demand_field.make_acceptance_prob_function()
    else:
        # Fallback acceptance probability function
        def acceptance_prob_fn(constrained_request, satellite, obs_pass):
            name = satellite.name.upper()
            if any(x in name for x in ["SKYSAT", "PELICAN", "TANAGER"]):
                return 0.70
            elif "UMBRA" in name:
                return 0.85
            elif "CAPELLA" in name or "ACADIA" in name:
                return 0.90
            elif "LOFT" in name or "YAM" in name:
                return 0.92
            elif "UBOTICA" in name or "HAMMER" in name or "ACCENTURE" in name:
                return 0.93
            elif "PERSISTENCE" in name or "LEMUR" in name:
                return 0.94
            elif "AEROCUBE" in name:
                return 0.95
            elif "ICEYE" in name:
                return 0.96
            return 0.85

    results = {}

    def _plots_subdir(scheduler_name):
        if plots_dir is None:
            return None
        d = os.path.join(plots_dir, f"sub{submission_cost:.2f}_exec{exec_cost:.2f}", scheduler_name)
        os.makedirs(d, exist_ok=True)
        return d

    # ── 1. STOCHASTIC LOG-LINEARIZED SCHEDULER ─────────────────────────────────
    print(f"    Running stochastic_log (sub={submission_cost:.2f}, exec={exec_cost:.2f})...")
    sto_start = time.time()
    random.seed(run_seed)
    np.random.seed(run_seed)

    world_sto, const_sto = create_world_and_constellations(cached_satellites, demand_field=demand_field)
    workflow_sto = create_volcano_workflow(volcano_db_locations, min_time, max_time)
    broker_sto = Broker(constellations=const_sto, world=world_sto, name="Broker-Sto-Log")
    broker_sto.add_workflow(workflow_sto)
    world_sto.add_broker(broker_sto)

    try:
        sto_kwargs = dict(
            current_time=world_sto.time,
            use_ilp=True,
            use_stochastic=True,
            stochastic_formulation="log_linearized",
            acceptance_probability_function=acceptance_prob_fn,
            execution_probability_function=_execution_prob_function,
            submission_cost_rate=submission_cost,
            execution_cost_rate=exec_cost,
            tax_rate=TAX_RATE,
            max_solver_time_s=MAX_SOLVER_TIME_S,
            solver_engine="GUROBI",
            update_timelines=False,
            update_requests=False,
            max_reschedule_depth=5,
        )
        subdir = _plots_subdir("stochastic_log")
        if subdir:
            sto_kwargs.update(plot_schedule=True, save_schedule_plot=True, results_path=subdir)

        broker_sto.schedule_workflow_redundant(**sto_kwargs)
        ticks = run_simulation_forward(world_sto)

        m = compute_metrics_v2(
            workflow_graph=broker_sto._workflow_graph,
            broker=broker_sto,
            observation_status_enum=ObservationStatus,
            submission_cost_rate=submission_cost,
            execution_cost_rate=exec_cost,
            verbose=False,
        )
        sto_elapsed = time.time() - sto_start
        print(f"      [Sto] TaskComp={m['task_completion_rate']:.3f}, GroupComp={m['group_completion_rate']:.3f}, "
              f"Qual={m['realized_quality']:.1f}, Cost={m['total_cost']:.1f}, Util={m['utility']:.1f}, "
              f"Passes/Task={m['submitted_passes_per_task']:.2f}, Elapsed={sto_elapsed:.1f}s")

        for k, v in m.items():
            results[f'sto_{k}'] = v
        results['sto_ticks'] = ticks
        results['sto_elapsed_s'] = sto_elapsed

        if results_dir:
            path = os.path.join(results_dir, f"run_sub{submission_cost:.2f}_exec{exec_cost:.2f}_stochastic_log.json")
            _save_run_json(path, 'stochastic_log', submission_cost, exec_cost, m, ticks, sto_elapsed)
    except Exception as e:
        print(f"      Stochastic FAILED: {e}")
        import traceback; traceback.print_exc()
        zm = _zero_metrics()
        for k, v in zm.items():
            results[f'sto_{k}'] = v

    # ── 2. DETERMINISTIC ILP SCHEDULER ─────────────────────────────────────────
    print(f"    Running deterministic (sub={submission_cost:.2f}, exec={exec_cost:.2f})...")
    det_start = time.time()
    random.seed(run_seed)
    np.random.seed(run_seed)

    world_det, const_det = create_world_and_constellations(cached_satellites, demand_field=demand_field)
    workflow_det = create_volcano_workflow(volcano_db_locations, min_time, max_time)
    broker_det = Broker(constellations=const_det, world=world_det, name="Broker-Det")
    broker_det.add_workflow(workflow_det)
    world_det.add_broker(broker_det)

    try:
        det_kwargs = dict(
            current_time=world_det.time,
            use_ilp=True,
            use_stochastic=False,
            max_solver_time_s=MAX_SOLVER_TIME_S,
            solver_engine="GUROBI",
            update_timelines=False,
            update_requests=False,
            tax_rate=TAX_RATE,
            max_reschedule_depth=1,
        )
        subdir = _plots_subdir("deterministic")
        if subdir:
            det_kwargs.update(plot_schedule=True, save_schedule_plot=True, results_path=subdir)

        broker_det.schedule_workflow_redundant(**det_kwargs)
        ticks = run_simulation_forward(world_det)

        m = compute_metrics_v2(
            workflow_graph=broker_det._workflow_graph,
            broker=broker_det,
            observation_status_enum=ObservationStatus,
            submission_cost_rate=submission_cost,
            execution_cost_rate=exec_cost,
            verbose=False,
        )
        det_elapsed = time.time() - det_start
        print(f"      [Det] TaskComp={m['task_completion_rate']:.3f}, GroupComp={m['group_completion_rate']:.3f}, "
              f"Qual={m['realized_quality']:.1f}, Cost={m['total_cost']:.1f}, Util={m['utility']:.1f}, "
              f"Passes/Task={m['submitted_passes_per_task']:.2f}, Elapsed={det_elapsed:.1f}s")

        for k, v in m.items():
            results[f'det_{k}'] = v
        results['det_ticks'] = ticks
        results['det_elapsed_s'] = det_elapsed

        if results_dir:
            path = os.path.join(results_dir, f"run_sub{submission_cost:.2f}_exec{exec_cost:.2f}_deterministic.json")
            _save_run_json(path, 'deterministic', submission_cost, exec_cost, m, ticks, det_elapsed)
    except Exception as e:
        print(f"      Deterministic FAILED: {e}")
        import traceback; traceback.print_exc()
        zm = _zero_metrics()
        for k, v in zm.items():
            results[f'det_{k}'] = v

    # ── 3. GREEDY SCHEDULER ────────────────────────────────────────────────────
    print(f"    Running greedy (sub={submission_cost:.2f}, exec={exec_cost:.2f})...")
    grd_start = time.time()
    random.seed(run_seed)
    np.random.seed(run_seed)

    world_grd, const_grd = create_world_and_constellations(cached_satellites, demand_field=demand_field)
    workflow_grd = create_volcano_workflow(volcano_db_locations, min_time, max_time)
    broker_grd = Broker(constellations=const_grd, world=world_grd, name="Broker-Greedy")
    broker_grd.add_workflow(workflow_grd)
    world_grd.add_broker(broker_grd)

    try:
        grd_kwargs = dict(
            current_time=world_grd.time,
            use_ilp=False,
            use_stochastic=False,
            max_solver_time_s=MAX_SOLVER_TIME_S,
            update_timelines=False,
            update_requests=False,
            tax_rate=TAX_RATE,
            max_reschedule_depth=1,
        )
        subdir = _plots_subdir("greedy")
        if subdir:
            grd_kwargs.update(plot_schedule=True, save_schedule_plot=True, results_path=subdir)

        broker_grd.schedule_workflow_redundant(**grd_kwargs)
        ticks = run_simulation_forward(world_grd)

        m = compute_metrics_v2(
            workflow_graph=broker_grd._workflow_graph,
            broker=broker_grd,
            observation_status_enum=ObservationStatus,
            submission_cost_rate=submission_cost,
            execution_cost_rate=exec_cost,
            verbose=False,
        )
        grd_elapsed = time.time() - grd_start
        print(f"      [Grd] TaskComp={m['task_completion_rate']:.3f}, GroupComp={m['group_completion_rate']:.3f}, "
              f"Qual={m['realized_quality']:.1f}, Cost={m['total_cost']:.1f}, Util={m['utility']:.1f}, "
              f"Passes/Task={m['submitted_passes_per_task']:.2f}, Elapsed={grd_elapsed:.1f}s")

        for k, v in m.items():
            results[f'grd_{k}'] = v
        results['grd_ticks'] = ticks
        results['grd_elapsed_s'] = grd_elapsed

        if results_dir:
            path = os.path.join(results_dir, f"run_sub{submission_cost:.2f}_exec{exec_cost:.2f}_greedy.json")
            _save_run_json(path, 'greedy', submission_cost, exec_cost, m, ticks, grd_elapsed)
    except Exception as e:
        print(f"      Greedy FAILED: {e}")
        import traceback; traceback.print_exc()
        zm = _zero_metrics()
        for k, v in zm.items():
            results[f'grd_{k}'] = v

    return results


def run_ablation_study():
    """Sweeps a cost parameter grid running Greedy + Deterministic + Stochastic_Log at each point."""
    print("\n" + "=" * 70)
    print("COST ABLATION STUDY: Greedy vs Deterministic vs Stochastic")
    print("=" * 70)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join("results", f"cost_ablation_{timestamp}")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n[Results] {results_dir}")
    print(f"[Plots]   {plots_dir}")

    print("\n[Init] Loading satellites and volcano database...")
    cached_satellites = load_satellites_once()
    volcano_db_locations = load_volcano_locations_from_database()

    min_time = SIMULATION_START

    # Build demand field — single source of truth for simulator and planner
    _demand_cfg = DemandFieldConfig(use_constant_probability=False)
    _horizon_s = lookahead_horizon_h * 3600.0
    demand_field = DemandField(config=_demand_cfg, reference_time=min_time, horizon_s=_horizon_s)

    for vloc in volcano_db_locations:
        demand_field.add_spike(vloc.lat_deg, vloc.lon_deg, min_time)

    _all_constellation_names = ["Planet", "Umbra", "Capella", "LOFT", "Ubotica", "Mission Control", "AC", "ICEYE"]
    print("[DemandField] Precomputing demand trajectories...")
    demand_field.precompute(_all_constellation_names)
    print("[DemandField] Precompute complete.")

    # Cost parameter grid
    submission_costs = [0.00, 0.05, 0.10, 0.20]
    execution_costs  = [0.05, 0.15, 0.30, 0.50]

    n_combos = len(submission_costs) * len(execution_costs)
    print(f"\n[Config] {len(submission_costs)} × {len(execution_costs)} = {n_combos} cost combinations")
    print(f"  Submission costs (c_sub): {submission_costs}")
    print(f"  Execution costs (c_exec):  {execution_costs}")

    all_results = []

    for sub_cost, exec_cost in product(submission_costs, execution_costs):
        print(f"\n--- submission_cost={sub_cost:.2f}, execution_cost={exec_cost:.2f} ---")

        result = run_single_comparison(
            sub_cost, exec_cost,
            run_seed=42,
            results_dir=results_dir,
            plots_dir=plots_dir,
            cached_satellites=cached_satellites,
            volcano_db_locations=volcano_db_locations,
            demand_field=demand_field,
        )
        result['submission_cost_param'] = sub_cost
        result['execution_cost_param'] = exec_cost

        # Comparative utility metrics (Stochastic vs Deterministic & Greedy)
        det_util = result.get('det_utility', 0.0)
        sto_util = result.get('sto_utility', 0.0)
        grd_util = result.get('grd_utility', 0.0)

        result['sto_vs_det_utility_diff'] = sto_util - det_util
        result['sto_vs_grd_utility_diff'] = sto_util - grd_util
        result['sto_vs_det_task_comp_diff'] = result.get('sto_task_completion_rate', 0.0) - result.get('det_task_completion_rate', 0.0)
        result['sto_vs_grd_task_comp_diff'] = result.get('sto_task_completion_rate', 0.0) - result.get('grd_task_completion_rate', 0.0)

        all_results.append(result)

        # Save combined result JSON for this parameter grid point
        with open(os.path.join(results_dir, f"result_sub{sub_cost:.2f}_exec{exec_cost:.2f}.json"), 'w') as f:
            json.dump({k: v for k, v in result.items() if not isinstance(v, (list, set, dict))}, f, indent=2)

    # ── SAVE SUMMARY CSVs ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    ablation_summary_csv = os.path.join(results_dir, "ablation_summary.csv")
    df.to_csv(ablation_summary_csv, index=False)
    print(f"\n[Summary Wide CSV]  {ablation_summary_csv}")

    # Long-format all_runs.csv (one row per scheduler × cost combination)
    detailed_rows = []
    for result in all_results:
        sub_cost  = result['submission_cost_param']
        exec_cost = result['execution_cost_param']
        for prefix, scheduler in [('sto', 'stochastic_log'), ('det', 'deterministic'), ('grd', 'greedy')]:
            detailed_rows.append({
                'submission_cost_param':    sub_cost,
                'execution_cost_param':     exec_cost,
                'scheduler':                 scheduler,
                'task_completion_rate':      result.get(f'{prefix}_task_completion_rate', 0.0),
                'group_completion_rate':     result.get(f'{prefix}_group_completion_rate', 0.0),
                'realized_quality':          result.get(f'{prefix}_realized_quality', 0.0),
                'total_cost':                result.get(f'{prefix}_total_cost', 0.0),
                'submission_cost':           result.get(f'{prefix}_submission_cost', 0.0),
                'execution_cost':            result.get(f'{prefix}_execution_cost', 0.0),
                'utility':                   result.get(f'{prefix}_utility', 0.0),
                'n_submissions':             result.get(f'{prefix}_n_submissions', 0),
                'n_accepted':                result.get(f'{prefix}_n_accepted', 0),
                'n_executed':                result.get(f'{prefix}_n_executed', 0),
                'n_rejected':                result.get(f'{prefix}_n_rejected', 0),
                'rejection_rate':            result.get(f'{prefix}_rejection_rate', 0.0),
                'submitted_passes_per_task': result.get(f'{prefix}_submitted_passes_per_task', 0.0),
                'exec_passes_per_completed': result.get(f'{prefix}_exec_passes_per_completed', 0.0),
                'ticks':                     result.get(f'{prefix}_ticks', 0),
                'elapsed_s':                 result.get(f'{prefix}_elapsed_s', 0.0),
            })

    detailed_df = pd.DataFrame(detailed_rows)
    all_runs_csv = os.path.join(results_dir, "all_runs.csv")
    detailed_df.to_csv(all_runs_csv, index=False)
    print(f"[Detailed Long CSV]  {all_runs_csv}")

    # ── SUMMARY STATISTICAL ANALYSIS ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLATION STUDY EXECUTIVE SUMMARY")
    print("=" * 70)

    for prefix, label in [('sto', 'STOCHASTIC_LOG'), ('det', 'DETERMINISTIC'), ('grd', 'GREEDY')]:
        col_util = f'{prefix}_utility'
        col_comp = f'{prefix}_task_completion_rate'
        col_pass = f'{prefix}_submitted_passes_per_task'

        if col_util in df.columns:
            print(f"\n{label}:")
            print(f"  Task Comp Rate:   {df[col_comp].mean():.3f} ± {df[col_comp].std():.3f} (Range: {df[col_comp].min():.3f} – {df[col_comp].max():.3f})")
            print(f"  Net Utility:      {df[col_util].mean():.2f} ± {df[col_util].std():.2f} (Range: {df[col_util].min():.2f} – {df[col_util].max():.2f})")
            print(f"  Redundancy Pass:  {df[col_pass].mean():.2f} submitted passes/task")

    # Best Stochastic Configuration
    if 'sto_utility' in df.columns:
        best_sto = df.loc[df['sto_utility'].idxmax()]
        print(f"\nBest Stochastic Net Utility Configuration:")
        print(f"  sub_cost={best_sto['submission_cost_param']:.2f}, exec_cost={best_sto['execution_cost_param']:.2f}")
        print(f"  Net Utility={best_sto['sto_utility']:.2f}, Task Comp Rate={best_sto['sto_task_completion_rate']:.3f}")

    # Stochastic Advantage Analysis
    df['sto_adv_det'] = df['sto_utility'] - df['det_utility']
    df['sto_adv_grd'] = df['sto_utility'] - df['grd_utility']

    best_adv_det = df.loc[df['sto_adv_det'].idxmax()]
    worst_adv_det = df.loc[df['sto_adv_det'].idxmin()]

    print(f"\nLargest Stochastic Advantage over Deterministic ILP:")
    print(f"  sub={best_adv_det['submission_cost_param']:.2f}, exec={best_adv_det['execution_cost_param']:.2f}: "
          f"Δutility={best_adv_det['sto_adv_det']:+.2f}, ΔTaskComp={best_adv_det['sto_vs_det_task_comp_diff']:+.3f}")

    print(f"\nSmallest Stochastic Advantage over Deterministic ILP:")
    print(f"  sub={worst_adv_det['submission_cost_param']:.2f}, exec={worst_adv_det['execution_cost_param']:.2f}: "
          f"Δutility={worst_adv_det['sto_adv_det']:+.2f}, ΔTaskComp={worst_adv_det['sto_vs_det_task_comp_diff']:+.3f}")

    n = len(df)
    sto_wins_det = (df['sto_utility'] > df['det_utility']).sum()
    sto_wins_grd = (df['sto_utility'] > df['grd_utility']).sum()
    det_wins_grd = (df['det_utility'] > df['grd_utility']).sum()

    print(f"\nWin/Loss Head-to-Head Record across {n} Cost Grid Points:")
    print(f"  Stochastic > Deterministic: {sto_wins_det}/{n} ({sto_wins_det/n*100:.0f}%)")
    print(f"  Stochastic > Greedy:        {sto_wins_grd}/{n} ({sto_wins_grd/n*100:.0f}%)")
    print(f"  Deterministic > Greedy:     {det_wins_grd}/{n} ({det_wins_grd/n*100:.0f}%)")

    return df, results_dir


if __name__ == "__main__":
    df, results_dir = run_ablation_study()
    print(f"\n[Done] All cost ablation results saved to {results_dir}")