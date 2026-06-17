import time
import os
import gurobipy as gp
from ortools.linear_solver.python import model_builder
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

load_dotenv()

# Problem definitions
# PROBLEMS = [
#     {"name": "Ship Tracking", "file": "problems/isolated_benchmark_problem.mps", "size": "small"},
#     {"name": "Volcano Scheduling", "file": "problems/isolated_benchmark_problem_large.mps", "size": "large"}
# ]
PROBLEMS = [
    {"name": "Volcano Scheduling", "file": "problems/isolated_benchmark_problem_large.mps", "size": "large"}
]

GAP_TESTS = [0.0, 0.01, 0.02, 0.05]
NUM_RUNS = 2  # Number of runs per problem/solver combo
RESULTS = {}

def benchmark_gurobi(mps_file, gap_tests, num_runs):
    """Run Gurobi with different MIP gaps, multiple times."""
    print("  Starting Gurobi benchmarks...")
    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)

        if os.environ.get("WLSACCESSID"):
            env.setParam("WLSACCESSID", os.environ.get("WLSACCESSID"))
            env.setParam("WLSSECRET", os.environ.get("WLSSECRET"))
            env.setParam("LICENSEID", int(os.environ.get("LICENSEID", 0)))
        env.start()
    except Exception as e:
        print(f"  ERROR: Failed to initialize Gurobi environment: {e}")
        return None

    results = []
    for gap in gap_tests:
        times = []
        objs = []

        for run in range(num_runs):
            try:
                m = gp.read(mps_file, env=env)
                m.Params.MIPGap = gap

                start_time = time.time()
                m.optimize()
                duration = time.time() - start_time

                obj_val = m.ObjVal if m.SolCount > 0 else float('nan')
                times.append(duration)
                objs.append(obj_val)
                print(f"    Gurobi {gap*100}% gap, run {run+1}/{num_runs}: {duration:.2f}s")
            except Exception as e:
                print(f"    WARNING: Gurobi run {run+1} failed: {e}")
                continue

        if not times:
            print(f"    ERROR: All Gurobi runs failed for gap {gap*100}%")
            continue

        # Convert to numpy arrays and filter out NaN/inf
        times_arr = np.array(times, dtype=float)
        objs_arr = np.array(objs, dtype=float)

        results.append({
            "gap": gap * 100,
            "times": times_arr.tolist(),
            "time_mean": float(np.mean(times_arr)),
            "time_std": float(np.std(times_arr)) if len(times_arr) > 1 else 0.0,
            "obj_mean": float(np.nanmean(objs_arr)),
            "obj_std": float(np.nanstd(objs_arr)) if len(objs_arr) > 1 else 0.0
        })

    return results if results else None

def benchmark_scip(mps_file, num_runs):
    """Run SCIP baseline, multiple times."""
    print("  Starting SCIP baseline...")

    times = []
    objs = []

    for run in range(num_runs):
        try:
            model = model_builder.ModelBuilder()
            if not model.import_from_mps_file(mps_file):
                print(f"    WARNING: Failed to load MPS file into SCIP on run {run+1}")
                continue

            scip_solver = model_builder.ModelSolver("SCIP")
            scip_solver.set_time_limit_in_seconds(500)

            start_time = time.time()
            scip_solver.solve(model)
            duration = time.time() - start_time

            obj = scip_solver.objective_value
            times.append(duration)
            objs.append(obj)
            print(f"    SCIP run {run+1}/{num_runs}: {duration:.2f}s")
        except Exception as e:
            print(f"    WARNING: SCIP run {run+1} failed: {e}")
            continue

    if not times:
        print("    ERROR: All SCIP runs failed")
        return None

    # Convert to numpy arrays and filter out NaN/inf
    times_arr = np.array(times, dtype=float)
    objs_arr = np.array(objs, dtype=float)

    return {
        "times": times_arr.tolist(),
        "time_mean": float(np.mean(times_arr)),
        "time_std": float(np.std(times_arr)) if len(times_arr) > 1 else 0.0,
        "obj_mean": float(np.nanmean(objs_arr)),
        "obj_std": float(np.nanstd(objs_arr)) if len(objs_arr) > 1 else 0.0
    }

def plot_results(results, output_dir="plots"):
    """Generate comparison plots."""
    Path(output_dir).mkdir(exist_ok=True)

    prob_names = list(results.keys())

    # Plot 1: Time vs Gap for each problem (with error bars)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (prob_name, data) in enumerate(results.items()):
        ax = axes[idx]

        # Gurobi times at different gaps with error bars
        if data["gurobi"]:
            gaps = [r["gap"] for r in data["gurobi"]]
            times_mean = [r["time_mean"] for r in data["gurobi"]]
            times_std = [r["time_std"] for r in data["gurobi"]]

            ax.errorbar(gaps, times_mean, yerr=times_std, marker='o', linewidth=2,
                        markersize=8, capsize=5, capthick=2, label='Gurobi')

        # SCIP baseline with error bar (if available)
        if data["scip"]:
            scip_mean = data["scip"]["time_mean"]
            scip_std = data["scip"]["time_std"]
            ax.axhline(scip_mean, color='red', linestyle='--', linewidth=2, label='SCIP')
            if data["gurobi"]:
                gaps = [r["gap"] for r in data["gurobi"]]
                ax.fill_between(gaps, scip_mean - scip_std, scip_mean + scip_std,
                                color='red', alpha=0.2)

        ax.set_xlabel('MIP Gap (%)', fontsize=12)
        ax.set_ylabel('Solve Time (s)', fontsize=12)
        ax.set_title(f'{prob_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_vs_gap.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/time_vs_gap.png")

    # Plot 2: Solver comparison (grouped bar chart with error bars)
    fig, ax = plt.subplots(figsize=(12, 6))

    prob_names = list(results.keys())
    x = np.arange(len(prob_names))
    width = 0.15

    # SCIP bars with error bars (if available)
    scip_available = [results[p]["scip"] is not None for p in prob_names]
    if any(scip_available):
        scip_means = [results[p]["scip"]["time_mean"] if results[p]["scip"] else 0 for p in prob_names]
        scip_stds = [results[p]["scip"]["time_std"] if results[p]["scip"] else 0 for p in prob_names]
        # Only plot bars where SCIP data exists
        for i, pname in enumerate(prob_names):
            if results[pname]["scip"]:
                ax.bar(x[i] - 2*width, scip_means[i], width, yerr=scip_stds[i],
                       label='SCIP' if i == 0 else '', color='#e74c3c', capsize=3)

    # Gurobi bars for each gap with error bars
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, gap in enumerate(GAP_TESTS):
        gap_means = []
        gap_stds = []
        for p in prob_names:
            if results[p]["gurobi"] and i < len(results[p]["gurobi"]):
                gap_means.append(results[p]["gurobi"][i]["time_mean"])
                gap_stds.append(results[p]["gurobi"][i]["time_std"])
            else:
                gap_means.append(0)
                gap_stds.append(0)

        # Only plot bars where data exists
        for j, pname in enumerate(prob_names):
            if gap_means[j] > 0:
                ax.bar(x[j] + (i-1)*width, gap_means[j], width, yerr=gap_stds[j],
                       label=f'Gurobi {gap*100}%' if j == 0 else '', color=colors[i], capsize=3)

    ax.set_xlabel('Problem', fontsize=12)
    ax.set_ylabel('Solve Time (s)', fontsize=12)
    ax.set_title('Solver Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prob_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/solver_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/solver_comparison.png")

    # Plot 3: Speedup comparison (SCIP baseline = 1.0) with error propagation
    fig, ax = plt.subplots(figsize=(10, 6))

    has_speedup_data = False
    for prob_name in prob_names:
        # Only plot speedup if both SCIP and Gurobi succeeded
        if results[prob_name]["scip"] and results[prob_name]["gurobi"]:
            has_speedup_data = True
            scip_mean = results[prob_name]["scip"]["time_mean"]
            speedup_means = [scip_mean / r["time_mean"] for r in results[prob_name]["gurobi"]]
            gaps = [r["gap"] for r in results[prob_name]["gurobi"]]

            # Error propagation for speedup (simplified)
            speedup_stds = []
            for r in results[prob_name]["gurobi"]:
                # Relative error propagation: σ(a/b) ≈ (a/b) * sqrt((σ_a/a)^2 + (σ_b/b)^2)
                rel_err = np.sqrt((results[prob_name]["scip"]["time_std"]/scip_mean)**2 +
                                 (r["time_std"]/r["time_mean"])**2)
                speedup_stds.append((scip_mean / r["time_mean"]) * rel_err)

            ax.errorbar(gaps, speedup_means, yerr=speedup_stds, marker='o', linewidth=2,
                       markersize=8, capsize=5, capthick=2, label=prob_name)

    if has_speedup_data:
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('MIP Gap (%)', fontsize=12)
        ax.set_ylabel('Speedup vs SCIP', fontsize=12)
        ax.set_title('Gurobi Speedup Relative to SCIP', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/speedup_comparison.png")
    else:
        plt.close(fig)
        print(f"Skipped: {output_dir}/speedup_comparison.png (no SCIP data for speedup)")

    # Plot 4: Objective value vs Gap for each problem
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (prob_name, data) in enumerate(results.items()):
        ax = axes[idx]

        # Gurobi objectives at different gaps with error bars
        if data["gurobi"]:
            gaps = [r["gap"] for r in data["gurobi"]]
            obj_means = [r["obj_mean"] for r in data["gurobi"]]
            obj_stds = [r["obj_std"] for r in data["gurobi"]]

            ax.errorbar(gaps, obj_means, yerr=obj_stds, marker='o', linewidth=2,
                        markersize=8, capsize=5, capthick=2, label='Gurobi')

        # SCIP baseline with error bar (if available)
        if data["scip"]:
            scip_obj_mean = data["scip"]["obj_mean"]
            scip_obj_std = data["scip"]["obj_std"]
            ax.axhline(scip_obj_mean, color='red', linestyle='--', linewidth=2, label='SCIP')
            if data["gurobi"]:
                gaps = [r["gap"] for r in data["gurobi"]]
                ax.fill_between(gaps, scip_obj_mean - scip_obj_std, scip_obj_mean + scip_obj_std,
                                color='red', alpha=0.2)

        ax.set_xlabel('MIP Gap (%)', fontsize=12)
        ax.set_ylabel('Objective Value', fontsize=12)
        ax.set_title(f'{prob_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/objective_vs_gap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/objective_vs_gap.png")

    # Plot 5: Objective value comparison (grouped bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(prob_names))
    width = 0.15

    # SCIP bars with error bars (if available)
    if any(results[p]["scip"] for p in prob_names):
        for i, pname in enumerate(prob_names):
            if results[pname]["scip"]:
                scip_obj_mean = results[pname]["scip"]["obj_mean"]
                scip_obj_std = results[pname]["scip"]["obj_std"]
                ax.bar(x[i] - 2*width, scip_obj_mean, width, yerr=scip_obj_std,
                       label='SCIP' if i == 0 else '', color='#e74c3c', capsize=3)

    # Gurobi bars for each gap with error bars
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, gap in enumerate(GAP_TESTS):
        for j, pname in enumerate(prob_names):
            if results[pname]["gurobi"] and i < len(results[pname]["gurobi"]):
                gap_obj_mean = results[pname]["gurobi"][i]["obj_mean"]
                gap_obj_std = results[pname]["gurobi"][i]["obj_std"]
                ax.bar(x[j] + (i-1)*width, gap_obj_mean, width, yerr=gap_obj_std,
                       label=f'Gurobi {gap*100}%' if j == 0 else '', color=colors[i], capsize=3)

    ax.set_xlabel('Problem', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Objective Value Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prob_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/objective_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/objective_comparison.png")

    # Plot 6: Time vs Objective tradeoff (Pareto-style)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, prob_name in enumerate(prob_names):
        ax = axes[idx]
        data = results[prob_name]

        # Gurobi points
        if data["gurobi"]:
            for r in data["gurobi"]:
                ax.errorbar(r["time_mean"], r["obj_mean"],
                           xerr=r["time_std"], yerr=r["obj_std"],
                           marker='o', markersize=8, capsize=4,
                           label=f'Gurobi {r["gap"]:.0f}%')

        # SCIP point (if available)
        if data["scip"]:
            ax.errorbar(data["scip"]["time_mean"], data["scip"]["obj_mean"],
                       xerr=data["scip"]["time_std"], yerr=data["scip"]["obj_std"],
                       marker='s', markersize=10, capsize=4, color='red',
                       label='SCIP')

        ax.set_xlabel('Solve Time (s)', fontsize=12)
        ax.set_ylabel('Objective Value', fontsize=12)
        ax.set_title(f'{prob_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_vs_objective_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/time_vs_objective_tradeoff.png")

def print_summary(results):
    """Print text summary table with statistics."""
    print("\n" + "=" * 90)
    print(f"{'Problem':<25} | {'Solver':<20} | {'Time (s)':<20} | {'Objective':<20}")
    print("-" * 90)

    for prob_name, data in results.items():
        # SCIP (if available)
        if data['scip']:
            scip_str = f"{data['scip']['time_mean']:.2f} ± {data['scip']['time_std']:.2f}"
            obj_str = f"{data['scip']['obj_mean']:.2f} ± {data['scip']['obj_std']:.2f}"
            print(f"{prob_name:<25} | {'SCIP':<20} | {scip_str:<20} | {obj_str:<20}")
        else:
            print(f"{prob_name:<25} | {'SCIP':<20} | {'TIMED OUT':<20} | {'N/A':<20}")

        # Gurobi (if available)
        if data["gurobi"]:
            for res in data["gurobi"]:
                solver_name = f"Gurobi ({res['gap']}% Gap)"
                time_str = f"{res['time_mean']:.2f} ± {res['time_std']:.2f}"
                obj_str = f"{res['obj_mean']:.2f} ± {res['obj_std']:.2f}"
                print(f"{'':<25} | {solver_name:<20} | {time_str:<20} | {obj_str:<20}")
        else:
            print(f"{'':<25} | {'Gurobi (ALL)':<20} | {'FAILED':<20} | {'N/A':<20}")

        print("-" * 90)

    print("=" * 90)

# Main execution
if __name__ == "__main__":
    print("=== MIP SOLVER BENCHMARK ===\n")

    for problem in PROBLEMS:
        print(f"\nBenchmarking: {problem['name']} ({problem['file']})")

        if not os.path.exists(problem['file']):
            print(f"  WARNING: File not found, skipping...")
            continue

        # Run benchmarks
        try:
            gurobi_results = benchmark_gurobi(problem['file'], GAP_TESTS, NUM_RUNS)
            scip_results = benchmark_scip(problem['file'], NUM_RUNS)

            # Store results if at least one solver succeeded
            if gurobi_results or scip_results:
                RESULTS[problem['name']] = {
                    "gurobi": gurobi_results,
                    "scip": scip_results
                }
                if not gurobi_results:
                    print(f"  WARNING: Gurobi failed for {problem['name']}, plotting SCIP only")
                if not scip_results:
                    print(f"  WARNING: SCIP failed/timed out for {problem['name']}, plotting Gurobi only")
            else:
                print(f"  WARNING: All solvers failed for {problem['name']}, skipping...")
        except Exception as e:
            print(f"  ERROR: Benchmark failed for {problem['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate plots
    if RESULTS:
        print("\n" + "=" * 80)
        print("Generating plots...")
        try:
            plot_results(RESULTS)
        except Exception as e:
            print(f"ERROR: Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()

        # Print summary
        try:
            print_summary(RESULTS)
        except Exception as e:
            print(f"ERROR: Failed to print summary: {e}")
            import traceback
            traceback.print_exc()

        print("\nBenchmark complete!")
    else:
        print("\nNo results to plot. Check errors above.")
