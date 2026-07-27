"""
Post-processing script for normal run results (volcano Monte Carlo comparisons).
Reads individual run JSON files across multiple timestamped folders and generates summary plots
focused strictly on Utility (computed as Quality - Cost), Task Completion, Quality, and Cost.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURATION — Define your run folders here
# ==============================================================================
# Base directory where results are stored
BASE_DIR = "/Users/davidf/Code/federated-scheduling/results"

# Folder names relative to BASE_DIR (or full absolute paths)
RUN_FOLDERS = [
    "volcano_2026-07-21_151739",
    "volcano_2026-07-21_161936",
    "volcano_2026-07-21_164001",
    "volcano_2026-07-22_092250",
    "volcano_2026-07-22_142732",
    "volcano_2026-07-22_172517"
]

# Output folder where combined plots & summary report will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_results")
# ==============================================================================

# Global Matplotlib Styling Configuration for Large Text
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

SCHEDULER_COLORS = {
    'greedy': '#E07B54',
    'deterministic': '#5B8DB8',
    'stochastic_log': '#6AAB6E',
    'stochastic': '#6AAB6E',
}

SCHEDULER_LABELS = {
    'greedy': 'Greedy',
    'deterministic': 'Deterministic ILP',
    'stochastic_log': 'Stochastic Log',
    'stochastic': 'Stochastic',
}


def _color(scheduler):
    return SCHEDULER_COLORS.get(scheduler, '#888888')


def _label(scheduler):
    return SCHEDULER_LABELS.get(scheduler, scheduler.capitalize())


def load_results_from_jsons(folders, base_dir="."):
    """
    Scans specified directories for run JSON files, parses them, and normalizes
    column names for post-processing and visualization.
    Recalculates utility explicitly as Quality - Cost.
    """
    records = []
    folder_run_key_map = {}
    next_global_run_id = 1

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.is_absolute():
            folder_path = Path(base_dir) / folder

        if not folder_path.exists():
            print(f"  [Warning] Folder not found: {folder_path} — Skipping.")
            continue

        json_files = list(folder_path.glob("*.json"))
        if not json_files:
            print(f"  [Warning] No JSON files found in {folder_path} — Skipping.")
            continue

        loaded_count = 0
        folder_name = folder_path.name

        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)

                if "scheduler" not in data:
                    continue

                data['folder'] = folder_name
                data['source_file'] = jf.name

                local_run_id = data.get('run', data.get('seed', jf.stem))
                mapping_key = (folder_name, local_run_id)

                if mapping_key not in folder_run_key_map:
                    folder_run_key_map[mapping_key] = next_global_run_id
                    next_global_run_id += 1

                data['run'] = folder_run_key_map[mapping_key]
                records.append(data)
                loaded_count += 1

            except Exception as e:
                print(f"  [Warning] Could not parse {jf.name}: {e}")

        print(f"  [Folder] {folder_name}: loaded {loaded_count} run JSONs")

    if not records:
        raise FileNotFoundError("No valid run JSON files were found in the provided directories.")

    df = pd.DataFrame(records)

    # Standardize & Normalize Quality and Cost fields
    if 'quality' not in df.columns and 'realized_quality' in df.columns:
        df['quality'] = df['realized_quality']

    if 'cost' not in df.columns and 'total_cost' in df.columns:
        df['cost'] = df['total_cost']

    # Force utility calculation: Utility = Quality - Cost
    if 'quality' in df.columns and 'cost' in df.columns:
        df['utility'] = df['quality'] - df['cost']
        print("  [Info] Calculated Utility directly as (Quality - Cost)")

    if 'task_completion_rate_pct' not in df.columns and 'task_completion_rate' in df.columns:
        df['task_completion_rate_pct'] = df['task_completion_rate'] * 100.0

    if 'scheduled' not in df.columns:
        if 'n_tasks_completed' in df.columns:
            df['scheduled'] = df['n_tasks_completed']
        elif 'n_accepted' in df.columns:
            df['scheduled'] = df['n_accepted']

    return df


def plot_metric_boxplots(df, output_dir):
    """Box plots showing distribution of key metrics per scheduler across runs."""
    print("  Generating metric boxplots...")

    metrics = [
        ('utility', 'Utility (Quality - Cost)', True),
        ('task_completion_rate_pct', 'Task Completion Rate (%)', True),
        ('n_tasks_completed', 'Tasks Completed', True),
        ('quality', 'Realized Quality', True),
        ('cost', 'Total Cost', False),
        ('completions_per_cost', 'Completions / Cost', True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    schedulers = df['scheduler'].unique()

    for idx, (metric, title, higher_better) in enumerate(metrics):
        ax = axes[idx]
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Column\n"{metric}"\nnot found', ha='center', va='center',
                    transform=ax.transAxes, color='grey', fontsize=20)
            ax.set_title(title, fontsize=20)
            continue

        data_per_sched = [df[df['scheduler'] == s][metric].dropna().values for s in schedulers]
        colors = [_color(s) for s in schedulers]
        labels = [_label(s) for s in schedulers]

        bp = ax.boxplot(data_per_sched, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        for i, data in enumerate(data_per_sched):
            if len(data) > 0:
                ax.scatter(i + 1, np.mean(data), marker='D', color='black',
                           zorder=5, s=50, label='Mean' if i == 0 else '')

        ax.set_xticks(range(1, len(schedulers) + 1))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=20)
        ax.set_ylabel(title, fontsize=17)

        if higher_better is True:
            ax.set_title(title + ' ↑', fontsize=15)
        elif higher_better is False:
            ax.set_title(title + ' ↓', fontsize=15)
        else:
            ax.set_title(title, fontsize=15)

        ax.grid(axis='y', alpha=0.4)

    legend_handle = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                                markersize=8, label='Mean')
    fig.legend(handles=[legend_handle], loc='lower right', frameon=True, fontsize=18)

    plt.suptitle('Scheduler Performance — Distribution Across Runs', fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_boxplots.png'), bbox_inches='tight')
    print("  [Saved] metric_boxplots.png")
    plt.close()


def plot_mean_bars(df, output_dir):
    """Bar chart of means ± std for core performance metrics."""
    print("  Generating mean bar charts...")

    metrics = [
        ('utility', 'Utility (Quality - Cost)'),
        ('task_completion_rate_pct', 'Task Completion (%)'),
        ('n_tasks_completed', 'Tasks Completed'),
        # ('quality', 'Realized Quality'),
        # ('cost', 'Total Cost'),
    ]

    schedulers = df['scheduler'].unique()
    x = np.arange(len(schedulers))

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Missing:\n{metric}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=15)
            continue

        means = [df[df['scheduler'] == s][metric].mean() for s in schedulers]
        stds = [df[df['scheduler'] == s][metric].std() for s in schedulers]
        colors = [_color(s) for s in schedulers]
        labels = [_label(s) for s in schedulers]

        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.8,
                      error_kw=dict(elinewidth=2, ecolor='#333333'))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=13)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(title, fontsize=16, pad=10)
        ax.grid(axis='y', alpha=0.4)

        # Annotate numbers over bars with larger text
        max_val = max(means) if len(means) > 0 and max(means) > 0 else 1
        for bar, mean, std in zip(bars, means, stds):
            y_offset = (std if not np.isnan(std) else 0) + 0.02 * max_val
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle('Scheduler Comparison — Means ± Std', fontsize=18, y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_bars.png'), bbox_inches='tight')
    print("  [Saved] mean_bars.png")
    plt.close()


def plot_per_run_lines(df, output_dir):
    """Line plots showing primary metrics per Monte Carlo run across all schedulers."""
    print("  Generating per-run line plots...")

    metrics = [
        ('utility', 'Utility (Quality - Cost)'),
        ('task_completion_rate_pct', 'Task Completion Rate (%)'),
        ('n_tasks_completed', 'Tasks Completed'),
        ('quality', 'Realized Quality'),
    ]

    schedulers = df['scheduler'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Missing:\n{metric}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=15)
            continue

        for sched in schedulers:
            sub = df[df['scheduler'] == sched].sort_values('run')
            ax.plot(sub['run'], sub[metric], marker='o', markersize=6,
                    color=_color(sched), label=_label(sched), alpha=0.85, linewidth=2)
            mean_val = sub[metric].mean()
            ax.axhline(mean_val, color=_color(sched), linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_xlabel('Global Run Index', fontsize=13)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.4)

    plt.suptitle('Per-Run Results Across All Folders (dashed = mean)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_run_lines.png'), bbox_inches='tight')
    print("  [Saved] per_run_lines.png")
    plt.close()


def plot_utility_vs_completion(df, output_dir):
    """Scatter of Utility vs Task Completion Rate — key efficacy view."""
    print("  Generating utility vs completion rate scatter...")

    if 'utility' not in df.columns or 'task_completion_rate_pct' not in df.columns:
        print("  Skipping: missing utility or task_completion_rate_pct columns")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    schedulers = df['scheduler'].unique()

    for sched in schedulers:
        sub = df[df['scheduler'] == sched]
        ax.scatter(sub['task_completion_rate_pct'], sub['utility'],
                   color=_color(sched), label=_label(sched),
                   s=90, alpha=0.75, edgecolors='white', linewidths=0.8)
        mean_x = sub['task_completion_rate_pct'].mean()
        mean_y = sub['utility'].mean()
        std_x = sub['task_completion_rate_pct'].std()
        std_y = sub['utility'].std()
        ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y,
                    fmt='D', color=_color(sched), markersize=12,
                    capsize=6, linewidth=2.5, zorder=5)

    ax.set_xlabel('Task Completion Rate (%)', fontsize=14)
    ax.set_ylabel('Utility (Quality - Cost)', fontsize=14)
    ax.set_title('Utility vs Task Completion Rate\n(diamonds = mean ± std, upper-right is better)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'utility_vs_completion.png'), bbox_inches='tight')
    print("  [Saved] utility_vs_completion.png")
    plt.close()


def plot_quality_vs_cost(df, output_dir):
    """Scatter of Quality vs Cost — trade-off view."""
    print("  Generating quality vs cost scatter...")

    if 'quality' not in df.columns or 'cost' not in df.columns:
        print("  Skipping: missing quality or cost columns")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    schedulers = df['scheduler'].unique()

    for sched in schedulers:
        sub = df[df['scheduler'] == sched]
        ax.scatter(sub['cost'], sub['quality'],
                   color=_color(sched), label=_label(sched),
                   s=90, alpha=0.75, edgecolors='white', linewidths=0.8)
        mean_x = sub['cost'].mean()
        mean_y = sub['quality'].mean()
        std_x = sub['cost'].std()
        std_y = sub['quality'].std()
        ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y,
                    fmt='D', color=_color(sched), markersize=12,
                    capsize=6, linewidth=2.5, zorder=5)

    ax.set_xlabel('Total Cost', fontsize=14)
    ax.set_ylabel('Realized Quality', fontsize=14)
    ax.set_title('Realized Quality vs Total Cost Trade-off\n(diamonds = mean ± std, upper-left is better)', fontsize=20)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_vs_cost.png'), bbox_inches='tight')
    print("  [Saved] quality_vs_cost.png")
    plt.close()


def plot_scheduler_advantage(df, output_dir):
    """
    Bar chart showing advantage of each non-baseline scheduler over greedy
    for Utility, Task Completion Rate, and Realized Quality.
    """
    print("  Generating scheduler advantage chart...")

    if 'greedy' not in df['scheduler'].values:
        print("  Skipping: no 'greedy' scheduler found for baseline comparison")
        return

    metrics = [
        ('utility', 'Utility (Quality - Cost)'),
        ('task_completion_rate_pct', 'Task Completion (%)'),
        ('quality', 'Realized Quality'),
    ]

    baseline = 'greedy'
    others = [s for s in df['scheduler'].unique() if s != baseline]

    if not others:
        print("  Skipping: only one scheduler present")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6))

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Missing:\n{metric}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{title} Advantage over Greedy', fontsize=15)
            continue

        baseline_mean = df[df['scheduler'] == baseline][metric].mean()
        labels, advantages, errors, colors = [], [], [], []

        for sched in others:
            sub = df[df['scheduler'] == sched][metric]
            adv = sub.mean() - baseline_mean

            merged = df[df['scheduler'].isin([baseline, sched])].pivot(
                index='run', columns='scheduler', values=metric)
            if baseline in merged.columns and sched in merged.columns:
                diffs = (merged[sched] - merged[baseline]).dropna()
                err = diffs.std() if len(diffs) > 1 else 0
            else:
                err = 0

            labels.append(_label(sched))
            advantages.append(adv)
            errors.append(err)
            colors.append(_color(sched))

        x = np.arange(len(labels))
        bars = ax.bar(x, advantages, yerr=errors, capsize=6, color=colors, alpha=0.8,
                      error_kw=dict(elinewidth=2, ecolor='#333333'))
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)
        ax.set_ylabel(f'Δ {title} vs Greedy', fontsize=13)
        ax.set_title(f'{title} Advantage over Greedy', fontsize=15)
        ax.grid(axis='y', alpha=0.4)

        for bar, val, err in zip(bars, advantages, errors):
            sign = '+' if val >= 0 else ''
            y_offset = (err if not np.isnan(err) else 0) + 0.02 * max(abs(v) for v in advantages + [1])
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    f'{sign}{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle('Advantage over Greedy Baseline (mean ± std of per-run differences)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scheduler_advantage.png'), bbox_inches='tight')
    print("  [Saved] scheduler_advantage.png")
    plt.close()


def generate_summary_report(df, output_dir, folder_list):
    """Text summary report focusing on primary performance metrics."""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    n_runs = df['run'].nunique()
    schedulers = df['scheduler'].unique()

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMBINED JSON RUNS — PRIMARY METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Source folders ({len(folder_list)}):\n")
        for folder in folder_list:
            f.write(f"  - {folder}\n")
        f.write(f"\nTotal aggregated Monte Carlo runs: {n_runs}\n")
        f.write(f"Schedulers: {', '.join(schedulers)}\n\n")

        metrics = [
            ('utility', 'Utility (Quality - Cost)'),
            ('task_completion_rate_pct', 'Task Completion Rate (%)'),
            ('n_tasks_completed', 'Tasks Completed'),
            ('quality', 'Realized Quality'),
            ('cost', 'Total Cost'),
            ('completions_per_cost', 'Completions / Cost'),
        ]

        f.write("PER-SCHEDULER STATISTICS\n")
        f.write("-" * 70 + "\n")
        for sched in schedulers:
            sub = df[df['scheduler'] == sched]
            f.write(f"\n{_label(sched).upper()} ({sched}):\n")
            for col, label in metrics:
                if col in sub.columns:
                    mean = sub[col].mean()
                    std = sub[col].std()
                    f.write(f"  {label:<30} {mean:>10.3f} ± {std:.3f}\n")

        if 'greedy' in df['scheduler'].values:
            f.write("\n" + "=" * 70 + "\n")
            f.write("ADVANTAGE OVER GREEDY BASELINE\n")
            f.write("-" * 70 + "\n")
            greedy = df[df['scheduler'] == 'greedy']
            for sched in [s for s in schedulers if s != 'greedy']:
                sub = df[df['scheduler'] == sched]
                f.write(f"\n{_label(sched).upper()} vs Greedy:\n")
                for col, label in [('utility', 'Utility (Quality - Cost)'), ('task_completion_rate_pct', 'Task Completion (%)'),
                                    ('n_tasks_completed', 'Tasks Completed'), ('quality', 'Realized Quality')]:
                    if col in sub.columns:
                        delta = sub[col].mean() - greedy[col].mean()
                        sign = '+' if delta >= 0 else ''
                        pct = (delta / greedy[col].mean() * 100) if greedy[col].mean() != 0 else 0
                        f.write(f"  {label:<30} {sign}{delta:>8.2f}  ({sign}{pct:.1f}%)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("BEST RUN PER SCHEDULER (by utility)\n")
        f.write("-" * 70 + "\n")
        if 'utility' in df.columns:
            for sched in schedulers:
                sub = df[df['scheduler'] == sched]
                if not sub.empty:
                    best = sub.loc[sub['utility'].idxmax()]
                    f.write(f"\n{_label(sched).upper()}:\n")
                    f.write(f"  Global Run ID:     {int(best['run'])}\n")
                    f.write(f"  Folder:            {best.get('folder', 'N/A')}\n")
                    f.write(f"  Source JSON:       {best.get('source_file', 'N/A')}\n")
                    f.write(f"  Utility:           {best['utility']:.3f}\n")
                    if 'task_completion_rate_pct' in best:
                        f.write(f"  Task Completion:   {best['task_completion_rate_pct']:.2f}%\n")
                    if 'n_tasks_completed' in best:
                        f.write(f"  Tasks Completed:   {int(best['n_tasks_completed'])}\n")

    print("  [Saved] summary_report.txt")


def main():
    print(f"\n{'='*70}")
    print("MULTIPLE RUNS JSON POST-PROCESSING — PRIMARY METRICS")
    print(f"{'='*70}\n")
    print(f"Loading JSON results from {len(RUN_FOLDERS)} folders...")

    try:
        all_runs_df = load_results_from_jsons(RUN_FOLDERS, BASE_DIR)
        n_runs = all_runs_df['run'].nunique()
        schedulers = list(all_runs_df['scheduler'].unique())
        print(f"\n  Successfully loaded {len(all_runs_df)} run results")
        print(f"  Aggregated dataset: {n_runs} total runs across {len(schedulers)} schedulers")
        print(f"  Schedulers: {schedulers}")
    except Exception as e:
        print(f"ERROR loading results: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating combined plots in: {OUTPUT_DIR}\n")

    plot_functions = [
        ('Metric Boxplots', lambda df, od: plot_metric_boxplots(df, od)),
        ('Mean Bar Charts', lambda df, od: plot_mean_bars(df, od)),
        ('Per-Run Lines', lambda df, od: plot_per_run_lines(df, od)),
        ('Utility vs Task Completion', lambda df, od: plot_utility_vs_completion(df, od)),
        ('Quality vs Cost', lambda df, od: plot_quality_vs_cost(df, od)),
        ('Scheduler Advantage', lambda df, od: plot_scheduler_advantage(df, od)),
        ('Summary Report', lambda df, od: generate_summary_report(df, od, RUN_FOLDERS)),
    ]

    failed = []
    for name, fn in plot_functions:
        try:
            fn(all_runs_df, OUTPUT_DIR)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            failed.append((name, str(e)))

    print(f"\n{'='*70}")
    if failed:
        print(f"POST-PROCESSING COMPLETE (with {len(failed)} errors)")
        for name, err in failed:
            print(f"  x {name}: {err[:80]}")
    else:
        print("POST-PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nCombined outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()