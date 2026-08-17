"""
Post-processing script for normal run results (volcano Monte Carlo comparisons).
Reads individual run JSON files across multiple timestamped folders and generates summary plots
focused on Total & Per-Completed-Task Utility, Quality, Cost, Task Completion Rate (%), Wall Time, MIP Gap,
and both 2D and 3D Pareto trade-off curves.
"""

import json
import os
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# CONFIGURATION — Define your run folders here
# ==============================================================================
BASE_DIR = "/Users/davidf/Code/federated-scheduling/results"

RUN_FOLDERS = [
    "earthquake_campaign_20260815_232847"
]

OUTPUT_DIR = os.path.join(BASE_DIR, RUN_FOLDERS[0], "combined_results")
# ==============================================================================

# Global Matplotlib Styling Configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# High-contrast, color-blind-friendly method palette.  Method identity is kept
# categorical in every Pareto plot; completion rate is represented separately.
SCHEDULER_COLORS = {
    'greedy': '#E69F00',          # orange
    'random': '#D55E00',          # vermilion
    'deterministic': '#0072B2',   # blue
    'stochastic_log': '#009E73',  # green
    'stochastic': '#CC79A7',      # magenta
}

SCHEDULER_LABELS = {
    'greedy': 'Greedy',
    'random': 'Random',
    'deterministic': 'Deterministic ILP',
    'stochastic_log': 'Stochastic Log',
    'stochastic': 'Stochastic',
}

SCHEDULER_MARKERS = {
    'greedy': 'o',           # Circle
    'random': 'X',           # Cross
    'deterministic': 's',    # Square
    'stochastic_log': '^',   # Triangle
    'stochastic': 'P',       # Filled plus (diamonds are reserved for means)
}

SCHEDULER_LINESTYLES = {
    'greedy': '-',
    'random': '--',
    'deterministic': '-.',
    'stochastic_log': ':',
    'stochastic': (0, (5, 1, 1, 1)),
}

# Used only if a result file contains a scheduler not listed above.  This keeps
# new methods from silently sharing the same gray fallback style.
EXTRA_COLORS = [
    '#56B4E9', '#F0E442', '#332288', '#88CCEE', '#44AA99',
    '#117733', '#999933', '#882255', '#AA4499', '#661100',
]
EXTRA_MARKERS = ['P', 'v', '<', '>', '*', 'h', 'p', '8', 'd', 'H']
EXTRA_LINESTYLES = [
    (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2)), (0, (2, 1)),
    (0, (4, 1, 1, 1, 1, 1)),
]


def register_scheduler_styles(schedulers):
    """Assign a unique color/marker/line style to any unconfigured method."""
    unknown = [s for s in schedulers if s not in SCHEDULER_COLORS]
    for i, sched in enumerate(unknown):
        SCHEDULER_COLORS[sched] = EXTRA_COLORS[i % len(EXTRA_COLORS)]
        SCHEDULER_MARKERS[sched] = EXTRA_MARKERS[i % len(EXTRA_MARKERS)]
        SCHEDULER_LINESTYLES[sched] = EXTRA_LINESTYLES[i % len(EXTRA_LINESTYLES)]


def _color(scheduler):
    return SCHEDULER_COLORS.get(scheduler, '#888888')


def _label(scheduler):
    return SCHEDULER_LABELS.get(scheduler, scheduler.capitalize())


def _marker(scheduler):
    return SCHEDULER_MARKERS.get(scheduler, 'o')


def _linestyle(scheduler):
    return SCHEDULER_LINESTYLES.get(scheduler, '-')


def get_pareto_frontier_2d(costs, qualities):
    """Computes 2D non-dominated points assuming cost minimization (X) and quality maximization (Y)."""
    if len(costs) == 0:
        return [], []
    sorted_pairs = sorted(zip(costs, qualities), key=lambda x: (x[0], -x[1]))
    
    pareto_costs = []
    pareto_qualities = []
    max_quality = -float('inf')

    for c, q in sorted_pairs:
        if q > max_quality:
            pareto_costs.append(c)
            pareto_qualities.append(q)
            max_quality = q

    return pareto_costs, pareto_qualities


def get_pareto_frontier_3d(costs, qualities, completions):
    """Computes 3D non-dominated points (Cost ↓, Quality ↑, Task Completion Rate ↑)."""
    if len(costs) == 0:
        return np.array([]), np.array([]), np.array([])

    pts = list(zip(costs, qualities, completions))
    pareto_pts = []

    for i, p1 in enumerate(pts):
        dominated = False
        for j, p2 in enumerate(pts):
            if i != j:
                if (p2[0] <= p1[0] and p2[1] >= p1[1] and p2[2] >= p1[2]) and \
                   (p2[0] < p1[0] or p2[1] > p1[1] or p2[2] > p1[2]):
                    dominated = True
                    break
        if not dominated:
            pareto_pts.append(p1)

    if not pareto_pts:
        return np.array([]), np.array([]), np.array([])

    pareto_pts = np.array(pareto_pts)
    return pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2]


def load_results_from_jsons(folders, base_dir="."):
    """
    Scans directories for run JSON files and extracts fields matching your specific JSON schema.
    Calculates per-completed-task normalized metrics (Cost/Completed Task, Quality/Completed Task, Utility/Completed Task).
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
        skipped_empty_count = 0
        folder_name = folder_path.name

        for jf in json_files:
            if jf.stat().st_size == 0:
                skipped_empty_count += 1
                continue

            try:
                with open(jf, 'r') as f:
                    data = json.load(f)

                if not data or not isinstance(data, dict):
                    skipped_empty_count += 1
                    continue

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

            except (json.JSONDecodeError, Exception) as e:
                print(f"  [Warning] Could not parse {jf.name} ({e}) — Skipping.")
                skipped_empty_count += 1

        print(f"  [Folder] {folder_name}: loaded {loaded_count} run JSONs (skipped {skipped_empty_count} empty/invalid)")

    if not records:
        raise FileNotFoundError("No valid run JSON files were found in the provided directories.")

    df = pd.DataFrame(records)

    # --- Standardize JSON Fields ---
    df['quality'] = df.get('realized_quality', df.get('quality', 0.0))
    df['cost'] = df.get('total_cost', df.get('cost', 0.0))

    if 'utility' in df.columns:
        df['utility'] = df['utility']
    else:
        df['utility'] = df['quality'] - df['cost']

    if 'task_completion_rate_pct' not in df.columns and 'task_completion_rate' in df.columns:
        df['task_completion_rate_pct'] = df['task_completion_rate'] * 100.0

    # Determine Valid Completed Task Count (n_tasks_completed_valid)
    if 'n_tasks_completed_valid' in df.columns:
        df['n_tasks_completed_valid'] = df['n_tasks_completed_valid']
    elif 'n_tasks_completed' in df.columns:
        df['n_tasks_completed_valid'] = df['n_tasks_completed']
    elif 'n_executed' in df.columns:
        df['n_tasks_completed_valid'] = df['n_executed']
    else:
        df['n_tasks_completed_valid'] = df.get('n_accepted', 1)

    # --- Calculate Per-Completed-Task Normalized Metrics ---
    valid_completed = df['n_tasks_completed_valid'].replace(0, np.nan)
    df['cost_per_completed_task'] = df['cost'] / valid_completed
    df['quality_per_completed_task'] = df['quality'] / valid_completed
    df['utility_per_completed_task'] = df['utility'] / valid_completed

    print("  [Info] Loaded JSONs successfully. Normalized metrics calculated across 'n_tasks_completed_valid'.")
    return df


def plot_pareto_2d(df, output_dir):
    """Quality vs Cost with completion encoded by color and method by marker."""
    print("  Generating 2D Pareto frontier plots...")

    required_cols = ['quality', 'cost', 'task_completion_rate_pct']
    valid_df = df.dropna(subset=required_cols).copy()
    if valid_df.empty:
        print("  Skipping 2D Pareto plot: missing required columns.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    schedulers = valid_df['scheduler'].unique()

    vmin = valid_df['task_completion_rate_pct'].min()
    vmax = valid_df['task_completion_rate_pct'].max()
    completion_norm = Normalize(vmin=vmin, vmax=vmax)
    completion_mappable = ScalarMappable(norm=completion_norm, cmap='viridis')
    completion_mappable.set_array([])

    scatter_handles = []
    for sched in schedulers:
        sub = valid_df[valid_df['scheduler'] == sched]
        ax1.scatter(
            sub['cost'], sub['quality'],
            c=sub['task_completion_rate_pct'], cmap='viridis', norm=completion_norm,
            marker=_marker(sched), s=100,
            alpha=0.82, edgecolors='#222222', linewidths=0.65,
            zorder=3
        )
        scatter_handles.append(
            plt.Line2D(
                [0], [0], marker=_marker(sched), linestyle='none',
                color='#777777', markerfacecolor='#777777',
                markeredgecolor='#222222', markersize=9,
                label=_label(sched)
            )
        )

    p_costs, p_quals = get_pareto_frontier_2d(valid_df['cost'].values, valid_df['quality'].values)
    ax1.plot(p_costs, p_quals, color='#111111', linestyle='--', linewidth=2.8, zorder=10)
    scatter_handles.append(
        plt.Line2D([0], [0], color='#111111', linestyle='--', linewidth=2.8,
                   label='Global 2D Pareto')
    )

    ax1.set_xlabel('Total Cost ↓', fontsize=14)
    ax1.set_ylabel('Realized Quality ↑', fontsize=14)
    ax1.set_title(
        'Global 2D Pareto: Quality vs Cost\n'
        '(Color = completion rate; marker = method)', fontsize=15
    )
    ax1.legend(
        handles=scatter_handles, fontsize=10, loc='lower right', title='Method / frontier'
    )
    cbar1 = fig.colorbar(completion_mappable, ax=ax1, pad=0.02)
    cbar1.set_label('Task Completion Rate (%)', fontsize=12)
    ax1.grid(alpha=0.4)

    frontier_handles = []
    for sched in schedulers:
        sub = valid_df[valid_df['scheduler'] == sched]
        ax2.scatter(
            sub['cost'], sub['quality'],
            c=sub['task_completion_rate_pct'], cmap='viridis', norm=completion_norm,
            marker=_marker(sched), s=85,
            alpha=0.72, edgecolors='#222222', linewidths=0.55,
            zorder=3
        )
        p_c, p_q = get_pareto_frontier_2d(sub['cost'].values, sub['quality'].values)
        ax2.plot(
            p_c, p_q, color='#333333', linestyle=_linestyle(sched),
            linewidth=2.4, alpha=0.78, zorder=2
        )
        frontier_handles.append(
            plt.Line2D(
                [0], [0], marker=_marker(sched), linestyle=_linestyle(sched),
                color='#333333', markerfacecolor='#777777',
                markeredgecolor='#222222', linewidth=2.2, markersize=8,
                label=f'{_label(sched)} frontier'
            )
        )

    ax2.set_xlabel('Total Cost ↓', fontsize=14)
    ax2.set_ylabel('Realized Quality ↑', fontsize=14)
    ax2.set_title(
        'Per-Method 2D Pareto Frontiers\n'
        '(Color = completion rate; marker/line style = method)', fontsize=15
    )
    ax2.legend(handles=frontier_handles, fontsize=10, loc='lower right')
    cbar2 = fig.colorbar(completion_mappable, ax=ax2, pad=0.02)
    cbar2.set_label('Task Completion Rate (%)', fontsize=12)
    ax2.grid(alpha=0.4)

    plt.suptitle('2D Pareto Analysis — Realized Quality vs Total Cost', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_2d_quality_vs_cost.png'), bbox_inches='tight')
    print("  [Saved] pareto_2d_quality_vs_cost.png")
    plt.close()


def plot_pareto_3d(df, output_dir):
    """Generates 3D Pareto frontier plots (Cost ↓, Quality ↑, Task Completion Rate ↑)."""
    print("  Generating 3D Pareto frontier plots...")

    required_cols = ['cost', 'quality', 'task_completion_rate_pct']
    valid_df = df.dropna(subset=required_cols).copy()
    if valid_df.empty:
        print("  Skipping 3D Pareto plot: missing required columns.")
        return

    fig = plt.figure(figsize=(20, 8))
    schedulers = valid_df['scheduler'].unique()

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    scatter_handles = []

    for sched in schedulers:
        sub = valid_df[valid_df['scheduler'] == sched]
        ax1.scatter(
            sub['cost'], sub['quality'], sub['task_completion_rate_pct'],
            color=_color(sched), marker=_marker(sched), s=85, alpha=0.72,
            edgecolors='white', linewidths=0.8
        )
        scatter_handles.append(
            plt.Line2D(
                [0], [0], marker=_marker(sched), linestyle='none',
                color=_color(sched), markerfacecolor=_color(sched),
                markeredgecolor='white', markersize=9, label=_label(sched)
            )
        )

    p_costs, p_quals, p_comps = get_pareto_frontier_3d(
        valid_df['cost'].values,
        valid_df['quality'].values,
        valid_df['task_completion_rate_pct'].values
    )

    if len(p_costs) > 0:
        ax1.scatter(
            p_costs, p_quals, p_comps,
            facecolors='none', edgecolors='#111111', marker='o',
            s=190, alpha=1.0, linewidths=1.8, zorder=10
        )
        if len(p_costs) >= 3:
            try:
                ax1.plot_trisurf(
                    p_costs, p_quals, p_comps, color='#555555',
                    alpha=0.10, edgecolor='#555555', linewidth=0.4
                )
            except Exception:
                pass

        scatter_handles.append(
            plt.Line2D(
                [0], [0], marker='o', linestyle='none', color='#111111',
                markerfacecolor='none', markeredgewidth=1.8, markersize=11,
                label='Global Pareto point'
            )
        )

    ax1.set_xlabel('Total Cost ↓', fontsize=12, labelpad=10)
    ax1.set_ylabel('Realized Quality ↑', fontsize=12, labelpad=10)
    ax1.set_zlabel('Completion Rate (%) ↑', fontsize=12, labelpad=10)
    ax1.set_title(
        'Global 3D Pareto Frontier\n'
        '(Color/marker = method; black ring = Pareto)', fontsize=14
    )
    ax1.legend(handles=scatter_handles, fontsize=9, loc='upper left', title='Method')
    ax1.view_init(elev=25, azim=135)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    for sched in schedulers:
        sub = valid_df[valid_df['scheduler'] == sched]
        p_c, p_q, p_comp = get_pareto_frontier_3d(
            sub['cost'].values,
            sub['quality'].values,
            sub['task_completion_rate_pct'].values
        )

        ax2.scatter(
            sub['cost'], sub['quality'], sub['task_completion_rate_pct'],
            color=_color(sched), marker=_marker(sched), s=55, alpha=0.22,
            edgecolors='none'
        )

        if len(p_c) > 0:
            ax2.scatter(
                p_c, p_q, p_comp,
                color=_color(sched), marker=_marker(sched),
                s=135, alpha=0.95, edgecolors='white', linewidths=1.0,
                label=f'{_label(sched)} Pareto'
            )
            if len(p_c) >= 3:
                try:
                    ax2.plot_trisurf(
                        p_c, p_q, p_comp, color=_color(sched),
                        alpha=0.08, edgecolor=_color(sched), linewidth=0.25
                    )
                except Exception:
                    pass

    ax2.set_xlabel('Total Cost ↓', fontsize=12, labelpad=10)
    ax2.set_ylabel('Realized Quality ↑', fontsize=12, labelpad=10)
    ax2.set_zlabel('Completion Rate (%) ↑', fontsize=12, labelpad=10)
    ax2.set_title('Per-Method 3D Pareto Points\n(Distinct color and marker)', fontsize=14)
    ax2.legend(fontsize=9, loc='upper left', title='Method')
    ax2.view_init(elev=25, azim=135)

    plt.suptitle('3D Pareto Trade-off Analysis', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_3d_quality_cost_completion.png'), bbox_inches='tight')
    print("  [Saved] pareto_3d_quality_cost_completion.png")
    plt.close()


def plot_utility_vs_completion(df, output_dir):
    """Plot utility against completion rate, with method mean ± one std."""
    print("  Generating utility vs task completion plot...")

    required_cols = ['scheduler', 'utility', 'task_completion_rate_pct']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  Skipping utility vs completion plot: missing {missing}.")
        return

    valid_df = df.dropna(subset=required_cols).copy()
    if valid_df.empty:
        print("  Skipping utility vs completion plot: no valid rows.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    schedulers = valid_df['scheduler'].unique()
    method_handles = []

    for sched in schedulers:
        sub = valid_df[valid_df['scheduler'] == sched]
        color = _color(sched)
        marker = _marker(sched)

        # Individual Monte Carlo realizations.
        ax.scatter(
            sub['task_completion_rate_pct'], sub['utility'],
            color=color, marker=marker, s=85, alpha=0.58,
            edgecolors='white', linewidths=0.65, zorder=2
        )

        # Method mean and ± one standard deviation in both dimensions.
        mean_completion = sub['task_completion_rate_pct'].mean()
        mean_utility = sub['utility'].mean()
        std_completion = sub['task_completion_rate_pct'].std()
        std_utility = sub['utility'].std()
        std_completion = 0.0 if pd.isna(std_completion) else std_completion
        std_utility = 0.0 if pd.isna(std_utility) else std_utility

        ax.errorbar(
            mean_completion, mean_utility,
            xerr=std_completion, yerr=std_utility,
            fmt='D', markersize=13, color=color, ecolor=color,
            markerfacecolor=color, markeredgecolor='#222222',
            markeredgewidth=1.2, elinewidth=2.4, capsize=7,
            alpha=0.98, zorder=6
        )

        method_handles.append(
            plt.Line2D(
                [0], [0], marker=marker, linestyle='none', color=color,
                markerfacecolor=color, markeredgecolor='white',
                markersize=9, label=_label(sched)
            )
        )

    mean_handle = plt.Line2D(
        [0], [0], marker='D', linestyle='none', color='#555555',
        markerfacecolor='#AFAFAF', markeredgecolor='#222222',
        markersize=10, label='Method mean ± 1 std'
    )

    ax.set_xlabel('Task Completion Rate (%) ↑', fontsize=14)
    ax.set_ylabel('Utility (Quality - Cost) ↑', fontsize=14)
    ax.set_title(
        'Utility vs Task Completion Rate\n'
        '(diamonds = method mean ± 1 std; upper-right is better)',
        fontsize=17, pad=12
    )
    ax.legend(handles=method_handles + [mean_handle], fontsize=11, loc='best')
    ax.grid(alpha=0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'utility_vs_completion.png'), bbox_inches='tight')
    print("  [Saved] utility_vs_completion.png")
    plt.close()


def plot_metric_boxplots(df, output_dir):
    """Box plots showing distribution of key absolute and normalized per-completed-task metrics."""
    print("  Generating metric boxplots...")

    metrics = [
        ('utility_per_completed_task', 'Utility / Completed Task', True),
        ('quality_per_completed_task', 'Quality / Completed Task', True),
        ('cost_per_completed_task', 'Cost / Completed Task', False),
        ('task_completion_rate_pct', 'Task Completion Rate (%)', True),
        ('n_tasks_completed_valid', 'Completed Tasks Count', True),
        ('utility', 'Total Utility (Quality - Cost)', True),
        ('quality', 'Realized Quality', True),
        ('cost', 'Total Cost', False),
        ('avg_wall_time_s', 'Avg Wall Time (s)', False),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    schedulers = df['scheduler'].unique()

    for idx, (metric, title, higher_better) in enumerate(metrics):
        ax = axes[idx]
        if metric not in df.columns or df[metric].dropna().empty:
            ax.text(0.5, 0.5, f'Column\n"{metric}"\nnot found / empty', ha='center', va='center',
                    transform=ax.transAxes, color='grey', fontsize=16)
            ax.set_title(title, fontsize=16)
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
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
        ax.set_ylabel(title, fontsize=13)

        if higher_better is True:
            ax.set_title(title + ' ↑', fontsize=14)
        elif higher_better is False:
            ax.set_title(title + ' ↓', fontsize=14)
        else:
            ax.set_title(title, fontsize=14)

        ax.grid(axis='y', alpha=0.4)

    legend_handle = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                                markersize=8, label='Mean')
    fig.legend(handles=[legend_handle], loc='lower right', frameon=True, fontsize=14)

    plt.suptitle('Scheduler Performance — Per-Completed-Task Efficiency & Absolute Metrics', fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_boxplots.png'), bbox_inches='tight')
    print("  [Saved] metric_boxplots.png")
    plt.close()


def plot_mean_bars(df, output_dir):
    """Bar chart of means ± std comparing per-completed-task efficiency and primary metrics."""
    print("  Generating mean bar charts...")

    metrics = [
        ('utility_per_completed_task', 'Utility / Completed Task'),
        ('quality_per_completed_task', 'Quality / Completed Task'),
        ('cost_per_completed_task', 'Cost / Completed Task'),
        ('task_completion_rate_pct', 'Task Completion (%)'),
        ('utility', 'Total Utility'),
        ('n_tasks_completed_valid', 'Completed Tasks'),
    ]

    schedulers = df['scheduler'].unique()
    x = np.arange(len(schedulers))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns or df[metric].dropna().empty:
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
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(title, fontsize=15, pad=10)
        ax.grid(axis='y', alpha=0.4)

        max_val = max([m for m in means if not np.isnan(m)] or [1])
        for bar, mean, std in zip(bars, means, stds):
            if np.isnan(mean):
                continue
            y_offset = (std if not np.isnan(std) else 0) + 0.02 * max_val
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    f'{mean:.2f}' if 'per_completed_task' in metric else f'{mean:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Scheduler Comparison — Means ± Std (Per-Completed-Task Efficiency)', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_bars.png'), bbox_inches='tight')
    print("  [Saved] mean_bars.png")
    plt.close()


def plot_computational_metrics(df, output_dir):
    """Bar charts for computational performance: Avg Wall Time (s) and Avg MIP Gap (%)."""
    print("  Generating computational performance plots (Wall Time & MIP Gap)...")

    schedulers = df['scheduler'].unique()
    x = np.arange(len(schedulers))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if 'avg_wall_time_s' in df.columns and not df['avg_wall_time_s'].dropna().empty:
        means_wt = [df[df['scheduler'] == s]['avg_wall_time_s'].mean() for s in schedulers]
        stds_wt = [df[df['scheduler'] == s]['avg_wall_time_s'].std() for s in schedulers]
        colors = [_color(s) for s in schedulers]
        labels = [_label(s) for s in schedulers]

        bars1 = ax1.bar(x, means_wt, yerr=stds_wt, capsize=6, color=colors, alpha=0.8,
                        error_kw=dict(elinewidth=2, ecolor='#333333'))
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)
        ax1.set_ylabel('Wall Clock Time (seconds)', fontsize=14)
        ax1.set_title('Avg Wall Time (s) ↓', fontsize=16, pad=10)
        ax1.grid(axis='y', alpha=0.4)

        max_val1 = max([m for m in means_wt if not np.isnan(m)] or [1])
        for bar, mean, std in zip(bars1, means_wt, stds_wt):
            if np.isnan(mean):
                continue
            y_off = (std if not np.isnan(std) else 0) + 0.02 * max_val1
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                     f'{mean:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Missing: avg_wall_time_s', ha='center', va='center', transform=ax1.transAxes)

    if 'avg_mip_gap_pct' in df.columns and not df['avg_mip_gap_pct'].dropna().empty:
        means_gap = [df[df['scheduler'] == s]['avg_mip_gap_pct'].mean() for s in schedulers]
        stds_gap = [df[df['scheduler'] == s]['avg_mip_gap_pct'].std() for s in schedulers]
        colors = [_color(s) for s in schedulers]
        labels = [_label(s) for s in schedulers]

        bars2 = ax2.bar(x, [0 if np.isnan(m) else m for m in means_gap],
                        yerr=[0 if np.isnan(s) else s for s in stds_gap],
                        capsize=6, color=colors, alpha=0.8,
                        error_kw=dict(elinewidth=2, ecolor='#333333'))
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)
        ax2.set_ylabel('MIP Gap (%)', fontsize=14)
        ax2.set_title('Avg MIP Gap (%) ↓', fontsize=16, pad=10)
        ax2.grid(axis='y', alpha=0.4)

        valid_means = [m for m in means_gap if not np.isnan(m)]
        max_val2 = max(valid_means) if valid_means and max(valid_means) > 0 else 1
        for bar, mean, std in zip(bars2, means_gap, stds_gap):
            if np.isnan(mean):
                ax2.text(bar.get_x() + bar.get_width() / 2, 0.05 * max_val2,
                         'N/A', ha='center', va='bottom', fontsize=11, color='gray')
            else:
                y_off = (std if not np.isnan(std) else 0) + 0.02 * max_val2
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                         f'{mean:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Missing: avg_mip_gap_pct', ha='center', va='center', transform=ax2.transAxes)

    plt.suptitle('Computational Efficiency Metrics', fontsize=18, y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computational_metrics.png'), bbox_inches='tight')
    print("  [Saved] computational_metrics.png")
    plt.close()


def plot_per_run_lines(df, output_dir):
    """Line plots showing primary and normalized per-completed-task metrics per Monte Carlo run."""
    print("  Generating per-run line plots...")

    metrics = [
        ('utility_per_completed_task', 'Utility / Completed Task'),
        ('task_completion_rate_pct', 'Task Completion Rate (%)'),
        ('cost_per_completed_task', 'Cost / Completed Task'),
        ('avg_wall_time_s', 'Avg Wall Time (s)'),
    ]

    schedulers = df['scheduler'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns or df[metric].dropna().empty:
            ax.text(0.5, 0.5, f'Missing / Empty:\n{metric}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=15)
            continue

        for sched in schedulers:
            sub = df[df['scheduler'] == sched].sort_values('run')
            if sub[metric].dropna().empty:
                continue
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


def plot_scheduler_advantage(df, output_dir):
    """Bar chart showing advantage of each non-baseline scheduler over greedy baseline."""
    print("  Generating scheduler advantage chart...")

    if 'greedy' not in df['scheduler'].values:
        print("  Skipping: no 'greedy' scheduler found for baseline comparison")
        return

    metrics = [
        ('utility_per_completed_task', 'Utility / Completed Task'),
        ('task_completion_rate_pct', 'Task Completion (%)'),
        ('quality_per_completed_task', 'Quality / Completed Task'),
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
                    f'{sign}{val:.2f}' if 'per_completed_task' in metric else f'{sign}{val:.1f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle('Advantage over Greedy Baseline (mean ± std of per-run differences)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scheduler_advantage.png'), bbox_inches='tight')
    print("  [Saved] scheduler_advantage.png")
    plt.close()


def generate_summary_report(df, output_dir, folder_list):
    """Text summary report focusing on primary absolute and per-completed-task efficiency metrics."""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    n_runs = df['run'].nunique()
    schedulers = df['scheduler'].unique()

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMBINED JSON RUNS — PRIMARY & NORMALIZED METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Source folders ({len(folder_list)}):\n")
        for folder in folder_list:
            f.write(f"  - {folder}\n")
        f.write(f"\nTotal aggregated Monte Carlo runs: {n_runs}\n")
        f.write(f"Schedulers: {', '.join(schedulers)}\n\n")

        metrics = [
            ('utility_per_completed_task', 'Utility / Completed Task'),
            ('quality_per_completed_task', 'Quality / Completed Task'),
            ('cost_per_completed_task', 'Cost / Completed Task'),
            ('task_completion_rate_pct', 'Task Completion Rate (%)'),
            ('n_tasks_completed_valid', 'Completed Tasks Count'),
            ('utility', 'Total Utility (Quality - Cost)'),
            ('quality', 'Realized Quality'),
            ('cost', 'Total Cost'),
            ('avg_wall_time_s', 'Avg Wall Time (s)'),
            ('avg_mip_gap_pct', 'Avg MIP Gap (%)'),
        ]

        f.write("PER-SCHEDULER STATISTICS\n")
        f.write("-" * 70 + "\n")
        for sched in schedulers:
            sub = df[df['scheduler'] == sched]
            f.write(f"\n{_label(sched).upper()} ({sched}):\n")
            for col, label in metrics:
                if col in sub.columns and not sub[col].dropna().empty:
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
                for col, label in [('utility_per_completed_task', 'Utility / Completed Task'),
                                    ('quality_per_completed_task', 'Quality / Completed Task'),
                                    ('cost_per_completed_task', 'Cost / Completed Task'),
                                    ('task_completion_rate_pct', 'Task Completion (%)'),
                                    ('n_tasks_completed_valid', 'Completed Tasks'),
                                    ('utility', 'Total Utility')]:
                    if col in sub.columns and not sub[col].dropna().empty:
                        delta = sub[col].mean() - greedy[col].mean()
                        sign = '+' if delta >= 0 else ''
                        pct = (delta / greedy[col].mean() * 100) if greedy[col].mean() != 0 else 0
                        f.write(f"  {label:<30} {sign}{delta:>8.3f}  ({sign}{pct:.1f}%)\n")

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
        register_scheduler_styles(schedulers)
        print(f"\n  Successfully loaded {len(all_runs_df)} run results")
        print(f"  Aggregated dataset: {n_runs} total runs across {len(schedulers)} schedulers")
        print(f"  Schedulers: {schedulers}")
    except Exception as e:
        print(f"ERROR loading results: {e}")
        traceback.print_exc()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating combined plots in: {OUTPUT_DIR}\n")

    plot_functions = [
        ('2D Pareto Quality vs Cost', lambda df, od: plot_pareto_2d(df, od)),
        ('3D Pareto Quality vs Cost vs Completion', lambda df, od: plot_pareto_3d(df, od)),
        ('Utility vs Task Completion', lambda df, od: plot_utility_vs_completion(df, od)),
        ('Metric Boxplots', lambda df, od: plot_metric_boxplots(df, od)),
        ('Computational Metrics', lambda df, od: plot_computational_metrics(df, od)),
        ('Mean Bar Charts', lambda df, od: plot_mean_bars(df, od)),
        ('Per-Run Lines', lambda df, od: plot_per_run_lines(df, od)),
        ('Scheduler Advantage', lambda df, od: plot_scheduler_advantage(df, od)),
        ('Summary Report', lambda df, od: generate_summary_report(df, od, RUN_FOLDERS)),
    ]

    failed = []
    for name, fn in plot_functions:
        try:
            fn(all_runs_df, OUTPUT_DIR)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            traceback.print_exc()
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
