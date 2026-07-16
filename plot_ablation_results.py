"""
Post-processing script for cost ablation study results.
Generates comprehensive summary plots from ablation study results folder.

Usage:
    python plot_ablation_results.py <results_folder>

Example:
    python plot_ablation_results.py results/cost_ablation_2026-07-08_100327
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_results_from_json(results_dir):
    """Load results from individual JSON files."""
    import json
    import glob

    # Find all JSON result files
    json_pattern = os.path.join(results_dir, "run_sub*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        raise FileNotFoundError(f"No JSON result files found in {results_dir}")

    print(f"  Found {len(json_files)} JSON result files")

    # Load all JSON files
    all_data = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_data.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not load {json_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid JSON files could be loaded")

    print(f"  Successfully loaded {len(all_data)} JSON files")

    # Build long-format dataframe (one row per scheduler+params)
    rows = []
    for data in all_data:
        # Check if this is individual run format or combined format
        if 'scheduler' in data:
            # Individual run format (per-scheduler JSON)
            rows.append({
                'submission_cost': data.get('submission_cost', 0),
                'execution_cost': data.get('execution_cost', 0),
                'scheduler': data['scheduler'],
                'scheduled': data.get('scheduled', 0),
                'attempts': data.get('attempts', 0),
                'accepted': data.get('accepted', 0),
                'executed': data.get('executed', 0),
                'rejections': data.get('rejections', 0),
                'rejection_rate': data.get('rejection_rate', 0),
                'acceptance_rate': data.get('acceptance_rate', 0),
                'execution_rate': data.get('execution_rate', 0),
                'quality': data.get('quality', 0),
                'cost': data.get('cost', 0),
                'submission_cost_value': data.get('submission_cost_value', 0),
                'execution_cost_value': data.get('execution_cost_value', 0),
                'utility': data.get('utility', 0),
                'ticks': data.get('ticks', 0),
                'elapsed_s': data.get('elapsed_s', 0),
            })
        else:
            # Combined format (both schedulers in one JSON) - legacy format
            # Extract submission and execution cost from filename or data
            sub_cost = data.get('submission_cost_param', 0)
            exec_cost = data.get('execution_cost_param', 0)

            # Stochastic row
            rows.append({
                'submission_cost': sub_cost,
                'execution_cost': exec_cost,
                'scheduler': 'stochastic',
                'scheduled': data.get('sto_scheduled', 0),
                'attempts': data.get('sto_attempts', 0),
                'accepted': data.get('sto_accepted', 0),
                'executed': data.get('sto_executed', 0),
                'rejections': data.get('sto_rejections', 0),
                'rejection_rate': data.get('sto_rejection_rate', 0),
                'acceptance_rate': data.get('sto_acceptance_rate', 0),
                'execution_rate': data.get('sto_execution_rate', 0),
                'quality': data.get('sto_quality', 0),
                'cost': data.get('sto_cost', 0),
                'submission_cost_value': data.get('sto_submission_cost', 0),
                'execution_cost_value': data.get('sto_execution_cost', 0),
                'utility': data.get('sto_utility', 0),
                'ticks': data.get('sto_ticks', 0),
                'elapsed_s': data.get('sto_elapsed_s', 0),
            })

            # Deterministic row
            rows.append({
                'submission_cost': sub_cost,
                'execution_cost': exec_cost,
                'scheduler': 'deterministic',
                'scheduled': data.get('det_scheduled', 0),
                'attempts': data.get('det_attempts', 0),
                'accepted': data.get('det_accepted', 0),
                'executed': data.get('det_executed', 0),
                'rejections': data.get('det_rejections', 0),
                'rejection_rate': data.get('det_rejection_rate', 0),
                'acceptance_rate': data.get('det_acceptance_rate', 0),
                'execution_rate': data.get('det_execution_rate', 0),
                'quality': data.get('det_quality', 0),
                'cost': data.get('det_cost', 0),
                'submission_cost_value': data.get('det_submission_cost', 0),
                'execution_cost_value': data.get('det_execution_cost', 0),
                'utility': data.get('det_utility', 0),
                'ticks': data.get('det_ticks', 0),
                'elapsed_s': data.get('det_elapsed_s', 0),
            })

    all_runs_df = pd.DataFrame(rows)

    # Remove duplicate entries (in case both individual and combined formats exist)
    all_runs_df = all_runs_df.drop_duplicates(
        subset=['submission_cost', 'execution_cost', 'scheduler']
    )

    return all_runs_df, None


def load_results(results_dir):
    """Load results from CSV files or JSON files as fallback."""
    all_runs_path = os.path.join(results_dir, "all_runs.csv")
    summary_path = os.path.join(results_dir, "ablation_summary.csv")

    # Try to load from CSV first
    if os.path.exists(all_runs_path):
        print("  Loading from CSV files...")
        all_runs_df = pd.read_csv(all_runs_path)

        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = None

        return all_runs_df, summary_df

    # Fallback to loading from JSON files
    print("  CSV not found, loading from individual JSON files...")
    return load_results_from_json(results_dir)


def plot_heatmap(ax, data, title, cmap='RdYlGn', vmin=None, vmax=None, center=None, cbar_label=''):
    """Helper function to plot a heatmap using matplotlib."""
    if center is not None:
        # For diverging colormaps, center around the specified value
        vmax_abs = max(abs(data.min().min() - center), abs(data.max().max() - center))
        vmin = center - vmax_abs
        vmax = center + vmax_abs

    im = ax.imshow(data.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels([f'{x:.2f}' for x in data.columns])
    ax.set_yticklabels([f'{y:.2f}' for y in data.index])

    # Add value annotations
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            ax.text(j, i, f'{data.values[i, j]:.1f}',
                   ha='center', va='center', color='black', fontsize=9)

    ax.set_title(title)


def plot_utility_heatmaps(df, output_dir):
    """Create heatmaps showing utility across cost parameter space."""
    print("  Generating utility heatmaps...")

    # Check if required columns exist
    required_cols = ['scheduler', 'submission_cost', 'execution_cost', 'utility']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pivot data for heatmaps
    try:
        stochastic_data = df[df['scheduler'] == 'stochastic'].pivot(
            index='execution_cost', columns='submission_cost', values='utility'
        )

        deterministic_data = df[df['scheduler'] == 'deterministic'].pivot(
            index='execution_cost', columns='submission_cost', values='utility'
        )
    except Exception as e:
        print(f"    Error pivoting data: {e}")
        raise

    # Calculate difference (stochastic advantage)
    difference_data = stochastic_data - deterministic_data

    # Plot stochastic utility
    plot_heatmap(axes[0], stochastic_data, 'Stochastic Utility',
                cmap='RdYlGn', cbar_label='Utility')
    axes[0].set_xlabel('Submission Cost')
    axes[0].set_ylabel('Execution Cost')

    # Plot deterministic utility
    plot_heatmap(axes[1], deterministic_data, 'Deterministic Utility',
                cmap='RdYlGn', cbar_label='Utility')
    axes[1].set_xlabel('Submission Cost')
    axes[1].set_ylabel('Execution Cost')

    # Plot difference (stochastic advantage)
    plot_heatmap(axes[2], difference_data, 'Stochastic Advantage\n(Stochastic - Deterministic)',
                cmap='RdBu_r', center=0, cbar_label='Advantage')
    axes[2].set_xlabel('Submission Cost')
    axes[2].set_ylabel('Execution Cost')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'utility_heatmaps.png'), bbox_inches='tight')
    print(f"  [Saved] utility_heatmaps.png")
    plt.close()


def plot_rejection_rates(df, output_dir):
    """Plot rejection rates comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmaps for rejection rates
    stochastic_rej = df[df['scheduler'] == 'stochastic'].pivot(
        index='execution_cost', columns='submission_cost', values='rejection_rate'
    )

    deterministic_rej = df[df['scheduler'] == 'deterministic'].pivot(
        index='execution_cost', columns='submission_cost', values='rejection_rate'
    )

    # Plot stochastic rejection rate
    plot_heatmap(axes[0], stochastic_rej, 'Stochastic Rejection Rate',
                cmap='YlOrRd', cbar_label='Rejection Rate (%)')
    axes[0].set_xlabel('Submission Cost')
    axes[0].set_ylabel('Execution Cost')

    # Plot deterministic rejection rate
    plot_heatmap(axes[1], deterministic_rej, 'Deterministic Rejection Rate',
                cmap='YlOrRd', cbar_label='Rejection Rate (%)')
    axes[1].set_xlabel('Submission Cost')
    axes[1].set_ylabel('Execution Cost')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rejection_rates.png'), bbox_inches='tight')
    print(f"  [Saved] rejection_rates.png")
    plt.close()


def plot_execution_rates(df, output_dir):
    """Plot execution success rates comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmaps for execution rates
    stochastic_exec = df[df['scheduler'] == 'stochastic'].pivot(
        index='execution_cost', columns='submission_cost', values='execution_rate'
    )

    deterministic_exec = df[df['scheduler'] == 'deterministic'].pivot(
        index='execution_cost', columns='submission_cost', values='execution_rate'
    )

    # Plot stochastic execution rate
    plot_heatmap(axes[0], stochastic_exec, 'Stochastic Execution Rate',
                cmap='YlGn', cbar_label='Execution Rate (%)')
    axes[0].set_xlabel('Submission Cost')
    axes[0].set_ylabel('Execution Cost')

    # Plot deterministic execution rate
    plot_heatmap(axes[1], deterministic_exec, 'Deterministic Execution Rate',
                cmap='YlGn', cbar_label='Execution Rate (%)')
    axes[1].set_xlabel('Submission Cost')
    axes[1].set_ylabel('Execution Cost')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_rates.png'), bbox_inches='tight')
    print(f"  [Saved] execution_rates.png")
    plt.close()


def plot_comparison_bars(df, output_dir):
    """Bar charts comparing key metrics."""
    print("  Generating comparison bars...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('utility', 'Utility'),
        ('rejection_rate', 'Rejection Rate (%)'),
        ('execution_rate', 'Execution Rate (%)'),
        ('quality', 'Total Quality'),
        ('cost', 'Total Cost'),
        ('scheduled', 'Scheduled Tasks')
    ]

    for idx, (metric, title) in enumerate(metrics):
        try:
            ax = axes[idx // 3, idx % 3]

            # Group by cost parameters and scheduler
            grouped = df.groupby(['submission_cost', 'execution_cost', 'scheduler'])[metric].mean().reset_index()

            # Create parameter combination labels
            grouped['params'] = grouped.apply(
                lambda row: f"sub={row['submission_cost']:.2f}\nexec={row['execution_cost']:.2f}",
                axis=1
            )

            # Get all unique parameter combinations (union of both schedulers)
            all_params_sto = set(grouped[grouped['scheduler'] == 'stochastic']['params'].values)
            all_params_det = set(grouped[grouped['scheduler'] == 'deterministic']['params'].values)
            all_params = sorted(all_params_sto | all_params_det)

            x = np.arange(len(all_params))
            width = 0.35

            # Get values for each scheduler, using reindex to handle missing values gracefully
            sto_df = grouped[grouped['scheduler'] == 'stochastic'].set_index('params')
            det_df = grouped[grouped['scheduler'] == 'deterministic'].set_index('params')

            sto_data = sto_df.reindex(all_params)[metric].fillna(0).values
            det_data = det_df.reindex(all_params)[metric].fillna(0).values

            # Plot bars
            ax.bar(x - width/2, sto_data, width, label='Stochastic', alpha=0.8)
            ax.bar(x + width/2, det_data, width, label='Deterministic', alpha=0.8)

            ax.set_xlabel('Cost Parameters')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(all_params, fontsize=8)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        except Exception as e:
            print(f"    Warning: Could not generate {title} bar chart: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bars.png'), bbox_inches='tight')
    print(f"  [Saved] comparison_bars.png")
    plt.close()


def plot_scatter_utility_vs_rejection(df, output_dir):
    """Scatter plot of utility vs rejection rate."""
    print("  Generating scatter plot...")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        stochastic = df[df['scheduler'] == 'stochastic']
        deterministic = df[df['scheduler'] == 'deterministic']

        ax.scatter(stochastic['rejection_rate'], stochastic['utility'],
                   s=100, alpha=0.6, label='Stochastic', marker='o')
        ax.scatter(deterministic['rejection_rate'], deterministic['utility'],
                   s=100, alpha=0.6, label='Deterministic', marker='s')

        # Add parameter labels (safely)
        for _, row in stochastic.iterrows():
            try:
                ax.annotate(f"({row['submission_cost']:.2f},{row['execution_cost']:.2f})",
                           (row['rejection_rate'], row['utility']),
                           fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
            except:
                pass

        ax.set_xlabel('Rejection Rate (%)')
        ax.set_ylabel('Utility')
        ax.set_title('Utility vs Rejection Rate\n(Higher utility and lower rejection is better)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'utility_vs_rejection.png'), bbox_inches='tight')
        print(f"  [Saved] utility_vs_rejection.png")
        plt.close()
    except Exception as e:
        print(f"    Scatter plot error: {e}")
        raise


def plot_quality_per_task(df, output_dir):
    """Plot quality per scheduled task."""
    print("  Generating quality per task...")

    try:
        df_copy = df.copy()  # Avoid modifying original
        df_copy['quality_per_task'] = df_copy['quality'] / df_copy['scheduled'].replace(0, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmaps for quality per task
        try:
            stochastic_qpt = df_copy[df_copy['scheduler'] == 'stochastic'].pivot(
                index='execution_cost', columns='submission_cost', values='quality_per_task'
            )
            plot_heatmap(axes[0], stochastic_qpt, 'Stochastic Quality per Task',
                        cmap='Blues', cbar_label='Quality/Task')
            axes[0].set_xlabel('Submission Cost')
            axes[0].set_ylabel('Execution Cost')
        except Exception as e:
            print(f"    Warning: Could not generate stochastic quality heatmap: {e}")
            axes[0].text(0.5, 0.5, 'Error generating heatmap', ha='center', va='center', transform=axes[0].transAxes)

        try:
            deterministic_qpt = df_copy[df_copy['scheduler'] == 'deterministic'].pivot(
                index='execution_cost', columns='submission_cost', values='quality_per_task'
            )
            plot_heatmap(axes[1], deterministic_qpt, 'Deterministic Quality per Task',
                        cmap='Blues', cbar_label='Quality/Task')
            axes[1].set_xlabel('Submission Cost')
            axes[1].set_ylabel('Execution Cost')
        except Exception as e:
            print(f"    Warning: Could not generate deterministic quality heatmap: {e}")
            axes[1].text(0.5, 0.5, 'Error generating heatmap', ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_per_task.png'), bbox_inches='tight')
        print(f"  [Saved] quality_per_task.png")
        plt.close()
    except Exception as e:
        print(f"    Quality per task error: {e}")
        raise


def plot_performance_metrics(df, output_dir):
    """Plot runtime performance metrics."""
    print("  Generating performance metrics...")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Group by scheduler and compute averages
        grouped = df.groupby('scheduler').agg({
            'elapsed_s': 'mean',
            'ticks': 'mean'
        }).reset_index()

        if len(grouped) == 0:
            print("    Warning: No scheduler data for performance metrics")
            axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)
        else:
            # Plot elapsed time
            ax = axes[0]
            x = np.arange(len(grouped))
            ax.bar(x, grouped['elapsed_s'], alpha=0.8)
            ax.set_ylabel('Elapsed Time (seconds)')
            ax.set_title('Average Runtime per Scheduler')
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['scheduler'])
            ax.grid(axis='y', alpha=0.3)

            # Plot ticks
            ax = axes[1]
            ax.bar(x, grouped['ticks'], alpha=0.8, color='orange')
            ax.set_ylabel('Simulation Ticks')
            ax.set_title('Average Simulation Ticks per Scheduler')
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['scheduler'])
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), bbox_inches='tight')
        print(f"  [Saved] performance_metrics.png")
        plt.close()
    except Exception as e:
        print(f"    Performance metrics error: {e}")
        raise


def generate_summary_report(df, output_dir):
    """Generate a text summary report."""
    report_path = os.path.join(output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COST ABLATION STUDY - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")

        for scheduler in ['stochastic', 'deterministic']:
            sched_data = df[df['scheduler'] == scheduler]
            f.write(f"\n{scheduler.upper()}:\n")
            f.write(f"  Avg Utility:         {sched_data['utility'].mean():.2f} ± {sched_data['utility'].std():.2f}\n")
            f.write(f"  Avg Rejection Rate:  {sched_data['rejection_rate'].mean():.2f}%\n")
            f.write(f"  Avg Execution Rate:  {sched_data['execution_rate'].mean():.2f}%\n")
            f.write(f"  Avg Quality:         {sched_data['quality'].mean():.2f}\n")
            f.write(f"  Avg Cost:            {sched_data['cost'].mean():.2f}\n")
            f.write(f"  Avg Scheduled:       {sched_data['scheduled'].mean():.2f}\n")
            f.write(f"  Avg Runtime:         {sched_data['elapsed_s'].mean():.2f}s\n")

        # Best configurations
        f.write("\n" + "="*70 + "\n")
        f.write("BEST CONFIGURATIONS\n")
        f.write("-"*70 + "\n")

        # Best stochastic utility
        best_sto = df[df['scheduler'] == 'stochastic'].loc[df[df['scheduler'] == 'stochastic']['utility'].idxmax()]
        f.write(f"\nBest Stochastic Utility:\n")
        f.write(f"  Submission Cost: {best_sto['submission_cost']:.2f}\n")
        f.write(f"  Execution Cost:  {best_sto['execution_cost']:.2f}\n")
        f.write(f"  Utility:         {best_sto['utility']:.2f}\n")
        f.write(f"  Rejection Rate:  {best_sto['rejection_rate']:.2f}%\n")
        f.write(f"  Execution Rate:  {best_sto['execution_rate']:.2f}%\n")

        # Lowest stochastic rejection
        lowest_rej = df[df['scheduler'] == 'stochastic'].loc[df[df['scheduler'] == 'stochastic']['rejection_rate'].idxmin()]
        f.write(f"\nLowest Stochastic Rejection Rate:\n")
        f.write(f"  Submission Cost: {lowest_rej['submission_cost']:.2f}\n")
        f.write(f"  Execution Cost:  {lowest_rej['execution_cost']:.2f}\n")
        f.write(f"  Rejection Rate:  {lowest_rej['rejection_rate']:.2f}%\n")
        f.write(f"  Utility:         {lowest_rej['utility']:.2f}\n")

        # Highest execution rate
        highest_exec = df[df['scheduler'] == 'stochastic'].loc[df[df['scheduler'] == 'stochastic']['execution_rate'].idxmax()]
        f.write(f"\nHighest Stochastic Execution Rate:\n")
        f.write(f"  Submission Cost: {highest_exec['submission_cost']:.2f}\n")
        f.write(f"  Execution Cost:  {highest_exec['execution_cost']:.2f}\n")
        f.write(f"  Execution Rate:  {highest_exec['execution_rate']:.2f}%\n")
        f.write(f"  Utility:         {highest_exec['utility']:.2f}\n")

        # Comparative analysis
        f.write("\n" + "="*70 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-"*70 + "\n\n")

        # Compare by cost parameters
        for (sub, ex), group in df.groupby(['submission_cost', 'execution_cost']):
            sto = group[group['scheduler'] == 'stochastic'].iloc[0]
            det = group[group['scheduler'] == 'deterministic'].iloc[0]

            f.write(f"\nSubmission={sub:.2f}, Execution={ex:.2f}:\n")
            f.write(f"  Utility:         Sto={sto['utility']:7.2f}  Det={det['utility']:7.2f}  Diff={sto['utility']-det['utility']:+7.2f}\n")
            f.write(f"  Rejection Rate:  Sto={sto['rejection_rate']:6.2f}%  Det={det['rejection_rate']:6.2f}%  Diff={sto['rejection_rate']-det['rejection_rate']:+6.2f}%\n")
            f.write(f"  Execution Rate:  Sto={sto['execution_rate']:6.2f}%  Det={det['execution_rate']:6.2f}%  Diff={sto['execution_rate']-det['execution_rate']:+6.2f}%\n")

    print(f"  [Saved] summary_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from cost ablation study results')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as results_dir)')

    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir if args.output_dir else results_dir

    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    print(f"\n{'='*70}")
    print("COST ABLATION STUDY - POST-PROCESSING")
    print(f"{'='*70}\n")
    print(f"Loading results from: {results_dir}")

    # Load data
    try:
        all_runs_df, summary_df = load_results(results_dir)
        print(f"  Loaded {len(all_runs_df)} data points")
        print(f"  Schedulers: {all_runs_df['scheduler'].unique()}")
        print(f"  Cost parameters: {len(all_runs_df) // 2} combinations")
    except Exception as e:
        print(f"ERROR loading results: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating plots in: {output_dir}\n")

    # Check data validity
    print(f"Data shape: {all_runs_df.shape}")
    print(f"Columns: {list(all_runs_df.columns)}")
    print(f"Schedulers: {all_runs_df['scheduler'].unique()}")

    required_cols = ['submission_cost', 'execution_cost', 'scheduler', 'utility', 'rejection_rate', 'execution_rate']
    missing_cols = [col for col in required_cols if col not in all_runs_df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return
    print(f"✓ All required columns present")

    # Generate all plots
    print("\nGenerating plots...")
    plot_functions = [
        ('Utility Heatmaps', plot_utility_heatmaps),
        ('Rejection Rates', plot_rejection_rates),
        ('Execution Rates', plot_execution_rates),
        ('Comparison Bars', plot_comparison_bars),
        ('Utility vs Rejection', plot_scatter_utility_vs_rejection),
        ('Quality per Task', plot_quality_per_task),
        ('Performance Metrics', plot_performance_metrics),
        ('Summary Report', generate_summary_report),
    ]

    failed_plots = []
    for plot_name, plot_func in plot_functions:
        try:
            plot_func(all_runs_df, output_dir)
        except Exception as e:
            print(f"  ✗ ERROR in {plot_name}: {e}")
            failed_plots.append((plot_name, str(e)))
            continue

    print(f"\n{'='*70}")
    if failed_plots:
        print(f"POST-PROCESSING COMPLETE (with {len(failed_plots)} warnings)")
    else:
        print("✓✓✓ POST-PROCESSING COMPLETE ✓✓✓")
    print(f"{'='*70}")
    print(f"\n✓ Plots saved to: {output_dir}")
    print(f"\nGenerated files:")
    plot_files = [
        'utility_heatmaps.png',
        'rejection_rates.png',
        'execution_rates.png',
        'comparison_bars.png',
        'utility_vs_rejection.png',
        'quality_per_task.png',
        'performance_metrics.png',
        'summary_report.txt'
    ]
    for plot_file in plot_files:
        full_path = os.path.join(output_dir, plot_file)
        if os.path.exists(full_path):
            print(f"  ✓ {plot_file}")
        else:
            print(f"  ✗ {plot_file} (NOT FOUND)")

    if failed_plots:
        print(f"\n⚠ Failed to generate ({len(failed_plots)}):")
        for plot_name, error in failed_plots:
            print(f"  • {plot_name}: {error[:60]}...")
    print()


if __name__ == "__main__":
    main()
