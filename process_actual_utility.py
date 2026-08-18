"""Reconstruct causally valid run metrics and generate the standard plots.

This script pairs every ``run_*.json`` summary with its corresponding
``executions_*.json`` file.  It uses the execution records for metrics that can
be corrupted by later case-specific summary rewrites:

* realized quality: best successful quality per causally valid task;
* completed tasks: number of distinct causally valid tasks;
* completion rate: valid completed tasks divided by the run's corrected
  reachability denominator.

Costs remain based on the run summary's submission and execution cost
components.  The execution-detail file is intentionally not used as the
primary cost source because it omits failed executions and causally invalid
executions, even though those bookings still incur cost.

The plotting functions and filenames are imported from ``plot_runs_results.py``
so both scripts produce the same figures and folder layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION â€” same pattern as plot_runs_results.py
# ==============================================================================
BASE_DIR = "/Users/davidf/Code/federated-scheduling/results"

RUN_FOLDERS = [
    "volcano_campaign_20260817_015417",
    "volcano_2026-08-17_194449"
]
# RUN_FOLDERS = [
#     "earthquake_campaign_20260817_002759",
# ]
HEDGING_SCHEDULER_ALIASES = {"stochastic_log", "stochastic_logical"}
HEDGING_SCHEDULER_ID = "hedging_milp"

# Fixed left-to-right order used by every plot and report.
SCHEDULER_ORDER = [
    HEDGING_SCHEDULER_ID,
    "deterministic",
    "greedy_n",
    "greedy",
    "random",
]

SCHEDULER_LABELS = {
    HEDGING_SCHEDULER_ID: "Hedging MILP",
    "deterministic": "Standard ILP",
    "greedy_n": "Greedy-N",
    "greedy": "Greedy",
    "random": "Random",
}

# Set to None to plot every available scheduler, or list a subset using the
# identifiers above, for example ["hedging_milp", "deterministic", "greedy_n"].
# The subset always retains SCHEDULER_ORDER rather than the list's input order.
PLOT_SCHEDULERS = None

# The corrected figures replace the previous figures in the usual directory.
OUTPUT_DIR = os.path.join(BASE_DIR, RUN_FOLDERS[0], "combined_results")
# ==============================================================================


def _load_plot_module():
    """Import the supplied plotting script from this script's directory."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        import plot_runs_results as plots
    except ImportError as exc:
        raise ImportError(
            "plot_runs_results.py must be in the same directory as "
            "process_actual_utility.py"
        ) from exc

    # Both legacy identifiers and the canonical identifier refer to the same
    # uncertainty-aware hedging method.
    if "stochastic_log" in plots.SCHEDULER_COLORS:
        for scheduler_id in (*HEDGING_SCHEDULER_ALIASES, HEDGING_SCHEDULER_ID):
            plots.SCHEDULER_COLORS.setdefault(
                scheduler_id, plots.SCHEDULER_COLORS["stochastic_log"]
            )
            plots.SCHEDULER_MARKERS.setdefault(
                scheduler_id, plots.SCHEDULER_MARKERS["stochastic_log"]
            )
            plots.SCHEDULER_LINESTYLES.setdefault(
                scheduler_id, plots.SCHEDULER_LINESTYLES["stochastic_log"]
            )
            plots.SCHEDULER_LABELS[scheduler_id] = "Hedging MILP"
    plots.SCHEDULER_LABELS.update(SCHEDULER_LABELS)
    return plots


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _as_nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _find_execution_file(run_file: Path) -> Path | None:
    """Return the execution-detail file paired with a run-summary file."""
    if not run_file.name.startswith("run_"):
        return None

    suffix = run_file.name[len("run_") :]
    exact = run_file.with_name(f"executions_{suffix}")
    if exact.exists():
        return exact

    # Tolerate copied files such as ``...(1).json``.
    stem = Path(suffix).stem
    candidates = sorted(run_file.parent.glob(f"executions_{stem}*.json"))
    return candidates[0] if len(candidates) == 1 else None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _valid_best_entries(executions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select the highest-quality causally valid execution for every task."""
    best: dict[str, dict[str, Any]] = {}
    for entry in executions:
        if not isinstance(entry, dict) or not entry.get("task"):
            continue
        if entry.get("is_valid", True) is not True:
            continue

        task = str(entry["task"])
        quality = _as_float(entry.get("quality"), 0.0)
        if task not in best or quality > _as_float(best[task].get("quality"), 0.0):
            best[task] = entry
    return best


def _completion_denominator(run: dict[str, Any], completed: int) -> int:
    """Choose the task denominator without removing parent-failure cascades.

    New earthquake runs store the case-specific denominator in
    ``n_tasks_reachable``.  It excludes exogenous impossibilities and tasks
    suppressed by a resolved logical gate, while parent failures remain misses.

    The fallback removes only geometry/timeline/truncation exclusions from the
    total.  In particular, ``n_tasks_parent_not_met`` is not subtracted.
    """
    stored = _as_nonnegative_int(run.get("n_tasks_reachable"))
    if stored is not None and stored >= completed:
        return stored

    total = _as_nonnegative_int(run.get("n_tasks_total"))
    if total is None:
        return completed

    exogenous = sum(
        _as_nonnegative_int(run.get(field)) or 0
        for field in (
            "n_tasks_no_passes",
            "n_tasks_inactive_timeline",
            "n_tasks_sim_truncated",
        )
    )
    return max(completed, total - exogenous)


def reconstruct_run(
    run: dict[str, Any],
    executions: list[dict[str, Any]],
    run_file: Path,
    execution_file: Path,
) -> dict[str, Any]:
    """Return one corrected run record without modifying either source file."""
    corrected = dict(run)
    best = _valid_best_entries(executions)
    selected = list(best.values())

    realized_quality = float(sum(_as_float(row.get("quality")) for row in selected))
    n_completed = len(selected)
    denominator = _completion_denominator(run, n_completed)
    completion_rate = n_completed / denominator if denominator else 0.0

    # The run summary contains costs for all billable outcomes, including failed
    # and causally invalid executions that do not appear in the exported detail
    # file.  Recombine its two components instead of charging only the pass that
    # supplied each task's credited quality.
    submission_cost = _as_float(run.get("submission_cost"), 0.0)
    if "execution_cost" in run:
        execution_cost = _as_float(run.get("execution_cost"), 0.0)
        cost_source = "run submission_cost + execution_cost"
    elif "total_cost" in run:
        execution_cost = max(0.0, _as_float(run.get("total_cost")) - submission_cost)
        cost_source = "run total_cost"
    else:
        execution_cost = float(
            sum(_as_float(row.get("exec_cost")) for row in executions)
        )
        cost_source = "execution-detail fallback"

    total_cost = submission_cost + execution_cost
    utility = realized_quality - total_cost

    completed_groups = {
        str(row.get("group") or row.get("task")) for row in selected
    }
    n_groups_completed = len(completed_groups)
    n_groups_total = _as_nonnegative_int(run.get("n_groups_total"))

    corrected.update(
        {
            "realized_quality": realized_quality,
            "quality": realized_quality,
            "submission_cost": submission_cost,
            "execution_cost": execution_cost,
            "total_cost": total_cost,
            "cost": total_cost,
            "utility": utility,
            "n_tasks_completed_valid": n_completed,
            "n_tasks_completed": n_completed,
            "n_tasks_reachable": denominator,
            "task_completion_rate": completion_rate,
            "reachable_task_completion_rate": completion_rate,
            "task_completion_rate_pct": 100.0 * completion_rate,
            "n_groups_completed": n_groups_completed,
            "corrected_from_executions": True,
            "quality_definition": "best valid execution per task",
            "completion_definition": "valid completed tasks / corrected reachable tasks",
            "cost_source": cost_source,
            "source_file": run_file.name,
            "execution_source_file": execution_file.name,
        }
    )

    if n_groups_total:
        corrected["group_completion_rate"] = n_groups_completed / n_groups_total

    n_tasks_total = _as_nonnegative_int(run.get("n_tasks_total"))
    if n_tasks_total:
        corrected["raw_task_completion_rate"] = n_completed / n_tasks_total

    completed_for_normalization = n_completed if n_completed else np.nan
    corrected["cost_per_completed_task"] = total_cost / completed_for_normalization
    corrected["quality_per_completed_task"] = (
        realized_quality / completed_for_normalization
    )
    corrected["utility_per_completed_task"] = utility / completed_for_normalization
    corrected["completions_per_cost"] = (
        n_completed / total_cost if total_cost > 0 else np.nan
    )
    return corrected


def load_corrected_results(folders: list[str], base_dir: str = ".") -> pd.DataFrame:
    """Load paired run/execution JSONs and reconstruct their primary metrics."""
    records: list[dict[str, Any]] = []
    run_id_map: dict[tuple[str, Any], int] = {}
    next_run_id = 1

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.is_absolute():
            folder_path = Path(base_dir) / folder

        if not folder_path.exists():
            print(f"  [Warning] Folder not found: {folder_path} â€” skipping")
            continue

        loaded = skipped = 0
        folder_name = folder_path.name
        for run_file in sorted(folder_path.glob("run_*.json")):
            execution_file = _find_execution_file(run_file)
            if execution_file is None:
                print(f"  [Warning] No unique execution file for {run_file.name} â€” skipping")
                skipped += 1
                continue

            try:
                run = _load_json(run_file)
                executions = _load_json(execution_file)
                if not isinstance(run, dict) or "scheduler" not in run:
                    raise ValueError("run JSON is not a scheduler summary")
                if not isinstance(executions, list):
                    raise ValueError("execution JSON is not a list")

                record = reconstruct_run(
                    run, executions, run_file=run_file, execution_file=execution_file
                )
                scheduler_source = str(record["scheduler"])
                record["scheduler_source"] = scheduler_source
                if scheduler_source in HEDGING_SCHEDULER_ALIASES:
                    record["scheduler"] = HEDGING_SCHEDULER_ID
                record["folder"] = folder_name

                local_id = run.get("run", run.get("seed", run_file.stem))
                mapping_key = (folder_name, local_id)
                if mapping_key not in run_id_map:
                    run_id_map[mapping_key] = next_run_id
                    next_run_id += 1
                record["run"] = run_id_map[mapping_key]

                records.append(record)
                loaded += 1
                print(
                    f"    {run_file.name}: Q={record['quality']:.4f}, "
                    f"C={record['cost']:.4f}, U={record['utility']:.4f}, "
                    f"completion={record['n_tasks_completed_valid']}/"
                    f"{record['n_tasks_reachable']} "
                    f"({record['task_completion_rate_pct']:.2f}%)"
                )
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                print(f"  [Warning] Could not reconstruct {run_file.name}: {exc}")
                skipped += 1

        print(
            f"  [Folder] {folder_name}: reconstructed {loaded} runs "
            f"(skipped {skipped})"
        )

    if not records:
        raise FileNotFoundError("No valid run/execution JSON pairs were found.")
    return pd.DataFrame(records)


def _canonical_plot_scheduler(value: str) -> str:
    """Convert configuration/CLI scheduler names to canonical identifiers."""
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "stochastic_log": HEDGING_SCHEDULER_ID,
        "stochastic_logical": HEDGING_SCHEDULER_ID,
        "hedging_ilp": HEDGING_SCHEDULER_ID,
        "hedging_milp": HEDGING_SCHEDULER_ID,
        "standard_ilp": "deterministic",
        "deterministic_ilp": "deterministic",
        "greedyn": "greedy_n",
    }
    return aliases.get(key, key)


def select_and_order_plot_schedulers(
    df: pd.DataFrame,
    requested: list[str] | None = None,
) -> pd.DataFrame:
    """Filter an optional scheduler subset and apply the fixed display order."""
    available = set(df["scheduler"].dropna().astype(str))

    if requested:
        selected = {_canonical_plot_scheduler(name) for name in requested}
        missing = selected - available
        if missing:
            raise ValueError(
                "Requested plot scheduler(s) not present in the results: "
                + ", ".join(sorted(missing))
            )
    else:
        selected = available

    ordered = [name for name in SCHEDULER_ORDER if name in selected]
    ordered.extend(sorted(selected - set(SCHEDULER_ORDER)))
    if not ordered:
        raise ValueError("The requested scheduler subset is empty.")

    rank = {name: index for index, name in enumerate(ordered)}
    plot_df = df[df["scheduler"].isin(ordered)].copy()
    plot_df["_scheduler_order"] = plot_df["scheduler"].map(rank)
    sort_columns = ["_scheduler_order"]
    if "run" in plot_df.columns:
        sort_columns.append("run")
    plot_df.sort_values(sort_columns, kind="stable", inplace=True)
    plot_df.drop(columns="_scheduler_order", inplace=True)

    labels = [SCHEDULER_LABELS.get(name, name) for name in ordered]
    print(f"  [Plots] Scheduler order: {' -> '.join(labels)}")
    return plot_df


def save_corrected_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the reconstructed table and one corrected JSON per run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "corrected_run_metrics.csv", index=False)

    corrected_dir = output_dir / "corrected_runs"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        source_stem = Path(str(row["source_file"])).stem
        destination = corrected_dir / f"corrected_{source_stem}.json"
        payload = {
            key: (None if isinstance(value, float) and np.isnan(value) else value)
            for key, value in row.to_dict().items()
        }
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str, allow_nan=False)

    print("  [Saved] corrected_run_metrics.csv and corrected_runs/*.json")


def save_hedging_timing_summary(df: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    """Report computational averages for the Hedging MILP runs only.

    Other schedulers do not currently save comparable solver statistics, so
    these values are intentionally reported as a standalone Hedging MILP
    summary rather than as a cross-scheduler performance comparison.
    """
    hedging = df[df["scheduler"] == HEDGING_SCHEDULER_ID].copy()
    if hedging.empty:
        print("  [Timing] No Hedging MILP runs found; timing summary skipped")
        return None

    metric_specs = [
        ("avg_solve_time_s", "mean_solve_time_s"),
        ("avg_wall_time_s", "mean_wall_time_s"),
        ("avg_mip_gap_pct", "mean_mip_gap_pct"),
        ("final_mip_gap_pct", "mean_final_mip_gap_pct"),
        ("n_planning_sessions", "mean_planning_sessions_per_run"),
    ]

    summary: dict[str, Any] = {
        "scheduler": "Hedging MILP",
        "n_runs": int(len(hedging)),
        "aggregation": "arithmetic mean across run-level summaries",
    }
    available = False
    for source, destination in metric_specs:
        if source not in hedging.columns:
            continue
        values = pd.to_numeric(hedging[source], errors="coerce").dropna()
        if values.empty:
            continue
        summary[destination] = float(values.mean())
        summary[f"n_runs_with_{source}"] = int(len(values))
        available = True

    if not available:
        print("  [Timing] Hedging MILP runs contain no solver timing fields")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hedging_milp_timing_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    lines = [
        "",
        "HEDGING MILP COMPUTATIONAL PERFORMANCE",
        "--------------------------------------",
        f"Runs included: {summary['n_runs']}",
    ]
    display_specs = [
        ("mean_solve_time_s", "Mean solve time per planning session", "s", 3),
        ("mean_wall_time_s", "Mean wall time per planning session", "s", 3),
        ("mean_mip_gap_pct", "Mean MIP gap", "%", 3),
        ("mean_final_mip_gap_pct", "Mean final MIP gap", "%", 3),
        ("mean_planning_sessions_per_run", "Mean planning sessions per run", "", 2),
    ]
    for key, label, unit, decimals in display_specs:
        if key in summary:
            suffix = f" {unit}" if unit else ""
            lines.append(f"{label}: {summary[key]:.{decimals}f}{suffix}")
    lines.extend(
        [
            "Note: comparable solver statistics were not saved for the other schedulers.",
            "These values therefore describe Hedging MILP performance only.",
        ]
    )

    timing_text = "\n".join(lines) + "\n"
    (output_dir / "hedging_milp_timing_summary.txt").write_text(
        timing_text, encoding="utf-8"
    )

    report_path = output_dir / "summary_report.txt"
    with report_path.open("a", encoding="utf-8") as stream:
        stream.write(timing_text)

    print(timing_text.rstrip())
    print("  [Saved] hedging_milp_timing_summary.json/.txt")
    print("  [Updated] summary_report.txt with Hedging MILP timing averages")
    return summary


def load_replan_records(folders: list[str], base_dir: str = ".") -> pd.DataFrame:
    """Read replans from every run JSON, independent of execution-file pairing."""
    records: list[dict[str, Any]] = []
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.is_absolute():
            folder_path = Path(base_dir) / folder
        if not folder_path.exists():
            continue

        for run_file in sorted(folder_path.glob("run_*.json")):
            try:
                run = _load_json(run_file)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(run, dict) or "scheduler" not in run:
                continue

            replans = run.get("replans", run.get("n_replans"))
            if replans is None:
                continue

            source_scheduler = str(run["scheduler"])
            scheduler = (
                HEDGING_SCHEDULER_ID
                if source_scheduler in HEDGING_SCHEDULER_ALIASES
                else _canonical_plot_scheduler(source_scheduler)
            )
            records.append(
                {
                    "scheduler": scheduler,
                    "replans": replans,
                    "seed": run.get("seed"),
                    "folder": folder_path.name,
                    "source_file": run_file.name,
                }
            )

    return pd.DataFrame(records)


def save_replans_by_scheduler(
    replan_df: pd.DataFrame, output_dir: Path
) -> dict[str, Any] | None:
    """Append mean replan counts for all schedulers to the main report."""
    if replan_df.empty or "replans" not in replan_df.columns:
        print("  [Replans] No usable replan records found")
        return None

    available = set(replan_df["scheduler"].dropna().astype(str))
    ordered = [scheduler for scheduler in SCHEDULER_ORDER if scheduler in available]
    ordered.extend(sorted(available - set(SCHEDULER_ORDER)))

    rows: list[dict[str, Any]] = []
    for scheduler in ordered:
        values = pd.to_numeric(
            replan_df.loc[replan_df["scheduler"] == scheduler, "replans"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "scheduler": scheduler,
                "label": SCHEDULER_LABELS.get(scheduler, scheduler),
                "n_runs": int(len(values)),
                "mean_replans": float(values.mean()),
                "std_replans": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_replans": int(values.min()),
                "max_replans": int(values.max()),
            }
        )

    if not rows:
        print("  [Replans] No scheduler contains numeric replan values")
        return None

    summary = {"metric": "replans", "schedulers": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "replans_by_scheduler.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    lines = ["", "MEAN REPLANS BY SCHEDULER", "-------------------------"]
    for row in rows:
        lines.append(
            f"{row['label']}: {row['mean_replans']:.2f} replans/run "
            f"(n={row['n_runs']}, std={row['std_replans']:.2f}, "
            f"range={row['min_replans']}-{row['max_replans']})"
        )
    report_text = "\n".join(lines) + "\n"

    (output_dir / "replans_by_scheduler.txt").write_text(
        report_text, encoding="utf-8"
    )
    with (output_dir / "summary_report.txt").open("a", encoding="utf-8") as stream:
        stream.write(report_text)

    print(report_text.rstrip())
    print("  [Saved] replans_by_scheduler.json/.txt")
    print("  [Updated] summary_report.txt with every scheduler's replans")
    return summary


def _plot_mean_bars_total_first(df: pd.DataFrame, output_dir: str, plots) -> None:
    """Plot overall metrics first and per-completed-task metrics second."""
    print("  Generating mean bar charts...")

    metrics = [
        ("task_completion_rate_pct", "Task Completion (%)"),
        ("utility", "Total Utility"),
        ("n_tasks_completed_valid", "Completed Tasks"),
        ("utility_per_completed_task", "Utility / Completed Task"),
        ("quality_per_completed_task", "Quality / Completed Task"),
        ("cost_per_completed_task", "Cost / Completed Task"),
    ]

    schedulers = df["scheduler"].unique()
    x = np.arange(len(schedulers))
    fig, axes = plots.plt.subplots(2, 3, figsize=(18, 10))

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        if metric not in df.columns or df[metric].dropna().empty:
            ax.text(
                0.5, 0.5, f"Missing:\n{metric}", ha="center", va="center",
                transform=ax.transAxes, fontsize=14,
            )
            ax.set_title(title, fontsize=15)
            continue

        means = [df[df["scheduler"] == s][metric].mean() for s in schedulers]
        stds = [df[df["scheduler"] == s][metric].std() for s in schedulers]
        colors = [plots._color(s) for s in schedulers]
        labels = [plots._label(s) for s in schedulers]

        bars = ax.bar(
            x, means, yerr=stds, capsize=6, color=colors, alpha=0.8,
            error_kw=dict(elinewidth=2, ecolor="#333333"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(title, fontsize=15, pad=10)
        ax.grid(axis="y", alpha=0.4)

        max_val = max([m for m in means if not np.isnan(m)] or [1])
        for bar, mean, std in zip(bars, means, stds):
            if np.isnan(mean):
                continue
            y_offset = (std if not np.isnan(std) else 0) + 0.02 * max_val
            value_text = (
                f"{mean:.2f}" if "per_completed_task" in metric else f"{mean:.1f}"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                value_text,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    plots.plt.suptitle(
        "Scheduler Comparison â€” Means Â± Std (Overall and Per-Completed-Task Metrics)",
        fontsize=18, y=1.02,
    )
    plots.plt.tight_layout()
    plots.plt.savefig(os.path.join(output_dir, "mean_bars.png"), bbox_inches="tight")
    print("  [Saved] mean_bars.png")
    plots.plt.close()


def generate_plots(
    df: pd.DataFrame,
    output_dir: Path,
    folders: list[str],
) -> list[tuple[str, str]]:
    """Call exactly the plotting suite used by plot_runs_results.py."""
    plots = _load_plot_module()
    plots.register_scheduler_styles(list(df["scheduler"].unique()))

    # This comparison does not include computational metrics because those
    # fields are populated only for the stochastic solver. Remove a stale copy
    # if the output directory was previously processed by the standard script.
    computational_plot = output_dir / "computational_metrics.png"
    if computational_plot.exists():
        computational_plot.unlink()

    # ``plot_scheduler_advantage`` expects at most one record for each
    # (run, scheduler) pair.  Timestamped result folders normally satisfy that,
    # but copied JSONs can create duplicates such as ``file(1).json``.
    advantage_df = df
    duplicate_mask = df.duplicated(subset=["run", "scheduler"], keep=False)
    if duplicate_mask.any():
        print(
            "  [Warning] Duplicate (run, scheduler) records found; using the "
            "last copy only for scheduler_advantage.png"
        )
        advantage_df = df.drop_duplicates(
            subset=["run", "scheduler"], keep="last"
        )

    plot_functions = [
        ("2D Pareto Quality vs Cost", plots.plot_pareto_2d),
        ("3D Pareto Quality vs Cost vs Completion", plots.plot_pareto_3d),
        ("Utility vs Task Completion", plots.plot_utility_vs_completion),
        ("Metric Boxplots", plots.plot_metric_boxplots),
        (
            "Mean Bar Charts",
            lambda data, target: _plot_mean_bars_total_first(
                data, target, plots
            ),
        ),
        ("Per-Run Lines", plots.plot_per_run_lines),
        (
            "Scheduler Advantage",
            lambda _data, target: plots.plot_scheduler_advantage(
                advantage_df, target
            ),
        ),
        (
            "Summary Report",
            lambda data, target: plots.generate_summary_report(
                data, target, folders
            ),
        ),
    ]

    failures: list[tuple[str, str]] = []
    for name, function in plot_functions:
        try:
            function(df, str(output_dir))
        except Exception as exc:  # keep generating the remaining independent plots
            print(f"  ERROR in {name}: {exc}")
            traceback.print_exc()
            failures.append((name, str(exc)))
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct valid metrics and generate the standard run plots."
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_DIR,
        help="Parent directory containing the run folders.",
    )
    parser.add_argument(
        "--run-folder",
        action="append",
        dest="run_folders",
        help="Run folder relative to --base-dir; repeat for multiple folders.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to <base>/<first run folder>/combined_results.",
    )
    parser.add_argument(
        "--plot-scheduler",
        action="append",
        dest="plot_schedulers",
        help=(
            "Scheduler to include in plots; repeat for a subset. Accepted names "
            "include hedging_milp, deterministic, greedy_n, greedy, and random. "
            "Defaults to PLOT_SCHEDULERS, or every available scheduler when that "
            "configuration value is None."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folders = args.run_folders or RUN_FOLDERS
    output_dir = Path(
        args.output_dir
        or os.path.join(args.base_dir, folders[0], "combined_results")
    )

    print("\n" + "=" * 72)
    print("CORRECTED ACTUAL-UTILITY POST-PROCESSING")
    print("=" * 72)
    print(f"Loading paired run and execution JSONs from {len(folders)} folder(s)...")

    try:
        df = load_corrected_results(folders, args.base_dir)
    except Exception as exc:
        print(f"ERROR loading results: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    save_corrected_metrics(df, output_dir)

    requested_schedulers = (
        args.plot_schedulers
        if args.plot_schedulers is not None
        else PLOT_SCHEDULERS
    )
    try:
        plot_df = select_and_order_plot_schedulers(df, requested_schedulers)
    except ValueError as exc:
        print(f"ERROR selecting plot schedulers: {exc}")
        raise SystemExit(1) from exc

    print(f"\nGenerating standard plots in: {output_dir}\n")
    failures = generate_plots(plot_df, output_dir, folders)
    save_hedging_timing_summary(df, output_dir)
    replan_df = load_replan_records(folders, args.base_dir)
    save_replans_by_scheduler(replan_df, output_dir)

    print("\n" + "=" * 72)
    if failures:
        print(f"POST-PROCESSING COMPLETE WITH {len(failures)} PLOT ERROR(S)")
        for name, error in failures:
            print(f"  x {name}: {error[:100]}")
    else:
        print("POST-PROCESSING COMPLETE")
    print("=" * 72)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
