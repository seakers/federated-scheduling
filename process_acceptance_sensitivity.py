"""Post-process acceptance_sensitivity_sweep.py results into comparison plots.

For each case study, shows how mean metrics move with planner p_accept error
margin (0 / 5 / 10 / 20 %), with one series per scheduler:

  - average utility
  - average task completion (%)
  - cost / completed task
  - utility / completed task

  python process_acceptance_sensitivity.py
  python process_acceptance_sensitivity.py results/acc_sens_2026-09-16_172333
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import process_actual_utility as pau

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = r"E:\Code\federated-scheduling\results"
SWEEP_FOLDER = "acc_sens_2026-09-17_085950"
# ==============================================================================

HEDGER_ALIASES = pau.HEDGING_SCHEDULER_ALIASES
HEDGER_ID = pau.HEDGING_SCHEDULER_ID

# Display order: hedger first, then common baselines.
SCHEDULER_ORDER = (
    HEDGER_ID,
    "deterministic",
    "greedy_n",
    "greedy",
    "random",
)

METRICS = (
    ("utility", "Mean utility", False),
    ("task_completion_rate_pct", "Mean task completion (%)", False),
    ("cost_per_completed_task", "Mean cost / completed task", True),
    ("utility_per_completed_task", "Mean utility / completed task", False),
)


def _load_plot_styles():
    try:
        import plot_runs_results as plots
    except ImportError as exc:
        raise SystemExit("plot_runs_results.py is required for colors/labels") from exc
    if "stochastic_log" in plots.SCHEDULER_COLORS:
        for sid in (*HEDGER_ALIASES, HEDGER_ID):
            plots.SCHEDULER_COLORS.setdefault(sid, plots.SCHEDULER_COLORS["stochastic_log"])
            if hasattr(plots, "SCHEDULER_LABELS"):
                plots.SCHEDULER_LABELS.setdefault(sid, "Hedging MILP")
    # AccSens plots: show Standard ILP (not Deterministic ILP).
    if hasattr(plots, "SCHEDULER_LABELS"):
        plots.SCHEDULER_LABELS["deterministic"] = "Standard ILP"
    return plots


def _canonical_scheduler(name: str) -> str:
    n = str(name).strip().lower()
    if n in HEDGER_ALIASES or n in ("hedging_milp", "hedging_ilp"):
        return HEDGER_ID
    return n


def _completed_count(row: pd.Series) -> float:
    for key in ("n_tasks_completed_valid", "n_tasks_completed"):
        if key in row.index and pd.notna(row[key]):
            try:
                v = float(row[key])
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    # Fallback from rate × reachable/total if present.
    rate = row.get("task_completion_rate")
    for den_key in ("n_tasks_reachable", "n_tasks_total"):
        den = row.get(den_key)
        if pd.notna(rate) and pd.notna(den) and float(den) > 0:
            return float(rate) * float(den)
    return np.nan


def load_per_run(sweep_dir: Path) -> pd.DataFrame:
    """Load acc_sens_per_run.csv, or rebuild from cell run_*.json files."""
    csv_path = sweep_dir / "acc_sens_per_run.csv"
    rows: list[dict] = []

    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        rows = df.to_dict(orient="records")
    else:
        for cfg_path in sorted(sweep_dir.glob("*/sweep_config.json")):
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            cfg_dir = cfg_path.parent
            for run_path in sorted(cfg_dir.glob("run_*.json")):
                try:
                    with open(run_path, encoding="utf-8") as fh:
                        rec = json.load(fh)
                except Exception as exc:
                    print(f"[Warning] {run_path}: {exc}")
                    continue
                rec["config_id"] = cfg.get("config_id", cfg_dir.name)
                rec["case_study"] = cfg.get("case_study", rec.get("case_study"))
                rec["relative_error"] = float(
                    cfg.get("relative_error", rec.get("relative_error", 0.0))
                )
                rows.append(rec)

    if not rows:
        raise SystemExit(f"No run records under {sweep_dir}")

    df = pd.DataFrame(rows)
    df["scheduler"] = df["scheduler"].map(_canonical_scheduler)
    df["relative_error"] = pd.to_numeric(df["relative_error"], errors="coerce").fillna(0.0)
    df["utility"] = pd.to_numeric(df.get("utility"), errors="coerce")
    df["total_cost"] = pd.to_numeric(
        df["total_cost"] if "total_cost" in df.columns else df.get("cost"),
        errors="coerce",
    )
    if "task_completion_rate" in df.columns:
        df["task_completion_rate"] = pd.to_numeric(
            df["task_completion_rate"], errors="coerce"
        )
    else:
        df["task_completion_rate"] = np.nan
    df["task_completion_rate_pct"] = 100.0 * df["task_completion_rate"]

    completed = df.apply(_completed_count, axis=1)
    df["n_completed_for_norm"] = completed
    df["cost_per_completed_task"] = df["total_cost"] / completed
    df["utility_per_completed_task"] = df["utility"] / completed
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "utility",
        "task_completion_rate_pct",
        "cost_per_completed_task",
        "utility_per_completed_task",
        "total_cost",
        "n_completed_for_norm",
    ]
    present = [m for m in metrics if m in df.columns]
    agg = (
        df.groupby(["case_study", "relative_error", "scheduler"], dropna=False)[present]
        .agg(["mean", "std", "count"])
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def _ordered_schedulers(present: list[str]) -> list[str]:
    ordered = [s for s in SCHEDULER_ORDER if s in present]
    extras = sorted(s for s in present if s not in ordered)
    return ordered + extras


def _error_tick_label(err: float) -> str:
    return f"{100.0 * float(err):.0f}%"


def plot_case_study(agg: pd.DataFrame, case: str, out_dir: Path, plots) -> None:
    sub = agg[agg["case_study"] == case].copy()
    if sub.empty:
        print(f"  [Skip] no rows for case_study={case}")
        return

    errors = sorted(sub["relative_error"].unique())
    schedulers = _ordered_schedulers(list(sub["scheduler"].unique()))
    plots.register_scheduler_styles(schedulers)

    n_err = len(errors)
    n_sched = len(schedulers)
    x = np.arange(n_err)
    width = min(0.8 / max(n_sched, 1), 0.18)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, (metric, title, lower_better) in zip(axes, METRICS):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in sub.columns:
            ax.set_title(f"{title} (missing)")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        for i, sched in enumerate(schedulers):
            means, stds = [], []
            for err in errors:
                row = sub[(sub["relative_error"] == err) & (sub["scheduler"] == sched)]
                if row.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(float(row[mean_col].iloc[0]))
                    std_v = row[std_col].iloc[0] if std_col in row.columns else np.nan
                    stds.append(0.0 if pd.isna(std_v) else float(std_v))

            offset = (i - (n_sched - 1) / 2.0) * width
            color = plots._color(sched)
            label = plots._label(sched)
            ax.bar(
                x + offset, means, width=width, yerr=stds, capsize=3,
                color=color, alpha=0.88, label=label,
                error_kw=dict(elinewidth=1.2, ecolor="#333333"),
            )

        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([_error_tick_label(e) for e in errors])
        ax.set_xlabel("Planner p_accept relative error")
        ax.grid(axis="y", alpha=0.35)
        if lower_better:
            ax.annotate(
                "lower is better", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="#666666",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=min(len(schedulers), 6),
        frameon=False, bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"Acceptance-probability estimate sensitivity - {case}",
        fontsize=15, y=1.06,
    )
    fig.tight_layout()
    out_path = out_dir / f"sensitivity_{case}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def plot_hedger_degradation(agg: pd.DataFrame, out_dir: Path, plots) -> None:
    """One panel per metric: hedger mean vs error margin, one line per case study."""
    hedger = agg[agg["scheduler"] == HEDGER_ID].copy()
    if hedger.empty:
        print("  [Skip] hedger degradation plot (no hedging runs)")
        return

    cases = sorted(hedger["case_study"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for ax, (metric, title, _) in zip(axes, METRICS):
        mean_col = f"{metric}_mean"
        for i, case in enumerate(cases):
            sub = hedger[hedger["case_study"] == case].sort_values("relative_error")
            if mean_col not in sub.columns or sub.empty:
                continue
            ax.plot(
                100.0 * sub["relative_error"].values,
                sub[mean_col].values,
                marker="o", linewidth=2, label=case, color=cmap(i),
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Planner p_accept relative error (%)")
        ax.grid(alpha=0.35)

    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(
        "Hedging MILP vs p_accept estimate error (mean over seeds)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / "sensitivity_hedger_vs_error.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def write_summary_tables(agg: pd.DataFrame, out_dir: Path) -> None:
    out_csv = out_dir / "sensitivity_by_case_error_scheduler.csv"
    agg.to_csv(out_csv, index=False)
    print(f"  [Saved] {out_csv}")

    # Compact wide table for the four headline metrics.
    pieces = []
    for metric, title, _ in METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in agg.columns:
            continue
        wide = agg.pivot_table(
            index=["case_study", "relative_error"],
            columns="scheduler",
            values=mean_col,
            aggfunc="first",
        )
        wide.columns = [f"{c}__{metric}" for c in wide.columns]
        pieces.append(wide)
    if pieces:
        wide_all = pd.concat(pieces, axis=1).reset_index()
        wide_path = out_dir / "sensitivity_means_wide.csv"
        wide_all.to_csv(wide_path, index=False)
        print(f"  [Saved] {wide_path}")


def process(sweep_dir: Path) -> None:
    plots = _load_plot_styles()
    out_dir = sweep_dir / "sensitivity_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AccSens post] loading {sweep_dir}")
    df = load_per_run(sweep_dir)
    print(
        f"  {len(df)} runs  |  cases={sorted(df['case_study'].dropna().unique())}  "
        f"|  schedulers={sorted(df['scheduler'].unique())}"
    )

    # Drop volcano cells that only have the hedger (failed baselines from old driver).
    counts = df.groupby(["case_study", "relative_error"])["scheduler"].nunique()
    thin = counts[counts < 2]
    if not thin.empty:
        print("  [Note] cells with <2 schedulers (incomplete -- often old volcano runs):")
        for (case, err), n in thin.items():
            print(f"    {case} @ {100*err:.0f}% -> {n} scheduler(s)")

    agg = aggregate(df)
    write_summary_tables(agg, out_dir)

    for case in sorted(df["case_study"].dropna().unique()):
        plot_case_study(agg, case, out_dir, plots)
    plot_hedger_degradation(agg, out_dir, plots)
    print(f"[AccSens post] done -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot acceptance-probability sensitivity sweep results",
    )
    ap.add_argument(
        "sweep_dir",
        nargs="?",
        default=os.path.join(BASE_DIR, SWEEP_FOLDER),
        help="Path to acc_sens_* results directory",
    )
    args = ap.parse_args()
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        raise SystemExit(f"Not a directory: {sweep_dir}")
    process(sweep_dir)


if __name__ == "__main__":
    main()
