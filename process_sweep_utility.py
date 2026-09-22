"""Cross-configuration post-processing for earthquake_config_sweep.py output.

process_actual_utility.py answers "how do the schedulers compare?" for ONE
configuration. This answers "how does that comparison change across
configurations?" -- the config becomes an axis instead of a constant.

Metrics are imported from process_actual_utility: quality, completion, and
utility come from each run summary (the live request table).  Execution files
are diagnostic only — older exports omitted dispatcher-legal OR-gate collects.

  python process_sweep_utility.py results\\sweep_2026-09-11_184943
  python process_sweep_utility.py <sweep_dir> --per-config-plots
  python process_sweep_utility.py <sweep_dir> --extra-folder earthquake_campaign_20260815_232847
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import process_actual_utility as pau

# ==============================================================================
# CONFIGURATION -- same hardcoded pattern as process_actual_utility.py
# ==============================================================================
BASE_DIR = r"E:\Code\federated-scheduling\results"

# The sweep directory to process, relative to BASE_DIR.
SWEEP_FOLDER = "sweep_2026-09-12_011124"

# Extra results folders included as unpositioned reference configs, for example
# a legacy campaign to hold the sweep against. They appear in the bar charts and
# report but not in the fleet-by-horizon heatmap, having no sweep metadata.
REFERENCE_FOLDERS: list[str] = [
    # "earthquake_campaign_20260815_232847",
]

# Also regenerate the standard single-config plot suite inside every config's
# own combined_results/, exactly as process_actual_utility.py would.
PER_CONFIG_PLOTS = False

OUTPUT_DIR = os.path.join(BASE_DIR, SWEEP_FOLDER, "cross_config")
# ==============================================================================

HEDGER = pau.HEDGING_SCHEDULER_ID
BASELINES = ("deterministic", "greedy_n", "greedy", "random")

# Metrics shown on the cross-config grid, and whether lower is better.
CROSS_METRICS = [
    ("utility", "Net Utility", False),
    ("task_completion_rate_pct", "Task Completion (%)", False),
    ("n_tasks_completed_valid", "Completed Tasks", False),
    ("utility_per_completed_task", "Utility / Completed Task", False),
    ("quality_per_completed_task", "Quality / Completed Task", False),
    ("cost_per_completed_task", "Cost / Completed Task", True),
]


# ==============================================================================
# DISCOVERY
# ==============================================================================

def discover_configs(sweep_dir: Path, extra_folders: list[str] | None = None) -> list[dict[str, Any]]:
    """Find every config directory in a sweep, newest-style metadata first.

    A directory qualifies if it holds run_*.json. sweep_config.json supplies the
    fleet/window metadata; folders without one (a hand-made or legacy results
    directory passed via --extra-folder) still load, with NaN metadata so they
    plot as an unpositioned reference.
    """
    configs: list[dict[str, Any]] = []

    for cfg_path in sorted(sweep_dir.glob("*/sweep_config.json")):
        cfg_dir = cfg_path.parent
        if not any(cfg_dir.glob("run_*.json")):
            print(f"  [Skip] {cfg_dir.name}: no run_*.json (config never produced results)")
            continue
        try:
            with cfg_path.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [Warning] {cfg_path}: {exc}")
            continue

        fleet = cfg.get("fleet") or {}
        windows = cfg.get("windows") or {}
        n_rgb = float(fleet.get("n_rgb", np.nan))
        n_sar = float(fleet.get("n_sar", np.nan))
        configs.append({
            "config_id": cfg.get("config_id", cfg_dir.name),
            "config_dir": cfg_dir,
            "fleet_key": cfg.get("fleet_key", ""),
            "windows_key": cfg.get("windows_key", ""),
            "n_rgb": n_rgb,
            "n_sar": n_sar,
            "fleet_total": n_rgb + n_sar,
            "horizon_h": float(windows.get("horizon_h", np.nan)),
            "extent_window_h": float(windows.get("extent_window_h", np.nan)),
            "is_reference": False,
        })

    for folder in extra_folders or []:
        cfg_dir = Path(folder)
        if not cfg_dir.is_absolute():
            for parent in (sweep_dir, Path(pau.BASE_DIR)):
                if (parent / folder).exists():
                    cfg_dir = parent / folder
                    break
        if not cfg_dir.exists() or not any(cfg_dir.glob("run_*.json")):
            print(f"  [Warning] Reference folder unusable: {cfg_dir}")
            continue
        configs.append({
            "config_id": cfg_dir.name,
            "config_dir": cfg_dir,
            "fleet_key": "reference",
            "windows_key": "reference",
            "n_rgb": np.nan, "n_sar": np.nan, "fleet_total": np.nan,
            "horizon_h": np.nan, "extent_window_h": np.nan,
            "is_reference": True,
        })

    return configs


def load_sweep(configs: list[dict[str, Any]], per_config_plots: bool = False) -> pd.DataFrame:
    """Reconstruct every config with process_actual_utility's exact semantics."""
    frames: list[pd.DataFrame] = []

    for cfg in configs:
        cfg_dir: Path = cfg["config_dir"]
        print(f"\n[Config] {cfg['config_id']}")
        try:
            df = pau.load_corrected_results([cfg_dir.name], str(cfg_dir.parent))
        except Exception as exc:
            print(f"  [Warning] Skipping {cfg['config_id']}: {exc}")
            continue

        for key in ("config_id", "fleet_key", "windows_key", "n_rgb", "n_sar",
                    "fleet_total", "horizon_h", "extent_window_h", "is_reference"):
            df[key] = cfg[key]
        frames.append(df)

        if per_config_plots:
            out = cfg_dir / "combined_results"
            out.mkdir(parents=True, exist_ok=True)
            pau.save_corrected_metrics(df, out)
            try:
                plot_df = pau.select_and_order_plot_schedulers(df, None)
                pau.generate_plots(plot_df, out, [cfg_dir.name])
                pau.save_hedging_timing_summary(df, out)
            except Exception as exc:
                print(f"  [Warning] Per-config plots failed for {cfg['config_id']}: {exc}")

    if not frames:
        raise FileNotFoundError("No config produced reconstructable run/execution pairs.")
    return pd.concat(frames, ignore_index=True)


# ==============================================================================
# ORDERING AND LABELS
# ==============================================================================

def order_configs(df: pd.DataFrame) -> list[str]:
    """Sort configs by horizon then fleet size, with references last."""
    keys = (df[["config_id", "horizon_h", "fleet_total", "is_reference"]]
            .drop_duplicates("config_id"))
    keys = keys.sort_values(
        by=["is_reference", "horizon_h", "fleet_total", "config_id"],
        na_position="last", kind="stable",
    )
    return list(keys["config_id"])


def config_label(df: pd.DataFrame, config_id: str) -> str:
    """Compact two-line tick label: fleet composition over campaign horizon."""
    row = df[df["config_id"] == config_id].iloc[0]
    if row["is_reference"] or not np.isfinite(row["fleet_total"]):
        return config_id.replace("earthquake_", "").replace("campaign_", "")
    return f"{row['n_rgb']:.0f}R+{row['n_sar']:.0f}S\n{row['horizon_h']:.0f}h"


def order_schedulers(df: pd.DataFrame) -> list[str]:
    available = set(df["scheduler"].dropna().astype(str))
    ordered = [s for s in pau.SCHEDULER_ORDER if s in available]
    ordered.extend(sorted(available - set(pau.SCHEDULER_ORDER)))
    return ordered


# ==============================================================================
# TABLES
# ==============================================================================

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m, _, _ in CROSS_METRICS] + [
        m for m in ("realized_quality", "total_cost", "group_completion_rate",
                    "avg_solve_time_s", "avg_mip_gap_pct", "final_mip_gap_pct",
                    "n_tasks_reachable")
        if m in df.columns
    ]
    metrics = [m for m in dict.fromkeys(metrics) if m in df.columns]
    agg = (df.groupby(["config_id", "fleet_key", "windows_key", "horizon_h",
                       "n_rgb", "n_sar", "fleet_total", "scheduler"],
                      dropna=False)[metrics]
             .agg(["mean", "std", "count"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def build_verdict(summary: pd.DataFrame) -> pd.DataFrame:
    """Per config: does hedging beat the best baseline, and by how much?"""
    rows: list[dict[str, Any]] = []
    for cid, sub in summary.groupby("config_id"):
        util = dict(zip(sub["scheduler"], sub["utility_mean"]))
        if HEDGER not in util:
            continue
        rivals = {k: v for k, v in util.items() if k in BASELINES and np.isfinite(v)}
        if not rivals:
            continue
        best = max(rivals, key=rivals.get)
        head = sub.iloc[0]
        row = {
            "config_id": cid,
            "fleet_key": head["fleet_key"], "windows_key": head["windows_key"],
            "horizon_h": head["horizon_h"], "fleet_total": head["fleet_total"],
            "hedge_utility": util[HEDGER],
            "best_baseline": pau.SCHEDULER_LABELS.get(best, best),
            "best_baseline_utility": rivals[best],
            "hedge_advantage": util[HEDGER] - rivals[best],
            "hedge_advantage_pct": (100.0 * (util[HEDGER] - rivals[best]) / abs(rivals[best])
                                    if rivals[best] else np.nan),
            "hedge_wins": bool(util[HEDGER] > rivals[best]),
            "n_runs": int(sub.loc[sub["scheduler"] == HEDGER, "utility_count"].iloc[0]),
        }
        hedge_row = sub[sub["scheduler"] == HEDGER]
        for src, dst in (("utility_std", "hedge_utility_std"),
                         ("avg_mip_gap_pct_mean", "mip_gap_pct"),
                         ("avg_solve_time_s_mean", "solve_time_s"),
                         ("task_completion_rate_pct_mean", "hedge_completion_pct")):
            row[dst] = float(hedge_row[src].iloc[0]) if src in hedge_row.columns else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("hedge_advantage", ascending=False)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, verdict: pd.DataFrame,
                 output_dir: Path) -> None:
    lines = ["=" * 78, "CROSS-CONFIGURATION SCHEDULER COMPARISON", "=" * 78, ""]
    lines.append(f"Configurations: {df['config_id'].nunique()}")
    lines.append(f"Runs total:     {len(df)}")
    lines.append(f"Seeds/config:   {sorted(df.groupby('config_id')['run'].nunique().unique())}")
    lines.append("Quality, completion, utility, and cost from run summaries "
                 "(live request table; dispatcher-enforced gates).")
    lines.append("")

    if not verdict.empty:
        lines += ["", "HEDGING MILP vs BEST BASELINE (ranked by advantage)",
                  "-" * 78]
        for _, r in verdict.iterrows():
            flag = "WINS " if r["hedge_wins"] else "loses"
            gap = "" if not np.isfinite(r.get("mip_gap_pct", np.nan)) \
                else f", mip_gap={r['mip_gap_pct']:.1f}%"
            lines.append(
                f"{flag} {r['config_id']:34s} adv={r['hedge_advantage']:+8.1f} "
                f"({r['hedge_utility']:.1f} vs {r['best_baseline_utility']:.1f} "
                f"{r['best_baseline']}, n={r['n_runs']}{gap})"
            )
        n_win = int(verdict["hedge_wins"].sum())
        lines += ["", f"Hedging wins in {n_win}/{len(verdict)} configurations."]
        suspect = verdict[verdict.get("mip_gap_pct", pd.Series(dtype=float)) > 25.0]
        if not suspect.empty:
            lines += ["",
                      "CAUTION: these configs carry a mean MIP gap above 25%, so their",
                      "hedging result reflects solver truncation as much as policy:"]
            lines += [f"  - {r['config_id']} ({r['mip_gap_pct']:.1f}%)"
                      for _, r in suspect.iterrows()]

    lines += ["", "", "PER-CONFIG MEANS", "-" * 78]
    for cid in order_configs(df):
        sub = summary[summary["config_id"] == cid]
        if sub.empty:
            continue
        head = sub.iloc[0]
        lines.append(f"\n{cid}  (fleet={head['fleet_total']:.0f}, "
                     f"horizon={head['horizon_h']:.0f}h)")
        for sched in order_schedulers(df):
            row = sub[sub["scheduler"] == sched]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(
                f"  {pau.SCHEDULER_LABELS.get(sched, sched):<14s} "
                f"U={row['utility_mean']:8.1f}  "
                f"completion={row['task_completion_rate_pct_mean']:5.1f}%  "
                f"tasks={row['n_tasks_completed_valid_mean']:5.1f}  "
                f"cost/task={row['cost_per_completed_task_mean']:6.2f}  "
                f"(n={int(row['utility_count'])})"
            )

    text = "\n".join(lines) + "\n"
    (output_dir / "cross_config_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print("  [Saved] cross_config_report.txt")


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_cross_config_bars(df: pd.DataFrame, output_dir: Path, plots) -> None:
    """The workhorse: config on x, one bar group per scheduler, six metrics."""
    configs = order_configs(df)
    schedulers = order_schedulers(df)
    labels = [config_label(df, c) for c in configs]
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(schedulers))

    fig, axes = plt.subplots(2, 3, figsize=(6.2 * 3, 10))
    for ax, (metric, title, lower_better) in zip(axes.flatten(), CROSS_METRICS):
        if metric not in df.columns:
            ax.text(0.5, 0.5, f"Missing:\n{metric}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            ax.set_title(title, fontsize=14)
            continue

        for j, sched in enumerate(schedulers):
            means, stds = [], []
            for cid in configs:
                vals = df[(df["config_id"] == cid) & (df["scheduler"] == sched)][metric]
                vals = pd.to_numeric(vals, errors="coerce").dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std() if len(vals) > 1 else 0.0)
            ax.bar(x + (j - (len(schedulers) - 1) / 2) * width, means, width,
                   yerr=stds, capsize=3, label=plots._label(sched),
                   color=plots._color(sched), alpha=0.88,
                   error_kw=dict(elinewidth=1.2, ecolor="#444444"))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(title + (" (lower better)" if lower_better else ""), fontsize=11)
        ax.set_title(title, fontsize=14, pad=8)
        ax.grid(axis="y", alpha=0.35)
        ax.axhline(0, color="#888888", lw=0.8)

    handles, legend_labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(schedulers),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Scheduler comparison across configurations  "
                 "(x tick = RGB+SAR satellites / campaign horizon)",
                 fontsize=16, y=1.045)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_config_bars.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  [Saved] cross_config_bars.png")


def plot_hedge_advantage_heatmap(verdict: pd.DataFrame, output_dir: Path) -> None:
    """Fleet x window grid of hedging advantage, diverging around zero."""
    grid = verdict[np.isfinite(verdict["fleet_total"]) & np.isfinite(verdict["horizon_h"])]
    if grid.empty:
        print("  [Skip] heatmap: no configs carry fleet/horizon metadata")
        return

    fleets = sorted(grid["fleet_total"].unique())
    horizons = sorted(grid["horizon_h"].unique())
    matrix = np.full((len(fleets), len(horizons)), np.nan)
    for _, r in grid.iterrows():
        matrix[fleets.index(r["fleet_total"]), horizons.index(r["horizon_h"])] = r["hedge_advantage"]

    # Scale to the 75th percentile, not the max: a single blowout config would
    # otherwise flatten every other cell to indistinguishable pale. Exact values
    # are annotated, so clipping the colour costs nothing.
    magnitudes = np.abs(matrix[np.isfinite(matrix)])
    span = float(np.percentile(magnitudes, 75)) if magnitudes.size else 1.0
    span = max(span, 1.0)
    clipped = bool(magnitudes.size and magnitudes.max() > span)

    fig, ax = plt.subplots(figsize=(1.9 * len(horizons) + 3.2, 1.5 * len(fleets) + 2.6))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h:.0f} h" for h in horizons], fontsize=12)
    ax.set_yticks(range(len(fleets)))
    ax.set_yticklabels([f"{f:.0f} sats" for f in fleets], fontsize=12)
    ax.set_xlabel("Campaign horizon", fontsize=13)
    ax.set_ylabel("Fleet size", fontsize=13)
    subtitle = "blue = hedging wins, red = a baseline wins"
    if clipped:
        subtitle += f"  (colour clipped at +/-{span:.0f}; labels are exact)"
    ax.set_title("Hedging MILP utility minus best baseline\n" + subtitle,
                 fontsize=14, pad=12)

    for i in range(len(fleets)):
        for j in range(len(horizons)):
            if not np.isfinite(matrix[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color="#999999")
                continue
            shade = "white" if abs(matrix[i, j]) > 0.62 * span else "#111111"
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center",
                    color=shade, fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, extend="both" if clipped else "neither",
                 label="Utility advantage")
    fig.tight_layout()
    fig.savefig(output_dir / "hedge_advantage_heatmap.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  [Saved] hedge_advantage_heatmap.png")


def plot_metric_trends(df: pd.DataFrame, output_dir: Path, plots,
                       axis: str, facet: str, filename: str,
                       axis_label: str, facet_label: str) -> None:
    """Metric vs one config axis, one line per scheduler, faceted by the other."""
    sub = df[np.isfinite(df[axis]) & np.isfinite(df[facet])]
    if sub.empty:
        print(f"  [Skip] {filename}: no configs carry both {axis} and {facet}")
        return

    facets = sorted(sub[facet].unique())
    metrics = [("utility", "Net Utility"),
               ("task_completion_rate_pct", "Task Completion (%)"),
               ("cost_per_completed_task", "Cost / Completed Task")]
    schedulers = order_schedulers(sub)

    fig, axes = plt.subplots(len(metrics), len(facets),
                             figsize=(4.6 * len(facets), 3.5 * len(metrics)),
                             squeeze=False, sharex=True)
    for r, (metric, title) in enumerate(metrics):
        for c, fval in enumerate(facets):
            ax = axes[r][c]
            pane = sub[sub[facet] == fval]
            for sched in schedulers:
                s = pane[pane["scheduler"] == sched]
                if s.empty:
                    continue
                g = (s.groupby(axis)[metric]
                      .agg(["mean", "std", "count"]).reset_index().sort_values(axis))
                err = g["std"].where(g["count"] > 1, 0.0)
                ax.errorbar(g[axis], g["mean"], yerr=err, capsize=3,
                            marker=plots._marker(sched), linestyle=plots._linestyle(sched),
                            color=plots._color(sched), label=plots._label(sched),
                            lw=2, markersize=7)
            ax.grid(alpha=0.35)
            ax.axhline(0, color="#888888", lw=0.8)
            if r == 0:
                ax.set_title(f"{facet_label} = {fval:.0f}", fontsize=13)
            if c == 0:
                ax.set_ylabel(title, fontsize=12)
            if r == len(metrics) - 1:
                ax.set_xlabel(axis_label, fontsize=12)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(schedulers),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Scheduler performance vs {axis_label.lower()}", fontsize=16, y=1.06)
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  [Saved] {filename}")


def plot_advantage_ranking(verdict: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bars of hedging advantage, so the winning configs read first."""
    if verdict.empty:
        return
    v = verdict.sort_values("hedge_advantage")
    y = np.arange(len(v))
    colors = ["#2166ac" if w else "#b2182b" for w in v["hedge_wins"]]

    fig, ax = plt.subplots(figsize=(10, 0.62 * len(v) + 2.4))
    ax.barh(y, v["hedge_advantage"], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(v["config_id"], fontsize=10)
    ax.axvline(0, color="#333333", lw=1.2)
    ax.set_xlabel("Hedging MILP utility minus best baseline", fontsize=12)
    ax.set_title("Where hedging pays off", fontsize=15, pad=10)
    ax.grid(axis="x", alpha=0.35)

    span = max(abs(v["hedge_advantage"].min()), abs(v["hedge_advantage"].max())) or 1.0
    for yi, (adv, rival) in enumerate(zip(v["hedge_advantage"], v["best_baseline"])):
        offset = 0.02 * span * (1 if adv >= 0 else -1)
        ax.text(adv + offset, yi, f"{adv:+.1f}  (vs {rival})",
                va="center", ha="left" if adv >= 0 else "right", fontsize=9)
    ax.set_xlim(-1.45 * span, 1.45 * span)

    fig.tight_layout()
    fig.savefig(output_dir / "hedge_advantage_ranking.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  [Saved] hedge_advantage_ranking.png")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    sweep_dir = Path(BASE_DIR) / SWEEP_FOLDER
    if not sweep_dir.exists():
        raise SystemExit(f"Sweep directory not found: {sweep_dir}")
    output_dir = Path(OUTPUT_DIR)

    print("\n" + "=" * 78)
    print("CROSS-CONFIGURATION POST-PROCESSING")
    print("=" * 78)
    print(f"Sweep: {sweep_dir}")

    configs = discover_configs(sweep_dir, REFERENCE_FOLDERS)
    if not configs:
        raise SystemExit(f"No config directories with run_*.json under {sweep_dir}")
    print(f"Found {len(configs)} configuration(s) with results.")

    df = load_sweep(configs, per_config_plots=PER_CONFIG_PLOTS)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    verdict = build_verdict(summary)
    df.to_csv(output_dir / "cross_config_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "cross_config_summary.csv", index=False)
    if not verdict.empty:
        verdict.to_csv(output_dir / "cross_config_verdict.csv", index=False)
    print("\n  [Saved] cross_config_run_metrics.csv, cross_config_summary.csv, "
          "cross_config_verdict.csv")

    plots = pau._load_plot_module()
    plots.register_scheduler_styles(list(df["scheduler"].unique()))

    print(f"\nGenerating cross-config figures in {output_dir}\n")
    tasks = [
        ("cross-config bars", lambda: plot_cross_config_bars(df, output_dir, plots)),
        ("advantage heatmap", lambda: plot_hedge_advantage_heatmap(verdict, output_dir)),
        ("advantage ranking", lambda: plot_advantage_ranking(verdict, output_dir)),
        ("vs horizon", lambda: plot_metric_trends(
            df, output_dir, plots, "horizon_h", "fleet_total",
            "metrics_vs_horizon.png", "Campaign horizon (h)", "Fleet")),
        ("vs fleet size", lambda: plot_metric_trends(
            df, output_dir, plots, "fleet_total", "horizon_h",
            "metrics_vs_fleet.png", "Fleet size (satellites)", "Horizon")),
    ]
    failures: list[tuple[str, str]] = []
    for name, fn in tasks:
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR in {name}: {exc}")
            traceback.print_exc()
            failures.append((name, str(exc)))

    write_report(df, summary, verdict, output_dir)

    print("=" * 78)
    if failures:
        print(f"COMPLETE WITH {len(failures)} FIGURE ERROR(S)")
        for name, err in failures:
            print(f"  x {name}: {err[:110]}")
    else:
        print("COMPLETE")
    print("=" * 78)
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
