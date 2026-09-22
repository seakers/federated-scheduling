"""
Config sweep for the earthquake damage-assessment study.

Runs earthquake_damage_assesment.py across combinations of fleet size and task
time windows WITHOUT editing that file or earthquake_utils.py. Each combination
runs in its own subprocess (fresh Gurobi env, no cross-config state leakage,
crash isolation), writing to results/sweep_<ts>/<config_id>/.

  python earthquake_config_sweep.py --dry-run          # print grid + validation
  python earthquake_config_sweep.py --runs 3
  python earthquake_config_sweep.py --runs 2 --schedulers stochastic_logical greedy
  python earthquake_config_sweep.py --rollup-only results/sweep_2026-09-11_180000
"""

import argparse
import datetime as dt
import itertools
import json
import glob
import os
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.abspath(__file__))

# ============================== THE GRID ====================================
# Fleet: (n_rgb, n_sar) slice sizes applied to the shared TLE fleet, matching
# load_satellites_for_earthquake's manual slicing.
FLEETS = {
    "fleet_small":   {"n_rgb": 18, "n_sar": 20},
    "fleet_mid":     {"n_rgb": 30, "n_sar": 20},
    "fleet_current": {"n_rgb": 50, "n_sar": 30},
}

# Windows. horizon_h is applied to BOTH earthquake_utils.CAMPAIGN_HORIZON_H and
# the driver's LOOKAHEAD_HORIZON_H: _validate_24h_timing requires them equal.
# Invariant (also enforced there): extent_window_h + max(closes) <= horizon_h.
WINDOWS = {
    "win_18h_tight": {   # current working tree
        "horizon_h": 18.0, "extent_window_h": 3.0,
        "urban": (0.0, 6.0), "triage": (6.0, 12.0), "final": (12.0, 15.0),
    },
    "win_24h_compressed": {   # the Aug 17 campaign
        "horizon_h": 24.0, "extent_window_h": 6.0,
        "urban": (1.0, 6.0), "triage": (6.0, 10.0), "final": (12.0, 18.0),
    },
    "win_56h_full": {   # the Aug 15 campaign, recovered from pass-time maxima
        "horizon_h": 56.0, "extent_window_h": 8.0,
        "urban": (1.0, 12.0), "triage": (6.0, 24.0), "final": (12.0, 48.0),
    },
}

BASELINES = ("greedy", "greedy_n", "deterministic", "random")
HEDGER = "stochastic_logical"


def validate_windows(w):
    """Mirror earthquake_utils._validate_24h_timing so bad combos fail in 0 ms
    rather than after a TLE load."""
    errs = []
    for name in ("urban", "triage", "final"):
        lo, hi = w[name]
        if lo < 0.0 or hi <= lo:
            errs.append(f"{name} window [{lo:g}, {hi:g}]: close must exceed open")
    latest = w["extent_window_h"] + max(w["urban"][1], w["triage"][1], w["final"][1])
    if latest > w["horizon_h"] + 1e-9:
        errs.append(f"latest close is EXT+{latest:g} h but horizon is {w['horizon_h']:g} h")
    return errs


def build_grid(fleets, windows):
    grid = []
    for (fk, fv), (wk, wv) in itertools.product(sorted(fleets.items()),
                                                sorted(windows.items())):
        grid.append({"config_id": f"{fk}__{wk}", "fleet_key": fk, "windows_key": wk,
                     "fleet": dict(fv), "windows": dict(wv),
                     "errors": validate_windows(wv)})
    return grid


# ============================== WORKER ======================================

def _apply_config(cfg):
    """Patch module globals in place. Safe because every consumer reads these at
    call time: the window constants inside create_earthquake_workflow, and
    MIN_ELEVATION_DEG inside _req."""
    import earthquake_utils as eu
    import earthquake_damage_assesment as drv
    from fame_geometry import InstrumentType

    w = cfg["windows"]
    eu.CAMPAIGN_HORIZON_H = float(w["horizon_h"])
    eu.EXTENT_WINDOW_H = float(w["extent_window_h"])
    eu.URBAN_OPEN_H,  eu.URBAN_CLOSE_H  = float(w["urban"][0]),  float(w["urban"][1])
    eu.TRIAGE_OPEN_H, eu.TRIAGE_CLOSE_H = float(w["triage"][0]), float(w["triage"][1])
    eu.FINAL_OPEN_H,  eu.FINAL_CLOSE_H  = float(w["final"][0]),  float(w["final"][1])
    if w.get("min_elevation_deg") is not None:
        eu.MIN_ELEVATION_DEG = float(w["min_elevation_deg"])

    # Must equal eu.CAMPAIGN_HORIZON_H or create_earthquake_workflow raises.
    drv.LOOKAHEAD_HORIZON_H = float(w["horizon_h"])
    if cfg.get("max_solver_time_s"):
        drv.MAX_SOLVER_TIME_S = int(cfg["max_solver_time_s"])
    if cfg.get("max_num_instances"):
        drv.MAX_NUM_INSTANCES = int(cfg["max_num_instances"])
    if cfg.get("schedulers"):
        drv.SCHEDULERS = list(cfg["schedulers"])   # drives both the run loop and aggregate()

    f = cfg["fleet"]
    n_rgb, n_sar = int(f["n_rgb"]), int(f["n_sar"])
    exclude = tuple(o.upper() for o in (f.get("exclude_operators") or []))
    dedupe = bool(f.get("dedupe", False))   # False reproduces the driver exactly
    _load_full = drv._load_satellites_once_shared

    def _fleet_loader(sim_start, horizon_h):
        full = _load_full(sim_start, horizon_h)
        if exclude:
            full = [s for s in full if not any(o in s.name.upper() for o in exclude)]
        rgb = [s for s in full if InstrumentType.RGB in s.instruments][:n_rgb]
        sar = [s for s in full if InstrumentType.SAR in s.instruments][:n_sar]
        fleet = rgb + sar
        if dedupe:
            seen, uniq = set(), []
            for s in fleet:
                if id(s) not in seen:
                    seen.add(id(s))
                    uniq.append(s)
            fleet = uniq
        print(f"[Sweep fleet] {len(fleet)} satellites "
              f"({len(rgb)} RGB, {len(sar)} SAR) out of {len(full)}")
        return fleet

    drv.load_satellites_for_earthquake = _fleet_loader
    return drv


def run_worker(config_path):
    os.chdir(REPO)
    with open(config_path) as fh:
        cfg = json.load(fh)
    drv = _apply_config(cfg)
    print(f"[Sweep] config={cfg['config_id']} horizon={cfg['windows']['horizon_h']:g}h "
          f"fleet={cfg['fleet']['n_rgb']}RGB+{cfg['fleet']['n_sar']}SAR "
          f"runs={cfg['runs']}")
    drv.run_comparison(num_monte_carlo_runs=int(cfg["runs"]),
                       results_dir=cfg["results_dir"],
                       plot_schedule=bool(cfg.get("plot_schedule", False)))


# ============================== ROLL-UP =====================================

def rollup(sweep_dir):
    import pandas as pd

    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(sweep_dir, "*", "sweep_config.json"))):
        cfg_dir = os.path.dirname(cfg_path)
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        for run_path in sorted(glob.glob(os.path.join(cfg_dir, "run_*.json"))):
            try:
                with open(run_path) as fh:
                    r = json.load(fh)
            except Exception as exc:
                print(f"[Warning] {run_path}: {exc}")
                continue
            for k, v in list(r.items()):
                if isinstance(v, str) and k not in ("scheduler", "sim_start"):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            r["config_id"] = cfg["config_id"]
            r["fleet_key"] = cfg["fleet_key"]
            r["windows_key"] = cfg["windows_key"]
            r["horizon_h"] = cfg["windows"]["horizon_h"]
            r["n_rgb"] = cfg["fleet"]["n_rgb"]
            r["n_sar"] = cfg["fleet"]["n_sar"]
            rows.append(r)

    if not rows:
        print(f"[Rollup] No run_*.json under {sweep_dir}")
        return None

    df = pd.DataFrame(rows)
    per_run = os.path.join(sweep_dir, "sweep_per_run.csv")
    df.to_csv(per_run, index=False)

    metrics = [m for m in ("utility", "total_cost", "realized_quality",
                           "task_completion_rate", "group_completion_rate",
                           "avg_solve_time_s", "avg_mip_gap_pct")
               if m in df.columns]
    agg = (df.groupby(["config_id", "fleet_key", "windows_key", "horizon_h",
                       "n_rgb", "n_sar", "scheduler"])[metrics]
             .agg(["mean", "std", "count"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    per_cfg = os.path.join(sweep_dir, "sweep_by_config.csv")
    agg.to_csv(per_cfg, index=False)

    # The question the sweep exists to answer: where does hedging beat every baseline?
    verdicts = []
    for cid, sub in agg.groupby("config_id"):
        util = dict(zip(sub["scheduler"], sub["utility_mean"]))
        if HEDGER not in util:
            continue
        rivals = {k: v for k, v in util.items() if k in BASELINES}
        if not rivals:
            continue
        best_rival = max(rivals, key=rivals.get)
        head = sub.iloc[0]
        verdicts.append({
            "config_id": cid, "horizon_h": head["horizon_h"],
            "n_rgb": head["n_rgb"], "n_sar": head["n_sar"],
            "hedge_utility": util[HEDGER],
            "best_baseline": best_rival, "best_baseline_utility": rivals[best_rival],
            "hedge_advantage": util[HEDGER] - rivals[best_rival],
            "hedge_wins": util[HEDGER] > rivals[best_rival],
            "mip_gap_pct": sub.loc[sub["scheduler"] == HEDGER, "avg_mip_gap_pct_mean"].mean()
                           if "avg_mip_gap_pct_mean" in sub.columns else float("nan"),
        })

    vdf = pd.DataFrame(verdicts).sort_values("hedge_advantage", ascending=False)
    vpath = os.path.join(sweep_dir, "sweep_verdict.csv")
    vdf.to_csv(vpath, index=False)

    print("\n" + "=" * 78)
    print("SWEEP VERDICT -- hedging utility vs best baseline, ranked")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(vdf.to_string(index=False))
    print(f"\n[Rollup] {per_run}\n[Rollup] {per_cfg}\n[Rollup] {vpath}")
    return vdf


# ============================== ORCHESTRATOR ================================

def _run_child(cfg_path, log_path, timeout_s, stream=True):
    """Run one config in a subprocess, teeing its output to console and log.

    PYTHONUNBUFFERED is required or the child block-buffers and nothing appears
    until it exits. A reader thread pumps the pipe so the timeout still fires
    while the child is silent -- Gurobi solves for minutes without printing.
    """
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--config-file", cfg_path]
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1)

        def _pump():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if stream:
                    sys.stdout.write("  | " + line)
                    sys.stdout.flush()

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
        pump.join(timeout=10)
    return proc.returncode, timed_out


def main():
    ap = argparse.ArgumentParser(description="Fleet x time-window sweep")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config-file", help=argparse.SUPPRESS)
    ap.add_argument("--runs", type=int, default=3, help="Seeds per config (42, 43, ...).")
    ap.add_argument("--schedulers", nargs="+", default=None,
                    help="Subset of schedulers. Screening tip: stochastic_logical greedy")
    ap.add_argument("--fleets", nargs="+", default=None, help="Subset of FLEETS keys.")
    ap.add_argument("--windows", nargs="+", default=None, help="Subset of WINDOWS keys.")
    ap.add_argument("--max-solver-time-s", type=int, default=None)
    ap.add_argument("--max-num-instances", type=int, default=None)
    ap.add_argument("--min-elevation-deg", type=float, default=None,
                    help="Override MIN_ELEVATION_DEG for every config.")
    ap.add_argument("--schedule-plots", action="store_true",
                    help="Emit per-run schedule PDFs (slow, large).")
    ap.add_argument("--timeout-s", type=int, default=None, help="Per-config wall clock cap.")
    ap.add_argument("--quiet", action="store_true",
                    help="Log only; do not mirror child output to the console.")
    ap.add_argument("--sweep-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollup-only", default=None, help="Re-aggregate an existing sweep dir.")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.config_file)
        return
    if args.rollup_only:
        rollup(args.rollup_only)
        return

    fleets = {k: FLEETS[k] for k in (args.fleets or FLEETS)}
    windows = {k: WINDOWS[k] for k in (args.windows or WINDOWS)}
    if args.min_elevation_deg is not None:
        windows = {k: dict(v, min_elevation_deg=args.min_elevation_deg)
                   for k, v in windows.items()}

    grid = build_grid(fleets, windows)
    good = [g for g in grid if not g["errors"]]
    bad = [g for g in grid if g["errors"]]

    print(f"[Sweep] {len(grid)} combinations: {len(good)} valid, {len(bad)} rejected")
    for g in grid:
        w, f = g["windows"], g["fleet"]
        tag = "OK  " if not g["errors"] else "SKIP"
        print(f"  {tag} {g['config_id']:34s} horizon={w['horizon_h']:>5g}h "
              f"ext={w['extent_window_h']:g} urban={tuple(w['urban'])} "
              f"triage={tuple(w['triage'])} final={tuple(w['final'])} "
              f"fleet={f['n_rgb']}+{f['n_sar']}")
        for e in g["errors"]:
            print(f"       ! {e}")
    if args.dry_run or not good:
        return

    sweep_dir = args.sweep_dir or os.path.join(
        "results", f"sweep_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    os.makedirs(sweep_dir, exist_ok=True)

    for i, g in enumerate(good, 1):
        cfg_dir = os.path.join(sweep_dir, g["config_id"])
        os.makedirs(cfg_dir, exist_ok=True)
        cfg = dict(g, results_dir=cfg_dir, runs=args.runs,
                   schedulers=args.schedulers,
                   max_solver_time_s=args.max_solver_time_s,
                   max_num_instances=args.max_num_instances,
                   plot_schedule=args.schedule_plots)
        cfg_path = os.path.join(cfg_dir, "sweep_config.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)

        log_path = os.path.join(cfg_dir, "sweep_run.log")
        print(f"\n[{i}/{len(good)}] {g['config_id']} -> {cfg_dir}", flush=True)
        started = dt.datetime.now()
        rc, timed_out = _run_child(cfg_path, log_path, args.timeout_s,
                                   stream=not args.quiet)
        if timed_out:
            print(f"       TIMEOUT after {args.timeout_s}s (partial runs kept)")
        elif rc != 0:
            print(f"       EXIT CODE {rc} -- see {log_path}")
        print(f"       done in {(dt.datetime.now() - started).total_seconds() / 60:.1f} min"
              f"  log: {log_path}", flush=True)

    rollup(sweep_dir)


if __name__ == "__main__":
    main()
