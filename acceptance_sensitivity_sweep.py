"""
Sensitivity of hedging MILP to mis-estimated acceptance probabilities.

Runs the three case studies (volcano, earthquake, multi-sensor riverflow) with
the *planner* seeing approximate p_accept (relative error margins 0 / 5 / 10 /
20 %), while the *simulator* keeps drawing against the true demand field.

Existing drivers are left unchanged: this file monkey-patches them only inside
worker subprocesses. A zero-error cell reproduces the current case-study
behaviour (exact planner knowledge).

Earthquake uses the 56 h campaign window with a mid-size fleet (30 RGB + 20 SAR).
Volcano uses volcano_stochastic_comparison_real.py (not the deprecated
volcano_benchmarking_hpc.py).

  python acceptance_sensitivity_sweep.py --dry-run
  python acceptance_sensitivity_sweep.py --runs 2 --in-process
  python acceptance_sensitivity_sweep.py --runs 2 --case-studies earthquake --errors 0.0 0.1
  python acceptance_sensitivity_sweep.py --resume --sweep-dir results/acc_sens_...
  python acceptance_sensitivity_sweep.py --rollup-only results/acc_sens_...
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))

# Relative error margins on planner p_accept: p_hat = p_true * U(1-δ, 1+δ).
ERROR_MARGINS = (0.0, 0.05, 0.10, 0.20)

# ---- Drivers (must stay in sync with the live comparison scripts) ------------
# earthquake -> earthquake_damage_assesment.py
# riverflow  -> riverflow_coobservation.py   (driver defaults; no horizon/fleet patch)
# volcano    -> volcano_stochastic_comparison_real.py
#
# Earthquake ONLY is re-parameterized away from the driver's in-file defaults
# (those defaults are LOOKAHEAD=18 h and load_satellites_for_earthquake's
# 50 RGB + 30 SAR).  Here we use the sweep's win_56h_full + fleet_mid so the
# sensitivity study matches the 56 h campaign with a mid-size fleet.
EARTHQUAKE_FLEET = {"n_rgb": 30, "n_sar": 20}   # fleet_mid in earthquake_config_sweep
EARTHQUAKE_WINDOWS = {                          # win_56h_full
    "horizon_h": 56.0,
    "extent_window_h": 8.0,
    "urban": (1.0, 12.0),
    "triage": (6.0, 24.0),
    "final": (12.0, 48.0),
}

CASE_STUDIES = ("earthquake", "riverflow", "volcano")  # volcano last: heaviest / flakiest

HEDGER_ALIASES = {
    "stochastic_log",
    "stochastic_logical",
    "hedging_milp",
    "hedging_ilp",
}
BASELINE_ALIASES = {
    "deterministic",
    "greedy",
    "greedy_n",
    "random",
    "super_random",
}


def _error_tag(rel_error: float) -> str:
    pct = int(round(100.0 * float(rel_error)))
    return f"err{pct:02d}pct"


def build_grid(case_studies, errors):
    grid = []
    for case in case_studies:
        for err in errors:
            cid = f"{case}__{_error_tag(err)}"
            cell = {
                "config_id": cid,
                "case_study": case,
                "relative_error": float(err),
            }
            # Earthquake-only overrides (riverflow/volcano keep their driver defaults).
            if case == "earthquake":
                cell["earthquake_fleet"] = dict(EARTHQUAKE_FLEET)
                cell["earthquake_windows"] = dict(EARTHQUAKE_WINDOWS)
            grid.append(cell)
    return grid


# ============================== WORKER HOOKS ================================

def _wrap_run_one_scheduler(drv, relative_error: float):
    """Install planner estimate error for each MC seed; simulator stays exact.

    Idempotent under in-process re-entry: always wraps the original driver
    function, never a previous AccSens wrapper.
    """
    if not getattr(drv, "_acc_sens_orig_run_one", None):
        drv._acc_sens_orig_run_one = drv.run_one_scheduler
    orig = drv._acc_sens_orig_run_one

    def wrapped(scheduler, seed, *args, **kwargs):
        demand_field = None
        # Positional layouts differ slightly across drivers; demand_field is
        # always the first DemandField-like arg after locations/events.
        for a in args:
            if hasattr(a, "install_planner_estimate_error") and hasattr(
                a, "make_acceptance_prob_function"
            ):
                demand_field = a
                break
        if demand_field is None:
            demand_field = kwargs.get("demand_field")

        if demand_field is not None:
            # Distinct estimate draw per MC seed; shared across schedulers in that seed.
            est_seed = int(seed) * 100_003 + int(round(1000.0 * relative_error))
            demand_field.install_planner_estimate_error(
                relative_error=relative_error,
                estimate_seed=est_seed,
            )
            print(
                f"[AccSens] planner p_accept relative_error={relative_error:.0%} "
                f"estimate_seed={est_seed} (simulator uses truth)"
            )
        return orig(scheduler, seed, *args, **kwargs)

    drv.run_one_scheduler = wrapped
    return orig


def _apply_earthquake_56h(drv_mod, fleet, windows, schedulers, max_solver_time_s, max_num_instances):
    """Reuse the same fleet/window patching as earthquake_config_sweep."""
    from earthquake_config_sweep import _apply_config

    cfg = {
        "config_id": "fleet_mid__win_56h_full",
        "fleet_key": "fleet_mid",
        "windows_key": "win_56h_full",
        "fleet": dict(fleet),
        "windows": dict(windows),
        "schedulers": schedulers,
        "max_solver_time_s": max_solver_time_s,
        "max_num_instances": max_num_instances,
    }
    return _apply_config(cfg)


def run_worker(config_path: str):
    os.chdir(REPO)
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    case = cfg["case_study"]
    rel_error = float(cfg["relative_error"])
    results_dir = cfg["results_dir"]
    runs = int(cfg["runs"])
    plot_schedule = bool(cfg.get("plot_schedule", False))
    schedulers = cfg.get("schedulers")

    print(
        f"[AccSens] case={case} relative_error={rel_error:.0%} "
        f"runs={runs} dir={results_dir}"
    )

    if case == "earthquake":
        import earthquake_damage_assesment as drv
        fleet = cfg.get("earthquake_fleet") or EARTHQUAKE_FLEET
        windows = cfg.get("earthquake_windows") or EARTHQUAKE_WINDOWS
        _apply_earthquake_56h(
            drv,
            fleet=fleet,
            windows=windows,
            schedulers=schedulers,
            max_solver_time_s=cfg.get("max_solver_time_s"),
            max_num_instances=cfg.get("max_num_instances"),
        )
        print(
            f"[AccSens] driver=earthquake_damage_assesment.py  "
            f"horizon={windows['horizon_h']:g}h  "
            f"fleet={fleet['n_rgb']}RGB+{fleet['n_sar']}SAR  "
            f"schedulers={schedulers or drv.SCHEDULERS}  "
            f"MAX_NUM_INSTANCES={drv.MAX_NUM_INSTANCES}  "
            f"MAX_SOLVER_TIME_S={drv.MAX_SOLVER_TIME_S}"
        )
        _wrap_run_one_scheduler(drv, rel_error)
        drv.run_comparison(
            num_monte_carlo_runs=runs,
            results_dir=results_dir,
            plot_schedule=plot_schedule,
        )
    elif case == "volcano":
        # Current comparison driver (volcano_benchmarking_hpc.py is deprecated).
        import volcano_stochastic_comparison_real as drv
        if schedulers:
            drv.SCHEDULERS = list(schedulers)
        if cfg.get("max_solver_time_s"):
            drv.MAX_SOLVER_TIME_S = int(cfg["max_solver_time_s"])
        if cfg.get("max_num_instances"):
            drv.MAX_NUM_INSTANCES = int(cfg["max_num_instances"])
        print(
            f"[AccSens] driver=volcano_stochastic_comparison_real.py  "
            f"(driver defaults)  schedulers={schedulers or drv.SCHEDULERS}  "
            f"MAX_NUM_INSTANCES={drv.MAX_NUM_INSTANCES}  "
            f"MAX_SOLVER_TIME_S={drv.MAX_SOLVER_TIME_S}"
        )
        _wrap_run_one_scheduler(drv, rel_error)
        drv.run_comparison(
            num_monte_carlo_runs=runs,
            results_dir=results_dir,
            schedulers=schedulers,
            plot_schedule=plot_schedule,
        )
    elif case == "riverflow":
        import riverflow_coobservation as drv
        if schedulers:
            drv.SCHEDULERS = list(schedulers)
        if cfg.get("max_solver_time_s"):
            drv.MAX_SOLVER_TIME_S = int(cfg["max_solver_time_s"])
        if cfg.get("max_num_instances") and hasattr(drv, "MAX_NUM_INSTANCES"):
            drv.MAX_NUM_INSTANCES = int(cfg["max_num_instances"])
        print(
            f"[AccSens] driver=riverflow_coobservation.py  "
            f"(driver defaults: horizon={drv.LOOKAHEAD_HORIZON_H:g}h, "
            f"lead={drv.DISPATCH_LEAD_H:g}h)  "
            f"schedulers={schedulers or drv.SCHEDULERS}  "
            f"MAX_NUM_INSTANCES={drv.MAX_NUM_INSTANCES}  "
            f"MAX_SOLVER_TIME_S={drv.MAX_SOLVER_TIME_S}"
        )
        _wrap_run_one_scheduler(drv, rel_error)
        drv.run_comparison(
            num_monte_carlo_runs=runs,
            results_dir=results_dir,
            plot_schedule=plot_schedule,
        )
    else:
        raise SystemExit(f"Unknown case_study: {case}")

    # Stamp every run_*.json with sensitivity metadata (non-destructive add).
    meta = {
        "case_study": case,
        "relative_error": rel_error,
        "planner_p_accept_mode": (
            "exact" if rel_error <= 0.0 else f"relative_error_{rel_error:.0%}"
        ),
    }
    for run_path in glob.glob(os.path.join(results_dir, "run_*.json")):
        try:
            with open(run_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            rec.update(meta)
            with open(run_path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, default=str)
        except Exception as exc:
            print(f"[Warning] could not stamp {run_path}: {exc}")


# ============================== ROLL-UP =====================================

def _is_hedger(name: str) -> bool:
    return str(name).lower() in HEDGER_ALIASES


def _is_baseline(name: str) -> bool:
    return str(name).lower() in BASELINE_ALIASES


def rollup(sweep_dir: str):
    import pandas as pd

    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(sweep_dir, "*", "sweep_config.json"))):
        cfg_dir = os.path.dirname(cfg_path)
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        for run_path in sorted(glob.glob(os.path.join(cfg_dir, "run_*.json"))):
            try:
                with open(run_path, encoding="utf-8") as fh:
                    r = json.load(fh)
            except Exception as exc:
                print(f"[Warning] {run_path}: {exc}")
                continue
            for k, v in list(r.items()):
                if isinstance(v, str) and k not in (
                    "scheduler", "sim_start", "case_study", "planner_p_accept_mode",
                ):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            r["config_id"] = cfg["config_id"]
            r["case_study"] = cfg["case_study"]
            r["relative_error"] = float(cfg["relative_error"])
            rows.append(r)

    if not rows:
        print(f"[Rollup] No run_*.json under {sweep_dir}")
        return None

    df = pd.DataFrame(rows)
    per_run = os.path.join(sweep_dir, "acc_sens_per_run.csv")
    df.to_csv(per_run, index=False)

    metrics = [
        m for m in (
            "utility", "total_cost", "realized_quality",
            "task_completion_rate", "group_completion_rate",
            "avg_solve_time_s", "avg_mip_gap_pct",
        )
        if m in df.columns
    ]
    agg = (
        df.groupby(["config_id", "case_study", "relative_error", "scheduler"])[metrics]
        .agg(["mean", "std", "count"])
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    per_cfg = os.path.join(sweep_dir, "acc_sens_by_config.csv")
    agg.to_csv(per_cfg, index=False)

    # Hedging advantage vs best baseline, and degradation vs exact (0% error).
    verdicts = []
    for (case, err), sub in agg.groupby(["case_study", "relative_error"]):
        util = dict(zip(sub["scheduler"], sub["utility_mean"]))
        hedgers = {k: v for k, v in util.items() if _is_hedger(k)}
        baselines = {k: v for k, v in util.items() if _is_baseline(k)}
        if not hedgers or not baselines:
            continue
        hedge_name = max(hedgers, key=hedgers.get)
        hedge_u = hedgers[hedge_name]
        best_base = max(baselines, key=baselines.get)
        verdicts.append({
            "case_study": case,
            "relative_error": err,
            "hedger": hedge_name,
            "hedge_utility": hedge_u,
            "best_baseline": best_base,
            "best_baseline_utility": baselines[best_base],
            "hedge_advantage": hedge_u - baselines[best_base],
            "hedge_wins": hedge_u > baselines[best_base],
        })

    vdf = pd.DataFrame(verdicts)
    if not vdf.empty:
        # Degradation relative to exact-knowledge cell within each case study.
        deg_rows = []
        for case, sub in vdf.groupby("case_study"):
            exact = sub.loc[sub["relative_error"] <= 1e-12]
            if exact.empty:
                continue
            u0 = float(exact["hedge_utility"].iloc[0])
            adv0 = float(exact["hedge_advantage"].iloc[0])
            for _, row in sub.iterrows():
                deg_rows.append({
                    **row.to_dict(),
                    "hedge_utility_vs_exact": row["hedge_utility"] - u0,
                    "hedge_advantage_vs_exact": row["hedge_advantage"] - adv0,
                })
        vdf = pd.DataFrame(deg_rows).sort_values(
            ["case_study", "relative_error"]
        )

    vpath = os.path.join(sweep_dir, "acc_sens_verdict.csv")
    vdf.to_csv(vpath, index=False)

    print("\n" + "=" * 78)
    print("ACCEPTANCE SENSITIVITY — hedging utility vs best baseline by error margin")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(vdf.to_string(index=False) if not vdf.empty else "(no verdict rows)")
    print(f"\n[Rollup] {per_run}\n[Rollup] {per_cfg}\n[Rollup] {vpath}")
    return vdf


# ============================== ORCHESTRATOR ================================

def _is_windowsapps_python(path: str) -> bool:
    return "windowsapps" in os.path.normpath(path).lower()


def _worker_python() -> str:
    """Interpreter used to spawn workers.

    Windows Store / WindowsApps Python venvs ship a redirector stub that often
    fails with: Unable to create process ... The file cannot be accessed by
    the system. Prefer a non-Store base executable when available.
    """
    candidates = []
    base = getattr(sys, "_base_executable", None)
    if base:
        candidates.append(base)
    candidates.append(sys.executable)
    # Real venv python next to the redirector (some layouts ship both).
    prefix_scripts = os.path.join(sys.prefix, "Scripts", "python.exe")
    candidates.append(prefix_scripts)

    seen = set()
    for c in candidates:
        if not c:
            continue
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c) and not _is_windowsapps_python(c):
            return c
    # Last resort: whatever we are running under.
    return os.path.abspath(sys.executable)


def _count_run_json(cfg_dir: str) -> int:
    return len(glob.glob(os.path.join(cfg_dir, "run_*.json")))


def _cell_is_done(cfg_dir: str, runs: int, schedulers) -> bool:
    """Heuristic: enough run_*.json files to cover runs × schedulers (or ≥1)."""
    n = _count_run_json(cfg_dir)
    if n <= 0:
        return False
    if schedulers:
        need = max(1, int(runs) * len(schedulers))
    else:
        # Unknown scheduler count — require at least one file per seed.
        need = max(1, int(runs))
    return n >= need


def _run_child(cfg_path, log_path, timeout_s, stream=True, retries=3):
    """Spawn a worker; retry on Windows Store CreateProcess flakiness."""
    python_exe = _worker_python()
    cmd = [
        python_exe, os.path.abspath(__file__),
        "--worker", "--config-file", os.path.abspath(cfg_path),
    ]
    last_rc, timed_out = 1, False

    for attempt in range(1, retries + 1):
        timed_out = False
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"[AccSens] spawn attempt {attempt}/{retries}: {cmd}\n")
                log.flush()
                proc = subprocess.Popen(
                    cmd, cwd=REPO,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    bufsize=1,
                )

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
            last_rc = proc.returncode if proc.returncode is not None else 1
        except OSError as exc:
            last_rc = 101
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"[AccSens] spawn OSError: {exc}\n")
            if stream:
                print(f"  | [AccSens] spawn OSError: {exc}")

        # Windows Store stub often surfaces as rc=101 with empty/near-empty log.
        spawn_flaky = (last_rc in (101, 103)) or timed_out is False and last_rc != 0 and (
            _count_run_json(os.path.dirname(cfg_path)) == 0
            and attempt < retries
            and os.path.getsize(log_path) < 500
        )
        if last_rc == 0 or timed_out or not spawn_flaky:
            break
        print(f"  [AccSens] spawn failed (rc={last_rc}); retry {attempt}/{retries} in 3s...")
        time.sleep(3.0)

    return last_rc, timed_out


def _run_in_process(cfg_path, log_path, stream=True):
    """Fallback when CreateProcess cannot launch the WindowsApps venv stub."""
    import contextlib
    import io
    import traceback

    buf = io.StringIO()
    rc = 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            run_worker(cfg_path)
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception:
        rc = 1
        buf.write(traceback.format_exc())

    text = buf.getvalue()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("[AccSens] in-process worker\n")
        log.write(text)
    if stream:
        for line in text.splitlines(True):
            sys.stdout.write("  | " + line)
        sys.stdout.flush()
    return rc, False


def main():
    ap = argparse.ArgumentParser(
        description="Planner p_accept estimate-error sensitivity across case studies",
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config-file", help=argparse.SUPPRESS)
    ap.add_argument("--runs", type=int, default=2,
                    help="Monte Carlo seeds per cell (42, 43, ...).")
    ap.add_argument("--case-studies", nargs="+", default=None,
                    choices=list(CASE_STUDIES),
                    help="Subset of case studies (default: all three).")
    ap.add_argument("--errors", nargs="+", type=float, default=None,
                    help="Relative error margins, e.g. 0.0 0.05 0.1 0.2")
    ap.add_argument("--schedulers", nargs="+", default=None,
                    help="Optional scheduler subset passed through to drivers.")
    ap.add_argument("--max-solver-time-s", type=int, default=None)
    ap.add_argument("--max-num-instances", type=int, default=None)
    ap.add_argument("--schedule-plots", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=None,
                    help="Per-cell wall-clock cap (seconds).")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sweep-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollup-only", default=None,
                    help="Re-aggregate an existing sensitivity sweep dir.")
    ap.add_argument("--in-process", action="store_true",
                    help="Run each cell in this process (avoids Windows Store "
                         "venv CreateProcess failures). Uses more RAM.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells that already have run_*.json under --sweep-dir.")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.config_file)
        return
    if args.rollup_only:
        rollup(args.rollup_only)
        return

    cases = tuple(args.case_studies or CASE_STUDIES)
    errors = tuple(args.errors if args.errors is not None else ERROR_MARGINS)
    grid = build_grid(cases, errors)

    worker_py = _worker_python()
    print(f"[AccSens] {len(grid)} cells "
          f"({len(cases)} case studies × {len(errors)} error margins)")
    print(f"[AccSens] worker python: {worker_py}")
    if _is_windowsapps_python(sys.executable) or _is_windowsapps_python(worker_py):
        print(
            "[AccSens] WARNING: Windows Store Python detected. Subprocess spawns "
            "often fail with 'The file cannot be accessed by the system'. "
            "Prefer --in-process, or recreate the venv with a non-Store Python."
        )
    for g in grid:
        extra = ""
        if g["case_study"] == "earthquake":
            f = g.get("earthquake_fleet") or EARTHQUAKE_FLEET
            w = g.get("earthquake_windows") or EARTHQUAKE_WINDOWS
            extra = (f"  fleet={f['n_rgb']}RGB+{f['n_sar']}SAR "
                     f"horizon={w['horizon_h']:g}h")
        elif g["case_study"] == "riverflow":
            extra = "  (riverflow_coobservation defaults)"
        elif g["case_study"] == "volcano":
            extra = "  (volcano_stochastic_comparison_real defaults)"
        print(f"  {g['config_id']:28s} err={g['relative_error']:.0%}{extra}")
    if args.dry_run:
        return

    sweep_dir = args.sweep_dir or os.path.join(
        "results",
        f"acc_sens_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}",
    )
    os.makedirs(sweep_dir, exist_ok=True)

    for i, g in enumerate(grid, 1):
        cfg_dir = os.path.join(sweep_dir, g["config_id"])
        os.makedirs(cfg_dir, exist_ok=True)

        if args.resume and _cell_is_done(cfg_dir, args.runs, args.schedulers):
            print(f"\n[{i}/{len(grid)}] {g['config_id']} SKIP (resume, "
                  f"{_count_run_json(cfg_dir)} run_*.json)", flush=True)
            continue

        cfg = dict(
            g,
            results_dir=cfg_dir,
            runs=args.runs,
            schedulers=args.schedulers,
            max_solver_time_s=args.max_solver_time_s,
            max_num_instances=args.max_num_instances,
            plot_schedule=args.schedule_plots,
        )
        cfg_path = os.path.join(cfg_dir, "sweep_config.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        log_path = os.path.join(cfg_dir, "sweep_run.log")
        print(f"\n[{i}/{len(grid)}] {g['config_id']} -> {cfg_dir}", flush=True)
        started = dt.datetime.now()

        if args.in_process:
            rc, timed_out = _run_in_process(
                cfg_path, log_path, stream=not args.quiet,
            )
        else:
            rc, timed_out = _run_child(
                cfg_path, log_path, args.timeout_s, stream=not args.quiet,
            )
            # Automatic fallback when the WindowsApps stub cannot spawn.
            if rc in (101, 103) or (
                rc != 0 and _count_run_json(cfg_dir) == 0
                and os.path.exists(log_path)
                and os.path.getsize(log_path) < 800
            ):
                print("  [AccSens] spawn unusable — falling back to in-process")
                rc, timed_out = _run_in_process(
                    cfg_path, log_path, stream=not args.quiet,
                )

        elapsed = (dt.datetime.now() - started).total_seconds()
        n_runs = _count_run_json(cfg_dir)
        if timed_out:
            print(f"  TIMEOUT after {elapsed:.0f}s")
        elif rc != 0:
            print(f"  FAILED rc={rc} after {elapsed:.0f}s  (see {log_path})")
        elif n_runs == 0:
            print(
                f"  FAILED (exit 0 but no run_*.json) after {elapsed:.0f}s  "
                f"(see {log_path})"
            )
        else:
            print(f"  OK in {elapsed:.0f}s ({n_runs} run_*.json)")

        # Give Windows a beat to release the Store redirector / Gurobi handles.
        time.sleep(1.0)

    rollup(sweep_dir)


if __name__ == "__main__":
    main()
