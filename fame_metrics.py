"""
Metrics for stochastic vs. deterministic scheduler comparison in FAME.

Design Principle:
- Measures REALIZED outcomes from broker._requests (actual simulation execution).
- Reachability Denominator: Reachable tasks are root tasks OR children whose parents succeeded.
- Causal Validation: Quality and completions are only credited if parent dependencies were met.
- True Redundancy: Tracks executed passes per completed task and submitted passes per request.
"""

import numpy as np


def _status_names(ObservationStatus):
    """Resolve status enum members tolerantly across FAME versions."""
    return {
        'rejected': getattr(ObservationStatus, 'CONSTELLATION_REJECTED', None) or getattr(ObservationStatus, 'REJECTED', None),
        'scheduled': ObservationStatus.SCHEDULED,
        'received': ObservationStatus.DATA_RECEIVED,
        'cancelled': getattr(ObservationStatus, 'CANCELLED', None),
        'execution_failed': getattr(ObservationStatus, 'EXECUTION_FAILED', None),
    }


def compute_metrics_v3(workflow_graph, broker, ObservationStatus,
                       submission_cost_rate, execution_cost_fn=None,
                       execution_cost_rate=0.0,
                       verbose=True,
                       sim_end_time=None):
    """
    Realized, causally-valid, reachability-adjusted metrics.

    sim_end_time: datetime at which the simulation stopped.  When provided,
    tasks whose every submitted pass has a fall time strictly after sim_end_time
    (i.e., the simulation ended before any of their passes could have resolved)
    are excluded from both numerator and denominator.  Pass broker.world.time
    (or equivalent) to activate this filter.
    """
    S = _status_names(ObservationStatus)
    reqs = broker._requests

    tasks = list(workflow_graph.nodes())
    obsreq_to_task = {t.observation_request: t for t in tasks}

    def group_of(task):
        return getattr(task, 'request_group', None) or getattr(task, 'name', str(id(task)))

    all_groups = set(group_of(t) for t in tasks)
    n_tasks_total = len(tasks)
    n_groups_total = len(all_groups)

    # Raw counts
    n_submissions = 0
    n_accepted = 0
    n_executed = 0
    n_rejected = 0
    n_cancelled = 0
    n_execution_failed = 0

    total_submission_cost = 0.0
    total_execution_cost = 0.0

    best_success_quality_by_task = {}     # task -> float
    executed_count_by_task = {}           # task -> int
    submitted_count_by_task = {}          # task -> int

    _accepted_statuses = {S['scheduled'], S['received']}
    if S['execution_failed'] is not None:
        _accepted_statuses.add(S['execution_failed'])

    # Precompute Q_MAX per task: max quality over all submitted passes.
    # Used for cost billing so costs don't depend on which specific pass was booked.
    q_max_by_task = {}
    for _, row in reqs.iterrows():
        rp = row['requested_pass']
        if rp is None:
            continue
        task = obsreq_to_task.get(row['request'])
        if task is None:
            continue
        try:
            q = task.rewarder(rp.highest)
        except Exception:
            q = 0.0
        if q > q_max_by_task.get(task, -np.inf):
            q_max_by_task[task] = q

    has_dispatch_time = 'dispatch_time' in reqs.columns

    for _, row in reqs.iterrows():
        rp = row['requested_pass']
        if rp is None:
            continue
        task = obsreq_to_task.get(row['request'])
        if task is None:
            continue
        status = row['status']

        try:
            q = task.rewarder(rp.highest)
        except Exception:
            q = 0.0

        q_max = q_max_by_task.get(task, q)
        _raw_dt = row['dispatch_time'] if has_dispatch_time else None
        # Convert pandas NaT to None so cost functions can check `is not None` safely
        try:
            dispatch_time = None if (_raw_dt != _raw_dt) else _raw_dt
        except Exception:
            dispatch_time = _raw_dt

        n_submissions += 1
        submitted_count_by_task[task] = submitted_count_by_task.get(task, 0) + 1
        total_submission_cost += submission_cost_rate * q_max

        if status == S['rejected']:
            n_rejected += 1
        if S['cancelled'] is not None and status == S['cancelled']:
            n_cancelled += 1
        if S['execution_failed'] is not None and status == S['execution_failed']:
            n_execution_failed += 1
            sat = row['satellite']
            if execution_cost_fn is not None:
                try:
                    total_execution_cost += execution_cost_fn(task, sat, rp, dispatch_time, q_max=q_max)
                except Exception:
                    total_execution_cost += execution_cost_rate * q_max
            else:
                total_execution_cost += execution_cost_rate * q_max
        if status in _accepted_statuses:
            n_accepted += 1
        if status == S['received']:
            n_executed += 1
            executed_count_by_task[task] = executed_count_by_task.get(task, 0) + 1
            sat = row['satellite']
            if execution_cost_fn is not None:
                try:
                    total_execution_cost += execution_cost_fn(task, sat, rp, dispatch_time, q_max=q_max)
                except Exception:
                    total_execution_cost += execution_cost_rate * q_max
            else:
                total_execution_cost += execution_cost_rate * q_max

            # DATA_RECEIVED means the satellite executed; credit quality regardless of
            # whether the data product is non-empty (spatial misses are treated as
            # successful executions — quality is geometry-based, not detection-based).
            prev = best_success_quality_by_task.get(task, -np.inf)
            if q > prev:
                best_success_quality_by_task[task] = q

    total_cost = total_submission_cost + total_execution_cost
    completed_tasks = set(best_success_quality_by_task.keys())

    # ---------------------------------------------------------------------------
    # 1. Reachability & Causal Dependency Analysis
    # ---------------------------------------------------------------------------
    task_success_map = {
        t: (t in completed_tasks and getattr(t, 'successful_execution', True))
        for t in tasks
    }

    # Tasks that received at least one pass candidate from the planner.
    tasks_with_passes = set(submitted_count_by_task.keys())

    # Tasks cut off by the simulation ending before any of their passes could
    # resolve.  A task is "sim-truncated" if it was dispatched (has at least one
    # SUBMITTED or SCHEDULED row) and ALL of those passes have fall times after
    # sim_end_time — meaning the simulation stopped before any accept/reject
    # notification or data callback could fire.
    sim_truncated_tasks: set = set()
    if sim_end_time is not None:
        _in_flight_statuses = {ObservationStatus.SUBMITTED, ObservationStatus.SCHEDULED}
        for task in tasks_with_passes:
            task_rows = reqs[reqs['request'] == task.observation_request]
            in_flight = task_rows[task_rows['status'].isin(_in_flight_statuses)]
            if in_flight.empty:
                continue
            # If the task already completed, it's not truncated
            if task in completed_tasks:
                continue
            # Check whether every in-flight pass falls after sim_end_time
            all_after = True
            for _, row in in_flight.iterrows():
                rp = row['requested_pass']
                if rp is None:
                    all_after = False
                    break
                try:
                    fall_t = rp.fall.time
                except Exception:
                    all_after = False
                    break
                if fall_t <= sim_end_time:
                    all_after = False
                    break
            if all_after:
                sim_truncated_tasks.add(task)

    def _timeline_was_active(task):
        """Returns False if any GREATER_OR_EQUAL timeline constraint was never satisfiable.

        Uses the task's observation window midpoint as the evaluation time; falls
        back to the constraint's own threshold value to determine whether the
        timeline was below the required minimum throughout the window.
        """
        tl_constraints = getattr(task, 'timeline_constraints', [])
        if not tl_constraints:
            return True
        obs_req = getattr(task, 'observation_request', None)
        if obs_req is not None:
            min_t = getattr(obs_req, 'min_time', None)
            max_t = getattr(obs_req, 'max_time', None)
            if min_t is not None and max_t is not None:
                eval_time = min_t + (max_t - min_t) / 2
            elif min_t is not None:
                eval_time = min_t
            else:
                eval_time = None
        else:
            eval_time = None

        for tc in tl_constraints:
            tl = getattr(tc, 'timeline', None)
            threshold = getattr(tc, 'value', None)
            tc_type = getattr(tc, 'type', None)
            if tl is None or threshold is None:
                continue
            # Only evaluate GREATER_OR_EQUAL constraints (the kind used for activity checks)
            if tc_type is not None and 'GREATER_OR_EQUAL' not in str(tc_type):
                continue
            if eval_time is None:
                continue
            try:
                tl_val = tl.get_value_at(eval_time)
            except Exception:
                continue
            if tl_val < threshold:
                return False
        return True

    def is_task_feasible(task):
        """A task is feasible if it had passes, timelines were active, and sim didn't cut it off."""
        if task not in tasks_with_passes:
            return False
        if not _timeline_was_active(task):
            return False
        if task in sim_truncated_tasks:
            return False
        return True

    # Build a reachability set: a task is reachable if it is feasible and all
    # START_IF_SUCCESSFUL parent dependencies were themselves reachable AND succeeded.
    # We evaluate in topological order so parent reachability is known first.
    try:
        import networkx as nx
        topo_order = list(nx.topological_sort(workflow_graph))
    except Exception:
        topo_order = tasks  # fallback if graph is not a DAG or nx unavailable

    task_reachable: dict = {}
    for task in topo_order:
        if not is_task_feasible(task):
            task_reachable[task] = False
            continue
        parents = list(workflow_graph.predecessors(task))
        reachable = True
        for p in parents:
            edge_data = workflow_graph.get_edge_data(p, task)
            for _, constraint in edge_data.items():
                c_class = constraint.get('constraint_class')
                c_type = constraint.get('constraint_type')
                if 'SUCCESS' in str(c_class) and 'START_IF_SUCCESSFUL' in str(c_type):
                    # Parent must have been reachable and succeeded
                    if not (task_reachable.get(p, False) and task_success_map.get(p, False)):
                        reachable = False
                        break
            if not reachable:
                break
        task_reachable[task] = reachable

    reachable_tasks = {t for t, ok in task_reachable.items() if ok}
    n_tasks_reachable = len(reachable_tasks)

    # Classify all tasks for diagnostics
    n_no_passes = sum(1 for t in tasks if t not in tasks_with_passes)
    n_inactive_timeline = sum(1 for t in tasks if t in tasks_with_passes and not _timeline_was_active(t))
    n_sim_truncated = len(sim_truncated_tasks)
    n_parent_not_met = sum(1 for t in tasks
                           if t in tasks_with_passes and _timeline_was_active(t)
                           and t not in sim_truncated_tasks
                           and not task_reachable.get(t, False))

    # Valid completed tasks must be reachable
    valid_completed_tasks = completed_tasks.intersection(reachable_tasks)
    n_tasks_completed_valid = len(valid_completed_tasks)

    # ---------------------------------------------------------------------------
    # 2. Metric Calculations
    # ---------------------------------------------------------------------------
    # Reachable Task Completion Rate (Fair Denominator)
    reachable_task_completion_rate = (
        n_tasks_completed_valid / n_tasks_reachable if n_tasks_reachable > 0 else 0.0
    )

    # Raw Completion Rate (Legacy reference)
    raw_task_completion_rate = len(completed_tasks) / n_tasks_total if n_tasks_total > 0 else 0.0

    # Group Completion
    completed_groups = set(group_of(t) for t in valid_completed_tasks)
    n_groups_completed = len(completed_groups)
    group_completion_rate = n_groups_completed / n_groups_total if n_groups_total > 0 else 0.0

    # Realized Quality (Credited ONLY for causally valid completions)
    realized_quality = float(sum(
        best_success_quality_by_task[t] for t in valid_completed_tasks
    )) if valid_completed_tasks else 0.0

    utility = realized_quality - total_cost

    # Redundancy
    if valid_completed_tasks:
        exec_passes_per_completed = np.mean([
            executed_count_by_task.get(t, 0) for t in valid_completed_tasks
        ])
    else:
        exec_passes_per_completed = 0.0

    tasks_with_submissions = list(submitted_count_by_task.keys())
    submitted_passes_per_task = (
        np.mean([submitted_count_by_task[t] for t in tasks_with_submissions])
        if tasks_with_submissions else 0.0
    )

    acceptance_rate = (n_accepted / n_submissions) if n_submissions > 0 else 0.0
    rejection_rate = (n_rejected / n_submissions) if n_submissions > 0 else 0.0
    completions_per_cost = (n_tasks_completed_valid / total_cost) if total_cost > 0 else float('nan')

    # ---------------------------------------------------------------------------
    # 3. Per-execution detail list
    # ---------------------------------------------------------------------------
    # Every DATA_RECEIVED pass, including those this function's AND-cascade
    # judged unreachable.  The dispatcher already enforced SUCCESS (and OR via
    # success_constraint_mode='any'); dropping those rows made the export a
    # strict subset of what actually flew.  is_valid still records the AND
    # judgement for diagnostics.  Post-processing scores the run summary, not
    # this list.
    execution_details = []
    for _, row in reqs[reqs['status'] == S['received']].iterrows():
        task = obsreq_to_task.get(row['request'])
        if task is None:
            continue
        rp = row['requested_pass']
        sat = row['satellite']
        _raw_dt = row['dispatch_time'] if has_dispatch_time else None
        try:
            _dt_valid = _raw_dt is not None and _raw_dt == _raw_dt
        except Exception:
            _dt_valid = False
        _dispatch_time = _raw_dt if _dt_valid else None
        q = q_max_by_task.get(task, 0.0)  # use precomputed Q_MAX for this task
        try:
            _pass_q = task.rewarder(rp.highest) if rp else 0.0
        except Exception:
            _pass_q = 0.0
        if execution_cost_fn is not None and rp is not None and sat is not None:
            try:
                _exec_cost = execution_cost_fn(task, sat, rp, _dispatch_time, q_max=q)
            except Exception:
                _exec_cost = 0.0
        else:
            _exec_cost = execution_cost_rate * q
        execution_details.append({
            'task': getattr(task, 'name', str(id(task))),
            'group': group_of(task),
            'satellite': sat.name if sat is not None else None,
            'pass_time': str(rp.highest.time) if rp else None,
            'quality': round(_pass_q, 4),
            'exec_cost': round(_exec_cost, 4),
            'is_valid': task in valid_completed_tasks,
        })
    execution_details.sort(key=lambda x: (x['group'], x['task']))

    # ---------------------------------------------------------------------------
    # 4. Planning session stats (accumulated by Broker across all replan calls)
    # ---------------------------------------------------------------------------
    import math as _math
    _sessions = getattr(broker, '_planning_sessions', [])
    def _is_finite(v):
        try:
            return v is not None and not _math.isnan(v)
        except (TypeError, ValueError):
            return False
    _solve_times = [s['solve_time_s'] for s in _sessions if _is_finite(s.get('solve_time_s'))]
    _wall_times  = [s['wall_s']       for s in _sessions if _is_finite(s.get('wall_s'))]
    _mip_gaps    = [s['mip_gap']      for s in _sessions if _is_finite(s.get('mip_gap'))]

    m = {
        # ---- PRIMARY: Reachable & Demand Metrics ----
        'task_completion_rate': reachable_task_completion_rate,  # Main metric uses reachable denominator
        'reachable_task_completion_rate': reachable_task_completion_rate,
        'group_completion_rate': group_completion_rate,
        'n_tasks_completed_valid': n_tasks_completed_valid,
        'n_tasks_reachable': n_tasks_reachable,
        'n_tasks_total': n_tasks_total,
        'n_groups_completed': n_groups_completed,
        'n_groups_total': n_groups_total,
        'realized_quality': realized_quality,
        'utility': utility,

        # ---- COST AXIS ----
        'total_cost': total_cost,
        'submission_cost': total_submission_cost,
        'execution_cost': total_execution_cost,
        'n_submissions': n_submissions,
        'n_accepted': n_accepted,
        'n_executed': n_executed,
        'n_rejected': n_rejected,
        'n_cancelled': n_cancelled,
        'n_execution_failed': n_execution_failed,

        # ---- REDUNDANCY & EFFICIENCY ----
        'submitted_passes_per_task': submitted_passes_per_task,
        'exec_passes_per_completed': exec_passes_per_completed,
        'replans': getattr(broker, '_n_replans', 0),
        'completions_per_cost': completions_per_cost,
        'acceptance_rate': acceptance_rate,
        'rejection_rate': rejection_rate,
        'raw_task_completion_rate': raw_task_completion_rate,

        # ---- FEASIBILITY BREAKDOWN (denominator diagnostics) ----
        'n_tasks_no_passes': n_no_passes,                  # orbital gap — excluded
        'n_tasks_inactive_timeline': n_inactive_timeline,  # timeline below threshold — excluded
        'n_tasks_sim_truncated': n_sim_truncated,          # sim ended before pass resolved — excluded
        'n_tasks_parent_not_met': n_parent_not_met,        # parent dependency failed — excluded

        # ---- PLANNING SESSION DIAGNOSTICS ----
        'n_planning_sessions': len(_sessions),
        'avg_solve_time_s': float(np.mean(_solve_times)) if _solve_times else float('nan'),
        'avg_wall_time_s': float(np.mean(_wall_times)) if _wall_times else float('nan'),
        'avg_mip_gap_pct': float(np.mean(_mip_gaps) * 100) if _mip_gaps else float('nan'),
        'final_mip_gap_pct': float(_sessions[-1].get('mip_gap', float('nan')) * 100) if _sessions else float('nan'),

        # ---- PER-EXECUTION DETAIL (popped before saving run_*.json) ----
        'execution_details': execution_details,
    }

    if verbose:
        _trunc_str = f", sim-truncated={n_sim_truncated}" if n_sim_truncated else ""
        print(f"   [Metrics] Tasks total={n_tasks_total}: reachable={n_tasks_reachable} "
              f"(no-passes={n_no_passes}, inactive-tl={n_inactive_timeline}"
              f"{_trunc_str}, parent-not-met={n_parent_not_met})")
        print(f"   [Metrics] Reachable Completion: tasks {n_tasks_completed_valid}/{n_tasks_reachable} "
              f"({100*reachable_task_completion_rate:.1f}%), groups {n_groups_completed}/{n_groups_total} "
              f"({100*group_completion_rate:.1f}%)")
        print(f"   [Metrics] Realized quality {realized_quality:.1f}, cost {total_cost:.1f} "
              f"(sub {total_submission_cost:.1f} + exec {total_execution_cost:.1f}), "
              f"utility {utility:.1f}")
        print(f"   [Metrics] Bookings: {n_submissions} submitted, {n_accepted} accepted, "
              f"{n_executed} executed ok, {n_execution_failed} exec-failed, "
              f"{n_rejected} rejected ({100*rejection_rate:.0f}% rej), "
              f"{n_cancelled} cancelled, {getattr(broker, '_n_replans', 0)} replans")
        print(f"   [Metrics] TRUE passes/task: {submitted_passes_per_task:.2f} submitted, "
              f"{exec_passes_per_completed:.2f} executed among completed")
        if _sessions:
            _gap_str = (f", avg MIP gap {np.mean(_mip_gaps)*100:.1f}%" if _mip_gaps else "")
            _solve_str = (f"avg solve {np.mean(_solve_times):.1f}s" if _solve_times else "")
            print(f"   [Metrics] Planning: {len(_sessions)} sessions, {_solve_str}{_gap_str}")

    return m


# Alias for backward compatibility with existing imports
compute_metrics = compute_metrics_v3


# ---------------------------------------------------------------------------
# Paired analysis across seeds + cost-frontier plotting
# ---------------------------------------------------------------------------

def paired_summary(records, schedulers, primary='task_completion_rate',
                   baseline='deterministic'):
    """
    Prints paired mean differences vs `baseline` with a bootstrap 95% CI.
    """
    import collections
    by = collections.defaultdict(dict)   # seed -> scheduler -> metrics
    for r in records:
        by[r['seed']][r['scheduler']] = r

    seeds = sorted(s for s in by if baseline in by[s])
    print(f"\n=== Paired analysis (n={len(seeds)} seeds, baseline={baseline}) ===")

    def boot_ci(diffs, iters=10000):
        diffs = np.asarray(diffs, float)
        if len(diffs) < 2:
            return (float('nan'), float('nan'))
        idx = np.random.randint(0, len(diffs), size=(iters, len(diffs)))
        means = diffs[idx].mean(axis=1)
        return (np.percentile(means, 2.5), np.percentile(means, 97.5))

    for sched in schedulers:
        if sched == baseline:
            continue
        for metric in (primary, 'utility', 'total_cost', 'group_completion_rate'):
            diffs = [by[s][sched][metric] - by[s][baseline][metric]
                     for s in seeds if sched in by[s]]
            if not diffs:
                continue
            md = float(np.mean(diffs))
            lo, hi = boot_ci(diffs)
            sig = "" if (lo <= 0 <= hi) else "  *"
            print(f"  {sched} vs {baseline}  Δ{metric}: "
                  f"{md:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{sig}")


def plot_cost_frontier(records, schedulers, out_path,
                       y='task_completion_rate', x='total_cost'):
    """
    Plots completion (y) vs total cost (x) across seeds and schedulers.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import collections

    fig, ax = plt.subplots(figsize=(7, 5))
    series = collections.defaultdict(list)
    for r in records:
        series[r['scheduler']].append((r[x], r[y], r.get('tax_rate', None)))

    for sched in schedulers:
        pts = series.get(sched, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, alpha=0.35, s=25, label=f"{sched} (runs)")

        taxes = [p[2] for p in pts]
        if all(t is not None for t in taxes) and len(set(taxes)) > 1:
            bytax = collections.defaultdict(list)
            for xx, yy, tt in pts:
                bytax[tt].append((xx, yy))
            curve = sorted((np.mean([a for a, _ in v]),
                            np.mean([b for _, b in v])) for v in bytax.values())
            cx = [c[0] for c in curve]
            cy = [c[1] for c in curve]
            ax.plot(cx, cy, '-o', linewidth=2, label=f"{sched} (mean curve)")
        else:
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=140, marker='D',
                       edgecolor='k', linewidth=1.2, label=f"{sched} (mean)")

    ax.set_xlabel(_pretty(x))
    ax.set_ylabel(_pretty(y))
    ax.set_title("Completion vs cost frontier (up-and-left is better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Frontier] saved to {out_path}")


def _pretty(k):
    return {
        'task_completion_rate': 'Reachable task completion rate',
        'group_completion_rate': 'Group (demand-unit) completion rate',
        'total_cost': 'Total realized cost',
        'realized_quality': 'Realized quality (best-of-successes)',
        'utility': 'Net utility',
    }.get(k, k)