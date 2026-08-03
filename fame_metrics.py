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
                       submission_cost_rate, execution_cost_rate,
                       verbose=True):
    """
    Realized, causally-valid, reachability-adjusted metrics.
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

        n_submissions += 1
        submitted_count_by_task[task] = submitted_count_by_task.get(task, 0) + 1
        total_submission_cost += submission_cost_rate * q

        if status == S['rejected']:
            n_rejected += 1
        if S['cancelled'] is not None and status == S['cancelled']:
            n_cancelled += 1
            # Submission cost already counted above; no execution cost for cancelled passes
        if S['execution_failed'] is not None and status == S['execution_failed']:
            n_execution_failed += 1
            # Accepted + executed but failed: pay full execution cost, zero quality
            total_execution_cost += execution_cost_rate * q
        if status in _accepted_statuses:
            n_accepted += 1
        if status == S['received']:
            n_executed += 1
            executed_count_by_task[task] = executed_count_by_task.get(task, 0) + 1
            total_execution_cost += execution_cost_rate * q

            # Track best execution quality per task (only successful executions)
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

    def is_task_causally_valid(task):
        """Checks if all required parent dependencies succeeded in simulation execution."""
        parents = list(workflow_graph.predecessors(task))
        if not parents:
            return True
        for p in parents:
            edge_data = workflow_graph.get_edge_data(p, task)
            for _, constraint in edge_data.items():
                c_class = constraint.get('constraint_class')
                c_type = constraint.get('constraint_type')
                # If constraint requires parent success, parent must have succeeded
                if 'SUCCESS' in str(c_class):
                    if 'START_IF_SUCCESSFUL' in str(c_type):
                        if not task_success_map.get(p, False):
                            return False
        return True

    # Filter reachable tasks (root tasks + children of successful parents)
    reachable_tasks = set(t for t in tasks if is_task_causally_valid(t))
    n_tasks_reachable = len(reachable_tasks)

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
    }

    if verbose:
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