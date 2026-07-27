"""
Metrics for the stochastic-vs-deterministic scheduler comparison.

Design principle (see discussion): the stochastic planner's ONLY lever is
redundant booking, whose promised benefit is "more of the underlying demand
completed under the same uncertainty, per unit of constellation cost." Every
primary metric here is therefore sourced from REALIZED outcomes in
broker._requests (what actually happened in simulation), with denominators
taken from the demand (total requests that existed) -- never from planner
behavior (node counts, "scheduled" flags), which the planner controls and can
game.

Key definitions
---------------
Demand unit = a request_group (one volcano / one ship track). A workflow node
  ("task") is one observation request; a group bundles the detection + its
  follow-ups. Completion is reported at BOTH granularities:
    - task-level: fraction of individual requests that got >=1 successful collect
    - group-level: fraction of demand units with >=1 successful collect anywhere
Realized quality = sum over COMPLETED tasks of the quality of the BEST
  succeeding collect (matches the formulation's best-of-successes crediting).
Total bookings submitted = rows in broker._requests with a requested_pass
  (the true "passes" count; this is what avg-passes-per-request should use).
Accepted bookings = submissions that were accepted (SCHEDULED or DATA_RECEIVED).
Executed collects = DATA_RECEIVED rows (an accepted booking that actually ran).
Replans = number of rejection events (each forces a reschedule in the loop).

The cost model mirrors the run script: submission cost paid per submission,
execution cost paid per executed collect.
"""

import numpy as np


def _status_names(ObservationStatus):
    """Resolve the status enum members we rely on, tolerantly."""
    return {
        'rejected': ObservationStatus.CONSTELLATION_REJECTED,
        'scheduled': ObservationStatus.SCHEDULED,
        'received': ObservationStatus.DATA_RECEIVED,
    }


def compute_metrics_v2(workflow_graph, broker, ObservationStatus,
                       submission_cost_rate, execution_cost_rate,
                       verbose=True):
    """
    Realized, demand-based metrics. Returns a flat dict.

    Parameters
    ----------
    workflow_graph : the broker's workflow graph (nodes are tasks)
    broker : has ._requests DataFrame with columns
             ['request', 'requested_pass', 'status', ...]
    ObservationStatus : the status enum (passed in to avoid import coupling)
    submission_cost_rate, execution_cost_rate : floats (fractions of quality)
    """
    S = _status_names(ObservationStatus)
    reqs = broker._requests

    # --- Map each workflow task to its rewarder and its demand group ----------
    tasks = list(workflow_graph.nodes())
    obsreq_to_task = {t.observation_request: t for t in tasks}

    def group_of(task):
        # request_group defaults to the task name; volcano workflows set it to
        # the volcano name so detection + follow-ups share a group.
        return getattr(task, 'request_group', None) or getattr(task, 'name', str(id(task)))

    all_groups = set(group_of(t) for t in tasks)
    n_tasks_total = len(tasks)
    n_groups_total = len(all_groups)

    # --- Walk the realized request log once -----------------------------------
    n_submissions = 0            # every booking submission with a pass
    n_accepted = 0               # accepted (scheduled or received)
    n_executed = 0               # actually collected (data received)
    n_rejected = 0               # rejected submissions (== replan triggers)

    total_submission_cost = 0.0
    total_execution_cost = 0.0

    # best succeeding quality per task, and which groups completed
    best_success_quality_by_task = {}     # task -> float
    executed_count_by_task = {}           # task -> int (for true passes/request)
    submitted_count_by_task = {}          # task -> int

    for _, row in reqs.iterrows():
        rp = row['requested_pass']
        if rp is None:
            continue
        task = obsreq_to_task.get(row['request'])
        if task is None:
            continue
        status = row['status']

        # quality of THIS pass (rewarder is evaluated at the pass highest point)
        try:
            q = task.rewarder(rp.highest)
        except Exception:
            q = 0.0

        n_submissions += 1
        submitted_count_by_task[task] = submitted_count_by_task.get(task, 0) + 1
        total_submission_cost += submission_cost_rate * q

        if status == S['rejected']:
            n_rejected += 1
        if status in (S['scheduled'], S['received']):
            n_accepted += 1
        if status == S['received']:
            n_executed += 1
            executed_count_by_task[task] = executed_count_by_task.get(task, 0) + 1
            total_execution_cost += execution_cost_rate * q
            # best-of-successes crediting
            prev = best_success_quality_by_task.get(task, -np.inf)
            if q > prev:
                best_success_quality_by_task[task] = q

    total_cost = total_submission_cost + total_execution_cost

    # --- Completion (the PRIMARY metric), demand-based ------------------------
    completed_tasks = set(best_success_quality_by_task.keys())  # >=1 successful collect
    n_tasks_completed = len(completed_tasks)

    completed_groups = set(group_of(t) for t in completed_tasks)
    n_groups_completed = len(completed_groups)

    task_completion_rate = n_tasks_completed / n_tasks_total if n_tasks_total else 0.0
    group_completion_rate = n_groups_completed / n_groups_total if n_groups_total else 0.0

    # --- Realized quality (count weighted by how good each completion was) -----
    realized_quality = float(sum(best_success_quality_by_task.values())) if completed_tasks else 0.0

    # --- Utility (realized quality net of realized cost) ----------------------
    utility = realized_quality - total_cost

    # --- TRUE passes-per-completed-request (fixes the node-count bug) ----------
    # Redundancy actually realized: executed collects per task that completed.
    if completed_tasks:
        exec_passes_per_completed = np.mean([executed_count_by_task.get(t, 0)
                                             for t in completed_tasks])
    else:
        exec_passes_per_completed = 0.0
    # Redundancy actually BOOKED: submissions per task that had any submission.
    tasks_with_submissions = list(submitted_count_by_task.keys())
    if tasks_with_submissions:
        submitted_passes_per_task = np.mean([submitted_count_by_task[t]
                                             for t in tasks_with_submissions])
    else:
        submitted_passes_per_task = 0.0

    # --- Efficiency framings (report, but NOT as primary) ---------------------
    # completions per unit cost, and per booking submitted
    completions_per_cost = (n_tasks_completed / total_cost) if total_cost > 0 else float('nan')
    acceptance_rate = (n_accepted / n_submissions) if n_submissions else 0.0
    rejection_rate = (n_rejected / n_submissions) if n_submissions else 0.0

    m = {
        # ---- PRIMARY: demand completion ----
        'task_completion_rate': task_completion_rate,
        'group_completion_rate': group_completion_rate,
        'n_tasks_completed': n_tasks_completed,
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
        # ---- TRUE redundancy (fixes the old avg_passes_per_request) ----
        'submitted_passes_per_task': submitted_passes_per_task,
        'exec_passes_per_completed': exec_passes_per_completed,
        # ---- SECONDARY / efficiency ----
        'replans': n_rejected,                 # each rejection forces a reschedule
        'completions_per_cost': completions_per_cost,
        'acceptance_rate': acceptance_rate,
        'rejection_rate': rejection_rate,
    }

    if verbose:
        print(f"   [Metrics] Completion: tasks {n_tasks_completed}/{n_tasks_total} "
              f"({100*task_completion_rate:.1f}%), groups {n_groups_completed}/{n_groups_total} "
              f"({100*group_completion_rate:.1f}%)")
        print(f"   [Metrics] Realized quality {realized_quality:.1f}, cost {total_cost:.1f} "
              f"(sub {total_submission_cost:.1f} + exec {total_execution_cost:.1f}), "
              f"utility {utility:.1f}")
        print(f"   [Metrics] Bookings: {n_submissions} submitted, {n_accepted} accepted, "
              f"{n_executed} executed, {n_rejected} rejected ({100*rejection_rate:.0f}% rej)")
        print(f"   [Metrics] TRUE passes/task: {submitted_passes_per_task:.2f} submitted, "
              f"{exec_passes_per_completed:.2f} executed among completed "
              f"(old node-count metric was structurally ~1.0 and meaningless)")

    return m


# ---------------------------------------------------------------------------
# Paired analysis across seeds + cost-frontier plotting
# ---------------------------------------------------------------------------

def paired_summary(records, schedulers, primary='task_completion_rate',
                   baseline='deterministic'):
    """
    records: list of dicts, each {'scheduler': str, 'seed': int, **metrics}.
    Prints paired mean differences vs `baseline` with a bootstrap 95% CI, for
    the primary metric and utility. Pairing is by seed (same rejection draws).
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
    Frontier plot: completion (y) vs cost (x), one series per scheduler, each
    point a seed; large marker = mean. If records carry a 'tax_rate' field the
    means are connected in ascending-cost order to show the swept curve.
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
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, alpha=0.35, s=25, label=f"{sched} (runs)")
        # mean point(s)
        taxes = [p[2] for p in pts]
        if all(t is not None for t in taxes) and len(set(taxes)) > 1:
            # swept curve: mean per tax level, connected by ascending cost
            bytax = collections.defaultdict(list)
            for xx, yy, tt in pts:
                bytax[tt].append((xx, yy))
            curve = sorted((np.mean([a for a, _ in v]),
                            np.mean([b for _, b in v])) for v in bytax.values())
            cx = [c[0] for c in curve]; cy = [c[1] for c in curve]
            ax.plot(cx, cy, '-o', linewidth=2, label=f"{sched} (mean curve)")
        else:
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=140, marker='D',
                       edgecolor='k', linewidth=1.2, label=f"{sched} (mean)")

    ax.set_xlabel(_pretty(x)); ax.set_ylabel(_pretty(y))
    ax.set_title("Completion vs cost frontier (up-and-left is better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"[Frontier] saved to {out_path}")


def _pretty(k):
    return {
        'task_completion_rate': 'Task completion rate',
        'group_completion_rate': 'Group (demand-unit) completion rate',
        'total_cost': 'Total realized cost',
        'realized_quality': 'Realized quality (best-of-successes)',
        'utility': 'Net utility',
    }.get(k, k)