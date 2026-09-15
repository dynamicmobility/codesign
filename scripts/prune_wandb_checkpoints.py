"""Delete all but the first, last, and best checkpoint artifact of ungrouped W&B runs.

A training run logs one `model` artifact version per eval (see
`minimal_mjx.utils.logging.save_model`), so a long run leaves dozens of ~100 MB
checkpoints behind. This keeps the three that carry information -- the initial policy,
the final policy, and the best-scoring one -- and deletes the rest.

Dry run by default; pass --execute to actually delete.
"""

import argparse
from collections import namedtuple

import wandb
import os

ENTITY  = os.environ["WANDB_ENTITY"]
PROJECT = 'codesign'
# wandb's own `wandb-history`/`wandb-events` artifacts share the run; only ours is 'model'.
ARTIFACT_TYPE = 'model'
# Metrics that rank a checkpoint, in priority order. All are maximized.
DESIGN_HYPERVOLUME = 'Pareto-Optimal Design Hypervolume'
FALLBACK_METRIC    = 'eval/episode_reward'
# Spread statistics carry 'hypervolume'-adjacent names but do not rank front quality.
SPREAD_TERMS = ('std', 'dev', 'spacing')

Plan = namedtuple('Plan', 'run keep drop metric reason')


def version_index(artifact):
    """The integer N of an artifact whose version is 'vN'."""
    return int(artifact.version.removeprefix('v'))


def choose_metric(run):
    """The metric that ranks `run`'s checkpoints: a hypervolume if it logged one.

    Hypervolume is the objective-space volume a run's Pareto front dominates above a
    reference point (`pymoo.indicators.hv.HV`, via `plot_design_pareto_progress`). For a
    multi-objective run it measures the whole front, whereas `eval/episode_reward` is one
    scalarization of it, so the hypervolume ranks such a checkpoint better. Spacing and
    standard deviations describe how a front is spread, not how good it is, so they are
    skipped.
    """
    keys = [key for key in run.summary.keys() if not key.startswith('_')]
    if DESIGN_HYPERVOLUME in keys:
        return DESIGN_HYPERVOLUME
    hypervolumes = sorted(
        key for key in keys
        if 'hypervolume' in key.lower()
        and not any(term in key.lower() for term in SPREAD_TERMS)
    )
    return hypervolumes[0] if hypervolumes else FALLBACK_METRIC


def best_step(run, metric, samples=100_000):
    """The `_step` at which `run` recorded its largest `metric`, or None if unusable.

    W&B downsamples a curve longer than `samples` rows, and a downsampled curve can miss
    the true optimum, so a full-length response is refused rather than trusted. Training
    logs one row per eval, far under the cap.
    """
    history = run.history(keys=[metric], samples=samples, pandas=True)
    if history.empty or metric not in history or len(history) >= samples:
        return None
    return int(history.loc[history[metric].idxmax(), '_step'])


def plan_run(run, min_evals):
    """Which of `run`'s checkpoints to keep and which to delete, or None to skip it.

    Checkpoints are matched to the metric curve through `artifact.metadata['iteration']`,
    which `save_model` sets to the same step the progress callback logs at. Grouped and
    still-running runs are left alone.
    """
    if run.group or run.state == 'running':
        return None
    artifacts = [a for a in run.logged_artifacts() if a.type == ARTIFACT_TYPE]
    if len(artifacts) < min_evals:
        return None

    artifacts.sort(key=version_index)
    keep = {version_index(artifacts[0]), version_index(artifacts[-1])}

    metric = choose_metric(run)
    step = best_step(run, metric)
    # A resumed run can re-save one step; the later version wins.
    by_iteration = {a.metadata.get('iteration'): a for a in artifacts}
    if step is None or step not in by_iteration:
        # Without a trustworthy best, keeping the whole run beats keeping the wrong three.
        return Plan(run, artifacts, [], metric, f'no {metric} match for best checkpoint')
    keep.add(version_index(by_iteration[step]))

    return Plan(
        run,
        [a for a in artifacts if version_index(a) in keep],
        [a for a in artifacts if version_index(a) not in keep],
        metric,
        f'best at step {step}',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--project', type=str, default=PROJECT)
    parser.add_argument('--min_evals', type=int, default=40,
                        help='Only prune runs with at least this many checkpoints')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Prune only this run, instead of the whole project')
    parser.add_argument('--execute', action='store_true',
                        help='Delete; without it, only report what would be deleted')
    args = parser.parse_args()

    api = wandb.Api()
    path = f'{args.entity}/{args.project}'
    runs = [api.run(f'{path}/{args.run_id}')] if args.run_id else api.runs(path)
    print(f'Scanning {len(runs)} run(s) in {path}...')

    plans, freed, skipped = [], 0, []
    for run in runs:
        plan = plan_run(run, args.min_evals)
        if plan is None:
            continue
        if not plan.drop:
            skipped.append(plan)
            continue
        plans.append(plan)
        bytes_ = sum(a.size or 0 for a in plan.drop)
        freed += bytes_
        print(f"  {run.id}  {run.name[:38]:38s} keep {len(plan.keep)} "
              f"delete {len(plan.drop):4d}  {bytes_ / 1e9:6.2f} GB  "
              f"{plan.metric} ({plan.reason})")

    for plan in skipped:
        print(f"  SKIP {plan.run.id}  {plan.run.name[:38]:38s} {plan.reason}")

    print(f'\n{len(plans)} runs to prune, '
          f'{sum(len(p.drop) for p in plans)} checkpoints, {freed / 1e9:.1f} GB')
    if not args.execute:
        print('Dry run; pass --execute to delete.')
        return

    deleted, failures = 0, []
    for plan in plans:
        print(f'Deleting {len(plan.drop)} checkpoints from {plan.run.id}...', flush=True)
        for artifact in plan.drop:
            try:
                # delete_aliases stays False so a version someone tagged raises instead.
                artifact.delete(delete_aliases=False)
                deleted += 1
            except Exception as error:
                failures.append((artifact.name, error))
    print(f'Deleted {deleted} checkpoints; {len(failures)} failed.')
    for name, error in failures[:20]:
        print(f'  {name}: {type(error).__name__}: {error}')


if __name__ == '__main__':
    main()
