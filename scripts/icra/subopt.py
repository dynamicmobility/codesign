"""Overlay the design Pareto fronts each algorithm searched out for one robot, with the PPO
policies trained on pairs called out along them.

Every algorithm's front is the ``nsga3_front.npz`` that ``nsga2_design_front.py`` wrote
under its :class:`icra.Run`'s ``dataset_path/<run id>/`` -- the non-dominated
``(design, tradeoff)`` pairs of that run's trained network. This draws one figure per
objective pair, with every algorithm's front on the same axes in its own colour, so the
fronts can be read against each other. An algorithm with no run configured, or whose
dataset has not been generated yet, is warned about and skipped.

Each figure also stars the pairs ``front_callouts.py`` calls out on the ``MDH`` front's
slice, and joins each star to the return of the PPO run trained on that one design and
tradeoff, so the join is the gap between the front and a policy specialised to the pair.
The PPO runs are fetched from W&B and rolled out here; a callout no PPO policy was
trained on is warned about and left unjoined.

Run from the repository root, where the configured dataset paths resolve:
``python -m scripts.icra.subopt --robot cheetah``.
"""

import argparse
from itertools import combinations
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop
import wandb
from brax.training import checkpoint as brax_checkpoint
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from moplayground.utils.pareto import get_nondominated

import codesign
from codesign.eval.parallel_eval import build_grid_rollout_fn
from codesign.optimizers.nsga2 import repetition_keys

from scripts import icra
from scripts.icra import front_callouts
from scripts.icra.nsga2_design_front import DETERMINISTIC, SEED, download_run

OUT_DIR    = "scripts/icra/outputs"
FRONT_NAME = "nsga3_front.npz"

# The algorithms overlaid, each with its own colour (Okabe-Ito, so the pair stays
# distinguishable in greyscale and to colourblind readers). PPO holds one run per callout
# rather than a front, so it is drawn by plot_callouts in PPO_COLOR.
ALGO_COLORS = {"MDH": "#0072b2", "MLP": "#d55e00"}
PPO_COLOR   = "#cc79a7"  # Okabe-Ito reddish purple
PPO_SIZE    = 60         # PPO marker area, in points squared


def load_front(algo: str, run) -> codesign.Grid | None:
    """The run's front grid, or None (with a warning) when there is nothing to read.

    ``FINAL_CONFIGS`` carries an :class:`icra.Run` only for the algorithms already
    searched; the others hold a placeholder that names no dataset.
    """
    if not isinstance(run, icra.Run):
        print(f"warning: {algo} has no run configured; skipping")
        return None
    path = Path(run.dataset_path) / run.run_id / FRONT_NAME
    if not path.exists():
        print(f"warning: {algo} has no dataset at {path}; skipping")
        return None
    return codesign.Grid.load(path)


def plot_front(ax, grid: codesign.Grid, objs, color: str, label: str, labels, show_dominated=False, set_lim=False) -> None:
    """One objective pair's projection of a front: its non-dominated points, joined."""
    rewards = grid.mean_rewards.reshape(-1, grid.n_r)[:, objs]
    mop.plot_pareto(
        ax                   = ax,
        pareto               = rewards,
        colors               = color,
        objective            = [labels[o] for o in objs],
        show_dominated       = True, #show_dominated,
        dominated_alpha      = 0.1,
        # nondominated_s       = 200,
        nondominated_s=       
        outline_nondominated = 2.0,
        set_lims             = set_lim,
        marker               = "o",
    )
    # A projection's own frontier, sorted along the x objective so the join is monotone.
    front = rewards[get_nondominated(rewards)]
    ax.plot(*front[np.argsort(front[:, 0])].T, color=color, lw=1.5, zorder=0, label=label)


def pair_key(design, tradeoff) -> tuple[float, ...]:
    """A ``(design, tradeoff)`` pair rounded as :func:`front_callouts.arm` writes a sweep
    arm, so a callout and the PPO run trained on it compare equal."""
    return tuple(round(float(x), front_callouts.DECIMALS) for x in (*design, *tradeoff))


def callouts(front: codesign.Grid, labels) -> dict[tuple[int, int], tuple]:
    """Every 2D slice's callouts on ``front``, as ``{objs: (returns, keys, names)}``.

    ``returns`` are the callouts' ``(N_CALLOUTS, n_r)`` returns on the front, ``keys``
    their :func:`pair_key`s and ``names`` the sweep arm names ``front_callouts.py`` gives.
    """
    rewards   = front.mean_rewards.reshape(-1, front.n_r)
    designs   = front.designs.reshape(-1, front.design_dim)
    tradeoffs = front.tradeoffs.reshape(-1, front.n_r)
    slices = {}
    for objs in combinations(range(front.n_r), 2):
        picks = front_callouts.slice_callouts(rewards, objs)
        name  = "-".join(labels[o].lower() for o in objs)
        slices[objs] = (
            rewards[picks],
            [pair_key(designs[i], tradeoffs[i]) for i in picks],
            [f"{name}-{rank}" for rank in range(1, len(picks) + 1)],
        )
    return slices


def load_ppo(run) -> tuple[dict, Path] | None:
    """The PPO run's config and checkpoint, or None (with a warning) when W&B holds no
    policy for it. A negative ``run.checkpoint`` fetches the run's last checkpoint."""
    if not isinstance(run, icra.Run):
        print(f"warning: PPO entry {run!r} names no run; skipping")
        return None
    try:
        return download_run(run)
    except (wandb.errors.CommError, ValueError) as error:
        print(f"warning: PPO run {run.run_id} has no policy to roll out ({error}); skipping")
        return None


def rollout_setup(config, checkpoint: Path) -> str:
    """What a compiled PPO rollout is built from: the env config bar the design and
    tradeoff, which the multi-objective env does not read, the network, and the episode
    length."""
    env_config = config.env_config.to_dict()
    env_config["codesign"].pop("default_design")
    env_config["reward"]["optimization"].pop("default_scalarization")
    network = (checkpoint / "ppo_network_config.json").read_text()
    return repr((env_config, network, config.learning_params.ppo_params.episode_length))


def compile_rollout(config, checkpoint: Path):
    """The run's jnp env, and a jitted :func:`build_grid_rollout_fn` of its PPO network.

    Returns ``(env, rollout)``; ``rollout`` takes the params of any run sharing this
    run's :func:`rollout_setup`.
    """
    env, _  = codesign.load_env(config, backend="jnp")
    network = brax_checkpoint.get_network(
        ppo_checkpoint.load_config(checkpoint), ppo_networks.make_ppo_networks
    )
    make_ppo_policy = ppo_networks.make_inference_fn(network)

    def make_policy(params, designs, tradeoffs, deterministic=True):
        # Trained on one design and tradeoff, the policy reads neither.
        return make_ppo_policy(params, deterministic=deterministic)

    rollout = build_grid_rollout_fn(
        env           = env,
        n_steps       = config.learning_params.ppo_params.episode_length,
        make_policy   = make_policy,
        deterministic = DETERMINISTIC,
    )
    return env, rollout


def ppo_returns(robot: str) -> dict[tuple[float, ...], tuple[str, np.ndarray]]:
    """Every configured PPO run's id and ``(n_r,)`` return, keyed by the :func:`pair_key` of
    the design and tradeoff it was trained on.

    Each policy takes one episode of its config's ``episode_length`` on the jnp backend,
    rolled out as ``make_pair_rollout`` rolls out the front's pairs, so its return is
    ``r(s_0) + sum_t r(s_t) (1 - done_t)``. The rewards stay per objective, each weighted
    one, rather than scalarized under the run's training tradeoff, which places the return
    in the front's objective space. Runs sharing a :func:`rollout_setup` share one compile.
    """
    rollouts, returns = {}, {}
    for run in icra.FINAL_CONFIGS[robot].get("PPO", []):
        if (loaded := load_ppo(run)) is None:
            continue
        config, checkpoint = loaded
        setup = rollout_setup(config, checkpoint)
        if setup not in rollouts:
            rollouts[setup] = compile_rollout(config, checkpoint)
        env, rollout = rollouts[setup]

        opt  = config.env_config.reward.optimization
        grid = codesign.Grid.crossed(
            config.env_config.codesign.default_design, opt.default_scalarization
        )
        (_, _, rewards), _ = rollout(
            repetition_keys(1, 1, SEED),
            jnp.asarray(grid.designs),
            jnp.asarray(grid.tradeoffs),
            grid.build_models(env, tiled=False),
            ppo_checkpoint.load(checkpoint.resolve()),
        )
        key = pair_key(grid.designs[0, 0], grid.tradeoffs[0, 0])
        returns[key] = (run.run_id, np.asarray(rewards)[0, 0, 0])
    return returns


def report(slices, ppo) -> None:
    """Print each callout's front return beside its PPO policy's, warning of a callout no
    PPO policy was trained on and of a PPO policy trained on no callout."""
    called = {}
    for returns, keys, names in slices.values():
        for ret, key, name in zip(returns, keys, names):
            called.setdefault(key, (name, ret))  # a shared endpoint keeps its first name

    fmt = lambda x: np.array2string(x, precision=1, floatmode="fixed")
    for key, (name, ret) in called.items():
        if key not in ppo:
            print(f"warning: no PPO policy was trained on callout {name}; left unjoined")
            continue
        run_id, ppo_ret = ppo[key]
        print(f"  {name}: {front_callouts.ALGO} {fmt(ret)}  PPO ({run_id}) {fmt(ppo_ret)}")
    for key in ppo.keys() - called.keys():
        print(f"warning: PPO run {ppo[key][0]} was trained on no {front_callouts.ALGO} "
              f"callout; not drawn")


def plot_callouts(ax, returns, keys, objs, ppo) -> None:
    """A slice's callouts starred on the front, each joined to the return of the PPO policy
    trained on its pair. ``returns`` and ``keys`` are one slice's from :func:`callouts`."""
    stars = returns[:, objs]
    ax.scatter(*stars.T, marker="*", s=front_callouts.STAR_SIZE,
               color=ALGO_COLORS[front_callouts.ALGO], edgecolors="black", linewidths=1.0,
               zorder=3)
    # (n_joined, 2, 2): each joined star, then its PPO return.
    joins = np.array([(star, ppo[key][1][list(objs)])
                      for star, key in zip(stars, keys) if key in ppo])
    for join in joins:
        ax.plot(*join.T, color=PPO_COLOR, lw=1.0, ls="--", zorder=1)
    if len(joins):
        ax.scatter(*joins[:, 1].T, marker="D", s=PPO_SIZE, color=PPO_COLOR,
                   edgecolors="black", linewidths=1.0, zorder=3, label="PPO")
    ax.margins(0.1)


def main(args) -> None:
    out_dir = Path(args.out_dir)
    fronts = {
        algo: grid
        for algo, run in icra.FINAL_CONFIGS[args.robot].items()
        if algo in ALGO_COLORS
        and (grid := load_front(algo, run)) is not None
    }
    if not fronts:
        print(f"no fronts found for {args.robot}; nothing to plot")
        return

    grid = next(iter(fronts.values()))
    labels = [label.capitalize() for label in codesign.objective_labels(grid.objectives)]

    slices, ppo = {}, {}
    if front_callouts.ALGO in fronts:
        slices = callouts(fronts[front_callouts.ALGO], labels)
        ppo    = ppo_returns(args.robot)
        report(slices, ppo)
    else:
        print(f"warning: no {front_callouts.ALGO} front to call out pairs on; drawing no PPO")

    for objs in combinations(range(grid.n_r), 2):
        fig, ax = plt.subplots(figsize=(5, 6), layout="constrained")
        for algo, front in fronts.items():
            plot_front(ax, front, objs, ALGO_COLORS[algo], algo, labels, 
                       show_dominated=True if 'MDH' == algo else False,
                       set_lim=True if 'MDH' == algo else False
            )
        # if objs in slices:
        #     plot_callouts(ax, *slices[objs][:2], objs, ppo)
        # ax.legend(fontsize=10, frameon=False)
        codesign.dress_axis(
            ax,
            tick_size=18,
            num_xticks=3,
            num_yticks=4,
            label_size=28
        )
        name = "_".join(labels[o].lower() for o in objs)
        fig.savefig(out_dir / f"subopt_{args.robot}_{name}.{args.format}")
        plt.close(fig)

    drawn = [*fronts, *(["PPO"] if ppo else [])]
    print(f"wrote {' vs '.join(drawn)} figures to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=str, default="cheetah",
                        choices=sorted(icra.FINAL_CONFIGS),
                        help="which robot's runs to overlay the fronts of")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="directory the figures are written into; the datasets are "
                             "read from each run's own dataset_path")
    parser.add_argument("--format", type=str, default="svg", choices=("svg", "png", "pdf"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
