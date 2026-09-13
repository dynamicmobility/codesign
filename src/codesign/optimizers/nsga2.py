"""Design-Pareto search with NSGA-II / NSGA-III over a trained multi-objective design network.

A trained ``mo_design_hypernetwork`` / ``mo_design_mlp`` is a policy *family* ``H(d, w)``:
a design ``d`` paired with a simplex tradeoff ``w`` picks out one policy, whose rollout
returns the env's per-objective reward vector ``R(d, w)``. The search therefore treats the
pair ``(d, w)`` as the decision variable and ``R(d, w)`` as the objectives, so every
individual is one point of the design x tradeoff frontier and the algorithm's diversity
mechanism spreads the population along it. Searching over ``d`` alone would need a
reduction of each design's whole front to a single vector, capping the frontier at one
point per design.

Everything here is in maximization space; ``DesignTradeoffProblem`` negates at the one
boundary where pymoo, which minimizes, needs it.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from moplayground.utils.pareto import get_nondominated, project_to_simplex
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.ref_dirs import get_reference_directions

from codesign.eval.parallel_eval import build_grid_rollout_fn
from codesign.learning.inference import load_mo_design_hypernetwork, load_mo_design_mlp
from codesign.utils.grid import Grid, sample_tradeoffs_cpu
from codesign.utils.model import normalize_design, sample_designs, unnormalize_design

# The MO algorithms this searches over; both key their policy on ``(design, tradeoff)``.
LOADERS = {
    "mo_design_hypernetwork": load_mo_design_hypernetwork,
    "mo_design_mlp"         : load_mo_design_mlp,
}

ALGORITHMS = ("nsga2", "nsga3")


def _make_policy_fn(inference_fn):
    """Adapt an MO inference fn to :func:`build_grid_rollout_fn`'s keyword protocol.

    The grid rollout calls ``make_policy(params=, designs=, tradeoffs=, deterministic=)``,
    which ``mo_design_mlp``'s singular ``design``/``tradeoff`` parameters do not match.
    """
    def make_policy(params, designs, tradeoffs, deterministic=True):
        return inference_fn(params, designs, tradeoffs, deterministic=deterministic)

    return make_policy


def repetition_keys(n_pairs: int, per_cell: int, seed: int) -> jnp.ndarray:
    """One reset key per repetition, shared by every cell: ``(n_pairs, 1, per_cell, 2)``.

    Keying on the repetition alone keeps the return a deterministic function of
    ``(d, w)``. :func:`codesign.eval.parallel_eval.grid_keys` draws a key per cell instead,
    so a pair's reset would vary with its slot in the population; NSGA-II is elitist and
    carries a survivor's recorded return forward untested, which would preserve lucky
    resets rather than good designs.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed + 2), per_cell)
    return jnp.broadcast_to(keys, (n_pairs, 1, per_cell, keys.shape[-1]))


def make_pair_rollout(
    env,
    config,
    *,
    n_steps: int,
    per_cell: int = 1,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    workers: int = 1,
):
    """Build a batched rollout of ``P`` ``(design, tradeoff)`` pairs, one env per pair.

    Args:
        env: a model-as-input MO env (e.g. ``MOCodesignCheetah``).
        config: the run config dict (as written to ``config.yaml`` at train time).
        n_steps: rollout length in env steps.
        per_cell: rollout repetitions per pair, averaged into its objective vector.
        checkpoint_path: explicit checkpoint dir; defaults to latest under ``save_dir/name``.
        seed: base PRNG seed, fixed across pairs (see :func:`repetition_keys`).
        deterministic: take the policy mode (vs. sampling) at each step.
        workers: threads compiling the per-design ``mjx.Model``s.

    Returns:
        ``rollout(designs, tradeoffs) -> (P, per_cell, n_r)``, the accumulated per-objective
        return of each pair. ``designs`` are ``(P, design_dim)`` in physical units,
        ``tradeoffs`` ``(P, n_r)`` on the simplex.

    The jitted rollout is built once here rather than per call, so a search evaluating a
    fixed ``P`` every generation compiles once instead of every generation.
    """
    inference_fn, params = LOADERS[config["algorithm"]](config, path=checkpoint_path)
    rollout_fn = build_grid_rollout_fn(
        env           = env,
        n_steps       = n_steps,
        make_policy   = _make_policy_fn(inference_fn),
        deterministic = deterministic,
    )

    def rollout(designs, tradeoffs):
        # ``P`` designs against one tradeoff apiece: cell (m, 0) carries its own
        # (d_m, w_m), which is what pairs the two arrays one to one.
        grid = Grid(
            designs    = np.asarray(designs, np.float32)[:, None],
            tradeoffs  = np.asarray(tradeoffs, np.float32)[:, None],
            per_cell   = per_cell,
            objectives = env.objectives,
        )
        (_, _, rewards), _ = rollout_fn(
            repetition_keys(len(grid.designs), per_cell, seed),
            # The policy is conditioned on designs in [0, 1]; the grid holds physical units.
            normalize_design(jnp.asarray(grid.designs), config=config),
            jnp.asarray(grid.tradeoffs),
            grid.build_models(env, tiled=False, workers=workers),
            params,
        )
        return np.asarray(rewards)[:, 0]  # (P, per_cell, n_r)

    return rollout


class SimplexRepair(Repair):
    """Project each decision vector's tradeoff block back onto the probability simplex.

    SBX and polynomial mutation are box operators, so they leave ``w`` with ``sum(w) != 1``.
    ``H(d, w)`` only ever saw simplex tradeoffs in training, and an off-simplex conditioning
    vector is out of distribution -- it returns numbers rather than an error, so the drift
    would go unnoticed. pymoo applies a repair after the operators and before evaluation.
    """

    def __init__(self, design_dim: int):
        super().__init__()
        self.design_dim = design_dim

    def _do(self, problem, X, **kwargs):
        X = np.array(X, dtype=float)
        X[:, self.design_dim:] = np.apply_along_axis(
            project_to_simplex, 1, X[:, self.design_dim:]
        )
        return X


class DesignTradeoffProblem(Problem):
    """NSGA-II's view of the search: the env's objectives over ``(design, tradeoff)`` pairs.

    The decision vector is ``[normalized_design, tradeoff]`` -- ``design_dim + n_r``
    coordinates all boxed in ``[0, 1]``, so a single ``eta`` means the same thing on every
    axis. ``_evaluate`` takes the whole population at once, which is what makes the
    population size the rollout's batch width.

    Every pair evaluated is kept, so :meth:`archive` holds the entire search and not just
    the population that survived it.
    """

    def __init__(self, rollout, design_limits, objectives):
        self.rollout = rollout
        self.low, self.high = np.asarray(design_limits, np.float64)
        self.objectives = list(objectives)
        self.design_dim = len(self.low)
        super().__init__(
            n_var = self.design_dim + len(self.objectives),
            n_obj = len(self.objectives),
            xl    = 0.0,
            xu    = 1.0,
        )
        self.evaluated = []

    def split(self, X) -> tuple[np.ndarray, np.ndarray]:
        """``[normalized_design, tradeoff]`` rows -> physical designs and their tradeoffs."""
        X = np.clip(np.asarray(X, np.float64), 0.0, 1.0)
        designs = unnormalize_design(X[:, :self.design_dim], self.low, self.high)
        return np.asarray(designs, np.float32), X[:, self.design_dim:].astype(np.float32)

    def _evaluate(self, X, out, *args, **kwargs):
        designs, tradeoffs = self.split(X)
        rewards = self.rollout(designs, tradeoffs)
        self.evaluated.append((designs, tradeoffs, rewards))
        out["F"] = -rewards.mean(axis=1)  # pymoo minimizes; the env's rewards are maximized

    def archive(self) -> Grid:
        """Every pair evaluated, as an ``(N, 1, per_cell, n_r)`` grid of one pair per cell."""
        designs, tradeoffs, rewards = (np.concatenate(x) for x in zip(*self.evaluated))
        return Grid(
            designs    = designs[:, None],
            tradeoffs  = tradeoffs[:, None],
            per_cell   = rewards.shape[1],
            rewards    = rewards[:, None],
            objectives = self.objectives,
        )


def pareto_front(grid: Grid) -> Grid:
    """The non-dominated cells of a one-pair-per-cell grid, in the same layout.

    Taken over the whole archive rather than the final population, which recovers frontier
    points that were found and later crowded out of the population.
    """
    keep = get_nondominated(grid.mean_rewards[:, 0])
    return dataclasses.replace(
        grid,
        designs   = grid.designs[keep],
        tradeoffs = grid.tradeoffs[keep],
        rewards   = grid.rewards[keep],
    )


def initial_population(design_dim, num_objectives, pop_size, seed) -> np.ndarray:
    """A space-filling initial population of ``[normalized_design, tradeoff]`` rows.

    Sobol designs crossed with tradeoffs drawn to include the simplex corners, rather than
    pymoo's i.i.d. uniform default: the frontier's extremes sit at the corners, so seeding
    them saves the search from having to rediscover them.
    """
    rng = np.random.default_rng(seed)
    designs = sample_designs(
        rng, pop_size, low=np.zeros(design_dim), high=np.ones(design_dim), dim=design_dim
    )
    tradeoffs = sample_tradeoffs_cpu(
        rng, pop_size, num_objectives, sampling="sparse-heavytail"
    )
    return np.hstack([designs, tradeoffs]).astype(np.float64)


def build_algorithm(name: str, pop_size: int, design_dim: int, num_objectives: int, seed: int):
    """The pymoo algorithm to search with, seeded and repaired for this problem.

    The two differ only in how they pick survivors out of the last front that does not fit:

      * ``nsga2`` -- crowding distance, the per-objective spacing to an individual's two
        neighbours. It measures density one axis at a time, which stops being a faithful
        estimate once there are three or more objectives, so the frontier tends to clump.
      * ``nsga3`` -- a fixed set of reference directions through the objective space,
        keeping the survivor associated with each. The spread is then imposed by the
        directions rather than inferred from the population, and holds as objectives grow.

    ``nsga3``'s directions are Riesz s-energy points rather than the Das-Dennis simplex
    lattice, which only lands on ``C(p + m - 1, m - 1)`` counts for an integer ``p``
    (``m = 3`` gives 3, 6, 10, 15, ...). Taking one direction per population slot keeps the
    rollout's batch width exactly ``pop_size`` for any value the caller asks for.
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'; expected one of {ALGORITHMS}.")

    shared = dict(
        n_offsprings         = pop_size,   # one batched rollout per generation
        sampling             = initial_population(
            design_dim, num_objectives, pop_size, seed
        ),
        repair               = SimplexRepair(design_dim),
        eliminate_duplicates = True,
    )
    if name == "nsga2":
        return NSGA2(pop_size=pop_size, **shared)
    return NSGA3(
        ref_dirs = get_reference_directions("energy", num_objectives, pop_size, seed=seed),
        pop_size = pop_size,
        **shared,
    )


def run_nsga(
    env,
    config,
    *,
    n_steps: int,
    budget: int,
    pop_size: int,
    algorithm: str = "nsga2",
    per_cell: int = 1,
    checkpoint_path: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    workers: int = 1,
    verbose: bool = True,
) -> tuple[Grid, Grid, object]:
    """Search the design x tradeoff frontier of a trained MO design network.

    Args:
        env: a model-as-input MO env (e.g. ``MOCodesignCheetah``).
        config: the run config dict; its ``algorithm`` must be one of :data:`LOADERS`.
        n_steps: rollout length in env steps.
        budget: total ``(design, tradeoff)`` pairs evaluated, over all generations.
        pop_size: pairs rolled out together, so ``budget / pop_size`` generations.
        algorithm: which of :data:`ALGORITHMS` to search with (see :func:`build_algorithm`).
        per_cell: rollout repetitions per pair, averaged into its objective vector.
        checkpoint_path: explicit checkpoint dir; defaults to latest under ``save_dir/name``.
        seed: PRNG seed for the rollouts and for pymoo's operators.
        deterministic: take the policy mode (vs. sampling) at each step.
        workers: threads compiling the per-design ``mjx.Model``s.
        verbose: print pymoo's per-generation table.

    Returns:
        ``(front, archive, result)`` -- the non-dominated pairs, every pair evaluated (both
        grids of one pair per cell), and pymoo's own result object.
    """
    if config["algorithm"] not in LOADERS:
        raise ValueError(
            f"'{config['algorithm']}' has no design x tradeoff policy to search; expected "
            f"one of {sorted(LOADERS)}."
        )
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'; expected one of {ALGORITHMS}.")

    problem = DesignTradeoffProblem(
        rollout = make_pair_rollout(
            env,
            config,
            n_steps         = n_steps,
            per_cell        = per_cell,
            checkpoint_path = checkpoint_path,
            seed            = seed,
            deterministic   = deterministic,
            workers         = workers,
        ),
        design_limits = (config["env_config"]["codesign"]["low"],
                         config["env_config"]["codesign"]["high"]),
        objectives    = env.objectives,
    )

    result = minimize(
        problem,
        build_algorithm(
            algorithm, pop_size, problem.design_dim, problem.n_obj, seed
        ),
        MaximumFunctionCallTermination(budget),
        seed    = seed,
        verbose = verbose,
    )

    archive = problem.archive()
    return pareto_front(archive), archive, result
