# codesign

Reinforcement learning over a robot's **design** and its **controller** at the same time.

A design-conditioned hypernetwork `H(d, w)` maps a design vector `d` (link lengths, say)
and a simplex tradeoff `w` over objectives to the weights of a policy/value MLP. Training
it on a batch of designs at once gives a single network that controls any design in the
range, so a design can be evaluated without retraining a policy for it. MuJoCo MJX
supplies one compiled model per design, and rollouts are `vmap`ped across them.

Three algorithms live in `src/codesign/hyperdesigners/`:

| Algorithm | Designs come from | Objectives |
|---|---|---|
| `design_hypernetwork` | a space-filling sample of the design box | one (scalarized) |
| `mo_design_hypernetwork` | a space-filling sample, crossed with sampled tradeoffs | vector-valued critic |
| `mo_design_predictor_hypernetwork` | a learned predictor `f(d \| w)`, trained by GRPO | vector-valued critic |

All three share the PPO scaffolding in `hyperdesigners/shared.py` and sample into a common
`Grid` (`utils/grid.py`): `M` designs x `K` tradeoffs, each cell rolled out `per_cell`
times, which is also the on-disk dataset format.

```
src/codesign/
  envs/           model-as-input MuJoCo envs (cheetah, RHex, TwoAxis) + wrappers
  hyperdesigners/ the three training algos, their networks, losses, and shared PPO parts
  eval/           batched rollouts of a trained checkpoint; rollout video
  learning/       rebuild networks from a brax checkpoint, without a live env
  optimizers/     TuRBO, for design search against a trained hypernetwork
  utils/          Grid, model building/compilation, plotting
scripts/train.py  entry point, driven by a YAML config in config/
```

## Running

```bash
python -m scripts.train --config config/mo_design_hypernetwork/cheetah1D.yaml
```

## Tests

```bash
pytest              # fast checks: shapes, schedule arithmetic, network widths
pytest --runslow    # everything, incl. MuJoCo compiles and a few training steps
```

Tests that compile models or take training steps are marked `slow` and skipped by default,
so the bare run stays quick. Envs are built once per config and shared across tests, since
a `spec.compile` dominates the runtime of everything that needs one.

| File | Covers |
|---|---|
| `tests/test_networks.py` | critic width per algorithm, the extras the PPO losses read |
| `tests/test_grid.py` | `Grid` shapes, model dedupe, batching, save/load |
| `tests/test_rollout.py` | every grid cell gets its own design, tradeoff, and model |
| `tests/test_training.py` | schedule arithmetic; all three algos end to end |

Narrow a run the usual way — `pytest --runslow tests/test_grid.py -k dedupe`.
