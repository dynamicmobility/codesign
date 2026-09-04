# codesign

Reinforcement learning over a robot's **design** and its **controller** at the same time.

The policy is conditioned on the robot's design, and trained on a batch of designs at
once, so one network controls any design in the range and a design can be evaluated
without retraining a policy for it. Mostly this uses a hypernetwork `H(d, w)` mapping a
design vector `d` (link lengths, say) and a simplex tradeoff `w` over objectives to the
weights of a policy/value MLP. MuJoCo MJX supplies one compiled model per design, and
rollouts are `vmap`ped across them.

Four algorithms live in `src/codesign/hyperdesigners/variants/`:

| Algorithm | Conditioned on the design by | Designs come from | Objectives |
|---|---|---|---|
| `design_mlp` | appending it to the observation | a space-filling sample of the design box | one (scalarized) |
| `design_hypernetwork` | generating the MLP weights | a space-filling sample of the design box | one (scalarized) |
| `mo_design_hypernetwork` | generating the MLP weights | a space-filling sample, crossed with sampled tradeoffs | vector-valued critic |
| `mo_design_predictor_hypernetwork` | generating the MLP weights | a learned predictor `f(d \| w)`, trained by GRPO | vector-valued critic |

Each lives in one module (`hyperdesigners/variants/<algo>.py`) holding its own networks,
loss, training loop and `setup_*` config wiring; what more than one algo reuses sits in
`hyperdesigners/networks.py` and `losses.py`. All four share the PPO scaffolding in
`hyperdesigners/shared.py` and sample into a common `Grid` (`utils/grid.py`): `M` designs x `K` tradeoffs, each cell rolled out `per_cell`
times, which is also the on-disk dataset format.

```
src/codesign/
  envs/           model-as-input MuJoCo envs (cheetah, RHex, TwoAxis) + wrappers
  hyperdesigners/ the shared PPO parts, plus variants/, one module per training algo
                  with its own networks, loss, training loop and config wiring
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
