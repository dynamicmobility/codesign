# codesign

Reinforcement learning over a robot's **design** and its **controller** at the same time.

The policy is conditioned on the robot's design, and trained on a batch of designs at
once, so one network controls any design in the range and a design can be evaluated
without retraining a policy for it. Mostly this uses a hypernetwork `H(d, w)` mapping a
design vector `d` (link lengths, say) and a simplex tradeoff `w` over objectives to the
weights of a policy/value MLP. MuJoCo MJX supplies one compiled model per design, and
rollouts are `vmap`ped across them.

Five algorithms live in `src/codesign/hyperdesigners/variants/`:

| Algorithm | Conditioned on the design by | Designs come from | Objectives |
|---|---|---|---|
| `design_mlp` | appending it to the observation | a space-filling sample of the design box | one (scalarized) |
| `design_hypernetwork` | generating the MLP weights | a space-filling sample of the design box | one (scalarized) |
| `design_lookup_hypernetwork` | generating the MLP weights, from a one-hot lookup | `M` designs sampled once and never resampled | one (scalarized) |
| `mo_design_hypernetwork` | generating the MLP weights | a space-filling sample, crossed with sampled tradeoffs | vector-valued critic |
| `mo_design_predictor_hypernetwork` | generating the MLP weights | a learned predictor `f(d \| w)`, trained by GRPO | vector-valued critic |

Each lives in one module (`hyperdesigners/variants/<algo>.py`) holding its own networks,
loss, training loop and `setup_*` config wiring; what more than one algo reuses sits in
`hyperdesigners/networks.py` and `losses.py`. All five share the PPO scaffolding in
`hyperdesigners/shared.py` and sample into a common `Grid` (`utils/grid.py`): `M` designs x `K` tradeoffs, each cell rolled out `per_cell`
times, which is also the on-disk dataset format.

`design_lookup_hypernetwork` is a warm-up stage for `design_hypernetwork`. Its feature map
is hardcoded to a one-hot lookup over the `M` fixed designs, so `flat(d_i) = W[i] + b`:
each row of `W` is one design's own policy, fed by that design's rollouts alone, and `b`
is frozen. Point a later `design_hypernetwork` run at the result with

```yaml
network_params:
  initialization_strategy: load_network
  warmup_checkpoint: results/<run>/<step>
```

which copies `W`/`b` and the observation normalizer across and regresses the fresh feature
MLPs onto `mlp(d_i) = e_i` so the run starts at the warm-up's policies. It needs
`num_features == the warm-up's num_designs`. One design per minibatch means a row of `W`
takes a gradient once every `M` steps, so Adam's effective step on it is ~`sqrt(M)` times
larger than usual: scale the warm-up's `learning_rate` down accordingly.

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

### Continuing a run

`num_timesteps` is a run's whole budget, so training it further is a matter of raising it
and pointing `--resume` at the run's own config:

```bash
# raise num_timesteps in the config, then
python -m scripts.train --config results/Sep05/<run>/config.yaml --resume
```

The job restarts from the newest checkpoint in that directory and trains the difference,
writing its checkpoints on the same step axis, appending to the same csvs and figures, and
logging to the same W&B run. Without `--resume` an existing run directory is still refused,
so nothing is overwritten by accident.

Each epoch writes `<run>/training_state`, holding the optimizer's moments, the observation
normalizer and every trained network, next to the brax checkpoints, which hold only what
inference needs. A run continued from that state, replaying the design schedule and the
RNG through the epochs already done, is the run it would have been had it never stopped --
provided its epochs stay the same size. An epoch is
`ceil(num_timesteps / (num_evals - 1)) ` steps rounded up to a whole training step, and a
continuation divides only what is left, so keeping the size means adding a whole number
`k` of the run's existing epochs and setting `num_evals` to `k + 1`:

```
epoch      = old num_timesteps / (old num_evals - 1)   # read it off the checkpoint spacing
num_evals  = k + 1
num_timesteps = old num_timesteps + k * epoch
```

Anything else still continues correctly; its epochs are just a different length, so the
checkpoints after the seam are spaced differently from those before it.

A run started before `training_state` existed can still be continued, from the params in
its newest checkpoint: Adam restarts from zero moments, so a few updates are scaled by the
gradient rather than by its running estimate. `design_mlp` refuses this path, its
checkpoints holding the policy but not the value network PPO trains against.

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
| `tests/test_resume.py` | finding the resume point; replaying a schedule; a run continued end to end |

Narrow a run the usual way — `pytest --runslow tests/test_grid.py -k dedupe`.
