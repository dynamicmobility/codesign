# Handoff: `design_hypernetwork` — hypernetwork PPO across robot designs

## Goal

Train a single policy/value pair that **adapts to the robot's design** instead of a fixed
morphology. A hypernetwork maps a robot design `d` (for `MAICheetah`, the back-leg length
scale `d ∈ [0.5, 2.0]`) to the weights of a policy MLP; a **separate** hypernetwork maps
`d` to the weights of a value MLP. Both are trained on a **single** reward with brax's
**clipped PPO** loss — no multi-objective / directive scalarization.

This reuses moplayground's affine hypernetwork formulation and the structure of its
hypernetwork PPO loop, but runs single-objective and on a **model-as-input** environment
(`MAICheetah`), and is driven by **minimal-mjx's** trainer.

## What was added

### 1. Parallel design rollout (warm-up / reference)
- `scripts/rollout_mai_cheetah.py` — builds N `MAICheetah` variants via
  `generate_model(d)`, stacks them into one batched `mjx.Model`, and `jit(vmap(...))`-rolls
  them out for 50 steps. The `stack_models` pattern here is the basis for batching designs
  during training.

### 2. New algorithm package: `src/codesign/hyperdesigners/`
- `models.py` — host-side design→model helpers (`stack_models`, `build_batched_model`,
  `sample_designs`, `normalize_design`). `generate_model` is a host-side `spec.compile()`
  (not jittable), so designs only change at host boundaries.
- `networks.py` — `make_design_hypernet_networks` builds the target policy/value MLPs
  (brax) and wraps moplayground's **`DualA2CHypernet`** (separate feature MLP + affine
  `W/b` per head) keyed on the **design**, *without* the preference-simplex normalization.
  Plus `make_design_inference_fn`.
- `losses.py` — `compute_design_hypernet_loss`: single-objective brax `compute_gae` +
  clipped surrogate; the hypernetwork is keyed on `data.design[:, 0]`. Adapted from
  `compute_morlax_loss` with the MO scalarization removed.
- `acting.py` — `DesignTransition` + `actor_step`/`generate_unroll` that thread the per-env
  stacked `mjx.Model` through `vmap(env.step, in_axes=(0,0,0))` and reimplement brax's
  `EpisodeWrapper` + `AutoResetWrapper` semantics (truncation at horizon, termination from
  env `done`, `discount = 1 - termination`, auto-reset to a per-slot `first_state`).
  `scalarize_reward` collapses `MAICheetah`'s multi-objective reward vector to one scalar.
- `design_hypernetwork.py` — `train_design_hypernetwork`, a single-device PPO loop forked
  from `moplayground.moppo.morlax.train`: resamples designs + rebuilds the stacked model at
  each eval boundary, unrolls, runs PPO SGD on the hypernetwork only, and evals on freshly
  sampled (held-out) designs. **v1 simplifications:** single device (no `pmap`); env state
  re-sampled each epoch; deterministic per-design resets.
- `factory.py` — `setup_design_hypernetwork(config) -> (train_fn, network_factory)`. This
  is the **`handle_params`** function, same shape as minimal-mjx's `setup_ppo`. It is
  intentionally **not** registered in minimal-mjx's `_ALGO_HANDLERS` (different algorithm
  than the brax-PPO minimal-mjx is based on); it is plugged in explicitly.

### 3. Env change
- `src/codesign/envs/MAICheetah.py` — added an `observation_size` property. The base
  `MjxEnv.observation_size` traces `reset(rng)`, but this env is model-as-input
  (`reset(rng, model)`), so the override traces against a nominal-design model (`d=1.0`).
  This is required so minimal-mjx's trainer can read `eval_env.observation_size`.

### 4. Config, run script, launcher
- `config/design_hypernetwork_cheetah.yaml` — `learning_params.{ppo_params, network_params,
  design_params}` (+ optional `reward_objective_weights` to scalarize the MO reward).
- `scripts/train_design_hypernetwork.py` — loads the config, builds `MAICheetah`, and drives
  the run through **`mm.learning.training.train(config, env, eval_env,
  handle_params=setup_design_hypernetwork)`**, so minimal-mjx owns the run directory,
  checkpointing (`save_model`) and progress plotting (`plot_progress`). Has a `--smoke`
  flag for a quick CPU run.
- `scripts/train.sh` — tmux launcher mirroring moplayground's `train.sh`, but with the
  `codesign` conda env and the `CODESIGN` session name.

## How it integrates with minimal-mjx

`mm.learning.training.train` is mostly orchestration (output dir, checkpoint, progress) and
delegates the algorithm to whatever `handle_params` returns — that `handle_params=` argument
is the "plug in without registering" hook. The model-as-input / multi-design nature did not
make it incompatible; only two small accommodations were needed:
1. `MAICheetah.observation_size` (above);
2. `train_design_hypernetwork` accepts (and ignores) `wrap_env_fn` / `eval_env`, which
   `mm.train` passes — brax's wrappers can't thread the per-design `mjx.Model`, so the
   algorithm keeps its own acting/eval and reuses only minimal-mjx's orchestration.

## How to run

```bash
# quick end-to-end check on CPU
JAX_PLATFORMS=cpu python scripts/train_design_hypernetwork.py --smoke

# full run
python scripts/train_design_hypernetwork.py --config config/design_hypernetwork_cheetah.yaml

# detached tmux launch (codesign env)
./scripts/train.sh config/design_hypernetwork_cheetah.yaml
```

Outputs land under `save_dir/name` (e.g. `results/design-hypernetwork-cheetah/test/`):
orbax checkpoints `(normalizer, hypernet)`, `config.yaml`, `progress.csv`, `progress.svg`.

## Status / verification

- Smoke runs pass end-to-end (via both the direct loop and `mm.train`): jits, builds the
  stacked model, finite `policy_loss` / `v_loss` / `entropy_loss`, eval reward stable on
  held-out designs, checkpoints + progress artifacts written.
- `scripts/test_mai_cheetah.py` still passes (no env regression).

## Notes / follow-ups

- **Reward scalarization:** `MAICheetah` emits a 3-vector MO reward with `alive` added to
  each component. We collapse it with a fixed objective-weight vector
  (`reward_objective_weights`, default all-ones) — the fixed-directive analogue of morlax's
  `sum(directive*rewards)`. Defaulting to ones triple-counts `alive`; tune if needed.
- **`v_loss` scale:** large because returns are unnormalized and `height` is weighted 10×;
  tune `reward_scaling`.
- **Possible upgrades:** multi-device `pmap`; carrying env state across epochs; richer
  (multi-dim) design spaces; a dedicated checkpoint loader / rollout for trained hypernets.
