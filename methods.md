# `mo_design_hypernetwork` — Methods

A knowledge-transfer document for the **multi-objective design hypernetwork** algorithm in
this repo (`codesign`). It describes what the algorithm computes, the models it uses, the
environment it trains on, the data flow, and the practical entry points (train / evaluate /
roll out). It assumes familiarity with PPO and JAX but not with this codebase.

---

## 1. One-paragraph summary

`mo_design_hypernetwork` trains a single **hypernetwork** `H(d, w)` that, given a robot
**design** `d` and a **tradeoff / objective-scalarization** `w`, emits the weights of a
policy MLP `π_{d,w}(a|s)` and a value MLP `V_{d,w}(s)`. It is the fusion of two existing
algorithms in the ecosystem:

- **`design_hypernetwork`** (this repo, `src/codesign/hyperdesigners/`) — a
  design-conditioned hypernetwork `H(d)` trained on a *model-as-input* (MAI) MuJoCo env,
  where each parallel env can have a different compiled robot model.
- **MORLAX** (`submodules/moplayground/src/moplayground/moppo/morlax.py`) — a
  *tradeoff*-conditioned hypernetwork `H(w)` for multi-objective RL, which scalarizes a
  per-objective reward *vector* by `w` (a dot product) before computing a single scalar
  advantage.

`mo_design_hypernetwork` conditions on **both** `d` and `w` (the concatenation
`[normalize(d), w]`), uses MORLAX's reward-scalarization-in-the-loss, and uses the
design-hypernetwork's model-as-input acting. At each training epoch it trains a **grid** of
`num_designs × num_tradeoffs` (design, tradeoff) cells simultaneously across the parallel
envs.

The end product is one set of hypernetwork weights that, at inference time, can produce a
specialized locomotion policy for *any* `(design, tradeoff)` pair — enabling fast Pareto /
co-design analysis without retraining per design or per objective weighting.

---

## 2. Why this exists (co-design motivation)

The broader goal is **computational co-design**: jointly reasoning about a robot's *physical
design* (here, a morphology parameter) and its *control policy*, across a *spectrum of
objective tradeoffs* (speed vs. energy vs. height). Training a separate policy for every
(design, tradeoff) pair is infeasible. A hypernetwork amortizes this: `H(d, w)` is trained
once over a distribution of designs and tradeoffs, and then queried cheaply to obtain the
policy for any specific pair. This makes it practical to sweep designs and trace Pareto
fronts.

---

## 3. The environment: `MAICheetah` (model-as-input Cheetah)

File: `src/codesign/envs/MAICheetah.py` (base: `src/codesign/envs/MAIBase.py`; multi-objective
mixin: `moplayground.envs.dmcontrol.cheetah.MOCheetah`).

### 3.1 Model-as-input (MAI)

Unlike a standard brax/MJX env, `MAICheetah` does **not** hold a fixed `mjx.Model`. Its API
takes the compiled model as an explicit argument:

```python
state = env.reset(rng, model)                 # model: an mjx.Model
state = env.step(state, action, model)
```

This is what makes per-env designs possible: we compile one `mjx.Model` per design and
`jax.vmap` `reset`/`step` over `(rng, model)` / `(state, action, model)`. Because model
compilation (`spec.compile()`) is a **host-side, non-jittable** operation, designs can only
change at host boundaries (i.e., once per training epoch), not inside the jitted inner loop.

### 3.2 The design parameter `d`

`MAICheetah.generate_model(d)` recompiles the cheetah XML with the **back thigh/shin scaled
by `d`** (a scalar; `design_dim = 1`):

```python
shin_pos     = [0.2, 0, -0.26]
new_shin_pos = shin_pos * d        # scale the back leg
# bshin body position, bthigh geom position & half-length are updated, then spec.compile()
```

So `d` is a **back-leg length scale**. The configured range is `d ∈ [0.5, 2.0]`
(`design_low`, `design_high`). Before being fed to the hypernetwork, designs are mapped to
`[0, 1]` via `normalize_design(d, low, high) = (d - low) / (high - low)`
(`src/codesign/utils/model.py`).

### 3.3 Observation

`_get_obs` (in `MOCheetah`) returns a **dict** with two identical entries:

```python
obs = concat([ data.qpos[1:], clip(data.qvel, -10, 10) ])   # shape (17,)
{'state': obs, 'privileged_state': obs}
```

The policy/value networks read the `state` key (`policy_obs_key = value_obs_key = "state"`),
so the effective observation dimension is **17**. Observation dims are **design-independent**
(the topology/DoF count is fixed; only link lengths change), which is why
`observation_size` can be inferred once from a nominal model (see §7.3).

### 3.4 Action

`action_size = 6` (the six actuated cheetah joints). Actions are clipped/scaled by
`params.action_scale` inside `step`. The action distribution is a `NormalTanhDistribution`
(param size = 12 = mean+std for 6 actions).

### 3.5 The reward **vector** and objectives

This is a genuinely multi-objective env. Per step, `reward_function` computes a dict of
scalar reward terms:

| key      | definition (MAICheetah)                                                     |
|----------|-----------------------------------------------------------------------------|
| `run`    | `min(4.0, (xposafter - xposbefore)/dt)` — forward speed, capped             |
| `energy` | `-sum(qfrc_actuator[3:]**2)` — negative actuator power (MAICheetah override)|
| `height` | `qpos[1] - DEFAULT_FF[1] + 0.2` — torso height above a reference            |
| `alive`  | `1.0` — shared survival bonus                                              |
| `done`   | termination penalty term                                                    |

`get_reward_and_metrics` (in `moplayground.envs.generic.mobase`) collapses these into a
**reward vector** of length `M = num_objectives`, one component per entry of
`reward.optimization.objectives`, using `reward.weights`:

```
objectives        = [['run'], ['energy'], ['height']]     # M = 3
shared_objectives = ['alive']
reward_vector[i]  = sum(weights[key]*rewards[key] for key in objectives[i])
                    + sum(weights[k]*rewards[k]  for k in shared_objectives)
```

So the env's `state.reward` has shape `(M,) = (3,)`: `[run, energy, height]` (each already
scaled by its config weight, with the shared `alive` bonus added to every component). The
objective order — `run, energy, height` — is the canonical order used everywhere (tradeoff
vectors, eval logging, the rollout script's `--tradeoff`).

**Termination**: `fall_termination` ends an episode if the torso pitches past ~80° or drops
below a height threshold.

---

## 4. The hypernetwork `H(d, w)`

Files: `src/codesign/hyperdesigners/networks.py` (wrappers) and
`submodules/moplayground/src/moplayground/moppo/networks.py` (`DualA2CHypernet`, the core).

### 4.1 What a hypernetwork does here

We first define two **template MLPs** with `brax`:

- a **policy network** (`policy_hidden_layer_sizes = [64, 64]`, output size 12),
- a **value network** (`value_hidden_layer_sizes = [64, 64]`, output size 1 — a **scalar**
  value head, as in MORLAX).

We then flatten each template's parameters into a vector and build a `DualA2CHypernet` that
*emits* those parameter vectors from a conditioning input. Concretely (`DualA2CHypernet`):

```
cond ──► policy_feature_mlp(hypersize) ──► f_π (num_features)
     └─► value_feature_mlp (hypersize) ──► f_V (num_features)

policy_weights = f_π · policy_W + policy_b     # shape = (#policy_params,)
value_weights  = f_V · value_W  + value_b      # shape = (#value_params,)
```

`policy_b` / `value_b` are **initialized to the flattened template weights**, and
`policy_W` / `value_W` are scaled by `W_variance` (default `0.0`). So at initialization the
hypernetwork outputs *exactly* the template MLP weights regardless of `cond`; training then
learns to modulate the weights as a function of `cond`.

**Only the hypernetwork is trained** (its feature MLPs + `W` + `b`). The template MLPs are
just structural scaffolding baked into `b`; there is no separately-trained policy/value MLP.
Trainable params are wrapped in `DesignHypernetParams(hypernetwork=...)`
(`src/codesign/hyperdesigners/losses.py`).

### 4.2 The conditioning input `cond`

The single change that distinguishes this from `design_hypernetwork` and MORLAX is the
conditioning vector. `make_mo_design_hypernet_networks` builds the hypernet with

```
num_objs (hypernet input dim) = design_dim + num_objectives   # 1 + 3 = 4
```

and everywhere the hypernet is applied, the input is the **concatenation**

```python
cond = concatenate([designs, directives], axis=-1)   # [d_norm (1), w (3)] -> (4,)
```

`designs` are `normalize_design`'d to `[0,1]`; `directives` (`w`) are simplex vectors from
the tradeoff sampler. **We do not re-normalize the concatenation** — `w` is already on the
simplex and `d` is already in `[0,1]`; L1-normalizing the concat would corrupt both.

Batched vs. single: if `cond` is 1-D (`(4,)`) the hypernet emits one policy/value MLP; if
`cond` is `(num_envs, 4)` it emits a *batch* of per-env MLPs (the `apply` vmaps the target
MLP over the env axis).

### 4.3 Inference-fn factory

`make_mo_design_inference_fn(networks)` returns

```python
inference_fn(params, designs, directives, deterministic=False) -> policy(obs, key)
```

where `params = (normalizer_params, hypernet_params)`. It concatenates `[designs,
directives]`, applies the hypernet to get per-env policy weights, and returns a standard
2-arg `policy(obs, key) -> (action, extras)`. The value head is not needed at acting time
(only in the loss), so it is discarded here.

---

## 5. The grid-search training scheme

This is the defining structural idea. Each epoch trains a **Cartesian grid** of designs ×
tradeoffs across the parallel envs.

Let:

```
num_cells      = num_designs * num_tradeoffs                 # e.g. 8 * 8 = 64
envs_per_cell  = num_envs // num_cells                       # e.g. 1024 // 64 = 16
```

Every env is assigned a `(design, tradeoff)` cell by this index ordering (single source of
truth, used identically in `build_grid` and the loss):

```
env = ((design_idx * num_tradeoffs) + tradeoff_idx) * envs_per_cell + rep
```

So 16 consecutive envs share exactly one `(design, tradeoff)` cell, one design occupies
`num_tradeoffs * envs_per_cell = 128` consecutive envs, etc.

`build_grid` (in `mo_design_hypernetwork.py`) constructs, per epoch:

- `designs_unique` — `[num_designs, design_dim]`, sampled uniformly in `[low, high]`
  (`model_lib.sample_designs`, host-side numpy).
- `tradeoffs_unique` — `[num_tradeoffs, num_objectives]` from `sample_tradeoffs` (§6).
- `batched_model` — a stacked `mjx.Model` with a leading env axis of `num_envs`. Only
  `num_designs` **unique** models are compiled; each is replicated
  `num_tradeoffs * envs_per_cell` times in grid order, then `model_lib.stack_models`
  stacks the leaves along axis 0. (Compiling 1024 models would be wasteful and slow.)
- `designs_full = repeat(designs_unique, reps, axis=0)` — `[num_envs, design_dim]`.
- `tradeoffs_full = tile(repeat(tradeoffs_unique, envs_per_cell), (num_designs, 1))` —
  `[num_envs, num_objectives]`.
- `designs_input = normalize_design(designs_full)`.

These per-env arrays (`designs_input`, `tradeoffs_full`) and the `batched_model` are passed
into the jitted training epoch. Within an epoch they are **constant** (models can't change
inside `jit`); they are re-sampled each outer iteration.

**Asserts** (must hold): `num_envs % num_cells == 0`, `num_eval_envs % num_cells == 0`,
`(batch_size * num_minibatches) % num_envs == 0`.

---

## 6. Tradeoff sampling

`sample_tradeoffs(rng, it, num_tradeoffs, num_objectives, sampling, alpha, warmup_frac, ...)`
(host-side numpy) ports MORLAX's `sample_preferences`. `num_tradeoffs` is the **sole driver**
of how many distinct tradeoffs are produced.

- `dense` — `num_tradeoffs` draws from `Dirichlet(alpha · 1_M)`.
- `sparse-heavytail` — `max(num_tradeoffs - num_objectives, 0)` Dirichlet draws **plus** the
  `num_objectives` axis-aligned one-hot corners (the extreme single-objective tradeoffs).
  E.g. `num_tradeoffs=8, num_objectives=3` → 5 simplex draws + 3 corners
  (`[1,0,0],[0,1,0],[0,0,1]`).
- `single-avg` — every tradeoff is the uniform `1/M`.
- **Warmup**: while `it < round(warmup_frac * num_evals_after_init)`, all tradeoffs are the
  uniform `1/M` (lets the policy first learn a competent average behavior before specializing).

Each returned tradeoff is a point on the probability simplex (non-negative, sums to 1),
shape `(num_tradeoffs, num_objectives)`.

---

## 7. The training loop

File: `src/codesign/hyperdesigners/mo_design_hypernetwork.py`,
`train_mo_design_hypernetwork(...)`.

### 7.1 Structure (single-device, `jax.jit`, no `pmap`)

```
initialize hypernet params, optimizer (adam + optional global-norm clip), obs normalizer
for it in range(num_evals_after_init):                 # outer epochs (host)
    build_grid(...) -> batched_model, designs_input, directives    # host, per epoch
    env_state = jit_reset(rngs, batched_model)
    training_epoch(...)   # jitted: scan over num_training_steps_per_epoch training_steps
    evaluate(...)         # per-objective + scalarized eval; logged via progress_fn
    policy_params_fn(...) # checkpoint hook
```

Each `training_step`:
1. Build the batched policy: `inference_fn((normalizer, hypernet), designs, directives)`.
2. `mo_generate_unroll` rolls out `unroll_length` steps over all `num_envs` (per-env model,
   per-env `(design, tradeoff)`), producing `MODesignTransition`s. A `jax.lax.scan` over
   `num_scans = batch_size*num_minibatches // num_envs` collects enough data.
3. Reshape rollout data `[num_scans, T, num_envs, ...]` → `[B, T, ...]`.
4. Update the observation normalizer's running statistics.
5. `num_updates_per_batch` epochs of PPO SGD: `sgd_step` shuffles into `num_minibatches`
   minibatches and applies `compute_mo_design_hypernet_loss` via `gradient_update_fn`.

`num_training_steps_per_epoch = ceil(num_timesteps / (num_evals_after_init *
env_step_per_training_step))`, where `env_step_per_training_step = batch_size * unroll_length
* num_minibatches`.

### 7.2 Acting (model-as-input, vector reward)

File: `src/codesign/hyperdesigners/mo_acting.py`.

`MODesignTransition` carries, per env: `observation, action, reward (vector, shape M),
design, directive, discount, next_observation, extras`. `mo_actor_step`:

- steps all envs with per-env models: `jax.vmap(env.step, in_axes=(0,0,0))`;
- keeps the **per-objective reward vector** (no scalarization at acting time — that happens
  in the loss, matching MORLAX);
- computes `termination` (env's own `done`, e.g. a fall) vs. `truncation` (episode-length
  horizon reached without termination);
- **manually auto-resets** finished envs back to their per-slot `first_state` via
  `_where_done` (the MAI env is not wrapped by brax's autoreset wrapper, so acting does it
  explicitly). `first_state` is the epoch's initial reset state, so a reset env returns to
  the correct design's initial condition.

### 7.3 Observation size & objective count (no throwaway model)

`obs_size = environment.observation_size` and
`num_objectives = len(environment.params.reward.optimization.objectives)`. `observation_size`
is inferred by the env itself via `jax.eval_shape(lambda rng: self.reset(rng,
self._mjx_model), key)` against the **already-compiled nominal model** — no fresh compile,
no live rollout. The normalizer is initialized from
`model_lib.observation_spec(obs_size)`, which turns the `{'state': (17,), ...}` shape dict
into a matching tree of `running_statistics` specs.

---

## 8. The loss (MORLAX scalarization + PPO/GAE)

File: `src/codesign/hyperdesigners/losses.py`, `compute_mo_design_hypernet_loss`. It is a
clone of the single-objective `compute_design_hypernet_loss` with **two** changes:

1. **Conditioning**: per-env policy/value weights come from the hypernet applied to the
   concatenation, using the value at the first timestep (design & tradeoff are constant over
   an episode segment):
   ```python
   cond = concatenate([data.design[:, 0], data.directive[:, 0]], axis=-1)   # [B, 4]
   policy_params, value_params = hypernetwork.apply(params.hypernetwork, cond)
   ```
2. **Reward scalarization** (the MORLAX step): after moving time to axis 0
   (`[B,T,·] → [T,B,·]`), the per-objective reward is dotted with the per-step directive to
   a **scalar** reward, *then* a single scalar GAE is computed:
   ```python
   rewards = sum(data.directive * data.reward, axis=2) * reward_scaling   # [T, B]
   ```

Everything else is standard clipped-PPO:
- scalar value baseline `V_{d,w}(s)` from the hypernet's value head;
- `ppo_losses.compute_gae(truncation, termination, rewards, values, bootstrap_value, λ, γ)`;
- optionally normalized advantages;
- clipped surrogate policy loss (`clipping_epsilon`), `0.25 * MSE` value loss, and an
  entropy bonus (`entropy_cost`).

Metrics returned: `total_loss, policy_loss, v_loss, entropy_loss`.

The key conceptual point: **one scalar advantage per env**, but the scalarization weights
differ per env (per cell). Because the hypernet is conditioned on that same `w`, gradients
teach it to produce policies specialized to each tradeoff — and, via the concatenated `d`,
to each design.

---

## 9. Evaluation

`evaluate(...)` builds a fresh design × tradeoff eval grid (`num_eval_envs` split into the
same cell structure), rolls out `episode_length` steps deterministically, and **accumulates
the per-objective return vector** `[num_eval_envs, num_objectives]` (masked by an `alive`
flag so post-termination steps don't contribute). It logs:

- `eval/episode_reward` — mean of the directive-scalarized return (`sum(w · return_vec)`),
- `eval/episode_reward_std`,
- `eval/episode_reward_obj{i}` — mean of each objective's raw return.

Reporting the **return vector** (not just the scalarized value) is what makes Pareto
structure visible: you can see how each objective's return moves as `w` sweeps the simplex.

---

## 10. Configuration reference

File: `config/mo_design_hypernetwork_cheetah.yaml`. Salient blocks:

```yaml
algorithm: "mo_design_hypernetwork"
env: "MAICheetah"
backend: jnp                       # jnp for training (MJX); np for single-thread rollout

env_config:
  reward:
    weights: { run, height, energy, alive, done }       # per-key reward weights
    optimization:
      objectives:        [['run'], ['energy'], ['height']]   # -> M = 3, this order
      shared_objectives: ['alive']
      labels:            ['Running Speed', 'Efficiency', 'Height']

learning_params:
  ppo_params:            # -> train_mo_design_hypernetwork(**ppo)
    num_timesteps, episode_length,
    num_envs: 1024, unroll_length: 20, batch_size: 512,
    num_minibatches: 2, num_updates_per_batch: 4,
    learning_rate, entropy_cost, discounting, reward_scaling,
    clipping_epsilon, gae_lambda, max_grad_norm,
    normalize_advantage, normalize_observations,
    num_evals: 15, num_eval_envs: 256, deterministic_eval, seed
  network_params:        # -> make_mo_design_hypernet_networks
    hypersize: [128, 128]          # hypernet feature-MLP hidden sizes
    num_features: 16               # hypernet feature dim
    policy_hidden_layer_sizes: [64, 64]
    value_hidden_layer_sizes:  [64, 64]
  design_params:
    num_designs: 8                 # designs per epoch
    design_low: 0.5, design_high: 2.0, design_dim: 1
  tradeoff_params:
    num_tradeoffs: 8               # tradeoffs per epoch (grid: 8 x 8 = 64 cells, 16 envs/cell)
    alpha: 1.0                     # Dirichlet concentration
    sampling: dense                # dense | sparse-heavytail | single-avg
    warmup_frac: 0.0               # fraction of epochs training only the uniform tradeoff
```

The wiring from config → train args is in
`src/codesign/hyperdesigners/factory.py::setup_mo_design_hypernetwork`: it binds
`make_mo_design_hypernet_networks` as the `network_factory` and
`train_mo_design_hypernetwork` with the design/tradeoff params + `**ppo_params`.

**Grid arithmetic sanity check** (default config): `1024 / (8 · 8) = 16` envs per
`(design, tradeoff)` cell; `64` cells total.

---

## 11. Running it

### Train
```bash
python scripts/train_mo_design_hypernetwork.py --config config/mo_design_hypernetwork_cheetah.yaml
```
Uses `minimal_mjx`'s trainer (`mm.learning.training.train`) with
`handle_params=setup_mo_design_hypernetwork`, logs to wandb, and checkpoints under
`save_dir/name` (e.g. `results/mo-design-hypernetwork-cheetah/mo-design-cheetah/<step>/`).

### Evaluate / roll out a single (design, tradeoff)
File: `scripts/rollout/rollout_single_design.py` (single-threaded, renders a video + reward
plot). It **dispatches on `config["algorithm"]`**, so the same script handles both
`design_hypernetwork` (`H(d)`) and `mo_design_hypernetwork` (`H(d, w)`). For the MO case it
requires both a design and a tradeoff (each with a default: `d = 1.0`, `w = uniform`), prints
the objective→weight mapping, and normalizes `w` onto the simplex.

```bash
python scripts/rollout/rollout_single_design.py \
  --config results/mo-design-hypernetwork-cheetah/mo-design-cheetah/config.yaml \
  --design 1.0 --tradeoff 1 0 0        # pure running (order: run, energy, height)
```

The loader `load_mo_design_hypernetwork` (`src/codesign/learning/inference.py`) rebuilds the
network from the checkpoint's saved architecture kwargs, re-binding `design_dim` (from
`design_params`), `num_objectives` (from `env_config.reward.optimization.objectives`), and
the init key — the same "rebind the call-time args" pattern as the single-objective loader.
The rollout core is `rollout_mo_design_hypernetwork_video` in
`src/codesign/eval/single_eval.py`.

---

## 12. File map

| File | Role |
|------|------|
| `src/codesign/hyperdesigners/mo_design_hypernetwork.py` | Training algo: grid build, tradeoff sampling, training loop, eval. |
| `src/codesign/hyperdesigners/mo_acting.py` | `MODesignTransition`, `mo_actor_step`, `mo_generate_unroll` (vector reward, per-env model, manual auto-reset). |
| `src/codesign/hyperdesigners/networks.py` | `make_mo_design_hypernet_networks` (widens conditioning to `design_dim+M`), `make_mo_design_inference_fn`. |
| `src/codesign/hyperdesigners/losses.py` | `compute_mo_design_hypernet_loss` (concat conditioning + directive scalarization + PPO/GAE); `DesignHypernetParams`. |
| `src/codesign/hyperdesigners/factory.py` | `setup_mo_design_hypernetwork` (config → train_fn + network_factory). |
| `src/codesign/hyperdesigners/acting.py` | Single-objective base acting (`_where_done`, `reset`) reused by `mo_acting`. |
| `src/codesign/envs/MAICheetah.py` / `MAIBase.py` | Model-as-input Cheetah; `generate_model(d)`, `observation_size`, vector reward. |
| `src/codesign/utils/model.py` | `sample_designs`, `normalize_design`, `stack_models`, `observation_spec`. |
| `src/codesign/learning/inference.py` | `load_mo_design_hypernetwork` / `load_mo_design_networks` (checkpoint → inference fn). |
| `src/codesign/eval/single_eval.py` | `rollout_mo_design_hypernetwork_video` (single-threaded eval/render). |
| `scripts/train_mo_design_hypernetwork.py` | Train entry point. |
| `scripts/rollout/rollout_single_design.py` | Roll out a trained `H(d)` or `H(d, w)` on one (design[, tradeoff]). |
| `config/mo_design_hypernetwork_cheetah.yaml` | Default training config. |
| `submodules/moplayground/src/moplayground/moppo/networks.py` | `DualA2CHypernet` (the hypernet core). |
| `submodules/moplayground/src/moplayground/moppo/morlax.py` | MORLAX reference algorithm. |

---

## 13. Key invariants & gotchas

- **Objective order is `[run, energy, height]`** everywhere (env reward vector, tradeoff
  vectors, eval `obj{i}`, rollout `--tradeoff`). It comes from
  `reward.optimization.objectives`. A `--tradeoff` of the wrong length is rejected.
- **Do not L1-normalize `cond = [d_norm, w]`.** `d` is in `[0,1]`, `w` is already a simplex
  point; normalizing the concatenation would destroy both signals. (The tradeoff `w` *is*
  simplex-normalized at sampling and at rollout time — just not jointly with `d`.)
- **Designs/tradeoffs are constant within a jitted epoch.** Model compilation is host-side
  and non-jittable, so `build_grid` runs on the host once per outer iteration; only
  `num_designs` unique models are compiled and then replicated in grid order.
- **Value head is scalar** (as in MORLAX), because the reward is scalarized *before* GAE.
  The advantage is a single scalar per env, using that env's tradeoff `w`.
- **Only the hypernetwork trains.** The template policy/value MLPs are folded into the
  hypernet's bias `b`; at init (`W_variance = 0`) the hypernet reproduces the templates
  exactly, then learns to modulate them by `(d, w)`.
- **Manual auto-reset**: MAI envs aren't wrapped by brax's autoreset, so `mo_actor_step`
  resets finished envs to the epoch's `first_state`. Truncation (horizon) vs. termination
  (fall) are tracked separately for correct GAE bootstrapping.
- **Grid divisibility asserts** must hold: `num_envs` and `num_eval_envs` each divisible by
  `num_designs * num_tradeoffs`, and `batch_size * num_minibatches` divisible by `num_envs`.
- **Checkpoint tuple is `(normalizer_params, hypernet_params)`** — a 2-tuple, like MORLAX
  (not the 3-tuple `(normalizer, policy, value)` of a standard brax PPO run). The loader
  rebinds `design_dim`, `num_objectives`, and an init key because those are passed at
  train-call time and not saved in the checkpoint's `network_factory_kwargs`.
```
