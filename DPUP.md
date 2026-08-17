# DPUP — Design Predictor Universal Policy

Review of the `mo_design_predictor_hypernetwork` algorithm: what the design predictor
`f(d | w)` is, how it is initialized and trained, and the issues its current training
scheme has. Written to be readable without prior knowledge of the repository.

Primary sources:

| File | Contents |
| --- | --- |
| `src/codesign/hyperdesigners/mo_design_predictor_hypernetwork.py` | training loop |
| `src/codesign/hyperdesigners/networks.py:197-330` | predictor architecture + inference fn |
| `src/codesign/hyperdesigners/losses.py:240-301` | GRPO loss |
| `src/codesign/utils/grid.py:120-178` | `DesignPredictorSampleGrid` |
| `src/codesign/hyperdesigners/factory.py:80-121` | config -> train fn wiring |
| `config/mo_design_predictor_hypernetwork/cheetah1D.yaml` | reference config |

Numbers below are for `cheetah1D.yaml` unless stated: `num_envs=4096`,
`num_tradeoffs=8`, `num_designs` (GRPO group size `G`) `=8`, so 64 cells x 64 envs per
cell; `resamples_per_epoch=20`, `num_evals=15` (14 epochs after init);
`episode_length=500`, `discounting=0.99`; design box `low=[0.5]`, `high=[2.0]`.

---

## 1. What the predictor is

`f(d | w)` is a stochastic policy whose "state" is a tradeoff `w` on the simplex and
whose "action" is a robot design `d`.

**Architecture** (`networks.py:269`). A plain MLP `[16, 16, 2*design_dim]`, built by
brax's `make_policy_network` with `distribution_type='tanh_normal'`. Input is `w` with
`obs_size=num_objectives`; a simplex vector is already O(1) in every coordinate, so no
observation normalizer is used. The `2*design_dim` outputs are split by
`NormalTanhDistribution` into `loc` and a raw scale, with `scale = softplus(raw) + 0.001`.

**Forward pass** (`networks.py:303-328`) produces three coupled quantities:

1. `raw ~ N(loc(w), scale(w))` — the *pre-tanh* sample.
2. `d_norm = 0.5 * (tanh(raw) + 1) in (0, 1)` — the design in normalized units.
3. `log_prob = log N(raw) - log |d tanh / d raw|` — the density of the squashed variable,
   evaluated at `raw`.

Keeping the sample in pre-tanh space is what makes the log-prob finite: `tanh` saturates,
so a density written directly on the squashed variable is numerically unusable near the
box edges. The extra `0.5 * (. + 1)` remap onto `[0, 1]` is affine with constant Jacobian,
so it cancels in every log-prob *ratio* and does not affect the GRPO objective.

`d_norm` is mapped to physical units by `unnormalize_design` before MuJoCo model
generation (`mo_design_predictor_hypernetwork.py:188`), then mapped back by
`normalize_design` for the hypernetwork's conditioning input (`:203-205`), so `f` and
`H(d, w)` share the same `[0, 1]` design convention.

## 2. Initialization

`design_networks.design_predictor_network.init(key_design_net)`
(`mo_design_predictor_hypernetwork.py:469`) is a bare lecun-uniform MLP init with no
output-layer rescaling. Instantiated with `design_dim=1`, `num_objectives=3`, that gives:

```
loc at the three simplex corners: [0.125, 0.124, 0.089]
scale:                           [0.658, 0.707, 0.647]
mode design (normalized):        [0.562, 0.562, 0.544]
sampled d_norm: mean 0.544  std 0.248   1%/99% quantiles: 0.056 / 0.964
```

Two consequences. First, **the predictor starts essentially constant in `w`** — all three
simplex corners map to a mode near 0.55. Second, it starts broad but not uniform: for
cheetah1D that is a back-leg scale of ~1.31 spanning roughly `[0.58, 1.87]`, with only ~3%
of samples in the bottom decile of the box and ~7% in the top. `tanh` never reaches the
endpoints.

## 3. Outputs

| Consumer | What it receives |
| --- | --- |
| `build_grid` (`:182-191`) | `n_tradeoffs x G` sampled designs -> unnormalized -> `DesignPredictorSampleGrid` -> one `mjx.Model` per (tradeoff, design) pair, tiled `per_cell` times |
| GRPO loss | `extras['raw_action']` `(T, G, design_dim)` and `extras['log_prob']` `(T, G)`, the behaviour log-probs |
| Checkpoint (`:475-480`) | `(normalizer_params, hypernet_params, design_predictor_params)` — a 3-tuple, against the 2-tuple `mo_design_hypernetwork` saves |
| Eval metrics (`:437-446`) | `eval/design_mode_obj{i}_dim{j}`, the **mode** design at each simplex corner. This is the alignment diagnostic: if `f` separates the extremes, these diverge |
| Rollout (`eval/rollout_video.py:226`) | mode, or a sample under `--sample_design`, at a user-chosen `w'` decoupled from the policy's `w` |

## 4. Training loop

Per resample (20 per epoch):

1. Draw 8 tradeoffs; draw `G=8` designs per tradeoff from `f(. | w)` -> 64 distinct robots
   (`:505`).
2. Reset all 4096 envs on those robots; run a PPO chunk training `H(d, w)` against this
   fixed grid (`:515`).
3. Roll out one deterministic episode per env under the just-trained policy, mask rewards
   after termination, discount and scalarize by that env's own `w` (`:339-374`), average
   the 64 repetitions per cell -> `values` of shape `(8, 8)` (`:533`).
4. GRPO: standardize values within each tradeoff's group, form the clipped ratio against
   the sampling-time log-prob, 4 full-batch epochs (`:536`, `losses.py:240-301`).

Two parts of this are worth keeping as-is. **Group-relative standardization is the right
baseline**: scalarized returns at different `w` live on different scales, so a global
baseline would be meaningless, and normalizing within a group restricts the comparison to
designs judged at the same tradeoff. **Termination handling is correct**: in
`rollout_returns`, `reward = nst.reward * alive` uses the pre-update `alive`, so the
terminating step's reward counts and everything afterwards is zeroed, with no bootstrap —
the Monte-Carlo return the GRPO advantage wants. Averaging 64 independent resets per
design also makes each value a genuinely low-variance estimate.

---

## 5. Issues

Ranked by expected impact.

### 5.1 The `H` / `f` feedback loop has no brake

**What.** The "reward" GRPO sees for a design is the return of the *current* hypernetwork,
and the hypernetwork is trained **only** on designs `f` proposed (`:505-508`). In
`mo_design_hypernetwork.py:129` designs come from a space-filling `sample_designs` over
the whole box; here coverage is entirely `f`'s support.

**Why it matters.** `f` favors a region -> `H` receives more training there -> measured
returns there rise -> the advantage pushes `f` further in -> `H` sees an even narrower
region. Nothing keeps `H` calibrated outside `f`'s support, so the value ranking becomes
self-fulfilling rather than informative. The predictor converges on whichever region it
happened to drift toward while `H` was still undertrained.

**Fix.** Reserve a fraction of the grid cells for uniformly-sampled designs so `H` stays
calibrated off `f`'s support, and/or freeze the predictor for the first few epochs while
`H` warms up. The second is nearly free: skip the `design_sgd` call while
`it < warmup_epochs`.

### 5.2 Nothing prevents variance collapse; `design_entropy_cost` is numerically inert

**What.** Measured entropy at init is ~0.01 nats. Driving `scale` all the way to
`min_std=0.001` only moves it to about -5.5. Multiplied by `entropy_cost=1e-3`, the whole
entropy term spans ~0.006 of the total loss, against a surrogate that is O(1) because
advantages are standardized.

**Why it matters.** The score function `grad log f` scales like `1/sigma`, so as `sigma`
shrinks the policy term grows while the entropy term does not — the bonus is weakest
exactly where it is needed. Once `sigma` bottoms out at `min_std`, all `G` designs in a
group are identical, the within-group value spread is pure rollout noise, and the
standardization at `losses.py:278-283` amplifies that noise into +-1 advantages. The
degeneracy guard there (`r_std <= 1e-6 * |r_mean|`) only catches near-exact ties, which
noisy 64-episode means will not produce, so `f` random-walks instead of stalling visibly.

**Fix.** Pass a real `min_std` to `NormalTanhDistribution` (`networks.py:256`) — a floor of
0.1-0.2 in pre-tanh space keeps a usable spread over the box. Raise `design_entropy_cost`
by two to three orders of magnitude. A KL trust region against the previous iterate is the
more principled version if the entropy floor proves too blunt.

### 5.3 `design_init_noise_std` and `design_noise_std_type` are dead under `tanh_normal`

**What.** brax reads both only in the `'normal'` branch of `make_policy_network`. Under
`tanh_normal` the module is a bare MLP and the scale comes from its output head, so those
arguments (`networks.py:219-220`) do nothing. The comment at `networks.py:265-268` states
this; the practical consequence is not stated.

**Why it matters.** There is no knob for the predictor's initial exploration width. It is
whatever `softplus(~0) ~ 0.69` gives, which is why the measured init spans ~[0.58, 1.87]
in physical units regardless of config.

**Fix.** Either expose `min_std` / `var_scale` on the `NormalTanhDistribution`
construction and drive them from the yaml, or drop the two dead arguments so the config
surface does not imply control it does not have.

### 5.4 The shared `alive` objective makes the GRPO signal largely `w`-independent

**What.** `mobase.py:75-76` adds `weights['alive'] * alive` to *every* objective
component, and `w` sums to 1, so `w . r` carries the full +1/step alive bonus at every
tradeoff. The other terms are predominantly positive as well (`run` capped at 4, `height`
= `qpos[1] - default + 0.2` at weight 10).

**Why it matters.** The discounted scalarized return is therefore dominated by how long
the current policy keeps this robot upright — nearly the same ranking for every `w`. Early
in training, when the policy is weak, that is the only legible signal, so `f` collapses
onto a single `w`-agnostic "stable" design before the run/height/energy distinctions ever
become visible. Combined with 5.1 this is the most likely observed failure.

**Detector.** `eval/design_mode_obj{i}_dim{j}` (`:437-446`). Three nearly identical corner
designs is the symptom.

**Fix.** Either move `alive` out of `shared_objectives` for this algorithm, or subtract a
survival baseline from the value before standardizing so the group comparison is on the
`w`-dependent part. Delaying predictor updates (5.1) also helps, since the survival
signal stops dominating once the policy can keep most designs alive.

### 5.5 Discounting the design value optimizes the wrong quantity

**What.** `discounts = 0.99 ** arange(500)` (`:369`) gives an effective horizon of ~100
steps, i.e. ~1 s of the 5 s episode.

**Why it matters.** Designs are ranked mostly on startup transients. Discounting exists
for GAE's bias/variance tradeoff inside PPO; a design is a *terminal* choice with no
bootstrapping, so the quantity you actually want to maximize is the full episode return. A
design that accelerates slowly but is faster at steady state is currently penalized.

**Fix.** Use `gamma = 1` (or a much larger gamma) in `discounted_scalarized` only, leaving
PPO's `discounting` untouched. One-line change with a real effect on which design wins.

### 5.6 Box-boundary optima are unreachable and their gradients vanish

**What.** `d_norm = 0.5 * (tanh(raw) + 1)` means `d = low` or `d = high` requires
`loc -> -inf` or `+inf`.

**Why it matters.** For a 1-D leg length against a mostly monotone objective such as "run
fast", the optimum plausibly *is* at a boundary. `loc` then grows slowly, the tanh
gradient decays, and the mode asymptotes near but never at the edge. Read a plateau in
`eval/design_mode_*` as saturation, not convergence.

**Fix.** Nothing cheap and clean; this is inherent to a squashed Gaussian on a box. If
boundary optima matter, a Beta distribution on `[0, 1]` reaches the edges with finite
parameters. Alternatively keep tanh and accept that the reported mode understates the
optimum, which the `TODO` at `networks.py:319-322` already touches on.

### 5.7 The eval Pareto plot does not mean what it means for `mo_design_hypernetwork`

**What.** `:449-454` builds a `DesignTradeoffRolloutGrid` whose `designs` field has shape
`(G, n_tradeoffs, design_dim)`, while the dataclass documents `(n_designs, design_dim)`.
`plot_design_paretos` colors one curve per index of the first axis.

**Why it matters.** Group member `g` at tradeoff `t` is a *different physical robot* than
group member `g` at tradeoff `t'` — under this algorithm designs are paired with their
tradeoff, not crossed with it. Each colored "per-design frontier" therefore connects
unrelated designs. Nothing crashes, because only `_design_label` reads `designs` and its
call site is commented out (`plotting.py:57`).

**Fix.** For this algorithm plot the union of all `(design, tradeoff)` points colored by
design *value* (a continuous colormap over `d`), not by group index — i.e.
`plot_design_objective_pareto` with per-point colors rather than
`plot_design_paretos`.

### 5.8 Host-side model compilation is now the bottleneck

**What.** `DesignPredictorSampleGrid.build_models` (`grid.py:145`) compiles
`n_tradeoffs * G = 64` MuJoCo models per resample, against 8 unique designs per resample
for `mo_design_hypernetwork`. Measured `generate_model` + `mjx.put_model` at **~91 ms per
design** (CPU, this machine).

**Why it matters.** ~5.8 s of serial Python per resample, ~2 min per epoch, ~27 min over a
14-epoch cheetah1D run, all on the critical path with nothing else running.

**Fix.** No correctness change needed. If it becomes limiting, lower
`resamples_per_epoch` (each resample buys one GRPO batch, so this trades predictor updates
for wall time), or cache compiled models keyed on a quantized design.

### 5.9 Minor: dead and silently-ignored config

* `warmup_frac` is accepted at `:78` and never used. `sample_tradeoffs` takes `it` and
  ignores it, while its docstring (`grid.py:31`) describes a warmup that is not
  implemented. Worth noting because a warmup is one of the fixes for 5.1.
* `design_sampling.sampling: random` in `cheetah1D.yaml` is never read —
  `factory.py:105-119` only consumes `tradeoff_sampling.sampling`.
* The predictor reuses PPO's `clipping_epsilon` (0.3) with no separate knob, so its trust
  region cannot be tuned independently of the policy's.

---

## 6. Suggested order of attack

1. **5.2** — a real `min_std` floor and a larger `design_entropy_cost`. Smallest change,
   directly targets the collapse mechanism.
2. **5.1** — mix uniformly-sampled designs into the grid and/or warm up `H` before the
   first predictor update.
3. **5.5** — undiscounted design values.

Those three are all small edits against the same failure mode: `f` committing to a design
before the value signal it is committing on is trustworthy. Re-check
`eval/design_mode_obj{i}_dim{j}` after each — corner designs that stay separated are the
sign the predictor is doing its job.
