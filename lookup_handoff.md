# `design_lookup_hypernetwork` — open items

Warm-up stage that trains `W` as a basis of per-design experts, plus the
`initialization_strategy: load_network` path that hands the result to a full
`design_hypernetwork` run. This file is the list of what is still unfinished or unchecked.

| Where | What |
|---|---|
| `src/codesign/hyperdesigners/variants/design_lookup_hypernetwork.py` | the whole algo: hypernetwork wrapper, networks, training loop, per-design eval, config wiring |
| `src/codesign/hyperdesigners/hypernetworks.py` | `LookupA2CHypernet`, `sobol_design_table` |
| `src/codesign/hyperdesigners/variants/design_hypernetwork.py` | `load_warmup_params` — the handoff |
| `src/codesign/utils/plotting.py` | `plot_design_rewards`, `plot_design_learning_curves`, `plot_design_rewards_progress` |
| `config/design_lookup_hypernetwork/` | `cheetah1D.yaml` (also the test config), `cheetah3D.yaml` |

## TODO

- [x] **`pytest -q tests/test_networks.py`** — 12 passed. The `designs.csv` write in
  `plot_design_rewards_progress`, `warn_off_table` and its two tests are verified; the
  `jax.errors.TracerArrayConversionError` reference in the tracer guard holds.

- [x] **3D warm-up restarted** as `results/Sep04/design-lookup-warmup-3D-longer-ppo-params/`,
  which has `designs.csv`, `progress.svg`, `design_rewards_progress.csv`, 30
  `eval_grid_<step>.npz` and `videos/`.

- [x] **`num_envs` vs `num_parallel_envs` — settled on `num_parallel_envs` everywhere except
  `ppo`.** `train_design_mlp`, `train_design_lookup_hypernetwork` and
  `train_mo_design_predictor` were renamed to match `train_design_hypernetwork` and
  `train_mo_design_hypernetwork`; the argument plays the same role in all five (it is what
  goes into `shared.Schedule.make`'s env slot). `config/ppo/*.yaml` and
  `design_hypernetwork_two_axis.yaml` keep `num_envs`, because those run minimal-mjx's own
  PPO. `scripts/wandb_sweep.py` gained `env_count_key` so it reads whichever name a config
  uses. This also fixed four configs that were already broken at HEAD — they passed
  `num_envs` to a trainer taking `num_parallel_envs`: `design_hypernetwork/cheetah1D.yaml`,
  `design_hypernetwork_rhex.yaml`, `mo_design_hypernetwork/cheetah1D.yaml`, `cheetah6D.yaml`.
  `--runslow` went from 5 failed / 13 passed to 2 failed / 16 passed.

- [ ] **`scripts/rollout_single.py` needs no lookup branch, but run it once.** It does not
  dispatch on algorithm: it calls `codesign.save_policy_rollout_video`, whose
  `design_lookup_hypernetwork` arm already exists (`src/codesign/eval/rollout_video.py:448`).
  `MO_ALGORITHMS` only gates `--tradeoff`, and a lookup run correctly lands in the
  "single-objective; tradeoffs are ignored" branch. Untested: `default_video_design(config)`
  when `--design` is omitted, which is almost certainly off the Sobol table and will snap to
  a neighbouring anchor (now with a warning). Run it with an explicit design from
  `designs.csv`.

- [ ] **Two pre-existing failures, untouched, neither lookup-related.**
  `test_algorithm_trains_and_evaluates[mo_design_hypernetwork]` raises `KeyError: 'per_cell'`
  (`setup_mo_design_hypernetwork` requires `tradeoff_sampling.per_cell`, which only
  `mo_design_hypernetwork/cheetah3D.yaml` defines). `[design_mlp]` raises
  `ValueError: Incompatible shapes for broadcasting: (2, 1, 1, 3) and requested shape
  (2, 1, 4, 2, 3)` — a real shape bug in its training path, not a config gap.

- [ ] **Optional: `parallel_eval.py` has no lookup branch.** Only matters if batched
  checkpoint evaluation is wanted for this algo.

## Things to know before running

**Scale `learning_rate` down by `sqrt(num_designs)`.** One design per minibatch means a row
of `W` sees a gradient once every `M` steps, so Adam's second moment decays in between and
the active-step update is `eta * g / sqrt(v_hat) ~ eta * sqrt(M)`. Measured 5.6x at `M = 32`
against `sqrt(32) = 5.66`; momentum is not the cause (`b1 = 0.9` gives 560, `b1 = 0.0` gives
555) and the per-design independence is unaffected. The shipped configs use
`1.77e-5 = 1e-4 / sqrt(32)`. At `design_hypernetwork`'s rate this diverges.

**Any off-table design snaps to its nearest anchor.** `features(d)` is
`one_hot(argmin_j ||d - D_j||^2)`, which always returns some anchor, so a design that is not
one of the `M` gets a neighbour's policy driving a robot that policy never trained on. It
does not error. `load_design_lookup_hypernetwork` now warns when the requested design is
farther from its nearest anchor than the anchors' own median nearest-neighbour spacing.

**Which designs those are.** `unit = Sobol(d=design_dim, seed=seed, scramble=True).random(M)`,
then `physical = low + unit * (high - low)`. The table the lookup compares against is
`unit` itself, since `normalize_design` inverts that map exactly. Row `i` is Sobol point `i`
in sequence order, not sorted, and owns row `i` of `policy_W`. Fully determined by
`(seed, num_designs, design_dim, low, high)`, all of which are in the run's `config.yaml`;
a run also writes them in physical units to `designs.csv`.

**Doing the handoff.** In the full run's `network_params`:

```yaml
initialization_strategy: load_network
warmup_checkpoint: results/<run>/<step>
```

`load_warmup_params` rebuilds the anchors from the warm-up's own `config.yaml` (rather than
assuming the two runs agree), copies `policy_W`/`policy_b`/`value_W`/`value_b` and the
observation normalizer, and regresses both fresh feature MLPs onto `mlp(d_i) = e_i` so the
run starts at the warm-up's policies instead of a random blend of them. It requires
`num_features == the warm-up's num_designs` — already true in
`config/design_hypernetwork/cheetah3D.yaml` and `cheetah6D.yaml`, both 32.

## Test state

`pytest -q tests/test_training.py --runslow`: **16 passed, 2 failed**, the two being the
non-lookup failures listed above. `pytest -q tests/test_networks.py`: **12 passed**.
A full `pytest -q --runslow` across every test file has not been rerun since.
