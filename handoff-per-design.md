# Handoff: per-design advantage scale, precomputed once per rollout

Status: **not implemented.** Stratified batching (`batching_strategy: stratified`) and
grouped advantage normalization (`num_advantage_groups`) *are* implemented and tested;
this document describes the follow-on change they leave available, why it is worth doing,
and what it costs.

## Background: what is already in place

`compute_design_hypernet_loss` standardizes advantages over the minibatch it is handed:

```python
grouped = advantages.reshape(advantages.shape[0], num_advantage_groups, -1)
mean = grouped.mean(axis=(0, 2), keepdims=True)
std  = grouped.std(axis=(0, 2), keepdims=True)
advantages = ((grouped - mean) / (std + 1e-8)).reshape(advantages.shape)
```

Under `batching_strategy: stratified` the minibatch is cell-major with every design
present in equal share, so `num_advantage_groups = num_designs` puts each design on its
own scale while the gradient still averages over all designs. That is the property the
change was for, and it holds.

## The residual problem

The statistic's sample size is tied to `num_minibatches`. Each design contributes

```
rows_per_design_per_minibatch = per_cell * num_scans / num_minibatches
```

rows to a minibatch, so its mean and std are estimated from
`rows_per_design_per_minibatch * unroll_length` transitions. Two things are wrong with
that:

1. **The nominal sample count overstates the real one.** The `unroll_length` timesteps
   within a row are not independent. GAE is an exponentially weighted sum along the
   trajectory, `A_t = sum_l (gamma*lambda)^l delta_{t+l}`, so consecutive `A_t` share most
   of their terms — with `gamma=0.98, lambda=0.95` the weights decay with a horizon of
   `1/(1 - 0.931) ~ 14` steps, which is most of a 20-step row. The effective sample size
   is therefore closer to the number of **independent rows** than to `rows x T`. At the
   sizing in `cheetah6D_stratified.yaml` (`num_minibatches=8`, 32 designs, `per_cell` 128)
   that is 16 effective samples, giving a std estimate with roughly `1/sqrt(2*16) ~ 18%`
   relative error.

2. **The statistic is re-drawn every gradient step.** `sgd_step` re-partitions the data on
   each of the `num_updates_per_batch` passes, and each of the `num_minibatches` steps
   within a pass gets its own group of rows. So the divisor applied to a design's
   advantages changes from step to step by a quantity that is pure estimation noise. The
   objective's scale therefore moves underneath the optimizer — a nonstationarity Adam's
   second-moment estimate has to absorb, on top of the real nonstationarity of PPO.

Neither is fatal — a normalization constant only has to be approximately right — but both
are avoidable, and avoiding them decouples `num_minibatches` from the statistic entirely,
which is the real win: the minibatch count then gets chosen on optimization grounds alone.

## The proposal

Compute the per-design advantage mean and std **once per rollout dataset**, over all of
that design's rows, and hold them fixed for every gradient step taken against that
dataset.

Concretely, in `make_training_chunk`'s `training_step` (`shared.py`), after the rollout
scan and the normalizer update but before the `sgd_step` scan:

1. Run one forward pass of the value network over the whole grid at the current
   `training_state.params`, producing `baseline` and `bootstrap_value` on the
   `(M, K, C*S, T)` layout.
2. Call `compute_gae` on it exactly as the loss does.
3. Reduce to `mean` and `std` per cell: `advantages.reshape(M*K, -1)` then
   `mean(axis=1)`, `std(axis=1)`. With `cheetah6D_stratified.yaml` that is
   `128 rows x 20 steps = 2560` transitions per design, 8x what a `num_minibatches=8`
   minibatch sees, and ~128 effective samples rather than ~16.
4. Thread the resulting `(M*K,)` vector into the loss, e.g. as a new field on the batch or
   an extra argument to `sgd_step`/`batch_step`, and have the loss index it by the group
   axis instead of computing `mean`/`std` itself.

The batching already delivers the group structure, so step 4 is the only place the loss
changes: replace the two reductions with a gather.

## Why a stale statistic is acceptable

The scale is computed with the value network as of the start of the update phase, while
the advantages it divides are recomputed inside the loss against the *current* value
network — which has moved by up to `num_updates_per_batch * num_minibatches` Adam steps by
the last minibatch. So the divisor is slightly stale.

This is fine, and the reason is worth being explicit about. Advantage standardization is
not part of the PPO objective's definition; it is a preconditioner on the policy-gradient
step size. Its job is to stop a design's contribution from scaling with the magnitude of
its returns. Getting that scale right to within a few percent is enough — and a divisor
estimated from 2560 transitions at slightly stale parameters is *more* accurate for that
purpose than one estimated from 320 transitions at exactly current parameters, because
the estimation error dominates the staleness error. Brax's convention of recomputing GAE
inside the loss is about keeping the *advantage* on-policy, which this change does not
touch; only its scale is frozen.

If the staleness ever needs bounding, the cheap check is to log the ratio of the frozen
std to the std the loss would have computed, per design, and confirm it stays near 1
across the update passes.

## Cost

- One extra forward pass of the value network over the rollout dataset per
  `training_step`, plus one `compute_gae`. The rollout itself is `unroll_length *
  num_scans` env steps of MuJoCo across `num_parallel_envs`, which dominates by a wide
  margin; the added cost should be in the noise, but it is worth measuring
  `training/sps` before and after rather than assuming.
- The plumbing is the awkward part: the statistic has to reach `compute_design_hypernet_loss`
  through `sgd_step` -> `batch_step` -> `gradient_update_fn`, none of which currently carry
  anything but `(params, normalizer_params, data, key)`. The least invasive route is
  probably to attach it to the `Grid` (or to a small wrapper struct alongside it) so it
  rides through the existing `data` argument and gets sliced by `batch_fn` along with
  everything else — which would also make it work unchanged under any batching strategy.

## Related, not done

- **`mo_design_hypernetwork`** normalizes per objective (`axis=(0, 1)` over `[T, B]`,
  leaving the objective axis `M` its own statistics) but pools over designs, exactly as
  `design_hypernetwork` did before this change. `batching_strategy: stratified` is
  available to it — `make_sgd_step` is shared — but its loss has no `num_advantage_groups`,
  so setting it there buys the design-mixed gradient without the per-design scale. Wiring
  the grouping in means normalizing over `(design, objective)` pairs jointly, which needs
  a two-axis fold rather than the one-axis fold used here.
- **The value loss is still pooled.** `v_loss = mse(vs - baseline)` has no normalization at
  all, so under any design-mixing strategy the value head's gradient is dominated by
  high-return designs. This is the one thing `batching_strategy: design` was protecting
  that stratification does not restore. Whether it matters is an empirical question — the
  value head is only a baseline, and a biased-toward-large-returns fit may be acceptable —
  but if `eval/worst_design_reward` degrades relative to a `design`-batched run while
  `eval/best_design_reward` holds, this is the first place to look. The fix would mirror
  the advantage one: divide each design's value error by that design's own return scale.

## Suggested validation

`config/design_hypernetwork/cheetah6D_stratified.yaml` is the A/B partner for
`cheetah6D.yaml`: it differs only in `batching_strategy`, `num_minibatches` (32 -> 8) and
`batch_size` (128 -> 512). Their product is held at 4096 so `num_scans` stays 1 and
`env_step_per_training_step` stays 81,920, leaving both runs on an identical 29 x 421
epoch schedule — the partition of each rollout dataset is the only difference. Per rollout
that is 32 all-design Adam steps instead of 128 single-design ones, with each design's
advantage scale estimated from 16 rows x 20 steps = 320 transitions.

Compare `eval/worst_design_reward`, not the pooled `eval/episode_reward` — the latter hides
exactly the failure mode at issue, one design stalling while the others carry the mean.

Note that `cheetah6D_shuffle.yaml` is **not** a valid third arm: it also differs from
`cheetah6D.yaml` in `learning_rate` (1e-5 vs 1e-4), `discounting`, `clipping_epsilon`,
`resamples_per_epoch` and `initialization_strategy`. A clean `shuffle` arm means copying
`cheetah6D_stratified.yaml` and changing `batching_strategy` alone.
