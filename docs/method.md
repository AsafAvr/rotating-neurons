# The ROTATE algorithm

This page describes exactly what [`rotate/pursuit.py`](../rotate/pursuit.py) computes, with the
hyperparameters used for the paper. For motivation and results see the
[paper](https://arxiv.org/abs/2604.06005).

## Setup

A gated MLP neuron has three weight vectors in the residual space (`d_model`):
`w_gate` and `w_in` (columns of `W_gate`, `W_in`, what the neuron *reads*) and `w_out`
(row of `W_out`, what it *writes*). Each is handled independently; below `w` is any one of them.

The vocabulary projection of a vector `v` is `v @ U`, where `U` (`d_model x |V|`) is the
unembedding `W_U`. For Llama layers `<= 12` we use the input embedding `W_E.T` instead,
because early-layer neurons align better with the (untied) embedding. Gemma ties the two.

A vector is a good *vocabulary channel* if its projection is **sparse**, i.e. a few tokens
stand far out from the rest. We measure this with the excess kurtosis of the projection
(over the vocabulary).

## One channel

ROTATE looks for a unit-norm direction close to `w` whose projection has high kurtosis.
The direction is parametrised as a Householder reflection of the neuron itself,
`H(u) w = w - 2 u (u . w) / |u|^2`, so it always has the same norm as `w` and the
optimisation over `u` is unconstrained. The loss for a batch of neurons is the mean of

```
loss(u) = - lambda_k * signed_log( kurtosis( (H(u) w) @ U * m ) )
          + lambda_f * (1 - cos(w, H(u) w))
```

* `m` is a per-token weight vector (see below); `signed_log(x) = sign(x) log(1 + |x|)` keeps
  the gradient scale comparable across neurons whose kurtosis differs by orders of magnitude.
* The second term keeps the channel close to the neuron, so channels stay *part of* `w`.

Optimisation: AdamW (lr `8e-4`, weight decay `0.01`), linear warm-up over 100 steps,
mixed precision on GPU, up to 10,000 steps. Every 100 steps the current candidate
`H(u) w` is scored by the kurtosis of its *unweighted* projection `(H(u) w) @ U`; the
best-scoring candidate (largest |kurtosis|) is returned as the channel. A batch stops early
once every neuron's loss has plateaued (std below `1e-4` over a 100-step window, checked
after 2,000 steps).

## Many channels: deflation by masking

After a channel `c` is found, the tokens that define it are removed from the objective:
take the z-scores of `c @ U`; if the projection's skewness is positive mask tokens with
`z > t`, if negative mask `z < -t` (both tails if |skewness| `<= 10`). Masked tokens get
weight `m = 0.01` for this neuron in all later iterations. The neuron vector `w` itself is
**not** changed, so every channel is a rotation of the original neuron. Repeating this
`C` times gives `C` channels ordered by discovery.

`t` is 4 for Gemma and 6 for Llama.

## Token weights

Before the first iteration (`rotate.models.default_token_weights`):

* **Gemma:** 5,295 special, unused and rare tokens
  ([`rotate/assets/gemma_outlier_token_ids.pt`](../rotate/assets/gemma_outlier_token_ids.pt)) get weight 0.
* **Both models:** tokens whose column of `U` has norm above the 99th percentile get weight
  `1e-4`, so a handful of high-norm tokens cannot dominate every projection.

## Hyperparameters used in the paper

| | Gemma-2-2B-it | Llama-3.1-8B-Instruct |
|---|---|---|
| projection | `W_U` | `W_E.T` for layers <= 12, else `W_U` |
| channels per neuron | 50 | 50 (30 for layers 8, 16, 26) |
| learning rate | 8e-4 | 8e-4 (2e-3 for layer 22) |
| `lambda_k` / `lambda_f` | 0.3 / 1.0 | 0.3 / 1.0 |
| mask threshold `t` | 4 | 6 (8 for layer 22) |
| steps / checkpoint interval | 10,000 / 100 | 10,000 / 100 |
| neurons per layer | the 99 in `data/neuron_indices_gemma.txt` | the 100 in `data/neuron_indices_llama.txt` |

Per-run exceptions on Gemma: layer 18 `out` used `lambda_k = 0.5`, and layer 4 `out` used
lr `2e-3`. The `metadata.json` written next to every run records its exact settings.

## Relation to the original training script

`rotate/pursuit.py` is a clean re-implementation of the paper configuration of the
original `train_projection_pursuit.py` (its `--mask-only` mode with loss-convergence
stopping). On identical inputs and seeds the two produce bit-identical channels and token
weights. Options of the original script that did not affect the paper runs (alpha-search
subtraction, kurtosis-patience early stopping, cross-iteration neuron killing) were left
out; subtraction-based deflation (used in the paper's ablations) is kept as
`rotate.pursuit.alpha_grid_search_batched`.
