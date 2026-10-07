"""ROTATE: Rotation-Optimized Token Alignment in weighT spacE.

The core algorithm of the paper, with no I/O, model loading or cluster logic.

Given a neuron weight vector w (d_model,) and a vocabulary projection matrix
U (d_model, vocab), ROTATE finds a sequence of *vocabulary channels*: unit-norm
directions close to w whose vocabulary projections are sparse (high kurtosis).

One channel is found per iteration:

  1. Parametrise a rotation R as a Householder reflection and optimise
         loss = -lambda_k * signed_log(kurtosis((R w) U * token_weights))
                + lambda_f * (1 - cos(w, R w))
     with AdamW. Every ``checkpoint_every`` steps the candidate R w is scored by the
     kurtosis of its *unweighted* projection; the best-scoring candidate is the channel.
  2. Deflate by masking: tokens in the channel's dominant tail (|z| > threshold on the
     side given by its skewness) get a small weight, so the next iteration is pushed
     towards a different part of the vocabulary. The neuron vector itself is unchanged.

Everything is batched over neurons: ``neuron_vecs`` is (N, d_model).
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def tensor_kurtosis_differentiable(tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Excess kurtosis (population moments) along ``axis``; differentiable."""
    mean = torch.mean(tensor, dim=axis, keepdim=True)
    std = torch.std(tensor, dim=axis, keepdim=True, unbiased=False)
    std = torch.clamp(std, min=1e-8)
    centered_tensor = tensor - mean
    mean_fourth_power = torch.mean(centered_tensor ** 4, dim=axis)
    std_fourth_power = std.squeeze(axis) ** 4
    return (mean_fourth_power / std_fourth_power) - 3


kurtosis = tensor_kurtosis_differentiable


def skewness(x: torch.Tensor, axis: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Skewness (population moments) along ``axis``."""
    mean = torch.mean(x, dim=axis, keepdim=True)
    std = torch.std(x, dim=axis, keepdim=True, unbiased=False).clamp_min(eps)
    return torch.mean(((x - mean) / std) ** 3, dim=axis)


def signed_log(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(|x| + 1): keeps the kurtosis gradient well scaled across neurons."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


# ---------------------------------------------------------------------------
# Rotation parametrisation
# ---------------------------------------------------------------------------

class BatchedOrthonormalRotations(nn.Module):
    """One product of ``k`` Householder reflections per neuron.

    A reflection H = I - 2 v v^T / |v|^2 is orthonormal for any v, so optimising v
    explores norm-preserving transformations of the neuron vector without constraints.
    The paper uses k = 1.
    """

    def __init__(self, num_neurons: int, hidden_size: int, k: int = 1, init_scale: float = 1e-7) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.hidden_size = hidden_size
        self.k = k
        self.householder_vectors = nn.Parameter(torch.randn(num_neurons, k, hidden_size) * init_scale)

    def forward(self, neuron_vecs: torch.Tensor) -> torch.Tensor:
        """Apply each neuron's reflections: (N, d) -> (N, d)."""
        eps = 1e-12
        y = neuron_vecs.unsqueeze(1)  # [N, 1, d]
        v = self.householder_vectors  # [N, k, d]
        for i in range(self.k):
            v_i = v[:, i:i + 1, :]
            v_norm_sq = torch.sum(v_i * v_i, dim=2, keepdim=True)
            proj = torch.sum(y * v_i, dim=2, keepdim=True)
            y = y - 2.0 * proj * v_i / (v_norm_sq + eps)
        return y.squeeze(1)


# ---------------------------------------------------------------------------
# Token weighting and masking
# ---------------------------------------------------------------------------

def deweight_high_norm_tokens(
    U: torch.Tensor,
    token_weights: torch.Tensor,
    value: float = 1e-4,
    percentile: float = 0.99,
) -> tuple[torch.Tensor, int]:
    """Give ``value`` weight to tokens whose column of U has norm above ``percentile``."""
    token_weights = token_weights.clone()
    token_norms = torch.norm(U, dim=0).float().cpu()
    high_norm = token_norms > torch.quantile(token_norms, percentile)
    token_weights[high_norm] = value
    return token_weights, int(high_norm.sum().item())


def tail_mask(logits: torch.Tensor, std_threshold: float, small_skew_eps: float) -> torch.Tensor:
    """Boolean mask of a channel's extreme tokens.

    Tokens with z-score beyond ``std_threshold`` on the tail indicated by the sign of the
    skewness; both tails if |skewness| <= ``small_skew_eps``. ``logits`` is (V,).
    """
    mean = torch.mean(logits)
    std = torch.std(logits, unbiased=False).clamp_min(1e-8)
    z = (logits - mean) / std
    skew = torch.mean(z ** 3).item()
    if abs(skew) <= small_skew_eps:
        return torch.abs(z) > std_threshold
    if skew > 0:
        return z > std_threshold
    return z < -std_threshold


@torch.no_grad()
def alpha_grid_search_batched(
    x_vocab: torch.Tensor,
    y_vocab: torch.Tensor,
    *,
    steps: int = 1000,
    std_threshold: float = 7.0,
    small_skew_eps: float = 0.05,
    avg_lambda: float = 5.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtraction-based deflation (used by the ``subtraction`` ablation, not the paper method).

    Grid-search alpha in [0, 1] for residual logits r = x - alpha * y, minimising
    kurtosis(r) + avg_lambda * mean(|r| over y's extreme tokens). alpha > 0 is returned
    only if it lowers the kurtosis of x. Inputs are (N, V); returns (alpha, objective), each (N,).
    """
    device = x_vocab.device
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device)
    base_kurtosis = tensor_kurtosis_differentiable(x_vocab, axis=1)

    y_mean = torch.mean(y_vocab, dim=1, keepdim=True)
    y_std = torch.std(y_vocab, dim=1, keepdim=True, unbiased=False).clamp_min(epsilon)
    y_z = (y_vocab - y_mean) / y_std
    y_skew = torch.mean(y_z ** 3, dim=1)

    extremes_mask = torch.zeros_like(y_vocab, dtype=torch.bool)
    neutral_rows = (torch.abs(y_skew) <= small_skew_eps).nonzero(as_tuple=False).view(-1)
    if neutral_rows.numel() > 0:
        extremes_mask[neutral_rows] = torch.abs(y_z[neutral_rows]) >= std_threshold
    pos_rows = (y_skew > small_skew_eps).nonzero(as_tuple=False).view(-1)
    if pos_rows.numel() > 0:
        extremes_mask[pos_rows] = y_z[pos_rows] >= std_threshold
    neg_rows = (y_skew < -small_skew_eps).nonzero(as_tuple=False).view(-1)
    if neg_rows.numel() > 0:
        extremes_mask[neg_rows] = y_z[neg_rows] <= -std_threshold

    marked_count = extremes_mask.sum(dim=1, keepdim=True).clamp_min(1)
    x_marked = torch.where(extremes_mask, torch.abs(x_vocab), torch.zeros_like(x_vocab))
    base_objective = base_kurtosis + avg_lambda * x_marked.sum(dim=1) / marked_count.squeeze(1)

    residuals = x_vocab.unsqueeze(0) - alphas.view(-1, 1, 1) * y_vocab.unsqueeze(0)  # [G, N, V]
    kurtosis_vals = tensor_kurtosis_differentiable(residuals, axis=2)  # [G, N]
    residuals_marked = torch.where(extremes_mask.unsqueeze(0), torch.abs(residuals), torch.zeros_like(residuals))
    avg_marked_vals = residuals_marked.sum(dim=2) / marked_count.squeeze(1).unsqueeze(0)
    objectives = kurtosis_vals + avg_lambda * avg_marked_vals

    best_idx = torch.argmin(objectives, dim=0)
    best_objective = objectives.gather(0, best_idx.view(1, -1)).squeeze(0)
    best_kurt = kurtosis_vals.gather(0, best_idx.view(1, -1)).squeeze(0)
    best_alpha = alphas[best_idx]

    no_improvement = (best_alpha > 0) & (best_kurt >= base_kurtosis)
    best_alpha = torch.where(no_improvement, torch.zeros_like(best_alpha), best_alpha)
    best_objective = torch.where(no_improvement, base_objective, best_objective)
    return best_alpha, best_objective


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

@dataclass
class PursuitConfig:
    """Hyperparameters of ROTATE. Defaults are the paper's (Gemma); Llama uses mask_std_threshold=6."""

    num_channels: int = 50          # iterations = channels found per neuron
    max_steps: int = 10_000         # optimisation steps per channel
    learning_rate: float = 8e-4
    warmup_steps: int = 100         # linear LR warm-up
    weight_decay: float = 0.01
    kurtosis_lambda: float = 0.3
    faithfulness_lambda: float = 1.0
    init_scale: float = 1.0         # std of the initial Householder vector
    k: int = 1                      # Householder reflections per rotation
    checkpoint_every: int = 100     # steps between candidate evaluations
    mask_std_threshold: float = 4.0
    mask_small_skew_eps: float = 10.0
    masked_token_weight: float = 1e-2
    # Stop a batch once every neuron's loss has plateaued (std over a window).
    convergence_window: int = 100
    convergence_std: float = 1e-4
    convergence_min_steps: int = 2000
    use_amp: Optional[bool] = None  # None: enabled on CUDA
    seed: Optional[int] = None


@dataclass
class ChannelStep:
    """Result of one ROTATE iteration for a batch of N neurons."""

    channels: torch.Tensor          # (N, d_model) best rotated vector per neuron
    kurtosis: torch.Tensor          # (N,) unweighted vocab kurtosis of each channel
    best_step: torch.Tensor         # (N,) step at which the channel was found
    masked_tokens: torch.Tensor     # (N,) number of tokens masked after this channel
    num_steps: int                  # optimisation steps actually run
    loss_history: list = field(default_factory=list)


class _ConvergenceTracker:
    """Per-neuron loss plateau detection over a sliding window."""

    def __init__(self, num_neurons: int, window: int, std_threshold: float, min_steps: int, device):
        self.window, self.std_threshold, self.min_steps = window, std_threshold, min_steps
        self.step = 0
        self.buffer = torch.zeros(num_neurons, window, device=device)
        self.filled = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.converged = torch.zeros(num_neurons, dtype=torch.bool, device=device)

    def update(self, losses: torch.Tensor) -> bool:
        """Record one step of per-neuron losses; return True once all neurons converged."""
        self.buffer[:, self.step % self.window] = losses
        self.step += 1
        self.filled = torch.clamp(self.filled + 1, max=self.window)
        if self.step >= self.min_steps:
            active = ~self.converged & (self.filled >= self.window)
            if active.any():
                window_std = torch.std(self.buffer[active], dim=1, unbiased=False)
                self.converged[active] = window_std < self.std_threshold
        return bool(self.converged.all())


def find_channel(
    neuron_vecs: torch.Tensor,
    U: torch.Tensor,
    token_weights: torch.Tensor,
    cfg: PursuitConfig,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
) -> ChannelStep:
    """Run one ROTATE iteration: optimise one rotation per neuron and pick the best candidate.

    Args:
        neuron_vecs: (N, d_model) neuron weight vectors.
        U: (d_model, V) vocabulary projection matrix (W_U, or W_E.T for early Llama layers).
        token_weights: (N, V) per-neuron token weights for the kurtosis objective.
        cfg: hyperparameters.
        callback: optional ``callback(step, per_neuron_loss, weighted_kurtosis)`` for logging.
    """
    device = neuron_vecs.device
    n = neuron_vecs.shape[0]
    use_amp = (device.type == "cuda") if cfg.use_amp is None else cfg.use_amp

    rotations = BatchedOrthonormalRotations(n, neuron_vecs.shape[1], k=cfg.k, init_scale=cfg.init_scale).to(device)
    optimizer = optim.AdamW(
        rotations.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.weight_decay
    )
    warmup = cfg.warmup_steps
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: float(s) / float(max(1, warmup)) if warmup > 0 and s < warmup else 1.0
    )
    scaler = GradScaler("cuda") if use_amp else None
    amp_ctx = autocast("cuda") if use_amp else nullcontext()
    tracker = _ConvergenceTracker(n, cfg.convergence_window, cfg.convergence_std, cfg.convergence_min_steps, device)

    best_abs_kurt = torch.zeros(n)
    best_kurt = torch.zeros(n)
    best_step = torch.full((n,), -1, dtype=torch.long)
    best_vecs: list[Optional[torch.Tensor]] = [None] * n
    history = []
    neuron_norm = torch.norm(neuron_vecs, p=2, dim=1)

    step = 0
    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            rotated = rotations(neuron_vecs)
            weighted_vocab = torch.matmul(rotated, U) * token_weights
            weighted_kurt = tensor_kurtosis_differentiable(weighted_vocab, axis=1)
            kurtosis_loss = -cfg.kurtosis_lambda * signed_log(weighted_kurt)
            cos_sim = torch.sum(neuron_vecs * rotated, dim=1) / (neuron_norm * torch.norm(rotated, p=2, dim=1) + 1e-8)
            faithfulness_loss = cfg.faithfulness_lambda * (1 - cos_sim)
            per_neuron_loss = kurtosis_loss + faithfulness_loss
            loss = per_neuron_loss.mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        if callback is not None:
            callback(step, per_neuron_loss.detach(), weighted_kurt.detach())
        if tracker.update(per_neuron_loss.detach()):
            break

        if step % cfg.checkpoint_every == 0:
            with torch.no_grad():
                history.append(float(loss.item()))
                cand_kurt = tensor_kurtosis_differentiable(torch.matmul(rotated, U), axis=1).cpu()
                for i in range(n):
                    value = float(cand_kurt[i])
                    if abs(value) > best_abs_kurt[i]:
                        best_abs_kurt[i] = abs(value)
                        best_kurt[i] = value
                        best_step[i] = step
                        best_vecs[i] = rotated[i].detach().cpu()

    with torch.no_grad():
        # Fallback (no candidate ever had non-zero kurtosis): the last evaluated rotation.
        final = rotated.detach().cpu()
        channels = torch.stack([v if v is not None else final[i] for i, v in enumerate(best_vecs)])
        best_step[best_step < 0] = step

    return ChannelStep(
        channels=channels,
        kurtosis=best_kurt,
        best_step=best_step,
        masked_tokens=torch.zeros(n, dtype=torch.long),
        num_steps=step + 1,
        loss_history=history,
    )


def mask_channel_tokens(
    channels: torch.Tensor, U: torch.Tensor, token_weights: torch.Tensor, cfg: PursuitConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deflation: down-weight each channel's extreme tokens in its neuron's token weights.

    Returns the updated (N, V) weights and the (N,) number of masked tokens.
    """
    token_weights = token_weights.clone()
    counts = torch.zeros(channels.shape[0], dtype=torch.long)
    with torch.no_grad():
        for i in range(channels.shape[0]):
            logits = torch.matmul(channels[i].to(U.device), U)
            mask = tail_mask(logits, cfg.mask_std_threshold, cfg.mask_small_skew_eps)
            token_weights[i, mask.to(token_weights.device)] = cfg.masked_token_weight
            counts[i] = int(mask.sum())
    return token_weights, counts


def find_channels(
    neuron_vecs: torch.Tensor,
    U: torch.Tensor,
    token_weights: Optional[torch.Tensor] = None,
    cfg: Optional[PursuitConfig] = None,
    progress: bool = True,
    callback: Optional[Callable[[int, int, torch.Tensor, torch.Tensor], None]] = None,
) -> dict:
    """Decompose neurons into ``cfg.num_channels`` vocabulary channels each.

    Args:
        neuron_vecs: (N, d_model) or (d_model,) neuron weight vectors.
        U: (d_model, V) vocabulary projection matrix, on the device to run on.
        token_weights: (V,) or (N, V) initial token weights (default: all ones). See
            ``rotate.models.default_token_weights`` for the paper's choice.
        cfg: hyperparameters (default: ``PursuitConfig()``).
        progress: show a tqdm bar over channels.
        callback: optional ``callback(channel_idx, step, per_neuron_loss, weighted_kurtosis)``.

    Returns:
        dict with
          ``channels``  (N, C, d_model) channel vectors, in discovery order,
          ``kurtosis``  (N, C) unweighted vocab kurtosis of each channel,
          ``masked_tokens`` (N, C), ``best_step`` (N, C), ``num_steps`` list[int],
          ``token_weights`` (N, V) final weights.
    """
    cfg = cfg or PursuitConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    squeeze = neuron_vecs.dim() == 1
    neuron_vecs = neuron_vecs.reshape(-1, neuron_vecs.shape[-1]).to(U.device, torch.float32)
    U = U.float()
    n, vocab = neuron_vecs.shape[0], U.shape[1]

    if token_weights is None:
        token_weights = torch.ones(vocab)
    if token_weights.dim() == 1:
        token_weights = token_weights.unsqueeze(0).expand(n, -1)
    token_weights = token_weights.to(U.device, torch.float32).clone()

    iterator = range(cfg.num_channels)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="channels")

    steps: list[ChannelStep] = []
    for c in iterator:
        cb = (lambda s, l, k, _c=c: callback(_c, s, l, k)) if callback is not None else None
        result = find_channel(neuron_vecs, U, token_weights, cfg, callback=cb)
        token_weights, result.masked_tokens = mask_channel_tokens(result.channels, U, token_weights, cfg)
        steps.append(result)

    out = {
        "channels": torch.stack([s.channels for s in steps], dim=1),
        "kurtosis": torch.stack([s.kurtosis for s in steps], dim=1),
        "masked_tokens": torch.stack([s.masked_tokens for s in steps], dim=1),
        "best_step": torch.stack([s.best_step for s in steps], dim=1),
        "num_steps": [s.num_steps for s in steps],
        "token_weights": token_weights.cpu(),
    }
    if squeeze:
        for key in ("channels", "kurtosis", "masked_tokens", "best_step", "token_weights"):
            out[key] = out[key][0]
    return out
