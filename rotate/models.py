"""Model registry, weight access and the vocabulary projection used by ROTATE.

Weights are read straight from the Hugging Face safetensors checkpoint (see
:mod:`rotate.weights`), so no model is instantiated: only the requested neurons and the
(un)embedding matrix are loaded. The conventions match TransformerLens, which the paper used:

    W_gate[:, i] = gate_proj.weight[i]     W_in[:, i] = up_proj.weight[i]
    W_out[i, :]  = down_proj.weight[:, i]  W_U = lm_head.weight.T   W_E = embed_tokens.weight
"""

from functools import lru_cache
from typing import Optional, Sequence

import torch

from .weights import Checkpoint

MODEL_CONFIGS = {
    "gemma": {
        "hf_name": "google/gemma-2-2b-it",
        "results_subdir": "google_gemma-2-2b-it",
        "n_layers": 26,
        "d_model": 2304,
        "d_mlp": 9216,
        # Layers <= this index are projected with W_E.T instead of W_U (None = always W_U).
        "use_embedding_up_to_layer": None,
        # Hyperparameters used for the paper runs (see docs/method.md).
        "pursuit": {"mask_std_threshold": 4.0},
    },
    "llama": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "results_subdir": "meta-llama_Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "d_model": 4096,
        "d_mlp": 14336,
        "use_embedding_up_to_layer": 12,
        "pursuit": {"mask_std_threshold": 6.0},
    },
}

MLP_TYPES = ("gate", "in", "out")
_MLP_PROJ = {"gate": "gate_proj", "in": "up_proj", "out": "down_proj"}


def resolve_model_key(model: str) -> str:
    """Accept either a short key ('gemma') or a HF name ('google/gemma-2-2b-it')."""
    if model in MODEL_CONFIGS:
        return model
    for key, cfg in MODEL_CONFIGS.items():
        if cfg["hf_name"] == model:
            return key
    raise KeyError(f"Unknown model {model!r}; expected one of {list(MODEL_CONFIGS)}")


def uses_embedding_matrix(model: str, layer: int) -> bool:
    """Whether a layer's neurons are read in vocabulary space through W_E.T rather than W_U.

    Early Llama layers are better aligned with the (untied) input embedding than with
    the unembedding, so layers <= 12 use W_E.T. Gemma ties the two matrices.
    """
    threshold = MODEL_CONFIGS[resolve_model_key(model)]["use_embedding_up_to_layer"]
    return threshold is not None and layer <= threshold


@lru_cache(maxsize=4)
def checkpoint(model: str, source: Optional[str] = None) -> Checkpoint:
    """The safetensors checkpoint of ``model``: its HF repo, or ``source`` (a repo id or local dir)."""
    return Checkpoint(source or MODEL_CONFIGS[resolve_model_key(model)]["hf_name"])


def load_tokenizer(model: str, source: Optional[str] = None):
    """The model's Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source or MODEL_CONFIGS[resolve_model_key(model)]["hf_name"])


def projection_matrix(
    model: str,
    layer: int,
    device="cpu",
    dtype: torch.dtype = torch.float32,
    source: Optional[str] = None,
) -> torch.Tensor:
    """The (d_model, vocab) matrix that maps a neuron weight vector to vocabulary logits."""
    ckpt = checkpoint(model, source)
    if uses_embedding_matrix(model, layer) or "lm_head.weight" not in ckpt:  # tied: W_U = W_E.T
        name = "model.embed_tokens.weight"
    else:
        name = "lm_head.weight"
    return ckpt.get(name).to(device=device, dtype=dtype).T


def get_neurons(
    model: str,
    layer: int,
    neuron_indices: Sequence[int],
    mlp_type: str,
    dtype: torch.dtype = torch.float32,
    source: Optional[str] = None,
) -> torch.Tensor:
    """Stacked (N, d_model) weight vectors of several neurons in W_gate, W_in or W_out."""
    if mlp_type not in MLP_TYPES:
        raise ValueError(f"Invalid mlp_type {mlp_type!r}; expected one of {MLP_TYPES}")
    name = f"model.layers.{layer}.mlp.{_MLP_PROJ[mlp_type]}.weight"
    indices = [int(i) for i in neuron_indices]
    ckpt = checkpoint(model, source)
    if mlp_type == "out":  # down_proj is (d_model, d_mlp): neurons are columns
        vecs = ckpt.get(name).T[indices]
    else:  # gate_proj / up_proj are (d_mlp, d_model): neurons are rows
        vecs = ckpt.get(name, rows=indices)
    return vecs.to(dtype)


def get_neuron(
    model: str,
    layer: int,
    neuron_idx: int,
    mlp_type: str,
    dtype: torch.dtype = torch.float32,
    source: Optional[str] = None,
) -> torch.Tensor:
    """The (d_model,) weight vector of one neuron in W_gate, W_in or W_out."""
    return get_neurons(model, layer, [neuron_idx], mlp_type, dtype=dtype, source=source)[0]


def load_outlier_token_ids(path=None) -> torch.Tensor:
    """Gemma-2 token ids excluded from the kurtosis objective (special, unused and rare tokens)."""
    from .paths import GEMMA_OUTLIER_TOKENS

    return torch.load(path or GEMMA_OUTLIER_TOKENS, map_location="cpu")


def default_token_weights(model: str, U: torch.Tensor, outlier_tokens_path=None) -> torch.Tensor:
    """Per-token weights for the kurtosis objective, as used in the paper.

    - Gemma: 5,295 special / unused / rare tokens (rotate/assets/gemma_outlier_token_ids.pt) get weight 0.
    - Both models: tokens whose projection column norm is above the 99th percentile get
      weight 1e-4, so a handful of high-norm tokens cannot dominate the kurtosis.
    """
    from .pursuit import deweight_high_norm_tokens

    vocab_size = U.shape[1]
    weights = torch.ones(vocab_size, dtype=torch.float32)
    if resolve_model_key(model) == "gemma":
        outliers = load_outlier_token_ids(outlier_tokens_path)
        outliers = outliers[outliers < vocab_size]
        weights[outliers] = 0.0
    weights, _ = deweight_high_norm_tokens(U, weights, value=1e-4, percentile=0.99)
    return weights
