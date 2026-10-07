"""ROTATE: disentangling MLP neuron weights into vocabulary channels.

Code for "Disentangling MLP Neuron Weights in Vocabulary Space"
(Avrahamy, Gur-Arieh, Geva, 2026; arXiv:2604.06005).

The method itself lives in :mod:`rotate.pursuit`; :mod:`rotate.models` reads a neuron's
weights and the vocabulary projection matrix straight from a Hugging Face checkpoint.
"""

__version__ = "1.0.0"

from .pursuit import (  # noqa: F401
    PursuitConfig,
    find_channel,
    find_channels,
    kurtosis,
    skewness,
)

__all__ = ["PursuitConfig", "find_channel", "find_channels", "kurtosis", "skewness"]
