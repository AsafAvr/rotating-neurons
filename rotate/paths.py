"""Filesystem locations.

    ROTATE_RESULTS   Where scripts/train_channels.py writes channels (default: <repo>/results),
                     one sub-directory per model, e.g. results/google_gemma-2-2b-it/layer_18/...
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = Path(os.environ.get("ROTATE_RESULTS") or REPO_ROOT / "results").expanduser()
DATA_DIR = REPO_ROOT / "data"

# 5,295 Gemma-2 token ids excluded from the kurtosis objective: special and <unused*>
# tokens plus rare, likely under-trained tokens (e.g. 'TestingModule', private-use glyphs)
# that would otherwise dominate vocabulary projections. Shipped inside the package.
GEMMA_OUTLIER_TOKENS = PACKAGE_DIR / "assets" / "gemma_outlier_token_ids.pt"


def model_dir(model: str) -> Path:
    """Results directory of one model, e.g. ``model_dir("gemma")``."""
    from .models import MODEL_CONFIGS, resolve_model_key

    return RESULTS_ROOT / MODEL_CONFIGS[resolve_model_key(model)]["results_subdir"]


def layer_dir(model: str, layer: int) -> Path:
    """Results directory of one layer of one model."""
    return model_dir(model) / f"layer_{layer}"
