#!/usr/bin/env python
"""Find vocabulary channels for MLP neurons with ROTATE.

Runs :func:`rotate.pursuit.find_channels` over the neurons of one
(model, layer, mlp_type) in chunks and writes

    <output_dir>/results.pt      {neuron_idx: {"iterations": [{"rotated_vector_best": Tensor(d_model), ...}]}}
    <output_dir>/metadata.json   arguments, token weighting and timing

The default output directory is
    $ROTATE_RESULTS/<model>/layer_<L>/<timestamp>_mask_<mlp_type>_iter_<C>_lr_<lr>_kl_<kl>/

Only the neurons' weights and the (un)embedding matrix are read from the checkpoint; the model
itself is never loaded.

Examples:
    # A few neurons, quick look
    python scripts/train_channels.py --model gemma --layer 18 --mlp-type gate --indices 9005,127

    # The paper's neurons for one layer
    python scripts/train_channels.py --model gemma --layer 18 --mlp-type gate \\
        --indices-file data/neuron_indices_gemma.txt

    # Whole layer (one GPU, ~hours)
    python scripts/train_channels.py --model llama --layer 22 --mlp-type gate
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from rotate.models import (
    MODEL_CONFIGS,
    default_token_weights,
    get_neurons,
    projection_matrix,
    uses_embedding_matrix,
)
from rotate.paths import layer_dir
from rotate.pursuit import PursuitConfig, find_channels


def parse_indices(spec: str) -> list[int]:
    """'0-3,10' -> [0, 1, 2, 3, 10]; duplicates are dropped, order is kept."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--mlp-type", required=True, choices=["gate", "in", "out"])
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--indices", type=str, help="Neuron indices, e.g. '0-999,1200'")
    sel.add_argument("--indices-file", type=str, help="File with neuron indices (one per line or comma-separated)")
    sel.add_argument("--first-n", type=int, help="Use neurons 0..N-1")
    p.add_argument("--source", type=str, default=None,
                   help="Checkpoint to read: a HF repo id or a local directory (default: the model's HF repo)")
    p.add_argument("--output-dir", type=str, default=None, help="Default: see module docstring")
    p.add_argument("--chunk-size", type=int, default=1000, help="Neurons optimised together on the GPU")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cuda' or 'cpu'")
    p.add_argument("--resume", action="store_true", help="Skip neurons already in <output-dir>/results.pt")

    d = PursuitConfig()
    h = p.add_argument_group("ROTATE hyperparameters (defaults = paper)")
    h.add_argument("--num-channels", type=int, default=d.num_channels)
    h.add_argument("--max-steps", type=int, default=d.max_steps)
    h.add_argument("--learning-rate", type=float, default=d.learning_rate)
    h.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    h.add_argument("--kurtosis-lambda", type=float, default=d.kurtosis_lambda)
    h.add_argument("--faithfulness-lambda", type=float, default=d.faithfulness_lambda)
    h.add_argument("--init-scale", type=float, default=d.init_scale)
    h.add_argument("--k", type=int, default=d.k, help="Householder reflections per rotation")
    h.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    h.add_argument("--mask-std-threshold", type=float, default=None,
                   help="z-score threshold for masking a channel's tokens (default: 4 for Gemma, 6 for Llama)")
    h.add_argument("--mask-small-skew-eps", type=float, default=d.mask_small_skew_eps)
    h.add_argument("--masked-token-weight", type=float, default=d.masked_token_weight)
    h.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_cfg = MODEL_CONFIGS[args.model]
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PursuitConfig(
        num_channels=args.num_channels,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        kurtosis_lambda=args.kurtosis_lambda,
        faithfulness_lambda=args.faithfulness_lambda,
        init_scale=args.init_scale,
        k=args.k,
        checkpoint_every=args.checkpoint_every,
        mask_std_threshold=(args.mask_std_threshold if args.mask_std_threshold is not None
                            else model_cfg["pursuit"]["mask_std_threshold"]),
        mask_small_skew_eps=args.mask_small_skew_eps,
        masked_token_weight=args.masked_token_weight,
        seed=args.seed,
    )

    if args.first_n is not None:
        neurons = list(range(args.first_n))
    elif args.indices:
        neurons = parse_indices(args.indices)
    elif args.indices_file:
        neurons = parse_indices(Path(args.indices_file).read_text().replace("\n", ","))
    else:
        neurons = list(range(model_cfg["d_mlp"]))

    print(f"Reading {args.source or model_cfg['hf_name']} (layer {args.layer}, {args.mlp_type}) on {args.device} ...")
    U = projection_matrix(args.model, args.layer, device=args.device, source=args.source)
    proj_name = "W_E.T" if uses_embedding_matrix(args.model, args.layer) else "W_U"
    token_weights = default_token_weights(args.model, U)
    print(f"Projection: {proj_name} {tuple(U.shape)} | down-weighted tokens: {int((token_weights != 1).sum())}")

    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = layer_dir(args.model, args.layer) / (
            f"{ts}_mask_{args.mlp_type}_iter_{cfg.num_channels}"
            f"_lr_{cfg.learning_rate:.0e}_kl_{cfg.kurtosis_lambda:.0e}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.pt"

    results = {}
    if args.resume and results_path.exists():
        results = torch.load(results_path, map_location="cpu", weights_only=False)
        print(f"Resuming: {len(results)} neurons already done")
    todo = [n for n in neurons if n not in results]
    all_vecs = get_neurons(args.model, args.layer, todo, args.mlp_type, source=args.source) if todo else None

    metadata = {
        "model": model_cfg["hf_name"],
        "source": args.source or model_cfg["hf_name"],
        "layer": args.layer,
        "mlp_type": args.mlp_type,
        "projection": proj_name,
        "pursuit": asdict(cfg),
        "chunk_size": args.chunk_size,
        "num_neurons": len(neurons),
        "start_time": datetime.now().isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Output: {run_dir}  |  {len(todo)} neurons to process")

    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        t0 = time.time()
        vecs = all_vecs[start:start + args.chunk_size]
        out = find_channels(vecs, U, token_weights, cfg, progress=True)
        for j, neuron in enumerate(chunk):
            results[neuron] = {
                "iterations": [
                    {
                        "rotated_vector_best": out["channels"][j, c].clone(),
                        "rotated_kurtosis": float(out["kurtosis"][j, c]),
                        "best_epoch": int(out["best_step"][j, c]),
                        "residual_masked_token_count": int(out["masked_tokens"][j, c]),
                    }
                    for c in range(cfg.num_channels)
                ],
            }
        tmp = results_path.with_suffix(".pt.tmp")
        torch.save(results, tmp)
        os.replace(tmp, results_path)
        print(f"Neurons {chunk[0]}..{chunk[-1]} done in {time.time() - t0:.0f}s "
              f"({len(results)}/{len(neurons)}); saved {results_path}")

    metadata["end_time"] = datetime.now().isoformat()
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
