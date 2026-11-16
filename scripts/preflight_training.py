"""Check a Higguf#7 training run before allocating a GPU session.

This script intentionally does not allocate a model or start training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parameter_count(model: dict) -> int:
    """Return the tied-embedding parameter count without allocating weights."""
    hidden = model["hidden_size"]
    heads = model["num_attention_heads"]
    kv_heads = model["num_key_value_heads"]
    if hidden % heads:
        raise ValueError("hidden_size must divide evenly into num_attention_heads")
    if heads % kv_heads:
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
    embedding = model["vocab_size"] * hidden
    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    attention = 2 * hidden**2 + 2 * hidden * kv_width
    feed_forward = 3 * hidden * model["intermediate_size"]
    norms = 2 * hidden
    output = 0 if model.get("tie_word_embeddings", True) else embedding
    return embedding + model["num_layers"] * (attention + feed_forward + norms) + hidden + output


def token_count(path: Path) -> int:
    size = path.stat().st_size
    if size % 4:
        raise ValueError(f"{path} is not an unsigned-32-bit token file (size is not divisible by 4)")
    return size // 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--minimum-gpus", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.minimum_gpus < 1:
        raise SystemExit("--minimum-gpus must be at least 1")
    if not arguments.config.is_file():
        raise SystemExit(f"Configuration does not exist: {arguments.config}")
    settings = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or not {"model", "training", "data"} <= settings.keys():
        raise SystemExit("Configuration must contain model, training, and data sections")
    model, training, data = settings["model"], settings["training"], settings["data"]
    try:
        parameters = parameter_count(model)
        path = Path(data["train_path"])
        if not path.is_file():
            raise ValueError(f"token data does not exist: {path}")
        tokens = token_count(path)
        required = training["micro_batch_size"] * (model["max_sequence_length"] + 1)
        if tokens < required:
            raise ValueError(f"token data has {tokens:,} tokens; at least {required:,} are needed for one micro-batch")
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Preflight failed: {error}") from error
    print(f"Configuration: {arguments.config}")
    print(f"Parameters: {parameters:,}")
    print(f"Token data: {tokens:,} tokens at {path}")
    if arguments.require_cuda:
        try:
            import torch
        except ImportError as error:
            raise SystemExit("Preflight failed: PyTorch is required to check CUDA") from error
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available < arguments.minimum_gpus:
            raise SystemExit(f"Preflight failed: requires {arguments.minimum_gpus} CUDA GPU(s), found {available}")
        print(f"CUDA: {available} GPU(s) available")
    print("Preflight passed. This confirms setup only; it does not prove the model is trained or safe to release.")


if __name__ == "__main__":
    main()
