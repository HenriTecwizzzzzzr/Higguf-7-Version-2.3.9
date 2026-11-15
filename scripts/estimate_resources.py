"""Print a no-allocation size estimate for a Higguf#7 YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from higguf.model import HiggufConfig
from higguf.resources import estimate_resources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    estimate = estimate_resources(HiggufConfig(**settings["model"]))
    print(f"Parameters: {estimate.parameters:,} ({estimate.parameters / 1e9:.3f}B)")
    print(f"BF16 weights: {estimate.bf16_weights_gib:.2f} GiB")
    print(f"AdamW state before activations: {estimate.adamw_state_gib:.2f} GiB")


if __name__ == "__main__":
    main()
