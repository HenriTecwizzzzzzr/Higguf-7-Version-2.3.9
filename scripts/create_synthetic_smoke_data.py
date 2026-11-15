"""Create deterministic fake token data to validate the training pipeline.

This script intentionally does not create a language corpus. It is only for
checking that random initialisation, batching, loss, backpropagation and
checkpointing execute end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/tokenized/train.bin")
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.tokens < 2 or arguments.vocab_size < 2:
        raise SystemExit("--tokens and --vocab-size must both be at least 2")
    generator = np.random.default_rng(arguments.seed)
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    generator.integers(0, arguments.vocab_size, size=arguments.tokens, dtype=np.uint32).tofile(destination)
    print(f"Wrote {arguments.tokens:,} synthetic tokens to {destination}")


if __name__ == "__main__":
    main()
