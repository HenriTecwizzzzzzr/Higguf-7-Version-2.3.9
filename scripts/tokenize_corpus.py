"""Tokenize a corpus into a contiguous uint32 stream for Higguf#7 training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(Path(arguments.tokenizer) / "tokenizer.json"))
    input_files = sorted(Path(arguments.input).rglob("*.txt"))
    if not input_files:
        raise SystemExit("No .txt files found in the input directory.")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "train.bin"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing token file: {destination}")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise SystemExit("Tokenizer does not define the required <eos> token.")
    token_count = 0
    with destination.open("wb") as encoded_output:
        for file in input_files:
            with file.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    ids = tokenizer.encode(line).ids + [eos_id]
                    np.asarray(ids, dtype=np.uint32).tofile(encoded_output)
                    token_count += len(ids)
    print(f"Wrote {token_count:,} tokens to {destination}")


if __name__ == "__main__":
    main()
