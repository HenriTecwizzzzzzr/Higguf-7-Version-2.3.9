"""Train Higguf#7's tokenizer exclusively on the supplied corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=65536)
    arguments = parser.parse_args()
    files = sorted(str(path) for path in Path(arguments.input).rglob("*.txt"))
    if not files:
        raise SystemExit("No .txt files found in the input directory.")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=arguments.vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files, trainer)
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination / "tokenizer.json"))
    print(f"Saved a {tokenizer.get_vocab_size():,}-token vocabulary to {destination}")


if __name__ == "__main__":
    main()
