"""Download the WikiText-103 training split with an auditable provenance record.

WikiText is Wikipedia-derived. Its dataset card declares CC BY-SA 3.0; use of
the downloaded corpus is intentionally blocked until a named reviewer accepts
that licence's attribution and share-alike obligations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from datasets import load_dataset

SOURCE_NAME = "Salesforce WikiText-103 raw training split"
SOURCE_URL = "https://huggingface.co/datasets/Salesforce/wikitext"
LICENSE = "CC-BY-SA-3.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-by", required=True, help="Name of the person accepting the source licence.")
    parser.add_argument("--output", default="data/raw/wikitext-103-train.txt")
    parser.add_argument("--manifest", default="data/corpus-manifest.jsonl")
    arguments = parser.parse_args()
    destination = Path(arguments.output)
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {destination}")

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written_characters = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for row in dataset:
            text = row["text"].strip()
            if text:
                output.write(text + "\n")
                written_characters += len(text) + 1
    record = {
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "retrieved_on": date.today().isoformat(),
        "sha256": sha256(destination),
        "language": "en",
        "rights_verified_by": arguments.approved_by,
        "notes": f"{written_characters:,} UTF-8 characters; source licence approved for this pilot.",
    }
    manifest = Path(arguments.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Downloaded {written_characters:,} characters to {destination}")
    print(f"Recorded SHA-256 provenance in {manifest}")


if __name__ == "__main__":
    main()
