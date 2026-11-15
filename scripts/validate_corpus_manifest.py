from __future__ import annotations

"""Fail closed when Higguf#7 corpus provenance metadata is incomplete."""

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "source_name",
    "source_url",
    "license",
    "retrieved_on",
    "sha256",
    "language",
    "rights_verified_by",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/corpus-manifest.jsonl")
    arguments = parser.parse_args()
    manifest = Path(arguments.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest does not exist: {manifest}")
    errors: list[str] = []
    usable_rows = 0
    for number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: invalid JSON ({error.msg})")
            continue
        missing = sorted(field for field in REQUIRED_FIELDS if not isinstance(row.get(field), str) or not row[field].strip())
        if missing:
            errors.append(f"line {number}: missing {', '.join(missing)}")
        elif row["source_name"].startswith("EXAMPLE ONLY"):
            errors.append(f"line {number}: replace the example row with verified data")
        else:
            usable_rows += 1
    if errors:
        raise SystemExit("Corpus manifest rejected:\n- " + "\n- ".join(errors))
    if not usable_rows:
        raise SystemExit("Corpus manifest has no usable rows")
    print(f"Verified provenance metadata for {usable_rows} source(s).")


if __name__ == "__main__":
    main()
