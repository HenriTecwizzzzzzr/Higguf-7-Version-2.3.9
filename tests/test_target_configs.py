from pathlib import Path

import yaml

from higguf.model import HiggufConfig
from higguf.resources import estimate_resources


def test_4b_pilot_has_the_exact_4b_parameter_count() -> None:
    root = Path(__file__).parents[1]
    full = yaml.safe_load((root / "configs" / "higguf-4b.yaml").read_text(encoding="utf-8"))
    pilot = yaml.safe_load((root / "configs" / "higguf-4b-pilot.yaml").read_text(encoding="utf-8"))

    assert estimate_resources(HiggufConfig(**pilot["model"])).parameters == 3_960_655_872
    assert estimate_resources(HiggufConfig(**pilot["model"])).parameters == estimate_resources(HiggufConfig(**full["model"])).parameters
    assert pilot["model"]["max_sequence_length"] == 8
    assert pilot["training"]["checkpoint_every_steps"] == 1
