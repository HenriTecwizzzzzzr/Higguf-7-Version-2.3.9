import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_training.py"
SPEC = importlib.util.spec_from_file_location("preflight_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_parameter_count_for_tiny_tied_model() -> None:
    model = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    }
    assert preflight.parameter_count(model) == 20_640


def test_parameter_count_rejects_incompatible_heads() -> None:
    model = {
        "vocab_size": 64,
        "hidden_size": 30,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    try:
        preflight.parameter_count(model)
    except ValueError as error:
        assert "divide evenly" in str(error)
    else:
        raise AssertionError("expected incompatible attention heads to fail")
