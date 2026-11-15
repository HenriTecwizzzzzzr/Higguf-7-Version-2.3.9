from higguf.model import HiggufConfig
from higguf.resources import estimate_resources


def test_6b_config_size_estimate() -> None:
    estimate = estimate_resources(
        HiggufConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=12288,
            num_layers=30,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_sequence_length=2048,
        )
    )
    assert estimate.parameters == 5_919_461_376
    assert 11.0 < estimate.bf16_weights_gib < 11.1
    assert 88.0 < estimate.adamw_state_gib < 88.3
