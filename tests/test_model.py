import torch

from higguf.model import HiggufConfig, HiggufForCausalLM


def test_tiny_model_runs_forward_and_backward() -> None:
    config = HiggufConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=16,
    )
    model = HiggufForCausalLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    result = model(tokens, labels=tokens)
    assert result["logits"].shape == (2, 16, config.vocab_size)
    assert result["loss"] is not None
    result["loss"].backward()
