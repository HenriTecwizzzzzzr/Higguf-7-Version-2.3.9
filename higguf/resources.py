"""Static model-size and training-memory estimates for Higguf#7."""

from __future__ import annotations

from dataclasses import dataclass

from .model import HiggufConfig


@dataclass(frozen=True)
class ResourceEstimate:
    parameters: int
    bf16_weights_gib: float
    adamw_state_gib: float


def estimate_resources(config: HiggufConfig) -> ResourceEstimate:
    """Estimate parameter count and minimum AdamW state, without allocating weights."""
    head_dim = config.hidden_size // config.num_attention_heads
    kv_width = config.num_key_value_heads * head_dim
    embedding = config.vocab_size * config.hidden_size
    attention = 2 * config.hidden_size**2 + 2 * config.hidden_size * kv_width
    feed_forward = 3 * config.hidden_size * config.intermediate_size
    layer_norms = 2 * config.hidden_size
    final_norm = config.hidden_size
    output = 0 if config.tie_word_embeddings else embedding
    per_layer = attention + feed_forward + layer_norms
    parameters = embedding + config.num_layers * per_layer + final_norm + output
    gib = 2**30
    return ResourceEstimate(
        parameters=parameters,
        bf16_weights_gib=parameters * 2 / gib,
        adamw_state_gib=parameters * 16 / gib,
    )
