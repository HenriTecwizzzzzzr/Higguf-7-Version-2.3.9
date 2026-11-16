import torch

from higguf.moe import MixtureOfExpertsMLP, TopKRouter


def test_router_selects_normalized_top_k_experts() -> None:
    router = TopKRouter(hidden_size=8, num_experts=4, experts_per_token=2)
    weights, indices = router(torch.randn(3, 8))
    assert weights.shape == (3, 2)
    assert indices.shape == (3, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))


def test_moe_runs_forward_and_backward() -> None:
    module = MixtureOfExpertsMLP(hidden_size=8, intermediate_size=16, num_experts=3, experts_per_token=2)
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    result = module(tokens)
    assert result.shape == tokens.shape
    result.square().mean().backward()
    assert tokens.grad is not None
