"""Experimental mixture-of-experts components for future Higguf#7 research.

These components are not used by the current dense 2B pilot. They are a
from-scratch implementation for experiments with compatible experts trained
under one router; they do not merge unrelated pretrained models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class TopKRouter(nn.Module):
    """Route each token to the highest-scoring compatible experts."""

    def __init__(self, hidden_size: int, num_experts: int, experts_per_token: int) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if not 1 <= experts_per_token <= num_experts:
            raise ValueError("experts_per_token must be between 1 and num_experts")
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = torch.softmax(self.gate(tokens), dim=-1)
        weights, indices = torch.topk(probabilities, self.experts_per_token, dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True), indices

    def load_balance_loss(self, tokens: Tensor) -> Tensor:
        """Auxiliary Switch-style loss that discourages router collapse."""
        probabilities = torch.softmax(self.gate(tokens), dim=-1)
        assignments = functional.one_hot(probabilities.argmax(dim=-1), self.num_experts).to(probabilities.dtype)
        mean_probability = probabilities.mean(dim=0)
        mean_assignment = assignments.mean(dim=0)
        return self.num_experts * torch.sum(mean_probability * mean_assignment)


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.down_proj(functional.silu(self.gate_proj(tokens)) * self.up_proj(tokens))


class MixtureOfExpertsMLP(nn.Module):
    """Sparse SwiGLU experts with learned top-k token routing."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, experts_per_token: int) -> None:
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, experts_per_token)
        self.experts = nn.ModuleList(ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts))

    def forward(self, tokens: Tensor) -> Tensor:
        original_shape = tokens.shape
        flattened = tokens.reshape(-1, original_shape[-1])
        weights, indices = self.router(flattened)
        output = torch.zeros_like(flattened)
        for expert_number, expert in enumerate(self.experts):
            for route_number in range(self.router.experts_per_token):
                selected = indices[:, route_number] == expert_number
                if selected.any():
                    expert_output = expert(flattened[selected])
                    output[selected] += weights[selected, route_number].unsqueeze(-1) * expert_output
        return output.reshape(original_shape)
