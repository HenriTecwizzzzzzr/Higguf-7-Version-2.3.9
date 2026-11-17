import torch

from higguf.chat import sample_next_token


def test_greedy_sampling_returns_highest_logit() -> None:
    assert sample_next_token(torch.tensor([0.0, 2.0, 1.0]), temperature=0, top_k=0) == 1


def test_top_k_sampling_stays_within_candidates() -> None:
    torch.manual_seed(7)
    assert sample_next_token(torch.tensor([0.0, 10.0, 1.0]), temperature=1.0, top_k=1) == 1
