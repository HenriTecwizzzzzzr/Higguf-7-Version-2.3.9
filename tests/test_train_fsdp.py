from pathlib import Path

import pytest
import torch
from torch import nn

from higguf.train_fsdp import initialise_meta_module, prune_rank_checkpoints


def test_prune_rank_checkpoints_keeps_recent_shards_for_its_rank(tmp_path: Path) -> None:
    for step in (1, 2, 3):
        (tmp_path / f"step-{step:08d}-rank-0.pt").write_bytes(b"checkpoint")
    (tmp_path / "step-00000001-rank-1.pt").write_bytes(b"other rank")

    prune_rank_checkpoints(tmp_path, rank=0, keep_last=2)

    assert not (tmp_path / "step-00000001-rank-0.pt").exists()
    assert (tmp_path / "step-00000002-rank-0.pt").exists()
    assert (tmp_path / "step-00000003-rank-0.pt").exists()
    assert (tmp_path / "step-00000001-rank-1.pt").exists()


def test_prune_rank_checkpoints_rejects_zero_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        prune_rank_checkpoints(tmp_path, rank=0, keep_last=0)


def test_meta_initialiser_does_not_materialise_child_modules() -> None:
    class Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = nn.Linear(4, 4, bias=False)

    with torch.device("meta"):
        parent = Parent()

    initialise_meta_module(parent, torch.device("cpu"))

    assert parent.child.weight.is_meta
    initialise_meta_module(parent.child, torch.device("cpu"))
    assert not parent.child.weight.is_meta
