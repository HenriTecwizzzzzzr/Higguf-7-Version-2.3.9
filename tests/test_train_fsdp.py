from pathlib import Path

import pytest

from higguf.train_fsdp import prune_rank_checkpoints


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
