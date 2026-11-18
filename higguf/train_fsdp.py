"""FSDP training entry point for multi-GPU Higguf#7 research pilots.

Run this module with ``torchrun``.  It deliberately writes one sharded state
file per rank rather than pretending a large model can be exported as a single
ordinary checkpoint.  It is a training mechanism, not evidence that a target
model has been trained or evaluated.
"""

from __future__ import annotations

import argparse
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import DecoderLayer, HiggufConfig, HiggufForCausalLM, RMSNorm
from .train import TokenDataset, infinite_batches, learning_rate, load_config


def initialise_meta_module(module: torch.nn.Module, device: torch.device) -> None:
    """Materialise a meta-device leaf module on this rank with Higguf init."""
    module.to_empty(device=device)

    def initialise(child: torch.nn.Module) -> None:
        if isinstance(child, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(child.weight, mean=0.0, std=0.02)
        elif isinstance(child, RMSNorm):
            child.reset_parameters()

    module.apply(initialise)


def build_optimizer(model: FSDP, training: dict) -> torch.optim.Optimizer:
    if training.get("optimizer", "adamw").lower() == "adafactor":
        if not hasattr(torch.optim, "Adafactor"):
            raise SystemExit("This PyTorch build does not provide torch.optim.Adafactor.")
        return torch.optim.Adafactor(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    return torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        betas=(0.9, 0.95),
    )


def save_sharded_checkpoint(model: FSDP, optimizer: torch.optim.Optimizer, directory: Path, step: int, rank: int) -> None:
    """Save rank-local FSDP model and optimizer shards for explicit recovery."""
    directory.mkdir(parents=True, exist_ok=True)
    config = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, config):
        state = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "step": step,
            "format": "fsdp-sharded-v1",
        }
    torch.save(state, directory / f"step-{step:08d}-rank-{rank}.pt")


def prune_rank_checkpoints(directory: Path, rank: int, keep_last: int) -> None:
    """Keep only the newest complete-step shards owned by this rank."""
    if keep_last < 1:
        raise ValueError("checkpoint_keep_last must be at least one")
    shards = sorted(directory.glob(f"step-*-rank-{rank}.pt"))
    for obsolete in shards[:-keep_last]:
        obsolete.unlink()


def load_sharded_checkpoint(
    model: FSDP, optimizer: torch.optim.Optimizer, directory: Path, step: int, rank: int
) -> int:
    """Restore one rank's matching FSDP shard and return its completed step."""
    checkpoint = directory / f"step-{step:08d}-rank-{rank}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing rank-local FSDP checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "fsdp-sharded-v1":
        raise ValueError("Checkpoint is not a Higguf FSDP sharded checkpoint.")
    if int(state.get("step", -1)) != step:
        raise ValueError(f"Checkpoint step mismatch in {checkpoint}")
    config = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, config):
        model.load_state_dict(state["model"])
        optimizer_state = FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
    optimizer.load_state_dict(optimizer_state)
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Higguf#7 FSDP research pilot with torchrun.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cpu-offload", action="store_true", help="Offload FSDP parameters between GPU operations.")
    parser.add_argument("--resume-dir", type=Path, help="Directory containing rank-local FSDP checkpoint shards.")
    parser.add_argument("--resume-step", type=int, help="Completed step number to restore from --resume-dir.")
    arguments = parser.parse_args()
    if arguments.steps < 1:
        raise SystemExit("--steps must be at least one")
    if (arguments.resume_dir is None) != (arguments.resume_step is None):
        raise SystemExit("--resume-dir and --resume-step must be provided together.")
    if arguments.resume_step is not None and arguments.resume_step < 1:
        raise SystemExit("--resume-step must be positive.")
    if not torch.cuda.is_available():
        raise SystemExit("FSDP training requires CUDA; no CUDA GPU is available.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise SystemExit("Launch with torchrun, for example: torchrun --standalone --nproc_per_node=2 -m higguf.train_fsdp ...")

    rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise SystemExit("This research launcher requires at least two ranks for sharded training.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        settings = load_config(str(arguments.config))
        training, model_settings = settings["training"], settings["model"]
        precision = training["precision"]
        if precision not in {"fp16", "bf16"}:
            raise SystemExit("FSDP launcher requires fp16 or bf16 precision.")
        keep_last = int(training.get("checkpoint_keep_last", 2))
        if keep_last < 1:
            raise SystemExit("checkpoint_keep_last must be at least one.")
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        random.seed(training["seed"] + rank)
        np.random.seed(training["seed"] + rank)
        torch.manual_seed(training["seed"] + rank)

        # Meta construction avoids allocating the entire target on every rank.
        with torch.device("meta"):
            raw_model = HiggufForCausalLM(HiggufConfig(**model_settings))
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={DecoderLayer})
        model = FSDP(
            raw_model,
            auto_wrap_policy=policy,
            mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
            cpu_offload=CPUOffload(offload_params=arguments.cpu_offload),
            device_id=device,
            param_init_fn=lambda module: initialise_meta_module(module, device),
            sync_module_states=True,
            use_orig_params=True,
        )
        optimizer = build_optimizer(model, training)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        start_step = 0
        if arguments.resume_dir is not None and arguments.resume_step is not None:
            start_step = load_sharded_checkpoint(model, optimizer, arguments.resume_dir, arguments.resume_step, rank)
            if rank == 0:
                print(f"Restored sharded checkpoint at step {start_step}.", flush=True)
        dataset = TokenDataset(settings["data"]["train_path"], model_settings["max_sequence_length"])
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        batches = DataLoader(dataset, batch_size=training["micro_batch_size"], sampler=sampler, drop_last=True)
        if len(batches) == 0:
            raise ValueError("token file does not contain enough sequences per distributed rank")
        iterator = infinite_batches(batches)
        tokens_per_step = training["micro_batch_size"] * training["gradient_accumulation_steps"] * model_settings["max_sequence_length"] * world_size
        max_steps = max(1, training["max_tokens"] // tokens_per_step)
        warmup_steps = training["warmup_tokens"] // tokens_per_step
        model.train()
        for step in range(start_step, start_step + arguments.steps):
            lr = learning_rate(step, max_steps, warmup_steps, training["learning_rate"], training["min_learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for _ in range(training["gradient_accumulation_steps"]):
                tokens = next(iterator).to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=dtype):
                    result = model(tokens[:, :-1], labels=tokens[:, :-1])
                loss = result["loss"]
                assert isinstance(loss, torch.Tensor)
                total_loss += loss.detach().item()
                scaler.scale(loss / training["gradient_accumulation_steps"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            finished = step + 1
            if rank == 0:
                print(f"step={finished} mean_loss={total_loss / training['gradient_accumulation_steps']:.4f} lr={lr:.2e}", flush=True)
            if finished % training["checkpoint_every_steps"] == 0 or finished == arguments.steps:
                output_dir = Path(training["output_dir"])
                save_sharded_checkpoint(model, optimizer, output_dir, finished, rank)
                dist.barrier()
                prune_rank_checkpoints(output_dir, rank, keep_last)
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
