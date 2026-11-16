"""Minimal, auditable training entry point for Higguf#7."""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .model import HiggufConfig, HiggufForCausalLM


class TokenDataset(Dataset[torch.Tensor]):
    def __init__(self, path: str, sequence_length: int) -> None:
        tokens = np.memmap(path, dtype=np.uint32, mode="r")
        if len(tokens) < sequence_length + 1:
            raise ValueError("token file is shorter than one training sequence")
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.sequence_length

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.sequence_length
        return torch.from_numpy(np.asarray(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def infinite_batches(batches: DataLoader[torch.Tensor]) -> Iterator[torch.Tensor]:
    """Repeat a DataLoader without caching every yielded batch in memory."""
    while True:
        yield from batches


def learning_rate(step: int, total_steps: int, warmup_steps: int, maximum: float, minimum: float) -> float:
    if step < warmup_steps:
        return maximum * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return minimum + (maximum - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    model: HiggufForCausalLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    settings = load_config(arguments.config)
    training, model_settings = settings["training"], settings["model"]
    random.seed(training["seed"])
    np.random.seed(training["seed"])
    torch.manual_seed(training["seed"])

    if training["precision"] not in {"fp32", "fp16", "bf16"}:
        raise SystemExit("precision must be fp32, fp16, or bf16")
    if training["precision"] != "fp32" and not torch.cuda.is_available():
        raise SystemExit("fp16/bf16 training requires CUDA. Use the smoke-test config for CPU validation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.float16 if training["precision"] == "fp16" else torch.bfloat16
    use_autocast = device.type == "cuda" and training["precision"] != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast and training["precision"] == "fp16")
    model = HiggufForCausalLM(HiggufConfig(**model_settings)).to(device)
    print(f"Higguf model parameters: {model.parameter_count():,}")
    dataset = TokenDataset(settings["data"]["train_path"], model_settings["max_sequence_length"])
    if len(dataset) < training["micro_batch_size"]:
        raise ValueError("token file does not contain enough complete sequences for one micro-batch")
    batches = DataLoader(dataset, batch_size=training["micro_batch_size"], shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        betas=(0.9, 0.95),
    )
    start_step = 0
    if arguments.resume:
        state = torch.load(arguments.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        print(f"Resumed checkpoint at step {start_step:,}")
    tokens_per_step = (
        training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        * model_settings["max_sequence_length"]
    )
    max_steps = max(1, training["max_tokens"] // tokens_per_step)
    warmup_steps = training["warmup_tokens"] // tokens_per_step
    checkpoint_dir = Path(training["output_dir"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_iterator = infinite_batches(batches)
    for step in range(start_step, start_step + arguments.steps):
        lr = learning_rate(step, max_steps, warmup_steps, training["learning_rate"], training["min_learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        accumulated_loss = 0.0
        for _ in range(training["gradient_accumulation_steps"]):
            tokens = next(batch_iterator).to(device)
            # The model shifts labels internally for next-token prediction.
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                result = model(tokens[:, :-1], labels=tokens[:, :-1])
            loss = result["loss"]
            assert loss is not None
            accumulated_loss += loss.detach().item()
            scaler.scale(loss / training["gradient_accumulation_steps"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        print(f"step={completed_step} loss={accumulated_loss / training['gradient_accumulation_steps']:.4f} lr={lr:.2e}")
        if completed_step % training["checkpoint_every_steps"] == 0:
            save_checkpoint(checkpoint_dir / f"step-{completed_step}.pt", model, optimizer, scaler, completed_step)
    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scaler, start_step + arguments.steps)


if __name__ == "__main__":
    main()
