"""A small terminal sampler for compatible Higguf#7 causal-LM checkpoints.

This is an inference interface, not evidence that a checkpoint is useful for
conversation.  The caller must explicitly acknowledge that a checkpoint has
been trained and evaluated before using it as a chatbot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from .model import HiggufConfig, HiggufForCausalLM


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device) -> HiggufForCausalLM:
    """Load a checkpoint produced by ``higguf.train``."""
    with config_path.open(encoding="utf-8") as source:
        config_data = yaml.safe_load(source)
    model = HiggufForCausalLM(HiggufConfig(**config_data["model"])).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in state:
        raise ValueError("Checkpoint does not contain a Higguf model state.")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Sample one token from final-position logits without changing the model."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    filtered = logits.float() / temperature
    if top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    return int(torch.multinomial(torch.softmax(filtered, dim=-1), 1).item())


@torch.inference_mode()
def generate(
    model: HiggufForCausalLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int | None,
) -> list[int]:
    """Generate tokens using a context window, suitable for the early pilot."""
    if not prompt_ids:
        raise ValueError("The prompt cannot be empty.")
    device = next(model.parameters()).device
    token_ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = token_ids[-model.config.max_sequence_length :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(inputs)["logits"]
        assert isinstance(logits, torch.Tensor)
        next_id = sample_next_token(logits[0, -1], temperature, top_k)
        token_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return token_ids[len(prompt_ids) :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compatible Higguf#7 checkpoint in a terminal.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Acknowledge that this interface does not make an early checkpoint chat-ready.",
    )
    arguments = parser.parse_args()
    if not arguments.allow_untrained:
        raise SystemExit(
            "Refusing to label an unevaluated checkpoint as a chatbot. "
            "Pass --allow-untrained to inspect an early research checkpoint."
        )
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device)
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))
    eos_id = tokenizer.token_to_id("<eos>")
    model = load_model(arguments.checkpoint, arguments.config, device)
    print("Higguf#7 research terminal. Type /quit to exit.")
    print("Warning: this checkpoint has not been shown to be a capable chatbot.")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if not prompt:
            continue
        prompt_ids = tokenizer.encode(f"User: {prompt}\nAssistant:").ids
        generated = generate(
            model, prompt_ids, arguments.max_new_tokens, arguments.temperature, arguments.top_k, eos_id
        )
        print("Higguf#7>", tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
