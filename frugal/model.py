"""Local MLX model with per-token logprob streaming.

Two backends:
  LocalModel     — mlx-lm  (Qwen, Llama, Phi, Gemma 2, …)
  VLMLocalModel  — mlx-vlm (Gemma 3 and other vision-language models used text-only)

Both expose the same generate() / format_prompt() interface so the rest of the
pipeline (entropy monitor, buddy escalation) is backend-agnostic.
"""
from __future__ import annotations

from typing import Generator
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


class LocalModel:
    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    ):
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temp: float = 0.7,
    ) -> Generator[tuple[str, float], None, None]:
        """
        Yield (text_chunk, entropy) pairs during generation.

        text_chunk may be a partial word (BPE) — the pipeline accumulates
        these into clauses before making routing decisions.

        entropy is Shannon entropy of the token's log-probability distribution,
        computed here in MLX (fast, avoids shipping the full vocab vector to Rust).
        """
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp),
        ):
            logprobs: mx.array = response.logprobs  # shape: (vocab_size,)
            probs = mx.exp(logprobs)
            entropy = float(-mx.sum(probs * logprobs).item())
            yield response.text, entropy

            if response.finish_reason is not None:
                break

    def format_prompt(self, user_message: str, enable_thinking: bool = False) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_message}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                # Tokenizer doesn't support enable_thinking — not a thinking model
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        return user_message


class VLMLocalModel:
    """mlx-vlm backend — same interface as LocalModel, for Gemma 3 and friends."""

    def __init__(self, model_path: str = "mlx-community/gemma-3-4b-it-4bit"):
        from mlx_vlm import load as vlm_load
        self.model, self.processor = vlm_load(model_path)
        self.model_path = model_path

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temp: float = 0.0,
    ) -> Generator[tuple[str, float], None, None]:
        from mlx_vlm import stream_generate as vlm_stream

        for response in vlm_stream(
            self.model,
            self.processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            verbose=False,
        ):
            logprobs = getattr(response, "logprobs", None)
            if logprobs is not None:
                probs = mx.exp(logprobs)
                entropy = float(-mx.sum(probs * logprobs).item())
            else:
                entropy = 0.0
            yield response.text, entropy

            if getattr(response, "finish_reason", None) is not None:
                break

    def format_prompt(self, user_message: str, **_kwargs) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template
        messages = [{"role": "user", "content": user_message}]
        return apply_chat_template(
            self.processor, self.model.config, messages, num_images=0
        )


def load_local_model(model_path: str) -> "LocalModel | VLMLocalModel":
    """Auto-detect backend from model path and return the right wrapper."""
    vlm_ids = ("gemma-3", "gemma-4", "paligemma")
    if any(tag in model_path.lower() for tag in vlm_ids):
        return VLMLocalModel(model_path)
    return LocalModel(model_path)
