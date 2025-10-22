"""Inference helpers ensuring answer-only decoding."""

from __future__ import annotations

from typing import Iterable, List

import jax
import jax.numpy as jnp

from quietreasoning.config import QuietReasoningConfig
from quietreasoning.memory.kv_cache import SnapKVCache, KVCacheState
from quietreasoning.model import QuietReasoningModel

Array = jnp.ndarray


class AnswerOnlyDecoder:
    def __init__(
        self,
        cfg: QuietReasoningConfig,
        rationale_token_ids: Iterable[int],
        penalty: float = 4.0,
        temperature: float = 0.7,
        max_length: int = 128,
    ) -> None:
        self.cfg = cfg
        self.model = QuietReasoningModel(cfg.model)
        self.rationale_token_ids = jnp.array(list(rationale_token_ids), dtype=jnp.int32)
        self.penalty = penalty
        self.temperature = temperature
        self.max_length = max_length
        self.kv_cache = SnapKVCache(
            keep_fraction=cfg.model.memory.kv.keep_fraction,
            stream_window=cfg.model.memory.kv.stream_window,
            quantization=cfg.model.memory.kv.quantization,
        )

    def _penalize_rationale(self, logits: Array) -> Array:
        mask = jnp.zeros_like(logits)
        mask = mask.at[..., self.rationale_token_ids].set(1.0)
        return logits - self.penalty * mask

    def generate(self, params: dict, prompt: Array) -> Array:
        """Greedy decode with workspace gating and rational suppression."""
        tokens = prompt
        logits = None
        cache: KVCacheState | None = None
        for _ in range(self.max_length):
            outputs = self.model.apply({"params": params}, tokens, deterministic=True)
            logits = outputs.logits[:, -1, :]
            logits = self._penalize_rationale(logits)
            next_token = jnp.argmax(logits, axis=-1, keepdims=True)
            tokens = jnp.concatenate([tokens, next_token], axis=1)
            if next_token[0, 0] == 0:
                break
        return tokens[:, prompt.shape[1]:], logits

