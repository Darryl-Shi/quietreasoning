"""Streaming KV cache with SnapKV head-aware selection and quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class KVCacheState:
    keys: Array
    values: Array
    key_scale: Array
    value_scale: Array
    positions: Array


class SnapKVCache:
    """Implements SnapKV keep ratios and quantization for long-context decoding."""

    def __init__(
        self,
        keep_fraction: float = 0.3,
        stream_window: int = 4096,
        quantization: str = "int8",
    ) -> None:
        self.keep_fraction = keep_fraction
        self.stream_window = stream_window
        self.quantization = quantization

    def _quantize(self, tensor: Array) -> Tuple[Array, Array]:
        if self.quantization == "none":
            return tensor, jnp.ones((tensor.shape[0], 1, 1))
        qmin, qmax = (-128, 127) if self.quantization == "int8" else (-8, 7)
        scale = jnp.max(jnp.abs(tensor), axis=-1, keepdims=True) / qmax
        scale = jnp.maximum(scale, 1e-6)
        quantized = jnp.clip(jnp.round(tensor / scale), qmin, qmax)
        return quantized.astype(jnp.int8), scale.astype(jnp.float32)

    def _dequantize(self, tensor: Array, scale: Array) -> Array:
        if self.quantization == "none":
            return tensor
        return tensor.astype(jnp.float32) * scale

    def prefill(self, keys: Array, values: Array, attn_scores: Array) -> KVCacheState:
        """Initializes the cache based on prefill attention scores."""
        batch, heads, seq_len, _ = keys.shape
        keep = jnp.maximum(1, int(self.keep_fraction * seq_len))
        importance = jnp.mean(attn_scores, axis=-1)
        top_scores, top_indices = jax.lax.top_k(importance, keep)
        gather_indices = jnp.sort(top_indices, axis=-1)

        def gather(data: Array) -> Array:
            return jnp.take_along_axis(
                data,
                gather_indices[..., None, None].repeat(data.shape[-1], axis=-1),
                axis=2,
            )

        kept_keys = gather(keys)
        kept_values = gather(values)
        q_keys, scale_k = self._quantize(kept_keys)
        q_values, scale_v = self._quantize(kept_values)
        return KVCacheState(
            keys=q_keys,
            values=q_values,
            key_scale=scale_k,
            value_scale=scale_v,
            positions=gather_indices,
        )

    def update(self, cache: KVCacheState, new_keys: Array, new_values: Array) -> KVCacheState:
        """Streams new kv pairs while enforcing window limits."""
        keep_window = self.stream_window
        keys_deq = self._dequantize(cache.keys, cache.key_scale)
        values_deq = self._dequantize(cache.values, cache.value_scale)

        concat_keys = jnp.concatenate([keys_deq, new_keys], axis=2)
        concat_vals = jnp.concatenate([values_deq, new_values], axis=2)
        positions = jnp.concatenate(
            [cache.positions, cache.positions[..., -1:] + jnp.arange(1, new_keys.shape[2] + 1)],
            axis=-1,
        )
        if concat_keys.shape[2] > keep_window:
            concat_keys = concat_keys[..., -keep_window:, :]
            concat_vals = concat_vals[..., -keep_window:, :]
            positions = positions[..., -keep_window:]

        q_keys, scale_k = self._quantize(concat_keys)
        q_values, scale_v = self._quantize(concat_vals)
        return KVCacheState(
            keys=q_keys,
            values=q_values,
            key_scale=scale_k,
            value_scale=scale_v,
            positions=positions,
        )

    def materialize(self, cache: KVCacheState) -> Tuple[Array, Array]:
        keys = self._dequantize(cache.keys, cache.key_scale)
        values = self._dequantize(cache.values, cache.value_scale)
        return keys, values
