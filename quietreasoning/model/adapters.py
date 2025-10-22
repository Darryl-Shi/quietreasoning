"""Adapter bank with LoRA and IA³ style gating."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


class LoRAAdapter(nn.Module):
    features: int
    rank: int
    alpha: float = 32.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        down = nn.Dense(
            self.rank,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_uniform(),
        )
        up = nn.Dense(
            self.features,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
        )
        return up(down(x)) * (self.alpha / self.rank)


class IA3Gate(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        gate = self.param(
            "gate",
            nn.initializers.ones,
            (self.features,),
        )
        return x * gate


class AdapterBank(nn.Module):
    """Collection of adapters that can be selected dynamically."""

    num_adapters: int
    features: int
    lora_rank: int
    use_ia3: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.adapters = tuple(
            LoRAAdapter(self.features, self.lora_rank, dtype=self.dtype, name=f"lora_{i}")
            for i in range(self.num_adapters)
        )
        self.ia3 = IA3Gate(self.features, dtype=self.dtype) if self.use_ia3 else None

    def __call__(self, x: Array, mask: Optional[Array]) -> Array:
        if mask is None:
            mask = jnp.ones((x.shape[0], self.num_adapters), dtype=self.dtype)
        outputs = []
        for i, adapter in enumerate(self.adapters):
            out = adapter(x)
            weight = mask[:, i][..., None]
            outputs.append(out * weight)
        stacked = jnp.sum(jnp.stack(outputs, axis=0), axis=0)
        if self.ia3 is not None:
            stacked = self.ia3(stacked)
        return stacked
