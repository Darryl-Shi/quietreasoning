"""Core transformer layers used by the Quiet Reasoning model."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

try:
    from jax.sharding import PartitionSpec as _PartitionSpec
except (ImportError, AttributeError):
    _PartitionSpec = None  # Compatible with older JAX versions

Array = jnp.ndarray
PRNGKey = jax.Array


def with_sharding_constraint(x: Array, spec: Any) -> Array:
    """Convenience wrapper allowing compilation without sharding meshes."""
    try:
        partition_spec = spec
        if _PartitionSpec is not None and spec is not None and not isinstance(spec, _PartitionSpec):
            if isinstance(spec, (tuple, list)):
                partition_spec = _PartitionSpec(*spec)
        return jax.lax.with_sharding_constraint(x, partition_spec)
    except (ValueError, TypeError, RuntimeError):
        return x


class RMSNorm(nn.Module):
    """Root mean square layer norm."""

    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.epsilon)
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        return normed * scale.astype(self.dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    hidden_dim: int
    out_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dense = nn.Dense(
            self.hidden_dim * 2,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )
        projected = dense(x)
        x_g, x_l = jnp.split(projected, 2, axis=-1)
        return nn.Dense(
            self.out_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(nn.swish(x_g) * x_l)


def apply_rotary_pos_emb(q: Array, k: Array, freqs: Array) -> Tuple[Array, Array]:
    """Apply RoPE to query and key projections."""

    def reshape_for_broadcast(x: Array) -> Array:
        return einops.rearrange(x, "b l h (d2 d) -> b l h d2 d", d2=2)

    q = reshape_for_broadcast(q)
    k = reshape_for_broadcast(k)
    cos, sin = freqs
    cos = cos[..., None]
    sin = sin[..., None]
    q1, q2 = jnp.split(q, 2, axis=-2)
    k1, k2 = jnp.split(k, 2, axis=-2)
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-2)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-2)
    q_rot = einops.rearrange(q_rot, "b l h d2 d -> b l h (d2 d)")
    k_rot = einops.rearrange(k_rot, "b l h d2 d -> b l h (d2 d)")
    return q_rot, k_rot


def rotary_frequencies(
    seq_len: int, dim: int, base: float = 10000.0, dtype: Any = jnp.float32
) -> Tuple[Array, Array]:
    """Generate RoPE cos/sin caches."""
    assert dim % 2 == 0, "RoPE dimension must be even."
    exponent = jnp.arange(0, dim, 2, dtype=dtype) / dim
    theta = 1.0 / (base**exponent)
    idx = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.einsum("i,j->ij", idx, theta)
    return jnp.cos(freqs), jnp.sin(freqs)


@struct.dataclass
class AttentionMetadata:
    """Auxiliary data for logging diagnostics."""

    attn_entropy: Optional[Array] = struct.field(default=None)
    kv_pruned: Optional[int] = struct.field(default=None)
    ssm_gate: Optional[Array] = struct.field(default=None)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional rotary embedding."""

    num_heads: int
    head_dim: int
    rotary: bool = True
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    attention_axes: Tuple[int, ...] = (-2,)

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Optional[Array],
        deterministic: bool,
        rope_cache: Optional[Tuple[Array, Array]] = None,
        workspace_summary: Optional[Array] = None,
        ssm_global_gate: Optional[Array] = None,
    ) -> Tuple[Array, AttentionMetadata]:
        qkv = nn.DenseGeneral(
            (self.num_heads, self.head_dim * 3),
            axis=-1,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = with_sharding_constraint(q, ("batch", "length", "heads", "kv"))
        k = with_sharding_constraint(k, ("batch", "length", "heads", "kv"))
        v = with_sharding_constraint(v, ("batch", "length", "heads", "kv"))

        if self.rotary and rope_cache is not None:
            cos, sin = rope_cache
            q, k = apply_rotary_pos_emb(q, k, (cos, sin))

        attn_weights = nn.attention.dot_product_attention(
            query=q,
            key=k,
            value=v,
            bias=None,
            dtype=self.dtype,
            precision=jax.lax.Precision.HIGHEST,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=mask is None,
            deterministic=deterministic,
            mask=mask,
            attention_axes=self.attention_axes,
        )
        out = nn.DenseGeneral(
            x.shape[-1],
            axis=(-2, -1),
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="out",
        )(attn_weights)

        entropy = -jnp.sum(
            jax.nn.log_softmax(jnp.abs(attn_weights), axis=-1) * jnp.abs(attn_weights),
            axis=-1,
        )

        metadata = AttentionMetadata(attn_entropy=jnp.mean(entropy))
        return out, metadata


class TransformerBlock(nn.Module):
    """Transformer block with RMSNorm, attention, and SwiGLU feed-forward."""

    d_model: int
    n_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    rotary: bool = True
    dtype: Any = jnp.float32
    residual_scale: float = 1.0
    layer_idx: int = 0
    ssm_layer: Optional[nn.Module] = None

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Optional[Array],
        deterministic: bool,
        rope_cache: Optional[Tuple[Array, Array]] = None,
        workspace_summary: Optional[Array] = None,
        ssm_global_gate: Optional[Array] = None,
    ) -> Tuple[Array, AttentionMetadata]:
        normed = RMSNorm(dtype=self.dtype, name=f"attn_norm_{self.layer_idx}")(x)
        attn_out, attn_meta = MultiHeadAttention(
            num_heads=self.n_heads,
            head_dim=self.d_model // self.n_heads,
            rotary=self.rotary,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name=f"mha_{self.layer_idx}",
        )(normed, mask, deterministic, rope_cache)
        x = x + attn_out * self.residual_scale

        if self.ssm_layer is not None:
            ssm_out, ssm_gate = self.ssm_layer(
                normed,
                workspace_summary,
                deterministic=deterministic,
                global_gate=ssm_global_gate,
            )
            x = x + ssm_out
            if ssm_global_gate is not None:
                ssm_gate = ssm_gate * jnp.squeeze(ssm_global_gate)
            attn_meta.ssm_gate = ssm_gate

        ff_input = RMSNorm(dtype=self.dtype, name=f"ff_norm_{self.layer_idx}")(x)
        ff_out = SwiGLU(self.ff_dim, self.d_model, dtype=self.dtype, name=f"swiglu_{self.layer_idx}")(ff_input)
        if self.dropout_rate > 0.0:
            ff_out = nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=deterministic)
        x = x + ff_out * self.residual_scale
        return x, attn_meta
