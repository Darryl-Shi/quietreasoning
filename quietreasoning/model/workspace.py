"""Latent workspace with slot attention and adaptive halting."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


class GatedResidualNetwork(nn.Module):
    hidden_dim: int
    output_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        gate = nn.sigmoid(
            nn.Dense(self.output_dim, dtype=self.dtype, kernel_init=nn.initializers.xavier_uniform())(x)
        )
        h = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x)
        h = nn.Dense(self.hidden_dim, dtype=self.dtype)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.output_dim, dtype=self.dtype)(h)
        return x + gate * h


class WorkspaceBlock(nn.Module):
    """Iterative latent workspace block."""

    num_slots: int
    slot_dim: int
    num_heads: int = 8
    max_steps: int = 3
    halting_threshold: float = 0.8
    halting_epsilon: float = 1e-3
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.init_slots = self.param(
            "init_slots",
            nn.initializers.normal(stddev=0.02),
            (self.num_slots, self.slot_dim),
        )
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            broadcast_dropout=False,
        )
        self.grn = GatedResidualNetwork(
            hidden_dim=self.slot_dim * 4,
            output_dim=self.slot_dim,
            dtype=self.dtype,
        )
        self.halt_proj = nn.Dense(
            1,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )

    def __call__(
        self,
        token_states: Array,
        token_mask: Optional[Array],
        *,
        deterministic: bool = True,
    ) -> Tuple[Array, Array, Array]:
        batch = token_states.shape[0]
        slots = jnp.tile(self.init_slots[None, ...], (batch, 1, 1))
        halting_acc = jnp.zeros((batch,), dtype=self.dtype)
        step_counts = jnp.zeros((batch,), dtype=self.dtype)

        def body(step: int, carry):
            slots, halting_acc, step_counts = carry
            attn_out = self.attn(
                slots,
                token_states,
                token_states,
                mask=token_mask,
                deterministic=deterministic,
                precision=jax.lax.Precision.HIGHEST,
            )
            candidate_slots = self.grn(slots + attn_out)
            pooled = jnp.mean(candidate_slots, axis=1)
            halt_prob = jnp.squeeze(jax.nn.sigmoid(self.halt_proj(pooled)), axis=-1)
            still_running = halting_acc < (self.halting_threshold - self.halting_epsilon)
            still_running_f = still_running.astype(self.dtype)

            halting_acc = halting_acc + still_running_f * halt_prob
            halting_acc = jnp.clip(halting_acc, a_min=0.0, a_max=1.0)
            step_counts = step_counts + still_running_f

            slots = jnp.where(
                still_running_f[:, None, None] > 0.0, candidate_slots, slots
            )
            return slots, halting_acc, step_counts

        slots, halting_acc, step_counts = jax.lax.fori_loop(
            0, self.max_steps, body, (slots, halting_acc, step_counts)
        )

        summary = jnp.mean(slots, axis=1)
        return summary, slots, step_counts

