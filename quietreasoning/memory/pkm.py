"""Product Key Memory implementation compatible with JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


@dataclass
class PKMReadResult:
    values: Array
    indices: Array
    scores: Array


class ProductKeyMemory(nn.Module):
    num_slots: int
    codebook_size: int
    value_dim: int
    query_dim: int
    coarse_topk: int = 64
    value_scale: float = 0.01

    def setup(self) -> None:
        half_dim = self.query_dim // 2
        self.codebook_a = self.param(
            "codebook_a",
            nn.initializers.normal(stddev=0.02),
            (self.codebook_size, half_dim),
        )
        self.codebook_b = self.param(
            "codebook_b",
            nn.initializers.normal(stddev=0.02),
            (self.codebook_size, half_dim),
        )
        self.values = self.param(
            "values",
            nn.initializers.normal(stddev=0.02),
            (self.num_slots, self.value_dim),
        )
        self.value_scale_param = self.param(
            "value_scale",
            lambda *_: jnp.array(self.value_scale, dtype=jnp.float32),
        )
        self.query_proj = nn.Dense(
            self.query_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="query_proj",
        )
        self.value_proj = nn.Dense(
            self.query_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="value_proj",
        )

    def _combine_indices(self, idx_a: Array, idx_b: Array) -> Array:
        return idx_a * self.codebook_size + idx_b

    def __call__(self, queries: Array, topk: int) -> PKMReadResult:
        """Reads from PKM given query vectors."""
        half_dim = self.query_dim // 2
        q = self.query_proj(queries)
        qa, qb = jnp.split(q, 2, axis=-1)
        codebook_a = jax.lax.stop_gradient(self.codebook_a)
        codebook_b = jax.lax.stop_gradient(self.codebook_b)

        score_a = jnp.dot(qa, codebook_a.T)
        score_b = jnp.dot(qb, codebook_b.T)

        top_sa, idx_a = jax.lax.top_k(score_a, min(self.coarse_topk, score_a.shape[-1]))
        top_sb, idx_b = jax.lax.top_k(score_b, min(self.coarse_topk, score_b.shape[-1]))

        def combine(scores_a: Array, scores_b: Array, ia: Array, ib: Array):
            pair_scores = (
                scores_a[:, None]
                + scores_b[None, :]
            )
            pair_scores = pair_scores.reshape(-1)
            pair_idx = self._combine_indices(
                jnp.repeat(ia, scores_b.shape[0]),
                jnp.tile(ib, scores_a.shape[0]),
            )
            take = min(topk, pair_scores.shape[0])
            best_scores, best_idx = jax.lax.top_k(pair_scores, take)
            gathered_idx = pair_idx[best_idx]
            return best_scores, gathered_idx

        scores, indices = jax.vmap(combine)(top_sa, top_sb, idx_a, idx_b)

        values = jnp.take(self.values, indices, axis=0, mode="wrap")
        scale = jnp.asarray(self.value_scale_param)
        weighted = jnp.einsum("bn,bnd->bd", jax.nn.softmax(scores, axis=-1), values * scale)
        return PKMReadResult(values=weighted, indices=indices, scores=scores)
