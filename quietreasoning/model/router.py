"""Workspace-conditioned router for memory and expert activation."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from quietreasoning.config import MemoryConfig, SSMConfig

Array = jnp.ndarray


@struct.dataclass
class RouterDecisions:
    ssm_gate: Array
    pkm_topk: Array
    use_pkm: Array
    episodic_k: Array
    use_knn: Array
    knn_lambda: Array
    adapter_mask: Array


@struct.dataclass
class RouterAux:
    z_loss: Array
    entropy_loss: Array
    logits: Array


class WorkspaceRouter(nn.Module):
    """Predicts routing decisions from workspace summary."""

    ssm_cfg: SSMConfig
    memory_cfg: MemoryConfig
    entropy_floor: float = 0.1

    @nn.compact
    def __call__(self, h_ws: Array, deterministic: bool = True) -> tuple[RouterDecisions, RouterAux]:
        hidden = nn.Dense(
            h_ws.shape[-1],
            kernel_init=nn.initializers.xavier_uniform(),
            name="router_hidden",
        )(h_ws)
        hidden = nn.gelu(hidden)

        ssm_logits = nn.Dense(1, name="ssm_gate")(hidden)
        ssm_gate = jax.nn.sigmoid(ssm_logits)

        pkm_logit = nn.Dense(1, name="pkm_gate")(hidden)
        pkm_prob = jax.nn.sigmoid(pkm_logit)
        pkm_topk_logits = nn.Dense(1, name="pkm_topk")(hidden)
        pkm_topk = jnp.clip(
            jnp.round(jnp.squeeze(jax.nn.sigmoid(pkm_topk_logits), -1) * self.memory_cfg.pkm.topk),
            1,
            self.memory_cfg.pkm.topk,
        )

        episodic_logits = nn.Dense(1, name="episodic_k")(hidden)
        episodic_k = jnp.clip(
            jnp.round(jax.nn.relu(jnp.squeeze(episodic_logits, -1)) * 16 + 4),
            4,
            64,
        )

        knn_logits = nn.Dense(1, name="knn_lambda")(hidden)
        knn_lambda = jnp.squeeze(jax.nn.sigmoid(knn_logits), -1) * self.memory_cfg.knn_lm.lambda_max
        use_knn = (knn_lambda > 0.05).astype(jnp.float32)

        adapter_scores = nn.Dense(self.memory_cfg.adapter.num_adapters, name="adapter_scores")(hidden)
        adapter_mask = jax.nn.sigmoid(adapter_scores)

        logits = jnp.concatenate(
            [
                jnp.squeeze(ssm_logits, -1),
                jnp.squeeze(pkm_logit, -1),
                jnp.squeeze(knn_logits, -1),
            ],
            axis=-1,
        )
        z_loss = jnp.mean(jnp.square(logits)) * self.ssm_cfg.z_loss_scale

        entropy = -jnp.mean(
            jax.nn.sigmoid(logits) * jnp.log(jax.nn.sigmoid(logits) + 1e-6)
            + (1.0 - jax.nn.sigmoid(logits)) * jnp.log(1.0 - jax.nn.sigmoid(logits) + 1e-6)
        )
        entropy_loss = jnp.maximum(0.0, self.entropy_floor - entropy)

        decisions = RouterDecisions(
            ssm_gate=jnp.squeeze(ssm_gate, -1),
            pkm_topk=pkm_topk,
            use_pkm=jnp.squeeze(pkm_prob, -1),
            episodic_k=episodic_k,
            use_knn=use_knn,
            knn_lambda=knn_lambda,
            adapter_mask=adapter_mask,
        )
        aux = RouterAux(z_loss=z_loss, entropy_loss=entropy_loss, logits=logits)
        return decisions, aux
