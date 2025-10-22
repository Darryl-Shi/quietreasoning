"""Loss functions for Quiet Reasoning."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from quietreasoning.config import QuietReasoningConfig
from quietreasoning.model.model import QuietReasoningOutputs

Array = jnp.ndarray


def cross_entropy_loss(logits: Array, targets: Array, mask: Optional[Array] = None) -> Tuple[Array, Dict[str, Array]]:
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot)
    if mask is not None:
        loss = loss * mask
        denom = jnp.maximum(jnp.sum(mask), 1.0)
    else:
        denom = logits.shape[0] * logits.shape[1]
    return jnp.sum(loss) / denom, {"nll": jnp.sum(loss), "tokens": denom}


def answer_only_penalty(logits: Array, rationale_mask: Array, penalty: float = 1.0) -> Array:
    probs = jax.nn.softmax(logits, axis=-1)
    rationale_prob = jnp.sum(probs * rationale_mask[..., None], axis=-1)
    return penalty * jnp.mean(rationale_prob)


def workspace_regularizers(
    outputs: QuietReasoningOutputs,
    targets: Dict[str, Array],
    weights: Dict[str, float],
) -> Tuple[Array, Dict[str, Array]]:
    slots = outputs.workspace_slots
    summary = outputs.workspace_summary
    losses = {}
    total = 0.0

    if weights.get("sparsity", 0.0) > 0:
        sparsity = jnp.mean(jnp.abs(slots))
        losses["workspace_sparsity"] = sparsity
        total += weights["sparsity"] * sparsity

    if weights.get("orth", 0.0) > 0:
        gram = jnp.einsum("bkd,bmd->bkm", slots, slots)
        identity = jnp.eye(gram.shape[-1], dtype=slots.dtype)
        orth = jnp.mean(jnp.square(gram - identity))
        losses["workspace_orth"] = orth
        total += weights["orth"] * orth

    if weights.get("next_latent", 0.0) > 0:
        diffs = jnp.diff(slots, axis=1)
        next_latent = jnp.mean(jnp.square(diffs))
        losses["workspace_next"] = next_latent
        total += weights["next_latent"] * next_latent

    if weights.get("infoNCE", 0.0) > 0 and "target_embeddings" in targets:
        temp = 0.1
        positives = jnp.einsum("bd,bd->b", summary, targets["target_embeddings"])
        negatives = jnp.einsum("bd,md->bm", summary, targets["target_bank"])
        logits = jnp.concatenate([positives[:, None], negatives], axis=1) / temp
        labels = jnp.zeros((summary.shape[0],), dtype=jnp.int32)
        info_nce = optax.softmax_cross_entropy(
            logits,
            jax.nn.one_hot(labels, logits.shape[-1]),
        )
        info_nce = jnp.mean(info_nce)
        losses["workspace_infoNCE"] = info_nce
        total += weights["infoNCE"] * info_nce

    return total, losses


def compute_total_loss(
    outputs: QuietReasoningOutputs,
    batch: Dict[str, Array],
    config: QuietReasoningConfig,
) -> Tuple[Array, Dict[str, Array]]:
    labels = batch["labels"]
    mask = batch.get("loss_mask")
    nll, stats = cross_entropy_loss(outputs.logits[:, :-1, :], labels[:, 1:], mask[:, 1:] if mask is not None else None)
    total = nll
    logs = {"nll": nll}

    if "rationale_mask" in batch:
        penalty = answer_only_penalty(outputs.logits, batch["rationale_mask"], penalty=5.0)
        total += penalty
        logs["answer_only_penalty"] = penalty

    workspace_weights = {
        "next_latent": config.model.workspace.loss_weights.next_latent,
        "sparsity": config.model.workspace.loss_weights.sparsity,
        "orth": config.model.workspace.loss_weights.orthogonality,
        "infoNCE": config.model.workspace.loss_weights.info_nce,
    }
    ws_loss, ws_logs = workspace_regularizers(outputs, batch, workspace_weights)
    total += ws_loss
    logs.update(ws_logs)

    total += outputs.router_aux.z_loss + outputs.router_aux.entropy_loss
    logs["router_z_loss"] = outputs.router_aux.z_loss
    logs["router_entropy_loss"] = outputs.router_aux.entropy_loss

    return total, logs

