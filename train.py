"""Entrypoint for Quiet Reasoning training on TPU."""

from __future__ import annotations

import functools
from typing import Dict

import jax
import jax.numpy as jnp

from quietreasoning import QuietReasoningConfig, build_train_step, create_train_state
from quietreasoning.training.stages import StageScheduler


def dummy_batch(cfg: QuietReasoningConfig, global_seed: int) -> Dict[str, jnp.ndarray]:
    key = jax.random.PRNGKey(global_seed)
    batch_size = max(1, cfg.training.batch_tokens // cfg.model.context)
    input_ids = jax.random.randint(
        key,
        (batch_size, cfg.model.context),
        minval=0,
        maxval=cfg.model.vocab_size,
    )
    labels = jnp.roll(input_ids, shift=-1, axis=1)
    mask = jnp.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": mask,
    }


def main() -> None:
    cfg = QuietReasoningConfig()
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, cfg)
    scheduler = StageScheduler(cfg.training)
    step_fn = build_train_step(cfg)

    tokens_per_step = cfg.training.batch_tokens
    for step in range(10):
        stage = scheduler.stage_at(float(state.tokens_seen))
        batch = dummy_batch(cfg, step)
        feature_gates = {
            name: jnp.array(1.0 if enabled else 0.0, dtype=jnp.float32)
            for name, enabled in stage.features.items()
        }
        state, logs = step_fn(state, batch, float(tokens_per_step), feature_gates)
        print(
            f"step={step} stage={stage.stage.name} "
            f"loss={float(logs['loss']):.4f} "
            f"workspace_gate={float(feature_gates['workspace']):.1f}"
        )


if __name__ == "__main__":
    main()
