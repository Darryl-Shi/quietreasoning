"""Optimizer utilities."""

from __future__ import annotations

import optax

from quietreasoning.config import OptimizerConfig


def build_optimizer(cfg: OptimizerConfig) -> optax.GradientTransformation:
    if cfg.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.name}")
    schedule = optax.constant_schedule(cfg.learning_rate)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.betas[0],
            b2=cfg.betas[1],
            weight_decay=cfg.weight_decay,
        ),
    )
    return optimizer
