"""Utilities for launching Quiet Reasoning training on TPU pods."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import jax
import jax.distributed as jdist


@dataclass
class TPULaunchConfig:
    topology: str = "2x2"
    coordinator_address: Optional[str] = None
    coordinator_port: int = 12345
    process_id: int = 0
    process_count: int = 1


def configure_environment(cfg: TPULaunchConfig) -> None:
    topology = cfg.topology
    if topology:
        parts = topology.lower().split("x")
        if len(parts) == 2:
            parts.append("1")
        bounds = ",".join(parts)
        os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", bounds)
    if cfg.coordinator_address:
        os.environ.setdefault("COORDINATOR_ADDRESS", f"{cfg.coordinator_address}:{cfg.coordinator_port}")
    os.environ.setdefault("JAX_USE_PJRT_C_API_ON_TPU", "true")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton=false")


def initialize_distributed(cfg: TPULaunchConfig) -> None:
    configure_environment(cfg)
    if jdist.is_initialized():
        return
    jax.distributed.initialize(
        coordinator_address=os.environ.get("COORDINATOR_ADDRESS", ""),
        num_processes=cfg.process_count,
        process_id=cfg.process_id,
    )


__all__ = ["TPULaunchConfig", "configure_environment", "initialize_distributed"]
