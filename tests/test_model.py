"""Unit tests for QuietReasoning models."""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quietreasoning.config import (
    AdapterConfig,
    EpisodicMemoryConfig,
    KNNLMConfig,
    MemoryConfig,
    ModelConfig,
    PKMConfig,
    QuietReasoningConfig,
    SnapKVConfig,
    SSMConfig,
    WorkspaceConfig,
)
from quietreasoning.model import QuietReasoningModel


def _tiny_config(ssm_blocks: list[int] | None = None) -> QuietReasoningConfig:
    """Build a small configuration suitable for unit tests."""

    if ssm_blocks is None:
        ssm_blocks = []

    d_model = 64

    workspace = WorkspaceConfig(
        slots=4,
        dim=32,
        inner_loop_max=1,
        halting_threshold=0.5,
        halting_epsilon=1e-3,
    )

    ssm = SSMConfig(
        blocks=ssm_blocks,
        state_dim=d_model,
        gating="workspace_scalar",
        z_loss_scale=1e-4,
    )

    memory = MemoryConfig(
        kv=SnapKVConfig(
            keep_fraction=0.25,
            quantization="int8",
            stream_window=64,
            max_context=256,
            fallback_keep_fraction=0.1,
        ),
        episodic=EpisodicMemoryConfig(
            backend="faiss_ivfpq",
            params={"nlist": 8, "m": 4, "nprobe": 2},
            ttl_days=1,
            write_trigger="surprise",
            cooldown_tokens=8,
            dedupe_threshold=0.5,
        ),
        pkm=PKMConfig(
            slots=128,
            codebooks=16,
            value_dim=32,
            topk=4,
            temperature=0.5,
        ),
        knn_lm=KNNLMConfig(
            enable=False,
            index="ivf_pq",
            size=1024,
            lambda_max=0.1,
            rare_entity_threshold=0.05,
        ),
        adapter=AdapterConfig(
            lora_rank=4,
            ia3=False,
            num_adapters=2,
            fusion=False,
        ),
    )

    model_cfg = ModelConfig(
        layers=2,
        d_model=d_model,
        n_heads=8,
        ffn_inner=128,
        context=128,
        rotary_embedding=True,
        vocab_size=512,
        workspace=workspace,
        ssm=ssm,
        memory=memory,
    )

    return QuietReasoningConfig(model=model_cfg)


def test_quiet_reasoning_model_forward_shapes():
    cfg = _tiny_config()
    model = QuietReasoningModel(config=cfg.model)

    batch, seq_len = 2, 16
    input_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)

    variables = model.init({"params": rng}, input_ids, deterministic=True)
    outputs = model.apply(variables, input_ids, deterministic=True)

    assert outputs.logits.shape == (batch, seq_len, cfg.model.vocab_size)
    assert outputs.workspace_summary.shape == (batch, cfg.model.workspace.dim)
    assert outputs.router_decisions.adapter_mask.shape == (batch, cfg.model.memory.adapter.num_adapters)


def test_quiet_reasoning_model_with_ssm_metadata():
    cfg = _tiny_config(ssm_blocks=[1])
    model = QuietReasoningModel(config=cfg.model)

    batch, seq_len = 1, 8
    input_ids = jnp.arange(batch * seq_len, dtype=jnp.int32).reshape(batch, seq_len)
    rng = jax.random.PRNGKey(42)

    variables = model.init({"params": rng}, input_ids, deterministic=True)
    outputs = model.apply(variables, input_ids, deterministic=True)

    first_meta = outputs.attention_metadata[0]
    assert first_meta.ssm_gate is not None
    assert first_meta.ssm_gate.shape[0] == batch
