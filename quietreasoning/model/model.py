"""Full Quiet Reasoning model definition."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from quietreasoning.config import ModelConfig
from quietreasoning.model.adapters import AdapterBank
from quietreasoning.model.layers import AttentionMetadata, TransformerBlock, rotary_frequencies
from quietreasoning.model.router import RouterAux, RouterDecisions, WorkspaceRouter
from quietreasoning.model.ssm import WorkspaceGatedSSM
from quietreasoning.model.workspace import WorkspaceBlock
from quietreasoning.memory.pkm import ProductKeyMemory

Array = jnp.ndarray


@struct.dataclass
class QuietReasoningOutputs:
    logits: Array
    workspace_summary: Array
    workspace_slots: Array
    workspace_steps: Array
    router_decisions: RouterDecisions
    router_aux: RouterAux
    attention_metadata: Tuple[AttentionMetadata, ...]
    pkm_values: Array


class QuietReasoningModel(nn.Module):
    """3B-class language model with latent workspace and memory routing."""

    config: ModelConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )
        self.workspace = WorkspaceBlock(
            num_slots=self.config.workspace.slots,
            slot_dim=self.config.workspace.dim,
            max_steps=self.config.workspace.inner_loop_max,
            halting_threshold=self.config.workspace.halting_threshold,
            halting_epsilon=self.config.workspace.halting_epsilon,
            dtype=self.dtype,
        )
        self.router = WorkspaceRouter(
            ssm_cfg=self.config.ssm,
            memory_cfg=self.config.memory,
        )
        self.adapter_bank = AdapterBank(
            num_adapters=self.config.memory.adapter.num_adapters,
            features=self.config.d_model,
            lora_rank=self.config.memory.adapter.lora_rank,
            use_ia3=self.config.memory.adapter.ia3,
            dtype=self.dtype,
        )
        self.pkm = ProductKeyMemory(
            num_slots=self.config.memory.pkm.slots,
            codebook_size=self.config.memory.pkm.codebooks,
            value_dim=self.config.memory.pkm.value_dim,
            query_dim=self.config.d_model,
            coarse_topk=64,
        )
        self.out_norm = nn.LayerNorm(dtype=self.dtype)
        self.output_proj = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )

    def _rope_cache(self, seq_len: int) -> Optional[Tuple[Array, Array]]:
        if not self.config.rotary_embedding:
            return None
        cos, sin = rotary_frequencies(
            seq_len,
            self.config.d_model // self.config.n_heads,
            dtype=self.dtype,
        )
        return cos, sin

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        stage_features: Optional[Dict[str, Array]] = None,
        deterministic: bool = True,
    ) -> QuietReasoningOutputs:
        tokens = self.embed(input_ids)
        seq_len = tokens.shape[1]
        rope_cache = self._rope_cache(seq_len)

        attention_mask_processed = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask_processed = nn.attention.make_attention_mask(
                    attention_mask > 0, attention_mask > 0
                )
            else:
                attention_mask_processed = attention_mask

        feature_dict = dict(stage_features) if stage_features is not None else {}

        def gate(name: str, default: float = 1.0) -> Array:
            val = feature_dict.get(name, default)
            val = jnp.asarray(val, dtype=self.dtype)
            if val.ndim == 0:
                val = jnp.broadcast_to(val, (tokens.shape[0],))
            elif val.shape[0] != tokens.shape[0]:
                val = jnp.broadcast_to(val, (tokens.shape[0],))
            return val

        workspace_gate = gate("workspace")
        ssm_gate = gate("ssm")
        pkm_gate = gate("pkm")
        adapter_gate = gate("adapters")
        retrieval_gate = gate("retrieval")
        knn_gate = gate("knn")

        attn_metadata: List[AttentionMetadata] = []
        x = tokens
        workspace_summary = None
        workspace_steps = None
        workspace_slots = None

        for layer_idx in range(self.config.layers):
            ssm_module = None
            if (layer_idx + 1) in self.config.ssm.blocks:
                ssm_module = WorkspaceGatedSSM(
                    d_model=self.config.d_model,
                    state_dim=self.config.ssm.state_dim,
                    dtype=self.dtype,
                    name=f"ssm_{layer_idx}",
                )

            block = TransformerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                ff_dim=self.config.ffn_inner,
                dropout_rate=0.0,
                rotary=self.config.rotary_embedding,
                dtype=self.dtype,
                layer_idx=layer_idx,
                ssm_layer=ssm_module,
                name=f"block_{layer_idx}",
            )
            x, meta = block(
                x,
                attention_mask_processed,
                deterministic=deterministic,
                rope_cache=rope_cache,
                workspace_summary=workspace_summary,
                ssm_global_gate=ssm_gate,
            )
            attn_metadata.append(meta)

            if layer_idx + 1 == 20:
                workspace_summary, workspace_slots, workspace_steps = self.workspace(
                    x,
                    None,
                    deterministic=deterministic,
                )

        if workspace_summary is None:
            workspace_summary, workspace_slots, workspace_steps = self.workspace(
                x,
                None,
                deterministic=deterministic,
            )

        workspace_summary = workspace_summary * workspace_gate[:, None]
        workspace_steps = workspace_steps * workspace_gate
        if workspace_slots is not None:
            workspace_slots = workspace_slots * workspace_gate[:, None, None]

        router_decisions, router_aux = self.router(workspace_summary)

        gated_router = RouterDecisions(
            ssm_gate=router_decisions.ssm_gate * ssm_gate,
            pkm_topk=router_decisions.pkm_topk,
            use_pkm=router_decisions.use_pkm * pkm_gate,
            episodic_k=router_decisions.episodic_k * retrieval_gate,
            use_knn=router_decisions.use_knn * knn_gate * retrieval_gate,
            knn_lambda=router_decisions.knn_lambda * knn_gate * retrieval_gate,
            adapter_mask=router_decisions.adapter_mask * adapter_gate[:, None],
        )

        pkm_result = self.pkm(workspace_summary, self.config.memory.pkm.topk)
        pkm_projected = nn.Dense(
            self.config.d_model,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="pkm_backproj",
        )(pkm_result.values)
        x = x + gated_router.use_pkm[:, None, None] * pkm_projected[:, None, :]

        adapter_residual = self.adapter_bank(x.mean(axis=1), gated_router.adapter_mask)
        x = x + adapter_residual[:, None, :] * adapter_gate[:, None, None]

        logits = self.output_proj(self.out_norm(x))
        return QuietReasoningOutputs(
            logits=logits,
            workspace_summary=workspace_summary,
            workspace_slots=workspace_slots,
            workspace_steps=workspace_steps,
            router_decisions=gated_router,
            router_aux=router_aux,
            attention_metadata=tuple(attn_metadata),
            pkm_values=pkm_result.values,
        )
