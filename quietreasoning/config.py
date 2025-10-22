"""Configuration schemas for Quiet Reasoning models and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WorkspaceLossWeights:
    next_latent: float = 1.0
    sparsity: float = 0.1
    orthogonality: float = 0.05
    info_nce: float = 0.2


@dataclass
class WorkspaceConfig:
    slots: int = 64
    dim: int = 1024
    inner_loop_max: int = 3
    halting_threshold: float = 0.8
    halting_epsilon: float = 1e-3
    loss_weights: WorkspaceLossWeights = field(default_factory=WorkspaceLossWeights)


@dataclass
class SSMConfig:
    blocks: List[int] = field(default_factory=lambda: [10, 20])
    state_dim: int = 128
    gating: str = "workspace_scalar"
    z_loss_scale: float = 1.0e-4


@dataclass
class SnapKVConfig:
    keep_fraction: float = 0.3
    quantization: str = "int8"
    stream_window: int = 4096
    max_context: int = 32768
    fallback_keep_fraction: float = 0.1


@dataclass
class EpisodicMemoryConfig:
    backend: str = "faiss_ivfpq"
    params: Dict[str, int] = field(
        default_factory=lambda: {"nlist": 4096, "m": 32, "nprobe": 16}
    )
    ttl_days: int = 30
    write_trigger: str = "bayesian_surprise"
    cooldown_tokens: int = 64
    dedupe_threshold: float = 0.92


@dataclass
class PKMConfig:
    slots: int = 4_000_000
    codebooks: int = 8192
    value_dim: int = 256
    topk: int = 32
    temperature: float = 0.7


@dataclass
class KNNLMConfig:
    enable: bool = True
    index: str = "ivf_pq"
    size: int = 5_000_000
    lambda_max: float = 0.5
    rare_entity_threshold: float = 0.2


@dataclass
class AdapterConfig:
    lora_rank: int = 16
    ia3: bool = True
    num_adapters: int = 8
    fusion: bool = True


@dataclass
class MemoryConfig:
    kv: SnapKVConfig = field(default_factory=SnapKVConfig)
    episodic: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    pkm: PKMConfig = field(default_factory=PKMConfig)
    knn_lm: KNNLMConfig = field(default_factory=KNNLMConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass
class ModelConfig:
    arch: str = "transformer"
    layers: int = 28
    d_model: int = 2304
    n_heads: int = 24
    ffn_inner: int = 6144
    context: int = 8192
    rotary_embedding: bool = True
    norm: str = "rmsnorm"
    tokenizer_path: Optional[str] = None
    vocab_size: int = 50_000
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    ssm: SSMConfig = field(default_factory=SSMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 2.5e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.05
    clip_grad_norm: float = 1.0


@dataclass
class StageSwitch:
    name: str
    tokens: float
    enable: Dict[str, bool] = field(default_factory=dict)
    lr_peak: Optional[float] = None
    distill: Optional[Dict[str, bool]] = None
    datasets: Optional[List[str]] = None
    quant: Optional[Dict[str, bool]] = None
    router_supervision: bool = False


@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    batch_tokens: int = 3_000_000
    stages: List[StageSwitch] = field(
        default_factory=lambda: [
            StageSwitch(
                name="warmstart_dense_ws",
                tokens=7e9,
                enable={
                    "workspace": True,
                    "pkm": False,
                    "adapters": False,
                    "ssm": False,
                    "retrieval": False,
                    "knn": False,
                },
            ),
            StageSwitch(
                name="pkm_adapters",
                tokens=7e9,
                enable={"pkm": True, "adapters": True, "knn": False},
                lr_peak=1.0e-4,
            ),
            StageSwitch(
                name="ssm_light", tokens=2e9, enable={"ssm": True, "knn": False}
            ),
            StageSwitch(
                name="latent_distill",
                tokens=1e9,
                distill={"teacher_cot": True, "latent": True, "visible_rationales": False},
            ),
            StageSwitch(
                name="retrieval_supervision",
                tokens=5e8,
                datasets=["PopQA"],
                enable={"retrieval": True, "knn": True, "pkm": True, "adapters": True},
                router_supervision=True,
            ),
            StageSwitch(
                name="compress_serve",
                tokens=0.0,
                quant={"awq_w4": True, "kv": True},
            ),
        ]
    )


@dataclass
class QuietReasoningConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
