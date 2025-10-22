# Quiet Reasoning

Quiet Reasoning is a 3B-class language model designed for **latent (non-emissive) reasoning** with a workspace-routed memory hierarchy. The implementation follows the Variant-1 (+ light SSM experts) specification and merges all of the engineering details from `spec.md` and `docs/training_recipe.md` into a runnable TPU project.

---

## Architecture Highlights
- **Tokenizer**: SentencePiece, vocab 50k.
- **Transformer trunk**: 28 layers, `d_model=2304`, 24 RoPE heads, SwiGLU FFN (6144 hidden), RMSNorm, 8k train / 32k eval context.
- **Latent workspace**: 64 slots × 1024 dim, slot-attention + GRN updates, learned halting (≤3 steps avg 1.2). Losses: next-latent prediction, sparsity, orthogonality, InfoNCE to answer embedding.
- **Light SSM experts**: Two Mamba-style selective SSM layers (blocks 10 & 20, state 128) gated by workspace summary with z-loss-stabilized router.
- **Memory hierarchy**:
  - Tier A: SnapKV with head-aware keep ratio (int8/int4 streams).
  - Tier B: Episodic FAISS/ScaNN store (Bayesian-surprise writes, TTL 30 days, dedupe).
  - Tier C: Product-Key Memory (4M slots, int8 values) + adapter bank (LoRA r=16, IA³).
  - Tail booster: Optional kNN-LM (5M entries, IVF-PQ, rare-entity routing).
- **Router**: Workspace-conditioned planner toggles SSM, PKM, episodic reads, kNN lambda, adapter set, with entropy floor + z-loss.
- **Answer-only decoding**: Workspace masked from LM head, rationale tokens penalized.

All components live under `quietreasoning/model`, `quietreasoning/memory`, and `quietreasoning/training`.

---

## Data Plan (Auto Pull Supported)

`run.py` can fetch the full mixture automatically (`--auto-pull`) or you can supply your own TFRecords via `--pretrain-pattern`. The default plan mirrors the spec:

| Bucket | Dataset(s) | Ratio | Notes |
| ------ | ---------- | ----: | ----- |
| High-quality filtered web | FineWeb / FineWeb-2 | **55%** | Filtered CommonCrawl, 2024-quality slice. |
| Web with quality signals | RedPajama-Data-v2 | **20%** | Multi-source with annotations for weighting/dedup. |
| Broad corpus | Dolma (filtered) | **10%** | Extra coverage beyond CC-only mixes. |
| Math text | OpenWebMath, Proof-Pile-2 | **8%** | Preserve LaTeX & math reasoning patterns. |
| Code | The Stack v2 | **7%** | Code tokens improve reasoning/tool use. |

Cleaning checklist baked into the plan (language ID ≥0.9, toxicity filters, global+local dedup, math/code retention).

**Alignment & post-training** (not auto-converted to TFRecords yet, but documented for completeness):
- Answer-only SFT: OpenAssistant (OASST1/2) + filtered OpenHermes-2.5.
- Preference/DPO: UltraFeedback (binarized).
- Math SFT: OpenMathInstruct-1.

**Retrieval supervision & memory seeds**:
- PopQA (router supervision).
- Wikipedia + StackExchange dumps for episodic/semantic stores.

---

## Training Schedule (Stages)

| Stage | Tokens | Modules | Purpose |
| ----- | ------ | ------- | ------- |
| `warmstart_dense_ws` | 7e9 | Workspace only | Stabilize latent losses without retrieval. |
| `pkm_adapters` | 7e9 | +PKM +Adapters | Anneal PKM temperature, activate adapters. |
| `ssm_light` | 2e9 | +SSM experts | Track usage histograms, z-loss. |
| `latent_distill` | 1e9 | Quiet-STaR distill | Teacher CoT hidden, answer-only outputs. |
| `retrieval_supervision` | 5e8 | Retrieval on | Route supervision with PopQA. |
| `compress_serve` | final | AWQ + int8 KV | Export weights for deployment. |

Losses and metrics (workspace sparsity/orthogonality, router z-loss, etc.) are implemented in `quietreasoning/training/losses.py`.

---

## Repository Layout
```
quietreasoning/
├── config.py                 # Model & training configs (workspace, memory, stages, optimizer)
├── model/                    # Transformer trunk, workspace, router, SSM, adapters
├── memory/                   # PKM, episodic store, kNN-LM, SnapKV cache
├── training/                 # Losses, data pipelines, stage scheduler, TPU launch, inference
├── eval/                     # PopQA + LongBench evaluators
├── tools/                    # PKM/kNN/episodic index utilities
├── utils/                    # Tokenizer + Bayesian surprise helpers
├── docs/training_recipe.md   # Full recipe & runbook (mirrors spec)
├── run.py                    # End-to-end training + evaluation (auto data pull)
└── train.py                  # Minimal trainer skeleton (dummy batch generator)
spec.md                       # Original engineering spec for Variant-1 (+ light SSM)
```

---

## Quickstart
1. **Install**
   ```bash
   pip install -e .
   ```
2. **Tokenizer choice**
   - For production-quality runs, train a fresh 50k SentencePiece model on your cleaned corpus.
   - For fast iteration, `run.py` defaults to the permissive `mistralai/Mistral-7B-v0.1` tokenizer (32k SPM). It will download this automatically unless you pass `--tokenizer` or `--train-tokenizer`.
3. **Auto-pull data (optional)**
   ```bash
   python run.py \
     --auto-pull \
     --output-dir checkpoints \
     --steps 1000000 \
     --tpu-topology 2x2
   ```
   This samples a configurable number of documents per bucket (`--pull-docs`) into TFRecords under `output-dir/data/`.
4. **Custom data**
   If you already have TFRecords:
   ```bash
   python run.py \
     --pretrain-pattern "gs://bucket/pretrain/*.tfrecord" \
     --tokenizer tokenizer.model \
     --popqa data/popqa.jsonl \
     --output-dir checkpoints
   ```
5. **Memory tooling**
   ```bash
   python -m quietreasoning.tools.build_indices pkm --path artifacts/pkm_values.npy
   python -m quietreasoning.tools.build_indices knn \
     --keys artifacts/knn_keys.npy \
     --values artifacts/knn_vals.npy \
     --output artifacts/knn.index \
     --dim 2304
   ```
   These are not run automatically by `run.py`; execute them when you are ready to seed PKM/kNN memory.
6. **Telemetry (optional)**
   Pass `--wandb-project your_project` (plus optional `--wandb-entity`, `--wandb-tags`) to log training/eval metrics, dataset manifests, and throughput to Weights & Biases. Non-root TPU hosts run in disabled mode.
7. **Multi-host launch**
   Use `launch_tpu.sh` to fan out the command across TPU VM workers:
   ```bash
   export REPO_URL=https://github.com/your/repo.git
   export WANDB_API_KEY=xxxxxxxxxxxxxxxx
   ./launch_tpu.sh PROJECT TPU_NAME ZONE "cd quietreasoning && python run.py --auto-pull --output-dir checkpoints --steps 1000000 --tpu-topology 2x2 --wandb-project quietreasoning --wandb-run train-run-1"
   ```
   Replace `PROJECT`, `TPU_NAME`, and `ZONE` with your setup. The script resolves worker indices and starts the job on each VM so JAX distributed init succeeds.

During training, the runner logs stage-aware metrics, evaluates on PopQA (and LongBench NarrativeQA if downloaded), and saves Orbax checkpoints inside `output-dir`.

---

## Evaluation & Acceptance Targets
- **PopQA**: `quietreasoning.eval.popqa.PopQAEvaluator` (expect +8–10 pts vs closed-book baseline with PKM+episodic, +10–12 with kNN).
- **Long context**: `quietreasoning.eval.long_context.LongContextEvaluator` (SnapKV compression ≤3% RULER/LongBench drop, ≤5% with ZipCache).
- **Answer-only compliance**: Rationale leakage ≤0.5% tokens (penalized in loss).

---

## Additional References
- Quiet-STaR latent reasoning ([arXiv:2403.09629](https://arxiv.org/abs/2403.09629))
- Mamba / selective SSM ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))
- SnapKV, PKM, kNN-LM, AWQ, AdapterFusion/LoRA, FAISS, LongBench — see citations embedded in `spec.md`.

---

## Licensing & Notes
No pretrained weights or raw corpora are shipped. Respect the licenses for all referenced datasets (FineWeb, RedPajama, Dolma, OpenWebMath, Proof-Pile-2, The Stack v2, PopQA, LongBench, etc.) and ensure TPU usage complies with your cloud provider. The code is ready for offline deployment with answer-only decoding to avoid rationale leakage.
