# Quiet Reasoning Training Recipe (TPU)

This document summarizes the end-to-end procedure for training the 3B Quiet Reasoning model with latent workspace, memory routing, and TPU-optimized execution.

## 1. Environment Setup
- **Hardware**: TPU v4-8 (minimum) or multi-host pod. Configure topology via `TPU_CHIPS_PER_HOST_BOUNDS`.
- **Software**: Python ≥3.10, CUDA not required; install project deps with `pip install -e .` (pulls JAX TPU wheels, Flax, Optax, FAISS, ScaNN, etc.).
- **Launch**: Use `quietreasoning.training.launch.initialize_distributed` to set coordinator address when scaling beyond one host.

## 2. Data Preparation
1. **Tokenizer**: SentencePiece 50k vocabulary. Ensure `config.model.tokenizer_path` points to the trained `.model` file.
2. **Pretraining corpus**: Create TFRecords with schema `{"text": tf.string}` and host them on GCS or local NVMe. File patterns feed `quietreasoning.training.data.build_pretrain_dataset`.
3. **Routed supervision**:
   - PopQA JSONL for long-tail questions (`{"question": str, "answer": str}`) used during Stage `retrieval_supervision`.
   - Long-context corpora (RULER/LongBench JSON) for evaluation, not backprop.

## 3. Memory Index Bootstrap
Run utilities under `quietreasoning/tools/build_indices.py`:

```bash
# Initialize Product Key Memory values
python -m quietreasoning.tools.build_indices pkm \
  --path artifacts/pkm_values.npy --slots 4000000 --value-dim 256

# Build kNN-LM FAISS index (keys.npy: float32 states, values.npy: int labels)
python -m quietreasoning.tools.build_indices knn \
  --keys artifacts/knn_keys.npy \
  --values artifacts/knn_vals.npy \
  --output artifacts/knn.index \
  --dim 2304 --factory IVF4096,PQ32

# Optional sanity check
python -m quietreasoning.tools.build_indices verify-knn \
  --index artifacts/knn.index --queries artifacts/knn_queries.npy
```

Episodic memory starts empty by default; populate with curated events via `episodic` command if desired.

## 4. Training Stages Overview
Stages are defined in `quietreasoning.config.TrainingConfig`. Each stage toggles modules and supervises specific losses:

| Stage | Tokens | Enabled Modules | Notes |
|-------|--------|-----------------|-------|
| `warmstart_dense_ws` | 7e9 | Trunk + workspace (PKM/SSM off) | Focus on latent losses without retrieval. |
| `pkm_adapters` | 7e9 | Adds PKM + adapters | Anneal PKM temperature, monitor hit entropy. |
| `ssm_light` | 2e9 | Enables Mamba SSM experts | Track `g_usage` via router logs. |
| `latent_distill` | 1e9 | Distill Quiet-STaR latent signals, rationale mask active | Only answer tokens allowed. |
| `retrieval_supervision` | 5e8 | Activates episodic + kNN routing | Optimize λ and router supervision with PopQA. |
| `compress_serve` | until convergence | Apply AWQ/int8 KV, export deployment weights | Optional fine-tuning for serving. |

Gradually unfreeze modules by stage-specific feature gates (workspace, PKM, adapters, SSM, retrieval, kNN).

## 5. Launching Training
Example single-host TPU script (`train.py` uses dummy data—replace with real loader):

```python
from quietreasoning import QuietReasoningConfig, create_train_state, build_train_step
from quietreasoning.training.data import build_pretrain_dataset, DatasetConfig
from quietreasoning.training.stages import StageScheduler

cfg = QuietReasoningConfig()
state = create_train_state(rng, cfg)
step_fn = build_train_step(cfg)
scheduler = StageScheduler(cfg.training)
dataset = build_pretrain_dataset(
    DatasetConfig(file_pattern="gs://bucket/pretrain/*.tfrecord",
                  batch_size=64, sequence_length=cfg.model.context),
    tokenizer_path=Path("tokenizer.model"),
    shard_id=jax.process_index(),
    num_shards=jax.process_count(),
)
```

Within the training loop:
1. Fetch batch from dataset iterator.
2. Query stage: `stage = scheduler.stage_at(float(state.tokens_seen))`.
3. Build feature gate dict `{name: jnp.array(1.0 if enabled else 0.0)}`.
4. Call `state, logs = step_fn(state, batch, tokens_per_step, feature_gates)`.
5. Log `logs` (loss, workspace metrics, router penalties) to TensorBoard or Bigtable.

For multi-host pods, wrap run with `quietreasoning.training.launch.initialize_distributed`.

## 6. Monitoring & Diagnostics
- **Workspace**: Track `workspace_steps`, sparsity, orthogonality losses. Ensure average inner-loop steps ≤1.2.
- **Router**: Monitor `router_aux` z-loss/entropy loss, SSM gate usage histograms.
- **Memory**: Measure PKM hit rate, episodic/kNN activation frequency. Use `verify-knn` to validate indices after updates.
- **KV cache**: During long-context eval, watch compression ratio from `SnapKVCache` and fallback rate to int8.

## 7. Evaluation
- **PopQA**: Use `quietreasoning.eval.popqa.PopQAEvaluator` with latest checkpoints; expect +8–10 points vs closed-book baseline.
- **RULER/LongBench**: Build dataset iterators producing `{"document","question","answer"}`. Run `LongContextEvaluator`; target ≤3% degradation with SnapKV int8 and ≤5% with further ZipCache compression.
- **Answer-only compliance**: Sample generations, ensure ≤0.5% rationale leakage. Increase penalty weight if leakage grows.

## 8. Export & Deployment
1. Apply AWQ weight quantization and int8 KV settings (Stage `compress_serve`).
2. Save checkpoints via Orbax; include memory artifacts (PKM values, kNN index).
3. For offline inference, package:
   - `params` checkpoint
   - `artifacts/pkm_values.npy`, `knn.index`, episodic snapshots
   - Tokenizer model
   - Decoder configuration with rationale penalty settings

Follow LMDeploy/ZiPCache integration docs to serve with compressed KV caches.

## 9. Troubleshooting Tips
- **Unstable router**: Increase entropy floor or z-loss scale, or cap top-k reads.
- **PKM collapse**: Raise temperature or add EMA smoothing for value updates.
- **Latency spikes**: Reduce episodic `k` or enforce SnapKV fallback to int8.
- **Memory overflow**: Lower `snapkv_keep_frac` or shorten stream window during long contexts.

This recipe mirrors `spec.md` acceptance criteria while providing concrete steps to reproduce TPU training runs.

