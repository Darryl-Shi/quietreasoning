# Engineering Spec — Variant 1 (+ light SSM experts)

## 0) Scope & Success Criteria

**Goal:** Ship a 3B-class LM that performs latent (non-emissive) reasoning, scales knowledge beyond parameters via memory layers, and runs fully offline.
**Accept** when: (i) answer-only decoding (no CoT leakage) with **≤1.3×** FLOP overhead vs dense baseline, (ii) **≥+5–10 pts** on long-tail recall (PopQA) with memory enabled, (iii) **≤10 GB** RAM for indices at starter scale, (iv) long-context throughput sustained with compressed KV. ([arXiv][1])

---

## 1) Architecture Overview (3.0B dense trunk + light SSM experts)

### 1.1 Model Core

* **Tokenizer:** SentencePiece, vocab=50k.
* **Trunk:** 28 Transformer blocks; **d_model=2304**, **n_heads=24** (RoPE), **FFN=SwiGLU** with inner=6144, norm=RMSNorm. Context 8k (train) → 32k (eval).
* **Latent workspace (non-emissive):** K=64 slots, d_ws=1024; cross-attend to token states; **T=0–3** inner-loop refinements with learned halting; **masked from LM head**.
* **Light SSM experts:** Insert **2 Mamba-style SSM layers** (linear-time selective SSM) at blocks 10 and 20. Each is **conditionally enabled** via a **workspace-gated scalar** g∈[0,1] (top-1 “use or skip”). Router stabilized with **z-loss** and capacity clamp. ([arXiv][2])

### 1.2 Memory Hierarchy & Routing (workspace-routed)

* **Tier A — Short-term KV:** **SnapKV** head-aware selection on prefill (keep 20–40% positions/head) + streaming window=2–4k; **int8/int4 KV** supported. ([NeurIPS Proceedings][3])
* **Tier B — Episodic store (writeable):** FAISS/HNSW/ScaNN index on local NVMe; **event-boundary writes** triggered by Bayesian-surprise spikes + dedupe + TTL. ([GitHub][4])
* **Tier C — Semantic memory:** (i) **Product-Key Memory (PKM)** layer with 4M slots (two 8k codebooks; values 256-d int8; top-32 read); (ii) **Adapter bank** (LoRA r=16, IA³ gating) with optional AdapterFusion at inference. ([NeurIPS Papers][5])
* **Optional tail booster:** **kNN-LM** datastore (5M entries; IVF-PQ/HNSW; λ-interpolation only when router predicts “rare entity”). ([arXiv][6])

### 1.3 Dataflow (inference)

1. Tokens → trunk (lower 20 blocks).
2. Workspace cross-attends → forms **summary h_ws**.
3. Router(h_ws) issues **cost-aware plan**: {enable SSM g, PKM top-k, episodic reads k, kNN-LM λ, adapters set}.
4. Execute plan while running top-8 blocks + inner-loop (0–3 steps) until halting.
5. LM head decodes **answers only** (workspace is masked; rationale tokens explicitly penalized in training). ([arXiv][7])

---

## 2) Module Specs

### 2.1 Latent Workspace Block

* **Update:** Slot-attention style cross-attn (8 heads, q=W slots, k/v=token states) → GRN → residual.
* **Regularizers:** L1 sparsity on slot activations; orthogonality on slot basis (‖WᵀW−I‖); **next-latent prediction** loss (predict W_{t+1}); **InfoNCE** tying final h_ws to correct target answer embedding. ([arXiv][8])
* **Halting:** scalar halting probability from pooled slots; early-exit when Δloss < ε or conf>τ.

### 2.2 SSM Expert (Mamba-style) layer

* **Hidden dim:** 2304; state dim 128; conv kernel 4; selective scan impl.
* **Gating:** g = sigmoid(MLP(h_ws)); output = (1−g)*x + g*SSM(x).
* **Stability:** **z-loss** on router logits + entropy floor; log per-batch “g usage” histogram. ([Hugging Face][9])

### 2.3 PKM layer

* **Keys:** product of two 8k codebooks; PQ-compressed; search top-32.
* **Values:** 256-d int8; projection back to d_model via A 256×2304.
* **Write/refresh:** on training steps only; EMA updates + temperature annealing to avoid collapse. ([NeurIPS Papers][5])

### 2.4 Episodic store

* **Encoder:** mean-pool last-layer token reps at event boundary.
* **Index:** FAISS IVF-PQ (nlist=4096, m=32) or HNSW (M=32, ef=128).
* **Write trigger:** Bayesian surprise z_t > μ+σ; cooldown 64 tokens; de-dup cosine>0.92; TTL=30 days. ([NIPS Papers][10])

### 2.5 KV cache & long-context

* **Prefill:** SnapKV picks head-wise salient positions from an observation window (e.g., last 128).
* **Decode:** streaming window with sinks; **INT8/INT4** KV via LMDeploy-style online quant. Optionally ZipCache/PQCache for higher compression. ([NeurIPS Proceedings][3])

---

## 3) Concrete Hyperparameters

| Component                | Value (tunable)                             |
| ------------------------ | ------------------------------------------- |
| Layers / d_model / heads | 28 / 2304 / 24                              |
| FFN inner                | 6144 (SwiGLU)                               |
| Context window           | 8k train → 32k eval                         |
| Workspace slots / dim    | **K=64 / d_ws=1024**                        |
| Inner-loop max steps     | **T=3** (avg ~1.2)                          |
| SSM experts              | **2 layers** (blocks 10 & 20), state=128    |
| PKM                      | **4M** slots; top-32; value 256-d int8      |
| Adapters                 | LoRA r=16 on attention/FFN; IA³ on FFN      |
| kNN-LM                   | 5M entries, IVF-PQ; λ∈[0,0.5] (rare-only)   |
| KV                       | SnapKV keep 20–40% head-positions; **int8** |

---

## 4) Training Recipe (staged)

### 4.1 Data & teacher

* **Pretrain corpora:** **FineWeb** (quality-filtered CC; 15T tokens) + **SlimPajama-627B** + **RedPajama-v2** (with quality signals). Start from an open 1–3B base or train 50–100B tokens. ([NeurIPS Papers][11])
* **Teacher for latent distill:** ≥13B strong model producing CoT; CoT used **only** as latent target (no emission). **Quiet-STaR** objective for token-wise internal rationales. ([arXiv][1])

### 4.2 Optimizer & schedules (all stages)

* **AdamW** (β1=0.9, β2=0.95, wd=0.05); cosine LR.
* **LRs:** Stage-0 pretrain peak 2.5e-4; later stages 1e-4→5e-5.
* **Batching:** global 2–4M tokens/step (gradient-accum).
* **Stability:** grad-clip 1.0; norm-loss 1e-3; z-loss 1e-4 (routers). ([Hugging Face][9])

### 4.3 Stages

**Stage 0 — Dense + workspace warm-start (7–10B tokens)**

* Enable workspace from step 0 with **aux losses only** (next-latent; sparsity; orthogonality). No retrieval/PKM yet.

**Stage 1 — Turn on PKM + adapters (5–10B tokens)**

* Insert PKM in upper third of blocks; anneal memory temp from 2.0→0.7 over 100k steps; LoRA r=16 on attn & FFN; IA³ on FFN. ([NeurIPS Papers][5])

**Stage 2 — Light SSM experts (2–3B tokens)**

* Add two Mamba layers; **workspace-gated usage**; apply **z-loss** + entropy penalty to keep average g≈0.4 with stdev>0.1. ([arXiv][2])

**Stage 3 — Latent distillation (1–3B tokens)**

* Teacher generates CoT; student gets **latent targets** via auxiliary loss on workspace states (no visible rationales). Mix in answer-only NLL. ([arXiv][7])

**Stage 4 — Retrieval supervision (episodic & kNN) (≤1B tokens)**

* On **PopQA**/rare-entity sets: force router to query episodic/kNN when popularity<τ; cross-entropy on router decisions + λ for interpolation. ([arXiv][12])

**Stage 5 — Consolidation jobs (offline)**

* Promote high-hit episodic items into adapters/PKM (shadow A/B; rollback snapshots).

**Stage 6 — Compression & serving**

* **AWQ** W4/W3 weight-only quant; **INT8/INT4 KV** (LMDeploy/vLLM configs); optional **ZipCache/PQCache** for extreme contexts. ([proceedings.mlsys.org][13])

---

## 5) Evaluation & Gates (core)

* **Reasoning (answer-only):** GSM8K/ARC-C/BBH slices.
* **Long-tail factual:** **PopQA** closed-book vs **+PKM** vs **+kNN**. ([arXiv][12])
* **Long-context:** **RULER** + **LongBench** (throughput & accuracy with KV compression toggles). Pass if **≤3%** drop at 5–10× KV compression (SnapKV/ZipCache). ([arXiv][14])

---

## 6) Systems & Serving

### 6.1 TPU training

* **Hardware:** 8–16× TPU-v4; **bf16**; activation checkpointing.
* **Sharding:** tensor-parallel=2, pipeline-parallel=2 (upper blocks pipelined with workspace/PKM).
* **Datastores:** PKM values on HBM; PKM keys (PQ) and episodic/kNN indices on host RAM/NVMe with fused batched RPC.

### 6.2 Inference (offline)

* **Quantization:** AWQ W4; KV **int8** default; enable **SnapKV** on prefill; sliding window 2–4k on decode. ([proceedings.mlsys.org][13])
* **Planner:** per-request budgets (e.g., ≤2 PKM reads; ≤1 episodic query; ≤32 kNN neighbors).
* **Observability:** log (i) router decisions, (ii) PKM/kNN/episodic hit-rates, (iii) KV compression ratio vs latency.

---

## 7) Risk Controls (run-time & training)

* **Latent leakage:** penalize rationale tokens; mask workspace to head; audit n-gram markers. ([arXiv][7])
* **Router collapse:** track entropy & per-path utilization; apply **z-loss** and capacity factors. ([Hugging Face][9])
* **KV over-compression:** auto-fallback from int4→int8 on spike in loss/latency; SnapKV keep-ratio floor. ([NeurIPS Proceedings][3])

---

## 8) Config Skeletons

### 8.1 Model (YAML)

```yaml
model:
  arch: transformer
  layers: 28
  d_model: 2304
  n_heads: 24
  ffn_inner: 6144
  rope: true
  norm: rmsnorm
  context: 8192
workspace:
  slots: 64
  dim: 1024
  inner_loop_max: 3
  losses: {next_latent:1.0, sparsity:0.1, orth:0.05, infoNCE:0.2}
experts:
  ssm:
    blocks: [10, 20]
    state_dim: 128
    gating: workspace_scalar
    z_loss: 1.0e-4
memory:
  kv:
    snapkv_keep_frac: 0.3
    quant: int8
    stream_window: 4096
  episodic:
    backend: faiss_ivfpq
    params: {nlist:4096, m:32, nprobe:16}
    ttl_days: 30
    write_trigger: bayesian_surprise
  pkm:
    slots: 4000000
    codebooks: 8192
    value_dim: 256
    topk: 32
  knn_lm:
    enable: true
    index: ivf_pq
    size: 5000000
    lambda_max: 0.5
adapters:
  lora_r: 16
  ia3: true
```

### 8.2 Training stages (YAML)

```yaml
train:
  optimizer: {name: adamw, lr: 2.5e-4, betas:[0.9,0.95], wd:0.05}
  clip_grad_norm: 1.0
  batch_tokens: 3000000
stages:
  - name: warmstart_dense_ws
    tokens: 7e9
    enable: {workspace:true, pkm:false, adapters:false, ssm:false, retrieval:false}
  - name: pkm_adapters
    tokens: 7e9
    enable: {pkm:true, adapters:true}
    lr_peak: 1.0e-4
  - name: ssm_light
    tokens: 2e9
    enable: {ssm:true}
  - name: latent_distill
    tokens: 1e9
    distill: {teacher_cot:true, latent:true, visible_rationales:false}
  - name: retrieval_supervision
    tokens: 5e8
    datasets: [PopQA]
    router_supervision: true
  - name: compress_serve
    quant: {awq_w4:true, kv:int8}
```

---

## 9) Key Implementation Notes (concise)

### 9.1 Workspace & inner loop (pseudo-code)

```python
def workspace_block(h_tok, W, steps=3):
    for t in range(steps):
        W = cross_attn(query=W, key=h_tok, value=h_tok) + W
        W = grn(ffn(W))                                  # gated res norm
        if halt(W): break
    h_ws = mean_pool(W)                                  # summary
    return h_ws, W
```

### 9.2 Router (workspace-routed)

```python
def plan(h_ws):
    g_ssm = sigmoid(MLP(h_ws))                 # SSM on/off
    use_pkm, topk = gate_pkm(h_ws)             # product-key read
    use_knn, lam  = gate_knn(h_ws)             # for rare entities
    episodic_k    = gate_episodic(h_ws)        # ANN k
    adapters      = select_adapters(h_ws)      # LoRA/IA3 set
    L_z = z_loss(router_logits) + lb_loss()    # stability
    return Decisions(...), L_z
```

### 9.3 PKM read (sketch)

```python
idx, scr = product_key_lookup(q=h_ws)   # PQ keys, top-32
v = values[idx]                          # int8 -> dequant -> 256d
h = (scr @ v) @ W_v2h                    # 256->d_model
```

---

## 10) Runbook

### 10.1 Build indices (offline, small start)

* **PKM:** init random, trainable in graph.
* **kNN-LM:** encode 5M contexts → FAISS IVF-PQ (m=32); store fp16 centroids. ([GitHub][4])
* **Episodic:** start empty; writes gated by event-boundary detector.

### 10.2 Training (TPU; PyTorch-XLA or JAX)

1. Stage-0: launch pretrain; monitor **latent losses** only.
2. Stage-1: enable PKM/adapters; anneal memory temp; watch PKM hit-entropy.
3. Stage-2: enable SSM; track **g usage hist**. ([arXiv][2])
4. Stage-3: run **Quiet-STaR** latent distill; ensure **answer-only outputs** on eval. ([arXiv][1])
5. Stage-4: route-supervise on **PopQA**; tune λ. ([arXiv][12])
6. Stage-6: export **AWQ** + **int8 KV**; validate RULER/LongBench under SnapKV/ZipCache. ([proceedings.mlsys.org][13])

---

## 11) Acceptance Tests (must pass)

* **No-CoT leakage:** ≤0.5% generated tokens match rationale lexicon; manual spot-checks. ([arXiv][1])
* **Long-tail** (PopQA): **+8 pts** vs closed-book baseline with {PKM+Episodic} on; **+10–12 pts** with {+kNN}. ([arXiv][12])
* **Long-context:** With SnapKV (×5–10 KV compression), **≤3%** drop on RULER/LongBench and **≥25%** memory saving; with ZipCache/PQCache extended compression **≤5%** drop. ([NeurIPS Proceedings][3])
* **Latency budget:** avg inner-loop steps ≤1.2; PKM/kNN lookups ≤3 per answer.

---

### References (key evidence)

Quiet-STaR latent reasoning; PKM; kNN-LM; SnapKV; LMDeploy KV-quant; Mamba/SSM; AdapterFusion/LoRA; FAISS; Long-context eval (RULER/LongBench); PopQA. ([arXiv][1])


[1]: https://arxiv.org/abs/2403.09629?utm_source=chatgpt.com "Quiet-STaR: Language Models Can Teach Themselves to ..."
[2]: https://arxiv.org/abs/2312.00752?utm_source=chatgpt.com "Linear-Time Sequence Modeling with Selective State Spaces"
[3]: https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf?utm_source=chatgpt.com "SnapKV: LLM Knows What You Are Looking for before ..."
[4]: https://github.com/facebookresearch/faiss?utm_source=chatgpt.com "facebookresearch/faiss: A library for efficient similarity ..."
[5]: https://papers.neurips.cc/paper/9061-large-memory-layers-with-product-keys.pdf?utm_source=chatgpt.com "Large Memory Layers with Product Keys"
[6]: https://arxiv.org/abs/1911.00172?utm_source=chatgpt.com "Generalization through Memorization: Nearest Neighbor ..."
[7]: https://arxiv.org/html/2403.09629v1?utm_source=chatgpt.com "Quiet-STaR: Language Models Can Teach Themselves to ..."
[8]: https://arxiv.org/pdf/1807.03748?utm_source=chatgpt.com "Representation learning with contrastive predictive coding"
[9]: https://huggingface.co/blog/moe?utm_source=chatgpt.com "Mixture of Experts Explained"
[10]: https://papers.nips.cc/paper/2822-bayesian-surprise-attracts-human-attention?utm_source=chatgpt.com "Bayesian Surprise Attracts Human Attention"
[11]: https://papers.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf?utm_source=chatgpt.com "The FineWeb Datasets: Decanting the Web for the Finest ..."
[12]: https://arxiv.org/abs/2212.10511?utm_source=chatgpt.com "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories"
[13]: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for On-Device ..."
[14]: https://arxiv.org/abs/2404.06654?utm_source=chatgpt.com "RULER: What's the Real Context Size of Your Long ..."
