"""End-to-end Quiet Reasoning training runner with automatic data pulls."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import time
import os
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import jax
import jax.distributed as jdist
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from jax.experimental import multihost_utils
from huggingface_hub import hf_hub_download
from orbax import checkpoint as orbax_checkpoint

from quietreasoning import QuietReasoningConfig, build_train_step, create_train_state
from quietreasoning.eval.long_context import LongContextEvaluator
from quietreasoning.eval.popqa import PopQAEvaluator
from quietreasoning.training.data import DatasetConfig, build_popqa_dataset, build_pretrain_dataset
from quietreasoning.training.launch import TPULaunchConfig, initialize_distributed
from quietreasoning.training.stages import StageScheduler
from quietreasoning.utils.tokenizer import SentencePieceTokenizer

LOGGER = logging.getLogger("quietreasoning.run")


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class DataSource:
    name: str
    dataset_id: str
    subset: Optional[str]
    split: str
    ratio: float
    text_field: Optional[str] = "text"


PRETRAIN_PLAN: List[DataSource] = [
    DataSource("fineweb1", "HuggingFaceFW/fineweb", None, "train", 0.60, "text"),
    DataSource("redpajama_v2", "togethercomputer/RedPajama-Data-v2", "default", "train", 0.20, "text"),
    # Dolma script is unsupported in streaming loader; user can add custom corpora externally.
    DataSource("openwebmath", "open-web-math/open-web-math", None, "train", 0.07, "content"),
    DataSource("proof_pile_2", "xavierdurawa/proof-pile-2-streaming", None, "train", 0.03, "content"),
    DataSource("the_stack_v2", "bigcode/the-stack-v2", "default", "train", 0.10, "content"),
]


@dataclass
class DataManifest:
    pretrain_glob: Path
    popqa_path: Path
    long_context_path: Optional[Path]
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quiet Reasoning training runner")
    parser.add_argument("--pretrain-pattern", type=str, default="", help="TFRecord glob for pretraining data")
    parser.add_argument("--tokenizer", type=Path, default=None, help="SentencePiece tokenizer path (local file)")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train a custom SentencePiece tokenizer (not implemented)")
    parser.add_argument("--pretrained-tokenizer", type=str, default="mistralai/Mistral-7B-v0.1", help="HF repo for pretrained tokenizer")
    parser.add_argument("--tokenizer-file", type=str, default="tokenizer.model", help="Filename inside HF repo to fetch")
    parser.add_argument("--popqa", type=Path, default=None, help="PopQA dataset (jsonl); auto-downloaded if absent")
    parser.add_argument("--long-context", type=Path, default=None, help="Long context evaluation set (jsonl); optional")
    parser.add_argument("--output-dir", type=Path, required=True, help="Checkpoint and log directory")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--eval-every", type=int, default=10_000, help="Evaluation frequency (steps)")
    parser.add_argument("--log-every", type=int, default=100, help="Logging frequency (steps)")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size (tokens_per_step = batch_size * context)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tpu-topology", type=str, default="", help="Topology string (e.g., 2x2); if empty no distributed init")
    parser.add_argument("--coordinator", type=str, default="", help="Fully qualified coordinator address host:port")
    parser.add_argument("--eval-samples", type=int, default=512, help="Number of PopQA examples to evaluate")
    parser.add_argument("--long-samples", type=int, default=64, help="Number of long-context samples to evaluate")
    parser.add_argument("--auto-pull", action="store_true", help="Automatically download dataset plan into output/data/")
    parser.add_argument("--pull-docs", type=int, default=50_000, help="Total documents to sample for auto-pulled pretrain TFRecords")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--wandb-project", type=str, default="", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="Weights & Biases entity")
    parser.add_argument("--wandb-run", type=str, default="", help="Weights & Biases run name")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated list of W&B tags")
    return parser.parse_args()


def maybe_initialize_tpu(args: argparse.Namespace) -> None:
    if args.tpu_topology:
        address, port = (args.coordinator.split(":") + ["12345"])[:2] if args.coordinator else ("", "12345")
        process_id = int(os.environ.get("JAX_PROCESS_INDEX", 0))
        process_count = int(os.environ.get("JAX_PROCESS_COUNT", 1))
        cfg = TPULaunchConfig(
            topology=args.tpu_topology,
            coordinator_address=address or None,
            coordinator_port=int(port),
            process_id=process_id,
            process_count=process_count,
        )
        initialize_distributed(cfg)


def take_examples(stream: Iterable[dict], limit: int) -> Iterator[dict]:
    for idx, example in enumerate(stream):
        if idx >= limit:
            break
        yield example


def extract_text(example: dict, preferred_key: Optional[str]) -> Optional[str]:
    candidates = []
    if preferred_key and preferred_key in example:
        candidates.append(example[preferred_key])
    for key in ("text", "content", "body", "cleaned_text", "input", "question"):
        if key in example:
            candidates.append(example[key])
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, list):
            value = " ".join([str(v) for v in value if v])
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        value = str(value).strip()
        if value:
            return value
    return None


def serialize_text(text: str) -> bytes:
    feature = {
        "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode("utf-8")])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def download_pretrain_sources(output_dir: Path, total_docs: int) -> Path:
    data_dir = output_dir / "data" / "pretrain"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    LOGGER.info("Auto-pulling pretrain datasets (total docs target=%d)...", total_docs)
    for source in PRETRAIN_PLAN:
        target = max(1, int(total_docs * source.ratio))
        tfrecord_path = data_dir / f"{source.name}.tfrecord"
        if tfrecord_path.exists():
            LOGGER.info("Skipping %s (already exists)", source.name)
            manifest.append({"name": source.name, "path": str(tfrecord_path), "examples": "existing"})
            continue
        LOGGER.info("Downloading %s (%s), target docs=%d", source.name, source.dataset_id, target)
        subset = source.subset
        if subset is None:
            try:
                from datasets import get_dataset_config_names

                config_names = get_dataset_config_names(source.dataset_id)
            except Exception as err:
                LOGGER.warning("Unable to list configs for %s: %s", source.dataset_id, err)
                config_names = []
            if config_names:
                subset = config_names[0]
                LOGGER.info("Using config %s for dataset %s", subset, source.dataset_id)
        if subset is None:
            dataset = load_dataset(
                source.dataset_id,
                split=source.split,
                streaming=True,
            )
        else:
            dataset = load_dataset(
                source.dataset_id,
                subset,
                split=source.split,
                streaming=True,
            )
        count = 0
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for example in take_examples(dataset, target):
                text = extract_text(example, source.text_field)
                if not text:
                    continue
                writer.write(serialize_text(text))
                count += 1
        LOGGER.info("Wrote %d examples to %s", count, tfrecord_path)
        manifest.append({"name": source.name, "path": str(tfrecord_path), "examples": count})
    manifest_path = output_dir / "data" / "pretrain_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return Path(str(data_dir / "*.tfrecord"))


def download_popqa_dataset(output_dir: Path, eval_samples: int) -> Path:
    target_path = output_dir / "data" / "popqa.jsonl"
    if target_path.exists():
        return target_path
    LOGGER.info("Auto-pulling PopQA (samples=%d)", eval_samples)
    dataset = load_dataset("akariasai/PopQA", split="test", streaming=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w") as f:
        for record in take_examples(dataset, eval_samples):
            question = record.get("question") or record.get("query") or ""
            answer = record.get("answer") or record.get("answer_aliases", [""])[0]
            json.dump({"question": question, "answer": answer}, f)
            f.write("\n")
    return target_path


LONG_BENCH_REPO = "THUDM/LongBench"
LONG_BENCH_ARCHIVE = "data.zip"
LONG_BENCH_MEMBER = "data/narrativeqa.jsonl"


def download_long_context_dataset(output_dir: Path, long_samples: int) -> Optional[Path]:
    target_path = output_dir / "data" / "long_context.jsonl"
    if target_path.exists():
        return target_path
    LOGGER.info("Auto-pulling LongBench NarrativeQA (samples=%d)", long_samples)
    try:
        archive_path = hf_hub_download(LONG_BENCH_REPO, filename=LONG_BENCH_ARCHIVE, repo_type="dataset")
    except Exception as exc:  # pragma: no cover - optional dataset
        LOGGER.warning("Failed to download LongBench archive: %s", exc)
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(LONG_BENCH_MEMBER) as member:
                with target_path.open("w") as f:
                    for raw_line in member:
                        record = json.loads(raw_line.decode("utf-8"))
                        document = record.get("context") or record.get("document") or record.get("input") or ""
                        question = record.get("question") or record.get("query") or ""
                        answers = record.get("answers") or record.get("answer") or record.get("output") or ""
                        if isinstance(answers, list):
                            answer = answers[0]
                        else:
                            answer = answers
                        json.dump(
                            {
                                "document": str(document),
                                "question": str(question),
                                "answer": str(answer),
                            },
                            f,
                        )
                        f.write("\n")
                        count += 1
                        if count >= long_samples:
                            break
    except KeyError as exc:  # pragma: no cover - optional dataset
        LOGGER.warning("LongBench archive missing %s: %s", LONG_BENCH_MEMBER, exc)
        target_path.unlink(missing_ok=True)
        return None
    except Exception as exc:  # pragma: no cover - optional dataset
        LOGGER.warning("Failed to unpack LongBench NarrativeQA split: %s", exc)
        target_path.unlink(missing_ok=True)
        return None
    if count == 0:
        LOGGER.warning("LongBench NarrativeQA yielded no records; check archive integrity")
        target_path.unlink(missing_ok=True)
        return None
    return target_path


def ensure_pretrained_tokenizer(args: argparse.Namespace) -> Path:
    if args.tokenizer and Path(args.tokenizer).exists():
        return Path(args.tokenizer)
    if args.train_tokenizer:
        raise NotImplementedError(
            "Custom tokenizer training is not implemented in run.py. "
            "Please train a SentencePiece model offline and pass --tokenizer."
        )
    target_dir = args.output_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / args.tokenizer_file
    if destination.exists():
        LOGGER.info("Using existing tokenizer at %s", destination)
        return destination
    LOGGER.info(
        "Downloading pretrained tokenizer %s:%s",
        args.pretrained_tokenizer,
        args.tokenizer_file,
    )
    downloaded_path = hf_hub_download(
        repo_id=args.pretrained_tokenizer,
        filename=args.tokenizer_file,
    )
    shutil.copy(downloaded_path, destination)
    LOGGER.info("Saved tokenizer to %s", destination)
    return destination


def init_wandb_run(args: argparse.Namespace, cfg: QuietReasoningConfig, tokenizer_path: Path):
    if not args.wandb_project:
        return None
    import wandb  # type: ignore

    config_payload = {
        "model": asdict(cfg.model),
        "training": asdict(cfg.training),
        "tokenizer": str(tokenizer_path),
        "auto_pull": bool(args.auto_pull),
    }
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run or None,
        tags=tags or None,
        config=config_payload,
    )
    LOGGER.info("Initialized Weights & Biases run: %s", run.name)
    return run


def to_python_scalar(value):
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        value = np.asarray(value)
        if value.size == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, (float, int)):
        return float(value)
    return value


def autopull_data(args: argparse.Namespace, wandb_run=None) -> DataManifest:
    output_dir = args.output_dir
    manifest_path = output_dir / "data" / "data_manifest.json"

    if jax.process_index() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        pretrain_glob = download_pretrain_sources(output_dir, args.pull_docs)
        popqa_path = download_popqa_dataset(output_dir, args.eval_samples)
        long_path = download_long_context_dataset(output_dir, args.long_samples)
        manifest_payload = {
            "pretrain_glob": str(pretrain_glob),
            "popqa": str(popqa_path),
            "long_context": str(long_path) if long_path else "",
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            json.dump(manifest_payload, f, indent=2)
        if wandb_run is not None:
            wandb_run.log({"data/pretrain_docs_target": args.pull_docs})
            wandb_run.config.update({"data_manifest": manifest_payload}, allow_val_change=True)
    multihost_utils.sync_global_devices("data_autopull_barrier")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    with manifest_path.open() as f:
        payload = json.load(f)
    pretrain_glob = Path(payload["pretrain_glob"])
    popqa_path = Path(payload["popqa"])
    long_context_path = Path(payload["long_context"]) if payload.get("long_context") else None
    return DataManifest(
        pretrain_glob=pretrain_glob,
        popqa_path=popqa_path,
        long_context_path=long_context_path,
        manifest_path=manifest_path,
    )


def prepare_datasets(
    cfg: QuietReasoningConfig,
    args: argparse.Namespace,
) -> Iterator[Dict[str, np.ndarray]]:
    if not args.pretrain_pattern:
        raise ValueError("Pretrain pattern must be supplied via --pretrain-pattern or --auto-pull.")
    dataset_cfg = DatasetConfig(
        file_pattern=args.pretrain_pattern,
        batch_size=args.batch_size,
        sequence_length=cfg.model.context,
    )
    dataset = build_pretrain_dataset(
        dataset_cfg,
        tokenizer_path=args.tokenizer,
        shard_id=jax.process_index(),
        num_shards=jax.process_count(),
    )
    return dataset


def sample_iterator(it: Iterable[dict], limit: int) -> list[dict]:
    return list(itertools.islice(it, limit))


def save_checkpoint(checkpointer: orbax_checkpoint.Checkpointer, directory: Path, state, step: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ckpt_path = directory / f"step_{step}"
    LOGGER.info("Saving checkpoint to %s", ckpt_path)
    checkpointer.save(str(ckpt_path), state)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    maybe_initialize_tpu(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ensure_pretrained_tokenizer(args)
    args.tokenizer = tokenizer_path

    cfg = QuietReasoningConfig()
    cfg.model.tokenizer_path = str(tokenizer_path)

    wandb_run = init_wandb_run(args, cfg, tokenizer_path)

    if args.auto_pull:
        manifest = autopull_data(args, wandb_run)
        if not args.pretrain_pattern:
            args.pretrain_pattern = str(manifest.pretrain_glob)
        if args.popqa is None:
            args.popqa = manifest.popqa_path
        if args.long_context is None and manifest.long_context_path is not None:
            args.long_context = manifest.long_context_path

    if not args.pretrain_pattern:
        raise ValueError("No pretrain data supplied. Provide --pretrain-pattern or enable --auto-pull.")
    if args.popqa is None:
        raise ValueError("PopQA path is required or enable --auto-pull to fetch it automatically.")

    rng = jax.random.PRNGKey(args.seed + jax.process_index())
    state = create_train_state(rng, cfg)
    scheduler = StageScheduler(cfg.training)
    dataset_iter = prepare_datasets(cfg, args)

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    popqa_data = sample_iterator(build_popqa_dataset(args.popqa), args.eval_samples)
    popqa_eval = PopQAEvaluator(cfg, tokenizer)

    long_eval = None
    long_data: List[dict] = []
    if args.long_context and Path(args.long_context).exists():
        with Path(args.long_context).open("r") as f:
            for line in itertools.islice(f, args.long_samples):
                long_data.append(json.loads(line))
        if long_data:
            long_eval = LongContextEvaluator(cfg, tokenizer)

    step_fn = build_train_step(cfg)
    tokens_per_step = args.batch_size * cfg.model.context
    checkpointer = orbax_checkpoint.PyTreeCheckpointer()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.time()
    for step in range(args.steps):
        step_wall = time.time()
        stage = scheduler.stage_at(float(state.tokens_seen))
        batch = next(dataset_iter)
        feature_gates = {
            name: jnp.array(1.0 if enabled else 0.0, dtype=jnp.float32)
            for name, enabled in stage.features.items()
        }
        state, logs = step_fn(state, batch, float(tokens_per_step), feature_gates)
        step_duration = time.time() - step_wall

        if step % args.log_every == 0 and jax.process_index() == 0:
            LOGGER.info(
                "step=%d stage=%s loss=%.4f workspace_steps=%.3f",
                step,
                stage.stage.name,
                float(logs["loss"]),
                float(logs["workspace_steps"]),
            )
            if wandb_run is not None:
                metrics = {
                    "train/loss": to_python_scalar(logs.get("loss")),
                    "train/workspace_steps": to_python_scalar(logs.get("workspace_steps")),
                    "train/ssm_gate": to_python_scalar(logs.get("ssm_gate", 0.0)),
                    "train/router_z_loss": to_python_scalar(logs.get("router_z_loss", 0.0)),
                    "train/router_entropy_loss": to_python_scalar(logs.get("router_entropy_loss", 0.0)),
                    "train/stage_index": stage.index,
                    "train/stage_name": stage.stage.name,
                    "train/stage_tokens": stage.consumed_tokens,
                    "train/tokens_seen": to_python_scalar(state.tokens_seen),
                    "time/step_sec": step_duration,
                    "time/tokens_per_sec": tokens_per_step / max(step_duration, 1e-6),
                }
                wandb_run.log(metrics, step=step)

        if step % args.eval_every == 0 and step > 0 and jax.process_index() == 0:
            LOGGER.info("Running PopQA evaluation...")
            pq_metrics = popqa_eval.evaluate(state.params, popqa_data)
            LOGGER.info(
                "PopQA: accuracy=%.3f recall@5=%.3f total=%d",
                pq_metrics.accuracy,
                pq_metrics.recall_at_5,
                pq_metrics.total_examples,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/popqa_accuracy": pq_metrics.accuracy,
                        "eval/popqa_recall_at5": pq_metrics.recall_at_5,
                        "eval/popqa_total": pq_metrics.total_examples,
                    },
                    step=step,
                )
            if long_eval is not None and long_data:
                LOGGER.info("Running long-context evaluation...")
                lc_metrics = long_eval.evaluate(state.params, long_data)
                LOGGER.info(
                    "LongCtx: EM=%.3f F1=%.3f latency=%.1fms total=%d",
                    lc_metrics.exact_match,
                    lc_metrics.f1,
                    lc_metrics.latency_ms,
                    lc_metrics.total_examples,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/longctx_em": lc_metrics.exact_match,
                            "eval/longctx_f1": lc_metrics.f1,
                            "eval/longctx_latency_ms": lc_metrics.latency_ms,
                            "eval/longctx_total": lc_metrics.total_examples,
                        },
                        step=step,
                    )
            save_checkpoint(checkpointer, output_dir, state, step)
            if wandb_run is not None:
                wandb_run.log({"checkpoints/last_step": step}, step=step)

    if jax.process_index() == 0:
        save_checkpoint(checkpointer, output_dir, state, args.steps)
        LOGGER.info("Training complete.")
        if wandb_run is not None:
            total_time = time.time() - train_start
            wandb_run.log({"time/total_training_sec": total_time, "training/steps": args.steps}, step=args.steps)
            wandb_run.finish()


if __name__ == "__main__":
    main()
