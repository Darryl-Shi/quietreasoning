"""End-to-end Quiet Reasoning training runner."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as orbax_checkpoint

from quietreasoning import QuietReasoningConfig, build_train_step, create_train_state
from quietreasoning.eval.long_context import LongContextEvaluator
from quietreasoning.eval.popqa import PopQAEvaluator
from quietreasoning.training.data import DatasetConfig, build_popqa_dataset, build_pretrain_dataset
from quietreasoning.training.launch import TPULaunchConfig, initialize_distributed
from quietreasoning.training.stages import StageScheduler
from quietreasoning.utils.tokenizer import SentencePieceTokenizer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("quietreasoning.run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quiet Reasoning training runner")
    parser.add_argument("--pretrain-pattern", type=str, required=True, help="TFRecord glob for pretraining data")
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece tokenizer path")
    parser.add_argument("--popqa", type=Path, required=True, help="PopQA dataset (jsonl)")
    parser.add_argument("--long-context", type=Path, required=False, help="Long context evaluation set (jsonl)")
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
    return parser.parse_args()


def maybe_initialize_tpu(args: argparse.Namespace) -> None:
    if args.tpu_topology:
        address, port = (args.coordinator.split(":") + ["12345"])[:2] if args.coordinator else ("", "12345")
        cfg = TPULaunchConfig(
            topology=args.tpu_topology,
            coordinator_address=address or None,
            coordinator_port=int(port),
            process_id=jax.process_index(),
            process_count=jax.process_count(),
        )
        initialize_distributed(cfg)


def prepare_datasets(
    cfg: QuietReasoningConfig,
    args: argparse.Namespace,
) -> Iterator[Dict[str, np.ndarray]]:
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
    maybe_initialize_tpu(args)

    cfg = QuietReasoningConfig()
    cfg.model.tokenizer_path = str(args.tokenizer)

    rng = jax.random.PRNGKey(args.seed + jax.process_index())
    state = create_train_state(rng, cfg)
    scheduler = StageScheduler(cfg.training)
    dataset_iter = prepare_datasets(cfg, args)

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    popqa_data = sample_iterator(build_popqa_dataset(args.popqa), args.eval_samples)
    popqa_eval = PopQAEvaluator(cfg, tokenizer)

    long_eval = None
    long_data = []
    if args.long_context and args.long_context.exists():
        with args.long_context.open("r") as f:
            for line in itertools.islice(f, args.long_samples):
                long_data.append(json.loads(line))
        long_eval = LongContextEvaluator(cfg, tokenizer)

    step_fn = build_train_step(cfg)
    tokens_per_step = args.batch_size * cfg.model.context
    checkpointer = orbax_checkpoint.PyTreeCheckpointer()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        stage = scheduler.stage_at(float(state.tokens_seen))
        batch = next(dataset_iter)
        feature_gates = {
            name: jnp.array(1.0 if enabled else 0.0, dtype=jnp.float32)
            for name, enabled in stage.features.items()
        }
        state, logs = step_fn(state, batch, float(tokens_per_step), feature_gates)

        if step % args.log_every == 0 and jax.process_index() == 0:
            LOGGER.info(
                "step=%d stage=%s loss=%.4f workspace_steps=%.3f",
                step,
                stage.stage.name,
                float(logs["loss"]),
                float(logs["workspace_steps"]),
            )

        if step % args.eval_every == 0 and step > 0 and jax.process_index() == 0:
            LOGGER.info("Running PopQA evaluation...")
            pq_metrics = popqa_eval.evaluate(state.params, popqa_data)
            LOGGER.info(
                "PopQA: accuracy=%.3f recall@5=%.3f total=%d",
                pq_metrics.accuracy,
                pq_metrics.recall_at_5,
                pq_metrics.total_examples,
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
            save_checkpoint(checkpointer, output_dir, state, step)

    if jax.process_index() == 0:
        save_checkpoint(checkpointer, output_dir, state, args.steps)
        LOGGER.info("Training complete.")


if __name__ == "__main__":
    main()

