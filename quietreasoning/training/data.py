"""Input pipelines for Quiet Reasoning training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from quietreasoning.config import QuietReasoningConfig
from quietreasoning.utils.tokenizer import SentencePieceTokenizer


@dataclass
class DatasetConfig:
    file_pattern: str
    batch_size: int
    sequence_length: int
    shuffle_buffer: int = 10_000
    cycle_length: int = 8
    deterministic: bool = False


def _load_files(file_pattern: str) -> tf.data.Dataset:
    files = tf.io.gfile.glob(file_pattern)
    if not files:
        raise ValueError(f"No files matched pattern: {file_pattern}")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files))
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return dataset


def _tokenize_example(example: tf.Tensor, tokenizer: SentencePieceTokenizer, seq_len: int) -> tf.Tensor:
    text = example.numpy().decode("utf-8")
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(tokens) < seq_len:
        tokens = tokens + [tokenizer.processor.pad_id()] * (seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]
    return tf.convert_to_tensor(tokens, dtype=tf.int32)


def build_pretrain_dataset(
    cfg: DatasetConfig,
    tokenizer_path: Path,
    *,
    shard_id: int = 0,
    num_shards: int = 1,
) -> Iterator[dict[str, np.ndarray]]:
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    dataset = _load_files(cfg.file_pattern)

    def parse_fn(serialized):
        example = tf.io.parse_single_example(
            serialized,
            features={"text": tf.io.FixedLenFeature([], tf.string)},
        )
        return example["text"]

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def tokenize_py(text: tf.Tensor) -> tf.Tensor:
        return tf.py_function(
            func=lambda x: _tokenize_example(x, tokenizer, cfg.sequence_length),
            inp=[text],
            Tout=tf.int32,
        )

    dataset = dataset.map(tokenize_py, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(cfg.shuffle_buffer)
    dataset = dataset.repeat()
    dataset = dataset.batch(cfg.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    it = iter(dataset)
    shard_step = 0
    while True:
        tokens = next(it).numpy()
        if shard_step % num_shards != shard_id:
            shard_step += 1
            continue
        labels = np.roll(tokens, shift=-1, axis=1)
        loss_mask = np.ones_like(tokens)
        yield {
            "input_ids": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
        }
        shard_step += 1


def build_popqa_dataset(path: Path) -> Iterator[dict[str, str]]:
    import json

    with tf.io.gfile.GFile(str(path), "r") as f:
        for line in f:
            record = json.loads(line)
            yield {"question": record["question"], "answer": record["answer"]}


def pad_to_batch(arrays: Iterable[np.ndarray], batch_size: int, pad_value: int) -> np.ndarray:
    arrays = list(arrays)
    if not arrays:
        raise ValueError("No arrays provided.")
    width = arrays[0].shape[-1]
    out = np.full((batch_size, width), pad_value, dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays[:batch_size]):
        out[i] = arr
    return out


def prepare_tokens(
    tokenizer: SentencePieceTokenizer,
    texts: Iterable[str],
    sequence_length: int,
) -> np.ndarray:
    encoded = [
        tokenizer.encode(text, add_bos=True, add_eos=True)[:sequence_length]
        for text in texts
    ]
    return pad_to_batch(encoded, len(encoded), tokenizer.processor.pad_id())

