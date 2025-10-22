"""PopQA evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import jax
import jax.numpy as jnp
import numpy as np

from quietreasoning.config import QuietReasoningConfig
from quietreasoning.training.inference import AnswerOnlyDecoder
from quietreasoning.utils.tokenizer import SentencePieceTokenizer


@dataclass
class PopQAMetrics:
    accuracy: float
    recall_at_5: float
    total_examples: int


class PopQAEvaluator:
    def __init__(self, cfg: QuietReasoningConfig, tokenizer: SentencePieceTokenizer) -> None:
        self.cfg = cfg
        self.decoder = AnswerOnlyDecoder(cfg, rationale_token_ids=[], penalty=6.0)
        self.tokenizer = tokenizer

    def evaluate(self, params: dict, dataset: Iterable[dict[str, str]]) -> PopQAMetrics:
        correct = 0
        correct_top5 = 0
        total = 0

        for example in dataset:
            question = example["question"]
            answer = example["answer"]
            prompt_tokens = jnp.array([self.tokenizer.encode(question)], dtype=jnp.int32)
            generations, logits = self.decoder.generate(params, prompt_tokens)
            generated_text = self.tokenizer.decode(np.asarray(generations[0]))
            if answer.lower() in generated_text.lower():
                correct += 1

            probs = jax.nn.softmax(logits, axis=-1)
            top5 = jnp.argsort(probs, axis=-1)[:, -5:]
            if any(
                self.tokenizer.decode([int(token)]).lower() in answer.lower()
                for token in np.asarray(top5[0])
            ):
                correct_top5 += 1
            total += 1

        if total == 0:
            return PopQAMetrics(accuracy=0.0, recall_at_5=0.0, total_examples=0)
        return PopQAMetrics(
            accuracy=correct / total,
            recall_at_5=correct_top5 / total,
            total_examples=total,
        )

