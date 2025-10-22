"""Long-context evaluation harness (RULER/LongBench-style)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from quietreasoning.config import QuietReasoningConfig
from quietreasoning.training.inference import AnswerOnlyDecoder
from quietreasoning.utils.tokenizer import SentencePieceTokenizer


@dataclass
class LongContextMetrics:
    exact_match: float
    f1: float
    latency_ms: float
    total_examples: int


class LongContextEvaluator:
    def __init__(self, cfg: QuietReasoningConfig, tokenizer: SentencePieceTokenizer) -> None:
        self.cfg = cfg
        self.decoder = AnswerOnlyDecoder(cfg, rationale_token_ids=[], penalty=4.0)
        self.tokenizer = tokenizer

    @staticmethod
    def _f1_score(pred: str, gold: str) -> float:
        pred_tokens = pred.lower().split()
        gold_tokens = gold.lower().split()
        common = sum((pred_tokens.count(w) and gold_tokens.count(w)) for w in set(pred_tokens))
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, params: dict, dataset: Iterable[dict[str, str]]) -> LongContextMetrics:
        em = 0
        f1 = 0.0
        total_latency = 0.0
        total = 0

        for sample in dataset:
            document = sample["document"]
            question = sample["question"]
            answer = sample["answer"]

            prompt = document + "\n\nQuestion: " + question + "\nAnswer:"
            prompt_tokens = jnp.array([self.tokenizer.encode(prompt)], dtype=jnp.int32)
            start = time.time()
            generations, _ = self.decoder.generate(params, prompt_tokens)
            total_latency += (time.time() - start) * 1000.0
            pred = self.tokenizer.decode(np.asarray(generations[0]))
            if answer.lower() in pred.lower():
                em += 1
            f1 += self._f1_score(pred, answer)
            total += 1

        if total == 0:
            return LongContextMetrics(exact_match=0.0, f1=0.0, latency_ms=0.0, total_examples=0)
        return LongContextMetrics(
            exact_match=em / total,
            f1=f1 / total,
            latency_ms=total_latency / total,
            total_examples=total,
        )

