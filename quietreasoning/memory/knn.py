"""kNN-LM datastore integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class KNNLookupResult:
    scores: np.ndarray
    values: np.ndarray
    indices: np.ndarray


class KNNDatastore:
    """Wrapper around FAISS IVF-PQ datastore used for kNN-LM interpolation."""

    def __init__(self, dim: int, size: int, index_factory: str = "IVF4096,PQ32") -> None:
        self.dim = dim
        self.size = size
        self.index_factory = index_factory
        self.index = None
        self.values: Optional[np.ndarray] = None

    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        if faiss is None:
            raise RuntimeError("faiss must be installed to build the kNN datastore.")
        self.values = values.astype(np.int32)
        self.index = faiss.index_factory(self.dim, self.index_factory, faiss.METRIC_INNER_PRODUCT)
        if not self.index.is_trained:
            self.index.train(keys)
        self.index.add(keys)

    def search(self, query: np.ndarray, k: int = 32) -> KNNLookupResult:
        if self.index is None or self.values is None:
            raise RuntimeError("Datastore has not been built.")
        distances, indices = self.index.search(query[None, :], k)
        return KNNLookupResult(
            scores=distances[0],
            values=self.values[indices[0]],
            indices=indices[0],
        )

    def interpolate(self, logits: np.ndarray, lookup: KNNLookupResult, lam: float) -> np.ndarray:
        vocab = logits.shape[-1]
        knn_scores = np.zeros_like(logits)
        np.add.at(knn_scores, lookup.values, lookup.scores)
        knn_prob = np.exp(knn_scores - knn_scores.max())
        knn_prob = knn_prob / np.sum(knn_prob)
        base_prob = np.exp(logits - logits.max())
        base_prob = base_prob / np.sum(base_prob)
        blended = (1 - lam) * base_prob + lam * knn_prob
        return np.log(blended + 1e-8)

