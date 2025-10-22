"""Episodic memory store backed by FAISS/ScaNN with TTL and deduplication."""

from __future__ import annotations

import dataclasses
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    import scann  # type: ignore
except ImportError:  # pragma: no cover
    scann = None


@dataclasses.dataclass
class EpisodicEntry:
    embedding: np.ndarray
    metadata: dict
    timestamp: float


class EpisodicMemory:
    """Manages episodic events with approximate nearest neighbour index."""

    def __init__(
        self,
        dim: int,
        backend: str = "faiss_ivfpq",
        ttl_days: int = 30,
        dedupe_threshold: float = 0.92,
        cooldown_tokens: int = 64,
        params: Optional[dict] = None,
    ) -> None:
        self.dim = dim
        self.backend = backend
        self.ttl_seconds = ttl_days * 24 * 3600
        self.dedupe_threshold = dedupe_threshold
        self.cooldown_tokens = cooldown_tokens
        self.params = params or {"nlist": 4096, "m": 32, "nprobe": 16}
        self.entries: List[EpisodicEntry] = []
        self._index = None
        self._last_write_time = -np.inf
        self._token_clock = 0

    def _build_faiss_index(self) -> None:
        if faiss is None:
            raise RuntimeError("FAISS backend requested but faiss is not installed.")
        quantizer = faiss.IndexFlatIP(self.dim)
        nlist = self.params.get("nlist", 4096)
        m = self.params.get("m", 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8)
        embeddings = np.stack([e.embedding for e in self.entries], axis=0).astype(np.float32)
        if embeddings.size > 0:
            index.train(embeddings)
            index.add(embeddings)
        index.nprobe = self.params.get("nprobe", 16)
        self._index = index

    def _build_scann_index(self) -> None:
        if scann is None:
            raise RuntimeError("ScaNN backend requested but scann is not installed.")
        embeddings = np.stack([e.embedding for e in self.entries], axis=0).astype(np.float32)
        if embeddings.size == 0:
            self._index = None
            return
        searcher = (
            scann.scann_ops_pybind.builder(embeddings, 10, "dot_product")
            .tree(num_leaves=self.params.get("nlist", 4096), num_leaves_to_search=self.params.get("nprobe", 16))
            .score_bruteforce()
            .build()
        )
        self._index = searcher

    def _ensure_index(self) -> None:
        if self.backend.startswith("faiss") and self._index is None:
            self._build_faiss_index()
        elif self.backend.startswith("scann") and self._index is None:
            self._build_scann_index()

    def prune_expired(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        keep = []
        for entry in self.entries:
            if now - entry.timestamp <= self.ttl_seconds:
                keep.append(entry)
        self.entries = keep
        self._index = None

    def _is_duplicate(self, embedding: np.ndarray) -> bool:
        if not self.entries:
            return False
        existing = np.stack([e.embedding for e in self.entries], axis=0)
        norms = np.linalg.norm(existing, axis=-1) * np.linalg.norm(embedding)
        sims = existing @ embedding / np.clip(norms, a_min=1e-6, a_max=None)
        return bool(np.max(sims) >= self.dedupe_threshold)

    def tick_tokens(self, tokens: int) -> None:
        self._token_clock += tokens

    def should_write(self, surprise: float, threshold: float, now: Optional[float] = None) -> bool:
        now = now or time.time()
        cooldown_ok = self._token_clock >= self.cooldown_tokens
        return surprise > threshold and cooldown_ok and (now - self._last_write_time) > 5.0

    def add_event(
        self,
        embedding: np.ndarray,
        metadata: Optional[dict],
        surprise: float,
        threshold: float,
        now: Optional[float] = None,
    ) -> bool:
        now = now or time.time()
        if not self.should_write(surprise, threshold, now):
            return False
        if self._is_duplicate(embedding):
            return False
        self.entries.append(EpisodicEntry(embedding=embedding.astype(np.float32), metadata=metadata or {}, timestamp=now))
        self._last_write_time = now
        self._token_clock = 0
        self._index = None
        return True

    def query(self, query: np.ndarray, topk: int = 8) -> Tuple[np.ndarray, List[dict]]:
        if not self.entries:
            return np.zeros((0, self.dim), dtype=np.float32), []
        self._ensure_index()
        query = query.astype(np.float32)
        if self.backend.startswith("faiss") and self._index is not None:
            distances, indices = self._index.search(query[None, :], topk)
            idx = indices[0]
        elif self.backend.startswith("scann") and self._index is not None:
            idx, distances = self._index.search(query, final_num_neighbors=topk)
        else:
            embeddings = np.stack([e.embedding for e in self.entries], axis=0)
            sims = embeddings @ query / (np.linalg.norm(embeddings, axis=-1) * np.linalg.norm(query) + 1e-6)
            idx = np.argsort(-sims)[:topk]

        vectors = np.stack([self.entries[int(i)].embedding for i in idx if i >= 0], axis=0)
        meta = [self.entries[int(i)].metadata for i in idx if i >= 0]
        return vectors, meta

