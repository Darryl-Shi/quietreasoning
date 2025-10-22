"""Utilities for building memory indices (PKM, kNN-LM, episodic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except (ImportError, OSError):  # pragma: no cover
    faiss = None

from quietreasoning.memory.episodic import EpisodicEntry, EpisodicMemory
from quietreasoning.memory.knn import KNNDatastore


def initialize_pkm_values(path: Path, slots: int, value_dim: int) -> None:
    values = np.random.randn(slots, value_dim).astype(np.float32) * 0.02
    np.save(path, values)
    print(f"Initialized PKM values at {path} with shape {values.shape}")


def build_knn_index(keys: Path, values: Path, output: Path, dim: int, factory: str) -> None:
    if faiss is None:
        raise RuntimeError("faiss must be installed to build kNN indices.")
    key_matrix = np.load(keys).astype(np.float32)
    value_matrix = np.load(values).astype(np.int32)
    if key_matrix.shape[0] != value_matrix.shape[0]:
        raise ValueError("Key and value matrix row counts must match.")
    datastore = KNNDatastore(dim=dim, size=key_matrix.shape[0], index_factory=factory)
    datastore.build(key_matrix, value_matrix)
    faiss.write_index(datastore.index, str(output))  # type: ignore
    np.save(output.with_suffix(".meta.npy"), value_matrix)
    print(f"kNN index written to {output}")


def export_episodic(entries_path: Path, backend: str, ttl_days: int) -> None:
    with entries_path.open("r") as f:
        entries = json.load(f)
    embeddings = [np.array(e["embedding"], dtype=np.float32) for e in entries]
    meta = [e.get("metadata", {}) for e in entries]
    memory = EpisodicMemory(
        dim=len(embeddings[0]),
        backend=backend,
        ttl_days=ttl_days,
    )
    for emb, m in zip(embeddings, meta):
        memory.entries.append(EpisodicEntry(embedding=emb, metadata=m, timestamp=0.0))
    memory.prune_expired(0.0)
    print(f"Episodic memory built with {len(memory.entries)} entries using backend={backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quiet Reasoning index builder")
    sub = parser.add_subparsers(dest="command", required=True)

    pkm_cmd = sub.add_parser("pkm", help="initialize PKM values")
    pkm_cmd.add_argument("--path", type=Path, required=True)
    pkm_cmd.add_argument("--slots", type=int, default=4_000_000)
    pkm_cmd.add_argument("--value-dim", type=int, default=256)

    knn_cmd = sub.add_parser("knn", help="build kNN-LM datastore")
    knn_cmd.add_argument("--keys", type=Path, required=True)
    knn_cmd.add_argument("--values", type=Path, required=True)
    knn_cmd.add_argument("--output", type=Path, required=True)
    knn_cmd.add_argument("--dim", type=int, required=True)
    knn_cmd.add_argument("--factory", type=str, default="IVF4096,PQ32")

    epi_cmd = sub.add_parser("episodic", help="prepare episodic memory entries")
    epi_cmd.add_argument("--input", type=Path, required=True)
    epi_cmd.add_argument("--backend", type=str, default="faiss_ivfpq")
    epi_cmd.add_argument("--ttl-days", type=int, default=30)

    verify_cmd = sub.add_parser("verify-knn", help="verify a kNN index with probe queries")
    verify_cmd.add_argument("--index", type=Path, required=True)
    verify_cmd.add_argument("--queries", type=Path, required=True)
    verify_cmd.add_argument("--k", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pkm":
        initialize_pkm_values(args.path, args.slots, args.value_dim)
    elif args.command == "knn":
        build_knn_index(args.keys, args.values, args.output, args.dim, args.factory)
    elif args.command == "episodic":
        export_episodic(args.input, args.backend, args.ttl_days)
    elif args.command == "verify-knn":
        if faiss is None:
            raise RuntimeError("faiss must be installed to verify kNN indices.")
        index = faiss.read_index(str(args.index))
        queries = np.load(args.queries).astype(np.float32)
        distances, neighbors = index.search(queries, args.k)
        print(
            f"Verified index {args.index} with {queries.shape[0]} queries; "
            f"mean distance={float(distances.mean()):.4f}"
        )
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
