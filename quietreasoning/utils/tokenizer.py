"""SentencePiece tokenizer utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: Path) -> None:
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        pieces = self.processor.encode(text, out_type=int)
        tokens = []
        if add_bos and self.processor.bos_id() >= 0:
            tokens.append(self.processor.bos_id())
        tokens.extend(pieces)
        if add_eos and self.processor.eos_id() >= 0:
            tokens.append(self.processor.eos_id())
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        # SentencePiece expects plain Python ints; JAX/NumPy scalars trigger a runtime error.
        normalized_tokens = [int(token) for token in tokens]
        try:
            return self.processor.decode_ids(normalized_tokens)
        except RuntimeError as exc:  # pragma: no cover - debugging aid for dtype mismatches
            preview = normalized_tokens[:12]
            raise RuntimeError(
                "SentencePiece decode failed for token preview "
                f"{preview} (len={len(normalized_tokens)})."
            ) from exc
