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
        return self.processor.decode(list(tokens))

