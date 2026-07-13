"""Char-level tokenizer for Typst math strings (v1).

Turns a Typst label like ``x^(2) + 1`` into a list of integer token IDs the
decoder learns to emit, and back again. Char-level is a deliberate v1 choice:
Typst math is a small, closed alphabet, so the vocabulary is tiny, there is
**zero out-of-vocabulary risk** once the vocab is built from a representative
corpus, and encode/decode is trivially lossless and easy to debug. A byte-level
BPE (shorter sequences via merges like ``sqrt(`` → one token) is the planned v2.

Layout of the vocabulary: the four special tokens occupy fixed IDs 0-3, then the
corpus characters follow in sorted order::

    0 <pad>   padding for batching
    1 <bos>   beginning of sequence (decoder start)
    2 <eos>   end of sequence (stop signal)
    3 <unk>   any character not seen at build time
    4.. the sorted set of characters in the corpus

The tokenizer serializes to a small JSON file (``save``/``load``) so the exact
same vocab is reused at train and inference time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class CharTokenizer:
    """A reversible mapping between Typst strings and integer token IDs."""

    def __init__(self, vocab: list[str]) -> None:
        # ``vocab`` is the full ordered token list (specials first); index == ID.
        self.id_to_token: list[str] = list(vocab)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(vocab)}
        # Cache the special IDs for hot-path use.
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    # -- construction ---------------------------------------------------------

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Build a tokenizer from a corpus: collect every distinct character."""
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        # Specials at fixed low IDs, then characters sorted for determinism.
        vocab = SPECIALS + sorted(chars)
        return cls(vocab)

    # -- core API -------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode a string to token IDs, optionally wrapping in <bos>…<eos>."""
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in text]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a string, dropping specials by default."""
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        out = []
        for i in ids:
            if skip_special and i in specials:
                continue
            out.append(self.id_to_token[i])
        return "".join(out)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "char-level-v1",
            "special_tokens": SPECIALS,
            "vocab": self.id_to_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocab"])
