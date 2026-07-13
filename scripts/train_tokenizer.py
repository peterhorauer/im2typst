#!/usr/bin/env python
"""Build the char-level tokenizer from the training corpus.

Reads the Typst labels from the train split, builds a :class:`CharTokenizer`,
saves it to JSON, and reports stats useful for configuring the model:

* vocabulary size (the decoder's output dimension),
* the longest encoded sequence (informs the decoder's ``max_length``),
* a round-trip check that encode→decode is lossless,
* an out-of-vocabulary check against val/test (any char the model would see at
  eval time but the tokenizer never learned).

The tokenizer is built from **train only** — building it on val/test would leak
information about the eval sets. Example::

    ./venv/bin/python scripts/train_tokenizer.py --data data --out data/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.tokenizer import CharTokenizer  # noqa: E402


def read_labels(split_dir: Path) -> list[str]:
    """Return the list of Typst label strings from a split's labels.jsonl."""
    path = split_dir / "labels.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line)["typst"] for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"),
                   help="dataset directory (holds train/ val/ test/)")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: <data>/tokenizer.json)")
    args = p.parse_args()

    train = read_labels(args.data / "train")
    if not train:
        sys.exit(f"error: no labels in {args.data / 'train'} — generate a dataset first.")

    tok = CharTokenizer.build(train)
    out_path = args.out or (args.data / "tokenizer.json")
    tok.save(out_path)

    # --- round-trip check: encode→decode must reproduce every train label ----
    mismatches = sum(1 for s in train if tok.decode(tok.encode(s)) != s)

    # --- longest encoded sequence (incl. <bos>/<eos>) → decoder max_length ----
    max_len = max(len(tok.encode(s)) for s in train)

    # --- out-of-vocabulary check on val/test ---------------------------------
    eval_texts = read_labels(args.data / "val") + read_labels(args.data / "test")
    oov = {ch for s in eval_texts for ch in s if ch not in tok.token_to_id}

    print(f"Built char-level tokenizer from {len(train)} train labels → {out_path}")
    print(f"  vocab size      : {tok.vocab_size} "
          f"({len(tok.id_to_token) - 4} chars + 4 specials)")
    print(f"  max seq length  : {max_len} tokens (use as decoder max_length)")
    print(f"  round-trip loss : {mismatches} mismatches "
          f"({'lossless ✓' if mismatches == 0 else 'PROBLEM ✗'})")
    if eval_texts:
        if oov:
            print(f"  OOV in val/test : {sorted(oov)!r}  ← WARNING: enlarge the "
                  f"train corpus so the tokenizer covers these characters.")
        else:
            print(f"  OOV in val/test : none ✓ ({len(eval_texts)} eval labels checked)")

    # Show the learned character set for a quick eyeball.
    chars = "".join(tok.id_to_token[4:])
    print(f"  characters      : {chars!r}")

    # Demo one round trip so the mapping is legible.
    sample = train[0]
    ids = tok.encode(sample)
    print(f"\n  example label   : {sample!r}")
    print(f"  encoded ids     : {ids}")
    print(f"  decoded back    : {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
