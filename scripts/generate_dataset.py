#!/usr/bin/env python
"""Generate a split (image, Typst-label) dataset.

Samples N valid Typst math expressions, partitions them into disjoint
train/val/test splits, renders each to a PNG with the Typst CLI, and writes a
JSONL manifest per split::

    data/
    ├── train/
    │   ├── images/000000.png …
    │   └── labels.jsonl        # {"image": "images/000000.png", "typst": "..."}
    ├── val/
    │   ├── images/…
    │   └── labels.jsonl
    └── test/
        ├── images/…
        └── labels.jsonl

Labels are deduplicated *before* splitting (always — a duplicate draw is
re-sampled), so no formula string appears in more than one split: evaluation
never sees a training label. Because the labels are generated (not parsed),
every pair is correct by construction. Example::

    ./venv/bin/python scripts/generate_dataset.py --n 10000 --out data --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.generator import Grammar          # noqa: E402
from im2typst.render import (                    # noqa: E402
    RenderOptions, TypstRenderError, render_to_png,
)


def collect_labels(g: Grammar, n: int, max_attempts: int) -> list[str]:
    """Draw ``n`` distinct labels; re-sample whenever a duplicate comes up."""
    labels: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(labels) < n and attempts < max_attempts:
        attempts += 1
        label = g.sample_typst()
        if label in seen:      # already drawn — retry with a fresh sample
            continue
        seen.add(label)
        labels.append(label)
    if len(labels) < n:
        print(f"WARNING: only drew {len(labels)}/{n} labels "
              f"(hit attempt cap {max_attempts}; grammar may be exhausted at "
              f"this depth).", file=sys.stderr)
    return labels


def split_labels(labels: list[str], fracs: tuple[float, float, float],
                 seed: int) -> dict[str, list[str]]:
    """Shuffle and partition ``labels`` into train/val/test by ``fracs``."""
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_train = int(total * fracs[0])
    n_val = int(total * fracs[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],   # remainder → avoids off-by-one loss
    }


def render_split(name: str, labels: list[str], out_dir: Path,
                 opts: RenderOptions) -> tuple[int, int]:
    """Render one split into ``out_dir/name/``. Returns (written, failed)."""
    split_dir = out_dir / name
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    labels_path = split_dir / "labels.jsonl"

    written = 0
    failed = 0
    with labels_path.open("w") as manifest:
        for label in tqdm(labels, desc=name, unit="img"):
            rel = f"images/{written:06d}.png"
            try:
                render_to_png(label, split_dir / rel, opts)
            except TypstRenderError as exc:
                failed += 1
                print(f"[skip] {exc}", file=sys.stderr)
                continue
            manifest.write(json.dumps({"image": rel, "typst": label}) + "\n")
            written += 1
    print(f"  {name}: {written} pairs ({failed} render failures) → {split_dir}",
          file=sys.stderr)
    return written, failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="total number of pairs to generate")
    p.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    p.add_argument("--max-depth", type=int, default=4, help="recursion depth bound")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    p.add_argument("--ppi", type=int, default=200, help="render resolution")
    p.add_argument("--train-frac", type=float, default=0.8, help="train split fraction")
    p.add_argument("--val-frac", type=float, default=0.1, help="val split fraction")
    p.add_argument("--test-frac", type=float, default=0.1, help="test split fraction")
    args = p.parse_args()

    fracs = (args.train_frac, args.val_frac, args.test_frac)
    if abs(sum(fracs) - 1.0) > 1e-6:
        p.error(f"--train/--val/--test fracs must sum to 1.0 (got {sum(fracs)}).")

    g = Grammar(max_depth=args.max_depth, seed=args.seed)
    opts = RenderOptions(ppi=args.ppi)

    # 1) draw distinct labels (fast, no rendering), 2) split, 3) render each split.
    labels = collect_labels(g, args.n, max_attempts=args.n * 20)
    splits = split_labels(labels, fracs, seed=args.seed)

    print(f"Split {len(labels)} labels → "
          f"train {len(splits['train'])} / val {len(splits['val'])} / "
          f"test {len(splits['test'])}. Rendering…", file=sys.stderr)

    total_written = 0
    for name in ("train", "val", "test"):
        written, _ = render_split(name, splits[name], args.out, opts)
        total_written += written

    print(f"Done. Wrote {total_written} pairs across 3 splits under {args.out}.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
