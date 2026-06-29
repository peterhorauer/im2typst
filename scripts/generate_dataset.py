#!/usr/bin/env python
"""Generate a dataset of (image, Typst-label) pairs.

Samples N valid Typst math expressions, renders each to a PNG with the Typst
CLI, and writes a JSONL manifest pairing every image with its exact label::

    data/
    ├── images/000000.png …
    └── labels.jsonl        # {"image": "images/000000.png", "typst": "..."}

Because the labels are generated (not parsed), every pair is correct by
construction. Example::

    ./venv/bin/python scripts/generate_dataset.py --n 1000 --out data --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.generator import Grammar          # noqa: E402
from im2typst.render import (                    # noqa: E402
    RenderOptions, TypstRenderError, render_to_png,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="number of pairs to generate")
    p.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    p.add_argument("--max-depth", type=int, default=4, help="recursion depth bound")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    p.add_argument("--ppi", type=int, default=200, help="render resolution")
    p.add_argument("--unique", action="store_true",
                   help="skip duplicate label strings")
    args = p.parse_args()

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out / "labels.jsonl"

    g = Grammar(max_depth=args.max_depth, seed=args.seed)
    opts = RenderOptions(ppi=args.ppi)

    seen: set[str] = set()
    written = 0
    failed = 0
    # Bound attempts so --unique on a small grammar can't loop forever.
    max_attempts = args.n * 20

    with labels_path.open("w") as manifest:
        attempts = 0
        while written < args.n and attempts < max_attempts:
            attempts += 1
            label = g.sample_typst()
            if args.unique and label in seen:
                continue
            seen.add(label)

            rel = f"images/{written:06d}.png"
            try:
                render_to_png(label, args.out / rel, opts)
            except TypstRenderError as exc:
                failed += 1
                print(f"[skip] {exc}", file=sys.stderr)
                continue

            manifest.write(json.dumps({"image": rel, "typst": label}) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  … {written}/{args.n}", file=sys.stderr)

    print(f"Wrote {written} pairs to {args.out} "
          f"({failed} render failures, {attempts} samples drawn).",
          file=sys.stderr)
    if written < args.n:
        print(f"WARNING: only produced {written}/{args.n} "
              f"(hit attempt cap {max_attempts}).", file=sys.stderr)


if __name__ == "__main__":
    main()
