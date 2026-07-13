#!/usr/bin/env python
"""Build an HTML contact sheet to eyeball (image, Typst-label) pairs.

Reads a dataset's ``labels.jsonl`` and emits a single self-contained HTML page
showing every rendered formula next to its exact label string, so you can skim
the whole dataset and catch any image↔label mismatch, clipping, or unreadable
formula before training. The page is written *inside* the data directory so the
relative ``images/...`` paths resolve directly — just open it in a browser::

    python scripts/make_contact_sheet.py --data data
    # then open data/contact_sheet.html

Use ``--n`` to review a random sample instead of the whole set.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

# Make the repo root importable (kept consistent with the other scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>im2typst contact sheet — {count} pairs</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f5f5f5; }}
  h1 {{ font-size: 18px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat({cols}, 1fr);
    gap: 16px;
  }}
  .card {{
    background: #fff; border: 1px solid #ddd; border-radius: 8px;
    padding: 10px; display: flex; flex-direction: column; gap: 8px;
  }}
  .imgwrap {{
    background: #fff; min-height: 60px;
    display: flex; align-items: center; justify-content: center;
    border-bottom: 1px solid #eee; padding-bottom: 8px;
  }}
  .imgwrap img {{ max-width: 100%; height: auto; }}
  .label {{
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 12px; color: #222; word-break: break-word; white-space: pre-wrap;
  }}
  .meta {{ font-size: 11px; color: #999; }}
</style>
</head>
<body>
<h1>im2typst contact sheet — {count} pairs{note}</h1>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

_CARD = """  <div class="card">
    <div class="imgwrap"><img src="{src}" alt="{alt}" loading="lazy"></div>
    <div class="label">{label}</div>
    <div class="meta">{meta}</div>
  </div>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"),
                   help="dataset directory (holds labels.jsonl + images/)")
    p.add_argument("--out", type=Path, default=None,
                   help="output HTML path (default: <data>/contact_sheet.html)")
    p.add_argument("--n", type=int, default=None,
                   help="review a random sample of N pairs (default: all)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--cols", type=int, default=2,
                   help="number of formulas side by side (default: 2)")
    args = p.parse_args()

    labels_path = args.data / "labels.jsonl"
    if not labels_path.exists():
        sys.exit(f"error: {labels_path} not found — generate a dataset first.")

    with labels_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        sys.exit(f"error: {labels_path} is empty.")

    note = ""
    total = len(rows)
    if args.n is not None and args.n < total:
        rows = random.Random(args.seed).sample(rows, args.n)
        note = f" (random sample of {args.n} / {total}, seed {args.seed})"

    out_path = args.out or (args.data / "contact_sheet.html")

    cards = []
    for i, row in enumerate(rows):
        cards.append(_CARD.format(
            src=html.escape(row["image"], quote=True),
            alt=html.escape(row["typst"], quote=True),
            label=html.escape(row["typst"]),
            meta=html.escape(f"#{i}  ·  {row['image']}"),
        ))

    page = _PAGE.format(
        count=len(rows), note=note, cols=args.cols, cards="\n".join(cards),
    )
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path} — {len(rows)} pairs. Open it in a browser.")


if __name__ == "__main__":
    main()
