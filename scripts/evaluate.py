#!/usr/bin/env python
"""Run a saved checkpoint over a whole split and report exact-match.

Loads a checkpoint written by ``train_model.py --save`` and generates a
prediction for every example in a split (val/test/train), comparing against
the gold Typst label — the held-out counterpart to ``predict.py`` (which only
handles one image at a time)::

    python scripts/evaluate.py --model runs/milestone2 --data data --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.data import FormulaDataset                          # noqa: E402
from im2typst.metrics import cer                                  # noqa: E402
from im2typst.model import DEFAULT_TROCR, load_image_processor    # noqa: E402
from im2typst.tokenizer import CharTokenizer                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="checkpoint dir saved by train_model.py --save")
    p.add_argument("--data", type=Path, default=Path("data"), help="dataset directory")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--trocr", default=DEFAULT_TROCR,
                   help="checkpoint the image processor comes from (must match training)")
    p.add_argument("--device", default="cuda", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n", type=int, default=None, help="limit to first N examples")
    p.add_argument("--show-mismatches", type=int, default=10,
                   help="how many gold/pred mismatches to print")
    args = p.parse_args()

    device = torch.device(args.device)
    tok = CharTokenizer.load(args.model / "tokenizer.json")
    image_processor = load_image_processor(args.trocr)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    dataset = FormulaDataset(args.data / args.split, tok, image_processor, limit=args.n)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating {len(dataset)} examples from {args.split} on {device}.")

    exact = 0
    total = 0
    total_cer = 0.0
    mismatches = []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split, unit="batch"):
            pixel_values = batch["pixel_values"].to(device)
            gen = model.generate(pixel_values)
            for g in gen:
                pred = tok.decode(g.tolist())
                gold = dataset.rows[idx]["typst"]
                example_cer = cer(gold, pred)
                total_cer += example_cer
                if pred == gold:
                    exact += 1
                elif len(mismatches) < args.show_mismatches:
                    mismatches.append((gold, pred, example_cer))
                total += 1
                idx += 1

    print(f"\nExact match: {exact}/{total} ({100 * exact / total:.1f}%)")
    print(f"Mean CER:    {total_cer / total:.4f}  (0 = perfect, 1 = fully wrong length-for-length)")
    if mismatches:
        print(f"\nFirst {len(mismatches)} mismatches:")
        for gold, pred, example_cer in mismatches:
            print(f"  gold: {gold}")
            print(f"  pred: {pred}")
            print(f"  CER:  {example_cer:.4f}")
            print()


if __name__ == "__main__":
    main()
