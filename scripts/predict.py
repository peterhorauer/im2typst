#!/usr/bin/env python
"""Run a saved checkpoint on one image and print its predicted Typst string.

Loads a checkpoint written by ``train_model.py --save`` (a
``VisionEncoderDecoderModel`` + our tokenizer) and generates a prediction for a
single PNG — handy for eyeballing a checkpoint on val/test images or any
screenshot, without re-running the training/eval loop::

    ./venv/bin/python scripts/predict.py --model runs/trocr-v1 --image data/val/images/000000.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.model import DEFAULT_TROCR, load_image_processor  # noqa: E402
from im2typst.tokenizer import CharTokenizer                     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="checkpoint dir saved by train_model.py --save")
    p.add_argument("--image", type=Path, required=True, help="PNG to predict on")
    p.add_argument("--trocr", default=DEFAULT_TROCR,
                   help="checkpoint the image processor comes from (must match training)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    tok = CharTokenizer.load(args.model / "tokenizer.json")
    image_processor = load_image_processor(args.trocr)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        gen = model.generate(pixel_values)
    print(tok.decode(gen[0].tolist()))


if __name__ == "__main__":
    main()
