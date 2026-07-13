"""PyTorch dataset that pairs a rendered formula image with its token IDs.

This is the glue between the generated dataset on disk and the model: for each
example it loads the PNG, runs it through the encoder's **image processor**
(resize/normalize → pixel tensor), and encodes the Typst label with **our**
:class:`~im2typst.tokenizer.CharTokenizer` into decoder target IDs.

The vision side (image processor) and the text side (tokenizer) are independent
— exactly why we can pair a pretrained TrOCR image processor with our own
char-level tokenizer. Each item is ``{"pixel_values", "labels"}``, the two
tensors a ``VisionEncoderDecoderModel`` needs for a training step.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer

# Loss ignores this target index, so padding positions don't contribute a gradient.
IGNORE_INDEX = -100


class FormulaDataset(Dataset):
    """(image, Typst-label) pairs from one split, ready for the model."""

    def __init__(self, split_dir, tokenizer: CharTokenizer, image_processor,
                 max_length: int = 128, limit: int | None = None) -> None:
        self.split_dir = Path(split_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        with (self.split_dir / "labels.jsonl").open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_label(self, text: str) -> torch.Tensor:
        """Encode → pad/truncate to max_length → mask padding with IGNORE_INDEX."""
        ids = self.tokenizer.encode(text)[: self.max_length]
        pad = self.max_length - len(ids)
        ids = ids + [self.tokenizer.pad_id] * pad
        # Replace pad IDs with IGNORE_INDEX so the loss skips them.
        labels = [t if t != self.tokenizer.pad_id else IGNORE_INDEX for t in ids]
        return torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = Image.open(self.split_dir / row["image"]).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)          # [3, H, W]
        return {
            "pixel_values": pixel_values,
            "labels": self._encode_label(row["typst"]),
        }
