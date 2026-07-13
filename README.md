# im2typst
A formula screenshot to typst math converter

# goal
The goal is to create a image 2 typst converter which is able to take any formula found online and easily convert it into a typst formula.

# TODO
Add all kinds of mathematical and informatical, as well as physical characters and calculations to dataset generation
- First do so with typst characters as input only.
- Afterwards expand to LaTeX and other characters if needed.

Change the Character Tokenizer to be a BPE tokenizer to support "sqrt" as a token for example.


# Python Environment setup

First create the python environment
```bash
gh repo clone peterhorauer/im2typst
cd im2typst

python -m venv .
```

Under Linux source the bin/activate with following command
```bash 
source ./bin/activate
```

To confirm run `which python`

Upgrade the pip installation
```bash
python -m pip install --upgrade pip
```

And install all dependencies
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install requirements.txt
```

# Generating training data

We don't use a public dataset — we synthesize our own `(image, typst)` pairs. A
recursive grammar emits random *valid* Typst math, and the Typst CLI renders
each expression to a PNG. Because we generate the strings ourselves, every label
is correct by construction.

**Prerequisite:** the Typst CLI must be installed and on your `PATH` (or set the
`TYPST_BIN` env var). Check with `typst --version` (developed against 0.15.0).

## Preview formulas (no rendering)

Eyeball what the grammar produces before committing to a full render run:

```bash
python scripts/demo_generate.py --n 20 --max-depth 4 --seed 0
```

| flag | default | meaning |
|------|---------|---------|
| `--n` | 15 | how many formulas to print |
| `--max-depth` | 4 | recursion depth bound (higher = more nested) |
| `--seed` | none | RNG seed for reproducible output |

## Render a dataset

Generate the actual `(image, label)` pairs, partitioned into disjointREADME.md
train/val/test splits:

```bash
python scripts/generate_dataset.py --n 10000 --out data --seed 0
```

This writes one folder per split, each with its own images and manifest:

```
data/
├── train/
│   ├── images/000000.png, 000001.png, …
│   └── labels.jsonl   # one {"image": "images/000000.png", "typst": "..."} per line
├── val/
│   ├── images/…
│   └── labels.jsonl
└── test/
    ├── images/…
    └── labels.jsonl
```

| flag | default | meaning |
|------|---------|---------|
| `--n` | 100 | total number of pairs to generate |
| `--out` | `data` | output directory |
| `--max-depth` | 4 | recursion depth bound |
| `--seed` | 0 | RNG seed for reproducibility |
| `--ppi` | 200 | render resolution (pixels per inch) |
| `--train-frac` | 0.8 | train split fraction |
| `--val-frac` | 0.1 | val split fraction |
| `--test-frac` | 0.1 | test split fraction |

Labels are always deduplicated **before** splitting (a duplicate draw is
re-sampled), so no formula string appears in more than one split — evaluation
never sees a training label. The three
fractions must sum to 1.0; for larger datasets `90/5/5`
(`--train-frac 0.9 --val-frac 0.05 --test-frac 0.05`) gives more to training
while still leaving thousands of eval examples.

The same `--seed` reproduces the same dataset. The `data/` directory is
gitignored — regenerate it rather than committing it.

## Sanity-check the dataset

Before training, eyeball that the rendered images actually match their labels.
This builds a self-contained HTML page showing each formula next to its exact
Typst string:

Point `--data` at a single split directory (it holds the `labels.jsonl` +
`images/` the tool needs):

```bash
python scripts/make_contact_sheet.py --data data/train
# then open the page in a browser:
xdg-open data/train/contact_sheet.html
```

Review a random subset instead of the whole set, or change how many formulas
sit side by side:

```bash
python scripts/make_contact_sheet.py --data data/train --n 60 --cols 2
```

| flag | default | meaning |
|------|---------|---------|
| `--data` | `data` | split directory (holds `labels.jsonl` + `images/`) |
| `--out` | `<data>/contact_sheet.html` | output HTML path |
| `--n` | all | review a random sample of N pairs |
| `--cols` | 2 | how many formulas side by side |
| `--seed` | 0 | sampling seed |

The HTML is written inside the split folder so the relative `images/...` paths
resolve — don't move it out of that folder or the images won't load.

# Training the tokenizer

The model predicts a *sequence of tokens*, so we need a tokenizer that maps
Typst strings to integer IDs and back. For v1 this is **char-level**: Typst math
is a small, closed alphabet, so the vocabulary is tiny and there is zero
out-of-vocabulary risk. It is built from the **train split only** (building it
on val/test would leak information about the eval sets):

```bash
python scripts/train_tokenizer.py --data data
```

This writes `data/tokenizer.json` and prints stats you'll need for the model:

```
vocab size      : 54 (50 chars + 4 specials)
max seq length  : 98 tokens (use as decoder max_length)
round-trip loss : 0 mismatches (lossless ✓)
OOV in val/test : none ✓ (2000 eval labels checked)
```

| flag | default | meaning |
|------|---------|---------|
| `--data` | `data` | dataset directory (holds `train/ val/ test/`) |
| `--out` | `<data>/tokenizer.json` | output path for the tokenizer JSON |

Reuse it in code with:

```python
from im2typst.tokenizer import CharTokenizer
tok = CharTokenizer.load("data/tokenizer.json")
ids = tok.encode("x^(2) + 1")   # [<bos>, …, <eos>]
tok.decode(ids)                 # 'x^(2) + 1'
```

The four special tokens occupy fixed IDs: `<pad>`=0, `<bos>`=1, `<eos>`=2,
`<unk>`=3. If `train_tokenizer.py` reports any OOV characters in val/test,
enlarge the training set so the tokenizer covers them before training.

# Training the model

The model is a **Vision Encoder-Decoder**: a pretrained **TrOCR** image encoder
(reused as-is, along with its bundled image processor) paired with a decoder
whose vocabulary is resized to **our** char-level tokenizer. The vision side and
the text side are independent, so the tokenizer can be swapped later without
touching the encoder.

Three pieces make it up:

- `im2typst/data.py` — `FormulaDataset`: loads each PNG → pixel tensor (image
  processor) and encodes its Typst label → token IDs (tokenizer).
- `im2typst/model.py` — `build_model`: loads pretrained TrOCR and re-points its
  decoder at our vocab.
- `scripts/train_model.py` — the training entry point.

> **Training strategy, milestone tracking, and results** live in
> [TRAINING.md](TRAINING.md) — read it for the plan beyond the first run.

## Overfit smoke test

Before any large run, prove the pipeline learns by overfitting a small subset —
loss should collapse toward zero and the model should reproduce those exact
labels. If it can't overfit a handful of examples, something is miswired.

```bash
python scripts/train_model.py --data data --n 16 --epochs 80
```

The first run downloads the pretrained TrOCR checkpoint (a few hundred MB). It
prints the per-epoch loss, then generates a few train images and reports how
many decode back to the exact gold label (`Exact match: N/M`). See
[TRAINING.md](TRAINING.md) for the milestone-1 result and what comes next.

| flag | default | meaning |
|------|---------|---------|
| `--data` | `data` | dataset directory (needs `train/` + `tokenizer.json`) |
| `--n` | 200 | how many train examples to use |
| `--epochs` | 30 | passes over the subset |
| `--batch-size` | 4 | examples per step |
| `--lr` | 5e-5 | AdamW learning rate |
| `--max-length` | 128 | decoder max sequence length |
| `--trocr` | `microsoft/trocr-small-printed` | pretrained encoder/decoder checkpoint |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--eval-samples` | 5 | train examples to decode after training |
| `--save` | none | optional directory to save the model + tokenizer |

## Full training

The code is device-agnostic — pass `--device cuda` on a GPU machine and raise
`--n` to cover the whole train split (8000 for the default dataset):

```bash
python scripts/train_model.py --data data --device cuda --n 8000 --epochs 10 --save runs/trocr-v1
```

CPU is fine for the overfit smoke test, but a real run over the full 8k train
set wants a **cloud GPU (CUDA)** — an integrated GPU is not worth the ROCm
setup. Move the run to Colab or a rented GPU for the full fit.

