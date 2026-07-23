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

python -m venv ./venv
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
pip install -r requirements.txt
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
firefox data/train/contact_sheet.html
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

Each epoch also runs a **val check** (loss, exact-match, and CER on the `val`
split, disjoint from train) and prints it next to the train loss, gradient
norm, and learning rate:

```
  epoch  12  avg loss 0.1830  grad_norm 2.1140  lr 5.00e-05  val loss 2.9412  val exact 3/50 (6.0%)  val CER 0.4118
```

- **val loss vs train loss** is the earliest signal for a train/val gap — a
  continuous number, so it shows overfitting starting several epochs before
  val exact-match (all-or-nothing per sequence) would reveal anything.
- **val CER** (character error rate — `im2typst/metrics.py`, Levenshtein edit
  distance normalized by gold length) scores *how close* a wrong prediction
  was instead of pass/fail, which matters since near-misses (one dropped
  character, a confused Greek letter) look identical to total garbage under
  exact-match alone.
- **grad_norm** is the gradient's norm before any clipping (nothing is
  actually clipped here) — a free signal for training instability; a sudden
  spike usually means something's off.
- **lr** is just the optimizer's current learning rate — constant today since
  there's no scheduler yet, logged so a future warmup/decay schedule has
  somewhere to show up.

All of these (loss, grad norm, lr, and the full val breakdown) are per-epoch
histories written to `training_log.json` (see below), so a training run can
be plotted after the fact, not just read off stdout.

| flag | default | meaning |
|------|---------|---------|
| `--data` | `data` | dataset directory (needs `train/` + `tokenizer.json`) |
| `--n` | 200 | how many train examples to use |
| `--epochs` | 30 | passes over the subset |
| `--batch-size` | 4 | examples per step |
| `--lr` | 5e-5 | AdamW learning rate |
| `--max-length` | 128 | decoder max sequence length |
| `--trocr` | `microsoft/trocr-small-printed` | pretrained encoder/decoder checkpoint |
| `--device` | `cuda` | `cpu` or `cuda` |
| `--eval-samples` | 10 | train examples to decode after training |
| `--val-n` | 100 | val examples to check exact-match on each epoch (0 disables) |
| `--val-every` | 1 | run the val check every N epochs (0 disables) |
| `--save` | none | optional directory to save the model + tokenizer |
| `--save-every` | 5 | checkpoint to `--save` every N epochs (0 disables; final save always happens) |
| `--resume` | none | continue training from a checkpoint dir saved by `--save`, instead of starting over from the pretrained TrOCR checkpoint |

Every `--save` also writes **`training_log.json`** next to the model — a list
of every training invocation that built the checkpoint: the exact command line,
timestamp, hyperparameters, and per-epoch histories for loss, gradient norm,
learning rate, and val (loss/exact-match/CER), plus (on completion) the final
train exact-match. `--resume` appends to this list rather than replacing it,
so a checkpoint's full training history — how many epochs total, across how
many separate runs, and what every curve looked like each time — is readable
straight from the file, e.g.:

```bash
python -m json.tool runs/milestone3/training_log.json
```

### Keep-best checkpoint

Whenever `--save` is set (no extra flag needed) and the val check is active,
the epoch with the best-so-far val exact-match fraction is also saved
separately to **`<save>/best/`** (model + tokenizer + a `best_info.json` with
the epoch, val loss/exact-match/CER that earned it, and a timestamp). This
protects against exactly the failure mode where training regresses after a
good epoch (overfitting, noise, or otherwise) and `--save-every`'s periodic
snapshot lands on a worse epoch than one you already passed:

```bash
python -m json.tool runs/milestone3-full/best/best_info.json
```

`--resume` carries the best-so-far fraction forward from the checkpoint
you're resuming from, so it keeps comparing against the true best across the
whole lineage, not just the current invocation.

## Continuing training from a checkpoint

If a run didn't train long enough (loss still high, near-misses like a dropped
`(` or a confused Greek letter), you don't need to start over from the
pretrained TrOCR checkpoint — resume from the saved model and keep going:

```bash
python scripts/train_model.py --data data --device cuda --n 400 --epochs 20 \
  --resume runs/milestone3 --save runs/milestone3
```

`--resume` loads the saved weights *and* that checkpoint's own `tokenizer.json`
(not `data/tokenizer.json`) so the decoder's vocab stays aligned with what it
was trained on. Point `--save` at the same directory to keep checkpointing in
place, or a new one to branch off. Note the optimizer state (Adam moments) is
not saved/restored, so the first few resumed epochs behave like a warm restart
rather than a perfectly seamless continuation — this is normal and still far
cheaper than retraining from the pretrained checkpoint.

## Full training

The code is device-agnostic — pass `--device cuda` on a GPU machine and raise
`--n` to cover the whole train split (8000 for the default dataset):

```bash
python scripts/train_model.py --data data --device cuda --n 8000 --epochs 10 --save runs/trocr-v1
```

CPU is fine for the overfit smoke test, but a real run over the full 8k train
set wants a **cloud GPU (CUDA)** — an integrated GPU is not worth the ROCm
setup. Move the run to Colab or a rented GPU for the full fit.

## Predict with a saved checkpoint

Once you have a `--save` directory (a checkpoint or the final model), run it on
any single image — not just the train examples the eval loop checks:

```bash
python scripts/predict.py --model runs/trocr-v1 --image data/val/images/000000.png
```

It prints the predicted Typst string. `--trocr` must match whatever checkpoint
you trained from (default `microsoft/trocr-small-printed`) since only the
*image processor* is re-loaded from it — the trained weights come from
`--model`.

## Evaluate on a split (val/test)

`predict.py` only handles one image; to check generalization, run the
checkpoint over an entire held-out split and get an exact-match score:

```bash
python scripts/evaluate.py --model runs/milestone2 --data data --split val
```

It prints `Exact match: N/M (xx.x%)` and `Mean CER` (character error rate —
edit distance normalized by gold length; 0 = perfect, closer to 1 = further
off), plus a handful of gold/pred mismatches with their individual CER to
eyeball (`--show-mismatches`, default 10). Use `--split test` for the final
check, `--n` to cap how many examples to evaluate, and `--batch-size` to trade
memory for speed on GPU.

| flag | default | meaning |
|------|---------|---------|
| `--model` | *required* | checkpoint dir saved by `train_model.py --save` |
| `--data` | `data` | dataset directory (holds `train/ val/ test/`) |
| `--split` | `val` | which split to evaluate (`train`, `val`, `test`) |
| `--trocr` | `microsoft/trocr-small-printed` | must match the checkpoint's training run |
| `--device` | `cuda` | `cpu` or `cuda` |
| `--batch-size` | 8 | examples generated per batch |
| `--n` | all | limit to the first N examples |
| `--show-mismatches` | 10 | how many gold/pred mismatches to print |

## Plot training curves

`training_log.json` (written whenever `train_model.py` is run with `--save`)
records per-epoch loss, gradient norm, learning rate, and val loss/exact-match/CER.
Render it to a PNG:

```bash
python scripts/plot_training.py --log runs/milestone3-full/training_log.json
```

Saves `training_plot.png` next to the log (override with `--out`) with five
panels: train+val loss, val exact-match %, val CER, gradient norm, and
learning rate — each on its own axis rather than sharing one plot, since their
scales (loss ~0.1-8, grad norm ~1-50, lr ~1e-5) would otherwise flatten the
smaller ones to invisible lines. The script reconstructs a single continuous
x-axis across `--save-every` checkpoints and `--resume` chains automatically,
so you don't need to run it once per invocation.

