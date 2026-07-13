# im2typst
A formula screenshot to typst math converter

# goal
The goal is to create a image 2 typst converter which is able to take any formula found online and easily convert it into a typst formula.

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

Generate the actual `(image, label)` pairs, partitioned into disjoint
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

