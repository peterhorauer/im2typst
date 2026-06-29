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

Generate the actual `(image, label)` pairs:

```bash
python scripts/generate_dataset.py --n 1000 --out data --seed 0 --unique
```

This writes:

```
data/
├── images/000000.png, 000001.png, …
└── labels.jsonl        # one {"image": "images/000000.png", "typst": "..."} per line
```

| flag | default | meaning |
|------|---------|---------|
| `--n` | 100 | number of pairs to generate |
| `--out` | `data` | output directory |
| `--max-depth` | 4 | recursion depth bound |
| `--seed` | 0 | RNG seed for reproducibility |
| `--ppi` | 200 | render resolution (pixels per inch) |
| `--unique` | off | skip duplicate label strings |

The same `--seed` reproduces the same dataset. The `data/` directory is
gitignored — regenerate it rather than committing it.

