#!/usr/bin/env python
"""Print sample synthetic Typst formulas to stdout (no rendering).

For eyeballing the grammar's output before committing to a full render run::

    ./venv/bin/python scripts/demo_generate.py --n 20 --max-depth 4 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.generator import Grammar  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=15, help="how many to print")
    p.add_argument("--max-depth", type=int, default=4, help="recursion depth bound")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = p.parse_args()

    g = Grammar(max_depth=args.max_depth, seed=args.seed)
    for _ in range(args.n):
        print(g.sample_typst())


if __name__ == "__main__":
    main()
