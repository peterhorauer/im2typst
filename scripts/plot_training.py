#!/usr/bin/env python
"""Plot per-epoch training curves from a training_log.json written by train_model.py.

    python scripts/plot_training.py --log runs/milestone3-full/training_log.json

Produces one figure with five panels:
  1. train + val loss (same units — the core overfitting check)
  2. val exact-match %
  3. val CER
  4. gradient norm
  5. learning rate

Grad norm and LR get their own panels because their scales (roughly 1-50 and
~1e-5 respectively) would otherwise flatten to invisible lines next to loss.
Val exact-match and CER are kept apart too: they can diverge (CER dropping on
near-misses while exact-match stays flat), which is exactly the kind of signal
worth seeing on its own axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_curves(log_path: Path) -> dict:
    """Flatten a training_log.json's runs into continuous, epoch-indexed curves.

    A single command invocation can log several entries (periodic --save-every
    snapshots sharing one `timestamp`, then a final one) whose arrays each
    restart from epoch 1 of that invocation — later entries are strict
    prefixes-supersets of earlier ones, not a continuation. Keeping only the
    last entry per `timestamp` avoids re-plotting the same epochs twice. Each
    kept entry is then placed on the global x-axis via
    `cumulative_epochs - epochs_this_run`, which is correct across --resume
    chains without needing to know the invocation order explicitly.

    Older logs (written before val loss/CER/grad-norm/LR tracking existed)
    are missing `grad_norm_per_epoch`/`lr_per_epoch` entirely, and their val
    entries (`val_exact_match_per_epoch`, the pre-rename field) only have
    `exact`/`total`, no `loss`/`cer`. Each series therefore keeps its own
    epoch list rather than sharing one, so a field missing from an old run
    just leaves that series shorter instead of breaking the others.
    """
    runs = json.loads(log_path.read_text())["runs"]

    last_by_timestamp: dict[str, dict] = {}
    for run in runs:
        last_by_timestamp[run["timestamp"]] = run
    kept = list(last_by_timestamp.values())

    loss_epochs: list[int] = []
    loss: list[float] = []
    grad_epochs: list[int] = []
    grad_norm: list[float] = []
    lr_epochs: list[int] = []
    lr: list[float] = []
    val_epochs: list[int] = []
    val_exact_pct: list[float] = []
    val_loss_epochs: list[int] = []
    val_loss: list[float] = []
    val_cer_epochs: list[int] = []
    val_cer: list[float] = []

    for run in kept:
        offset = run["cumulative_epochs"] - run["epochs_this_run"]
        for i, l in enumerate(run.get("loss_per_epoch", []), start=1):
            loss_epochs.append(offset + i)
            loss.append(l)
        for i, g in enumerate(run.get("grad_norm_per_epoch", []), start=1):
            grad_epochs.append(offset + i)
            grad_norm.append(g)
        for i, r in enumerate(run.get("lr_per_epoch", []), start=1):
            lr_epochs.append(offset + i)
            lr.append(r)
        # val_per_epoch is the current field name; val_exact_match_per_epoch
        # is what older train_model.py versions wrote.
        val_entries = run.get("val_per_epoch", run.get("val_exact_match_per_epoch", []))
        for v in val_entries:
            e = offset + v["epoch"]
            val_epochs.append(e)
            val_exact_pct.append(100 * v["exact"] / v["total"])
            if "loss" in v:
                val_loss_epochs.append(e)
                val_loss.append(v["loss"])
            if "cer" in v:
                val_cer_epochs.append(e)
                val_cer.append(v["cer"])

    return {
        "loss_epochs": loss_epochs, "loss": loss,
        "grad_epochs": grad_epochs, "grad_norm": grad_norm,
        "lr_epochs": lr_epochs, "lr": lr,
        "val_epochs": val_epochs, "val_exact_pct": val_exact_pct,
        "val_loss_epochs": val_loss_epochs, "val_loss": val_loss,
        "val_cer_epochs": val_cer_epochs, "val_cer": val_cer,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True, help="path to training_log.json")
    p.add_argument("--out", type=Path, default=None,
                   help="output image path (default: <log's dir>/training_plot.png)")
    args = p.parse_args()

    c = _load_curves(args.log)
    out = args.out or args.log.parent / "training_plot.png"

    def _no_data(ax) -> None:
        ax.text(0.5, 0.5, "no data in this log\n(older train_model.py)",
                ha="center", va="center", transform=ax.transAxes, color="gray")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    (ax_loss, ax_exact, ax_cer), (ax_grad, ax_lr, ax_unused) = axes
    ax_unused.axis("off")

    ax_loss.plot(c["loss_epochs"], c["loss"], label="train")
    if c["val_loss_epochs"]:
        ax_loss.plot(c["val_loss_epochs"], c["val_loss"], label="val", marker="o")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    if c["val_epochs"]:
        ax_exact.plot(c["val_epochs"], c["val_exact_pct"], marker="o", color="tab:green")
    else:
        _no_data(ax_exact)
    ax_exact.set_title("Val exact-match")
    ax_exact.set_xlabel("epoch")
    ax_exact.set_ylabel("%")
    ax_exact.set_ylim(0, 100)

    if c["val_cer_epochs"]:
        ax_cer.plot(c["val_cer_epochs"], c["val_cer"], marker="o", color="tab:red")
    else:
        _no_data(ax_cer)
    ax_cer.set_title("Val CER")
    ax_cer.set_xlabel("epoch")
    ax_cer.set_ylabel("error rate")
    ax_cer.set_ylim(bottom=0)

    if c["grad_epochs"]:
        ax_grad.plot(c["grad_epochs"], c["grad_norm"], color="tab:purple")
    else:
        _no_data(ax_grad)
    ax_grad.set_title("Gradient norm")
    ax_grad.set_xlabel("epoch")

    if c["lr_epochs"]:
        ax_lr.plot(c["lr_epochs"], c["lr"], color="tab:orange")
        ax_lr.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    else:
        _no_data(ax_lr)
    ax_lr.set_title("Learning rate")
    ax_lr.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
