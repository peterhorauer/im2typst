#!/usr/bin/env python
"""Overfit a small subset to prove the image→Typst pipeline learns.

This is milestone 1 of model training (CLAUDE.md step 4): before any large run,
train on a handful of examples and confirm the loss collapses toward zero and
the model can reproduce those exact labels. If it can't overfit a few hundred
examples, something is wired wrong — cheaper to find out now.

    ./venv/bin/python scripts/train_model.py --data data --n 200 --epochs 30

The first run downloads the pretrained TrOCR checkpoint (a few hundred MB).
On CPU this is a smoke test, not a real training run — keep --n small.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from im2typst.data import FormulaDataset          # noqa: E402
from im2typst.model import (                       # noqa: E402
    DEFAULT_TROCR, build_model, load_image_processor,
)
from im2typst.tokenizer import CharTokenizer       # noqa: E402


def _load_prior_runs(save_dir: Path) -> list[dict]:
    """Prior run entries from a checkpoint's own log, so --resume chains append
    rather than lose the history of commands/epochs that built the checkpoint."""
    log_path = save_dir / "training_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text())["runs"]
    return []


def _write_log(save_dir: Path, prior_runs: list[dict], run_entry: dict) -> None:
    all_runs = prior_runs + [run_entry]
    (save_dir / "training_log.json").write_text(json.dumps({"runs": all_runs}, indent=2))


def _val_exact_match(model, val_dataset, val_loader, tok, device) -> tuple[int, int]:
    """Generate predictions for the val subset and count exact matches.

    Leaves the model in train() mode on return so the caller's training loop
    doesn't need to remember to flip it back.
    """
    model.eval()
    exact = 0
    idx = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            gen = model.generate(pixel_values)
            for g in gen:
                pred = tok.decode(g.tolist())
                gold = val_dataset.rows[idx]["typst"]
                exact += pred == gold
                idx += 1
    model.train()
    return exact, idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"), help="dataset directory")
    p.add_argument("--n", type=int, default=200, help="how many train examples to overfit")
    p.add_argument("--epochs", type=int, default=30, help="passes over the subset")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="AdamW learning rate")
    p.add_argument("--max-length", type=int, default=128, help="decoder max length")
    p.add_argument("--trocr", default=DEFAULT_TROCR, help="pretrained TrOCR checkpoint")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--eval-samples", type=int, default=5,
                   help="how many train examples to decode after training")
    p.add_argument("--val-n", type=int, default=50,
                   help="how many val examples to check each epoch for exact-match (0 disables)")
    p.add_argument("--val-every", type=int, default=1,
                   help="run the val exact-match check every N epochs (0 disables)")
    p.add_argument("--save", type=Path, default=None, help="optional dir to save the model")
    p.add_argument("--save-every", type=int, default=5,
                   help="checkpoint to --save every N epochs (0 disables)")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue training from a checkpoint dir saved by --save, "
                        "instead of starting over from the pretrained TrOCR checkpoint")
    args = p.parse_args()

    device = torch.device(args.device)
    image_processor = load_image_processor(args.trocr)

    prior_runs: list[dict] = []
    cumulative_epochs = 0
    if args.resume:
        # Load the checkpoint's own tokenizer, not data/tokenizer.json — the
        # decoder's embedding/lm_head rows were sized to (and trained on) this
        # exact vocab, so IDs must come from the same file to stay aligned.
        print(f"Resuming from checkpoint {args.resume}…")
        tok = CharTokenizer.load(args.resume / "tokenizer.json")
        model = VisionEncoderDecoderModel.from_pretrained(args.resume).to(device)
        prior_runs = _load_prior_runs(args.resume)
        if prior_runs:
            cumulative_epochs = prior_runs[-1]["cumulative_epochs"]
    else:
        tok = CharTokenizer.load(args.data / "tokenizer.json")
        print(f"Loading TrOCR ({args.trocr}) + resizing decoder to vocab {tok.vocab_size}…")
        model = build_model(tok, args.trocr, max_length=args.max_length).to(device)

    dataset = FormulaDataset(args.data / "train", tok, image_processor,
                             max_length=args.max_length, limit=args.n)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Overfitting {len(dataset)} examples for {args.epochs} epochs "
          f"on {device}.")

    val_dataset = val_loader = None
    if args.val_n and args.val_every:
        val_dataset = FormulaDataset(args.data / "val", tok, image_processor,
                                     max_length=args.max_length, limit=args.val_n)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            print(f"Checking val exact-match on {len(val_dataset)} examples "
                  f"every {args.val_every} epoch(s).")
        else:
            val_dataset = None

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    loss_history: list[float] = []
    val_history: list[dict] = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
        avg_loss = total / len(loader)
        loss_history.append(avg_loss)

        val_str = ""
        if val_loader is not None and epoch % args.val_every == 0:
            v_exact, v_total = _val_exact_match(model, val_dataset, val_loader, tok, device)
            val_history.append({"epoch": epoch, "exact": v_exact, "total": v_total})
            val_str = f"  val exact {v_exact}/{v_total} ({100 * v_exact / v_total:.1f}%)"
        print(f"  epoch {epoch:3d}  avg loss {avg_loss:.4f}{val_str}")

        # Periodic checkpoint so a long CPU run isn't all-or-nothing on interrupt.
        if args.save and args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
            args.save.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.save)
            tok.save(args.save / "tokenizer.json")
            _write_log(args.save, prior_runs, {
                "timestamp": started_at,
                "command": " ".join(sys.argv),
                "resumed_from": str(args.resume) if args.resume else None,
                "n_train": len(dataset),
                "epochs_this_run": epoch,
                "epochs_this_run_planned": args.epochs,
                "cumulative_epochs": cumulative_epochs + epoch,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "trocr": args.trocr,
                "device": str(device),
                "loss_per_epoch": loss_history,
                "val_n": args.val_n,
                "val_exact_match_per_epoch": val_history,
                "status": "interrupted-or-in-progress",
            })
            print(f"  [checkpoint] saved to {args.save} after epoch {epoch}")

    # --- did it actually learn? decode a few training examples ---------------
    model.eval()
    print(f"\nGenerating {args.eval_samples} train examples (want exact matches):")
    exact = 0
    with torch.no_grad():
        for i in range(min(args.eval_samples, len(dataset))):
            item = dataset[i]
            pv = item["pixel_values"].unsqueeze(0).to(device)
            gen = model.generate(pv)   # length bounded by generation_config.max_length
            pred = tok.decode(gen[0].tolist())
            gold = dataset.rows[i]["typst"]
            ok = pred == gold
            exact += ok
            print(f"  [{'✓' if ok else '✗'}] gold: {gold}")
            if not ok:
                print(f"      pred: {pred}")
    print(f"\nExact match: {exact}/{min(args.eval_samples, len(dataset))} "
          f"({'pipeline learns ✓' if exact else 'not yet — train longer / check wiring'})")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save)
        tok.save(args.save / "tokenizer.json")
        n_eval = min(args.eval_samples, len(dataset))
        _write_log(args.save, prior_runs, {
            "timestamp": started_at,
            "command": " ".join(sys.argv),
            "resumed_from": str(args.resume) if args.resume else None,
            "n_train": len(dataset),
            "epochs_this_run": args.epochs,
            "cumulative_epochs": cumulative_epochs + args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "trocr": args.trocr,
            "device": str(device),
            "loss_per_epoch": loss_history,
            "val_n": args.val_n,
            "val_exact_match_per_epoch": val_history,
            "train_eval_exact_match": f"{exact}/{n_eval}",
            "status": "complete",
        })
        print(f"Saved model + tokenizer + training_log.json to {args.save}")


if __name__ == "__main__":
    main()
