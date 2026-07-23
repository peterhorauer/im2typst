# Training strategy & milestones

This file tracks the model-training approach, the milestone ladder, results so far, and the strategy for future runs. For **how to run** training, see the "Training the model" section of [README.md](README.md).

## Architecture (recap)

- **Vision Encoder-Decoder**: a pretrained **TrOCR** image encoder (reused as-is with its bundled image processor) + a decoder whose vocabulary is resized to our char-level `CharTokenizer`.
- Dev checkpoint: `microsoft/trocr-small-printed` (light enough for CPU smoke tests; the "printed" variant matches our clean renders). Consider `trocr-base-*` for the real run if the small model underfits.
- The vision side (image processor + encoder) and text side (tokenizer + decoder) are **decoupled** — swapping the tokenizer later (char → BPE) only re-sizes the decoder; the encoder is untouched.

## Milestone ladder

### Milestone 1 — Overfit smoke test ✅ done (2026-07-13)

- **Goal:** prove the full pipeline learns end to end before spending on a big run.
- **Setup:** 16 train examples, 80 epochs, CPU, `trocr-small-printed`.
- **Result:** loss **8.1 → 0.28**; **3/8 exact matches**; the misses were *near*-misses (one dropped character, one unbalanced paren, one degenerate repetition on the longest label).
- **Verdict:** the image → pixels → encoder → decoder → tokens → loss → generation → decode loop is validated. Near-misses (not noise) confirm it is learning structure.
- **Gotcha logged:** transformers v5 requires generation params on `model.generation_config`, not `model.config`.

### Milestone 2 — Clean overfit (push to ~8/8) ✅ done (2026-07-22)

- **Goal:** confirm the model can *perfectly* memorize a small set — rules out
  capacity/wiring/truncation problems.
- **Setup:** 50 train examples (from a fresh 500-image dataset), 200 epochs,
  CUDA (new hardware), `trocr-small-printed`.
- **Result:** train predictions matched very well (exact match or near exact match) — confirmed with `predict.py` on a train image (exact match). Evaluated on the **val** split (50 examples, `scripts/evaluate.py`): **0/50 exact match**, and predictions differed wildly from the gold labels — not near-misses, effectively unrelated output.
- **Verdict:** expected for this milestone, not a bug. With only 50 train examples and 200 epochs, the model memorized those specific images rather than learning to generalize — there's no pressure to learn anything beyond a near lookup-table mapping, so total failure on unseen images is the normal outcome. Double-checked `predict.py`/`evaluate.py` load the same checkpoint + tokenizer consistently, ruling out an eval-script bug.
- **Next:** Milestone 3 — train on the full train split (not an overfit subset) with moderate epochs and re-check val exact-match. Note the dataset is currently only 500 images (400 train) — well below the 10k–100k scale TRAINING.md calls for, so real generalization likely needs a bigger generated dataset even after Milestone 3.

### Milestone 3 — First real run (generalization)

- **Goal:** train on the full 8k train split and measure generalization on the held-out **val** split.
- **Requires (not yet built):**
  - a **validation loop** (val loss + exact-match each epoch) — the current script is train-only,
  - **checkpointing** / keep-best-model.
- **Strategy:** GPU (CUDA); batch 16–32; 10–30 epochs; linear/cosine warmup; track val exact-match and early-stop on it. Move to `trocr-base` if underfitting.
- **Success:** high val exact-match on labels never seen in training.

### Milestone 4 — Evaluation & robustness (roadmap step 5)

- **Metrics:** exact-match, token edit distance, BLEU, and **render-match** (compile the prediction, re-render, compare to the input image — proves *semantic* correctness even when the string differs).
- **Robustness:** regenerate the dataset with augmentation (fonts, sizes, background tint, noise) via `RenderOptions` so the model generalizes to real screenshots.
- **Tokenizer v2:** byte-level BPE (shorter sequences; keyword tokens like `sqrt`). Swap the decoder vocab and retrain — encoder unchanged.

## Strategy notes for future runs

### Compute
- **CPU (Ryzen 7735U):** fine for overfit/smoke tests only.
- **Cloud GPU (Colab T4 / rented A10/A100):** the real answer (10–100× faster). Code is device-agnostic: `--device cuda`.

### What to add before the big run
- Validation loop (per-epoch val loss + metrics).
- Checkpoint/save-best (`--save` exists; extend to periodic or best-only).
- The metrics harness from milestone 4.

### Scaling knobs
- **Data:** 10k now; scale to 50–100k for a strong model. Label generation is cheap; **rendering** is the cost — plan render time accordingly.
- **`max_length`:** keep ≥ the tokenizer's reported max (98); default 128 is safe.
- **Model size:** start `trocr-small`; move to `trocr-base` if it underfits.

### Reproducibility
Record per run: dataset seed, TrOCR checkpoint, lr, batch size, epochs, and the resulting metrics (use the run log below).

## Run log

| date | milestone | model | data | epochs | result |
|------|-----------|-------|------|--------|--------|
| 2026-07-13 | 1 — overfit smoke test | trocr-small-printed | 16 train | 80 | loss 0.28, 3/8 exact (near-misses) |
| 2026-07-22 | 2 — clean overfit | trocr-small-printed | 50 train (500-image set) | 200 | train: exact match; val (50): 0/50, wildly off — expected (pure memorization, no generalization pressure) |
