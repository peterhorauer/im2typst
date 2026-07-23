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

### Milestone 3 — First real run (generalization) ✅ done (2026-07-23)

- **Goal:** train on the full train split and measure generalization on the held-out **val** split.

**Attempt 1 — 400 examples, data-scale bottleneck found:**
- **Setup:** 400 train examples (the full train split of the then-current 500-image dataset), CUDA, `trocr-small-printed`, `runs/milestone3`. Trained 30 epochs, then resumed (`--resume`) for 30 more — 60 epochs total.
- **Result:** loss **0.3 → 0.09**. `scripts/evaluate.py` on **train**: 31/50 exact match. On **val**: 1/50 exact match, with the remaining mismatches visibly closer (fewer/smaller errors) than Milestone 2's "wildly off" predictions.
- **Verdict:** an **overfitting signature**, not an epoch-starved one — train performance (62%) far ahead of val (2%), widening as training continued. Root cause diagnosed as **dataset size/diversity** (only 400 train, well below the 10k–100k target), not training duration.

**Attempt 2 — 9,000 examples, fix confirmed:**
- **Setup:** 9,000 train examples, CUDA, `trocr-small-printed`, batch size 8 (16 OOM'd on the 3.63 GiB GPU), 15 epochs, `runs/milestone3-full`.
- **Result:** loss **1.58 → 0.026**, still falling at epoch 15. Val exact-match rose 0% → peaked **92% (epoch 13)**, ending **86% (epoch 15)**. Final check on the held-out **test** split (`scripts/evaluate.py --split test`): **416/500 exact match (83.2%)**.
- **Verdict:** confirms the Attempt 1 diagnosis — scaling train data 400 → 9,000 fixed the overfitting; val/test now track train instead of being "wildly off". Train loss kept improving through epoch 15 while val oscillated in the 74–92% band from epoch 8 on — a generalization plateau for this data size/epoch count, not a regression.
- **Note:** this run predates the keep-best-checkpoint feature, so the saved checkpoint is just epoch 15 (86% val) rather than the true best epoch seen (epoch 13, 92% val). Future runs auto-preserve the best epoch via `--save`.
- **Success criterion met:** high val/test exact-match (83–92%) on labels never seen in training.
- **Built along the way (previously listed as "not yet built"):** per-epoch **val loss + exact-match + CER** check, gradient-norm/LR logging, **keep-best checkpointing** (auto-enabled with `--save`), and `scripts/plot_training.py` for visualizing `training_log.json`.
- **Next:** Milestone 4 — augment the dataset (fonts, sizes, noise, rotation via `RenderOptions`) so the model generalizes beyond clean synthetic renders to real screenshots, and add the render-match/BLEU metrics.

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
| 2026-07-23 | 3 — first real run, attempt 1 | trocr-small-printed | 400 train (full split, 500-image set) | 60 (30 + 30 resumed) | loss 0.3→0.09; train 31/50 exact; val 1/50 exact — overfitting signature, dataset too small, not epoch-starved |
| 2026-07-23 | 3 — first real run, attempt 2 | trocr-small-printed | 9,000 train | 15 | loss 1.58→0.026; val exact-match 0%→92% (peak epoch 13), 86% at epoch 15; **test 416/500 (83.2%)** — data-scale fix confirmed |
