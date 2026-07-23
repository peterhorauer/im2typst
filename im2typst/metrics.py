"""Sequence-comparison metrics for predicted vs. gold Typst strings."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (character-level)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def cer(gold: str, pred: str) -> float:
    """Character error rate: edit distance normalized by gold length.

    Unlike exact-match (binary pass/fail), CER scores *how close* a wrong
    prediction was — e.g. one dropped character out of 40 gives CER 0.025,
    not the same "failure" as a wholly unrelated prediction.
    """
    if len(gold) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(gold, pred) / len(gold)
