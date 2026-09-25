"""Small, dependency-free statistics helpers."""

from __future__ import annotations

import math
import random
from typing import Hashable, Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with Wilson 95% interval: (p, lo, hi). NaNs when n == 0."""
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_diff(a: dict[Hashable, float], b: dict[Hashable, float], iters: int = 2000,
                          seed: int = 0) -> tuple[float, float, float, int]:
    """Mean(b - a) over keys present in both, with a 95% percentile bootstrap interval.
    Returns (diff, lo, hi, n_paired)."""
    keys = sorted(set(a) & set(b), key=str)
    n = len(keys)
    if n == 0:
        return (math.nan, math.nan, math.nan, 0)
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1], n)


def binary_metrics(pairs: Sequence[tuple[int, int]]) -> dict:
    """pairs of (label, prediction), 1 = unsafe / deny. Returns precision/recall/F1/FPR/counts."""
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0)
    tn = sum(1 for y, p in pairs if y == 0 and p == 0)
    prec = tp / (tp + fp) if tp + fp else math.nan
    rec = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else math.nan
    fpr = fp / (fp + tn) if fp + tn else math.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))
    return s[idx]
