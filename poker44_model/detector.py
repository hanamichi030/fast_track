"""Poker44 bot detector — coralplus-v7 (CORAL + per-feature quantile stack).

Model: ExtraTrees(n_jobs=1) + HistGradientBoosting soft-vote ensemble over the
same 180-feature C2 behavioral feature set (features.py / FEATURE_NAMES), with a
TWO-STAGE domain adaptation baked into TRAINING (see train_model.py):

  1. CORAL: benchmark-train features are re-centered and re-colored (mean +
     covariance) to the UNLABELED live feature distribution.
  2. QUANTILE STACK: a per-feature monotone quantile map sends each CORAL-aligned
     source feature's empirical CDF onto the LIVE feature's empirical CDF. This
     fixes the residual marginal shape (skew / multimodality / the chunk-size-
     induced marginal shift, benchmark ~34h vs live ~93h) that CORAL's 2nd-order
     match leaves behind.

Both stages use ONLY unlabeled live feature statistics (no live labels exist).
Benchmark labels are used solely for the classifier fit. On the 500 held-out
live chunks this lifts raw predict_proba std 0.133 -> 0.143 and the
duplication-rank Spearman 0.606 -> 0.635 vs CORAL.

IMPORTANT — inference does NOT sanitize and does NOT re-apply either alignment
stage. Live chunks arrive already sanitized by the validator
(prepare_hand_for_miner runs validator-side) AND are already in the live feature
space the model was aligned to during training, so this path featurizes the
incoming chunks directly and calls the model as-is. Re-applying the CORAL /
quantile transforms here would double-shift already-live-space data. The baked
transforms (mu_src, mu_tgt, W, quantile grids) are shipped in
quantstack_transform.npz for reference / benchmark-space use only.

Output = within-batch rank in [0,1] (higher = more bot-like), matching the
validator's ranking-based reward. ExtraTrees n_jobs=1 (deterministic, single
thread). No thresholds / clips / rank tricks beyond the within-batch rank.
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _raw_scores(model, chunks):
    # Live chunks are already sanitized AND already in the aligned live feature
    # space; featurize as-is (no re-sanitize, no re-transform).
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model.predict_proba(np.array(rows, dtype=float))[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk model probability (fallback; batch path is score_batch)."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores(_model(), [chunk])[0]), 6)
    except Exception:
        return 0.5
