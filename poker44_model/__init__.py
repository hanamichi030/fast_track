"""Participant-owned model package for the Poker44 miner — coralplus-v7.

model_name: poker-coralplus-v7

Bot detector = ExtraTrees(n_jobs=1) + HistGradientBoosting soft-vote ensemble
over the 180-feature C2 behavioral feature set, trained with a two-stage domain
adaptation baked into TRAINING: (1) CORAL aligns benchmark-train features (mean +
covariance) to the UNLABELED live feature distribution, then (2) a per-feature
monotone QUANTILE map sends each CORAL-aligned marginal onto the live feature
marginal (fixing residual marginal shape incl. the benchmark ~34h vs live ~93h
chunk-size shift). Only unlabeled live feature statistics are used; benchmark
labels are used solely for the classifier fit (no live labels exist).

Inference does NOT re-sanitize (live hands are already sanitized validator-side)
and does NOT re-apply either alignment stage (live is already in the aligned
space); scores are within-batch ranks. See features.py (extraction +
FEATURE_NAMES), train_model.py (two-stage training), quantstack_transform.npz
(baked mu_src/mu_tgt/W + quantile grids), model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
