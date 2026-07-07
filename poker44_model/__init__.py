"""Participant-owned model package for the Poker44 miner — poker13-ensemble.

Bot detector = widened soft-vote bag: ExtraTrees (seeds 17/24) + RandomForest
(seed 17) + HistGradientBoosting (seed 17) over the 180 sanitization-invariant
behavioral features (the C2 feature set: v3 features minus the fragile
identity / raw-magnitude aggregates). Trained on the full public Poker44
benchmark v1.14 (886 labeled groups through 2026-07-07, including the
pattern_hardened_v2 bot releases) with every hand passed through the
validator's prepare_hand_for_miner (train==serve); scored by within-batch
ranking. Inference does NOT re-sanitize (live hands arrive already sanitized
validator-side). No capture-fitted domain adaptation and no query-chunk
fitting. See detector.py (inference), features.py (extraction +
FEATURE_NAMES), train_model.py (reproducible training), model.joblib (the
fitted artifact).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
