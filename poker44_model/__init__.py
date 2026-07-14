"""Participant-owned model package for the Poker44 miner — poker13-lgbm.

Bot detector = tuned LightGBM (1200 trees, lr 0.02, 63 leaves — the measured
grouped-by-date AP ceiling on the public benchmark) over the 180
sanitization-invariant C2 behavioral features (features.py FEATURE_NAMES),
plus an isotonic calibrator fit on grouped-by-date out-of-fold predictions
and a reward-fit, FPR-capped within-batch decision layer tuned for the
0.1.34 validator reward: a small top fraction of each served batch crosses
the hard 0.5 operating point (no hard zero, hard human-FPR well under the
0.10 cap), while the transform stays monotone in the calibrated probability.
Training hands pass through the validator's prepare_hand_for_miner
(train==serve); inference does NOT re-sanitize. No capture-fitted domain
adaptation and no query-chunk fitting. See detector.py (inference + decision
layer), features.py (extraction), model.joblib (the fitted artifact: LGBM +
isotonic + decision constants).
"""

from poker44_model.detector import score_chunk, score_batch  # noqa: F401

__all__ = ["score_batch", "score_chunk"]
