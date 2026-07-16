"""Participant-owned model package for the Poker44 miner — poker13-varfuse.

Bot detector = a within-batch rank-fused, low-capacity 3-member ensemble over
the 180 sanitization-invariant C2 behavioral features (features.py
FEATURE_NAMES):
  * stack = StackingClassifier[LGBM + XGB + RandomForest] -> logistic meta
  * mono  = monotone-constrained LGBM bag on the sign-stable subspace
  * mlp   = StandardScaler -> PCA -> sklearn MLPClassifier
The three members are genuinely decorrelated (pairwise OOF Spearman 0.71-0.83);
their within-batch ranks are fused .35/.30/.35. A strictly-monotone, tie-free
decision layer maps the fused rank to [0,1] so the top ~10% of each served
batch crosses the hard 0.5 operating point (no hard zero, bounded human FPR)
while AP / recall@FPR are exactly those of the fused ranking. No isotonic
calibration (isotonic is non-injective and merges ranks on OOD windows).
Training hands pass through the validator's prepare_hand_for_miner
(train==serve); inference does NOT re-sanitize. No capture-fitted domain
adaptation and no query-chunk fitting. See detector.py (inference + fusion +
decision layer), features.py (extraction), model.joblib (the fitted members,
fusion weights, and monotone-subspace index).
"""

from poker44_model.detector import score_chunk, score_batch  # noqa: F401

__all__ = ["score_batch", "score_chunk"]
