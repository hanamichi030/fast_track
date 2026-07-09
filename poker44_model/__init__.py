"""Participant-owned model package for the Poker44 miner — poker13-mlp.

Bot detector = BagMLP: a 5-member bag of standardized Torch MLPs
(512-256-128, dropout 0.3, early-stopped on validation loss) over the 180
sanitization-invariant C2 behavioral features (see features.py
FEATURE_NAMES). Trees collapse to a near-flat predict_proba on the
validator-sanitized live feed; standardized MLPs extrapolate and preserve a
discriminative within-batch ordering. Output is a rank-anchored logistic
tuned for the 0.1.34 validator reward (0.35*AP + 0.30*recall@FPR<=5% +
0.30*threshold blocks): the top ~10% of each served batch crosses 0.5, so a
true bot is essentially always flagged (no hard zero) while the hard human
FPR stays under the 0.10 cap. Training hands pass through the validator's
prepare_hand_for_miner (train==serve); inference does NOT re-sanitize. No
capture-fitted domain adaptation and no query-chunk fitting. See detector.py
(inference + output transform), mlp_bag.py / mlp_member.py (model classes),
features.py (extraction), model.joblib (the fitted artifact).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
