"""Participant-owned model package for the Poker44 miner — poker-c2-ensemble.

C2 variant: WIDENED soft-vote bag. Bot detector = ExtraTrees +
HistGradientBoosting + RandomForest + second-seed ExtraTrees soft-vote over the
EXACT C2 (v5 sani) behavioral feature set (v3 features minus the fragile
identity / raw-magnitude aggregates). The added RandomForest and second-seed
ExtraTrees are pure variance reduction over the C2 vote — snapshot-agnostic, no
capture-fitted domain adaptation. Trained on benchmark hands passed through the
validator's prepare_hand_for_miner (train==serve), same sanitizer as C2; scored
by within-batch ranking. Inference does NOT re-sanitize (live hands arrive
already sanitized validator-side). features.py / detector.py are byte-identical
to C2; only model.joblib differs (a distinct fitted artifact). See detector.py
(inference), features.py (extraction + FEATURE_NAMES), train_model.py (training),
model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
