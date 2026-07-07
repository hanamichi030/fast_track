"""Reproducible training for poker13-ensemble -> writes model.joblib.

Widened soft-vote bag (ExtraTrees seed17 + ExtraTrees seed24 + RandomForest
seed17 + HistGradientBoosting seed17), all n_jobs=1, min_samples_leaf=2 on
the forests and max_iter=500 on the HGB, over the 180 sanitization-invariant
FEATURE_NAMES (see features.py). Trained on the FULL public Poker44 benchmark
v1.14 (886 labeled chunk groups, releases 2026-05-26 .. 2026-07-07, including
the pattern_hardened_v2 bot releases 2026-07-06/07).

Selection evidence (date-held-out, validator reward = 0.75*AP + 0.25*recall@FPR<=5%),
two folds (train<=07-06 test 07-07; train<=07-05 test 07-06..07):
  - this config beats the leaf=4 / hgb_iter=300 base on BOTH folds
    (0.9006/0.8719 vs 0.8972/0.8677) — the only tested knobs that did;
  - the base itself scores 0.897 vs 0.893 for the prior-generation artifact
    at identical training coverage on the unseen 07-07 hardened release;
  - hardened-only or hardened-upweighted training mixes LOSE to training on
    all releases equally; dropping the sig_* duplication features costs
    ~-0.015 reward even on hardened bots (they still carry real signal).

DEPLOYABLE RULE: every learner is n_jobs=1 (n_jobs=-1 deadlocks the
validator's batched predict path).

Every raw benchmark hand is passed through the validator's
`prepare_hand_for_miner` (payload_view.py) BEFORE feature extraction, so the
training distribution matches what the validator serves (train==serve). Live
chunks are already sanitized validator-side, so inference does NOT re-sanitize.

    python3 poker44_model/train_model.py --data <path to benchmark raw dir> \
        --payload-view <path to poker44/validator/payload_view.py>

The benchmark is public: https://api.poker44.net/api/v1/benchmark (no auth).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import typing

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier,
                              VotingClassifier)

from poker44_model.features import chunk_features, FEATURE_NAMES

SEED = 17


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional  # payload_view uses Optional but never imports it
    fn = pv.prepare_hand_for_miner

    def sanitize_chunk(chunk):
        out = []
        for h in (chunk or []):
            try:
                out.append(fn(h))
            except Exception:
                out.append(h)
        return out

    return sanitize_chunk


def load(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def build_ensemble(seed=SEED):
    # DEPLOYABLE: every learner n_jobs=1 (n_jobs=-1 deadlocks batched predict).
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                              random_state=seed, n_jobs=1)
    et2 = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                               random_state=seed + 7, n_jobs=1)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                random_state=seed, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=500, l2_regularization=1.0,
                                         random_state=seed)
    return VotingClassifier(
        estimators=[("et", et), ("et2", et2), ("rf", rf), ("hgb", hgb)],
        voting="soft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to benchmark raw chunk JSON dir")
    ap.add_argument("--payload-view", required=True,
                    help="path to poker44/validator/payload_view.py (the sanitizer)")
    args = ap.parse_args()

    sanitize_chunk = _load_sanitizer(args.payload_view)

    data = load(args.data)
    rows, y = [], []
    for g, l in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE: sanitize raw hands
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
    X = np.array(rows, dtype=float)
    y = np.array(y)

    model = build_ensemble().fit(X, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features, seed={SEED})")


if __name__ == "__main__":
    main()
