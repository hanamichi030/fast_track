"""v7_coralplus candidate CORAL_QUANTSTACK.

CORAL (v6_da) aligns only the 2nd-order (covariance) statistics of the
benchmark-train features to the UNLABELED live feature covariance, then retrains
C2's ensemble on the aligned features. Inference is identity (live is already in
live space). It hit live raw-STD 0.133 / dup-corr 0.606.

Residual gap after CORAL:
  (A) covariance alignment matches only means + 2nd moments; per-feature MARGINAL
      SHAPE (skew, multimodality, quantiles) is still benchmark-shaped.
  (B) chunk size: benchmark ~34 hands/chunk vs live ~93. Many features
      (hand_count, signature top/unique shares, entropy/quantile aggregates over
      more hands) have marginals that shift purely with chunk length. CORAL only
      matches their 2nd moment; the full marginal is wrong.

QUANTSTACK stacks a second, marginal-level domain-adaptation step ON TOP of
CORAL: after CORAL aligns the joint 2nd-order statistics, apply a per-feature
monotone QUANTILE MAP that sends each CORAL-aligned source feature's empirical
CDF onto the LIVE feature's empirical CDF. Covariance handles the joint 2nd
order; the quantile map fixes the residual marginal shape (including the
chunk-size-induced marginal shift, axis B) feature by feature. Then retrain C2's
exact ensemble on the doubly-aligned features.

    X1 = (X_src - mu_src) @ W + mu_tgt                       # CORAL
    X2[:, j] = Q_tgt_j( F_aligned_j( X1[:, j] ) )            # per-feature quantile map
    model = Ensemble.fit(X2, y)

Q_tgt_j is the live target quantile function for feature j; F_aligned_j is the
empirical CDF of the CORAL-aligned source feature j. Both estimated from the
UNLABELED live captures (no live labels — feature marginals only). The map is
monotone per feature so it cannot invent rank structure the trees would exploit
spuriously; it only reshapes each feature's marginal to match live.

Inference is IDENTITY on live features (live is already in live space) — both
alignment steps are baked into TRAINING, exactly like CORAL. Bakes
model.joblib + quantstack_transform.npz (CORAL params + per-feature quantile
grids) for reference / benchmark-space use.

Ensemble = ExtraTrees(n_jobs=1) + HistGBM soft-vote, hyperparameters identical to
C2 / CORAL. No thresholds / clips / rank tricks.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import typing

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import chunk_features, FEATURE_NAMES  # noqa: E402

RAW_DIR = "/root/ares/Poker/train/raw"
PV_PATH = "/root/ares/Poker/main/poker44/validator/payload_view.py"
CAP_DIRS = [
    "/root/ares/Poker/Poker44-uid7/poker44_model/captures",
    "/root/ares/Poker/Poker44-uid73/poker44_model/captures",
]
EPS = 1e-6            # Tikhonov ridge on covariances (numerical stability)
N_QGRID = 1000        # quantile grid resolution for the marginal map


# --- sanitizer (train==serve) ------------------------------------------------
def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional
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


# --- data --------------------------------------------------------------------
def load_benchmark(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def load_live_unique():
    """Unique live chunk sets deduped by top-level chunk_hash (unlabeled)."""
    seen = {}
    for cd in CAP_DIRS:
        for f in glob.glob(os.path.join(cd, "*", "*", "chunks.json")):
            d = json.load(open(f))
            seen.setdefault(d["chunk_hash"], d["chunks"])
    live = []
    for h in sorted(seen):
        live.extend(seen[h])
    return live


def featurize(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


# --- CORAL -------------------------------------------------------------------
def _sqrtm_psd(C):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _inv_sqrtm_psd(C):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, EPS, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def fit_coral(X_src, X_tgt):
    d = X_src.shape[1]
    mu_src = X_src.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)
    Cs = np.cov(X_src, rowvar=False) + EPS * np.eye(d)
    Ct = np.cov(X_tgt, rowvar=False) + EPS * np.eye(d)
    W = _inv_sqrtm_psd(Cs) @ _sqrtm_psd(Ct)
    return mu_src, mu_tgt, W


def apply_coral(X, mu_src, mu_tgt, W):
    return (X - mu_src) @ W + mu_tgt


# --- per-feature quantile map (marginal alignment) ---------------------------
def fit_quantile_map(X_src_aligned, X_tgt, n_grid=N_QGRID):
    """For each feature j build a monotone map sending the CORAL-aligned source
    marginal onto the live target marginal. Represented by a shared probability
    grid p and, per feature, the source-quantile knots (src_q) and the
    target-quantile knots (tgt_q). Applying: for value x, find its rank via
    src_q (empirical CDF), then read tgt_q at that probability."""
    p = np.linspace(0.0, 1.0, n_grid)
    d = X_src_aligned.shape[1]
    src_q = np.empty((d, n_grid))
    tgt_q = np.empty((d, n_grid))
    for j in range(d):
        src_q[j] = np.quantile(X_src_aligned[:, j], p)
        tgt_q[j] = np.quantile(X_tgt[:, j], p)
    return p, src_q, tgt_q


def apply_quantile_map(X, p, src_q, tgt_q):
    """Map each column through F_src_aligned (empirical CDF) then Q_tgt (target
    quantile). Monotone, feature-wise. Degenerate (constant) source columns are
    passed through unchanged so the map is well defined everywhere."""
    out = np.empty_like(X, dtype=float)
    d = X.shape[1]
    for j in range(d):
        sq = src_q[j]
        tq = tgt_q[j]
        # empirical CDF of aligned source: value -> probability via interp on knots
        if sq[-1] - sq[0] <= 1e-12:
            out[:, j] = X[:, j]  # constant source feature; leave untouched
            continue
        pr = np.interp(X[:, j], sq, p)          # F_src_aligned(x) in [0,1]
        out[:, j] = np.interp(pr, p, tq)        # Q_tgt(pr)
    return out


# --- ensemble (identical to C2 / CORAL) --------------------------------------
def build_ensemble(seed=0):
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=1)  # baked n_jobs=1
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=300, l2_regularization=1.0,
                                         random_state=seed)
    return VotingClassifier(estimators=[("et", et), ("hgb", hgb)], voting="soft")


def main():
    sanitize_chunk = _load_sanitizer(PV_PATH)

    # source: sanitized benchmark (labeled)
    data = load_benchmark(RAW_DIR)
    src_chunks = [sanitize_chunk(g) for g, _ in data]
    y = np.array([l for _, l in data])
    X_src = featurize(src_chunks)

    # target: unlabeled live captures (already sanitized) — marginals + covariance
    live_chunks = load_live_unique()
    X_tgt = featurize(live_chunks)

    print(f"src {X_src.shape}  tgt {X_tgt.shape}  pos={int(y.sum())}")

    # step 1: CORAL 2nd-order alignment
    mu_src, mu_tgt, W = fit_coral(X_src, X_tgt)
    X1 = apply_coral(X_src, mu_src, mu_tgt, W)

    # step 2: per-feature quantile map to live marginals (stacked on CORAL)
    p, src_q, tgt_q = fit_quantile_map(X1, X_tgt)
    X2 = apply_quantile_map(X1, p, src_q, tgt_q)

    model = build_ensemble(seed=0).fit(X2, y)

    joblib.dump(model, os.path.join(HERE, "model.joblib"))
    np.savez(os.path.join(HERE, "quantstack_transform.npz"),
             mu_src=mu_src, mu_tgt=mu_tgt, W=W,
             qmap_p=p, qmap_src_q=src_q, qmap_tgt_q=tgt_q,
             feature_names=np.array(FEATURE_NAMES))
    print("wrote model.joblib + quantstack_transform.npz")


if __name__ == "__main__":
    main()
