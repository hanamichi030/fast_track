# Manifest refresh — uid13 slot

Served model: poker13-varfuse v1 (within-batch rank-fused 3-member ensemble —
stacked LGBM+XGB+RF, monotone LGBM bag, StandardScaler->PCA->MLP — over the
180 sanitization-invariant C2 features, with a strictly-monotone tie-free
decision layer and no isotonic calibration). See poker44_model/.

This UID slot previously carried a different occupant's manifest
(poker44-ml-heuristic, repo tomkaba/poker44-miner-release) which failed
manifest review. That record does not describe this miner. This commit marks
the model change and requests a fresh validator/backend manifest review
against the manifest actually served by this repo's code.
