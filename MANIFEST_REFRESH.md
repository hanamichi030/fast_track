# Manifest refresh — uid13 slot

Served model: poker13-lgbm v2 (tuned LightGBM + isotonic calibration with a
reward-fit, FPR-capped within-batch decision layer over the 180
sanitization-invariant C2 features). v2 = same fitted artifact as v1 with a
simplified decision-layer implementation. See poker44_model/.

This UID slot previously carried a different occupant's manifest
(poker44-ml-heuristic, repo tomkaba/poker44-miner-release) which failed
manifest review. That record does not describe this miner. This commit marks
the model change and requests a fresh validator/backend manifest review
against the manifest actually served by this repo's code.
