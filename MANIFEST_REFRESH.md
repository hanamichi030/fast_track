# Manifest refresh — uid13 slot

Served model: poker13-ensemble v1 (widened soft-vote bag, C2 feature set,
n_jobs=1), trained on the full public Poker44 benchmark v1.14. See
poker44_model/ (train_model.py reproduces model.joblib).

This UID slot previously carried a different occupant's manifest
(poker44-ml-heuristic, repo tomkaba/poker44-miner-release) which failed
manifest review. That record does not describe this miner. This commit marks
the ownership/model change and requests a fresh validator/backend manifest
review against the manifest actually served by this repo's code.
