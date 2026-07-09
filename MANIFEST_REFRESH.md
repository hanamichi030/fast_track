# Manifest refresh — uid13 slot

Served model: poker13-mlp v1 (BagMLP: 5 standardized Torch MLPs over the 180
sanitization-invariant C2 features, rank-anchored logistic output tuned for
the 0.1.34 validator reward). See poker44_model/.

This UID slot previously carried a different occupant's manifest
(poker44-ml-heuristic, repo tomkaba/poker44-miner-release) which failed
manifest review. That record does not describe this miner. This commit marks
the ownership/model change and requests a fresh validator/backend manifest
review against the manifest actually served by this repo's code.
