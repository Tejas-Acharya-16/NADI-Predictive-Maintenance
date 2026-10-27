# HGB tuning record

The bounded search used the same five engine folds and horizons for every candidate. The HGB family covered target caps 100, 125, and 150 plus uncapped labels; 5/20 and 10/30 trailing windows; initial-reading differences; training-only near-constant sensor removal; time-feature removal; a small-leaf setting (`max_iter=240`, `learning_rate=0.05`, `max_leaf_nodes=15`, `min_samples_leaf=15`, `l2_regularization=2.0`); and `absolute_error` loss.

The validation winner uses the standard bounded setting: `max_iter=180`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=False`, and cap 100. It was selected by lower pooled PHM08 mean, then RMSE. This was a limited, documented search rather than an exhaustive hyperparameter sweep.
