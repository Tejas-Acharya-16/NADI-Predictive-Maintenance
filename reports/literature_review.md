# Literature review and comparison limits

NASA C-MAPSS is a simulated multivariate turbofan benchmark with censored test trajectories and one RUL value per test engine. The project uses FD001 only for the completed experiments.

Ensarioglu, Inkaya, and Emel (2023), Applied Sciences 13(21), 11893, motivates the compact CNN-LSTM, sensor-difference, and change-point discussion. The project implements compact CNN/LSTM/CNN-LSTM and initial-reading differences. It does not claim to reproduce the paper's change-point labels because a complete run-to-failure change point would be unavailable for a censored deployment trajectory.

Ramasso and Saxena (2014) informs the grouped validation and protocol caution. The 2021 Symmetry deep-learning paper is retained as an architecture reference. Published scores are not treated as directly comparable without matching labels, preprocessing, censoring, split, and PHM08 aggregation.
