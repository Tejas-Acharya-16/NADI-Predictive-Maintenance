# Literature context and scope

Ramasso and Saxena's C-MAPSS benchmarking review identifies sensor noise,
operating conditions, and multiple simultaneous fault modes as important limits
on prognostic-model generalization. This supports reporting FD002 and FD004
separately rather than relying only on pooled metrics.

- Ramasso, E. and Saxena, A., *Performance Benchmarking and Analysis of
  Prognostic Methods for CMAPSS Datasets*, International Journal of
  Prognostics and Health Management, 2014. https://doi.org/10.36001/ijphm.2014.v5i2.2236

The requested adaptive-TCN reference was reviewed as a research lead, but its
full adaptive-dilation and cross-dimension architecture was not reproduced.
The completed study isolates target cap, fold-fitted condition normalization,
causal baseline deltas, trailing-window choices, tuned HGB, Extra Trees, and
Random Forest first. A compact temporal model should be a separate candidate
on the same full-range grouped folds, using a stated compute budget and early
stopping, before it can be compared with the selected HGB.
