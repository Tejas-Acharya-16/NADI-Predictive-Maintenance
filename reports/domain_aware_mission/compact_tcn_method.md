# Compact TCN with sensor and temporal attention

This CPU-bounded architecture uses three dilated residual TCN blocks (dilations 1, 2, 4), channel attention and temporal attention. Each fold fits its condition normalizer and scaling only on development engines. It is inspired by the cited multi-scale attention literature; it is not a reproduction of the published architecture.
