# Multi Loss Optimization

## [Strategies for Balancing Multiple Loss Functions in Deep Learning](https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0)

1. Transform to single-task learning

Each loss function may have different magnitudes, making their direct (weighted) summation meaningles.

- Use Initial Loss Value
- Use Prior Loss Value
- Use Real-time Loss Value
- Weighting Via Uncertainty

2. Manipulate Gradient of Each Loss
