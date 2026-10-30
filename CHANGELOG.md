# Changelog

## 0.1.0 — 2026-10-30

### Added

- Installable package metadata with a small `kahm.KAHMClassifier` interface.
- Standard samples-by-features inputs, preserved labels, in-memory fitting, and
  compatibility with scikit-learn cloning, pipelines, and grid search.
- A bundled-data classification comparison with a fixed development/test split,
  shared cross-validation folds, six configurations per tuned method, and full
  result records.
- A README, classifier guide, paper-to-code map, citation metadata, contribution
  guide, and explicit software-license status.
- Numerical, estimator, and experiment-protocol tests; CI and distribution builds.

### Repaired

- Duplicate distance computation in the classifier's soft-score path.
- Avoidable Gaussian query underflow in normalized reconstruction and unstable
  normalization of very small positive class scores.
- Reference-kernel evaluation overhead without changing the intended Gaussian
  formula.
- Degenerate local reference sets, invalid block/dimension inputs, and
  roundoff-negative squared distances.

### Compatibility and limits

- Existing top-level classification and regression imports remain available.
- The adapter changes array orientation at its boundary; legacy functions retain
  the features-by-samples convention.
- Numerical repairs can change earlier floating-point outputs in affected edge
  cases. The new example is not a reproduction of historical paper results.
- Normalized class scores are not guaranteed to be calibrated probabilities.
- Regression and optional quantum-optimization scripts remain legacy research
  interfaces; this release's polished example concerns classification.
- No repository-wide software license is assigned by this release.
