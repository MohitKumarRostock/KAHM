# Classification guide

## Public estimator

```python
from kahm import KAHMClassifier
```

The adapter follows the scikit-learn estimator interface: `fit(X, y)`,
`predict(X)`, `predict_proba(X)`, `get_params()`, and `set_params()`.
Inputs have shape `(n_samples, n_features)`; targets are one-dimensional discrete
labels. Each of at least two classes must have two or more observations.

The adapter accepts finite, dense real-valued data. Missing values, sparse arrays,
sample weights, online updates, and multilabel targets are not supported in this
release. Scaling is an explicit pipeline step rather than an implicit operation
inside the estimator.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `subspace_dim` | `10` | Requested local principal-subspace dimension; the learner reduces it when rank is lower |
| `block_size` | `100` | Target reference-block size; maps to legacy `Nb` |
| `alpha` | `10.0` | Positive exponent for normalized affinity scores; does not change nearest-reconstruction prediction |
| `max_train_per_class` | `None` | Optional reference cap; `None` retains all available class observations |
| `random_state` | `0` | Seed for reference subsampling; no effect when subsampling is unused |
| `batch_size` | `None` | Optional query batch size during prediction |

The legacy learner treats a `None` seed as zero. `block_size` is not a strict
maximum: the number of blocks is approximately
`round(n_class_samples / block_size)`, and observations are divided into balanced
consecutive groups. The implementation does not learn a clustering for this
classifier. The block count is capped at `floor(n_class_samples / 2)` so that
every block has at least two reference observations. Ordering therefore matters
when more than one block is used.

The adapter fits in memory and makes no automatic model files or cache
directories. The legacy implementation stores reference values in float32;
subsequent computation and stored factors may use higher precision. A larger
reference block increases the cost of local dense kernel linear algebra. No
large-scale performance guarantee is implied by the small digits example.

## Preprocessing and model selection

Place a scaler inside a `Pipeline` so that cross-validation fits it only on each
training fold:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from kahm import KAHMClassifier

pipeline = Pipeline([
    ("scale", MinMaxScaler(feature_range=(-1, 1), clip=False)),
    ("classifier", KAHMClassifier(block_size=64)),
])
search = GridSearchCV(
    pipeline,
    {"classifier__subspace_dim": [4, 8, 16]},
    scoring="balanced_accuracy",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=1,
    error_score="raise",
)
# Fit on a development set; reserve another set for final evaluation.
# search.fit(X_development, y_development)
```

Standardization and min–max scaling change the KAHM geometry. Neither is a
universally correct default. Training-fitted min–max scaling with `clip=False`
allows future values outside the training range; clipping would define a
different transformation. The released comparison chooses between both scalers
on development data for every tuned method.

The core uses the inherited reconstruction and folding-score construction.
This adapter is not a new classification algorithm, and it does not add the
forecasting representations or risk certificates from separate manuscripts.

## Labels and normalized scores

`classes_` contains sorted original labels. `predict` returns labels with their
original dtype. Columns of `predict_proba` follow `classes_`.

For class folding distances `D_c`, the normalized scores are proportional to
`(1 - D_c) ** alpha`. A stable normalization is used; there is no top-k truncation
in the new adapter. These quantities are not certified or empirically calibrated
probabilities. Probability calibration, if needed, requires an appropriate
separate calibration protocol. The digits comparison does not score log loss
or claim calibrated confidence.

The affine weights used inside a local reconstruction are a different object
from the outer class scores. They sum to one where normalization is defined, but
individual weights can be negative.

## Numerical conventions

This release repairs avoidable numerical and computational problems in the
reference implementation. The changelog records the exact scope. Tests compare
ordinary inputs against the earlier mathematical formulas and cover the repaired
edge cases.

Stable Gaussian row scaling inside normalized reconstruction addresses all-zero
query kernels caused by exponent underflow. It does not establish robustness to
every form of signed-denominator cancellation or extreme input. Warnings and
explicit degenerate-case behavior are part of the implementation contract.
The row factor cancels only when the signed normalizer is nonzero. An exactly
zero normalizer emits a warning and retains the legacy divisor-of-one
convention; that result has no affine-normalization guarantee.

Constant local reference sets describe a single-point affine hull; their
reconstruction is that point. Invalid dimensions and block parameters are
reported explicitly rather than left to fail in a later matrix operation.

## Legacy API compatibility

Existing modules remain importable:

```python
from kahm_classification import train_kahm_classifier, kahm_classify
```

Those functions use `(n_features, n_samples)` arrays. The new estimator uses
`(n_samples, n_features)`. Transpose when crossing this boundary; do not transpose
inputs to `KAHMClassifier`.

The legacy API offers disk-backed models and other advanced options that the
small adapter does not expose. Existing regression and quantum-optimization
modules are packaged for compatibility, but they do not have the same
classification-example validation scope.

## Reproduction and limits

The recorded example has one dataset and one held-out split. It reports all
selected models, their full development searches, and test predictions. It does
not establish a universal ranking, evaluate privacy, or reproduce the papers'
classification or forecasting experiments.

For the exact protocol and the distinction between fit cost, prediction cost,
and statistical scope, see [examples/PROTOCOL.md](../examples/PROTOCOL.md).
