# Digits classification: fixed example protocol

This protocol is specified before the first release benchmark is run. It is an
approachable classification example, not a claim of broad superiority or a
reproduction of a published KAHM experiment.

## Question and data

How does the repository's KAHM classifier compare with familiar classifiers on
the same small, multiclass dataset and under the same model-selection protocol?

Use all 1,797 observations and all 64 input features from scikit-learn's bundled
`load_digits()` dataset. The features are the original 8-by-8 image intensities;
there are no pretrained features, augmentations, downloads, or selected classes.
The dataset and its provenance are documented by
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

The example concerns digit classification. It is not an evaluation of privacy,
federated learning, uncertainty calibration, or a deployment population.

## Split and selection

1. With `random_state=42`, reserve a stratified 25% holdout: 450 test examples.
   The remaining 1,347 examples form the development set.
2. Create three shuffled stratified development folds with `random_state=42`.
   Reuse the identical fold indices for every method.
3. Fit preprocessing inside each training fold using a scikit-learn `Pipeline`.
   The validation and test examples never participate in fitting a scaler.
4. Select each tuned method's configuration by mean development-fold balanced
   accuracy. A tie is broken by the first candidate in the declared grid order.
5. Refit the selected pipeline on the complete development set.
6. Complete **all** development searches before evaluating any test predictions.
   Evaluate only the selected pipeline for each method on the holdout.

The script records the original dataset indices for the development/test split
and every validation fold, together with a dataset checksum. The seed, split,
grids, and metric are not chosen after inspecting test performance. No candidate
is selected on the test results. No alternative seed is tried to improve a score.

## Declared candidates

For every tuned method, the outer grid is the same ordered pair of scalers:

1. `StandardScaler()`;
2. `MinMaxScaler(feature_range=(-1, 1), clip=False)`.

The three model settings below are tried in the listed order within each scaler.
Each tuned method therefore has **six** candidate pipelines and **18** validation
fits, followed by one refit. Equal candidate counts do not imply equal effective
capacity, compute cost, or an exhaustive search of each method's possibilities.

| Method | Three model settings | Other fixed choices |
|---|---|---|
| KAHM | `subspace_dim = 4, 8, 16` | `block_size=64`, `alpha=10`, `random_state=42`, all development examples retained, in-memory models |
| Logistic regression | `C = 0.1, 1, 10` | L2 regularization, `solver="lbfgs"`, `max_iter=5000` |
| RBF SVM | `C = 1, 10, 100` | `kernel="rbf"`, `gamma="scale"`, no probability fitting |
| k-nearest neighbors | `n_neighbors = 1, 3, 5` | `weights="distance"`, Euclidean distance |

An untuned `DummyClassifier(strategy="most_frequent")` is a reference only. It
uses the same folds and development/test split, with one candidate. Its smaller
search effort is explicit in the report.

KAHM's `block_size` is a target block size, not a strict upper bound on local
reference-set size. The legacy construction partitions each class's training
observations into consecutive, approximately balanced blocks. This example does
not substitute a different clustering or reconstruction algorithm.

## Metrics and computational cost

The primary selection metric is **balanced accuracy**. Report holdout accuracy,
balanced accuracy, and macro F1 for every selected method. Do not use log loss,
calibration claims, or probability-quality conclusions: KAHM's normalized scores
are not automatically calibrated probabilities.

Record two wall-clock quantities separately:

- **Search plus refit:** the full `GridSearchCV.fit` call, including fold-local
  preprocessing, all declared configurations, selection, and the selected refit.
- **Test prediction:** one `predict` call on all 450 held-out examples, including
  the selected pipeline's preprocessing.

Data loading, package imports, plotting, and writing the report occur outside
these timers. Search uses `n_jobs=1`, and a `threadpoolctl` context limits numerical
library pools to one thread. KAHM fitting and inference are serial. Timings are
single measurements on the recorded environment, with no warm-up repetition or
selection of the fastest run. They are descriptive and hardware-dependent.

## Failures, artifacts, and interpretation

Failed model fits must not disappear from the record. The runner retains warning
messages, candidate-level validation scores, and failed-method status. If any
candidate fails or has a non-finite score, the report is explicitly marked
**incomplete**, and the command exits unsuccessfully after writing available
results. A successful result therefore requires all declared candidates to run.

The output directory contains:

- `results.json`: protocol, environment, source/dataset hashes, selected settings,
  metrics, timing, warnings, and completion status;
- `cv_results.csv`: every candidate's three validation scores and timing;
- `predictions.csv`: original held-out index, true label, and each final model's
  prediction;
- `splits.json`: development/test and cross-validation indices;
- `report.md`: an automatically generated readable result summary;
- `comparison.png` / `comparison.svg`: holdout performance and measured cost;
- `kahm_confusion.png` / `kahm_confusion.svg`: KAHM holdout confusion matrix.

The held-out images come from one dataset and one split. This example provides no
population confidence interval, significance test, independent replication, or
evidence across domains. Its purpose is to make the implementation inspectable
and straightforward to compare, including when a baseline performs better.

Protocol identifier: `digits-classification-v1`.
