# A reproducible classification example

The Digits example compares KAHM with logistic regression, an RBF SVM,
distance-weighted nearest neighbors, and a dummy reference. It uses the small
dataset bundled with scikit-learn, so it needs no dataset download, GPU, or
pretrained model.

Dataset provenance and terms are documented by
[scikit-learn's Digits dataset page](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
and the original
[UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
The dataset is credited to E. Alpaydin and C. Kaynak. This project does not
relicense third-party datasets; consult their original terms.

## Run it

From the repository root, install the package and example dependencies:

```bash
python -m pip install -e ".[examples]"
python examples/classification.py --output-dir artifacts/digits
```

Open `artifacts/digits/report.md` for the result table, selected parameters, and
scope of the conclusions. Open `comparison.svg` for the performance/cost figure
and `kahm_confusion.svg` for the held-out confusion matrix. PNG copies are also
written for convenience.

The output directory must be new or empty. To repeat the example, choose a fresh
`--output-dir`; the runner preserves previous results instead of overwriting them.

The script prints progress during model selection. A complete run must finish
all 25 declared candidates (six per tuned model and one dummy), with three
development folds, before evaluating the selected models on 450 held-out images.
There is no faster variant that silently changes the published protocol.

## What makes the comparison useful

- The test split, validation folds, metric, and parameter grids are declared in
  [PROTOCOL.md](PROTOCOL.md).
- Every tuned method sees exactly the same folds and receives six candidate
  pipelines, including the same two preprocessing options.
- Scalers are learned inside each fold. The test images never fit a scaler or
  select a model.
- Every candidate's validation scores and every final prediction are retained.
- Fitting/model-selection time and test-prediction time are measured separately.
- Failed fits and warnings remain visible. An incomplete evaluation exits with a
  nonzero status after writing the available evidence.

Equal search counts do not imply equal computational budgets or equivalent model
classes. This is a tutorial on one dataset and split. A baseline may perform
better than KAHM, and the report presents that outcome directly.

## Inspect the evidence

| File | Contents |
|---|---|
| `results.json` | Selected settings, holdout metrics, timing, warnings, environment, source and data checksums |
| `cv_results.csv` | All candidate parameters and validation-fold results |
| `predictions.csv` | Original test-image indices, true labels, and the selected models' predictions |
| `splits.json` | Original dataset indices for every train/validation/test partition |
| `report.md` | Generated human-readable interpretation and links to the figures |

`results.json` includes the exact SHA-256 of the protocol used for that run.
Timings will vary by machine; class-label predictions can also vary across
numerical-library versions. The report records the environment rather than
assuming bitwise portability.
