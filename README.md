# KAHM · Kernel Affine Hull Machines

**Local geometric models for classification and prediction.**

KAHMs reconstruct an input from class-specific reference observations using
kernel-based affine combinations. A classifier compares the resulting
reconstruction discrepancies across classes. This repository provides the
reference Python implementation and a small scikit-learn classification adapter.

**Start with [the classification example](examples/README.md):** handwritten
digits, four tuned methods, identical validation folds, and a reserved test set.
The example reports every method, including when a baseline performs better.

| Start here | What you get |
| --- | --- |
| [Installation and quickstart](#installation) | A normal Python environment and a row-oriented estimator |
| [Classification example](examples/README.md) | KAHM, logistic regression, RBF SVM, kNN, and a dummy reference |
| [Recorded results](examples/results/digits/report.md) | Measured scores, costs, selected settings, and limitations |
| [Evaluation protocol](examples/PROTOCOL.md) | The split, grids, and metrics specified before the first run |
| [Classifier guide](docs/CLASSIFICATION.md) | Parameters, numerical conventions, and legacy compatibility |
| [Research and citation](#research-and-citation) | Original method and related implementations |

## Installation

Python **3.10 or newer** is required. Install from a checkout:

```bash
git clone https://github.com/MohitKumarRostock/KAHM.git
cd KAHM
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install --upgrade pip
python -m pip install ".[examples]"
```

For a small installation without plotting dependencies, use
`python -m pip install .`. A built wheel can also be installed directly when
available in [Releases](https://github.com/MohitKumarRostock/KAHM/releases).
This release does **not** require or assume a package published on PyPI.

## Quickstart

The new estimator uses the standard **samples × features** convention. Put any
learned scaling inside a pipeline:

```python
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from kahm import KAHMClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    MinMaxScaler(feature_range=(-1, 1), clip=False),
    KAHMClassifier(subspace_dim=8, block_size=64),
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, prediction):.3f}")
```

These are illustrative fixed parameters. The comparison example selects
parameters using development folds and evaluates the reserved test set only
after all searches are complete:

```bash
python examples/classification.py --output-dir runs/digits
```

It writes a report, plots, every candidate's validation scores, split indices,
held-out predictions, and environment details. See
[the example guide](examples/README.md) for the exact outputs.

## How the classifier works

1. Divide each class's reference observations into approximately balanced blocks.
2. Fit the inherited KAHM local reconstructor for each block.
3. Score a query with its reconstruction discrepancy and angular discrepancy.
4. Take the best local match within each class, then choose the closest class.

The current block construction uses consecutive observations within each class;
it is not k-means. Input ordering, scaling, local dimension, and the reference
budget can affect results. Kernel reconstruction weights are affine weights and
can be negative.

`predict_proba` exposes normalized reconstruction-affinity scores for API
compatibility. Their normalization does **not** establish probability calibration.
The example evaluates class predictions, without probability-quality claims.

## Scope of this release

- `from kahm import KAHMClassifier`: tested classification adapter with in-memory
  fitting, label preservation, cloning, pipelines, and cross-validation support.
- Existing top-level functions such as `train_kahm_classifier`, `kahm_classify`,
  and the regression modules remain available with their original
  **features × samples** convention.
- The regression and optional quantum-optimization scripts are legacy research
  interfaces; they are not the focus of this classification release.
- The digits example is one dataset and one split. It is an installation and
  comparison example, not evidence of universal superiority, privacy, or
  deployment readiness.

See [the changelog](CHANGELOG.md) for the numerical repairs in this version and
[the classifier guide](docs/CLASSIFICATION.md) for limitations.

## Research and citation

The method originates in:

**Kumar, M., Moser, B. A., and Fischer, L. (2024).**
*On Mitigating the Utility-Loss in Differentially Private Learning: A New
Perspective by a Geometrically Inspired Kernel Approach.* Journal of Artificial
Intelligence Research, 79, 515–567.
[Paper](https://jair.org/index.php/jair/article/view/15071) ·
[DOI](https://doi.org/10.1613/jair.1.15071)

The collaborative-learning development is:

**Kumar, M., et al. (2025).**
*Geometrically Inspired Kernel Machines for Collaborative Learning Beyond
Gradient Descent.* Journal of Artificial Intelligence Research, 83, Article 16.
[Paper](https://jair.org/index.php/jair/article/view/16821) ·
[DOI](https://doi.org/10.1613/jair.1.16821)

| Repository | Relationship |
| --- | --- |
| **This repository: KAHM** | Python reference components and the installable classification entry point |
| [GIKM](https://github.com/software-competence-center-hagenberg/GIKM) | Institutional MATLAB/Python implementations and application experiments |
| [gradient-free-fl](https://github.com/MohitKumarRostock/gradient-free-fl) | Related collaborative-learning Python project |
| [Kernel Affine Hull Koopman Machine](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine) | Dynamical-system experiments and reproduction records |

Use [CITATION.cff](CITATION.cff) for the original-method citation. When reporting
software results, also record this repository's version and exact commit. The
classification example introduced here is separate from the published papers'
experimental results.

## Development

```bash
python -m pip install -e ".[dev,examples]"
python -m pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducible bug reports and evaluation
changes. Questions and defects belong in
[GitHub Issues](https://github.com/MohitKumarRostock/KAHM/issues).

## License status

A repository-wide software license has not yet been declared. See
[LICENSE_STATUS.md](LICENSE_STATUS.md). Dataset attribution and dependency
licenses are separate; see the example guide.
