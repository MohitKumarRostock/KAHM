# Contributing

## Development environment

Use Python 3.10 or newer and a virtual environment:

```bash
python -m pip install -e ".[dev,examples]"
python -m pytest
python -m build
```

The `kahm` package is a small adapter around the reference top-level modules.
Preserve the legacy features-by-samples interfaces when changing those modules.
The public estimator uses samples-by-features arrays.

## Report a defect

Include the exact commit or version, Python and dependency versions, input shape
and dtype, parameter values, the complete error, and the smallest example that
reproduces it. Use synthetic or public data in an issue. Identify whether you are
using the estimator or a legacy function.

For numerical defects, include a comparison to the intended formula and a test
that exercises the failure. For estimator changes, test labels, serialization,
and pipeline/cross-validation behavior as relevant.

## Change an experiment

The published digits protocol is fixed. If you change its split, data, grids,
metrics, or methods after observing the results, give the experiment a new
protocol identifier and keep the original record. Do not overwrite the recorded
release results with a favorable rerun.

Run new experiments into an untracked directory:

```bash
python examples/classification.py --output-dir runs/digits
```

Report all declared methods and failed configurations. Distinguish a new fitting
seed from a new data source, and descriptive timings from hardware-independent
complexity claims. Keep preprocessing inside the training folds.

## Propose a change

Use a focused pull request describing the problem, change, resulting behavior,
and verification. New numerical conventions should be documented in the
changelog. Keep generated models and local environments out of commits.

Repository licensing is currently undecided; see
[LICENSE_STATUS.md](LICENSE_STATUS.md) before proposing distribution terms.

## Release workflow

Pull requests run the test suite on Linux (Python 3.10 and 3.12) and Windows
(Python 3.12). After those checks pass on `main`, the workflow reads the package
version. If that version has no GitHub release, it builds distributions, verifies
the wheel in a fresh environment outside the checkout, and publishes a research
prerelease with checksums. It does not publish to PyPI.

To prepare a later release, update the package version and `kahm.__version__`,
add matching notes in `docs/releases/vVERSION.md`, update citation metadata and
the changelog, and submit a pull request. Existing GitHub releases are not
overwritten. The `workflow_dispatch` action can retry publication on `main`
after a transient failure.
