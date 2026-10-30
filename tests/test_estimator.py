"""Behavioral checks for the public estimator and packaged legacy API."""

from pathlib import Path

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kahm import KAHMClassifier
from kahm_classification import kahm_classify, train_kahm_classifier


@pytest.fixture
def classification_data():
    X, codes = make_blobs(n_samples=72, n_features=4, centers=3, cluster_std=0.7, random_state=17)
    labels = np.asarray(["zebra", "ant", "yak"])[codes]
    # Make first-seen class order deliberately different from sorted order.
    order = np.argsort(codes, kind="stable")
    return X[order], labels[order]


def test_adapter_matches_legacy_predictions_and_score_columns(classification_data):
    X, y = classification_data
    adapter = KAHMClassifier(subspace_dim=2, block_size=100, alpha=3.0).fit(X, y)
    legacy = train_kahm_classifier(
        X.T, y, subspace_dim=2, Nb=100, save_ae_to_disk=False, verbose=False, model_dtype="float64"
    )
    expected_labels, legacy_scores = kahm_classify(
        legacy, X.T, n_jobs=1, mode="soft", return_probabilities=True,
        alpha=3.0, topk=0, show_progress=False,
    )
    assert_array_equal(adapter.classes_, np.unique(y))
    assert_array_equal(adapter.predict(X), expected_labels)
    column_order = [list(legacy["classes_"]).index(label) for label in adapter.classes_]
    assert_allclose(adapter.predict_proba(X), legacy_scores[column_order].T)
    assert adapter.predict(X).dtype == y.dtype


def test_clone_pipeline_and_cross_validation(classification_data):
    X, y = classification_data
    estimator = KAHMClassifier(subspace_dim=2, block_size=100, random_state=23)
    assert clone(estimator).get_params() == estimator.get_params()
    pipeline = make_pipeline(StandardScaler(), estimator)
    search = GridSearchCV(
        pipeline,
        {"kahmclassifier__subspace_dim": [1, 2]},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=11),
        error_score="raise",
    ).fit(X, y)
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert search.score(X, y) > 0.85
    assert_array_equal(estimator.get_params()["subspace_dim"], 2)
    assert not hasattr(estimator, "model_")


def test_in_memory_fit_and_serialization(classification_data, tmp_path, monkeypatch):
    X, y = classification_data
    monkeypatch.chdir(tmp_path)
    estimator = KAHMClassifier(subspace_dim=2).fit(X, y)
    assert list(Path.cwd().iterdir()) == []
    path = tmp_path / "classifier.joblib"
    joblib.dump(estimator, path)
    reloaded = joblib.load(path)
    assert_array_equal(reloaded.predict(X), estimator.predict(X))
    assert_allclose(reloaded.predict_proba(X), estimator.predict_proba(X))


def test_numeric_labels_and_batched_scores(classification_data):
    X, text_labels = classification_data
    labels = np.asarray([{"ant": -7, "yak": 11, "zebra": 34}[label] for label in text_labels])
    estimator = KAHMClassifier(subspace_dim=2, batch_size=5).fit(X, labels)
    scores = estimator.predict_proba(X)
    assert scores.shape == (len(X), 3)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)
    assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)
    assert_array_equal(estimator.predict(X), estimator.classes_[scores.argmax(axis=1)])
    assert estimator.predict(X).dtype == labels.dtype
    estimator.set_params(batch_size=None)
    assert_allclose(scores, estimator.predict_proba(X), atol=1e-7)


def test_subsampling_is_reproducible(classification_data):
    X, y = classification_data
    parameters = dict(subspace_dim=2, block_size=100, max_train_per_class=12, random_state=29)
    first = KAHMClassifier(**parameters).fit(X, y)
    second = KAHMClassifier(**parameters).fit(X, y)
    assert_allclose(first.predict_proba(X), second.predict_proba(X))
    assert all(bank[0]["y_data"].shape[1] == 12 for bank in first.model_["classifier"])


@pytest.mark.parametrize(
    "parameters",
    [
        {"subspace_dim": 0}, {"subspace_dim": 1.5}, {"block_size": 1},
        {"alpha": 0}, {"alpha": float("nan")}, {"max_train_per_class": 1},
        {"random_state": -1}, {"batch_size": 0},
    ],
)
def test_invalid_parameters_fail_early(parameters, classification_data):
    X, y = classification_data
    with pytest.raises(ValueError):
        KAHMClassifier(**parameters).fit(X, y)


def test_input_validation_and_unfitted_state(classification_data):
    X, y = classification_data
    with pytest.raises(NotFittedError):
        KAHMClassifier().predict(X)
    with pytest.raises(ValueError, match="two classes"):
        KAHMClassifier().fit(X, np.zeros(len(y), dtype=int))
    bad_y = y.copy()
    bad_y[0] = "singleton"
    with pytest.raises(ValueError, match="two samples"):
        KAHMClassifier().fit(X, bad_y)
    bad_X = X.copy()
    bad_X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        KAHMClassifier().fit(bad_X, y)
    with pytest.raises(ValueError, match="Unknown label type"):
        KAHMClassifier().fit(X, np.linspace(0, 1, len(y)))
    estimator = KAHMClassifier(subspace_dim=2).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        estimator.predict(X[:, :-1])
