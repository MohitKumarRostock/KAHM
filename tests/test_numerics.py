"""Regression checks for the numerical and performance repairs.

These tests compare independent formula evaluations, saved pre-repair outputs,
and boundary cases that failed in the original implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import kahm_classification as classification
from autoencoder import autoencoder
from autoencoder_filtering_extended import autoencoder_filtering_extended
from dim_reduce import dim_reduce
from distance_matrix import distance_matrix
from kxa_matrix import kxa_matrix
from kxx_matrix import kxx_matrix
from parallel_autoencoders import parallel_autoencoders


def test_training_kernel_matches_original_pair_equation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 23))
    factor = rng.normal(size=(4, 4))
    precision = factor.T @ factor
    expected = np.ones((23, 23))
    for i in range(23):
        for j in range(23):
            delta = x[:, i] - x[:, j]
            expected[i, j] = np.exp(-0.5 / 4 * (delta @ precision @ delta))
    actual = kxx_matrix(x, precision)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
    assert_allclose(actual, actual.T, rtol=2e-13, atol=2e-15)
    assert_array_equal(np.diag(actual), np.ones(23))


def test_ordinary_reconstruction_preserves_pre_repair_outputs():
    # Outputs recorded from upstream commit 33f008c301c7cfca2d20e017b54383d822b5cf58.
    rng = np.random.default_rng(73)
    references = rng.normal(size=(3, 8))
    queries = rng.normal(size=(3, 4))
    expected_reconstruction = np.array([
        [0.2156692025975190, -0.2643372477368292, 0.1878919560666976, 0.0173086663664811],
        [0.7451530371253718, -0.2272934976845609, 0.0410341427285848, 0.4173556255412748],
        [-0.6947655830433616, 0.0690789425067747, -0.6464198510545035, -0.4536451106242128],
    ])
    expected_distances = [0.4109534809993184, 0.5112166499552261, 0.2977298549515095, 0.3233544588722209]
    actual, distances = autoencoder_filtering_extended(queries, autoencoder(references, 2), "folding")
    assert_allclose(actual, expected_reconstruction, rtol=1e-10, atol=1e-12)
    assert_allclose(distances, expected_distances, rtol=1e-10, atol=1e-12)


def test_row_shift_preserves_normalized_weights_but_not_kernel_api():
    rng = np.random.default_rng(4)
    references = rng.normal(size=(2, 15))
    bank = autoencoder(references, 2)
    queries = rng.normal(size=(2, 9))
    projected = bank["PC"].T @ queries
    raw = kxa_matrix(projected, bank["y_data_n_subspace"], bank["sqrt_weights_matrix"])
    shifted = kxa_matrix(projected, bank["y_data_n_subspace"], bank["sqrt_weights_matrix"], row_shift=True)
    assert_allclose(shifted.max(axis=1), 1.0)
    assert not np.allclose(raw, shifted)
    raw_weights = raw @ bank["B"]
    shifted_weights = shifted @ bank["B"]
    assert_allclose(
        raw_weights / raw_weights.sum(axis=1, keepdims=True),
        shifted_weights / shifted_weights.sum(axis=1, keepdims=True),
        rtol=1e-12, atol=1e-12,
    )


def test_normalized_query_survives_gaussian_underflow():
    bank = autoencoder(np.array([[1.0, 2.0, 3.0, 4.0]]), 1)
    query = np.array([[1000.0]])
    raw = kxa_matrix(bank["PC"].T @ query, bank["y_data_n_subspace"], bank["sqrt_weights_matrix"])
    assert not raw.any()  # The unchanged raw kernel legitimately underflows.
    reconstruction, distance = autoencoder_filtering_extended(query, bank, "folding")
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(distance).all()
    assert np.linalg.norm(reconstruction) > 0


def test_cancellation_is_reported_and_exact_zero_convention_is_explicit():
    bank = {
        "PC": np.ones((1, 1)),
        "y_data_n_subspace": np.array([[-1.0, 1.0]]),
        "sqrt_weights_matrix": np.ones((1, 1)),
        "kerneltype": "Gaussian",
        "B": np.array([[1.0, -1.0], [0.0, 0.0]]),
        "y_data": np.array([[1.0, 2.0]]),
    }
    with pytest.warns(RuntimeWarning, match="signed KAHM weight normalizer"):
        reconstruction, distance = autoencoder_filtering_extended(np.zeros((1, 1)), bank, "folding")
    assert_allclose(reconstruction, [[-1.0]])
    assert np.isfinite(distance).all()


def test_constant_reference_bank_reconstructs_its_point():
    references = np.tile([[1.5], [-2.0]], (1, 4))
    bank = autoencoder(references, 2)
    queries = np.array([[0.0, 10.0, 1.5], [0.0, -10.0, -2.0]])
    actual, distance = autoencoder_filtering_extended(queries, bank, "folding")
    assert_allclose(actual, np.tile([[1.5], [-2.0]], (1, 3)))
    assert np.isfinite(distance).all()
    assert distance[-1] == pytest.approx(0.0, abs=1e-8)


def test_block_partition_does_not_create_singleton_banks():
    references = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
    banks = parallel_autoencoders(references, subspace_dim=1, Nb=2)
    assert len(banks) == 1
    assert banks[0]["y_data"].shape[1] == 3
    with pytest.raises(ValueError, match="at least two"):
        parallel_autoencoders(references, subspace_dim=1, Nb=1)


def test_reconstruction_chunk_boundary_matches_individual_calls():
    rng = np.random.default_rng(3)
    bank = autoencoder(rng.normal(size=(3, 12)), 2)
    queries = rng.normal(size=(3, 1003))
    reconstruction, distance = autoencoder_filtering_extended(queries, bank, "folding")
    left = autoencoder_filtering_extended(queries[:, :501], bank, "folding")
    right = autoencoder_filtering_extended(queries[:, 501:], bank, "folding")
    assert_allclose(reconstruction, np.concatenate([left[0], right[0]], axis=1), rtol=1e-11, atol=1e-12)
    assert_allclose(distance, np.concatenate([left[1], right[1]]), rtol=1e-11, atol=1e-12)


def test_squared_distance_is_nonnegative_and_matches_direct_differences():
    rng = np.random.default_rng(22)
    a = rng.normal(size=(9, 5))
    actual = distance_matrix(a, a)
    expected = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=2)
    assert (actual >= 0).all()
    assert_allclose(actual, expected, atol=1e-13)


def test_soft_scores_keep_tiny_affinity_ratios_and_handle_all_zero():
    distances = np.array([[0.99, 1.0], [0.999, 1.0]])
    actual = classification._normalized_affinity_scores(distances, 100.0, None)
    expected_ratio = ((1 - distances[1, 0]) / (1 - distances[0, 0])) ** 100
    assert actual[0, 0] == pytest.approx(1.0)
    assert actual[1, 0] == pytest.approx(expected_ratio, rel=1e-11, abs=0)
    assert_allclose(actual[:, 1], [0.5, 0.5])
    assert_allclose(actual.sum(axis=0), 1.0)


def test_soft_prediction_evaluates_each_class_once_and_tuning_agrees(monkeypatch):
    model = {"classifier": [{"index": 0}, {"index": 1}], "classes_": np.array([0, 1]), "n_classes": 2}
    distances = np.array([[0.99, 0.999, 1.0], [0.999, 0.99, 1.0]], dtype=np.float32)
    calls = []

    def evaluate(x, banks, distance_type, **kwargs):
        index = banks[0]["index"]
        calls.append(index)
        return distances[index]

    monkeypatch.setattr(classification, "_call_combine_multiple_autoencoders_extended", evaluate)
    labels, probabilities = classification.kahm_classify(
        model, np.zeros((1, 3)), mode="soft", return_probabilities=True,
        alpha=100.0, topk=0, show_progress=False,
    )
    assert calls == [0, 1]
    assert_array_equal(labels, [0, 1, 0])
    assert_array_equal(probabilities.argmax(axis=0), [0, 1, 0])
    result = classification.tune_soft_params_ce(
        model, np.zeros((1, 3)), np.array([0, 1, 0]),
        alphas=[100.0], topks=[None], verbose=False, show_progress=False,
    )
    assert result.best_cross_entropy == pytest.approx(np.log(2.0) / 3, abs=1e-12)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_projection_dimension_is_rejected(n):
    with pytest.raises(ValueError, match="positive integer"):
        dim_reduce(np.array([[1.0, 2.0]]), n)


def test_nonfinite_training_references_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        autoencoder(np.array([[1.0, np.nan]]), 1)
