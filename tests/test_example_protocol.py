"""Check the shared split and search budget without using holdout outcomes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "classification.py"
SPEC = importlib.util.spec_from_file_location("classification_example", EXAMPLE_PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_split_is_disjoint_reproducible_and_covers_every_example():
    _, y = load_digits(return_X_y=True)
    development, test, folds = example.make_splits(y)
    development2, test2, folds2 = example.make_splits(y)
    np.testing.assert_array_equal(development, development2)
    np.testing.assert_array_equal(test, test2)
    assert (len(development), len(test)) == (1347, 450)
    assert set(development).isdisjoint(test)
    assert set(development) | set(test) == set(range(len(y)))
    seen_validation = []
    for (train, validation), (train2, validation2) in zip(folds, folds2):
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(validation, validation2)
        assert set(train).isdisjoint(validation)
        assert set(train) | set(validation) == set(range(len(development)))
        assert set(development[train]).isdisjoint(test)
        assert set(development[validation]).isdisjoint(test)
        seen_validation.extend(validation)
    assert sorted(seen_validation) == list(range(len(development)))


def test_all_tuned_methods_have_same_six_candidate_preprocessing_budget():
    specs = example.model_specs()
    assert [name for name, _, _ in specs[:4]] == list(example.TUNED_METHODS)
    for name, pipeline, grid in specs:
        assert "scale" in pipeline.named_steps
        if name in example.TUNED_METHODS:
            candidates = list(ParameterGrid(grid))
            assert len(candidates) == 6
            assert sum(isinstance(candidate["scale"], StandardScaler) for candidate in candidates) == 3
            assert sum(isinstance(candidate["scale"], MinMaxScaler) for candidate in candidates) == 3
            for candidate in candidates:
                scaler = candidate["scale"]
                assert not hasattr(scaler, "n_features_in_")
                if isinstance(scaler, MinMaxScaler):
                    assert scaler.feature_range == (-1, 1)
                    assert scaler.clip is False
        else:
            assert name == "Dummy (most frequent)"
            assert len(ParameterGrid(grid)) == 1


def test_nonempty_output_directory_is_preserved(tmp_path):
    previous = tmp_path / "report.md"
    previous.write_text("Previous experiment result", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty directory"):
        example.run(tmp_path)
    assert previous.read_text(encoding="utf-8") == "Previous experiment result"
    assert list(tmp_path.iterdir()) == [previous]
