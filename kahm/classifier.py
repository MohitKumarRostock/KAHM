"""A scikit-learn adapter around the reference KAHM classifier."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from kahm_classification import kahm_classify, train_kahm_classifier


class KAHMClassifier(ClassifierMixin, BaseEstimator):
    """Classify samples using per-class kernel affine hull reconstructions.

    This adapter delegates learning and scoring to the existing implementation.
    It changes the public array convention to ``(n_samples, n_features)`` and
    stores its fitted reference banks in memory. It does not standardize, scale,
    or otherwise preprocess inputs. Put training-fitted preprocessing in a
    scikit-learn ``Pipeline`` when it is needed.

    Parameters
    ----------
    subspace_dim : int, default=10
        Requested local principal-subspace dimension. The reference learner
        reduces it when the data have lower effective rank.
    block_size : int, default=100
        Reference-block size parameter, named ``Nb`` in the legacy API.
        The learner partitions each class into approximately
        ``round(n_class_samples / block_size)`` consecutive blocks. This is
        neither a hard upper bound on block size nor a learned clustering rule.
        Sample order within each class can therefore affect a multi-block fit.
    alpha : float, default=10.0
        Positive exponent for the normalized scores returned by
        ``predict_proba``. It does not tune the reconstruction models or the
        nearest-reconstruction classification rule. These scores are not
        guaranteed to be calibrated probabilities.
    max_train_per_class : int or None, default=None
        Optional reference-sample cap per class, applied by the legacy learner.
        At least two reference observations per class are required.
    random_state : int or None, default=0
        Seed for reference subsampling. The underlying learner interprets
        ``None`` as zero. Without subsampling this parameter has no effect.
    batch_size : int or None, default=None
        Optional number of query samples per scoring batch.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels; probability columns follow this order.
    n_features_in_ : int
        Number of features seen during fitting.
    model_ : dict
        Fitted legacy model with class banks ordered by ``classes_``. Its label
        values are integer indices into ``classes_``.

    Notes
    -----
    All classes need at least two observations. Dense, finite real-valued inputs
    are required; sparse arrays and missing values are not supported.
    ``predict`` uses the class with the smallest folding distance. As in the
    reference implementation, the local learner stores reference observations
    with float32 precision. The model is intended for trusted-data workflows;
    only load serialized Python models from trusted sources.
    """

    def __init__(
        self,
        subspace_dim=10,
        block_size=100,
        alpha=10.0,
        max_train_per_class=None,
        random_state=0,
        batch_size=None,
    ):
        self.subspace_dim = subspace_dim
        self.block_size = block_size
        self.alpha = alpha
        self.max_train_per_class = max_train_per_class
        self.random_state = random_state
        self.batch_size = batch_size

    def _check_parameters(self):
        for name, minimum in (("subspace_dim", 1), ("block_size", 2)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name, minimum in (("max_train_per_class", 2), ("batch_size", 1)):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be None or an integer >= {minimum}.")
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("alpha must be a finite positive number.")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError("random_state must be None or an integer in [0, 2**32).")

    def fit(self, X, y):
        """Fit class reference models on dense row-oriented samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training observations. No internal scaling is performed.
        y : array-like of shape (n_samples,)
            Discrete class labels, with at least two observations per class.

        Returns
        -------
        self : KAHMClassifier
            The fitted classifier.
        """
        self._check_parameters()
        X, y = validate_data(self, X, y, dtype=np.float64, accept_sparse=False)
        check_classification_targets(y)
        classes, encoded_y = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("KAHMClassifier requires at least two classes.")
        counts = np.bincount(encoded_y)
        if np.any(counts < 2):
            raise ValueError("KAHMClassifier requires at least two samples in every class.")

        model = train_kahm_classifier(
            X.T,
            encoded_y,
            subspace_dim=int(self.subspace_dim),
            Nb=int(self.block_size),
            random_state=None if self.random_state is None else int(self.random_state),
            input_scale=1.0,
            max_train_per_class=self.max_train_per_class,
            model_dtype="float64",
            save_ae_to_disk=False,
            verbose=False,
        )
        # Legacy training preserves first-seen class order. Reorder the fitted
        # banks, without refitting or reordering observations inside each bank,
        # so labels and probability columns follow the sklearn convention.
        order = np.argsort(np.asarray(model["classes_"], dtype=np.int64))
        model["classifier"] = [model["classifier"][int(i)] for i in order]
        model["classes_"] = np.arange(len(classes), dtype=np.int64)
        self.classes_ = classes
        self.model_ = model
        return self

    def _validate_query(self, X):
        check_is_fitted(self, attributes=["model_", "classes_"])
        self._check_parameters()
        return validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)

    def predict(self, X):
        """Return original class labels for row-oriented query samples."""
        X = self._validate_query(X)
        indices = kahm_classify(
            self.model_,
            X.T,
            n_jobs=1,
            mode="hard",
            batch_size=self.batch_size,
            show_progress=False,
        )
        return self.classes_[np.asarray(indices, dtype=np.int64)]

    def predict_proba(self, X):
        """Return normalized reconstruction-affinity scores.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
            Nonnegative row-normalized scores in ``classes_`` order, using
            ``(1 - folding_distance) ** alpha`` and no top-k truncation. They
            are not guaranteed to be calibrated probabilities. Calibration,
            if required, needs separate training/validation data.
        """
        X = self._validate_query(X)
        _, probabilities = kahm_classify(
            self.model_,
            X.T,
            n_jobs=1,
            mode="soft",
            return_probabilities=True,
            alpha=float(self.alpha),
            topk=0,
            batch_size=self.batch_size,
            show_progress=False,
        )
        return np.asarray(probabilities).T
