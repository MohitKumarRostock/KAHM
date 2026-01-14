"""
kahm_classification.py

KAHM-based multiclass classification using per-class autoencoders (AEs).

Core idea
---------
- Train one (or a small list of) OTFL autoencoder(s) per class using inputs from that class.
- At inference time, evaluate each class-AE on a new sample to obtain a distance value.
  Distances are assumed to lie in [0, 1] and be monotone with (1 - probability).

Soft probability mapping (recommended for calibrated probabilities)
------------------------------------------------------------------
Given distances D of shape (C, N):

    S = (1 - D) ** alpha
    keep only top-k scores per sample (optional)
    P = S / sum_c S

Prediction:
    y_hat = argmax_c P[c, :]

Important note:
- For strictly monotone transforms (alpha > 0), argmax P is the same as argmin D.
  Therefore alpha/topk primarily affect probability calibration and cross-entropy,
  not the predicted class (except for degenerate ties).

Autotuning support
------------------
- tune_soft_params_ce(...) evaluates a grid over (alpha, topk) on a validation set,
  stores best values in the model dict as:
      model['soft_alpha'], model['soft_topk']
  using multiclass cross-entropy (negative log-likelihood).

Model dict keys (stable)
------------------------
- classifier: list of per-class AEs (or joblib paths)
- classifier_dir: base directory for joblib AE paths (optional)
- model_id: optional model identifier (useful to avoid cache collisions)
- classes_: array/list of original labels in model's class-index order
- n_classes: number of classes
- input_scale: scalar used to scale X at train & infer (for OTFL gating)
- soft_alpha / soft_topk: tuned soft parameters (optional)

Requires:
    numpy, joblib
    and your OTFL helpers:
      - parallel_autoencoders
      - combine_multiple_autoencoders_extended
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Literal, overload
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike
from joblib import dump, load

from parallel_autoencoders import parallel_autoencoders
from combine_multiple_autoencoders_extended import combine_multiple_autoencoders_extended

import gc as _gc
import os
import tempfile


# ----------------------------
# Public result types
# ----------------------------

@dataclass(frozen=True)
class SoftTuningResultCE:
    """Return type for tune_soft_params_ce."""

    best_alpha: float
    best_topk: int | None
    best_cross_entropy: float
    best_accuracy: float


# ----------------------------
# Precision helpers
# ----------------------------

def _as_float_ndarray(x: Any, *, min_dtype: DTypeLike = np.float32) -> np.ndarray:
    """Convert input to a floating ndarray without downcasting precision.

    - Integer/bool inputs are promoted to float64.
    - Floating inputs keep their existing precision unless below `min_dtype`,
      in which case they are promoted to `min_dtype`.
    """
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64, copy=False)
    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)


def _scale_like(X: np.ndarray, scale: float, *, inplace: bool = False) -> np.ndarray:
    """Scale X by `scale` without unintentionally upcasting dtype."""
    if scale == 1.0:
        return X
    if X.dtype.kind not in "fc":
        X = X.astype(np.float32, copy=False)
    s = np.asarray(scale, dtype=X.dtype)
    if inplace and X.flags.writeable:
        np.multiply(X, s, out=X, casting="unsafe")
        return X
    return (X * s).astype(X.dtype, copy=False)


def _ae_as_list(ae: Any) -> list:
    """Ensure an autoencoder spec is passed as a flat list.

    The OTFL helper `combine_multiple_autoencoders_extended` expects a sequence of
    autoencoder components. Some training pipelines store:
      - a list/tuple of components, or
      - a single component (dict/ndarray-like).

    This normalizes both cases and also flattens a common accidental nesting:
        [ [component, ...] ].
    """
    if isinstance(ae, (list, tuple)):
        if len(ae) == 1 and isinstance(ae[0], (list, tuple)):
            return list(ae[0])
        return list(ae)
    return [ae]


def _call_combine_multiple_autoencoders_extended(
    X: np.ndarray,
    AE_list: Sequence[Any],
    distance_type: str,
    *,
    n_jobs: int | None = None,
):
    """Call `combine_multiple_autoencoders_extended` with best-effort support for `n_jobs`.

    The upstream OTFL helper signature can vary by version. We attempt to pass `n_jobs`
    if available; otherwise we fall back to the 3-argument call.
    """
    fn: Any = combine_multiple_autoencoders_extended
    try:
        return fn(X, AE_list, distance_type, n_jobs=n_jobs)
    except TypeError:
        return fn(X, AE_list, distance_type)


# ----------------------------
# Soft probability mapping helpers
# ----------------------------

def _topk_truncate_inplace(S: np.ndarray, k: int, *, chunk_cols: int = 64) -> None:
    """Zero all but the top-k entries per column of S in-place (memory-aware)."""
    if S.ndim != 2:
        raise ValueError("S must be 2D shaped (C, N).")
    C, N = S.shape
    k = int(k)
    if k <= 0 or k >= C or N == 0:
        return

    # Cap temporary index memory (~32 MiB by default).
    max_tmp_bytes = 32 * 1024 * 1024
    # indices are int64, values float64/float32 (worst-case 16 bytes per entry for argsort temp-ish)
    max_cols = max(1, min(int(chunk_cols), int(max_tmp_bytes // (max(1, C) * 16))))
    for s in range(0, N, max_cols):
        e = min(N, s + max_cols)
        sub = S[:, s:e]
        # Keep top-k per column
        idx = np.argpartition(sub, kth=C - k, axis=0)[: C - k, :]
        # Zero everything except top-k => zero the smallest (C-k)
        sub[idx, np.arange(e - s)] = 0.0


def _get_soft_params_from_model(
    model: dict,
    alpha: Optional[float],
    topk: Optional[int | None],
) -> Tuple[float, int | None]:
    """Resolve soft parameters.

    Resolution order:
      - If an explicit argument is provided, use it.
      - Otherwise, use the value stored in the model (if present).
      - Otherwise, fall back to sensible defaults.

    Semantics:
      - `topk=None` means "use the model setting" if present.
      - To *disable* top-k truncation, either:
          * set `model['soft_topk'] = None`, or
          * pass `topk=0` (or any non-positive value).
    """
    if alpha is None:
        alpha = model.get("soft_alpha", None)

    topk_from_model = False
    if topk is None:
        if "soft_topk" in model:
            topk = model.get("soft_topk", None)
            topk_from_model = True

    alpha_resolved = float(alpha) if alpha is not None else 10.0

    if topk is None and not topk_from_model:
        topk_resolved: int | None = 10
    else:
        topk_resolved = topk

    return alpha_resolved, topk_resolved


# ----------------------------
# Label helpers
# ----------------------------

def _stable_unique_labels(y: np.ndarray) -> list:
    """Stable unique labels preserving first occurrence order."""
    seen = {}
    out = []
    for v in y.tolist() if isinstance(y, np.ndarray) else list(y):
        if v not in seen:
            seen[v] = True
            out.append(v)
    return out


def _map_labels_to_indices(model: dict, y: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to 0..C-1 indices using model['classes_']."""
    classes = model.get("classes_", None)
    if classes is None:
        raise KeyError("model must contain 'classes_' to map labels.")
    # Normalize classes to a Python list for robust hashing
    if isinstance(classes, np.ndarray):
        classes_list = classes.tolist()
    else:
        classes_list = list(classes)

    lookup = {lab: i for i, lab in enumerate(classes_list)}
    y_arr = np.asarray(y).reshape(-1)
    try:
        return np.array([lookup[v] for v in y_arr.tolist()], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Found a label not present in model['classes_']: {e}") from e


def _normalize_y_input(y: Any, N: int) -> np.ndarray:
    """Normalize y to a 1D array of length N (supports (N,), (1,N), (N,1))."""
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and 1 in y_arr.shape:
        y_arr = y_arr.reshape(-1)
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D vector of labels (or a 2D vector with one singleton dimension).")
    if y_arr.size != N:
        raise ValueError(f"y length mismatch: expected {N} labels, got {y_arr.size}.")
    return y_arr


# ----------------------------
# Training
# ----------------------------

def train_kahm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    subspace_dim: int,
    Nb: int,
    random_state: int | None = 0,
    verbose: bool = True,
    input_scale: float = 1.0,
    # Limit training samples per class to control model size
    max_train_per_class: int | None = None,
    # Downcast arrays inside the trained classifier to reduce RAM.
    model_dtype: str = "auto",
    # Disk-backed classifier storage
    save_ae_to_disk: bool = True,
    ae_dir: str | Path | None = None,
    ae_cache_root: str | Path = "kahm_ae_cache",
    overwrite_ae_dir: bool = False,
    model_id: str | None = None,
    ae_compress: int = 3,
) -> dict:
    """Train a KAHM-based multiclass classifier from labeled data.

    Parameters
    ----------
    X:
        Input matrix shaped (D_in, N).
    y:
        Labels for each sample (length N). Can be numeric or hashable objects.
    subspace_dim, Nb:
        OTFL autoencoder hyperparameters (passed to parallel_autoencoders).
    input_scale:
        Scalar multiplier applied to X both at train and infer time (for AE gating).
    max_train_per_class:
        If set, subsample at most this many training samples per class for AE training.
        This is the highest-ROI lever to reduce disk/RAM footprint when C is large.
    save_ae_to_disk:
        If True, save each per-class AE to disk and store relative joblib paths in model['classifier'].

    Returns
    -------
    model dict (see module docstring).
    """
    X = _as_float_ndarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array shaped (D_in, N).")
    D_in, N = X.shape

    y_vec = _normalize_y_input(y, N=N)

    # Choose a working dtype.
    md0 = str(model_dtype).lower().strip()
    if md0 in ("auto", "none", ""):
        work_dtype = np.result_type(X.dtype)
    elif md0 in ("float32", "f32"):
        work_dtype = np.float32
    elif md0 in ("float64", "f64"):
        work_dtype = np.float64
    else:
        raise ValueError("model_dtype must be one of {'auto','float32','float64'}")

    X = X.astype(work_dtype, copy=False)

    # Stable class order (important when labels are non-numeric)
    classes_list = _stable_unique_labels(y_vec)
    C = len(classes_list)
    if C < 2:
        raise ValueError(f"Need at least 2 classes; got {C}.")

    # Map labels to 0..C-1
    label_to_idx = {lab: i for i, lab in enumerate(classes_list)}
    y_idx = np.array([label_to_idx[v] for v in y_vec.tolist()], dtype=np.int64)

    counts = np.bincount(y_idx, minlength=C)
    if counts.min() < 2:
        # OTFL AE training is not well-defined with <=1 sample
        bad = np.where(counts < 2)[0].tolist()
        bad_labels = [classes_list[i] for i in bad]
        raise ValueError(
            "Some classes have <2 samples; cannot train per-class autoencoders reliably. "
            f"Bad class indices={bad}, labels={bad_labels}."
        )

    if verbose:
        print(f"Training KAHM classifier on {N} samples.")
        print(f"Input dim: {D_in} | #classes: {C}")
        print(f"Input scaling factor (input_scale): {input_scale}")

    # Apply input scaling (for AE-based assignment)
    if float(input_scale) != 1.0:
        X = _scale_like(X, float(input_scale), inplace=False)

    # Optional subsampling per class (controls model size)
    X_clf = X
    y_idx_clf = y_idx
    if max_train_per_class is not None:
        m = int(max_train_per_class)
        if m <= 0:
            raise ValueError("max_train_per_class must be a positive integer or None.")
        if verbose:
            print(f"Subsampling AE training data: max_train_per_class={m}")

        rng = np.random.RandomState(int(random_state) if random_state is not None else 0)
        keep_parts: list[np.ndarray] = []
        for c in range(C):
            idx_c = np.where(y_idx == c)[0]
            if idx_c.size <= m:
                keep_parts.append(idx_c)
            else:
                keep_parts.append(rng.choice(idx_c, size=m, replace=False))
        keep_idx = np.concatenate(keep_parts).astype(np.int64, copy=False)
        keep_idx.sort()
        X_clf = X[:, keep_idx]
        y_idx_clf = y_idx[keep_idx]
        if verbose:
            print(f"AE training samples: {X_clf.shape[1]} (was {X.shape[1]})")

    # Ensure float32 for OTFL code (reduces peak memory)
    if X_clf.dtype != np.float32:
        X_clf = X_clf.astype(np.float32, copy=False)

    # Train per-class AEs
    if verbose:
        print("Training per-class autoencoders (parallel_autoencoders)...")

    AE_arr: list[Any] = []

    if bool(save_ae_to_disk):
        from uuid import uuid4

        run_id = str(model_id) if model_id is not None else uuid4().hex[:10]
        model_id_for_model = run_id

        if ae_dir is None:
            root = Path(ae_cache_root)
            root.mkdir(parents=True, exist_ok=True)
            ae_dir_resolved = (root / f"kahm_cls_{run_id}").resolve()
        else:
            ae_dir_resolved = Path(ae_dir).resolve()

        if ae_dir_resolved.exists():
            has_existing = any(ae_dir_resolved.rglob("*.joblib"))
            if has_existing and not bool(overwrite_ae_dir):
                raise FileExistsError(
                    f"AE directory already contains joblib files: {ae_dir_resolved}\n"
                    "Refusing to overwrite to prevent cache collisions. "
                    "Pass overwrite_ae_dir=True or choose a different ae_dir."
                )
        ae_dir_resolved.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Saving per-class AEs to: {ae_dir_resolved}")

        for c in range(C):
            idx_c = np.where(y_idx_clf == c)[0]
            if idx_c.size == 0:
                raise RuntimeError(f"Internal error: class {c} has 0 training samples after subsampling.")
            X_c = X_clf[:, idx_c]
            if verbose:
                print(f" Class {c + 1}/{C}: training autoencoder on {X_c.shape[1]} samples ...")

            AE_c_list = parallel_autoencoders(
                X_c,
                subspace_dim=subspace_dim,
                Nb=Nb,
                n_jobs=1,
                verbose=False,
            )

            fname = f"class_{c:05d}.joblib"
            fpath = ae_dir_resolved / fname
            dump(AE_c_list, str(fpath), compress=int(ae_compress))
            AE_arr.append(fname)

        classifier_dir_for_model = str(ae_dir_resolved)
        clf = AE_arr
    else:
        model_id_for_model = str(model_id) if model_id is not None else None
        for c in range(C):
            idx_c = np.where(y_idx_clf == c)[0]
            if idx_c.size == 0:
                raise RuntimeError(f"Internal error: class {c} has 0 training samples after subsampling.")
            X_c = X_clf[:, idx_c]
            if verbose:
                print(f" Class {c + 1}/{C}: training autoencoder on {X_c.shape[1]} samples ...")
            AE_c_list = parallel_autoencoders(
                X_c,
                subspace_dim=subspace_dim,
                Nb=Nb,
                n_jobs=1,
                verbose=False,
            )
            AE_arr.append(AE_c_list)
        classifier_dir_for_model = None
        clf = AE_arr

    # Downcast model arrays in AEs (best-effort)
    md = str(model_dtype).lower().strip()
    if md in ("auto", "none", ""):
        _dtype = work_dtype
    elif md in ("float32", "f32"):
        _dtype = np.float32
    elif md in ("float64", "f64"):
        _dtype = np.float64
    else:
        _dtype = work_dtype

    def _downcast_obj(obj):
        if isinstance(obj, np.ndarray) and obj.dtype.kind in "fc":
            return obj.astype(_dtype, copy=False)
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = _downcast_obj(v)
            return obj
        if isinstance(obj, (list, tuple)):
            out = [_downcast_obj(v) for v in obj]
            return out if isinstance(obj, list) else tuple(out)
        return obj

    if not bool(save_ae_to_disk):
        try:
            clf = _downcast_obj(clf)
        except Exception:
            pass

    if verbose:
        print("Training finished.")

    return {
        "classifier": clf,
        "classifier_dir": classifier_dir_for_model,
        "model_id": model_id_for_model,
        "classes_": np.asarray(classes_list, dtype=object),
        "n_classes": int(C),
        "input_scale": float(input_scale),
        # soft params (optional; filled by tune_soft_params_ce)
        "soft_alpha": None,
        "soft_topk": None,
        # tuning memory behavior
        "tune_memmap_threshold_bytes": 512 * 1024 * 1024,
    }


# ----------------------------
# Optional: preload classifier for repeated inference calls
# ----------------------------

def preload_kahm_classifier(model: dict, *, n_jobs: int = 1, prefer: str = "threads") -> None:
    """Load all per-class autoencoders into memory once and cache them in model['_classifier_cache'].

    Useful when the classifier is disk-backed (joblib paths) and you call kahm_classify()
    repeatedly in an outer loop.

    Notes
    -----
    - For very large numbers of classes, preloading may be infeasible due to RAM.
    - The cache is a runtime optimization only. save_kahm_classifier() strips keys starting with "_".
    """
    from joblib import Parallel, delayed, load as _load

    clf = model.get("classifier", None)
    if not isinstance(clf, (list, tuple)) or len(clf) == 0:
        raise TypeError("model['classifier'] must be a non-empty list/tuple to preload.")

    # Already materialized
    if not isinstance(clf[0], (str, os.PathLike, Path)):
        model["_classifier_cache"] = list(clf)
        return

    base_dir = model.get("classifier_dir", None)

    def _resolve_one(p: str | os.PathLike) -> str:
        pp = Path(p)
        if not pp.is_absolute() and base_dir is not None:
            pp = Path(base_dir) / pp
        return str(pp)

    paths = [_resolve_one(p) for p in clf]
    loaded = Parallel(n_jobs=int(n_jobs), prefer=str(prefer))(delayed(_load)(p) for p in paths)
    model["_classifier_cache"] = loaded


# ----------------------------
# Inference
# ----------------------------

@overload
def kahm_classify(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: Literal[False] = False,
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray: ...


@overload
def kahm_classify(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: Literal[True],
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...


def kahm_classify(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: bool = False,
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Predict class labels (and optionally probabilities) for new inputs.

    Parameters
    ----------
    mode:
        - "hard": choose class with minimum distance (argmin D).
        - "soft": compute probabilities via (1 - D) ** alpha and optional top-k truncation.
          Note: predicted labels are typically the same as hard mode; soft mainly affects P.
    return_probabilities:
        If True, also return probability matrix P shaped (C, N_new) with rows aligned to model['classes_'].
    alpha, topk:
        Soft parameters. If None, will use stored model values if present, otherwise defaults.
    batch_size:
        For large N_new, compute distances in batches to control memory usage.
        In hard mode, this reduces peak memory during distance evaluation.
    """
    # ----------------------------
    # Local helpers
    # ----------------------------
    def _maybe_tqdm_total(total: int, desc: str, unit: str):
        if not show_progress:
            return None
        try:
            from tqdm import tqdm  # type: ignore
            return tqdm(total=total, desc=desc, unit=unit, leave=False)
        except Exception:
            return None

    def _is_pathlike(x) -> bool:
        return isinstance(x, (str, os.PathLike, Path))

    def _resolve_ae_path(ae_ref):
        if not _is_pathlike(ae_ref):
            return ae_ref
        p = Path(ae_ref)
        base_dir = model.get("classifier_dir", None)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        return str(p)

    def _extract_pc(obj):
        if isinstance(obj, dict) and "PC" in obj:
            return obj.get("PC", None)
        if isinstance(obj, (list, tuple)):
            for it in obj:
                if isinstance(it, dict) and "PC" in it:
                    return it.get("PC", None)
        return None

    def _guard_pc_shape(ae_obj, *, D_in: int, ae_ref=None, class_idx: int | None = None) -> None:
        pc = _extract_pc(ae_obj)
        if pc is None or not isinstance(pc, np.ndarray) or pc.ndim != 2:
            return
        if pc.shape[0] == D_in:
            return
        if pc.shape[1] == D_in:
            raise ValueError(
                f"Incompatible AE projection matrix PC shape {pc.shape} for input dimension D_in={D_in} "
                f"(class {class_idx}). This strongly suggests you are loading the wrong AE files "
                f"(cache collision / overwritten kahm_ae_cache).\n"
                f"Model classifier_dir={model.get('classifier_dir')!r}, model_id={model.get('model_id')!r}, "
                f"AE ref={ae_ref!r}."
            )
        raise ValueError(
            f"Incompatible AE projection matrix PC shape {pc.shape} for input dimension D_in={D_in} "
            f"(class {class_idx}). Model classifier_dir={model.get('classifier_dir')!r}, "
            f"model_id={model.get('model_id')!r}, AE ref={ae_ref!r}."
        )

    def _load_ae_maybe(ae_ref, *, class_idx: int | None = None):
        cache = model.get("_classifier_cache", None)
        if isinstance(cache, (list, tuple)) and class_idx is not None and 0 <= class_idx < len(cache):
            ae_obj = cache[class_idx]
            _guard_pc_shape(ae_obj, D_in=int(X_new.shape[0]), ae_ref="(cache)", class_idx=class_idx + 1)
            return ae_obj, False

        if _is_pathlike(ae_ref):
            resolved = _resolve_ae_path(ae_ref)
            ae_obj = load(resolved)
            _guard_pc_shape(
                ae_obj,
                D_in=int(X_new.shape[0]),
                ae_ref=resolved,
                class_idx=(class_idx + 1 if class_idx is not None else None),
            )
            return ae_obj, True

        _guard_pc_shape(
            ae_ref,
            D_in=int(X_new.shape[0]),
            ae_ref="(in-memory)",
            class_idx=(class_idx + 1 if class_idx is not None else None),
        )
        return ae_ref, False

    # ----------------------------
    # Input validation + scaling
    # ----------------------------
    X_new = _as_float_ndarray(X_new)
    if X_new.ndim != 2:
        raise ValueError("X_new must be 2D shaped (D_in, N_new).")

    input_scale = float(model.get("input_scale", 1.0))
    if input_scale != 1.0:
        X_new = _scale_like(X_new, float(input_scale), inplace=False)

    AE_arr = model.get("_classifier_cache", model.get("classifier", None))
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError("Expected model['classifier'] (or model['_classifier_cache']) to be a non-empty list/tuple.")

    classes = model.get("classes_", None)
    if classes is None:
        raise KeyError("model must contain 'classes_'.")

    C_eff = int(model.get("n_classes", len(AE_arr)))
    if len(AE_arr) != C_eff:
        C_eff = len(AE_arr)  # trust actual list length
    if np.asarray(classes).size != C_eff:
        raise ValueError(f"Mismatch: model['classes_'] has {np.asarray(classes).size} entries but classifier has {C_eff}.")

    N_new = int(X_new.shape[1])
    out_dtype = np.float32
    distance_type = "folding"

    # -----------------------
    # FAST PATH (no probabilities OR hard one-hot):
    # - If probabilities are not requested, we only need the argmin distances.
    # - If mode == "hard" and probabilities are requested, return a one-hot matrix
    #   without computing the full distance matrix.
    # -----------------------
    if mode not in ("hard", "soft"):
        raise ValueError("mode must be either 'hard' or 'soft'.")

    if (not return_probabilities) or (mode == "hard"):
        best_dist = np.full((N_new,), np.inf, dtype=np.float64)
        best_idx = np.zeros((N_new,), dtype=np.int64)

        bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else None
        pbar = _maybe_tqdm_total(C_eff * N_new, desc="KAHM: distance eval", unit="sample")
        try:
            for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
                AE_c, from_disk = _load_ae_maybe(AE_ref, class_idx=c)
                try:
                    if bs is None:
                        d = _call_combine_multiple_autoencoders_extended(
                            X_new, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs
                        )
                        d = np.asarray(d, dtype=np.float64).reshape(-1)
                        if d.size != N_new:
                            raise ValueError(
                                f"Distance vector from class {c + 1} has shape {d.shape}; expected ({N_new},)."
                            )
                        mask = d < best_dist
                        best_dist[mask] = d[mask]
                        best_idx[mask] = c
                        if pbar is not None:
                            pbar.update(N_new)
                    else:
                        for start in range(0, N_new, bs):
                            end = min(start + bs, N_new)
                            X_batch = X_new[:, start:end]
                            d = _call_combine_multiple_autoencoders_extended(
                                X_batch, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs
                            )
                            d = np.asarray(d, dtype=np.float64).reshape(-1)
                            if d.size != (end - start):
                                raise ValueError(
                                    f"Distance vector from class {c + 1} has shape {d.shape}; expected ({end-start},)."
                                )
                            cols = np.arange(start, end)
                            mask = d < best_dist[cols]
                            upd = cols[mask]
                            best_dist[upd] = d[mask]
                            best_idx[upd] = c
                            if pbar is not None:
                                pbar.update(end - start)
                finally:
                    if from_disk:
                        del AE_c
                        _gc.collect()
        finally:
            if pbar is not None:
                pbar.close()

        classes_arr = np.asarray(classes, dtype=object).reshape(-1)
        y_pred = classes_arr[best_idx]

        if return_probabilities:
            P_hard = np.zeros((C_eff, N_new), dtype=out_dtype)
            P_hard[best_idx, np.arange(N_new)] = 1.0
            return y_pred, P_hard

        return y_pred

    # Soft mode with probabilities requested: compute full distance matrix D.
    alpha_resolved, topk_resolved = _get_soft_params_from_model(model, alpha, topk)
    if topk_resolved is not None and int(topk_resolved) <= 0:
        topk_resolved = None

    # Distance computation (optionally batched)
    D = np.empty((C_eff, N_new), dtype=np.float32)
    bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else None

    pbar = _maybe_tqdm_total(C_eff * N_new, desc="KAHM: distance eval", unit="sample")
    try:
        for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
            AE_c, from_disk = _load_ae_maybe(AE_ref, class_idx=c)
            try:
                if bs is None:
                    d = _call_combine_multiple_autoencoders_extended(
                        X_new, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs
                    )
                    d = np.asarray(d, dtype=np.float32).reshape(-1)
                    if d.size != N_new:
                        raise ValueError(
                            f"Distance vector from class {c + 1} has shape {d.shape}; expected ({N_new},)."
                        )
                    D[c, :] = d
                    if pbar is not None:
                        pbar.update(N_new)
                else:
                    for start in range(0, N_new, bs):
                        end = min(start + bs, N_new)
                        X_batch = X_new[:, start:end]
                        d = _call_combine_multiple_autoencoders_extended(
                            X_batch, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs
                        )
                        d = np.asarray(d, dtype=np.float32).reshape(-1)
                        if d.size != (end - start):
                            raise ValueError(
                                f"Distance vector from class {c + 1} has shape {d.shape}; expected ({end-start},)."
                            )
                        D[c, start:end] = d
                        if pbar is not None:
                            pbar.update(end - start)
            finally:
                if from_disk:
                    del AE_c
                    _gc.collect()
    finally:
        if pbar is not None:
            pbar.close()

    # Hard predictions from D (argmin)
    best_idx = np.argmin(D, axis=0).astype(np.int64, copy=False)
    classes_arr = np.asarray(classes, dtype=object).reshape(-1)
    y_pred = classes_arr[best_idx]

    alpha_resolved, topk_resolved = _get_soft_params_from_model(model, alpha, topk)
    if topk_resolved is not None and int(topk_resolved) <= 0:
        topk_resolved = None

    # Distance computation (optionally batched)
    # Note: Return probabilities implies holding P (C,N) anyway, so D is usually acceptable in float32.
    D = np.empty((C_eff, N_new), dtype=np.float32)
    bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else None

    pbar = _maybe_tqdm_total(C_eff * N_new, desc="KAHM: distance eval", unit="sample")
    try:
        for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
            AE_c, from_disk = _load_ae_maybe(AE_ref, class_idx=c)
            try:
                if bs is None:
                    d = _call_combine_multiple_autoencoders_extended(X_new, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                    d = np.asarray(d, dtype=np.float32).reshape(-1)
                    if d.size != N_new:
                        raise ValueError(f"Distance vector from class {c + 1} has shape {d.shape}; expected ({N_new},).")
                    D[c, :] = d
                    if pbar is not None:
                        pbar.update(N_new)
                else:
                    for start in range(0, N_new, bs):
                        end = min(start + bs, N_new)
                        X_batch = X_new[:, start:end]
                        d = _call_combine_multiple_autoencoders_extended(X_batch, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                        d = np.asarray(d, dtype=np.float32).reshape(-1)
                        if d.size != (end - start):
                            raise ValueError(
                                f"Distance vector from class {c + 1} has shape {d.shape}; expected ({end-start},)."
                            )
                        D[c, start:end] = d
                        if pbar is not None:
                            pbar.update(end - start)
            finally:
                if from_disk:
                    del AE_c
                    _gc.collect()
    finally:
        if pbar is not None:
            pbar.close()

    # Hard predictions from D (argmin)
    best_idx = np.argmin(D, axis=0).astype(np.int64, copy=False)
    classes_arr = np.asarray(classes, dtype=object).reshape(-1)
    y_pred = classes_arr[best_idx]


    # Soft probabilities
    w = (1.0 - D.astype(np.float64, copy=False))
    np.clip(w, 0.0, 1.0, out=w)
    a_f = float(alpha_resolved)
    if a_f != 1.0:
        np.power(w, a_f, out=w)

    if topk_resolved is not None:
        kk = int(topk_resolved)
        if 0 < kk < C_eff:
            _topk_truncate_inplace(w, kk)

    denom = w.sum(axis=0, dtype=np.float64)
    zero_cols = denom <= 1e-12
    denom = np.where(zero_cols, 1.0, denom)
    w /= denom[None, :]

    if np.any(zero_cols):
        w[:, zero_cols] = 1.0 / C_eff

    P = w.astype(out_dtype, copy=False)

    if return_probabilities:
        return y_pred, P
    return y_pred


def kahm_predict_proba(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Convenience wrapper: return P only (shape: (C, N_new))."""
    _, P = kahm_classify(
        model,
        X_new,
        n_jobs=n_jobs,
        mode="soft",
        return_probabilities=True,
        alpha=alpha,
        topk=topk,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return P


# ----------------------------
# Soft parameter tuning (cross-entropy)
# ----------------------------

def tune_soft_params_ce(
    model: dict,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    alphas: Iterable[float] = (2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0),
    topks: Iterable[int | None] = (None, 2, 5, 10, 20, 50),
    n_jobs: int = -1,
    verbose: bool = True,
    show_progress: bool = True,
    # Stability / optional regularization
    eps: float = 1e-12,
    label_smoothing: float = 0.0,
) -> SoftTuningResultCE:
    """Tune alpha/topk on a validation set by minimizing cross-entropy.

    This computes the full distance matrix once (optionally memmapped to disk if large),
    and then streams over validation samples to evaluate each (alpha, topk) candidate
    with low additional memory overhead.

    y_val can be original labels (matching model['classes_']).
    """
    Xv = _as_float_ndarray(X_val)
    if Xv.ndim != 2:
        raise ValueError("X_val must be 2D shaped (D_in, N_val).")
    N_val = int(Xv.shape[1])

    yv = _normalize_y_input(y_val, N=N_val)

    # Apply input scaling
    input_scale = float(model.get("input_scale", 1.0))
    if input_scale != 1.0:
        Xv = _scale_like(Xv, float(input_scale), inplace=False)

    AE_arr = model.get("_classifier_cache", model.get("classifier", None))
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError("Expected model['classifier'] (or model['_classifier_cache']) to be a non-empty list/tuple.")

    C_eff = int(model.get("n_classes", len(AE_arr)))
    if len(AE_arr) != C_eff:
        C_eff = len(AE_arr)

    y_idx = _map_labels_to_indices(model, yv)
    if y_idx.min() < 0 or y_idx.max() >= C_eff:
        raise ValueError("y_val contains labels outside the model's class set.")

    # Progress helper
    def _maybe_tqdm_total(total: int, desc: str, unit: str):
        if not show_progress:
            return None
        try:
            from tqdm import tqdm  # type: ignore
            return tqdm(total=total, desc=desc, unit=unit, leave=False)
        except Exception:
            return None

    # Decide whether to memmap D_val
    D_bytes = C_eff * N_val * 4  # float32 estimate
    use_memmap = D_bytes > int(model.get("tune_memmap_threshold_bytes", 512 * 1024 * 1024))

    D_mm: np.memmap | None = None
    tmp_path: str | None = None

    base_dir = model.get("classifier_dir", None)

    def _is_pathlike(x) -> bool:
        return isinstance(x, (str, os.PathLike, Path))

    def _resolve_ae_ref(ae_ref):
        if isinstance(ae_ref, (str, os.PathLike, Path)):
            pp = Path(ae_ref)
            if not pp.is_absolute() and base_dir is not None:
                pp = Path(base_dir) / pp
            return str(pp)
        return ae_ref

    def _extract_pc(obj):
        if isinstance(obj, dict) and "PC" in obj:
            return obj.get("PC", None)
        if isinstance(obj, (list, tuple)):
            for it in obj:
                if isinstance(it, dict) and "PC" in it:
                    return it.get("PC", None)
        return None

    def _guard_pc(ae_obj, *, D_in: int, ae_ref=None, class_idx: int | None = None) -> None:
        pc = _extract_pc(ae_obj)
        if pc is None or not isinstance(pc, np.ndarray) or pc.ndim != 2:
            return
        if pc.shape[0] == D_in:
            return
        if pc.shape[1] == D_in:
            raise ValueError(
                f"Incompatible AE PC shape {pc.shape} for input dimension D_in={D_in} (class {class_idx}). "
                f"This strongly suggests you are loading the wrong AE files (cache collision). "
                f"classifier_dir={model.get('classifier_dir')!r}, model_id={model.get('model_id')!r}, ae_ref={ae_ref!r}."
            )
        raise ValueError(
            f"Incompatible AE PC shape {pc.shape} for input dimension D_in={D_in} (class {class_idx}). "
            f"classifier_dir={model.get('classifier_dir')!r}, model_id={model.get('model_id')!r}, ae_ref={ae_ref!r}."
        )

    try:
        if use_memmap:
            tmp = tempfile.NamedTemporaryFile(prefix="kahm_Dval_cls_", suffix=".dat", delete=False)
            tmp_path = tmp.name
            tmp.close()
            D_mm = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(C_eff, N_val))
            D_val_mat = D_mm
            if verbose:
                print(f"Using on-disk memmap for D_val: {tmp_path}")
        else:
            D_val_mat = np.empty((C_eff, N_val), dtype=np.float32)

        # Compute full distance matrix once
        pbar = _maybe_tqdm_total(C_eff * N_val, desc="Tune: distance eval", unit="sample")
        try:
            for c, ae_ref in enumerate(AE_arr):
                ref = _resolve_ae_ref(ae_ref)
                if _is_pathlike(ref):
                    AE_c = load(ref)
                    from_disk = True
                else:
                    AE_c = ref
                    from_disk = False

                _guard_pc(AE_c, D_in=int(Xv.shape[0]), ae_ref=str(ref) if _is_pathlike(ref) else "(in-memory)", class_idx=c + 1)

                try:
                    d = _call_combine_multiple_autoencoders_extended(Xv, _ae_as_list(AE_c), "folding", n_jobs=n_jobs)
                    d = np.asarray(d, dtype=np.float32).reshape(-1)
                    if d.size != N_val:
                        raise ValueError(f"Distance vector from class {c + 1} has shape {d.shape}; expected ({N_val},).")
                    D_val_mat[c, :] = d
                    if pbar is not None:
                        pbar.update(N_val)
                finally:
                    if from_disk:
                        del AE_c
                        _gc.collect()
        finally:
            if pbar is not None:
                pbar.close()

        if isinstance(D_val_mat, np.memmap):
            D_val_mat.flush()

        # Grid evaluation in batches (streaming)
        # Choose a safe evaluation batch size
        bs_cfg = int(model.get("tune_eval_batch_size", 4096))
        # float64 work buffer (C*B*8)
        work_max_bytes = int(model.get("tune_work_max_bytes", 256 * 1024 * 1024))
        bs_mem = max(1, int(work_max_bytes // (max(1, C_eff) * 8)))
        bs = max(1, min(int(N_val), int(bs_cfg), int(bs_mem)))

        if verbose and N_val > bs:
            print(f"Evaluation batch size: {bs} (streaming)")

        work = np.empty((C_eff, bs), dtype=np.float64)
        denom = np.empty((bs,), dtype=np.float64)

        best_ce = np.inf
        best_alpha = None
        best_topk = None
        best_acc = -np.inf

        ls = float(label_smoothing)
        if not (0.0 <= ls < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")

        one = 1.0
        zero = 0.0

        for a in alphas:
            a_f = float(a)
            for k in topks:
                total_loss = 0.0
                total_correct = 0
                total_count = 0

                for start_col in range(0, N_val, bs):
                    end_col = min(N_val, start_col + bs)
                    B = end_col - start_col

                    w = work[:, :B]
                    # w = (1 - D) ** alpha
                    np.subtract(one, D_val_mat[:, start_col:end_col], out=w)
                    np.clip(w, zero, one, out=w)
                    if a_f != 1.0:
                        np.power(w, a_f, out=w)

                    # top-k truncation
                    if k is not None:
                        kk = int(k)
                        if kk <= 0:
                            kk = None
                        if kk is not None and 0 < kk < C_eff:
                            _topk_truncate_inplace(w, kk)

                    # denom and p_true
                    dcol = denom[:B]
                    dcol[:] = w.sum(axis=0, dtype=np.float64)
                    zero_cols = dcol <= eps
                    if np.any(zero_cols):
                        dcol[zero_cols] = one

                    # predicted class for accuracy (argmax w)
                    pred = np.argmax(w, axis=0).astype(np.int64, copy=False)
                    yi = y_idx[start_col:end_col]
                    total_correct += int(np.sum(pred == yi))
                    total_count += int(B)

                    # Normalize in-place only if needed for label smoothing; otherwise compute p_true as num/den.
                    if ls > 0.0:
                        np.divide(w, dcol, out=w)
                        if np.any(zero_cols):
                            w[:, zero_cols] = 1.0 / C_eff
                        p_true = w[yi, np.arange(B)]
                        p_true = np.maximum(p_true, eps)
                        log_p_true = np.log(p_true)

                        # mean log prob across classes (for smoothed targets)
                        log_w = np.log(np.maximum(w, eps))
                        mean_log = log_w.mean(axis=0)
                        loss_b = -((1.0 - ls) * log_p_true + ls * mean_log)
                    else:
                        num = w[yi, np.arange(B)]
                        p_true = num / dcol
                        p_true = np.maximum(p_true, eps)
                        loss_b = -np.log(p_true)

                    total_loss += float(np.sum(loss_b))

                ce = float(total_loss / max(1, total_count))
                acc = float(total_correct / max(1, total_count))

                if verbose:
                    print(f"  alpha={a_f:g}, topk={k}: CE={ce:.6f}, acc={acc:.4f}")

                # Primary criterion: cross-entropy; secondary: accuracy
                better = (ce < best_ce) or (np.isclose(ce, best_ce) and acc > best_acc)
                if better:
                    best_ce = ce
                    best_alpha = a_f
                    best_topk = (None if (k is None or int(k) <= 0) else int(k))
                    best_acc = acc

        model["soft_alpha"] = float(best_alpha if best_alpha is not None else 10.0)
        model["soft_topk"] = best_topk

        if verbose:
            print(f"Best soft params: alpha={model['soft_alpha']}, topk={model['soft_topk']}, CE={best_ce:.6f}, acc={best_acc:.4f}")

        return SoftTuningResultCE(
            best_alpha=float(model["soft_alpha"]),
            best_topk=model["soft_topk"],
            best_cross_entropy=float(best_ce),
            best_accuracy=float(best_acc),
        )

    finally:
        try:
            if D_mm is not None:
                del D_mm
        except Exception:
            pass
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ----------------------------
# Save / Load
# ----------------------------

def save_kahm_classifier(model: dict, path: str) -> None:
    """Save a KAHM classifier to disk.

    Notes
    -----
    - Runtime-only keys starting with "_" (e.g., _classifier_cache) are stripped.
    - If `model['classifier_dir']` is an absolute path, it is saved *relative* to the model
      file directory whenever possible, improving portability when moving the saved artifact
      together with its classifier directory.
    """
    model_to_save = {k: v for k, v in model.items() if not str(k).startswith("_")}

    clf_dir = model_to_save.get("classifier_dir", None)
    if isinstance(clf_dir, (str, os.PathLike)):
        try:
            model_dir = Path(path).resolve().parent
            clf_dir_p = Path(clf_dir)
            if clf_dir_p.is_absolute():
                model_to_save["classifier_dir"] = os.path.relpath(str(clf_dir_p), str(model_dir))
        except Exception:
            pass

    dump(model_to_save, path)
    print(f"KAHM classifier saved to {path}")


def load_kahm_classifier(path: str, *, base_dir: str | None = None) -> dict:
    """Load a KAHM classifier from disk.

    Portability behavior
    --------------------
    - If `base_dir` is provided, it takes precedence.
    - Otherwise, if the stored `classifier_dir` is relative, it is resolved relative to the
      directory containing `path`.
    """
    model = load(path)

    if base_dir is not None:
        model["classifier_dir"] = str(base_dir)
    else:
        clf_dir = model.get("classifier_dir", None)
        if isinstance(clf_dir, (str, os.PathLike)):
            p = Path(clf_dir)
            if not p.is_absolute():
                model_dir = Path(path).resolve().parent
                model["classifier_dir"] = str((model_dir / p).resolve())

    print(f"KAHM classifier loaded from {path}")
    return model


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    np.random.seed(0)

    MODEL_PATH = "kahm_classifier_example.joblib"

    # Toy data: 3 classes in 5D input
    D_in, N = 5, 9000
    C = 3
    X = np.random.randn(D_in, N).astype(np.float32)
    y = np.zeros((N,), dtype=np.int64)
    y[N // 3 : 2 * N // 3] = 1
    y[2 * N // 3 :] = 2

    # Shift class means
    X[:, y == 1] += 1.5
    X[:, y == 2] -= 1.5
    X = np.tanh(X)

    # Split: train / val / test
    N_train = int(0.7 * N)
    N_val = int(0.15 * N)

    X_train, y_train = X[:, :N_train], y[:N_train]
    X_val, y_val = X[:, N_train : N_train + N_val], y[N_train : N_train + N_val]
    X_test, y_test = X[:, N_train + N_val :], y[N_train + N_val :]

    model = train_kahm_classifier(
        X_train,
        y_train,
        subspace_dim=20,
        Nb=10,
        random_state=0,
        verbose=True,
        input_scale=1.0,
        save_ae_to_disk=False,
    )

    tune_res = tune_soft_params_ce(
        model,
        X_val,
        y_val,
        alphas=(2.0, 5.0, 10.0, 11.0,12.0,13.0,14.0, 15.0, 16.0, 17.0,18.0,19.0,20.0),
        topks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 20, 25, 50, 100),
        verbose=True,
    )

    y_pred, P = kahm_classify(model, X_test, mode="soft", return_probabilities=True, batch_size=1024)
    acc = float(np.mean(y_pred.astype(np.int64) == y_test.astype(np.int64)))
    nll = float(-np.mean(np.log(np.maximum(P[y_test, np.arange(P.shape[1])], 1e-12))))
    print(f"Test accuracy={acc:.4f}, Test NLL={nll:.6f}")

    save_kahm_classifier(model, MODEL_PATH)
    loaded = load_kahm_classifier(MODEL_PATH)
    y_pred2 = kahm_classify(loaded, X_test, mode="hard", batch_size=1024)
    acc2 = float(np.mean(y_pred2.astype(np.int64) == y_test.astype(np.int64)))
    print(f"Reloaded model test accuracy (hard)={acc2:.4f}")
