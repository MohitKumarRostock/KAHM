"""
kahm_regression.py

KAHM-based multivariate regression via output clustering.

Soft / "true" regression (distance-matrix based):
- combine_multiple_autoencoders_extended is used with one autoencoder per output cluster
  to obtain per-cluster distances D of shape (C_eff, N_new).
- Distances are assumed to lie in [0, 1] and be monotone with (1 - probability).

Sharpened + truncated probability mapping (recommended when C_eff is large):
    S = (1 - D) ** alpha
    keep only top-k scores per sample (optional)
    P = S / sum_c S

Prediction:
    Y_hat = cluster_centers @ P

Autotuning support:
- tune_soft_params(...) evaluates a grid over (alpha, topk) on a validation set,
  stores best values in the model dict as:
      model['soft_alpha'], model['soft_topk']

Inference defaults:
- kahm_regress(..., mode='soft', alpha=None, topk=None) will automatically use
  stored model values if present, otherwise falls back to alpha=10, topk=10.

Requires:
    pip install scikit-learn numpy joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Literal, overload
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike
from sklearn.cluster import KMeans, MiniBatchKMeans
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
class SoftTuningResult:
    """Return type for tune_soft_params."""

    best_alpha: float
    best_topk: int | None
    best_mse: float




@dataclass(frozen=True)
class NLMSCenterTuningResult:
    """Return type for tune_cluster_centers_nlms."""

    mu: float
    epsilon: float
    epochs: int
    batch_size: int
    final_mse: float
    mse_history: Tuple[float, ...]

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
        # Promote integers/bools to a sensible floating default.
        arr = arr.astype(np.float64, copy=False)

    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)

def _scale_like(X: np.ndarray, scale: float, *, inplace: bool = False) -> np.ndarray:
    """Scale X by `scale` without unintentionally upcasting dtype.

    - If X is float32 and scale is a Python float, `X * scale` would upcast to float64.
      This helper ensures the scale is cast to X.dtype first.
    - If `inplace=True` and X is writeable, scaling is performed in-place to avoid a full copy.
    """
    if scale == 1.0:
        return X
    if X.dtype.kind not in 'fc':
        X = X.astype(np.float32, copy=False)
    s = np.asarray(scale, dtype=X.dtype)
    if inplace and X.flags.writeable:
        np.multiply(X, s, out=X, casting='unsafe')
        return X
    # out-of-place but dtype-stable
    return (X * s).astype(X.dtype, copy=False)



def _ae_as_list(ae: Any) -> list:
    """Ensure an autoencoder spec is passed as a flat list.

    The OTFL helper `combine_multiple_autoencoders_extended` expects a sequence of
    autoencoder components. In some training pipelines each cluster stores:
      - a list/tuple of components, or
      - a single component (dict/ndarray-like).

    This normalizes both cases and also flattens a common accidental nesting
    pattern: [ [component, ...] ].
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
    fn: Any = combine_multiple_autoencoders_extended  # `Any` silences Pylance for version-dependent signatures
    try:
        return fn(X, AE_list, distance_type, n_jobs=n_jobs)
    except TypeError:
        return fn(X, AE_list, distance_type)

def l2_normalize_columns(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize columns of a 2D matrix (D, N). Safe for non-directional data when not used."""
    M = _as_float_ndarray(M)
    if M.ndim != 2:
        raise ValueError(f"Expected 2D matrix for l2_normalize_columns; got shape={M.shape}")
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms = np.maximum(norms, eps).astype(M.dtype, copy=False)
    return M / norms


def _should_auto_l2_normalize_targets(Y: np.ndarray) -> bool:
    """Heuristic: decide if Y columns are approximately unit-norm (directional data)."""
    Y = _as_float_ndarray(Y)
    if Y.ndim != 2 or Y.shape[1] == 0:
        return False
    norms = np.linalg.norm(Y, axis=0)
    # Robust statistics
    p10, p50, p90 = np.percentile(norms, [10, 50, 90]).tolist()
    # Treat as directional if most norms are near 1
    return (0.90 <= p50 <= 1.10) and (p10 >= 0.80) and (p90 <= 1.20)


# ----------------------------
# Training helpers
# ----------------------------

def train_kahm_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters: int,
    subspace_dim: int = 20,
    Nb: int = 100,
    random_state: int | None = 0,
    verbose: bool = True,
    input_scale: float = 1.0,
    # ----------------
    # Scalability controls
    # ----------------
    # For large n_clusters, scikit-learn's full KMeans can be both slow and memory hungry.
    # Use MiniBatchKMeans to reduce peak memory and make training practical.
    kmeans_kind: str = "auto",  # {'auto','full','minibatch'}
    kmeans_batch_size: int = 4096,
    # Limit the number of training samples used by the KAHM classifier per output cluster.
    # This directly limits the size of the stored classifier model and is the highest-ROI
    # lever when C is large.
    max_train_per_cluster: int | None = None,
    # Downcast arrays inside the trained classifier to reduce RAM.
    # (Does not change the external model API.)
    model_dtype: str = "auto",
    # Optional: normalize output cluster centroids (useful for directional/unit-norm embedding targets)
    #   - "none": do nothing (default; fully general)
    #   - "l2": always L2-normalize centroids
    #   - "auto_l2": normalize only if Y columns appear approximately unit-norm
    cluster_center_normalization: str = "none",
    # ----------------
    # Disk-backed classifier storage
    # ----------------
    # If True, save each per-cluster autoencoder to disk and store file paths in the model.
    # This prevents RAM growth when the number of clusters is large.
    save_ae_to_disk: bool = True,
    # Directory to store per-cluster AEs. If None and save_ae_to_disk=True,
    # a unique directory under ae_cache_root will be created.
    ae_dir: str | Path | None = None,
    ae_cache_root: str | Path = "kahm_ae_cache",
    overwrite_ae_dir: bool = False,
    # Optional ID to make AE directories stable/reproducible.
    model_id: str | None = None,
    # joblib compression level for AE shards (0..9). 3 is a good speed/size trade-off.
    ae_compress: int = 3,
) -> dict:
    """Train a KAHM-based regressor via output clustering."""
    X = _as_float_ndarray(X)
    Y = _as_float_ndarray(Y)

    # Choose a shared working dtype.
    # - If model_dtype is "auto"/"none": preserve input precision (up to float32 minimum from _as_float_ndarray).
    # - If model_dtype is "float32"/"float64": use that dtype during training to reduce peak RAM.
    md0 = str(model_dtype).lower().strip()
    if md0 in ("auto", "none", ""):
        work_dtype = np.result_type(X.dtype, Y.dtype)
    elif md0 in ("float32", "f32"):
        work_dtype = np.float32
    elif md0 in ("float64", "f64"):
        work_dtype = np.float64
    else:
        raise ValueError("model_dtype must be one of {'auto','float32','float64'}")

    X = X.astype(work_dtype, copy=False)
    Y = Y.astype(work_dtype, copy=False)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays shaped (D, N).")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of samples (columns). "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}."
        )

    D_in, N = X.shape
    D_out, _ = Y.shape

    if N < 2 * n_clusters:
        raise ValueError(
            f"Not enough samples ({N}) to ensure at least 2 samples per cluster "
            f"for n_clusters={n_clusters}. Need N >= 2 * n_clusters."
        )

    if verbose:
        print(f"Training KAHM regressor on {N} samples.")
        print(f"Input dim:  {D_in}, Output dim: {D_out}")
        print(f"Requested number of clusters: {n_clusters}")
        print(f"Input scaling factor (input_scale): {input_scale}")

    # Apply input scaling (for AE-based cluster assignment)
    if input_scale != 1.0:
        X = _scale_like(X, float(input_scale), inplace=False)

    # 1) K-means on outputs
    if verbose:
        print("Running K-means on outputs...")

    Y_T = Y.T  # (N, D_out)

    kind = str(kmeans_kind).lower().strip()
    if kind not in ("auto", "full", "minibatch"):
        raise ValueError("kmeans_kind must be one of {'auto','full','minibatch'}")

    use_minibatch = (kind == "minibatch") or (kind == "auto" and int(n_clusters) >= 2000)

    if use_minibatch:
        if verbose:
            print(
                f"Using MiniBatchKMeans (n_clusters={n_clusters}, batch_size={int(kmeans_batch_size)}) "
                "to reduce peak memory."
            )
        kmeans = MiniBatchKMeans(
            n_clusters=int(n_clusters),
            random_state=random_state,
            batch_size=int(kmeans_batch_size),
            n_init="auto",
            reassignment_ratio=0.01,
        )
        kmeans.fit(Y_T)
    else:
        kmeans = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
        kmeans.fit(Y_T)
    labels_zero_based = kmeans.labels_.astype(int)

    # 1a) Merge singleton clusters into nearest non-singleton cluster
    counts = np.bincount(labels_zero_based, minlength=n_clusters)
    singletons = np.where(counts == 1)[0]

    if singletons.size > 0 and verbose:
        print(f"Merging {singletons.size} singleton cluster(s)...")

    if singletons.size > 0:
        centers = kmeans.cluster_centers_
        # Precompute squared norms of centers once (avoids large temporaries during singleton merging).
        c2 = np.einsum("ij,ij->i", centers, centers)
        for cl in singletons:
            sample_indices = np.where(labels_zero_based == cl)[0]
            if sample_indices.size != 1:
                continue
            s_idx = sample_indices[0]
            y_sample = Y_T[s_idx]

            # Squared distance: ||c - y||^2 = ||c||^2 + ||y||^2 - 2 c·y
            y2 = float(np.dot(y_sample, y_sample))
            d2 = c2 + y2 - 2.0 * centers.dot(y_sample)

            candidates = np.where(counts >= 2)[0]
            if candidates.size == 0:
                candidates = np.where(counts >= 1)[0]

            candidates = candidates[candidates != cl]
            if candidates.size == 0:
                continue

            target = candidates[np.argmin(d2[candidates])]
            labels_zero_based[s_idx] = target
            counts[target] += 1
            counts[cl] -= 1


    # Free the transposed matrix used for KMeans to reduce peak RAM.
    try:
        del Y_T
    except Exception:
        pass
    # 1c) Drop empty clusters, remap to contiguous labels
    used_clusters = np.unique(labels_zero_based)
    n_clusters_eff = used_clusters.size

    if verbose:
        print(f"Effective number of clusters after preprocessing: {n_clusters_eff}")

        # Vectorized remap (avoids Python-object overhead for large N)
    map_arr = np.full(int(n_clusters), -1, dtype=np.int32)
    map_arr[used_clusters.astype(np.int64)] = np.arange(n_clusters_eff, dtype=np.int32)
    labels_mapped_zero = map_arr[labels_zero_based.astype(np.int64)]
    if np.any(labels_mapped_zero < 0):
        raise RuntimeError("Internal error while remapping cluster labels (found unmapped label).")
    labels_one_based = labels_mapped_zero + 1  # OTFL expects labels 1..C_eff
  # OTFL expects labels 1..C_eff

    cluster_centers = np.zeros((D_out, n_clusters_eff), dtype=work_dtype)
    for new_c in range(n_clusters_eff):
        mask = labels_mapped_zero == new_c
        cluster_centers[:, new_c] = Y[:, mask].mean(axis=1)

    # Optional centroid normalization for directional targets (e.g., unit-norm embeddings)
    cc_req = str(cluster_center_normalization).lower().strip()
    cc_applied = "none"
    if cc_req in ("l2", "auto_l2"):
        do_norm = (cc_req == "l2") or _should_auto_l2_normalize_targets(Y)
        if do_norm:
            cluster_centers = l2_normalize_columns(cluster_centers)
            cc_applied = "l2"
    if verbose:
        print(f"cluster center normalization: {cc_applied}")

    final_counts = np.bincount(labels_mapped_zero, minlength=n_clusters_eff)
    if verbose:
        print(f"Min cluster size after preprocessing: {final_counts.min()}")

    # 2) Optionally subsample per output cluster for autoencoder training
    #    This is the most effective way to control memory when n_clusters is large.
    X_clf = X
    labels_one_based_clf = labels_one_based

    if max_train_per_cluster is not None:
        m = int(max_train_per_cluster)
        if m <= 0:
            raise ValueError("max_train_per_cluster must be a positive integer or None")
        if verbose:
            print(f"Subsampling autoencoder training data: max_train_per_cluster={m}")

        rng = np.random.RandomState(int(random_state) if random_state is not None else 0)
        keep_idx_parts: list[np.ndarray] = []
        for c in range(n_clusters_eff):
            idx = np.where(labels_mapped_zero == c)[0]
            if idx.size <= m:
                keep_idx_parts.append(idx)
            else:
                keep_idx_parts.append(rng.choice(idx, size=m, replace=False))

        keep_idx = np.concatenate(keep_idx_parts).astype(np.int64, copy=False)
        keep_idx.sort()
        X_clf = X[:, keep_idx]
        labels_one_based_clf = labels_one_based[keep_idx]
        if verbose:
            print(f"Autoencoder training samples: {X_clf.shape[1]} (was {X.shape[1]})")

    # 3) Train per-cluster autoencoders

    # Y is no longer needed after computing cluster_centers; free it before classifier training.
    try:
        del Y
    except Exception:
        pass
    _gc.collect()

    if verbose:
        print("Training per-cluster autoencoders (parallel_autoencoders)...")

    # Ensure float32 to minimize peak memory inside OTFL autoencoder code
    if X_clf.dtype != np.float32:
        X_clf = X_clf.astype(np.float32, copy=False)
    labels_one_based_clf = labels_one_based_clf.astype(np.int32, copy=False)

    # Train one autoencoder per output cluster.

    # Optionally save AEs to disk to prevent RAM growth when C_eff is large.
    # IMPORTANT: Using a single shared cache directory across different trained models can lead to
    # silent cache collisions. Therefore, when save_ae_to_disk=True and ae_dir is not provided,
    # we create a unique directory under ae_cache_root.
    AE_arr: list[Any] = []

    if bool(save_ae_to_disk):
        from pathlib import Path
        from uuid import uuid4
        import os

        run_id = str(model_id) if model_id is not None else uuid4().hex[:10]

        if ae_dir is None:
            root = Path(ae_cache_root)
            root.mkdir(parents=True, exist_ok=True)
            ae_dir_resolved = (root / f"kahm_{run_id}").resolve()
        else:
            ae_dir_resolved = Path(ae_dir).resolve()

        # Refuse overwriting unless explicitly allowed
        if ae_dir_resolved.exists():
            has_existing = any(ae_dir_resolved.rglob("*.joblib"))
            if has_existing and not bool(overwrite_ae_dir):
                raise FileExistsError(
                    f"AE directory already contains joblib files: {ae_dir_resolved}\n"
                    "Refusing to overwrite to prevent model/corpus cache collisions. "
                    "Pass overwrite_ae_dir=True or choose a different ae_dir."
                )
        ae_dir_resolved.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Saving per-cluster autoencoders to: {ae_dir_resolved}")

        for c in range(n_clusters_eff):
            idx_c = np.where(labels_one_based_clf == (c + 1))[0]
            if idx_c.size < 2:
                raise ValueError(
                    f"Cluster {c + 1} has only {idx_c.size} sample(s) after subsampling; "
                    "need at least 2 to train an autoencoder. "
                    "Increase max_train_per_cluster or reduce n_clusters."
                )

            X_c = X_clf[:, idx_c]
            if verbose:

                print(f" Cluster {c + 1}/{n_clusters_eff}: training autoencoder on {X_c.shape[1]} samples ...")
            AE_c_list = parallel_autoencoders(
                X_c,
                subspace_dim=subspace_dim,
                Nb=Nb,
                n_jobs=1,
                verbose=False,
            )

            # Shard subdirectories to avoid placing many thousands of files in one folder
            shard = ae_dir_resolved / f"{c // 1000:03d}"
            shard.mkdir(parents=True, exist_ok=True)

            ae_path = shard / f"ae_cluster_{c + 1:05d}.joblib"
            dump(AE_c_list, ae_path, compress=int(ae_compress))

            # Store paths relative to classifier_dir for portability
            AE_arr.append(os.path.relpath(str(ae_path), str(ae_dir_resolved)))

            del X_c, idx_c, AE_c_list

        # Disk-backed classifier: store relative joblib paths
        clf = AE_arr

        # Record classifier directory + ID in model
        classifier_dir_for_model = str(ae_dir_resolved)
        model_id_for_model = run_id

    else:
        # In-memory classifier (legacy behavior)
        model_id_for_model = None
        classifier_dir_for_model = None

        for c in range(n_clusters_eff):
            idx_c = np.where(labels_one_based_clf == (c + 1))[0]
            if idx_c.size < 2:
                raise ValueError(
                    f"Cluster {c + 1} has only {idx_c.size} sample(s) after subsampling; "
                    "need at least 2 to train an autoencoder. "
                    "Increase max_train_per_cluster or reduce n_clusters."
                )

            X_c = X_clf[:, idx_c]
            if verbose:

                print(f" Cluster {c + 1}/{n_clusters_eff}: training autoencoder on {X_c.shape[1]} samples ...")
            AE_c_list = parallel_autoencoders(
                X_c,
                subspace_dim=subspace_dim,
                Nb=Nb,
                n_jobs=1,
                verbose=False,
            )
            AE_arr.append(AE_c_list)

        clf = AE_arr

    # 4) Downcast model arrays to reduce RAM footprint
    #    If you need even more savings, you can change PC to float16,
    #    but that may affect numerical stability in some OTFL implementations.
    md = str(model_dtype).lower().strip()
    if md in ("auto", "none", ""):
        _dtype = work_dtype
    elif md in ("float32", "f32"):
        _dtype = np.float32
    elif md in ("float64", "f64"):
        _dtype = np.float64
    else:
        raise ValueError("model_dtype must be one of {'auto','float32','float64'}")

    def _downcast_obj(obj):
        if isinstance(obj, np.ndarray) and obj.dtype.kind in "fc":
            return obj.astype(_dtype, copy=False)
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = _downcast_obj(v)
            return obj
        if isinstance(obj, (list, tuple)):
            out = [ _downcast_obj(v) for v in obj ]
            return out if isinstance(obj, list) else tuple(out)
        return obj

    try:
        clf = _downcast_obj(clf)
    except Exception:
        # If OTFL returns unexpected structures, keep the model unmodified.
        pass

    if verbose:
        print("Training finished.")

    return {
        "classifier": clf,
        "classifier_dir": classifier_dir_for_model,
        "model_id": model_id_for_model,
        "cluster_centers_init": cluster_centers.copy(),  # (D_out, C_eff) initial centers
        "cluster_centers": cluster_centers,  # (D_out, C_eff) tuned/current centers
        "n_clusters": int(n_clusters_eff),
        "input_scale": float(input_scale),
        # soft params (optional; filled by tune_soft_params)
        "soft_alpha": None,
        "soft_topk": None,
        "cluster_centers_normalization_requested": cc_req,
        "cluster_centers_normalization": cc_applied,
    }

# ----------------------------
# Soft probability mapping
# ----------------------------

def _topk_truncate_inplace(S: np.ndarray, k: int, *, chunk_cols: int = 64) -> None:
    """Zero all but the top-k entries per column of S in-place (memory-aware).

    This helper is designed to avoid allocating an index matrix of shape (C, N),
    which can dominate peak RAM when C and N are large.

    Implementation detail:
      - processes columns in chunks, limiting temporary index arrays to (C, chunk_cols)
    """
    if S.ndim != 2:
        raise ValueError("S must be 2D shaped (C, N).")
    C, N = S.shape
    k = int(k)
    if k <= 0 or k >= C or N == 0:
        return

    # Cap temporary index memory (~32 MiB by default).
    max_index_bytes = 32 * 1024 * 1024
    max_chunk = max(1, int(max_index_bytes // (8 * max(1, C))))
    chunk_cols = max(1, min(int(chunk_cols), max_chunk))

    for start in range(0, N, chunk_cols):
        end = min(N, start + chunk_cols)
        sub = S[:, start:end]  # (C, B)
        # argpartition returns full indices for the submatrix, but B is bounded.
        idx = np.argpartition(sub, -k, axis=0)
        idx_top = idx[-k:, :]  # (k, B)
        vals_top = np.take_along_axis(sub, idx_top, axis=0).copy()
        sub.fill(S.dtype.type(0.0))
        np.put_along_axis(sub, idx_top, vals_top, axis=0)


def distances_to_probabilities_one_minus_sharp(
    distance_matrix: np.ndarray,
    *,
    alpha: float = 10.0,
    topk: int | None = 10,
    eps: float = 1e-12,
    inplace: bool = False,
) -> np.ndarray:
    """
    Convert distances D (C, N) into probabilities P (C, N):

        S = (1 - D) ** alpha
        (optional) keep only top-k scores per column
        P = S / sum_c S

    Notes on memory
    ---------------
    - This implementation avoids creating an additional full-size `P` matrix.
      The returned array *is* the score/probability buffer.
    - If `inplace=True`, the input matrix is overwritten (or copied if not writeable).

    Assumes D in [0,1]. Uses uniform fallback if sum_c S == 0 for a sample.
    """
    D = _as_float_ndarray(distance_matrix)
    if D.ndim != 2:
        raise ValueError("distance_matrix must be 2D shaped (C, N).")

    C, N = D.shape
    dtype = D.dtype
    one = dtype.type(1.0)
    zero = dtype.type(0.0)
    eps_t = dtype.type(eps)

    if inplace:
        S = D if D.flags.writeable else D.copy()
        np.subtract(one, S, out=S)
    else:
        # Allocate a single full-size buffer for scores/probabilities.
        S = np.subtract(one, D)

    # Clamp to [0, 1] to avoid negative scores if D has minor numerical drift.
    np.clip(S, zero, one, out=S)

    a = float(alpha)
    if a != 1.0:
        np.power(S, dtype.type(a), out=S)

    if topk is not None:
        k = int(topk)
        if 0 < k < C:
            _topk_truncate_inplace(S, k)

    denom = S.sum(axis=0, dtype=dtype)  # (N,)
    zero_cols = denom <= eps_t
    if np.any(zero_cols):
        denom = denom.copy()
        denom[zero_cols] = one

    # Normalize in-place (broadcast along rows).
    np.divide(S, denom, out=S)

    if np.any(zero_cols):
        S[:, zero_cols] = dtype.type(1.0 / C)

    return S

def _ensure_distance_matrix_shape(
    D: np.ndarray,
    C_eff: int,
    N: int,
    labels: Optional[np.ndarray] = None,
    *,
    square_sample: int = 512,
) -> np.ndarray:
    """Ensure distance matrix D is shaped (C_eff, N).

    For non-square shapes this is unambiguous:
      - (C_eff, N): returned as-is
      - (N, C_eff): transposed

    For the corner case C_eff == N (square matrix), shape alone is ambiguous.
    If `labels` (length N) are provided (as returned by OTFL), we disambiguate by
    choosing the orientation whose argmin assignments best match the labels.
    """
    if D.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D; got shape {D.shape}.")

    # Unambiguous cases
    if D.shape == (C_eff, N) and (C_eff != N or D.shape != (N, C_eff)):
        return D
    if D.shape == (N, C_eff) and (C_eff != N or D.shape != (C_eff, N)):
        return D.T

    # If we reach here, shapes are either square (C_eff == N) or otherwise mismatched.
    if D.shape != (C_eff, N) and D.shape != (N, C_eff):
        raise ValueError(
            f"distance_matrix shape mismatch. Expected {(C_eff, N)} or {(N, C_eff)}, got {D.shape}."
        )

    # Square ambiguity: C_eff == N and D is (N, N)
    if C_eff != N:
        # Should not happen due to checks above, but keep safe.
        return D if D.shape == (C_eff, N) else D.T

    if labels is None:
        # Cannot disambiguate; keep backward-compatible behavior:
        # treat D as already (C, N). (Downstream code assumes axis-0 is clusters.)
        return D

    lab = np.asarray(labels).reshape(-1)
    if lab.size != N:
        raise ValueError(f"labels length mismatch for square distance matrix: labels={lab.size}, N={N}")

    # Sample indices to keep this cheap for very large N
    if square_sample is not None and int(square_sample) > 0 and int(square_sample) < N:
        idx = np.linspace(0, N - 1, int(square_sample), dtype=np.int64)
    else:
        idx = np.arange(N, dtype=np.int64)

    # Hypothesis A: D is (C, N)
    pred_cn = np.argmin(D[:, idx], axis=0).astype(np.int64)
    acc_cn = float((pred_cn == lab[idx].astype(np.int64)).mean())

    # Hypothesis B: D is (N, C) (so D.T is (C, N))
    pred_nc = np.argmin(D[idx, :], axis=1).astype(np.int64)
    acc_nc = float((pred_nc == lab[idx].astype(np.int64)).mean())

    return D if acc_cn >= acc_nc else D.T


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

    Important semantics
    -------------------
    - `topk=None` means "use the model setting".
    - To *disable* top-k truncation, either:
        * set `model['soft_topk'] = None`, or
        * pass `topk=0` (or any non-positive value).
    """
    if alpha is None:
        alpha = model.get("soft_alpha", None)

    # Distinguish: unspecified vs. model explicitly storing None (meaning "disable top-k").
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
# Prediction
# ----------------------------

# ----------------------------
# Optional: preload classifier for repeated inference calls
# ----------------------------

def preload_kahm_classifier(model: dict, *, n_jobs: int = 1, prefer: str = "threads") -> None:
    """
    Load all per-cluster autoencoders into memory once and cache them in model['_classifier_cache'].

    This is useful when you call kahm_regress() repeatedly in an outer loop (e.g., corpus batching)
    and the classifier is disk-backed (joblib paths). Preloading avoids re-loading AEs from disk
    for each call, at the cost of increased RAM usage.

    Notes
    -----
    - For very large numbers of clusters (e.g., 10k+), preloading may be infeasible due to RAM.
    - The cache is a runtime optimization only. save_kahm_regressor() strips keys starting with "_".
    """
    import os
    from pathlib import Path
    from joblib import Parallel, delayed, load

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
    loaded = Parallel(n_jobs=int(n_jobs), prefer=str(prefer))(delayed(load)(p) for p in paths)
    model["_classifier_cache"] = loaded


@overload
def kahm_regress(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: Literal[False] = False,
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # For large N_new, compute in batches to control memory usage
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray: ...


@overload
def kahm_regress(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: Literal[True],
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # For large N_new, compute in batches to control memory usage
    batch_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]: ...


def kahm_regress(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: bool = False,
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # For large N_new, compute in batches to control memory usage
    batch_size: Optional[int] = None,
    # Progress bar (uses tqdm if installed)
    show_progress: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Predict outputs for new inputs using a trained KAHM regressor.

    (Docstring unchanged — omitted here for brevity)
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

    def _guard_pc_shape(ae_obj, *, D_in: int, ae_ref=None, cluster_idx: int | None = None) -> None:
        pc = _extract_pc(ae_obj)
        if pc is None:
            return
        if not isinstance(pc, np.ndarray) or pc.ndim != 2:
            return
        if pc.shape[0] == D_in:
            return
        if pc.shape[1] == D_in:
            raise ValueError(
                f"Incompatible AE projection matrix PC shape {pc.shape} for input dimension D_in={D_in} "
                f"(cluster {cluster_idx}). This strongly suggests you are loading the wrong AE files "
                f"(cache collision / overwritten kahm_ae_cache).\n"
                f"Model classifier_dir={model.get('classifier_dir')!r}, model_id={model.get('model_id')!r}, "
                f"AE ref={ae_ref!r}."
            )
        raise ValueError(
            f"Incompatible AE projection matrix PC shape {pc.shape} for input dimension D_in={D_in} "
            f"(cluster {cluster_idx}). Model classifier_dir={model.get('classifier_dir')!r}, "
            f"model_id={model.get('model_id')!r}, AE ref={ae_ref!r}."
        )

    def _load_ae_maybe(ae_ref, *, cluster_idx: int | None = None):
        cache = model.get("_classifier_cache", None)
        if isinstance(cache, (list, tuple)) and cluster_idx is not None and 0 <= cluster_idx < len(cache):
            ae_obj = cache[cluster_idx]
            _guard_pc_shape(ae_obj, D_in=int(X_new.shape[0]), ae_ref="(cache)", cluster_idx=cluster_idx + 1)
            return ae_obj, False

        if _is_pathlike(ae_ref):
            resolved = _resolve_ae_path(ae_ref)
            ae_obj = load(resolved)
            _guard_pc_shape(
                ae_obj,
                D_in=int(X_new.shape[0]),
                ae_ref=resolved,
                cluster_idx=(cluster_idx + 1 if cluster_idx is not None else None),
            )
            return ae_obj, True

        _guard_pc_shape(
            ae_ref,
            D_in=int(X_new.shape[0]),
            ae_ref="(in-memory)",
            cluster_idx=(cluster_idx + 1 if cluster_idx is not None else None),
        )
        return ae_ref, False

    def _update_topk_inplace(best_d, best_i, worst_pos, worst_val, d, cluster_idx: int) -> None:
        mask = d < worst_val
        if not np.any(mask):
            return
        cols = np.nonzero(mask)[0]
        pos = worst_pos[cols]
        best_d[pos, cols] = d[cols]
        best_i[pos, cols] = cluster_idx
        sub = best_d[:, cols]
        new_wpos = np.argmax(sub, axis=0)
        worst_pos[cols] = new_wpos
        worst_val[cols] = sub[new_wpos, np.arange(cols.size)]

    def _soft_predict_from_topk(idx, dist, *, centers, alpha_f: float, out_dtype: np.dtype) -> np.ndarray:
        k, nb = dist.shape
        w = 1.0 - dist
        np.clip(w, 0.0, 1.0, out=w)
        if alpha_f != 1.0:
            np.power(w, alpha_f, out=w)

        denom = w.sum(axis=0)
        zero_cols = denom <= 1e-12
        denom = np.where(zero_cols, 1.0, denom)
        w /= denom

        Y_hat = np.zeros((centers.shape[0], nb), dtype=out_dtype)
        idx_safe = idx.copy()
        idx_safe[idx_safe < 0] = 0

        for i in range(k):
            Y_hat += centers[:, idx_safe[i, :]] * w[i, :][None, :]

        if np.any(zero_cols):
            mean_center = centers.mean(axis=1, keepdims=True).astype(out_dtype, copy=False)
            Y_hat[:, zero_cols] = mean_center
        return Y_hat

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

    cluster_centers = _as_float_ndarray(model["cluster_centers"])  # (D_out, C_eff)
    if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
        cluster_centers = l2_normalize_columns(cluster_centers)

    C_eff = int(cluster_centers.shape[1])
    if len(AE_arr) != C_eff:
        raise ValueError(f"Mismatch: got {len(AE_arr)} autoencoders but cluster_centers has C_eff={C_eff} clusters.")

    N_new = int(X_new.shape[1])
    out_dtype = np.result_type(cluster_centers.dtype, np.float32)
    distance_type = "folding"

    # -----------------------
    # HARD MODE
    # -----------------------
    if mode == "hard":
        best_dist = np.full((N_new,), np.inf, dtype=np.float64)
        best_idx = np.zeros((N_new,), dtype=np.int64)

        bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else None
        # Total distance-evals in hard mode ~ C_eff * N_new
        pbar = _maybe_tqdm_total(C_eff * N_new, desc="KAHM hard: distance eval", unit="sample")

        try:
            for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
                AE_c, from_disk = _load_ae_maybe(AE_ref, cluster_idx=c)
                try:
                    if bs is None:
                        d = _call_combine_multiple_autoencoders_extended(X_new, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                        d = np.asarray(d, dtype=np.float64).reshape(-1)
                        if d.size != N_new:
                            raise ValueError(
                                f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_new},)."
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
                            d = _call_combine_multiple_autoencoders_extended(X_batch, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                            d = np.asarray(d, dtype=np.float64).reshape(-1)
                            if d.size != (end - start):
                                raise ValueError(
                                    f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({end-start},)."
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

        Y_pred = cluster_centers[:, best_idx]
        if return_probabilities:
            P_hard = np.zeros((C_eff, N_new), dtype=out_dtype)
            P_hard[best_idx, np.arange(N_new)] = 1.0
            return Y_pred, P_hard
        return Y_pred

    # -----------------------
    # SOFT MODE
    # -----------------------
    if mode != "soft":
        raise ValueError("mode must be either 'hard' or 'soft'.")

    alpha_resolved, topk_resolved = _get_soft_params_from_model(model, alpha, topk)

    # FAST TOP-K path
    if (not return_probabilities) and (topk_resolved is not None):
        k_req = int(topk_resolved)
        if 0 < k_req < C_eff:
            bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else N_new
            bs = max(1, bs)

            slices: list[tuple[int, int]] = [(s, min(s + bs, N_new)) for s in range(0, N_new, bs)]
            k = min(k_req, C_eff)

            best_d_list: list[np.ndarray] = []
            best_i_list: list[np.ndarray] = []
            worst_pos_list: list[np.ndarray] = []
            worst_val_list: list[np.ndarray] = []

            for (s, e) in slices:
                nb = e - s
                bd = np.full((k, nb), np.inf, dtype=np.float64)
                bi = np.full((k, nb), -1, dtype=np.int64)
                wp = np.zeros((nb,), dtype=np.int64)
                wv = np.full((nb,), np.inf, dtype=np.float64)
                best_d_list.append(bd)
                best_i_list.append(bi)
                worst_pos_list.append(wp)
                worst_val_list.append(wv)

            # Progress over total distance-evals ~ C_eff * N_new
            pbar = _maybe_tqdm_total(C_eff * N_new, desc="KAHM soft: distance eval", unit="sample")
            try:
                for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
                    AE_c, from_disk = _load_ae_maybe(AE_ref, cluster_idx=c)
                    try:
                        for b_idx, (s, e) in enumerate(slices):
                            X_batch = X_new[:, s:e]
                            d = _call_combine_multiple_autoencoders_extended(X_batch, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                            d = np.asarray(d, dtype=np.float64).reshape(-1)
                            if d.size != (e - s):
                                raise ValueError(
                                    f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({e-s},)."
                                )
                            _update_topk_inplace(
                                best_d_list[b_idx],
                                best_i_list[b_idx],
                                worst_pos_list[b_idx],
                                worst_val_list[b_idx],
                                d,
                                c,
                            )
                            if pbar is not None:
                                pbar.update(e - s)
                    finally:
                        if from_disk:
                            del AE_c
                            _gc.collect()
            finally:
                if pbar is not None:
                    pbar.close()

            # Predict from top-k buffers (optional short progress)
            Y_pred = np.empty((cluster_centers.shape[0], N_new), dtype=out_dtype)
            alpha_f = float(alpha_resolved)

            pbar2 = _maybe_tqdm_total(N_new, desc="KAHM soft: assemble", unit="sample")
            try:
                for b_idx, (s, e) in enumerate(slices):
                    bd = best_d_list[b_idx]
                    bi = best_i_list[b_idx]

                    order = np.argsort(bd, axis=0)
                    bd_sorted = np.take_along_axis(bd, order, axis=0)
                    bi_sorted = np.take_along_axis(bi, order, axis=0)

                    Y_pred[:, s:e] = _soft_predict_from_topk(
                        bi_sorted, bd_sorted, centers=cluster_centers, alpha_f=alpha_f, out_dtype=out_dtype
                    )
                    if pbar2 is not None:
                        pbar2.update(e - s)
            finally:
                if pbar2 is not None:
                    pbar2.close()

            return Y_pred

    # Dense fallback (unchanged)
    if batch_size is not None and int(batch_size) > 0 and not return_probabilities:
        bs = int(batch_size)

        tmp = tempfile.NamedTemporaryFile(prefix="kahm_D_", suffix=".dat", delete=False)
        tmp_path = tmp.name
        tmp.close()

        D_mm = None
        # Progress over total distance-evals ~ C_eff * N_new
        pbar = _maybe_tqdm_total(C_eff * N_new, desc="KAHM soft(dense): distance eval", unit="sample")

        try:
            D_mm = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(C_eff, N_new))

            for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
                AE_c, from_disk = _load_ae_maybe(AE_ref, cluster_idx=c)
                try:
                    for start in range(0, N_new, bs):
                        end = min(start + bs, N_new)
                        X_batch = X_new[:, start:end]
                        d = _call_combine_multiple_autoencoders_extended(X_batch, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                        d = np.asarray(d, dtype=np.float32).reshape(-1)
                        if d.size != (end - start):
                            raise ValueError(
                                f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({end-start},)."
                            )
                        D_mm[c, start:end] = d
                        if pbar is not None:
                            pbar.update(end - start)
                finally:
                    if from_disk:
                        del AE_c
                        _gc.collect()

            if isinstance(D_mm, np.memmap):
                D_mm.flush()

            Y_pred = np.empty((cluster_centers.shape[0], N_new), dtype=out_dtype)

            pbar2 = _maybe_tqdm_total(N_new, desc="KAHM soft(dense): predict", unit="sample")
            try:
                for start in range(0, N_new, bs):
                    end = min(start + bs, N_new)
                    N_b = end - start
                    D_batch = np.asarray(D_mm[:, start:end], dtype=np.float64)
                    D_batch = _ensure_distance_matrix_shape(_as_float_ndarray(D_batch), C_eff, N_b, labels=None)
                    P_batch = distances_to_probabilities_one_minus_sharp(
                        D_batch, alpha=float(alpha_resolved), topk=topk_resolved, inplace=True
                    )
                    Y_pred[:, start:end] = cluster_centers @ P_batch
                    if pbar2 is not None:
                        pbar2.update(N_b)
            finally:
                if pbar2 is not None:
                    pbar2.close()

            return Y_pred

        finally:
            if pbar is not None:
                pbar.close()
            try:
                if D_mm is not None:
                    del D_mm
            except Exception:
                pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Non-batched dense path (and return_probabilities=True) unchanged
    D = np.empty((C_eff, N_new), dtype=np.float64)
    for c, AE_ref in enumerate(model.get("classifier", AE_arr)):
        AE_c, from_disk = _load_ae_maybe(AE_ref, cluster_idx=c)
        try:
            d = _call_combine_multiple_autoencoders_extended(X_new, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
            d = np.asarray(d, dtype=np.float64).reshape(-1)
            if d.size != N_new:
                raise ValueError(
                    f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_new},)."
                )
            D[c, :] = d
        finally:
            if from_disk:
                del AE_c
                _gc.collect()

    D = _ensure_distance_matrix_shape(_as_float_ndarray(D), C_eff, N_new, labels=None)
    P = distances_to_probabilities_one_minus_sharp(D, alpha=float(alpha_resolved), topk=topk_resolved, inplace=True)
    Y_pred = cluster_centers @ P

    if return_probabilities:
        return Y_pred, P
    return Y_pred

def tune_soft_params(
    model: dict,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    alphas: Sequence[float] = (5.0, 10.0, 15.0, 20.0),
    topks: Sequence[int | None] = (5, 10, 15, 20),
    n_jobs: int = -1,
    verbose: bool = True,
) -> SoftTuningResult:
    """
    Tune (alpha, topk) on a validation set and store the best choice in `model`.

    Notes
    -----
    - Computes the full distance matrix once for X_val and reuses it for all grid points.
    - Objective: minimize mean squared error (MSE) over all output dimensions and samples.
    - Supports disk-backed classifiers (joblib paths) as well as in-memory AEs.
    """
    import os
    import gc as _gc
    from pathlib import Path
    from joblib import load
    import tempfile

    X_val = _as_float_ndarray(X_val)
    Y_val = _as_float_ndarray(Y_val)

    if X_val.ndim != 2 or Y_val.ndim != 2:
        raise ValueError("X_val and Y_val must be 2D shaped (D, N).")

    if X_val.shape[1] != Y_val.shape[1]:
        raise ValueError("X_val and Y_val must have the same number of samples (columns).")

    # Apply same scaling used during training
    input_scale = float(model.get("input_scale", 1.0))
    Xv = _scale_like(X_val, float(input_scale), inplace=False) if input_scale != 1.0 else X_val

    cluster_centers = _as_float_ndarray(model["cluster_centers"])
    if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
        cluster_centers = l2_normalize_columns(cluster_centers)

    C_eff = int(cluster_centers.shape[1])
    N_val = int(Xv.shape[1])

    # Resolve classifier entries
    clf_cache = model.get("_classifier_cache", None)
    clf = clf_cache if isinstance(clf_cache, (list, tuple)) and len(clf_cache) else model.get("classifier", None)

    if isinstance(clf, (str, os.PathLike, Path)):
        # Directory path containing *.joblib
        p = Path(clf)
        if p.is_dir():
            files = sorted(p.rglob("*.joblib"))
            if len(files) == 0:
                raise TypeError(f"Classifier directory contains no *.joblib: {p}")
            AE_arr = [str(f) for f in files]
        else:
            raise TypeError("classifier as a string must be a directory path.")
    elif isinstance(clf, (list, tuple)) and len(clf) > 0:
        AE_arr = list(clf)
    else:
        raise TypeError(
            "Could not resolve model autoencoders. Expected model['classifier'] (or '_classifier_cache') "
            "to be a non-empty list/tuple of AEs or AE joblib paths, or a directory path."
        )

    if len(AE_arr) != C_eff:
        raise ValueError(
            f"Mismatch: got {len(AE_arr)} autoencoders but cluster_centers has C_eff={C_eff} clusters."
        )

    base_dir = model.get("classifier_dir", None)

    def _resolve_ae_ref(ae_ref):
        if isinstance(ae_ref, (str, os.PathLike, Path)):
            pp = Path(ae_ref)
            if not pp.is_absolute() and base_dir is not None:
                pp = Path(base_dir) / pp
            return str(pp)
        return ae_ref

    def _guard_pc(ae_obj, *, D_in: int, ae_ref=None, cluster_idx: int | None = None) -> None:
        # Same guard as in kahm_regress (fast, prevents silent cache collisions).
        def _extract_pc(obj):
            if isinstance(obj, dict) and "PC" in obj:
                return obj.get("PC", None)
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    if isinstance(it, dict) and "PC" in it:
                        return it.get("PC", None)
            return None

        pc = _extract_pc(ae_obj)
        if pc is None or not isinstance(pc, np.ndarray) or pc.ndim != 2:
            return
        if pc.shape[0] == D_in:
            return
        if pc.shape[1] == D_in:
            raise ValueError(
                f"Incompatible AE PC shape {pc.shape} for input dimension D_in={D_in} (cluster {cluster_idx}). "
                f"This strongly suggests you are loading the wrong AE files (cache collision). "
                f"classifier_dir={model.get('classifier_dir')!r}, model_id={model.get('model_id')!r}, ae_ref={ae_ref!r}."
            )
        raise ValueError(
            f"Incompatible AE PC shape {pc.shape} for input dimension D_in={D_in} (cluster {cluster_idx}). "
            f"classifier_dir={model.get('classifier_dir')!r}, model_id={model.get('model_id')!r}, ae_ref={ae_ref!r}."
        )

    # Decide whether to memmap D_val (rarely needed for typical N_val)
    D_bytes = C_eff * N_val * 8  # float64 estimate
    use_memmap = D_bytes > int(model.get("tune_memmap_threshold_bytes", 512 * 1024 * 1024))

    D_mm: np.memmap | None = None
    tmp_path: str | None = None
    try:
        if use_memmap:
            tmp = tempfile.NamedTemporaryFile(prefix="kahm_Dval_", suffix=".dat", delete=False)
            tmp_path = tmp.name
            tmp.close()
            D_mm = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(C_eff, N_val))
            D_val = D_mm
            if verbose:
                print(f"Using on-disk memmap for D_val: {tmp_path}")
        else:
            D_val = np.empty((C_eff, N_val), dtype=np.float64)

        # Compute full distance matrix once
        for c, ae_ref in enumerate(AE_arr):
            ref = _resolve_ae_ref(ae_ref)
            if isinstance(ref, (str, os.PathLike, Path)):
                AE_c = load(ref)
                from_disk = True
            else:
                AE_c = ref
                from_disk = False

            _guard_pc(AE_c, D_in=int(Xv.shape[0]), ae_ref=str(ref) if isinstance(ref, (str, os.PathLike, Path)) else "(in-memory)", cluster_idx=c + 1)

            try:
                d = _call_combine_multiple_autoencoders_extended(Xv, _ae_as_list(AE_c), "folding", n_jobs=n_jobs)
                d = np.asarray(d).reshape(-1)
                if d.size != N_val:
                    raise ValueError(
                        f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_val},)."
                    )
                if isinstance(D_val, np.memmap):
                    D_val[c, :] = np.asarray(d, dtype=np.float32)
                else:
                    D_val[c, :] = np.asarray(d, dtype=np.float64)
            finally:
                if from_disk:
                    del AE_c
                    _gc.collect()

        if isinstance(D_val, np.memmap):
            # Ensure data is written to disk; keep memmap dtype (typically float32) to avoid
            # materializing the full distance matrix into RAM.
            D_val.flush()

        # Keep D_val in its existing dtype; downstream tuning streams over columns to control peak RAM.
        D_val_mat = _ensure_distance_matrix_shape(_as_float_ndarray(D_val), C_eff, N_val, labels=None)

        alphas = tuple(alphas)
        topks = tuple(topks)
        if len(alphas) == 0:
            raise ValueError("alphas must contain at least one value.")
        if len(topks) == 0:
            raise ValueError("topks must contain at least one value.")

        best_mse = float("inf")
        best_alpha: float = float(alphas[0])
        best_topk: int | None = topks[0]

        if verbose:
            print("Tuning soft parameters on validation set...")
            print(f"Grid: alphas={list(alphas)}, topks={list(topks)}")
            print(f"Validation samples: {N_val}, clusters: {C_eff}")

        # Stream evaluation in column batches to avoid allocating a full (C_eff, N_val) work buffer.
        work_max_bytes = int(model.get("tune_eval_work_max_bytes", 256 * 1024 * 1024))
        bs_cfg = int(model.get("tune_eval_batch_cols", 4096))
        bs_mem = max(1, int(work_max_bytes // (max(1, C_eff) * 8)))  # float64 work buffer
        bs = max(1, min(int(N_val), int(bs_cfg), int(bs_mem)))

        if verbose and N_val > bs:
            print(f"Evaluation batch size: {bs} (streaming)")

        work = np.empty((C_eff, bs), dtype=np.float64)
        denom = np.empty((bs,), dtype=np.float64)

        one = 1.0
        zero = 0.0
        eps = 1e-12

        for a in alphas:
            a_f = float(a)
            for k in topks:
                sse = 0.0
                count = 0

                for start_col in range(0, N_val, bs):
                    end_col = min(N_val, start_col + bs)
                    B = end_col - start_col

                    w = work[:, :B]
                    np.subtract(one, D_val_mat[:, start_col:end_col], out=w)
                    np.clip(w, zero, one, out=w)
                    if a_f != 1.0:
                        np.power(w, a_f, out=w)

                    if k is not None:
                        kk = int(k)
                        if 0 < kk < C_eff:
                            _topk_truncate_inplace(w, kk)

                    dcol = denom[:B]
                    dcol[:] = w.sum(axis=0, dtype=np.float64)
                    zero_cols = dcol <= eps
                    if np.any(zero_cols):
                        dcol[zero_cols] = one

                    np.divide(w, dcol, out=w)

                    if np.any(zero_cols):
                        w[:, zero_cols] = 1.0 / C_eff

                    Y_hat_b = cluster_centers @ w
                    diff = Y_hat_b - Y_val[:, start_col:end_col]
                    sse += float(np.sum(diff * diff))
                    count += int(diff.size)

                mse = float(sse / max(1, count))

                if verbose:
                    print(f"  alpha={a_f:g}, topk={k}: MSE={mse:.6f}")

                if mse < best_mse:
                    best_mse = mse
                    best_alpha = a_f
                    best_topk = k


        model["soft_alpha"] = best_alpha
        model["soft_topk"] = best_topk

        if verbose:
            print(f"Best soft params: alpha={best_alpha}, topk={best_topk}, val MSE={best_mse:.6f}")

        return SoftTuningResult(best_alpha=best_alpha, best_topk=best_topk, best_mse=best_mse)

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
def tune_cluster_centers_nlms(
    model: dict,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mu: float = 0.1,
    epsilon: float | None = None,
    epochs: int = 1,
    batch_size: int = 1024,
    shuffle: bool = True,
    random_state: int | None = 0,
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # Optional anchoring: pull centers toward the initial KMeans centroids
    anchor_lambda: float = 0.0,
    # Performance
    n_jobs: int = -1,
    preload_classifier: bool = False,
    verbose: bool = True,
) -> NLMSCenterTuningResult:
    """Tune (refine) cluster centers via Normalized LMS (NLMS).

    Model structure
    ---------------
    The regressor predicts: Y_hat = C @ p(x),
    where C is model['cluster_centers'] (D_out, K) and p(x) are soft assignment
    probabilities computed from AE distances (independent of C).

    This function refines C by minimizing mean-squared error using an online,
    per-sample (or per-mini-batch) NLMS update:

        C <- C + (mu / (epsilon + mu*||p||^2)) * (y - C p) p^T

    Parameters
    ----------
    mu:
        Step size. Typical range for NLMS is (0, 1]. Start with 0.1..0.5.
    epsilon:
        Small positive constant to prevent division by zero when ||p||^2 is tiny.
        If None, a dtype-aware default is chosen (good general default: 1e-8).
    epochs:
        Number of passes over the dataset.
    batch_size:
        Number of samples per mini-batch. NLMS is defined per sample; mini-batching
        applies the same per-sample normalization but accumulates updates efficiently.
    anchor_lambda:
        If > 0, applies a light "pull" toward model['cluster_centers_init'] (if present),
        which stabilizes tuning when data are noisy.

    Returns
    -------
    NLMSCenterTuningResult with MSE history.
    """
    X = _as_float_ndarray(X)
    Y = _as_float_ndarray(Y)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D matrices shaped (D, N).")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y must have the same number of samples; got X.shape={X.shape}, Y.shape={Y.shape}.")

    C = model.get("cluster_centers", None)
    if C is None:
        raise KeyError("model must contain 'cluster_centers' to tune.")
    C = _as_float_ndarray(C)
    if C.ndim != 2:
        raise ValueError(f"model['cluster_centers'] must be 2D; got shape={C.shape}")
    if C.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Output dimension mismatch: model centers have D_out={C.shape[0]} but Y has D_out={Y.shape[0]}."
        )

    if epochs <= 0:
        raise ValueError("epochs must be >= 1.")
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1.")
    mu = float(mu)
    if not (mu > 0.0):
        raise ValueError("mu must be > 0.")

    # Choose epsilon: it should be much smaller than typical ||p||^2.
    # With probability vectors, ||p||^2 is in [1/K, 1]. For K up to 1e5, 1/K = 1e-5.
    if epsilon is None:
        epsilon = 1
    epsilon = float(epsilon)
    if not (epsilon > 0.0):
        raise ValueError("epsilon must be > 0.")

    N = X.shape[1]
    rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

    # Preserve initial centers for optional anchoring.
    if "cluster_centers_init" not in model or model.get("cluster_centers_init") is None:
        model["cluster_centers_init"] = np.asarray(model["cluster_centers"], dtype=np.float64).copy()

    C0 = None
    if anchor_lambda and anchor_lambda > 0.0:
        C0 = _as_float_ndarray(model.get("cluster_centers_init")).astype(np.float64, copy=False)

    # Work in float64 for stability; assign back later.
    orig_dtype = C.dtype
    C_work = np.asarray(C, dtype=np.float64).copy()
    model["cluster_centers"] = C_work

    # Optional preload (can be very RAM-heavy when K is large).
    if preload_classifier:
        try:
            preload_kahm_classifier(model, n_jobs=max(1, int(abs(n_jobs) if n_jobs != 0 else 1)))
        except Exception:
            # Preload is a performance optimization; proceed without it.
            pass

    mse_history: list[float] = []

    for ep in range(int(epochs)):
        idx = np.arange(N)
        if shuffle:
            rng.shuffle(idx)

        sse = 0.0
        count = 0

        for start in range(0, N, int(batch_size)):
            end = min(N, start + int(batch_size))
            sel = idx[start:end]
            Xb = X[:, sel]
            Yb = Y[:, sel]

            # Compute probabilities p(x) using the model's classifier + soft mapping.
            # Note: p(x) is independent of the cluster centers, so this is well-defined.
            Yhat_b, P_b = kahm_regress(
                model,
                Xb,
                n_jobs=int(n_jobs),
                mode="soft",
                return_probabilities=True,
                alpha=alpha,
                topk=topk,
                batch_size=None,
            )

            # Ensure shapes
            if P_b.ndim != 2 or P_b.shape[1] != (end - start):
                raise RuntimeError(f"Unexpected probability matrix shape: {P_b.shape}")
            if Yhat_b.ndim != 2 or Yhat_b.shape != Yb.shape:
                raise RuntimeError(f"Unexpected prediction shape: {Yhat_b.shape} vs Yb {Yb.shape}")

            E = (Yb - Yhat_b).astype(np.float64, copy=False)  # (D_out, B)
            P64 = np.asarray(P_b, dtype=np.float64)

            # Per-sample normalization term ||p||^2 (B,)
            p_norm2 = np.sum(P64 * P64, axis=0)
            step = mu / (epsilon + (mu*p_norm2))  # (B,)

            # Accumulate update: sum_i step_i * e_i p_i^T
            E_scaled = E * step[None, :]
            C_work += E_scaled @ P64.T

            # Optional anchoring toward initial centers
            if C0 is not None:
                C_work -= (mu * float(anchor_lambda)) * (C_work - C0)

            # Track MSE on the fly
            sse += float(np.sum(E * E))
            count += int(E.size)
            if verbose:
                print(f"[NLMS] count {end}/{N} | MSE={(sse/max(1, count)):.6g}...")

        mse = sse / max(1, count)
        mse_history.append(float(mse))

        if verbose:
            print(f"[NLMS] batch {ep+1}/{epochs} | MSE={mse:.6g} | mu={mu:g} | eps={epsilon:g}")

    # If the user requested/auto-applied L2 normalization for targets, preserve it after tuning.
    if model.get("cluster_centers_normalization") == "l2":
        norms = np.linalg.norm(C_work, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        C_work /= norms

    # Restore dtype if needed
    if orig_dtype != C_work.dtype:
        model["cluster_centers"] = C_work.astype(orig_dtype, copy=False)

    return NLMSCenterTuningResult(
        mu=float(mu),
        epsilon=float(epsilon),
        epochs=int(epochs),
        batch_size=int(batch_size),
        final_mse=float(mse_history[-1] if mse_history else np.nan),
        mse_history=tuple(mse_history),
    )


def save_kahm_regressor(model: dict, path: str) -> None:
    """Save a KAHM regressor to disk.

    Notes
    -----
    - Runtime-only keys starting with "_" (e.g., _classifier_cache) are stripped.
    - If `model['classifier_dir']` is an absolute path, it is saved *relative* to the model
      file directory whenever possible, improving portability when moving the saved artifact
      together with its classifier directory.
    """
    import os
    from pathlib import Path

    model_to_save = {k: v for k, v in model.items() if not str(k).startswith("_")}

    # Make classifier_dir portable (relative to the model file) if feasible.
    clf_dir = model_to_save.get("classifier_dir", None)
    if isinstance(clf_dir, (str, os.PathLike)):
        try:
            model_dir = Path(path).resolve().parent
            clf_dir_p = Path(clf_dir)
            # Store a relative path if classifier_dir is absolute or explicitly resolved.
            if clf_dir_p.is_absolute():
                model_to_save["classifier_dir"] = os.path.relpath(str(clf_dir_p), str(model_dir))
        except Exception:
            # Best-effort only; keep as-is on any failure.
            pass

    dump(model_to_save, path)
    print(f"KAHM regressor saved to {path}")


def load_kahm_regressor(path: str, *, base_dir: str | None = None) -> dict:
    """Load a KAHM regressor from disk.

    Parameters
    ----------
    path:
        Path to the saved regressor (joblib).
    base_dir:
        Optional override for `model['classifier_dir']`. Use this when relocating the model
        or when the classifier directory is stored separately.

    Portability behavior
    --------------------
    - If `base_dir` is provided, it takes precedence.
    - Otherwise, if the stored `classifier_dir` is relative, it is resolved relative to the
      directory containing `path`.
    """
    import os
    from pathlib import Path

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

    print(f"KAHM regressor loaded from {path}")
    return model
# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    np.random.seed(0)

    MODEL_PATH = "kahm_regressor_example.joblib"

    # Toy data
    D_in, D_out, N = 5, 3, 10_000
    X = np.tanh(np.random.randn(D_in, N))
    Y = np.vstack(
        [
            2 * X[0, :] + 0.5 * X[1, :] ** 2 + 0.1 * np.random.randn(N),
            -X[2, :] + np.sin(X[3, :]) + 0.1 * np.random.randn(N),
            X[4, :] * 1.5 + 0.1 * np.random.randn(N),
        ]
    )

    # Split: train / val / test
    N_train = int(0.7 * N)
    N_val = int(0.15 * N)
    X_train = X[:, :N_train]
    Y_train = Y[:, :N_train]

    X_val = X[:, N_train:N_train + N_val]
    Y_val = Y[:, N_train:N_train + N_val]

    X_test = X[:, N_train + N_val:]
    Y_test = Y[:, N_train + N_val:]

    model = train_kahm_regressor(
        X_train,
        Y_train,
        n_clusters=1000,
        subspace_dim=20,
        Nb=100,
        random_state=0,
        verbose=True,
        input_scale=0.5,
        save_ae_to_disk = False,
        cluster_center_normalization="none",
    )

    def _mse(yhat: np.ndarray, ytrue: np.ndarray) -> float:
        return float(np.mean((yhat - ytrue) ** 2))

    def _r2_overall(yhat: np.ndarray, ytrue: np.ndarray) -> float:
        residual_ss = float(np.sum((yhat - ytrue) ** 2))
        total_ss = float(np.sum((ytrue - ytrue.mean(axis=1, keepdims=True)) ** 2))
        return 1.0 - residual_ss / total_ss

    # ------------------------------------------------------------
    # 1) Tune alpha/topk on validation and store into model
    # ------------------------------------------------------------
    tune_res = tune_soft_params(
        model,
        X_val,
        Y_val,
        alphas=(2.0, 5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 20.0, 25.0, 50.0),
        topks=(2, 5, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200, 300, 400, 500),
        n_jobs=-1,
        verbose=True,
    )

    # Baseline (before center refinement)
    Y_val_pred_soft_before = kahm_regress(model, X_val, mode="soft", batch_size=1024)
    mse_val_before = _mse(Y_val_pred_soft_before, Y_val)

    print(
        f"\nSoft-params chosen on val: alpha={tune_res.best_alpha}, topk={tune_res.best_topk} | "
        f"Val MSE (soft, before NLMS centers): {mse_val_before:.6f}"
    )

    # ------------------------------------------------------------
    # 2) Refine cluster centers with NLMS (linear-parameter tuning)
    # ------------------------------------------------------------
    # This tunes model['cluster_centers'] while keeping AE gating fixed.
    # Start with mu in [0.1, 0.5].
    nlms_res = tune_cluster_centers_nlms(
        model,
        X_train,
        Y_train,
        mu=0.1,
        epsilon=1,
        epochs=20,
        batch_size=1024,
        shuffle=True,
        random_state=0,
        anchor_lambda=0.0,   # set e.g. 1e-3 to pull gently toward initial KMeans centers
        n_jobs=-1,
        preload_classifier=True,
        verbose=True,
        alpha = tune_res.best_alpha,
        topk = tune_res.best_topk
    )

    Y_val_pred_soft_after = kahm_regress(model, X_val, mode="soft", batch_size=1024)
    mse_val_after = _mse(Y_val_pred_soft_after, Y_val)

    print(
        f"NLMS center tuning: final train MSE (reported)={nlms_res.final_mse:.6f} | "
        f"Val MSE (soft, after NLMS centers): {mse_val_after:.6f}"
    )

    # Save/load
    save_kahm_regressor(model, MODEL_PATH)
    loaded_model = load_kahm_regressor(MODEL_PATH)

    # Hard prediction
    Y_pred_hard = kahm_regress(loaded_model, X_test, mode="hard", batch_size=1024)

    # Soft prediction: uses stored (soft_alpha, soft_topk) automatically
    Y_pred_soft = kahm_regress(loaded_model, X_test, mode="soft", return_probabilities=False, batch_size=1024)
    print(f"\nStored soft_alpha={loaded_model.get('soft_alpha')}, soft_topk={loaded_model.get('soft_topk')}")
    print(f"Test MSE (hard): {_mse(Y_pred_hard, Y_test):.6f} | R^2 (hard): {_r2_overall(Y_pred_hard, Y_test):.4f}")
    print(f"Test MSE (soft): {_mse(Y_pred_soft, Y_test):.6f} | R^2 (soft): {_r2_overall(Y_pred_soft, Y_test):.4f}")
