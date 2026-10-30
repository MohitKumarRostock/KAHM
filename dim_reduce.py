# dim_reduce.py
import numpy as np

def dim_reduce(y_data: np.ndarray, n: int):
    y_data = np.asarray(y_data, dtype=float)
    if y_data.ndim != 2 or y_data.shape[0] == 0 or y_data.shape[1] < 2:
        raise ValueError("y_data must have shape (features, samples) with at least two samples.")
    if not np.isfinite(y_data).all():
        raise ValueError("y_data must contain only finite values.")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    d, N = y_data.shape

    data = y_data - y_data.mean(axis=1, keepdims=True)
    covariance = (1.0 / (N - 1)) * (data @ data.T)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    idx = np.argsort(-eigvals)
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    eps = np.finfo(float).eps
    valid = np.where(eigvals_sorted > eps)[0]
    if valid.size == 0:
        if np.all(y_data == y_data[:, :1]):
            # Constant references have a zero-dimensional affine hull.
            return np.empty((0, N), dtype=float), np.empty((d, 0), dtype=float)
        raise ValueError("Reference variation is below numerical resolution; rescale the input features.")

    max_dim = valid[-1] + 1
    n_eff = min(n, max_dim)

    PC = eigvecs_sorted[:, :n_eff]
    y_data_n_subspace = PC.T @ y_data

    return y_data_n_subspace, PC
