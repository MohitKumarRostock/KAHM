# dim_reduce.py
import numpy as np

def dim_reduce(y_data: np.ndarray, n: int):
    y_data = np.asarray(y_data, dtype=float)
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
        raise ValueError("No eigenvalues > eps")

    max_dim = valid[-1] + 1
    n_eff = min(n, max_dim)

    PC = eigvecs_sorted[:, :n_eff]
    y_data_n_subspace = PC.T @ y_data

    return y_data_n_subspace, PC