import numpy as np

def autoencoder(y_data: np.ndarray, subspace_dim: int, min_range: float = 1e-3):
    y_data = np.asarray(y_data, dtype=np.float32)
    D, N = y_data.shape

    if subspace_dim > (N - 1):
        subspace_dim = N - 1

    # initial projection
    from dim_reduce import dim_reduce  # import where you actually have it
    y_data_n_subspace, PC = dim_reduce(y_data, subspace_dim)

    # shrink subspace_dim if range too small
    while True:
        ranges = y_data_n_subspace.max(axis=1) - y_data_n_subspace.min(axis=1)
        if np.min(ranges) >= min_range or subspace_dim <= 1:
            break
        subspace_dim -= 1
        y_data_n_subspace, PC = dim_reduce(y_data, subspace_dim)

    # --- FIX IS HERE ---
    # cov(AE.y_data_n_subspace') in MATLAB → covariance of rows (features)
    cov_matrix = np.cov(y_data_n_subspace.T, rowvar=False, bias=False)
    cov_matrix = np.atleast_2d(cov_matrix)   # ensure 2D even when 1D subspace

    # you can keep inv, or for safety use pinv:
    weights_matrix = np.linalg.inv(cov_matrix)
    # weights_matrix = np.linalg.pinv(cov_matrix)  # <- safer alternative

    kerneltype = "Gaussian"

    from kxx_matrix import kxx_matrix
    from kernel_regularized_least_squares import kernel_regularized_least_squares

    Kxx = kxx_matrix(y_data_n_subspace, weights_matrix, kerneltype)
    B = kernel_regularized_least_squares(Kxx, y_data)

    # matrix square root of weights_matrix
    eigvals, eigvecs = np.linalg.eigh(weights_matrix)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    s = np.sqrt(eigvals_clipped).astype(eigvecs.dtype, copy=False)
    sqrt_weights_matrix = (eigvecs * s[None, :]) @ eigvecs.T


    AE = {
        "y_data_n_subspace": y_data_n_subspace,
        "PC": PC,
        "kerneltype": kerneltype,
        "B": B,
        "y_data": y_data,
        "sqrt_weights_matrix": sqrt_weights_matrix,
    }

    return AE