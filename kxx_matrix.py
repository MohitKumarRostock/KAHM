import numpy as np

def kxx_matrix(x_matrix, weights_matrix, kerneltype="gaussian"):
    """
    Python version of:
        function [Kxx] = KxxMatrix(x_matrix, weights_matrix, kerneltype)
    % x_matrix is n x N
    % weights_matrix is n x n

    Parameters
    ----------
    x_matrix : (n, N) array_like
        Data matrix with N samples as columns and n features as rows.
    weights_matrix : (n, n) array_like
        Weight (e.g., precision / inverse covariance) matrix.
    kerneltype : str, optional
        Type of kernel, currently only 'gaussian' is implemented.

    Returns
    -------
    Kxx : (N, N) ndarray
        Kernel matrix.
    """
    kerneltype = kerneltype.lower()

    if kerneltype == "gaussian":
        X = np.asarray(x_matrix, dtype=float)      # shape (n, N)
        W = np.asarray(weights_matrix, dtype=float)

        n, N = X.shape
        Kxx = np.zeros((N, N), dtype=float)

        # Double loop as in the MATLAB code
        for i in range(N - 1):
            for j in range(i + 1, N):
                delta = X[:, i] - X[:, j]         # shape (n,)
                val = delta.T @ W @ delta        # scalar
                Kxx[i, j] = val
                Kxx[j, i] = val

        Kxx = np.exp(-(0.5 / n) * Kxx)

        return Kxx

    else:
        raise ValueError(f"Unknown kernel type: {kerneltype!r}")