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

        if X.ndim != 2 or X.shape[0] == 0 or W.shape != (X.shape[0], X.shape[0]):
            raise ValueError("Expected X with shape (n, N), n > 0, and weights with shape (n, n).")
        n, N = X.shape
        Kxx = np.empty((N, N), dtype=float)

        # Same delta.T @ W @ delta equation as the original pair loops. Chunking
        # removes the Python loop over individual pairs without allocating an
        # n-by-N-by-N temporary or subtracting large squared norms.
        chunk_size = max(1, min(N, (8 * 1024 * 1024) // max(8 * n * N, 1)))
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            delta = X[:, start:end, None] - X[:, None, :]
            Kxx[start:end] = np.einsum("ibj,ik,kbj->bj", delta, W, delta, optimize=True)
        np.maximum(Kxx, 0.0, out=Kxx)
        np.fill_diagonal(Kxx, 0.0)
        Kxx *= -(0.5 / n)
        np.exp(Kxx, out=Kxx)
        return Kxx

    else:
        raise ValueError(f"Unknown kernel type: {kerneltype!r}")
