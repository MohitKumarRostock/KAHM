import numpy as np
from distance_matrix import distance_matrix  # Python version of DistanceMatrix.m


def kxa_matrix(x_matrix, a_matrix, sqrt_weights_matrix, kerneltype="gaussian", *, row_shift=False):
    """
    Python version of:
        function [Kxa] = KxaMatrix(x_matrix, a_matrix, sqrt_weights_matrix, kerneltype)

    Parameters
    ----------
    x_matrix : ndarray, shape (n, N)
        Data matrix X (columns are samples).
    a_matrix : ndarray, shape (n, M)
        Data matrix A (columns are samples).
    sqrt_weights_matrix : ndarray, shape (n, n)
        Square root of the weight (precision) matrix.
    kerneltype : str, optional
        Kernel type, currently only 'gaussian' supported.
    row_shift : bool, optional
        If True, subtract each row's largest log-kernel value before taking
        exponentials. Use only where a later row normalization cancels this
        common factor. The default returns the original Gaussian kernel.

    Returns
    -------
    Kxa : ndarray, shape (N, M)
        Kernel matrix between x_matrix and a_matrix.
    """
    kerneltype = kerneltype.lower()

    if kerneltype == "gaussian":
        X = np.asarray(x_matrix, dtype=float)
        A = np.asarray(a_matrix, dtype=float)
        S = np.asarray(sqrt_weights_matrix, dtype=float)

        if X.ndim != 2 or A.ndim != 2 or X.shape[0] == 0 or X.shape[0] != A.shape[0]:
            raise ValueError("Queries and references must have the same positive feature dimension.")
        n, _ = X.shape
        if S.shape != (n, n) or A.shape[1] == 0:
            raise ValueError("Expected an n-by-n precision square root and nonempty references.")

        # Wx = sqrt_weights_matrix * x_matrix
        # Wa = sqrt_weights_matrix * a_matrix
        Wx = S @ X
        Wa = S @ A

        # DistanceMatrix(Wx', Wa') in MATLAB → distance_matrix(Wx.T, Wa.T) in Python
        dist_mat = distance_matrix(Wx.T, Wa.T)

        # Kxa = exp(-(0.5/n) * distMat);
        log_kernel = -(0.5 / n) * dist_mat
        if row_shift:
            if not np.isfinite(log_kernel).all():
                raise FloatingPointError("Nonfinite kernel distances; rescale the input features.")
            log_kernel -= log_kernel.max(axis=1, keepdims=True)
        Kxa = np.exp(log_kernel)

        return Kxa

    else:
        raise ValueError(f"Unknown kernel type: {kerneltype!r}")
