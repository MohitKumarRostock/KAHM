import numpy as np
from distance_matrix import distance_matrix  # Python version of DistanceMatrix.m


def kxa_matrix(x_matrix, a_matrix, sqrt_weights_matrix, kerneltype="gaussian"):
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

        n, _ = X.shape

        # Wx = sqrt_weights_matrix * x_matrix
        # Wa = sqrt_weights_matrix * a_matrix
        Wx = S @ X
        Wa = S @ A

        # DistanceMatrix(Wx', Wa') in MATLAB → distance_matrix(Wx.T, Wa.T) in Python
        dist_mat = distance_matrix(Wx.T, Wa.T)

        # Kxa = exp(-(0.5/n) * distMat);
        Kxa = np.exp(-(0.5 / n) * dist_mat)

        return Kxa

    else:
        raise ValueError(f"Unknown kernel type: {kerneltype!r}")