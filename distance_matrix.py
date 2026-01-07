import numpy as np

def distance_matrix(A, B):
    """
    Python version of:
        function [distMat] = DistanceMatrix(A,B)

    Computes the squared Euclidean distance matrix between rows of A and B.

    Parameters
    ----------
    A : array_like, shape (numA, d)
    B : array_like, shape (numB, d)

    Returns
    -------
    distMat : ndarray, shape (numA, numB)
        Squared distances between each row of A and each row of B.
        If A and B have different numbers of columns, returns an empty (0, 0) array.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    numA, d = A.shape
    numB, dB = B.shape

    if dB != d:
        return np.empty((0, 0))

    # Vectorized equivalent of the MATLAB helpA/helpB trick:
    # dist(i,j) = sum_k (A_ik - B_jk)^2
    A_sq = np.sum(A**2, axis=1, keepdims=True)        # (numA, 1)
    B_sq = np.sum(B**2, axis=1, keepdims=True).T      # (1, numB)
    distMat = A_sq + B_sq - 2 * (A @ B.T)             # (numA, numB)

    return distMat