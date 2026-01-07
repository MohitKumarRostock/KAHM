import numpy as np

def kernel_regularized_least_squares(KernelMatrix, LabelsMatrix,
                                     tol=1e-3, max_iter=100, verbose=False):
    """
    Python version of:
        function [B] = kernel_regularized_least_squares(KernelMatrix, LabelsMatrix)

    Parameters
    ----------
    KernelMatrix : (N, N) array_like
        Kernel / Gram matrix.
    LabelsMatrix : (M, N) array_like
        Label matrix; columns correspond to the same samples as KernelMatrix.
    tol : float, optional
        Convergence tolerance for the change in lambda (default: 1e-3).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    verbose : bool, optional
        If True, prints iteration info (like MATLAB fprintf).

    Returns
    -------
    B : (N, N) ndarray
        Inverse of (lambda * I + KernelMatrix) at convergence.
    """

    K = np.asarray(KernelMatrix, dtype=float)
    Y = np.asarray(LabelsMatrix, dtype=float)

    # N = size(LabelsMatrix,2);
    N = Y.shape[1]

    # tv0 = numel(LabelsMatrix);
    tv0 = Y.size

    # tv1 = sum(sum(LabelsMatrix.^2,2))/tv0;
    tv1 = np.sum(Y ** 2) / tv0

    # tau = 2*tv1; e = 0.5*tv1; lambda = tau + e;
    tau = 2.0 * tv1
    e = 0.5 * tv1
    lam = tau + e

    lam_prev = lam
    lam_chg = 1.0
    itr_count = 0

    B = None  # will be set in the loop

    # while (lambda_chg > 0.001) && (itr_count < 100)
    while (lam_chg > tol) and (itr_count < max_iter):
        # B = inv(lambda*eye(N) + KernelMatrix);
        B = np.linalg.inv(lam * np.eye(N) + K)

        # temp_matrix = KernelMatrix*B;
        temp_matrix = K @ B

        # tv = sum(sum((LabelsMatrix-LabelsMatrix*temp_matrix').^2,2));
        residual = Y - Y @ temp_matrix.T
        tv = np.sum(residual ** 2)

        # lambda = tau + (tv/tv0);
        lam = tau + (tv / tv0)

        # lambda_chg = abs(lambda-lambda_ini);
        lam_chg = abs(lam - lam_prev)
        lam_prev = lam

        itr_count += 1

    if verbose:
        print(f"iterations = {itr_count}, regularization = {lam:.6f}, N = {N}.")

    return B