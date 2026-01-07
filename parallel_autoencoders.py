import numpy as np
from joblib import Parallel, delayed

from autoencoder import autoencoder  # Python version of Autoencoder.m


def _fit_cluster_autoencoder(y_data, indices, subspace_dim):
    """Train one autoencoder on a subset of columns."""
    return autoencoder(y_data[:, indices], subspace_dim)


def parallel_autoencoders(
    y_data: np.ndarray,
    subspace_dim: int,
    Nb: int,
    n_jobs: int = 1,
    verbose: bool = False,
):
    """
    Python version of:
        function [AE_arr] = parallelAutoencoders(y_data, subspace_dim, Nb)

    Parameters
    ----------
    y_data : ndarray, shape (D, N)
        Data matrix with N samples (columns) and D features (rows).
    subspace_dim : int
        Target subspace dimension passed to each autoencoder.
    Nb : int
        Block size parameter used to determine number of clusters.
        Roughly N_clusters ≈ round(N / Nb).
    n_jobs : int, optional
        Number of parallel jobs (CPU cores) over clusters.
        1  -> no parallelism (serial, current behaviour)
        -1 -> use all cores
    verbose : bool, optional
        If True, prints basic info.

    Returns
    -------
    AE_arr : list
        List of AE objects (dicts if using the previous autoencoder() implementation).
    """
    y_data = np.ascontiguousarray(y_data, dtype=np.float32)
    D, N = y_data.shape  # noqa: F841 (D is not used but kept for clarity)

    if Nb <= 0:
        raise ValueError("Nb must be positive.")

    # N_clusters = round(N / Nb);  (MATLAB round: half up)
    N_clusters = int(np.floor(N / Nb + 0.5))
    N_clusters = max(1, N_clusters)

    if verbose:
        print(f"parallel_autoencoders: N={N}, Nb={Nb} -> N_clusters={N_clusters}")

    # If only one cluster, just train a single autoencoder
    if N_clusters <= 1:
        return [autoencoder(y_data, subspace_dim)]

    # --- Vectorised cluster assignment (equivalent logic, less Python loop) ---
    # q = floor(N / N_clusters), r = mod(N, N_clusters)
    q = N // N_clusters
    r = N % N_clusters

    # First r clusters have (q + 1) samples, the rest have q samples
    counts = np.full(N_clusters, q, dtype=int)
    counts[:r] += 1

    # Split indices 0..N-1 into clusters according to counts
    splits = np.cumsum(counts)[:-1]
    all_indices = np.arange(N, dtype=int)
    cluster_indices_list = np.split(all_indices, splits)  # list of 1D index arrays

    if verbose:
        sizes = [idx.size for idx in cluster_indices_list]
        print("Cluster sizes:", sizes)

    # --- Train one autoencoder per cluster ---
    if n_jobs == 1:
        # Serial (no nested parallelism if classifier already uses n_jobs)
        AE_arr = [
            _fit_cluster_autoencoder(y_data, indices, subspace_dim)
            for indices in cluster_indices_list
        ]
    else:
        # Parallel over clusters
        AE_arr = Parallel(n_jobs=n_jobs)(
            delayed(_fit_cluster_autoencoder)(y_data, indices, subspace_dim)
            for indices in cluster_indices_list
        )

    return AE_arr