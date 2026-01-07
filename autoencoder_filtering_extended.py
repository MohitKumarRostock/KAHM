import numpy as np
from kxa_matrix import kxa_matrix  # Python version of KxaMatrix.m


def autoencoder_filtering_extended(y_data_new, AE, distance_type):
    """
    Python version of:
        function [hat_y_data,distance] = AutoencoderFilteringExtended(y_data_new,AE,distance_type)

    Parameters
    ----------
    y_data_new : ndarray, shape (D, N_new)
        New data (columns are samples).
    AE : dict
        Autoencoder object as returned by `autoencoder()`:
        keys: 'PC', 'y_data_n_subspace', 'sqrt_weights_matrix',
              'kerneltype', 'B', 'y_data'.
    distance_type : {'folding','product','minimum','maximum','cosine'}
        Distance combination type.

    Returns
    -------
    hat_y_data : ndarray, shape (D, N_new)
        Reconstructed data.
    distance : ndarray, shape (N_new,)
        Distance for each sample.
    """
    y_data_new = np.asarray(y_data_new, dtype=float)

    # Project new data into subspace: x_data = (AE.PC)' * y_data_new
    PC = AE["PC"]
    x_data = PC.T @ y_data_new  # shape: (subspace_dim, N_new)
    N = x_data.shape[1]

    D = AE["y_data"].shape[0]

    # Reconstruct hat_y_data in chunks if N > 1000
    if N > 1000:
        hat_y_data = np.zeros((D, N), dtype=float)

        for start in range(0, N, 1000):
            end = min(N, start + 1000)

            Kxa = kxa_matrix(
                x_data[:, start:end],
                AE["y_data_n_subspace"],
                AE["sqrt_weights_matrix"],
                AE["kerneltype"],
            )  # shape: (N_chunk, N_train)

            weights = Kxa @ AE["B"]  # (N_chunk, N_train)

            sum_weights = weights.sum(axis=1, keepdims=True)  # (N_chunk, 1)
            # Avoid division by zero: if sum_weights == 0, keep the row as zeros.
            sum_weights_safe = sum_weights.copy()
            sum_weights_safe[sum_weights_safe == 0] = 1.0
            weights = weights / sum_weights_safe

            hat_y_data[:, start:end] = AE["y_data"] @ weights.T  # (D, N_chunk)
    else:
        Kxa = kxa_matrix(
            x_data,
            AE["y_data_n_subspace"],
            AE["sqrt_weights_matrix"],
            AE["kerneltype"],
        )  # (N_new, N_train)

        weights = Kxa @ AE["B"]  # (N_new, N_train)
        sum_weights = weights.sum(axis=1, keepdims=True)  # (N_new, 1)
        # Avoid division by zero: if sum_weights == 0, keep the row as zeros.
        sum_weights_safe = sum_weights.copy()
        sum_weights_safe[sum_weights_safe == 0] = 1.0
        weights = weights / sum_weights_safe

        hat_y_data = AE["y_data"] @ weights.T  # (D, N_new)

    # ---------- Distances ----------
    distance_type = distance_type.lower()

    # Column-wise Euclidean norms (like vecnorm(...,2) in MATLAB)
    diff = hat_y_data - y_data_new
    distance_1 = 1.0 - np.exp(-np.linalg.norm(diff, axis=0))

    # Cosine-based distance
    dot = np.sum(hat_y_data * y_data_new, axis=0)
    norm_hat = np.linalg.norm(hat_y_data, axis=0)
    norm_new = np.linalg.norm(y_data_new, axis=0)

    denom = norm_hat * norm_new
    # Safe cosine similarity: avoid division by zero
    cos_sim = np.zeros_like(dot)
    mask = denom != 0
    cos_sim[mask] = dot[mask] / denom[mask] # For denom == 0, cos_sim stays 0.
    # Avoid tiny numerical issues outside [-1, 1]
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    distance_2 = np.arccos(cos_sim) / np.pi

    if distance_type == "folding":
        distance = np.sqrt(distance_1**2 + distance_2**2) / np.sqrt(2.0)
    elif distance_type == "product":
        distance = distance_1 * distance_2
    elif distance_type == "minimum":
        distance = np.minimum(distance_1, distance_2)
    elif distance_type == "maximum":
        distance = np.maximum(distance_1, distance_2)
    elif distance_type == "cosine":
        distance = distance_2
    else:
        raise ValueError(f"Unknown distance_type: {distance_type!r}")

    return hat_y_data, distance