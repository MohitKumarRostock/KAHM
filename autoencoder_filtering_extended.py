import numpy as np
import warnings
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

    Notes
    -----
    Gaussian query rows are rescaled before affine-weight normalization to
    prevent all-zero underflow. This common factor cancels algebraically; it
    does not resolve cancellation between signed weights. A near-zero signed
    normalizer emits RuntimeWarning. For an exactly zero normalizer, the
    historical divisor-of-one convention is retained, so that row is not a
    normalized affine reconstruction. Identical-reference banks reconstruct
    their constant reference point.
    """
    y_data_new = np.asarray(y_data_new, dtype=float)
    if y_data_new.ndim != 2 or not np.isfinite(y_data_new).all():
        raise ValueError("y_data_new must be a finite matrix shaped (features, samples).")

    # Project new data into subspace: x_data = (AE.PC)' * y_data_new
    PC = AE["PC"]
    x_data = PC.T @ y_data_new  # shape: (subspace_dim, N_new)
    N = x_data.shape[1]

    D = AE["y_data"].shape[0]

    if y_data_new.shape[0] != D:
        raise ValueError("Query feature count does not match the fitted reference bank.")

    if "constant_reconstruction" in AE:
        hat_y_data = np.broadcast_to(AE["constant_reconstruction"], (D, N)).copy()
    else:
        hat_y_data = np.empty((D, N), dtype=float)
        for start in range(0, N, 1000):
            end = min(N, start + 1000)

            Kxa = kxa_matrix(
                x_data[:, start:end],
                AE["y_data_n_subspace"],
                AE["sqrt_weights_matrix"],
                AE["kerneltype"],
                row_shift=True,
            )  # shape: (N_chunk, N_train)

            weights = Kxa @ AE["B"]  # (N_chunk, N_train)

            sum_weights = weights.sum(axis=1, keepdims=True)  # (N_chunk, 1)
            # Detect signed cancellation separately from exponent underflow.
            scale = np.abs(weights).sum(axis=1, keepdims=True)
            threshold = np.finfo(float).eps * max(1, weights.shape[1]) * scale
            if np.any(np.abs(sum_weights) <= threshold):
                warnings.warn(
                    "Near-zero signed KAHM weight normalizer; reconstruction may be unstable. "
                    "Exactly zero normalizers retain the legacy divisor-of-one convention.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            sum_weights_safe = sum_weights.copy()
            sum_weights_safe[sum_weights_safe == 0] = 1.0
            weights = weights / sum_weights_safe

            hat_y_data[:, start:end] = AE["y_data"] @ weights.T  # (D, N_chunk)

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
