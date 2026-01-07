import numpy as np
from autoencoder_filtering_extended import autoencoder_filtering_extended
# ^ Python version of AutoencoderFilteringExtended.m


def combine_multiple_autoencoders_extended(y_data_new, AE_arr, distance_type):
    """
    Python version of:
        function [distance] = combineMultipleAutoencodersExtended(y_data_new, AE_arr, distance_type)

    Parameters
    ----------
    y_data_new : ndarray, shape (D, N)
        New data samples (columns are samples).
    AE_arr : list
        List of autoencoder objects (dicts if you use the earlier `autoencoder` implementation).
    distance_type : str
        Distance type passed to `autoencoder_filtering_extended`
        (e.g. 'folding', 'product', 'minimum', 'maximum', 'cosine').

    Returns
    -------
    distance : ndarray, shape (N,)
        Minimal distance over all autoencoders for each sample.
    """
    y_data_new = np.asarray(y_data_new, dtype=float)

    D, N = y_data_new.shape
    Q = len(AE_arr)

    distance_matrix = np.zeros((Q, N), dtype=float)

    for i in range(Q):
        # Ignore reconstructed data, keep only distances
        _, dist = autoencoder_filtering_extended(y_data_new, AE_arr[i], distance_type)
        distance_matrix[i, :] = dist

    # Minimum over autoencoders (rows), per sample (column)
    distance = np.min(distance_matrix, axis=0)

    return distance