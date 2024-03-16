import numpy as np

def moving_average(x, w):
    """
    Compute the moving average of a given data array.

    Parameters
    ----------
    x : numpy array
        The input data array to compute the moving average on.
    w : int
        The window size for the moving average.

    Returns
    -------
    numpy array
        The moving average of the input data array over the specified window size.

    Notes
    -----
    This function computes the moving average using a convolution approach, which
    can be more efficient than a straightforward implementation for large datasets.
    The 'valid' mode in `np.convolve` ensures that the returned moving average
    only contains values where the window is fully within the input array.
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_iqr_filter_symmetric(x, k):
    """
    Calculate the interquartile range (IQR) of the data, using a symmetric window.

    Parameters
    ----------
    x : numpy array
        The input data array to compute the IQR on.
    k : int
        The half window size. The total window size will be `2*k`, centered around each point.

    Returns
    -------
    numpy array
        The IQR values of the input data array, computed over a window of size `2*k` for each point.

    Notes
    -----
    The interquartile range (IQR) is the difference between the 75th and 25th percentiles of the data.
    This function computes the IQR in a rolling fashion, where for each point in the data array,
    a window of size `2*k` is used to calculate the IQR.
    """
    iqr = []

    for i in range(k, len(x) - k):
        window = x[i - k: i + k + 1]  # Include the element at position i
        q1, q3 = np.percentile(window, [25, 75])
        iqr.append(q3 - q1)

    return np.array(iqr)
