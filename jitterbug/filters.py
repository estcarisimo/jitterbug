import numpy as np


def moving_average(x, w):
    """
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_iqr_filter_symmetric(x, k):
    """
    FILL.

    input
    ----------
    day: FILL

    return
    ----------
    X: FILL
    """
    
    iqr = []
    
    for i in range(k, len(x) - k):
        q1, q3 = np.percentile(x[i - k: i + k], [25, 75])
        iqr.append(q3 - q1)
    
    return np.array(iqr)