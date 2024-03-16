import jitterbug.preprocessing as preproc
from jitterbug.signal_energy import JitterDispersion
from jitterbug.kstest import KSTest

def compute_jitter_dispersion(change_points, epoch_mins, mins, K, L, jitter_dispersion_threshold=0.25):
    """
    Computes the jitter dispersion values and identifies significant changes in the jitter signal.

    Parameters
    ----------
    change_points : numpy array
        Array of potential change points in the jitter signal.
    epoch_mins : numpy array
        Epoch timestamps corresponding to the minimum RTT measurements.
    mins : numpy array
        Minimum RTT measurements.
    K : int
        Parameter defining the window size for the IQR calculation.
    L : int
        Parameter defining the window size for the moving average calculation.
    jitter_dispersion_threshold : float, optional
        Threshold value for determining significant changes in jitter dispersion. Default is 0.25.

    Returns
    -------
    numpy array
        Array of jitter dispersion values indicating significant changes.
    """
    # Preprocessing to compute jitter dispersion
    epoch_jitter_dispersion, jitter_dispersion = preproc.jitter_dispersion(epoch_mins, mins, K, L)

    # Compute jitter dispersion changes
    jd = JitterDispersion(epoch_jitter_dispersion, jitter_dispersion, change_points)
    jd.fit(jitter_dispersion_threshold)

    return jd.getJitterDispersionValues()

def compute_ks_test(change_points, epoch_rtt, rtt):
    """
    Computes the Kolmogorov-Smirnov (KS) test to detect significant changes in the jitter distribution.

    Parameters
    ----------
    change_points : numpy array
        Array of potential change points in the jitter distribution.
    epoch_rtt : numpy array
        Epoch timestamps corresponding to the RTT measurements.
    rtt : numpy array
        Round Trip Time (RTT) measurements.

    Returns
    -------
    numpy array
        Array of results from the KS test indicating significant changes in the jitter distribution.
    """
    # Preprocessing to compute jitter values
    epoch_jitter, jitter = preproc.jitter(epoch_rtt, rtt)

    # Perform KS test
    ks = KSTest(epoch_jitter, jitter, change_points)
    ks.fit()
    
    return ks.getKSTestResults()
