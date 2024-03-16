import numpy as np
import scipy.stats

ALPHA = 0.05

class KSTest:
    """
    A class to perform the Kolmogorov-Smirnov test (KS test) for detecting changes in jitter distribution.

    Attributes
    ----------
    epoch : numpy array
        Array of timestamps for each jitter measurement.
    jitter : numpy array
        Array of jitter measurements.
    change_points : numpy array
        Array of timestamps where potential change points in the jitter distribution are located.
    change_jitter_regime : list
        List of tuples indicating the start and end of a change in jitter distribution and whether it's significant.

    Methods
    -------
    fit():
        Executes the KS test on the jitter data to find significant changes in distribution between epochs.
    getKSTestResults():
        Returns the results of the KS test analysis.
    """

    def __init__(self, epoch, jitter, change_points):
        """
        Initializes the KSTest class with epoch data, jitter measurements, and potential change points.

        Parameters
        ----------
        epoch : numpy array
            Array of timestamps for each jitter measurement.
        jitter : numpy array
            Array of jitter measurements.
        change_points : numpy array
            Array of timestamps where potential change points in the jitter distribution are located.
        """
        self.epoch = epoch
        self.jitter = jitter
        self.change_points = change_points
        self.change_jitter_regime = []

    def __get_signal_slice(self, i):
        """
        Helper function to extract slices of the jitter signal between consecutive change points.

        Parameters
        ----------
        i : int
            Index to specify the section of the signal to retrieve based on change points.

        Returns
        -------
        numpy array
            Slice of the jitter array between the ith and (i+1)th change points.
        """
        idx = (self.epoch > self.change_points[i - 1]) & (self.epoch <= self.change_points[i])
        return self.jitter[idx]

    def __ks2wrapper(self, x, y):
        """
        Wrapper function for performing a two-sample Kolmogorov-Smirnov test.

        Parameters
        ----------
        x, y : numpy array
            Data samples to compare.

        Returns
        -------
        tuple
            The KS statistic and p-value of the test. Returns (np.nan, np.nan) if the test fails.
        """
        try:
            stat, pvalue = scipy.stats.ks_2samp(x, y)
            return stat, pvalue
        except:
            return np.nan, np.nan

    def fit(self):
        """
        Executes the KS test on the jitter data to find significant changes in distribution between epochs.
        """
        for i in range(1, len(self.change_points) - 1):
            j1 = self.__get_signal_slice(i)
            j2 = self.__get_signal_slice(i + 1)

            if len(j1) > 0 and len(j2) > 0:
                _, pvalue12 = self.__ks2wrapper(j1, j2)
                self.change_jitter_regime.append((self.change_points[i], self.change_points[i + 1], pvalue12 < ALPHA))

    def getKSTestResults(self):
        """
        Returns the results of the KS test analysis.

        Returns
        -------
        numpy array
            An array of tuples indicating the start and end of a change in jitter distribution and whether it's significant.
        """
        return np.array(self.change_jitter_regime)
