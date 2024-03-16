import numpy as np
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
from bayesian_changepoint_detection.priors import const_prior
import bayesian_changepoint_detection.offline_likelihoods as offline_ll
from functools import partial

BCP_MIN_SAMPLES = 1
OFFCD_TRUNCATION = -40

class BCP:
    """
    Bayesian Change Point (BCP) detection class for identifying change points in time series data.

    Attributes
    ----------
    epoch : numpy array
        Timestamps of the data points.
    data : numpy array
        Data points of the time series.
    min_time_elapsed : int
        Minimum time between two change points.
    cpd_threshold : float
        Threshold for determining significant change points.

    Methods
    -------
    fit():
        Executes the BCP algorithm to find change points in the data.
    getChangePoints():
        Returns the detected change points.
    """

    def __init__(self, epoch, data, min_time_elapsed=1800, cpd_threshold=0.25):
        """
        Initializes the BCP class with time series data and parameters for change point detection.

        Parameters
        ----------
        epoch : numpy array
            Timestamps of the data points.
        data : numpy array
            Data points of the time series.
        min_time_elapsed : int, optional
            Minimum time between two change points, default is 1800.
        cpd_threshold : float, optional
            Threshold for determining significant change points, default is 0.25.

        Raises
        ------
        Exception
            If the epoch and data lengths differ, or there are insufficient samples for change point detection.
        """
        if len(epoch) != len(data):
            raise Exception("Epochs and data have different lengths.")
        if len(data) <= BCP_MIN_SAMPLES:
            raise Exception("Insufficient samples to compute change point detection.")

        self.epoch = epoch
        self.data = data
        self.min_time_elapsed = min_time_elapsed
        self.cpd_threshold = cpd_threshold

    def __bcp(self):
        """
        Private method to detect change points in the time series using Bayesian Change Point (BCP) detection.
        """
        prior_function = partial(const_prior, p=1 / (len(self.data) + 1))
        _, _, Pcp = offline_changepoint_detection(
            self.data, 
            prior_function,
            offline_ll.StudentT(), 
            truncate=OFFCD_TRUNCATION
        )

        self.bcp_t = self.epoch[1:]
        self.bcp_vals = np.exp(Pcp).sum(0)

    def __apply_cdp_th(self):
        """
        Private method to apply the change point detection threshold to the computed BCP values.
        """
        self.candidate_cp_t = self.bcp_t[self.bcp_vals > self.cpd_threshold]
        self.candidate_cp_t = np.sort(self.candidate_cp_t)

    def __find_cps(self):
        """
        Private method to finalize the change point detection process, finding the actual change points.
        """
        self.cps = []

        if len(self.candidate_cp_t) > 0:
            self.cps.append(self.candidate_cp_t[0])

            for i in range(1, len(self.candidate_cp_t)):
                if (self.candidate_cp_t[i] - self.candidate_cp_t[i - 1]) > self.min_time_elapsed:
                    self.cps.append(self.candidate_cp_t[i])

            self.cps.append(self.candidate_cp_t[-1])

    def fit(self):
        """
        Executes the Bayesian Change Point detection process to identify significant changes in the time series data.
        """
        self.__bcp()
        self.__apply_cdp_th()
        self.__find_cps()

    def getChangePoints(self):
        """
        Returns the detected change points after executing the BCP algorithm.

        Returns
        -------
        list
            A list of timestamps where significant change points have been detected.
        """
        return self.cps
