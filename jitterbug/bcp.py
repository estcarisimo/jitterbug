"""
"""

import numpy as np
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.offline_likelihoods import IndepentFeaturesLikelihood
from functools import partial

BCP_MIN_SAMPLES = 1
OFFCD_TRUNCATION = -40

class bcp:

    def __init__(self, epoch, data, min_time_elapsed=1800, cpd_threshold=0.25):

        if len(epoch) == len(data):
            if len(data) > BCP_MIN_SAMPLES:
                # usar para los umbral con decorator
                self.epoch = epoch
                self.data = data
                self.min_time_elapsed = min_time_elapsed
                self.cpd_threshold = cpd_threshold
            else:
                raise Exception("Insufficient samples (fewer than 2) \
                                 to compute change point detection")
        else:
            raise Exception("epochs and rtts have different lengths.")

    def __bcp(self):
        """
        Thi function detects change points in a time series.

        We apply BCP to detect changepoints in a time series.
        https://github.com/hildensia/bayesian_changepoint_detection
        """
        Q, P, Pcp = offline_changepoint_detection(
            self.data,
            partial(const_prior, p=1.0 / (len(self.data) + 1)),
            IndepentFeaturesLikelihood(),
            truncate=OFFCD_TRUNCATION
        )
        
        # Transform results
        self.bcp_t = self.epoch[1:]
        self.bcp_vals = np.exp(Pcp).sum(0)

    def __apply_cdp_th(self):

        self.candidate_cp_t = self.bcp_t[self.bcp_vals > self.cpd_threshold]
        self.candidate_cp_t = np.sort(self.candidate_cp_t)

    def __find_cps(self):

        self.cps = []
        self.cps.append(self.candidate_cp_t[0])

        for i in range(len(self.candidate_cp_t)):
            if i > 0 and (self.candidate_cp_t[i] - self.candidate_cp_t[i - 1]) > self.min_time_elapsed:
                self.cps.append(self.candidate_cp_t[i])
                
        self.cps.append(self.candidate_cp_t[-1])


    def fit(self):
        
        self.__bcp()
        self.__apply_cdp_th()
        self.__find_cps()
    

    def getChangePoints(self):
        return self.cps
