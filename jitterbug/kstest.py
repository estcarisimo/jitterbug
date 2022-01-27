import numpy as np
import scipy.stats


ALPHA = 0.05

class kstest:

    def __init__(self, epoch, jitter, change_points):
        self.epoch = epoch
        self.jitter = jitter
        self.change_points = change_points

    def __get_signal_slice(self, i):
        """Helper function"""
        idx = (self.epoch > self.change_points[i - 1]) & (self.epoch <= self.change_points[i])
        return self.jitter[idx]


    def __ks2wrapper(self, x, y):
        try:
            stat, pvalue = scipy.stats.ks_2samp(x, y)
            return stat, pvalue
        except:
            return np.nan, np.nan

    def fit(self):

        self.change_jitter_regime = []


        for i in range(1, len(self.change_points) - 1):

            j1 = self.__get_signal_slice(i)
            j2 = self.__get_signal_slice(i + 1)

            if len(j1) > 0 and len(j2) > 0:
                stat, pvalue12 = self.__ks2wrapper(j1, j2)

                self.change_jitter_regime.append((self.change_points[i], self.change_points[i + 1], pvalue12 < ALPHA)) 

        

    def getKSTestResults(self):
        return np.array(self.change_jitter_regime)