import numpy as np

class jitterDispersion:

    def __init__(self, epoch, jitter_dispersion, change_points):
        self.epoch = epoch
        self.jitter_dispersion = jitter_dispersion
        self.change_points = change_points

    def __get_signal_slice(self, i):
        """Helper function"""
        idx = (self.epoch > self.change_points[i - 1]) & (self.epoch <= self.change_points[i])
        return self.jitter_dispersion[idx]

    def fit(self, threshold=0.25):

        self.jitter_dispersion_mean_values = []


        for i in range(1, len(self.change_points) - 1):

            j1 = self.__get_signal_slice(i)
            j2 = self.__get_signal_slice(i + 1)

            if len(j1) > 0 and len(j2) > 0:

                self.jitter_dispersion_mean_values.append(
                    (
                        self.change_points[i],
                        self.change_points[i + 1],
                        np.mean(j2) > (np.mean(j1) + threshold)
                    )
                ) 
    
    def getJitterDispersionValues(self):
        return np.array(self.jitter_dispersion_mean_values)

