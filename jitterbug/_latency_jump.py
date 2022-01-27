import numpy as np

class latencyJumps:

    def __init__(self, epoch, rtt, change_points):
        self.epoch = epoch
        self.rtt = rtt
        self.change_points = change_points

    def __get_signal_slice(self, i):
        """Helper function"""
        idx = (self.epoch > self.change_points[i - 1]) & (self.epoch <= self.change_points[i])
        return self.rtt[idx]

    def fit(self, threshold=0.5):

        self.jumps = []


        for i in range(1, len(self.change_points) - 1):

            rtt1 = self.__get_signal_slice(i)
            rtt2 = self.__get_signal_slice(i + 1)

            if len(rtt1) > 0 and len(rtt1) > 0:
                jump =  np.mean(rtt2) > (np.mean(rtt1) + threshold)
                 
                self.jumps.append((self.change_points[i], self.change_points[i + 1], jump)) 

    def getLatencyJumps(self):
        return np.array(self.jumps)