import numpy as np

class LatencyJumps:
    """
    A class to identify significant jumps in latency over time within a given dataset.

    Attributes
    ----------
    epoch : numpy array
        Array of epoch timestamps for the round-trip time (RTT) measurements.
    rtt : numpy array
        Array of round-trip time (RTT) measurements.
    change_points : numpy array
        Array of timestamps indicating potential change points in the latency data.

    Methods
    -------
    fit(threshold=0.5):
        Analyzes the RTT data to find significant jumps in latency.
    getLatencyJumps():
        Returns the significant latency jumps identified.
    """

    def __init__(self, epoch, rtt, change_points):
        """
        Initializes the LatencyJumps class with the necessary time series data and change points.

        Parameters
        ----------
        epoch : numpy array
            Array of epoch timestamps for the RTT measurements.
        rtt : numpy array
            Array of RTT measurements.
        change_points : numpy array
            Array of timestamps indicating potential change points in the latency data.
        """
        self.epoch = epoch
        self.rtt = rtt
        self.change_points = change_points
        self.jumps = []

    def __get_signal_slice(self, i):
        """
        Helper function to extract slices of the RTT signal between consecutive change points.

        Parameters
        ----------
        i : int
            Index to specify the section of the signal to retrieve based on change points.

        Returns
        -------
        numpy array
            Slice of the RTT array between the ith and (i+1)th change points.
        """
        idx = (self.epoch > self.change_points[i - 1]) & (self.epoch <= self.change_points[i])
        return self.rtt[idx]

    def fit(self, threshold=0.5):
        """
        Analyzes the RTT data to identify significant jumps in latency between change points.

        Parameters
        ----------
        threshold : float, optional
            The value above which a difference between consecutive RTT measurements is considered a jump. Default is 0.5.
        """
        for i in range(1, len(self.change_points) - 1):
            rtt1 = self.__get_signal_slice(i)
            rtt2 = self.__get_signal_slice(i + 1)

            if len(rtt1) > 0 and len(rtt2) > 0:
                jump = np.mean(rtt2) > (np.mean(rtt1) + threshold)
                self.jumps.append((self.change_points[i], self.change_points[i + 1], jump))

    def getLatencyJumps(self):
        """
        Returns the significant latency jumps identified after analyzing the RTT data.

        Returns
        -------
        numpy array
            An array of tuples containing the start and end points of the significant jumps and a boolean indicating the presence of a jump.
        """
        return np.array(self.jumps)
