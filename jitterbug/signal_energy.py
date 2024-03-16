import numpy as np

class JitterDispersion:
    """
    A class for analyzing jitter dispersion to identify significant changes over time.

    Attributes
    ----------
    epoch : numpy array
        Array of epoch timestamps for the jitter dispersion measurements.
    jitter_dispersion : numpy array
        Array of jitter dispersion values.
    change_points : numpy array
        Array of timestamps indicating potential change points in the jitter dispersion.

    Methods
    -------
    fit(threshold=0.25):
        Analyzes jitter dispersion changes using the specified threshold to identify significant changes.
    getJitterDispersionValues():
        Returns the identified significant changes in jitter dispersion.
    """

    def __init__(self, epoch, jitter_dispersion, change_points):
        """
        Initializes the JitterDispersion class with epoch data, jitter dispersion measurements, and potential change points.

        Parameters
        ----------
        epoch : numpy array
            Array of epoch timestamps for the jitter dispersion measurements.
        jitter_dispersion : numpy array
            Array of jitter dispersion values.
        change_points : numpy array
            Array of timestamps indicating potential change points in the jitter dispersion.
        """
        self.epoch = epoch
        self.jitter_dispersion = jitter_dispersion
        self.change_points = change_points
        self.jitter_dispersion_mean_values = []

    def __get_signal_slice(self, i):
        """
        Helper function to extract slices of the jitter dispersion signal between consecutive change points.

        Parameters
        ----------
        i : int
            Index to specify the section of the signal to retrieve based on change points.

        Returns
        -------
        numpy array
            Slice of the jitter dispersion array between the ith and (i+1)th change points.
        """
        idx = (self.epoch > self.change_points[i - 1]) & (self.epoch <= self.change_points[i])
        return self.jitter_dispersion[idx]

    def fit(self, threshold=0.25):
        """
        Analyzes jitter dispersion changes using the specified threshold to identify significant changes.

        Parameters
        ----------
        threshold : float, optional
            The threshold value used to determine significant changes in jitter dispersion. Default is 0.25.
        """
        for i in range(1, len(self.change_points) - 1):
            j1 = self.__get_signal_slice(i)
            j2 = self.__get_signal_slice(i + 1)

            if len(j1) > 0 and len(j2) > 0:
                mean_increase = np.mean(j2) > (np.mean(j1) + threshold)
                self.jitter_dispersion_mean_values.append(
                    (self.change_points[i], self.change_points[i + 1], mean_increase)
                )

    def getJitterDispersionValues(self):
        """
        Returns the identified significant changes in jitter dispersion.

        Returns
        -------
        numpy array
            An array of tuples containing the start and end points of the significant changes and a boolean
            indicating whether there is an increase in jitter dispersion.
        """
        return np.array(self.jitter_dispersion_mean_values)
