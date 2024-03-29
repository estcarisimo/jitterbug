import numpy as np

class CongestionInference:
    """
    A class to infer network congestion events based on latency jumps and jitter analysis.

    Attributes
    ----------
    latency_jumps : numpy array
        Array containing the latency jump analysis results.
    jitter_analysis : numpy array
        Array containing the jitter analysis results.
    congestion : bool
        Indicates whether a congestion event is currently inferred.
    congestion_inferences : list
        List of tuples representing the inferred congestion states over time.

    Methods
    -------
    fit():
        Processes the latency jumps and jitter analysis to infer congestion events.
    getInferences():
        Returns the inferred congestion events as an array.
    """

    def __init__(self, latency_jumps, jitter_analysis):
        """
        Initializes the CongestionInference class with latency jump and jitter analysis data.

        Parameters
        ----------
        latency_jumps : numpy array
            Array containing the latency jump analysis results.
        jitter_analysis : numpy array
            Array containing the jitter analysis results.
        """
        self.latency_jumps = latency_jumps
        self.jitter_analysis = jitter_analysis
        self.congestion = False
        self.congestion_inferences = []

    def fit(self):
        """
        Processes the latency jumps and jitter analysis to infer congestion events.
        """
        self.congestion_inferences = []
        for jump, jitter in zip(self.latency_jumps, self.jitter_analysis):
            # Determine if there is a new congestion state based on latency jump and jitter conditions
            if jump[2] and jitter[2]:
                self.congestion = True
            elif not jump[2]:
                self.congestion = False

            self.congestion_inferences.append((jump[0], jump[1], self.congestion))



    def getInferences(self):
        """
        Returns the inferred congestion events.

        Returns
        -------
        numpy array
            An array of tuples representing the inferred congestion states over time.
        """
        return np.array(self.congestion_inferences)
