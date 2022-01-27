import numpy as np

class congestionInference:

    def __init__(self, latency_jumps, jitter_analysis):
        
        self.latency_jumps = latency_jumps
        self.jitter_analysis = jitter_analysis

        self.congestion = False


    def fit(self):

        self.congestion_inferences = []

        for i in range(len(self.latency_jumps)):

            if self.congestion and self.latency_jumps[i][2]:
                # The link was already congested and it has jumped to a state of larger mean RTT
                continue
            else:
                if self.latency_jumps[i][2] and self.jitter_analysis[i][2]:
                    self.congestion = True
                else:
                    self.congestion = False


            self.congestion_inferences.append((self.latency_jumps[i][0], self.latency_jumps[i][1], self.congestion))

    def getInferences(self):
        return np.array(self.congestion_inferences)
