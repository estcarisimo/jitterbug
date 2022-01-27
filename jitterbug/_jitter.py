import jitterbug.preprocessing as preproc

from jitterbug.signal_energy import jitterDispersion
from jitterbug.kstest import kstest


def compute_jiiter_dispersion(change_points, epoch_mins, mins, K, L, jitter_dispresion_threshold=0.25):
    # preproc
    epoch_jitter_dispresion, jitter_dispresion = preproc.jiiter_dispersion(epoch_mins, mins, K, L)

    # jd
    jd = jitterDispersion(epoch_jitter_dispresion, jitter_dispresion, change_points)
    jd.fit(jitter_dispresion_threshold)

    return jd.getJitterDispersionValues()



def compute_ks_test(change_points, epoch_rtt, rtt):
    # preproc 
    epoch_jitter, jitter = preproc.jiiter(epoch_rtt, rtt)

    # ks
    ks = kstest(epoch_jitter, jitter, change_points)
    ks.fit()
    return ks.getKSTestResults()