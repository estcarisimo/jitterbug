import jitterbug.filters as filters


K = 4
L = 6
KL = K + int(L / 2)


def jiiter(epoch, rtt):
    """

    """
    if len(epoch) == len(rtt):
        if len(rtt) > 1:
            t = epoch[1:]
            x = rtt[1:] - rtt[:-1]
        else:
            raise Exception("Insufficient samples (fewer than 2) to compute jitter")
    else:
        raise Exception("epochs and rtts have different lengths.")
    
    return t, x

def jiiter_dispersion(epoch, mins, iqr_order, ma_order):
    """
    """
    if len(epoch) == len(mins):
        # chequear si no habia problema con ma_order, iqr_order o kl impares. En tal caso poner un if y excepcion
        kl = iqr_order + int(ma_order / 2)

        jmin_t, jmin_vals = jiiter(epoch, mins)
        
        t = jmin_t[kl:-(kl - 1)]
        x = filters.moving_average(filters.moving_iqr_filter_symmetric(jmin_vals, iqr_order), ma_order)
    else:
        raise Exception("epochs and mins have different lengths.")

    return t, x
