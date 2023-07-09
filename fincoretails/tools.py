import numpy as np
from scipy.special import erfc

def loglikelihood_ratio(loglikelihoodsA, loglikelihoodsB, normalized_ratio=True):

    llA, llB = loglikelihoodsA, loglikelihoodsB
    n = len(llA)
    assert(n == len(llB))

    R = (llA - llB).sum()
    mean_diff = llA.mean() - llB.mean()
    variance = (1/n) * np.sum(((llA-llB) - mean_diff)**2)

    p = erfc( np.abs(R) / np.sqrt(2*n*variance))

    if normalized_ratio:
        R = R/np.sqrt(n*variance)

    return R, p
