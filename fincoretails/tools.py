import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq

def loglikelihood_ratio(loglikelihoodsA, loglikelihoodsB, normalized_ratio=True):
    """
    Return the log-likelihood ratio R and the probability p
    that a random sample from a normally distributed R-value
    distribution would be larger or equal to the computed
    R-value. Typically, one classifies the R-value as ``significant''
    if p < 0.05, but also please use your brain more than
    just checking p < 0.05.
    """

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

def general_quantile(q, cdf, *parameters, left=0, right=1000):
    """
    Numerically compute a quantile of a distribution.
    Will automatically adapt the interval if there's no sign change
    in the function cdf-q

    Parameters
    ----------
    q : float
        The quantile rank, a number between 0 and 1.
    cdf : function
        The cumulative distribution function.
    *parameters :
        The parameters of the distribution.
    left : float, optional
        The lower bound of the interval for the root-finding algorithm. Default is 0.
    right : float, optional
        The upper bound of the interval for the root-finding algorithm. Default is 1000.

    Returns
    -------
    quantile : float
        The quantile corresponding to the given quantile rank.
    """

    if q == 1:
        return np.inf

    if q == 0:
        return 0

    assert(0 < q < 1)

    # Define the equation to solve: cdf(x) - q = 0
    func = lambda x: cdf(x, *parameters) - q

    interval = (left, right)

    while True:
        try:
            # Use the Brent's method to find a root of the equation
            quantile = brentq(func, *interval)
            break
        except ValueError as e:
            interval = (interval[1], interval[1]*10)

    return quantile

def aic(logLL, number_of_free_parameters, nsamples=None):
    k = number_of_free_parameters
    n = nsamples
    AIC = 2*(number_of_free_parameters - logLL)
    if n is None or n-k-1 <= 0:
        return AIC
    else:
        return AIC + 2*k*(k+1)/(n-k-1)
