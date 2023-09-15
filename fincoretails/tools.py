import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq

def loglikelihood_ratio(loglikelihoodsA, loglikelihoodsB, normalized_ratio=True):
    """
    Compute the log-likelihood ratio and the significance level.
    Return the log-likelihood ratio R and the probability p
    that a random sample from a normally distributed R-value
    distribution would be larger or equal to the computed
    R-value.

    Parameters
    ----------
    loglikelihoodsA : array-like
        Log-likelihoods under null hypothesis.
    loglikelihoodsB : array-like
        Log-likelihoods under alternative hypothesis.
    normalized_ratio : bool, optional, default = True
        If True (default), return the normalized log-likelihood ratio.

    Returns
    -------
    R : float
        The log-likelihood ratio. It is normalized if `normalized_ratio` is True.
    p : float
        The significance level, i.e., the probability that a random sample
        from a normally distributed R-value distribution would be larger or
        equal to the computed R-value.

    Notes
    -----
    Typically, one classifies the R-value as 'significant' if p < 0.05. However,
    also consider the context of the problem rather than strictly following this rule.
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
    left : float, optional, default = 0
        The lower bound of the interval for the root-finding algorithm. Default is 0.
    right : float, optional, default = 1000
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
    """
    Compute the Akaike Information Criterion (AIC).

    Parameters
    ----------
    logLL : float
        The log-likelihood.
    number_of_free_parameters : int
        The number of free parameters in the model.
    nsamples : int, optional, default = None
        The number of samples. If not provided or if the sample size is too small,
        the function will return the regular AIC.

    Returns
    -------
    AIC : float
        The Akaike Information Criterion.

    Notes
    -----
    If the number of samples and the number of free parameters are large enough,
    the function will return the corrected AIC.
    """

    # Compute the regular AIC
    k = number_of_free_parameters
    n = nsamples
    AIC = 2*(number_of_free_parameters - logLL)

    # return the corrected AIC, only if number of samples has been given however.
    if n is None or n-k-1 <= 0:
        return AIC
    else:
        return AIC + 2*k*(k+1)/(n-k-1)

def compute_ccdf(data):
    """
    This function takes in an array of data and returns two arrays (x, ccdf)
    that represent the step-wise complementary cumulative distribution function (CCDF).

    Parameters
    ----------
        data: numpy.ndarray
            An array of positive random variates.

    Returns
    -------
        x: numpy.ndarray
            Sorted unique data.
        ccdf: numpy.ndarray
            CCDF of the data.
    """
    # Step 1: Sort the data in ascending order
    sorted_data = np.sort(data)

    # Step 2: Get the unique data values and their counts
    unique, counts = np.unique(sorted_data, return_counts=True)

    # Step 3: Calculate the cumulative counts
    cum_counts = np.cumsum(counts)

    # Step 4: Calculate the complementary cumulative distribution (CCDF)
    ccdf = 1 - cum_counts / len(data)

    return unique, ccdf

