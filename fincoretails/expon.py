import numpy as np

def fit_params(data):
    lam = rate(data)
    return (lam,)

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def rate(data):
    mu = np.mean(data)
    return 1/mu

def pdf(x,rate):
    return rate * np.exp(-rate*x)

def loglikelihoods(data,rate):
    return np.log(pdf(data,rate))

def mean(rate):
    return 1/rate

def median(rate):
    return np.log(2)/rate

def variance(rate):
    return 1/rate**2

def second_moment(rate):
    return variance(rate) + mean(rate)**2

def neighbor_degree(rate):
    return second_moment(rate)/mean(rate)

def cdf(x,rate):
    return 1-np.exp(-rate*x)

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def quantile(q, rate):
    return -np.log(1-q)/rate

