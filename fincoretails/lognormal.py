import numpy as np
from scipy.special import erf, erfinv
from scipy.stats import lognorm as splognorm

def fit_params(data):
    mu, sigma = mu_and_sigma(data)
    return mu, sigma

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def mu_and_sigma(data):
    mu = np.mean(np.log(data))
    sigma2 = np.mean((np.log(data)-mu)**2)
    sigma = np.sqrt(sigma2)

    return mu, sigma

def logLL(data):
    mu, sigma = mu_and_sigma(data)
    n = len(data)
    lnLL = - n/2*(1+np.log(2*np.pi*sigma**2)) \
           - n*mu
    return lnLL

def pdf(x,mu,sigma):
    C = 1/np.sqrt(2*np.pi*sigma**2)/x
    scl = -(np.log(x)-mu)**2/2/sigma**2
    return C * np.exp(scl)

def loglikelihoods(data,mu,sigma):
    return np.log(pdf(data,mu,sigma))

def mean(mu,sigma):
    return np.exp(mu + sigma**2/2)

def median(mu,sigma=None):
    return np.exp(mu)

def variance(mu,sigma):
    return (np.exp(sigma**2)-1) * np.exp(2*mu+sigma**2)

def second_moment(mu,sigma):
    return variance(mu,sigma) + mean(mu,sigma)**2

def neighbor_degree(mu,sigma):
    return second_moment(mu,sigma)/mean(mu,sigma)

def cdf(x,mu,sigma):
    return 0.5*(1+erf( (np.log(x)-mu)/sigma/np.sqrt(2)))

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def quantile(q, mu, sigma):
    return np.exp(mu+np.sqrt(2)*sigma*erfinv(2*q-1))

def sample(Nsample, mu, sigma):
    Z = np.random.randn(Nsample)
    return np.exp(mu + sigma*Z)
