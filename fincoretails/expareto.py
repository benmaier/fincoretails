import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto, expon
from scipy.optimize import newton, brentq

from fincoretails.tools import general_quantile

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def quantile(q, *parameters):
    return general_quantile(q, cdf, *parameters)

def fit_params(data, minxmin=1.001):
    a, xm, logLL = alpha_xmin_and_log_likelihood(data, minxmin=minxmin)
    return a, xm

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def loglikelihoods(data, *parameters):
    return np.log(pdf(data, *parameters))

def get_normalization_constant(alpha,xmin):
    assert(alpha > 1)
    assert(xmin>0)
    y = xmin
    a = alpha
    ea = np.exp(a)
    C = a*(a-1) / y / (1+(a-1)*ea)
    return C

def Pcrit(alpha,xmin,C=None):
    a = alpha
    ea = np.exp(a)
    return 1 - a/((a-1)*ea + 1)

def sample(Nsample, alpha, xmin):
    assert(xmin>0)
    assert(alpha>1)
    y = xmin
    a = alpha

    ea = np.exp(a)
    C = a*(a-1) / y / (1+(a-1)*ea)
    P = Pcrit(alpha,xmin,C)
    u = np.random.rand(Nsample)
    ndx = np.where(u<P)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    xlow = y/a * np.log(C*y*ea/(C*y*ea-a*ulow))

    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)
    return samples

def alpha_and_log_likelihood_fixed_xmin(data, xmin, a0=1.5):
    n = len(data)
    Lambda = data[np.where(data>xmin)[0]]
    Eta = data[np.where(data<=xmin)[0]]
    nL = len(Lambda)
    nH = len(Eta)
    L = np.mean(np.log(Lambda/xmin))
    Hminus1 = np.mean(Eta/xmin) - 1
    H = Hminus1 + 1
    CL = nL*L
    CH = nH*Hminus1

    dLogLLda = lambda a: n*(1/a + 1/(a-1) - a*np.exp(a)/(1+np.exp(a)*(a-1))) - CL - CH

    a = newton(dLogLLda, a0,maxiter=10000)

    C = a*(a-1) / xmin / (1+(a-1)*np.exp(a))
    logLL = n*np.log(C) - a*(CL+CH)

    return a, logLL


def alpha_xmin_and_log_likelihood_fixed_xmin_index(data, j, xmins=None):

    if xmins is None:
        xmins = np.sort(np.unique(data))

    n = len(data)

    xleft = xmins[j]
    xright = xmins[j+1]

    Lambda = data[np.where(data>xleft)[0]]
    Eta = data[np.where(data<=xleft)[0]]
    H = np.mean(Eta)
    L = np.mean(np.log(Lambda))
    nL = len(Lambda)
    nH = len(Eta)

    def dlogLLdalpha(xcand):
        a = n / (nL+ nH*H/xcand)
        return   n*(1/a + 1/(a-1) - a*np.exp(a)/(1+np.exp(a)*(a-1)))\
               - nL*(L-np.log(xcand)) - nH*(H/xcand - 1)

    try:
        xmin_new = brentq(dlogLLdalpha,xleft,xright)
    except ValueError as e:
        return None, None, None

    xmin = xmin_new
    a = n / (nL+ nH*H/xmin)
    C = a*(a-1) / xmin / (1+(a-1)*np.exp(a))
    logLL = n*np.log(C) - a*nL*(L-np.log(xmin)) - a*nH*(H/xmin - 1)

    return a, xmin_new, logLL


def alpha_xmin_and_log_likelihood(data, minxmin=6):

    n = len(data)
    xmins = np.sort(np.unique(data))
    xmins = xmins[xmins>=minxmin]

    maxLogLL = -np.inf
    alphas, nLs, nSs, logLLs  = [], [],[], []
    sampled_xmins = []
    for j, xj in enumerate(xmins):
        alpha, xmincand, logLL = alpha_xmin_and_log_likelihood_fixed_xmin_index(data,j,xmins)

        if xmincand is None:
            continue
        if xmincand >= xmins[j] and xmincand <=xmins[j+1]:
            break

    return alpha, xmincand, logLL


def pdf_left(x, alpha, xmin, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin)
    return C * np.exp(-alpha*(x/xmin - 1))

def pdf_right(x, alpha, xmin, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin)
    return C * (xmin/x)**alpha

def pdf(x, alpha, xmin):
    """"""
    C = get_normalization_constant(alpha, xmin)
    result = np.piecewise(np.asanyarray(x,dtype=float),
                          (
                              x<0,
                              x<=xmin,
                              x>xmin
                          ),
                          (
                              0.,
                              pdf_left,
                              pdf_right,
                          ),
                          alpha, xmin, C,
                         )
    return result

def cdf_left(x, alpha, xmin, C=None):
    xm = xmin
    return ((1 - alpha)*np.exp(alpha*(xm - x)/xm) + (alpha - 1)*np.exp(alpha))/((alpha - 1)*np.exp(alpha) + 1)

def cdf_right(x, alpha, xmin, C=None):
    xm = xmin
    return (-xm**(alpha - 1)*alpha*x**(1 - alpha) + alpha*np.exp(alpha) - np.exp(alpha) + 1)/(alpha*np.exp(alpha) - np.exp(alpha) + 1)

def cdf(x, alpha, xmin):
    C = get_normalization_constant(alpha, xmin)
    result = np.piecewise(np.asanyarray(x,dtype=float),
                          (
                              x<0,
                              x<=xmin,
                              x>xmin
                          ),
                          (
                              0.,
                              cdf_left,
                              cdf_right,
                          ),
                          alpha, xmin, C,
                         )

    return result

def mean(alpha,xmin):
    assert(alpha > 2)
    xm = xmin
    return xm*alpha*(alpha - 1)*(6*alpha + (alpha - 3)**2 - (alpha - 2)*(alpha - np.exp(alpha) + 1) - 9)/((alpha - 2)*((alpha - 1)*np.exp(alpha) + 1)*(6*alpha + (alpha - 3)**2 - 9))

def median(alpha,xmin):
    a, xm = alpha, xmin
    return quantile(0.5, a, xm)

def second_moment(alpha,xmin):
    assert(alpha > 3)
    xm = xmin
    return xm**2*(alpha**3 + 2*alpha**2*np.exp(alpha) + 3*alpha**2 - 8*alpha*np.exp(alpha) + 2*alpha + 6*np.exp(alpha) - 6)/(alpha**2*(alpha**2*np.exp(alpha) - 4*alpha*np.exp(alpha) + alpha + 3*np.exp(alpha) - 3))

def variance(*args,**kwargs):
    return second_moment(*args,**kwargs) - mean(*args,**kwargs)**2

def neighbor_degree(*args,**kwargs):
    return second_moment(*args,**kwargs)/mean(*args,**kwargs)




if __name__=="__main__":

    dataset = 'sample'
    if dataset == 'contacts':
        data = np.loadtxt('/Users/bfmaier/forschung/2023-Leo-Master/data/first_week_C_observations.txt')
        data = np.round(data)
        data = np.array(data,dtype=int)
    elif dataset == 'sample':
        alphatrue = 4
        Nsample = 10_000
        xmintrue = 10
        data = sample(Nsample,alphatrue,xmintrue)
        print(f"{cdf(1000.,alphatrue,xmintrue)=}")
        print(f"{data.min()=}")
        print()
        print(f"{data.mean()=}")
        print(f"{data.std()**2=}")
        print(f"{np.median(data)=}")

        print(f"{mean(alphatrue,xmintrue)=}")
        print(f"{variance(alphatrue,xmintrue)=}")
        print(f"{median(alphatrue,xmintrue)=}")

        _dens,_be,_ = pl.hist(data,np.logspace(-1,5,101),density=True)
        pl.plot(_be, pdf(_be, alphatrue, xmintrue))
        pl.xscale('log')
        pl.yscale('log')
        pl.figure()
        pl.plot(_be,cdf(_be, alphatrue, xmintrue))
        pl.xscale('log')
        pl.yscale('log')
        pl.show()

    xmins = np.linspace(4,40,2001)

    fig, ax = pl.subplots(2,1,figsize=(4,8),sharex=True)
    logLs = []
    alphas = []
    hatalpha = None
    hatxmin = None
    maxLL = None
    for xmin in xmins:
        a, LL = alpha_and_log_likelihood_fixed_xmin(data, xmin)
        if maxLL is None:
            maxLL = LL
        else:
            if LL > maxLL:
                maxLL = LL
                hatalpha = a
                hatxmin = xmin
        logLs.append(LL)
        alphas.append(a)

    uniquedat = np.sort(np.unique(data))
    uniquedat = uniquedat[np.where(np.logical_and(uniquedat>4,uniquedat<40))]
    unq_alphas = []
    unq_logLs = []

    for xmin in uniquedat:
        a, LL = alpha_and_log_likelihood_fixed_xmin(data, xmin)
        unq_logLs.append(LL)
        unq_alphas.append(a)

    ax[0].plot(uniquedat, unq_logLs,'s',mfc='w')
    ax[1].plot(uniquedat, unq_alphas,'s',mfc='w')
    ax[0].plot(xmins, logLs)
    ax[1].plot(xmins, alphas)

#ax[1].set_xscale('log')
    ax[1].set_xlabel('xmin')
    ax[0].set_ylabel('log-likelihood')
    fig.tight_layout()

    a_, xmin_, LL_ = alpha_xmin_and_log_likelihood(data)
    ax[0].plot([xmin_,xmin_],ax[0].get_ylim(),'--')
    ax[1].plot([xmin_,xmin_],ax[1].get_ylim(),'--')
    ax[0].plot(xmins[[0,-1]],[LL_]*2,'--')

    fig, ax = pl.subplots(1,1)

    if dataset == 'contacts':
        be = np.logspace(1,5)
        be = np.concatenate([np.arange(1,10),be])
    else:
        be = np.logspace(-1,4,101)
    ax.hist(data,be,density=True)
    ax.plot([xmin_,xmin_],ax.get_ylim(),'--')
    print(hatalpha,hatxmin)

    if dataset=='sample':
        x = np.logspace(-1,4,1001)
        ax.plot(x, pdf(x, alphatrue, xmintrue))
    else:
        x = np.logspace(0,6,1001)
    ax.plot(x, pdf(x, hatalpha,hatxmin))

    ax.set_xscale('log')
    ax.set_yscale('log')


    pl.show()


