import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto, expon
from scipy.optimize import newton, bisect

from fincoretails.tools import general_quantile

def quantile(q, *parameters):
    return general_quantile(q, cdf, *parameters)

def fit_params(data, N, minxmin=4):
    a, xm, logLL = alpha_xmin_and_log_likelihood(data, minxmin=minxmin)
    return a, xm

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def pdf(x, alpha, xmin):
    ea = np.exp(alpha)
    C = alpha*(alpha-1) / xmin / (1+(alpha-1)*ea)
    if hasattr(x, '__len__'):
        x = np.array(x)
        cond = x<=xmin
        i0 = np.where(cond)[0]
        i1 = np.where(np.logical_not(cond))[0]
        result = np.zeros_like(x,dtype=float)
        result[i0] = C * np.exp(-alpha*(x[i0]/xmin - 1))
        result[i1] = C * (xmin/x[i1])**alpha
    else:
        if x <= xmin:
            return C * np.exp(-alpha*(x/xmin - 1))
        else:
            return C * (xmin/x)**alpha

def sample(alpha, xmin, Nsample):
    assert(xmin>0)
    assert(alpha>1)
    y = xmin
    a = alpha

    ea = np.exp(a)
    C = a*(a-1) / y / (1+(a-1)*ea)
    Pcrit = C*y/a * (ea-1)
    print(Pcrit, C*y*ea/a)
    u = np.random.rand(Nsample)
    ndx = np.where(u<Pcrit)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    xlow = y/a * np.log(C*y*ea/(C*y*ea-a*ulow))

    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)
    return samples

def alpha_and_log_likelihood_fixed_xmin(data, xmin, a0=1.8):
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

    #print(f"{xleft=}", f"{xright=}")

    Lambda = data[np.where(data>xleft)[0]]
    Eta = data[np.where(data<=xleft)[0]]
    H = np.mean(Eta)
    L = np.mean(np.log(Lambda))
    nL = len(Lambda)
    nH = len(Eta)

    def dlogLLdalpha(xcand):
        #xcand = a * nH * H / (n - a*nL)
        a = n / (nL+ nH*H/xcand)
        #print(f"{a=}", f"{xcand=}")
        return   n*(1/a + 1/(a-1) - a*np.exp(a)/(1+np.exp(a)*(a-1)))\
               - nL*(L-np.log(xcand)) - nH*(H/xcand - 1)

    try:
        xmin_new = bisect(dlogLLdalpha,xleft,xright)
    except ValueError as e:
        return None, None, None
    #except RuntimeError as e:
    #    return None, None, None

    #xmin_new = a * nH * H / (n - a*nL)
    #print(f"{xmin_new=}")
    xmin = xmin_new
    a = n / (nL+ nH*H/xmin)
    C = a*(a-1) / xmin / (1+(a-1)*np.exp(a))
    logLL = n*np.log(C) - a*nL*(L-np.log(xmin)) - a*nH*(H/xmin - 1)

    #a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xmin_new, a)

    return a, xmin_new, logLL


def alpha_xmin_and_log_likelihood(data, minxmin=6):

    n = len(data)
    xmins = np.sort(np.unique(data))
    xmins = xmins[xmins>=minxmin]

    maxLogLL = -np.inf
    alphas, nLs, nSs, logLLs  = [], [],[], []
    sampled_xmins = []
    for j, xj in enumerate(xmins):
        #a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xj)
        alpha, xmincand, logLL = alpha_xmin_and_log_likelihood_fixed_xmin_index(data,j,xmins)

        if xmincand is None:
            continue
        if xmincand >= xmins[j] and xmincand <=xmins[j+1]:
            break

    return alpha, xmincand, logLL

#def alpha_xmin_and_log_likelihood(data, minxmin=6):
#
#    n = len(data)
#    xmins = np.sort(np.unique(data))
#    xmins = xmins[xmins>=minxmin]
#
#    maxLogLL = -np.inf
#    alphas, nLs, nSs, logLLs,  = [], [],[], []
#    sampled_xmins = []
#    for j, xj in enumerate(xmins):
#        a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xj)
#        alphas.append(a)
#        logLLs.append(logLL)
#        sampled_xmins.append(xj)
#
#        if logLL > maxLogLL:
#            maxLogLL = logLL
#        else:
#            break
#
#    print(f"found maximum at xmin={xmins[j-1]}")
#
#    if j == 1:
#        check_indices = [0]
#    else:
#        check_indices = [j-1, j-2]
#
#    current_max_tuple = alphas[j-1], sampled_xmins[j-1], logLLs[j-1]
#
#    for k in check_indices:
#        print(f"now checking xmin={xmins[k]}")
#        alpha, xmincand, logLL = alpha_xmin_and_log_likelihood_fixed_xmin_index(data,k,alphas[k],alphas[k+1],xmins)
#        print(f"found new value {alpha=} and {xmincand=}")
#
#        if logLL is not None and logLL > maxLogLL:
#            maxLogLL = logLL
#            current_set = alpha, xmincand, logLL
#
#    return current_max_tuple


#def alpha_xmin_and_log_likelihood(data):
#
#    n = len(data)
#    xmins = np.sort(np.unique(data))
#
#    maxLogLL = -np.inf
#    alphas, nLs, Ls, logLLs = [], [],[], []
#    sampled_xmins = []
#    for j, xj in enumerate(xmins[10:]):
#        a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xj)
#        alphas.append(a)
#        logLLs.append(logLL)
#        sampled_xmins.append(xj)
#
#        if logLL > maxLogLL:
#            maxLogLL = logLL
#        else:
#            break
#
#    return alphas[j-1], sampled_xmins[j-1], logLLs[j-1]


if __name__=="__main__":

    dataset = 'sample'
    if dataset == 'contacts':
        data = np.loadtxt('first_week_C_observations.txt')
        data = np.round(data)
        data = np.array(data,dtype=int)
    elif dataset == 'sample':
        alphatrue = 2
        xmintrue = 10
        data = sample(alphatrue,xmintrue,1001)
        print(f"{data.min()=}")


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


