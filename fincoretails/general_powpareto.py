import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto, expon
from scipy.optimize import newton, brentq

from fincoretails.tools import general_quantile

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def quantile(q, *parameters):
    return general_quantile(q, cdf, *parameters)

def fit_params(data,beta_initial_values=(1.,),minxmin=None,maxxmin=None):
    a, xm, beta, logLL = alpha_xmin_beta_and_log_likelihood(data, beta0=beta_initial_values,minxmin=minxmin,maxxmin=maxxmin)
    return a, xm, beta

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def loglikelihoods(data, *parameters):
    return np.log(pdf(data, *parameters))

def get_normalization_constant(alpha,xmin,beta):
    assert(beta > -1)
    assert(alpha > 1)
    assert(xmin>0)
    y = xmin
    a = alpha
    b = beta
    C = (b+1)*(a-1)/y/(a+b)
    return C

def Pcrit(alpha,xmin,beta):
    a = alpha
    b = beta
    return (-1 + a)/(a + b)

def sample(Nsample, alpha, xmin, beta):
    assert(xmin>0)
    assert(alpha>1)
    assert(beta > -1)
    #assert(beta>0)
    y = xmin
    a = alpha
    b = beta

    C = get_normalization_constant(alpha,xmin,beta)
    P = Pcrit(alpha,xmin,beta)
    u = np.random.rand(Nsample)
    ndx = np.where(u<P)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    # inverse CDF lower
    xlow = (((1 + b)*ulow* y**b)/C)**(1/(1 + b))

    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)

    return samples

def alpha_beta_and_log_likelihood_fixed_xmin(data, xmin, b0=1.5):
    n = len(data)

    Lambda = data[np.where(data>xmin)[0]]
    Eta = data[np.where(data<=xmin)[0]]
    logL = np.mean(np.log(Lambda))
    logH = np.mean(np.log(Eta))
    nL = len(Lambda)
    nH = len(Eta)

    Sqrt = np.sqrt
    Log = np.log

    y = xmin
    logy = Log(y)

    bStrong = -1 + n/(+(Sqrt(nL)*Sqrt(nH)*Sqrt(logL-logy)*Sqrt(logy-logH)) + nH*(logy-logH))
    bWeak   = -1 + n/(-(Sqrt(nL)*Sqrt(nH)*Sqrt(logL-logy)*Sqrt(logy-logH)) + nH*(logy-logH))
    #print(bStrong)
    #print(bWeak)
    testbs = [bStrong]
    if bWeak > -1:
        testbs.append(bWeak)

    maxlogLL = -np.inf
    hata = None
    hatb = None
    for b in testbs:
        a = (1 - b + (1 + b)*Sqrt(1 + (4*n)/(nL*(1 + b)*(logL-logy))))/2.
        if np.isnan(a):
            continue
        C = get_normalization_constant(a,y,b)
        logLL = n*np.log(C) - a*nL*(logL-logy) - b *nH*(logy-logH)

        if logLL > maxlogLL and (a>1) and (b > -1):
            hata = a
            hatb = b
            maxlogLL = logLL

    return hata, hatb, maxlogLL

def alpha_xmin_beta_and_log_likelihood_fixed_xmin_index(data, j, xmins=None, beta0=[2.,]):

    if xmins is None:
        xmins = np.sort(np.unique(data))

    n = len(data)

    xleft = xmins[j]
    xright = xmins[j+1]

    Lambda = data[np.where(data>xleft)[0]]
    Eta = data[np.where(data<=xleft)[0]]
    logH = np.mean(np.log(Eta))
    logL = np.mean(np.log(Lambda))
    nL = len(Lambda)
    nH = len(Eta)


    b = (-n + 2*nL - n*logH + nL*logH + n*logL - nL*logL)/((n - nL)*(logH - logL))
    logy = nL/((n - nL)*(1 + b)) + logL
    a = (n + nH*b)/nL
    y = np.exp(logy)
    if not np.isnan(a) and a > 1 and not np.isnan(b) and b > -1:
        C = get_normalization_constant(a,y,b)
        logLL = n*np.log(C) - a*nL*(logL-logy) - b *nH*(logy-logH)

        _a, _b, _logLL = alpha_beta_and_log_likelihood_fixed_xmin(data, xleft)

        if (_logLL > logLL) or (a < 1) or (b < -1) or (y<xleft) or (y>=xright):
            return _a, xleft, _b, _logLL
        else:
            return a, y, b, logLL
    else:
        _a, _b, _logLL = alpha_beta_and_log_likelihood_fixed_xmin(data, xleft)
        return _a, xleft, _b, _logLL



def alpha_xmin_beta_and_log_likelihood(data, beta0=[2.,],stop_at_first_max=False,minxmin=None,maxxmin=None):

    n = len(data)
    xmins = np.sort(np.unique(data))

    if minxmin is None:
        minxmin = xmins[0]
    xmins = xmins[xmins>minxmin]
    if maxxmin is not None:
        xmins = xmins[xmins<=maxxmin]

    maxlogLL = -np.inf
    alphas, nLs, nSs, logLLs  = [], [],[], []
    sampled_xmins = []
    for j, xj in enumerate(xmins[:-1]):

        alpha, xmincand, beta, logLL = alpha_xmin_beta_and_log_likelihood_fixed_xmin_index(data,j,xmins)

        if xmincand >= xmins[j] and xmincand <xmins[j+1] and logLL > maxlogLL:
            hatxmin = xmincand
            hatalpha = alpha
            hatbeta = beta
            hatlogLL = logLL
            maxlogLL = logLL

            if stop_at_first_max:
                break

    return hatalpha, hatxmin, hatbeta, hatlogLL


def pdf_left(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    y = xmin
    return C * (x/y)**beta

def pdf_right(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    return C * (xmin/x)**alpha

def pdf(x, alpha, xmin,beta):
    """"""
    C = get_normalization_constant(alpha, xmin, beta)
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
                          alpha, xmin, beta, C,
                         )
    return result

def cdf_left(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    y = xmin
    b = beta
    return (C * x*(x/y)**b)/(1 + b)

def cdf_right(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    y = xmin
    p0 = Pcrit(alpha, xmin, beta)
    return C / (alpha-1) * (y-x*(y/x)**alpha) + p0

def cdf(x, alpha, xmin, beta):
    C = get_normalization_constant(alpha, xmin, beta)
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
                          alpha, xmin, beta, C,
                         )

    return result

def median(alpha,xmin,beta):
    a, xm, b = alpha, xmin, beta
    return quantile(0.5, a, xm, b)

def variance(*args,**kwargs):
    return second_moment(*args,**kwargs) - mean(*args,**kwargs)**2

def neighbor_degree(*args,**kwargs):
    return second_moment(*args,**kwargs)/mean(*args,**kwargs)

def mean(alpha,xmin,beta):
    assert(alpha > 2)
    y = xmin
    a = alpha
    b = beta
    return (2*(-1 + a)*a*(1 + b)*y) / ((-4 + a**2)*(a + b))

def second_moment(alpha,xmin,beta):
    assert(alpha > 3)
    y = xmin
    a = alpha
    b = beta

    return ((-1 + a)*(1 + b)*y**2) / ((-3 + a)*(3 + b))


if __name__=="__main__":

    dataset = 'sample'
    if dataset == 'contacts':
        data = np.loadtxt('/Users/bfmaier/forschung/2023-Leo-Master/data/first_week_C_observations.txt')
        data = np.round(data)
        data = np.array(data,dtype=int)
        #data = data[data>0]
    elif dataset == 'sample':
        np.random.seed(20)
        alphatrue = 3.1
        betatrue = -0.5
        Nsample = 10_000
        xmintrue = 10
        data = sample(Nsample,alphatrue,xmintrue, betatrue)
        print(f"{alphatrue=},{xmintrue=},{betatrue=}")
        #print(f"{cdf(1000.,alphatrue,xmintrue,betatrue)=}")
        #print(f"{data.min()=}")
        #print()
        print(f"{data.mean()=}")
        print(f"{data.std()**2=}")
        print(f"{np.median(data)=}")

        print(f"{mean(alphatrue,xmintrue,betatrue)=}")
        print(f"{variance(alphatrue,xmintrue,betatrue)=}")
        print(f"{median(alphatrue,xmintrue,betatrue)=}")

        _dens,_be,_ = pl.hist(data,np.logspace(-1,3,101),density=True)
        pl.plot(_be, pdf(_be, alphatrue, xmintrue, betatrue))
        pl.xscale('log')
        pl.yscale('log')
        pl.figure()
        pl.plot(_be,cdf(_be, alphatrue, xmintrue, betatrue))
        pl.xscale('log')
        pl.yscale('log')
        #pl.show()

    xmins = np.linspace(9.5,10.5,20_001)

    fig, ax = pl.subplots(3,1,figsize=(4,10),sharex=True)
    logLs = []
    alphas = []
    betas = []
    hatalpha = None
    hatxmin = None
    hatbeta = None
    maxLL = None
    for xmin in xmins:
        a, b, LL = alpha_beta_and_log_likelihood_fixed_xmin(data, xmin)
        if maxLL is None:
            maxLL = LL
        else:
            if LL > maxLL:
                maxLL = LL
                hatalpha = a
                hatxmin = xmin
                hatbeta = b
        logLs.append(LL)
        alphas.append(a)
        betas.append(b)

    uniquedat = np.sort(np.unique(data))
    #uniquedat = uniquedat[np.where(np.logical_and(uniquedat>1,uniquedat<40))]
    unq_alphas = []
    unq_betas = []
    unq_logLs = []

    for xmin in uniquedat:
        a, b, LL = alpha_beta_and_log_likelihood_fixed_xmin(data, xmin)
        unq_logLs.append(LL)
        unq_alphas.append(a)
        unq_betas.append(b)

    print(hatalpha, hatxmin, hatbeta)

    ax[0].plot(uniquedat, unq_logLs,'s',mfc='w')
    ax[1].plot(uniquedat, unq_alphas,'s',mfc='w')
    ax[2].plot(uniquedat, unq_betas,'s',mfc='w')
    ax[0].plot(xmins, logLs)
    ax[1].plot(xmins, alphas)
    ax[2].plot(xmins, betas)

    ax[1].set_xlabel('xmin')
    ax[0].set_ylabel('log-likelihood')
    fig.tight_layout()

    #a_, xmin_, LL_ = alpha_xmin_and_log_likelihood(data)
    a_, xmin_, b_, LL_ = alpha_xmin_beta_and_log_likelihood(data)
    print(a_, xmin_, b_)
    ax[0].plot([xmin_,xmin_],ax[0].get_ylim(),'--')
    ax[1].plot([xmin_,xmin_],ax[1].get_ylim(),'--')
    ax[2].plot([xmin_,xmin_],ax[2].get_ylim(),'--')
    ax[0].plot(xmins[[0,-1]],[LL_]*2,'--')
    ax[1].plot(xmins[[0,-1]],[a_]*2,'--')
    ax[2].plot(xmins[[0,-1]],[b_]*2,'--')

    fig, ax = pl.subplots(1,1)

    if dataset == 'contacts':
        be = np.logspace(1,5)
        be = np.concatenate([np.arange(1,10),be])
    else:
        be = np.logspace(-1,4,101)
    ax.hist(data,be,density=True)
    #ax.plot([xmin_,xmin_],ax.get_ylim(),'--')

    if dataset=='sample':
        x = np.logspace(-1,4,1001)
    else:
        x = np.logspace(0,6,1001)
        ax.plot(x, pdf(x, alphatrue, xmintrue,betatrue))
    #ax.plot(x, pdf(x, hatalpha,hatxmin,hat))
    ax.plot(x, pdf(x, a_,xmin_,b_))

    ax.set_xscale('log')
    ax.set_yscale('log')


    pl.show()


