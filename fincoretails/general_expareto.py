import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto, expon
from scipy.optimize import newton, brentq

from fincoretails.tools import general_quantile

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
    assert(alpha > 1)
    assert(xmin>0)
    y = xmin
    a = alpha
    b = beta
    eb = np.exp(b)
    C = b*(a-1) / y / (b+(a-1)*(eb-1))
    return C

def Pcrit(alpha,xmin,beta):
    a = alpha
    b = beta
    y = xmin
    eb = np.exp(b)
    return 1 - b/(1 - a + b + (-1 + a)*eb)

def sample(Nsample, alpha, xmin, beta):
    assert(xmin>0)
    assert(alpha>1)
    #assert(beta>0)
    y = xmin
    a = alpha
    b = beta

    eb = np.exp(b)
    C = get_normalization_constant(alpha,xmin,beta)
    P = Pcrit(alpha,xmin,beta)
    u = np.random.rand(Nsample)
    ndx = np.where(u<P)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    xlow = y/b * np.log(C*y*eb/(C*y*eb-b*ulow))

    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)
    return samples

def alpha_beta_and_log_likelihood_fixed_xmin(data, xmin, b0=1.5):
    n = len(data)

    Lambda = data[np.where(data>xmin)[0]]
    Eta = data[np.where(data<=xmin)[0]]
    H = np.mean(Eta)
    L = np.mean(np.log(Lambda))
    logH = np.log(H)
    nL = len(Lambda)
    nH = len(Eta)
    logxm = np.log(xmin)

    def alpha_given_beta(b,eb):
        return 1-0.5*b/(eb-1) * (1 - np.sqrt(1 + 4*n*(eb-1) / (b*nL * (L - logxm))))

    def dLogLLdb(b):
        eb = np.exp(b)
        a = alpha_given_beta(b,eb)
        return -nH*(H/xmin - 1) - (n/b) * (a-1)*(eb*(b-1)+1) / ((a-1)*eb-a+b+1)


    b = newton(dLogLLdb, b0,maxiter=10000)
    a = alpha_given_beta(b,np.exp(b))

    C = get_normalization_constant(a,xmin,b)
    logLL = n*np.log(C) - a *nL*(L-logxm) - b*nH*(H/xmin-1)

    return a, b, logLL

def alpha_xmin_beta_and_log_likelihood_fixed_xmin_index(data, j, xmins=None, beta0=[2.,]):

    if xmins is None:
        xmins = np.sort(np.unique(data))

    n = len(data)

    xleft = xmins[j]
    xright = xmins[j+1]

    Lambda = data[np.where(data>xleft)[0]]
    Eta = data[np.where(data<=xleft)[0]]
    H = np.mean(Eta)
    L = np.mean(np.log(Lambda))
    logH = np.log(H)
    nL = len(Lambda)
    nH = len(Eta)


    def z(b):
        eb = np.exp(b)
        return -2*n + nL * ( 1+ np.sqrt(
                                  1 + 4*n*(eb-1) \
                                        /\
                                    (  b*nL *(L-logH + np.log(1/(1-eb) + 1/b) ))\
                                  ))

    #def zprime(b):
    #    eb = np.exp(b)
    #    logs = (L - logH + np.log(1/(1-eb) + 1/b) )
    #    sqr = np.sqrt(1 + 4*n*(eb-1) /( b*nL * logs))
    #    fac = 2*n/b**2/(1-eb+b)
    #    hyperfac = -2+b**2+2*np.cosh(b)-2*b*np.sinh(b)

    #    return fac * (-1-eb**2+eb*(2+b**2)+eb*logs*hyperfac) / sqr / logs**2




    maxbeta = None
    maxlogLL = -np.inf

    for b0 in beta0:
        try:
            #b = newton(z,b0,zprime)
            b = newton(z,b0)
            bfac = b/(np.exp(b)-1)
            xm = H * b / (1-bfac)
            a = 1 + nH/nL * bfac
            C = get_normalization_constant(a,xm,b)
            logLL = n*np.log(C) - a *nL*(L-np.log(xm)) - b*nH*(H/xm-1)
            if logLL > maxlogLL:
                maxlogLL = logLL
                maxbeta = b
                alpha = a
                beta = b
                xmin = xm
        except ValueError as e:
            pass
        except RuntimeError as e:
            pass

    if maxbeta is None:
        return None, None, None, None
    else:
        return alpha, xmin, beta, logLL


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

        try:
            a0, b0, logLL0 = alpha_beta_and_log_likelihood_fixed_xmin(data, xj)
            thesebetas = beta0 + (b0,)
        except RuntimeError as e:
            thesebetas = beta0
            logLL0 = None

        alpha, xmincand, beta, logLL = alpha_xmin_beta_and_log_likelihood_fixed_xmin_index(data,j,xmins,beta0=thesebetas)

        if logLL0 is not None and logLL is not None and logLL0 > logLL:
            alpha = a0
            beta = b0
            xmincand = xj

        if xmincand is None:
            continue

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
    return C * np.exp(-beta*(x/xmin - 1))

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
    return C*y/beta*np.exp(beta)*(1-np.exp(-beta*x/y))

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
    eb = np.exp(b)
    return (1/b + (-b + a*(2 - a + b))/((-2 + a)*(1 - a + b + (-1 + a)*eb)))*y

def second_moment(alpha,xmin,beta):
    assert(alpha > 3)
    y = xmin
    a = alpha
    b = beta
    eb = np.exp(b)
    return ((-1 + a)*(6 + b*(6 + b*(3 + b)) - 6*eb + a*(-2 - b*(2 + b) + 2*eb))*y**2)/\
            ((-3 + a)*b**2*(1 - a + b + (-1 + a)*eb))


def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

if __name__=="__main__":

    dataset = 'sample'
    if dataset == 'contacts':
        data = np.loadtxt('/Users/bfmaier/forschung/2023-Leo-Master/data/first_week_C_observations.txt')
        data = np.round(data)
        data = np.array(data,dtype=int)
        #data = data[data>0]
    elif dataset == 'sample':
        np.random.seed(20)
        alphatrue = 4
        betatrue = -3
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

    xmins = np.linspace(2,40,2001)

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
    uniquedat = uniquedat[np.where(np.logical_and(uniquedat>1,uniquedat<40))]
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


