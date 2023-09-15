import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto
from scipy.optimize import newton, minimize, brentq

from fincoretails.tools import general_quantile
from fincoretails.general_powpareto import (
        pdf as general_pdf,
        cdf as general_cdf,
        get_normalization_constant as general_normalization_constant,
        mean as general_mean,
        variance as general_variance,
        median as general_median,
        sample as general_sample,
    )

def quantile(q, *parameters):
    return general_quantile(q, cdf, *parameters)

def fit_params(data, minxmin=1.001):
    a, xm, logLL = alpha_xmin_and_log_likelihood(data, minxmin=minxmin)
    return a, xm

def get_normalization_constant(alpha, xmin):
    return general_normalization_constant(alpha, xmin, alpha)

def pdf(x, alpha, xmin):
    return general_pdf(x, alpha, xmin, alpha)

def cdf(x, alpha, xmin):
    return general_cdf(x, alpha, xmin, alpha)

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def loglikelihoods(data, alpha, xmin):
    return np.log(pdf(data, alpha, xmin))

def sample(Nsample, alpha, xmin):
    return general_sample(Nsample, alpha, xmin, alpha)

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def mean(alpha,xmin):
    a, xm, b = alpha, xmin, alpha
    return general_mean(a, xm, b)

def median(alpha,xmin):
    a, xm, b = alpha, xmin, alpha
    return general_median(a, xm, b)

def variance(alpha,xmin):
    a, xm, b = alpha, xmin, alpha
    return general_variance(a, xm, b)

def second_moment(*args,**kwargs):
    return variance(*args,**kwargs) + mean(*args,**kwargs)**2

def neighbor_degree(*args,**kwargs):
    return second_moment(*args,**kwargs)/mean(*args,**kwargs)

def alpha_and_log_likelihood_fixed_xmin(data, xmin, a0=2):
    n = len(data)

    Lambda = data[np.where(data>xmin)[0]]
    Eta = data[np.where(data<=xmin)[0]]
    logL = np.mean(np.log(Lambda))
    logH = np.mean(np.log(Eta))
    logy = np.log(xmin)
    nL = len(Lambda)
    nH = len(Eta)

    Sqrt = np.sqrt
    logXoY = logL - logy
    logYoS = logy - logH

    offs = -nL*logXoY - (n - nL) *logYoS
    z = lambda a: (n*(1 + a**2))/(-a + a**3) + offs
    zPrime = lambda a: -((n*(-1 + 4*a**2 + a**4))/(a**2*(-1 + a**2)**2))

    try:
        a = newton(z,a0,zPrime)
    except RuntimeError as e:
        return None, None

    hata = np.nan
    logLL = np.nan
    if a > 1:
        C = get_normalization_constant(a, xmin)
        logLL = n*np.log(C) - nL*a*logXoY - nH*a*logYoS
        hata = a

    return hata, logLL

def alpha_xmin_and_log_likelihood_fixed_xmin_index(data, j, xmins=None):

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

    if nH == nL:
        return None, None, None


    a = n/(2*nL-n)
    logy = 1 - n**2/(2*n*nL - 2*nL**2) + ((n - nL)*logH - nL*logL)/(n - 2*nL)
    y = np.exp(logy)

    if not np.isnan(a) and a > 1 and y>0:
        C = get_normalization_constant(a,y)
        logLL = n*np.log(C) - nL*a*(logL-logy) - nH*a*(logy-logH)

        _a, _logLL = alpha_and_log_likelihood_fixed_xmin(data, xleft,a0=a)

        if _logLL is not None:
            if (_logLL > logLL) or (a < 1) or (y<xleft) or (y>=xright):
                return _a, xleft, _logLL
        elif a>1 and y>=xleft and y<xright:
            return a, y, logLL
        else:
            return None, None, None
    else:
        _a, _logLL = alpha_and_log_likelihood_fixed_xmin(data, xleft)
        return _a, xleft, _logLL



def alpha_xmin_and_log_likelihood(data, stop_at_first_max=False,minxmin=None,maxxmin=None):

    n = len(data)
    xmins = np.sort(np.unique(data))

    if minxmin is None:
        minxmin = xmins[0]
    xmins = xmins[xmins>minxmin]
    if maxxmin is not None:
        xmins = xmins[xmins<=maxxmin]

    maxlogLL = -np.inf
    hatalpha = np.nan
    hatxmin = np.nan
    for j, xj in enumerate(xmins[:-1]):

        alpha, xmincand, logLL = alpha_xmin_and_log_likelihood_fixed_xmin_index(data,j,xmins)

        if alpha is None:
            continue

        if xmincand >= xmins[j] and xmincand <xmins[j+1] and logLL > maxlogLL:
            hatxmin = xmincand
            hatalpha = alpha
            hatlogLL = logLL
            maxlogLL = logLL

            if stop_at_first_max:
                break

    return hatalpha, hatxmin, maxlogLL




if __name__=="__main__":
    import lognormal
    from bfmplot.tools import get_inset_axes
    import powerlaw

    data = np.loadtxt('/Users/bfmaier/forschung/2023-Leo-Master/data/first_week_C_observations.txt')

    data = np.round(data)
    data = np.array(data,dtype=int)

    betas = [0]
    xmin0 = 10
    alpha0 = 1.9
    Nsample = 8001
    datasets = []


    #datasets_names = ['real world contact data']
    datasets_names = []

    for ibeta, beta in enumerate(betas):
        datasets.append(sample(Nsample, alpha0, xmin0))
        datasets_names.append(f'synth. data, {xmin0=:4.2f}, {alpha0=:4.2f}')



    xmins = np.linspace(2,12,1001)

    for idata, (data, name) in enumerate(zip(datasets, datasets_names)):

        fig, axs = pl.subplots(3,1,figsize=(5,10),sharey='row')
        axs = axs.reshape(3,1)
        fig.suptitle(name)

        #fit = powerlaw.Fit(data)
        #print(np.sum(fit.lognormal.loglikelihoods(data)))
        #print(f"{fit.power_law.alpha=}")
        #print(f"{fit.power_law.xmin=}")
        #print(f"{lognormal.logLL(data)=}")
        #print(f"{lognormal.mu_and_sigma(data)=}")
        #print(f"{fit.lognormal.mu=}")
        #print(f"{fit.lognormal.sigma=}")


        print("=========", name)

        for ibeta, beta in enumerate(betas):

            ax = axs[:,ibeta]

            logLs = []
            alphas = []
            hatalpha = None
            hatxmin = None
            maxLL = None

            print("scanning artificial xmins")
            for xmin in xmins:
                a, LL = alpha_and_log_likelihood_fixed_xmin(data, xmin)
                if maxLL is None:
                    maxLL = LL
                else:
                    if LL is not None and LL > maxLL:
                        maxLL = LL
                        hatalpha = a
                        hatxmin = xmin
                logLs.append(LL)
                alphas.append(a)


            ax[0].plot(xmins, logLs)
            ax[1].plot(xmins, alphas)

#ax[1].set_xscale('log')
            ax[0].set_xlabel('xmin')
            ax[1].set_xlabel('xmin')
            ax[0].set_ylabel('log-likelihood')
            ax[1].set_ylabel('alpha')

            print("trying to find alpha and xmin")
            a_, xmin_, LL_ = alpha_xmin_and_log_likelihood(data)
            print(a_, xmin_, LL_)

            ax[0].set_title(f'kernel: -x^beta beta={beta:4.2f}')
            ax[0].plot([xmin_,xmin_],ax[0].get_ylim(),'--')
            ax[1].plot([xmin_,xmin_],ax[1].get_ylim(),'--')
            ax[0].plot(xmins[[0,-1]],[LL_]*2,'--')
            ax[1].plot(xmins[[0,-1]],[a_]*2,'--')
            ax[0].set_xlim(xmins[[0,-1]])
            ax[1].set_xlim(xmins[[0,-1]])

            _ax = ax[-1]

            if 'contact' in name:
                be = np.logspace(1,5)
                be = np.concatenate([np.arange(1,10),be])
            else:
                be = np.logspace(-1,4)
            _ax.hist(data,be,density=True,alpha=0.5)

            x = np.logspace(np.log10(be[0]),np.log10(be[-1]),1001)
            #_ax.plot(x, pdf(x, a_,xmin_,))

            _ax.set_xscale('log')
            _ax.set_yscale('log')

            _ax.plot([xmin_,xmin_],_ax.get_ylim(),'--')

            _ax.set_xlabel('x')

            iax = get_inset_axes(_ax,width='37%',height='25%',loc='upper right')
            be = np.concatenate((np.arange(int(2.5*xmin)),
                                 np.logspace(np.log10(int(2.5*xmin)+1), np.log10(data.max()), 50)
                                 ))
            iax.hist(data, be, density=True,alpha=0.5)
            iax.set_xlim(0,20)
            x = np.linspace(0,20,1001)
            #iax.plot(x, pdf(x, a_,xmin_))
            if not 'world' in name:
                iax.plot(x, pdf(x, alpha0,xmin0))

            #ax[0].text(0.05,0.1,
            #           f"a = {a_:4.2f} \nxmin = {xmin_:4.2f}\n logLL={int(LL_):d}",
            #           transform=ax[0].transAxes,
            #           ha='left',
            #           va='bottom',
            #           backgroundcolor='w',
            #        )



        axs[-1,0].set_ylabel('pdf')
        fig.tight_layout()
        fig.savefig(name+'.pdf')

    pl.show()


