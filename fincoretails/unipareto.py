import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto

from fincoretails.general_algpareto import (
        pdf as general_pdf,
        cdf as general_cdf,
        get_normalization_constant as general_normalization_constant,
        mean as general_mean,
        variance as general_variance,
        median as general_median,
    )

def fit_params(data):
    a, xm, logLL = alpha_xmin_and_log_likelihood(data)
    return a, xm

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def get_normalization_constant(alpha, xmin):
    return general_normalization_constant(alpha, xmin, alpha)

def pdf(x, alpha, xmin):
    return general_pdf(x, alpha, xmin, beta=0)

def cdf(x, alpha, xmin):
    return general_cdf(x, alpha, xmin, beta=0)

def loglikelihoods(data, alpha, xmin):
    return np.log(pdf(data, alpha, xmin))

def sample(Nsample, alpha, xmin):

    beta = 0

    C = get_normalization_constant(alpha, xmin)
    Pcrit = C*xmin

    u = np.random.rand(Nsample)
    ndx = np.where(u<=Pcrit)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    xlow = ulow/C

    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)
    return samples

def alpha_and_log_likelihood_fixed_xmin(data, xmin, return_nLambda_and_logx=False):
    n = len(data)
    Lambda = data[np.where(data>xmin)[0]]
    nL = len(Lambda)
    L = np.mean(np.log(Lambda))

    a = 0.5 + np.sqrt(0.25 + n/nL * 1/(L-np.log(xmin)))

    logLL = n * np.log( (a-1)/a / xmin ) - a*nL*(L-np.log(xmin))

    if not return_nLambda_and_logx:
        return a, logLL
    else:
        return a, logLL, nL, L

def alpha_xmin_and_log_likelihood(data):

    n = len(data)
    xmins = np.sort(np.unique(data))

    maxLogLL = -np.inf
    alphas, nLs, Ls, logLLs = [], [],[], []
    sampled_xmins = []
    for j, xj in enumerate(xmins):
        a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xj)
        alphas.append(a)
        logLLs.append(logLL)
        sampled_xmins.append(xj)

        if logLL > maxLogLL:
            maxLogLL = logLL
        else:
            break
    current_max_tuple = alphas[j-1], sampled_xmins[j-1], logLLs[j-1]

    return current_max_tuple

def mean(alpha,xmin):
    a, xm, b = alpha, xmin, 0
    return general_mean(a, xm, b)

def median(alpha,xmin,beta):
    a, xm, b = alpha, xmin, 0
    return general_median(a, xm, b)

def variance(alpha,xmin,beta):
    a, xm, b = alpha, xmin, 0
    return general_variance(a, xm, b)

def second_moment(*args,**kwargs):
    return variance(*args,**kwargs) + mean(*args,**kwargs)**2

def neighbor_degree(*args,**kwargs):
    return second_moment(*args,**kwargs)/mean(*args,**kwargs)

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def quantile(q,alpha,xmin):
    beta = 0
    return general_quantile(q,cdf,alpha,xmin,beta)







if __name__=="__main__":
    import lognormal
    from bfmplot.tools import get_inset_axes
    import powerlaw


    data = np.loadtxt('first_week_C_observations.txt')
    data = np.round(data)
    data = np.array(data,dtype=int)

    betas = [0]
    xmin = 4.5
    alpha = 2.1
    Nsample = 8001
    datasets = [data, sample(Nsample, alpha, xmin)]


    datasets_names = ['real world contact data']

    for ibeta, beta in enumerate(betas):
        datasets.append(sample(Nsample, alpha, xmin))
        datasets_names.append(f'synth. data, {xmin=:4.2f}, {alpha=:4.2f}, {beta=:4.2f}')



    xmins = np.linspace(2,12,1001)

    for idata, (data, name) in enumerate(zip(datasets, datasets_names)):

        fig, axs = pl.subplots(3,1,figsize=(10,10),sharey='row')
        axs = axs.reshape(3,1)
        fig.suptitle(name)

        fit = powerlaw.Fit(data)
        print(np.sum(fit.lognormal.loglikelihoods(data)))
        print(f"{fit.power_law.alpha=}")
        print(f"{fit.power_law.xmin=}")
        print(f"{lognormal.logLL(data)=}")
        print(f"{lognormal.mu_and_sigma(data)=}")
        print(f"{fit.lognormal.mu=}")
        print(f"{fit.lognormal.sigma=}")


        for ibeta, beta in enumerate(betas):

            ax = axs[:,ibeta]

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

            ax[0].plot(xmins, logLs)
            ax[1].plot(xmins, alphas)

#ax[1].set_xscale('log')
            ax[0].set_xlabel('xmin')
            ax[1].set_xlabel('xmin')
            ax[0].set_ylabel('log-likelihood')
            ax[1].set_ylabel('alpha')

            a_, xmin_, LL_ = alpha_xmin_and_log_likelihood(data)

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
            _ax.plot(x, pdf(x, a_,xmin_))

            _ax.set_xscale('log')
            _ax.set_yscale('log')

            _ax.plot([xmin_,xmin_],_ax.get_ylim(),'--')

            _ax.set_xlabel('x')

            iax = get_inset_axes(_ax,width='37%',height='25%',loc='upper right')
            be = np.concatenate((np.arange(int(2.5*xmin)),
                                 np.logspace(np.log10(int(2.5*xmin)+1), np.log10(data.max()), 50)
                                 ))
            iax.hist(data, be, density=True,alpha=0.5)
            iax.set_xlim(0,2.5*xmin_)
            x = np.linspace(0,2.5*xmin_,1001)
            iax.plot(x, pdf(x, a_,xmin_))

            ax[0].text(0.05,0.1,
                       f"a = {a_:4.2f} \nxmin = {xmin_:4.2f}\n logLL={int(LL_):d}",
                       transform=ax[0].transAxes,
                       ha='left',
                       va='bottom',
                       backgroundcolor='w',
                    )



        axs[-1,0].set_ylabel('pdf')
        fig.tight_layout()
        fig.savefig(name+'.pdf')

    

    pl.show()


