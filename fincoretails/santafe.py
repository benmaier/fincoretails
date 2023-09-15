import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto
from scipy.optimize import bisect
from scipy.special import erfc

def fit_params(data, N):
    t, rho = t_and_rho(data,N)
    return t, rho, N

def pdf(x, t, rho, N):
    y0 = inverse_normal_cdf_approx(1-x/(N-1))
    return 1/(N-1) * np.sqrt((1-rho)/rho) * \
               np.exp(\
                    -(0.5-rho)/rho*y0**2\
                    +t*np.sqrt(1-rho)/rho*y0\
                    -t**2/2/rho
                  )

def t_and_rho(data, N):
    x = 1 - data/(N-1)
    yk = inverse_normal_cdf_approx(x)
    yk2 = yk**2
    n = len(data)
    yk = np.mean(yk)
    yk2 = np.mean(yk2)

    rho = (-yk**2 + yk2) / (-yk**2 + yk2 + 1)
    t = yk*np.sqrt(1-rho)

    return t, rho

def loglikelihoods(data, t, rho, N):
    return np.log(pdf(data, t, rho, N))

def loglikelihood(data, t, rho, N):
    return np.sum(loglikelihoods(data, t, rho, N))

def inverse_normal_cdf_approx(x):

    a0 = 2.30753
    a1 = 0.27061
    b1 = 0.99229
    b2 = 0.04481

    if hasattr(x, '__len__'):
        sign = np.ones_like(x)
        x = np.array(x)
        iR = np.where(x>0.5)[0]
        sign[iR] = -1
        x[iR] = 1-x[iR]
    else:
        if x > 0.5:
            x = 1-x
            sign = -1
        else:
            sign = 1

    s = np.sqrt(-2*np.log(x))

    inverse_normal_cdf = sign * ((a0 + a1*s)/(1+b1*s+b2*s**2) - s)
    return inverse_normal_cdf



if __name__=="__main__":
    from fincoretails import lognormal
    from bfmplot.tools import get_inset_axes
    import powerlaw
    from fincoretails import general_algpareto as npow
    from fincoretails import expon as expo


    data = np.loadtxt('/Users/bfmaier/forschung/2023-Leo-Master/data/first_week_C_observations.txt')
    data = np.round(data)
    data = np.array(data,dtype=int)
    old_data = np.array(data)
    data = data[data>0]

    name = 'contact data'

    fig, axs = pl.subplots(1,1,figsize=(10,10),sharey='row')
    fig.suptitle(name)

    N = 5_800_000
    t, rho = t_and_rho(data[data>0],N)
    SF_logL = loglikelihoods(data,t,rho,N).sum()

    mu, sigma = lognormal.mu_and_sigma(data)
    LN_logL = lognormal.loglikelihoods(data,mu,sigma).sum()

    beta = 2
    alpha, _xmin, PL_logL = npow.alpha_xmin_and_log_likelihood(data,beta=beta)

    rate = expo.rate(data)
    EX_logL = expo.loglikelihoods(data,rate).sum()

    print(f"{SF_logL=}")
    print(f"{LN_logL=}")
    print(f"{PL_logL=}")
    print(f"{EX_logL=}")

    #fit = powerlaw.Fit(data)
    #print(np.sum(fit.lognormal.loglikelihoods(data)))
    #print(f"{fit.power_law.alpha=}")
    #print(f"{fit.power_law.xmin=}")
    #print(f"{lognormal.logLL(data)=}")
    #print(f"{lognormal.mu_and_sigma(data)=}")
    #print(f"{fit.lognormal.mu=}")
    #print(f"{fit.lognormal.sigma=}")



    #ax = axs
    #ax[0].set_title(f'kernel: -x^beta beta={beta:4.2f}')

    #logLs = []
    #alphas = []
    #hatalpha = None
    #hatxmin = None
    #maxLL = None
    #for xmin in xmins:
    #    a, LL = alpha_and_log_likelihood_fixed_xmin(data, xmin, beta)
    #    if maxLL is None:
    #        maxLL = LL
    #    else:
    #        if LL > maxLL:
    #            maxLL = LL
    #            hatalpha = a
    #            hatxmin = xmin
    #    logLs.append(LL)
    #    alphas.append(a)

    #ax[0].plot(xmins, logLs)
    #ax[1].plot(xmins, alphas)

#ax[#1].set_xscale('log')
    #ax[0].set_xlabel('xmin')
    #ax[1].set_xlabel('xmin')
    #ax[0].set_ylabel('log-likelihood')
    #ax[1].set_ylabel('alpha')

    #a_, xmin_, LL_ = alpha_xmin_and_log_likelihood(data, beta)
    #ax[0].plot([xmin_,xmin_],ax[0].get_ylim(),'--')
    #ax[1].plot([xmin_,xmin_],ax[1].get_ylim(),'--')
    #ax[0].plot(xmins[[0,-1]],[LL_]*2,'--')
    #ax[1].plot(xmins[[0,-1]],[a_]*2,'--')
    #ax[0].set_xlim(xmins[[0,-1]])
    #ax[1].set_xlim(xmins[[0,-1]])




    _ax = axs

    if 'contact' in name:
        be = np.logspace(1,5)
        be = np.concatenate([np.arange(1,10)-0.5,be])
    else:
        be = np.logspace(-1,4)

    _ax.hist(data,be,density=True,alpha=0.5)

    x = np.logspace(np.log10(be[0]),np.log10(be[-1]),1001)
    _ax.plot(x, pdf(x, t,rho,N))
    _ax.plot(x, lognormal.pdf(x, mu, sigma))
    _ax.plot(x, npow.pdf(x, alpha, _xmin, beta),lw=3)
    #_ax.plot(x, expo.pdf(x, rate),lw=2)
    #_ax.plot(x, pdf(x, t,0.08,N))

    _ax.set_xscale('log')
    _ax.set_yscale('log')

    _ax.set_ylim(1e-10,1)

    #_ax.plot([xmin_,xmin_],_ax.get_ylim(),'--')

    _ax.set_xlabel('x')
    xmin = 10

    iax = get_inset_axes(_ax,width='37%',height='25%',loc='upper right')
    be = np.concatenate((np.arange(int(2.5*xmin))-0.5,
                         np.logspace(np.log10(int(2.5*xmin)+1), np.log10(data.max()), 50)
                         ))
    iax.hist(data, be, density=True, histtype='step',lw=2)
    iax.set_xlim(0,2.5*xmin)
    x = np.linspace(1,2.5*xmin,1001)
    #print(np.mean(thispdf))
    iax.plot(x, pdf(x, t, rho, N))
    iax.plot(x, lognormal.pdf(x, mu, sigma))
    iax.plot(x, npow.pdf(x, alpha, _xmin, beta),lw=3)
    #iax.plot(x, expo.pdf(x, rate),lw=2)
    #iax.plot(x, pdf(x, t, 0.08, N))

    #ax[0].text(0.05,0.1,
    #           f"a = {a_:4.2f} \nxmin = {xmin_:4.2f}\n logLL={int(LL_):d}",
    #           transform=ax[0].transAxes,
    #           ha='left',
    #           va='bottom',
    #           backgroundcolor='w',
    #        )



#axs[-1,0].set_ylabel('pdf')
#fig.tight_layout()
#fig.savefig(name+'.pdf')


    pl.show()

