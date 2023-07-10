import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto
from scipy.optimize import newton, minimize, brentq

from fincoretails.tools import general_quantile

def quantile(q, *parameters):
    return general_quantile(q, cdf, *parameters)

def fit_params(data, minxmin=1.001,alpha0=2):
    a, xm, logLL = alpha_xmin_and_log_likelihood(data, minxmin=minxmin,alpha0=alpha0)
    return a, xm

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def get_normalization_constant(alpha, xmin):
    """"""
    beta = alpha
    assert(xmin>0)
    assert(alpha>1)
    assert(beta>=0)
    C = (alpha-1)*(beta+1) / xmin / (2*alpha*beta-beta+alpha)
    return C

def pdf(x, alpha, xmin):
    """"""

    beta = alpha

    C = get_normalization_constant(alpha, xmin)
    if hasattr(x, '__len__'):
        x = np.array(x)
        cond = x<=xmin
        i0 = np.where(cond)[0]
        i1 = np.where(np.logical_not(cond))[0]
        result = np.zeros_like(x,dtype=float)
        result[i0] = C * (2-(x[i0]/xmin)**beta)
        result[i1] = C * (xmin/x[i1])**alpha
        return result
    else:
        if x <= xmin:
            return C * (2-(x/xmin)**beta)
        else:
            return C * (xmin/x)**alpha

def loglikelihoods(data, alpha, xmin, beta):
    return np.log(pdf(data, alpha, xmin, beta))

def sample(Nsample, alpha, xmin):

    beta = alpha
    C = get_normalization_constant(alpha, xmin)
    Pcrit = C*xmin*(2-1/(beta+1))

    u = np.random.rand(Nsample)
    ndx = np.where(u<=Pcrit)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    xlow = []
    for v in ulow:
        F = lambda x: C*x*(2-1/(beta+1)*(x/xmin)**beta) - v
        xl = brentq(F, 0, xmin)
        xlow.append(xl)


    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)
    return samples

def alpha_and_log_likelihood_fixed_xmin(data, xmin,alpha0=1.5):
    n = len(data)
    Lambda = data[np.where(data>xmin)[0]]
    Eta = data[np.where(data<=xmin)[0]]
    nL = len(Lambda)
    nH = len(Eta)
    L = np.mean(np.log(Lambda/xmin))

    def dlogLLda(a):
        H = np.mean(np.log(Eta/xmin) / (2*(Eta/xmin)**(-a)-1))
        return -nL *L + 2*n*a/(a**2-1) - 2*n/a - nH * H

    try:
        anew = newton(dlogLLda, alpha0)
    except ValueError as e:
        return None, None
    except RuntimeError as e:
        return None, None

    a = anew
    C = get_normalization_constant(a, xmin)
    logLL = n*np.log(C) - a*nL*L + nH * np.mean(np.log(2-(Eta/xmin)**a))

    return a, logLL

def alpha_xmin_and_log_likelihood_fixed_xmin_index(data, j, a0, xmins=None):

    if xmins is None:
        xmins = np.sort(np.unique(data))

    n = len(data)

    xleft = xmins[j]
    xright = xmins[j+1]

    Lambda = data[np.where(data>xleft)[0]]
    Eta = data[np.where(data<=xleft)[0]]
    #EtaPrime = data[np.where(data<xleft)[0]]
    nL = len(Lambda)
    nH = len(Eta)
    L = np.mean(np.log(Lambda))

    def minusloglikeli(params):
        a, x = params
        C = get_normalization_constant(a,x)
        H = np.mean(np.log(2-(Eta/x)**a))
        L_ = L - np.log(x)
        return - (n*np.log(C) - a * nL * L_ + nH * H)

    try:
        res = minimize(minusloglikeli,
                       x0=[a0,0.5*(xleft+xright)],
                       bounds=[(1.0001,np.inf),(xleft, xright)],
                       )
    except ValueError as e:
        return None, None, None

    a, x = res.x

    logLL = -minusloglikeli(res.x)

    return a, x, logLL

def alpha_xmin_and_log_likelihood(data, minxmin=1.001, alpha0=2):

    n = len(data)
    xmins = np.sort(np.unique(data))
    xmins = xmins[xmins>=minxmin]

    maxLogLL = -np.inf
    alphas, nLs, Ls, logLLs = [], [],[], []
    sampled_xmins = []

    for j, xj in enumerate(xmins):
        a, xmincand, logLL = alpha_xmin_and_log_likelihood_fixed_xmin_index(data,j,alpha0,xmins)
        alphas.append(a)
        logLLs.append(logLL)
        sampled_xmins.append(xmincand)

        if logLL > maxLogLL:
            maxLogLL = logLL
        else:
            break


    current_max_tuple = alphas[j-1], sampled_xmins[j-1], logLLs[j-1]

    return current_max_tuple


def cdf(x,alpha,xmin):
    beta = alpha
    C = get_normalization_constant(alpha, xmin)
    Pcrit = C*xmin * (2-1/(beta+1))
    if hasattr(x, '__len__'):
        x = np.array(x)
        cond = x<=xmin
        i0 = np.where(cond)[0]
        i1 = np.where(np.logical_not(cond))[0]
        result = np.zeros_like(x)
        result[i0] = C*x[i0]* (2-1/(beta+1)*(x[i0]/xmin)**beta)
        result[i1] = C*xmin/(alpha-1) * (1-(xmin/x[i1])**(alpha-1)) + Pcrit
    else:
        if x <= xmin:
            return C*x * (2-1/(beta+1)*(x/xmin)**beta)
        else:
            return C*xmin/(alpha-1) * (1-(xmin/x)**(alpha-1)) + Pcrit

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)






if __name__=="__main__":
    import lognormal
    from bfmplot.tools import get_inset_axes
    import powerlaw


    data = np.loadtxt('first_week_C_observations.txt')
    data = np.round(data)
    data = np.array(data,dtype=int)

    betas = [0]
    xmin0 = 6
    alpha0 = 1.9
    Nsample = 8001
    datasets = [data]


    datasets_names = ['real world contact data']

    for ibeta, beta in enumerate(betas):
        datasets.append(sample(Nsample, alpha0, xmin0))
        datasets_names.append(f'synth. data, {xmin0=:4.2f}, {alpha0=:4.2f}')



    xmins = np.linspace(1.1,12,1001)

    for idata, (data, name) in enumerate(zip(datasets, datasets_names)):

        fig, axs = pl.subplots(3,1,figsize=(5,10),sharey='row')
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
            _ax.plot(x, pdf(x, a_,xmin_,))

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
            iax.plot(x, pdf(x, a_,xmin_))
            if not 'world' in name:
                iax.plot(x, pdf(x, alpha0,xmin0))

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


