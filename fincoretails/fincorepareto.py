import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pareto
from scipy.optimize import bisect, minimize
from scipy.special import erfc

from fincoretails.tools import general_quantile

def quantile(q, *parameters):
    return general_quantile(q, cdf, *parameters)

def fit_params(data, beta=None, beta_initial_values=(0.,3.)):
    if beta is not None:
        a, xm, logLL = alpha_xmin_and_log_likelihood(data, beta)
        return a, xm, beta
    else:
        curr_logLL = -np.inf
        curr_tuple = None
        for beta0 in beta_initial_values:
            a, xm, beta0, logLL = alpha_xmin_beta_logLL(data,beta0=beta0)
            if logLL > curr_logLL:
                curr_tuple = a, xm, beta0
                curr_logLL = logLL
        return curr_tuple

def loglikelihoods(data, alpha, xmin, beta):
    return np.log(pdf(data, alpha, xmin, beta))

def loglikelihood(data, *parameters):
    return np.sum(loglikelihoods(data, *parameters))

def get_normalization_constant(alpha, xmin, beta):
    assert(xmin>0)
    assert(alpha>1)
    assert(beta>=0)
    C = (alpha-1)*(beta+1) / xmin / (2*alpha*beta-beta+alpha)
    return C

def sample(Nsample, alpha, xmin, beta):

    C = get_normalization_constant(alpha, xmin, beta)
    Pcrit = C*xmin*(2-1/(beta+1))

    u = np.random.rand(Nsample)
    ndx = np.where(u<=Pcrit)[0]
    Nlow = len(ndx)
    ulow = u[ndx]

    xlow = []
    for v in ulow:
        F = lambda x: C*x*(2-1/(beta+1)*(x/xmin)**beta) - v
        xl = bisect(F, 0, xmin)
        xlow.append(xl)


    Nrest = Nsample - Nlow
    xhigh = pareto(alpha-1,scale=xmin).rvs(Nrest)
    samples = np.concatenate([xlow,xhigh])
    np.random.shuffle(samples)
    return samples

def alpha_and_log_likelihood_fixed_xmin(data, xmin, beta):
    n = len(data)
    Lambda = data[np.where(data>xmin)[0]]
    Eta = data[np.where(data<=xmin)[0]]
    nL = len(Lambda)
    nH = len(Eta)
    L = np.mean(np.log(Lambda/xmin))
    H = np.mean(np.log(2-(Eta/xmin)**beta))

    l = nL*L/n
    b = beta
    a = (3*b+1)/(4*b+2) + 1/(4*b+2) * np.sqrt((1+b)*(4+l+b*(8+l))/l)

    C = get_normalization_constant(a, xmin, b)
    logLL = n*np.log(C) - a*nL*L + nH*H

    return a, logLL

def alpha_xmin_and_log_likelihood_fixed_xmin_index(data, j, beta, xmins=None):

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
    b = beta

    def dlogLLdxmin(xcand):
        L = np.mean(np.log(Lambda/xcand))
        Hprime = np.mean(1/(2*(xcand/Eta)**b-1))

        l = nL*L/n
        a = (3*b+1)/(4*b+2) + 1/(4*b+2) * np.sqrt((1+b)*(4+l+b*(8+l))/l)

        return -n + a*nL + b*nH*Hprime

    try:
        xmin_new = bisect(dlogLLdxmin,xleft,xright)
    except ValueError as e:
        return None, None, None

    a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xmin_new, beta)

    return a, xmin_new, logLL

def alpha_xmin_and_log_likelihood(data, beta):

    n = len(data)
    xmins = np.sort(np.unique(data))

    maxLogLL = -np.inf
    alphas, nLs, Ls, logLLs = [], [],[], []
    sampled_xmins = []
    for j, xj in enumerate(xmins):
        a, logLL = alpha_and_log_likelihood_fixed_xmin(data, xj, beta)
        alphas.append(a)
        logLLs.append(logLL)
        sampled_xmins.append(xj)

        if logLL > maxLogLL:
            maxLogLL = logLL
        else:
            break

    if j == 1:
        check_indices = [0]
    else:
        check_indices = [j-1, j-2]

    current_max_tuple = alphas[j-1], sampled_xmins[j-1], logLLs[j-1]

    for k in check_indices:
        alpha, xmincand, logLL = alpha_xmin_and_log_likelihood_fixed_xmin_index(data,k,beta,xmins)

        if logLL is not None and logLL > maxLogLL:
            #print("it happened!")
            maxLogLL = logLL
            current_max_tuple = alpha, xmincand, logLL

    return current_max_tuple

def alpha_xmin_beta_logLL(data,beta0=2):

    func = lambda b: -alpha_xmin_and_log_likelihood(data, b)[-1]

    res = minimize(func, (beta0,), bounds=[(0,np.inf)])
    beta = res.x[0]
    alpha, xmin, logLL = alpha_xmin_and_log_likelihood(data, beta)

    return alpha, xmin, beta, logLL





def mean(alpha,xmin,beta):
    assert(alpha > 2)
    xm = xmin
    return xm*(alpha - 1)*(beta + 1)*(beta + (alpha - 2)*(beta + 1) + 2)/((alpha - 2)*(beta + 2)*(2*alpha*beta + alpha - beta))

def median(alpha,xmin,beta):
    a, xm, b = alpha, xmin, beta
    return quantile(0.5, a, xm, b)

def second_moment(alpha,xmin,beta):
    assert(alpha > 3)
    xm = xmin
    return xm**2*(alpha - 1)*(beta + 1)*(3*beta + (alpha - 3)*(2*beta + 3) + 9)/(3*(alpha - 3)*(beta + 3)*(2*alpha*beta + alpha - beta))

def variance(*args,**kwargs):
    return second_moment(*args,**kwargs) - mean(*args,**kwargs)**2

def neighbor_degree(*args,**kwargs):
    return second_moment(*args,**kwargs)/mean(*args,**kwargs)

def ccdf(x, *args,**kwargs):
    return 1-cdf(x, *args,**kwargs)

def pdf_left(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    return C * (2-(x/xmin)**beta)

def pdf_right(x, alpha, xmin, beta, C):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    return C * (xmin/x)**alpha

def pdf(x, alpha, xmin, beta):
    """"""
    C = get_normalization_constant(alpha, xmin, beta)
    result = np.piecewise(np.asanyarray(x,dtype=float),
                          (
                              x<0,
                              x<=xmin,
                              x>xmin
                          ),
                          (
                              lambda x, *args: 0,
                              pdf_left,
                              pdf_right,
                          ),
                          alpha, xmin, beta, C,
                         )
    return result

def cdf_left(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    return C*x * (2-1/(beta+1)*(x/xmin)**beta)

def cdf_right(x, alpha, xmin, beta, C=None):
    if C is None:
        C = get_normalization_constant(alpha, xmin, beta)
    Pcrit = C*xmin * (2-1/(beta+1))
    return C*xmin/(alpha-1) * (1-(xmin/x)**(alpha-1)) + Pcrit

def cdf(x, alpha, xmin, beta):
    """"""
    C = get_normalization_constant(alpha, xmin, beta)
    result = np.piecewise(np.asanyarray(x,dtype=float),
                          (
                              x<0,
                              x<=xmin,
                              x>xmin
                          ),
                          (
                              lambda x, *args: 0,
                              cdf_left,
                              cdf_right,
                          ),
                          alpha, xmin, beta, C,
                         )
    return result






if __name__=="__main__":
    import lognormal
    from bfmplot.tools import get_inset_axes
    import powerlaw


    data = np.loadtxt('/Users/bfmaier/forschung/2023-Leo-Master/data/subproject_estimate_powerlaw/first_week_C_observations.txt')
    data = np.round(data)
    data = np.array(data,dtype=int)

    betas = [0,1,2]
    xmin = 4.5
    alpha = 2.1
    Nsample = 8001
    datasets = [data]

    data = datasets[0]
    _,_,betaopt,_ = alpha_xmin_beta_logLL(data)
    betas.append(betaopt)


    datasets_names = ['real world contact data']

    for ibeta, beta in enumerate(betas):
        datasets.append(sample(Nsample, alpha, xmin, beta))
        datasets_names.append(f'synth. data, {xmin=:4.2f}, {alpha=:4.2f}, {beta=:4.2f}')



    xmins = np.linspace(2,12,1001)

    for idata, (data, name) in enumerate(zip(datasets, datasets_names)):

        fig, axs = pl.subplots(3,4,figsize=(10,10),sharey='row')
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
                a, LL = alpha_and_log_likelihood_fixed_xmin(data, xmin, beta)
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

            if ibeta == 3:
                a_, xmin_, beta, LL_ = alpha_xmin_beta_logLL(data, beta0=0)
            else:
                a_, xmin_, LL_ = alpha_xmin_and_log_likelihood(data, beta)

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
            _ax.plot(x, pdf(x, a_,xmin_,beta))

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
            iax.plot(x, pdf(x, a_,xmin_,beta))

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


