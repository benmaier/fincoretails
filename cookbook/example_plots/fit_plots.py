import numpy as np
import matplotlib.pyplot as pl
from bfmplot.sequences import bochenska2 as c

def plot(axlin, model, betas, betalabels, xmin=10,alpha=2,ylimlin=(0.0,0.1),ylimlog=(1e-3,1e-1),legloc='lower left',
         inset_bounds= (0.62,0.6,0.38,0.4),
         Nsample=5000,
         coloroffset=0,
         ):
    xlin = np.linspace(0,2.5*xmin,1001)
    xlog = np.logspace(np.log10(xmin/10),np.log10(xmin*10),1001)
    logbe = np.logspace(-2,np.log10(xmin*10),51)
    xlogpdf = np.sqrt(logbe[1:] * logbe[:-1])
    linbe = np.linspace(0,10*xmin,101)
    xlinpdf = 0.5*(linbe[1:] + linbe[:-1])

    axlog = axlin.inset_axes(inset_bounds)
    axlog.set_xscale('log')
    axlog.set_yscale('log')
    axlog.set_xlim(xlog[0],xlog[-1])
    axlog.set_ylim(ylimlog)
    axlin.set_ylim(ylimlin)
    axlin.set_xlim(xlin[0],xlin[-1])
    axlin.set_xlabel('x',loc='right')
    axlin.set_ylabel('pdf',loc='top')
    for i, (beta, lbl) in enumerate(zip(betas, betalabels)):

        try:
            X = model.sample(Nsample, alpha, xmin, beta)
            isforced = False
        except TypeError as e:
            X = model.sample(Nsample, alpha, xmin)
            isforced = True

        print(alpha, xmin, beta)


        pars = model.fit_params(X)
        hatxmin = pars[1]
        alpha = pars[0]

        if isforced:
            label = f"α̂ = {alpha:4.2f}"
        else:
            beta = pars[2]
            label = f"β̂ = {beta:4.2f}\nα̂ = {alpha:4.2f}"

        print(pars)
        pdflin, _ = np.histogram(X, bins=linbe, density=True)
        pdflog, _ = np.histogram(X, bins=logbe, density=True)

        axlin.plot(xlinpdf, pdflin, 's',mfc='w', color=c[i+coloroffset])
        axlog.plot(xlogpdf, pdflog, 's',mfc='w', color=c[i+coloroffset])
        #axlin.hist(X, bins=linbe, density=True, alpha=0.2, color=c[i])
        #axlog.hist(X, bins=logbe, density=True, alpha=0.2, color=c[i])

        axlin.plot(xlin, model.pdf(xlin, *pars),label=label, color=c[i+coloroffset])
        axlog.plot(xlog, model.pdf(xlog, *pars),label={}, color=c[i+coloroffset])

        axlin.plot([hatxmin]*2,ylimlin,'--',lw=1,color=c[i+coloroffset])
        axlog.plot([hatxmin]*2,ylimlog,'--',lw=1,color=c[i+coloroffset])
    if legloc is not None:
        axlin.legend(loc=legloc)

if __name__ == "__main__":
    np.random.seed(5)
    from fincoretails import (
            general_algpareto,
            general_expareto,
            general_powpareto,
            algpareto,
            expareto,
            powpareto,
        )

    fig, ax = pl.subplots(1,3,figsize=(10,3))

    betas = [0,1]
    betalabels = map(lambda x: 'β = '+x,['0', '1','2'])
    plot(ax[2], general_algpareto, betas, betalabels)
    ax[2].set_title('(c) alg-Pareto',loc='left')

    betas = [-0.5,1]
    betalabels = map(lambda x: 'β = '+x,['-1/2', '1','2'])
    plot(ax[1], general_expareto, betas, betalabels,legloc=(0.03,0.03))#,ylimlog=(1e-3,1))
    ax[1].set_title('(b) exp-Pareto',loc='left')

    betas = [-0.5,1]
    betalabels = map(lambda x: 'β = '+x,['-1/2', '1','2'])
    plot(ax[0], general_powpareto, betas, betalabels, legloc=(0.07,0.67))
    ax[0].set_title('(a) pow-Pareto',loc='left')

    for a in ax:
        a.set_xticks([0,10,20,25])
        a.set_yticks([0,0.05,0.1])

    fig.tight_layout()

    fig.savefig('fit_plots_0.pdf')

    fig, ax = pl.subplots(1,3,figsize=(10,3))
    betas = [2]
    betalabels = map(lambda x: 'β = '+x,['2'])
    plot(ax[2], algpareto, betas, betalabels,coloroffset=2)
    ax[2].set_title('(f) forced alg-Pareto',loc='left')

    betas = [2]
    betalabels = map(lambda x: 'β = '+x,['2'])
    plot(ax[1], expareto, betas, betalabels, coloroffset=2,ylimlog=(1e-3,.5))
    ax[1].set_title('(e) forced exp-Pareto',loc='left')

    betas = [2]
    betalabels = map(lambda x: 'β = '+x,['2'])
    plot(ax[0], powpareto, betas, betalabels, legloc='upper left', coloroffset=2)
    ax[0].set_title('(d) forced pow-Pareto',loc='left')

    for a in ax:
        a.set_xticks([0,10,20,25])
        a.set_yticks([0,0.05,0.1])

    fig.tight_layout()

    fig.savefig('fit_plots_1.pdf')

    pl.show()

    pl.show()
