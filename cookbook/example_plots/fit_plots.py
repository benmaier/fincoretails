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
            print(X)
        except TypeError as e:
            X = model.sample(Nsample, alpha, xmin)

        print(alpha, xmin, beta)


        pars = model.fit_params(X)
        hatxmin = pars[0]

        print(pars)
        pdflin, _ = np.histogram(X, bins=linbe, density=True)
        pdflog, _ = np.histogram(X, bins=logbe, density=True)

        axlin.plot(xlinpdf, pdflin, 's',mfc='w', color=c[i+coloroffset])
        axlog.plot(xlogpdf, pdflog, 's',mfc='w', color=c[i+coloroffset])
        #axlin.hist(X, bins=linbe, density=True, alpha=0.2, color=c[i])
        #axlog.hist(X, bins=logbe, density=True, alpha=0.2, color=c[i])

        axlin.plot(xlin, model.pdf(xlin, *pars),label=lbl, color=c[i+coloroffset])
        axlog.plot(xlog, model.pdf(xlog, *pars),label=lbl, color=c[i+coloroffset])

    axlin.plot([xmin,xmin],ylimlin,'--',lw=1,color='#999999')
    axlog.plot([xmin,xmin],ylimlog,'--',lw=1,color='#999999')
    if legloc is not None:
        axlin.legend(loc=legloc)

if __name__ == "__main__":
    np.random.seed(5)
    from fincoretails import general_algpareto, general_expareto, general_powpareto, algpareto, expareto, powpareto

    fig, ax = pl.subplots(1,3,figsize=(10,3))

    betas = [0,1]
    betalabels = map(lambda x: 'β = '+x,['0', '1','2'])
    plot(ax[2], general_algpareto, betas, betalabels)
    ax[2].set_title('(c) alg-Pareto',loc='left')

    betas = [-0.5,1]
    betalabels = map(lambda x: 'β = '+x,['-1/2', '1','2'])
    plot(ax[1], general_expareto, betas, betalabels)#,ylimlog=(1e-3,1))
    ax[1].set_title('(b) exp-Pareto',loc='left')

    betas = [-0.5,1]
    betalabels = map(lambda x: 'β = '+x,['-1/2', '1','2'])
    plot(ax[0], general_powpareto, betas, betalabels, legloc=(0.09,0.75))
    ax[0].set_title('(a) pow-Pareto',loc='left')

    fig.tight_layout()

    fig.savefig('fit_plots_0.pdf')

    fig, ax = pl.subplots(1,3,figsize=(10,3))
    betas = [2]
    betalabels = map(lambda x: 'β = '+x,['2'])
    plot(ax[2], algpareto, betas, betalabels,coloroffset=2)
    ax[2].set_title('(c) forced alg-Pareto',loc='left')

    betas = [2]
    betalabels = map(lambda x: 'β = '+x,['2'])
    plot(ax[1], expareto, betas, betalabels, coloroffset=2)#,ylimlog=(1e-3,1))
    ax[1].set_title('(b) forced exp-Pareto',loc='left')

    betas = [2]
    betalabels = map(lambda x: 'β = '+x,['2'])
    plot(ax[0], powpareto, betas, betalabels, legloc=(0.09,0.75), coloroffset=2)
    ax[0].set_title('(a) forced pow-Pareto',loc='left')

    fig.tight_layout()

    fig.savefig('fit_plots_1.pdf')

    pl.show()

    pl.show()
