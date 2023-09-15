import numpy as np
import matplotlib.pyplot as pl

def plot(axlin, model, betas, betalabels, xmin=10,alpha=2,ylimlin=(0.0,0.1),ylimlog=(1e-3,1e-1),legloc='lower left',
         inset_bounds = (0.62,0.6,0.38,0.4),
         ):
    xlin = np.linspace(0,2.5*xmin,1001)
    xlog = np.logspace(np.log10(xmin/10),np.log10(xmin*10),1001)

    axlog = axlin.inset_axes(inset_bounds)
    axlog.set_xscale('log')
    axlog.set_yscale('log')
    axlog.set_ylim(ylimlog)
    axlin.set_ylim(ylimlin)
    axlin.set_xlabel('x',loc='right')
    axlin.set_ylabel('pdf',loc='top')
    for beta, lbl in zip(betas, betalabels):
        axlin.plot(xlin, model.pdf(xlin, alpha, xmin, beta),label=lbl)
        axlog.plot(xlog, model.pdf(xlog, alpha, xmin, beta),label=lbl)

    axlin.plot([xmin,xmin],ylimlin,'--',lw=1,color='#999999')
    axlog.plot([xmin,xmin],ylimlog,'--',lw=1,color='#999999')
    if legloc is not None:
        axlin.legend(loc=legloc)

if __name__ == "__main__":
    from fincoretails import general_algpareto, general_expareto, general_powpareto

    fig, ax = pl.subplots(1,3,figsize=(10,3))

    betas = [0,1,2]
    betalabels = map(lambda x: 'β = '+x,['0', '1','2'])
    plot(ax[2], general_algpareto, betas, betalabels)
    ax[2].set_title('(c) alg-Pareto',loc='left')

    betas = [-0.5,1,2]
    betalabels = map(lambda x: 'β = '+x,['-1/2', '1','2'])
    plot(ax[1], general_expareto, betas, betalabels,ylimlog=(1e-3,0.5))
    ax[1].set_title('(b) exp-Pareto',loc='left')

    betas = [-0.5,1,2]
    betalabels = map(lambda x: 'β = '+x,['-1/2', '1','2'])
    plot(ax[0], general_powpareto, betas, betalabels, legloc=(0.09,0.65))
    ax[0].set_title('(a) pow-Pareto',loc='left')

    for a in ax:
        a.set_xticks([0,10,20,25])
        a.set_yticks([0,0.05,0.1])

    fig.tight_layout()

    fig.savefig('example_plots.pdf')

    pl.show()
