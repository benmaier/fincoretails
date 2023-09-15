import numpy as np
import matplotlib.pyplot as pl
from fincoretails import compute_ccdf, distributions


def analysis(data, models, axpdf=None,axcdf=None, params=None,nbins=100):

    if axpdf is None and axcdf is None:
        fig, axs = pl.subplots(1,2,figsize=(7,3))
        axpdf, axcdf = axs

    if not hasattr(models,'__len__'):
        models = [models]


    belog = np.logspace(np.log10(np.min(data[data>0])*10),
                        np.log10(np.max(data)+1),
                        nbins+1)
    belin = np.logspace(np.min(data[data>0]),
                        np.max(data)+1,
                        nbins+1)
    belog2 = np.logspace(np.log10(belog[0]),
                         np.log10(belog[1]),
                         1001)
    xlog = np.sqrt(belog[1:]*belog[:-1])
    belin2 = np.logspace(belin[0],
                         belin[1],
                         1001)
    xlin = 0.5*(belog[1:]+belog[:-1])


    if params is None:
        params = []

        for model in models:
            print(model.__name__)
            pars = model.fit_params(data)
            params.append(pars)

    if axpdf is not None:
        axpdf.hist(data, bins=belog, density=True,alpha=0.5)

        for pars, model in zip(params, models):
            axpdf.plot(xlog, model.pdf(xlog, *pars))

        axpdf.set_yscale('log')
        axpdf.set_xscale('log')

    if axcdf is not None:
        xc, cc = compute_ccdf(data)
        axcdf.step(xc, cc, where='post')

        for pars, model in zip(params, models):
            print(model.__name__, pars, model.loglikelihood(data, *pars))
            axcdf.plot(xlog, model.ccdf(xlog, *pars))

        axcdf.set_yscale('log')
        axcdf.set_xscale('log')



if __name__=="__main__":

    from fincoretails import distributions, unipareto

    Nsample = 2000
    atrue = 2
    ytrue = 3
    data = unipareto.sample(Nsample,atrue,ytrue)
    print(distributions[:-1])

    analysis(data, distributions[:-1])

    pl.show()

