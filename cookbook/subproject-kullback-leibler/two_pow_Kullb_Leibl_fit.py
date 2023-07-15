import numpy as np
from scipy.optimize import newton

def get_normalization_constant(x0,beta,y,alpha):
    assert(x0<y)
    assert(alpha>1)
    assert(beta!=1)
    assert(y>0)
    assert(x0>0)
    b = beta
    a = alpha
    return ((-1 + a)*(-1 + b))/(y*(-a + b + (y/x0)**(-1 + b)))


def pdf_left(x,x0,beta,y,alpha,C=None):
    if C is None:
        C = get_normalization_constant(x0,beta,y,alpha)
    return C * (y/x)**beta

def pdf_right(x,x0,beta,y,alpha,C=None):
    if C is None:
        C = get_normalization_constant(x0,beta,y,alpha)
    return C * (y/x)**alpha

def cdf_left(x,x0,beta,y,alpha,C=None):
    if C is None:
        C = get_normalization_constant(x0,beta,y,alpha)
    a = alpha
    b = beta
    return (C*(x**b*x0 - x*x0**b)*y**b)/((-1 + b)*(x*x0)**b)

def cdf_right(x,x0,beta,y,alpha,C=None):
    if C is None:
        C = get_normalization_constant(x0,beta,y,alpha)
    a = alpha
    b = beta
    P0 = cdf_left(y,x0,b,y,a,C)
    return (C*(y - x*(y/x)**a))/(-1 + a) + P0


def pdf(x,x0,beta,y,alpha):
    C = get_normalization_constant(x0,beta,y,alpha)
    x = np.asanyarray(x,dtype=float)
    result = np.piecewise(x,
                          (
                              x<x0,
                              np.logical_and(x>=x0,x<=y),
                              x>y
                          ),
                          (
                              0.0,
                              pdf_left,
                              pdf_right,
                          ),
                          x0,beta,y,alpha,C,
                         )
    return result

def cdf(x,x0,beta,y,alpha):
    C = get_normalization_constant(x0,beta,y,alpha)
    x = np.asanyarray(x,dtype=float)
    result = np.piecewise(x,
                          (
                              x<x0,
                              np.logical_and(x>=x0,x<=y),
                              x>y
                          ),
                          (
                              0.0,
                              cdf_left,
                              cdf_right,
                          ),
                          x0,beta,y,alpha,C,
                         )
    return result

def get_a_b_given_y(p, xedges, y, x0, xmid='arithmetic', dx=None, b0=0):

    assert(xedges[0] >= x0)

    if (type(xmid) == str):
        if xmid == 'geometric':
            xmid = np.sqrt(xedges[1:]*xedges[:-1])
        elif xmid == 'arithmetic':
            xmid = 0.5*(xedges[1:]+xedges[:-1])
        else:
            raise ValueError('xmid "{xmid}" is unknown')

    if dx is None:
        dx = np.diff(xedges)

    p = p / p.dot(dx)

    assert(len(p) + 1 == len(xedges))

    iS = np.where(xedges[:-1] < y)[0]
    iL = np.where(xedges[:-1] >= y)[0]
    S = xmid[iS]
    L = xmid[iL]
    dxS = dx[iS]
    dxL = dx[iL]
    PS = p[iS]
    PL = p[iL]
    pS = PS.dot(dxS)
    pL = PL.dot(dxL)
    logS = np.log(S).dot(PS*dxS) / pS
    logL = np.log(L).dot(PL*dxL) / pL
    logY = np.log(y)
    logYX0 = logY - np.log(x0)
    logSY = logS - logY
    logLY = logL - logY

    Sqrt = np.sqrt

    def aPlus_from_da(b):
        return 1/(1 - (x0/y)**(-1 + b)) + ((-1 + b)*\
                 ((1 + b)/(-1 + b) + Sqrt(1 + (-4 + 4*(y/x0)**(-1 + b))/((-1 + b)*pL*logLY))))/\
                 (2.*(1 - (y/x0)**(-1 + b)))

    def aMinus_from_da(b):
        return 1/(1 - (x0/y)**(-1 + b)) + ((-1 + b)*\
                 ((1 + b)/(-1 + b) - Sqrt(1 + (-4 + 4*(y/x0)**(-1 + b))/((-1 + b)*pL*logLY))))/\
                 (2.*(1 - (y/x0)**(-1 + b)))

    def a_from_db(b):
        return -((y*(-1 + (-1 + b)*b*pS*logSY) -
                    x0*(y/x0)**b*(-1 + (-1 + b)*pS*logSY + (-1 + b)*logYX0))/
                  (y - (-1 + b)*pS*(y - x0*(y/x0)**b)*logSY + x0*(y/x0)**b*(-1 + (-1 + b)*logYX0)))

    def diff_a_plus(b):
        return aPlus_from_da(b) - a_from_db(b)

    def diff_a_minus(b):
        return aMinus_from_da(b) - a_from_db(b)

    bs = []
    for b0 in [-1,0,0.5,2]:
        for diffa in [diff_a_plus, diff_a_minus]:
            try:
                b = newton(diffa,b0,maxiter=1000)
            except RuntimeError as e:
                b = np.nan
            bs.append(b)
            print(b)

    hatb = None
    hata = None
    minDKL = +np.inf

    for b in bs:

        if np.isnan(b) or b == -1:
            continue

        a = a_from_db(b)

        if np.isnan(a) or a <= 1:
            continue

        C = get_normalization_constant(x0,b,y,a)
        DKL = - np.log(C) - logY*(pS*b + pL*a) + b*pS*logS + a*pL*logL

        if DKL < minDKL:
            hata = a
            hatb = b
            minDKL = DKL

    return hata, hatb, minDKL

if __name__=="__main__":

    import  matplotlib.pyplot as pl

    x0_ = 0.1
    y_ = 1
    a_ = 2
    b_ = 0.9
    pars_ = (x0_,b_,y_,a_)


    bin_edges = be = np.logspace(-1,2,501)

    x = np.sqrt(be[1:]*be[:-1])
    dx = np.diff(be)
    CDF = cdf(be,*pars_)
    #p = pdf(x,*pars_)
    p = np.diff(CDF)/np.diff(be)

    figp, axp = pl.subplots(1,1,figsize=(4,4))
    axp.plot(x, p)
    axp.set_xscale('log')
    axp.set_yscale('log')


    avals = []
    bvals = []
    minDKLs = []

    ys = np.logspace(-1,2,501)[1:-1]
    for y in ys:
        a, b, minDKL = get_a_b_given_y(p, be, y, x0_,xmid=x, dx=dx)
        avals.append(a)
        bvals.append(b)
        minDKLs.append(minDKL)

    iM = np.nanargmin(minDKLs)
    print(f"{avals[iM]=}")
    print(f"{bvals[iM]=}")
    print(f"{ys[iM]=}")

    hata = avals[iM]
    hatb = bvals[iM]
    haty = ys[iM]

    axp.plot(x, pdf(x, x0_,b_,y_,a_))
    axp.plot(x, pdf(x, x0_,hatb,haty,hata))

    fig, ax = pl.subplots(3,1,figsize=(3,8),sharex=True)

    ax[0].plot(ys, minDKLs)
    ax[1].plot(ys, avals)
    ax[2].plot(ys, bvals)
    ax[0].set_xscale('log')

    pl.show()


