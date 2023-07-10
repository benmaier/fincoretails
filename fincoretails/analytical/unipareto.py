import sympy as sy

a, xm, nL, lnx, n = sy.symbols("a x_min n_Lambda lnx n")

def tex(expr):
    _y = sy.symbols(r"\xm")
    this = expr.replace(y, _y)
    return sy.latex(this)

lnxm = sy.log(xm)

logL = n*sy.log(a-1)-n*sy.log(a)-n*sy.log(xm) - a*nL*(lnx - lnxm)

dLda = sy.diff(logL, a)
dLdxm = sy.diff(logL, xm)

_ahat = sy.solve(sy.Eq(dLdxm,0), a)[0]
_xm = sy.solve(sy.Eq(dLda.replace(a, _ahat),0), xm)[0]
print("--------ahat--------")
sy.pprint(_ahat)
print("--------xmhat--------")
sy.pprint(_xm)

ahat = n/nL
lnhatxm = lnx - nL/(n-nL)
hatxm = sy.exp(lnhatxm)
assert((_ahat-ahat).simplify()==0)
assert((hatxm-_xm).simplify()==0)

d2Lda2 = sy.diff(dLda,a).replace(a,ahat).replace(xm,hatxm)
d2Ldxmda = sy.diff(dLda,xm).replace(a,ahat).replace(xm,hatxm)
d2Ldadxm = sy.diff(dLdxm,a).replace(a,ahat).replace(xm,hatxm)
d2Ldxm2 = sy.diff(dLdxm,xm).replace(a,ahat).replace(xm,hatxm)

M = sy.Matrix([ [d2Lda2, d2Ldxmda], [d2Ldadxm, d2Ldxm2] ])

print("--------hessian det-------")
sy.pprint(sy.simplify(M.det()))

