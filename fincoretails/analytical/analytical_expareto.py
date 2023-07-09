import sympy as sy

a, C, xm, n, nL, nS, L, S, x, x_, u  = sy.symbols(r"alpha C x_min n n_Lambda n_S \meanlnx \meanx x xprime u")

_C = a*(a-1) / ((a-1)*sy.exp(a)+1) / xm

logL = n * sy.log(_C) - a*nL*(L - sy.log(xm)) - a*nS*(S/xm-1)


def tex(expr):
    _L = sy.symbols(r"\meanlnx")
    _S = sy.symbols(r"\meanx")
    this = expr.replace(L, _L)
    this = this.replace(S, _S)
    return sy.latex(this)

#fac = 1/(n-1)*sy.sqrt((1-rho)/rho) * sy.exp(-t**2/2/rho)
#L = fac * np.exp((2*rho-1)/(2*rho) * n*yk2 + t*np.sqrt(1-rho)/rho * n*yk)

print("==== logL ======")
sy.pprint(logL)
print(tex(logL))

dLda = sy.simplify(sy.diff(logL,a))
print("==== dLda ======")
sy.pprint(dLda)
print("==== ahat ======")
#sy.pprint(ahat)


print("==== dLdxmin ======")
dLdxm = sy.diff(logL,xm)
sy.pprint(dLdxm)
print("---")
eq = sy.Eq(dLdxm, 0)
print("==== xmhat ======")
xmhat = sy.solve(eq,xm)[0]
sy.pprint(xmhat)
print(tex(xmhat))
print("==== ahat ======")
ahat = sy.solve(eq,a)[0]
sy.pprint(ahat)
print(tex(ahat))

print("==== d2Ldxmin2 ======")
d2Ldxm2 = sy.diff(dLdxm,xm)
d2Ldxm2 = d2Ldxm2.replace(xm, xmhat)
sy.pprint(sy.simplify(d2Ldxm2))

print()
print()
print("=======================")

P = sy.integrate(C*sy.exp(-a*(x_/xm-1)),(x_,0,x),conds='none')
P = sy.simplify(P)
print("===== P =======")
sy.pprint(P)


print("===== Pcrit =======")
Pcrit = sy.simplify(P.replace(x,xm))
sy.pprint(Pcrit)


print("===== xr =======")
xr = sy.solve(sy.Eq(P,u),x)[0]
sy.pprint(xr)
