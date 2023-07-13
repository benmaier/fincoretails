import sympy as sy

a, b, C, xm, n, nL, nS, L, S, x, x_, u  = sy.symbols(r"alpha beta C x_min n n_Lambda n_S \meanlnx \meanx x xprime u")


#         β⋅(α - 1)
#───────────────────────────
#    ⎛   β            β    ⎞
#\xm⋅⎝α⋅ℯ  - α + β - ℯ  + 1⎠

eb = sy.exp(b)
_C = b*(a-1) / ((a-1)*(eb-1) + b) /xm
__C = b*(a-1) / (a*(eb-1)+b-eb+1) / xm
assert(sy.simplify(_C -__C) == 0)


logL = n * sy.log(_C) - a*nL*(L - sy.log(xm)) - b*nS*(S/xm-1)
logxm = sy.log(xm)


def tex(expr):
    _L = sy.symbols(r"\meanlnx")
    _S = sy.symbols(r"\meanx")
    _xm = sy.symbols(r"\xm")
    this = expr.replace(L, _L)
    this = expr.replace(xm, _xm)
    this = this.replace(S, _S)
    return sy.latex(this)

def py(expr):
    _a = sy.symbols(r"a")
    _b = sy.symbols(r"b")
    _xm = sy.symbols("y")
    _L = sy.symbols(r"L")
    _S = sy.symbols(r"S")
    _nL = sy.symbols(r"nL")
    _nS = sy.symbols(r"nS")
    this = expr.replace(a, _a)
    this = this.replace(b, _b)
    this = this.replace(xm, _xm)
    this = this.replace(L, _L)
    this = this.replace(S, _S)
    this = this.replace(nL, _nL)
    this = this.replace(nS, _nS)
    this = sy.printing.pycode(this)
    this = this.replace("math.exp(b)","eb")
    this = this.replace("math.exp(2*b)","eb**2")
    this = this.replace("math.log(y)","logy")
    this = this.replace("math.","np.")

    return this

#fac = 1/(n-1)*sy.sqrt((1-rho)/rho) * sy.exp(-t**2/2/rho)
#L = fac * np.exp((2*rho-1)/(2*rho) * n*yk2 + t*np.sqrt(1-rho)/rho * n*yk)

print("\n\n\n=== logL ======")
sy.pprint(logL)
print(tex(logL))

dLda = sy.simplify(sy.diff(logL,a))
print("\n\n\n=== dLda ======")
#print("dLdA control=",sy.simplify(dLda -  (n/(a-1) - n*(eb-1)/(a*(eb-1)+b-eb+1) - nL*(L-logxm)))
sy.pprint(dLda)
print(tex(dLda))
print("\n\n\n=== ahat ======")
ahat = sy.solve(sy.Eq(dLda,0),a)[1]
_ahat1 =    1/2 - b/2/(eb-1) * (1-sy.sqrt( 1 + 4*n*(eb-1) / b/nL/(L-sy.log(xm))))
ahat1 = sy.simplify(ahat)
sy.pprint(ahat1)
print(tex(ahat1))
sy.pprint(sy.simplify(ahat1 -_ahat1))
import sys
sys.exit(1)


print("\n\n\n=== dLdxmin ======")
dLdxm = sy.diff(logL,xm)
sy.pprint(dLdxm)
print("---")
eq = sy.Eq(dLdxm, 0)
print("\n\n\n=== xmhat ======")
xmhat1 = sy.solve(eq,xm)[0]
sy.pprint(xmhat1)
print(tex(xmhat1))
print("y = " + py(xmhat1))
#print("\n\n\n=== ahat ======")
#ahat = sy.solve(eq,a)[0]
#sy.pprint(ahat)
#print(tex(ahat))

print()
print()
print()
print("\n\n\n=== dLdbeta ======")
dLdb = sy.diff(logL,b)
sy.pprint(dLdb)
print("---")
eq = sy.Eq(dLdb, 0)
xmhat2 = sy.simplify(sy.solve(eq,xm)[0])
sy.pprint(xmhat2)

print("\n\n\n======= ahat2 =========")
print(type(xmhat1), type(xmhat2))
eq = sy.Eq(xmhat1, xmhat2)
sy.pprint(eq)
ahat2 = sy.solve(eq,a)[0]
sy.pprint(ahat2)
print("a = " + py(ahat2))

print("\n\n\n========= dA =========")
dA = ahat1-ahat2
dA = sy.simplify(dA)
sy.pprint(dA)
print()
print("eq = " + py(dA))

deqdb = sy.simplify(dA.diff(b))
print("prime = " + py(deqdb))


#print("\n\n\n=== d2Ldxmin2 ======")
#d2Ldxm2 = sy.diff(dLdxm,xm)
#d2Ldxm2 = d2Ldxm2.replace(xm, xmhat)
#sy.pprint(sy.simplify(d2Ldxm2))

#print()
#print()
#print("\n\n\n======================")
#
#P = sy.integrate(C*sy.exp(-a*(x_/xm-1)),(x_,0,x),conds='none')
#P = sy.simplify(P)
#print("\n\n\n==== P =======")
#sy.pprint(P)
#
#
#print("\n\n\n==== Pcrit =======")
#Pcrit = sy.simplify(P.replace(x,xm))
#sy.pprint(Pcrit)
#
#
#print("\n\n\n==== xr =======")
#xr = sy.solve(sy.Eq(P,u),x)[0]
#sy.pprint(xr)
