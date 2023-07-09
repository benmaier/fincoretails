import sympy as sy

rho, t, yk2, yk, n, N = sy.symbols("rho t ybar_2 ybar_1 n N")

sy.pprint(yk2)

logL = -n*sy.log(N-1) + n/2*(sy.log(1-rho)-sy.log(rho))\
       -(1-2*rho)/2/rho * n * yk2 \
       + n*t* sy.sqrt(1-rho)/rho * yk \
       -t**2*n/2/rho

def tex(expr):
    _yk, _yk2 = sy.symbols(r"\yk \yksq")
    this = expr.replace(yk, _yk)
    this = this.replace(yk2, _yk2)
    return sy.latex(this)

#fac = 1/(n-1)*sy.sqrt((1-rho)/rho) * sy.exp(-t**2/2/rho)
#L = fac * np.exp((2*rho-1)/(2*rho) * n*yk2 + t*np.sqrt(1-rho)/rho * n*yk)

print("==== logL ======")
sy.pprint(logL)
print(tex(logL))

dLdt = sy.diff(logL,t)
that = sy.solve(sy.Eq(dLdt,0),t)[0]
print("==== dLdt ======")
sy.pprint(dLdt)
print("==== that ======")
sy.pprint(that)

print("==== dLdrho ======")
dLdrho = sy.diff(logL,rho)
sy.pprint(dLdrho)
print("---")
eq = sy.Eq(dLdrho.replace(t, that), 0)
sy.pprint(sy.simplify(eq))
rhohat = sy.solve(eq,rho)[0]
print("==== rhohat ======")
sy.pprint(rhohat)
print(tex(rhohat))
