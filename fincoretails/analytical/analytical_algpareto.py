import sympy as sy

C, a, y, n, nL, nS, L, X, x = sy.symbols("C alpha y n n_L n_S L X x")

_C = (a**2-1)/(2*a) / y

#logL = n*sy.log(_C)

pS = C*(2-(x/y)**a)
pL = C*(y/x)**a

def tex(expr):
    _y = sy.symbols(r"\xm")
    this = expr.replace(y, _y)
    return sy.latex(this)

print("=========== constant ========")

CS = sy.integrate(pS, (x,0,y), conds="none")
CL = sy.integrate(pL, (x,y,sy.oo), conds="none")

__C = CS+CL
sy.pprint(sy.simplify(__C))

print("========= dlogLda =========")

logL = n*sy.log(_C) - a*nL*(L-sy.log(y)) + nS*sy.log(2-(X/y)**a)
#dlogLda = 2*a*n / (a**2-1) - n/a - nL*L - nS * sy.log(X/y)/((X/y)**(-a)-1)

#dlogLda = sy.simplify(sy.diff(logL,a))
dlogLda = sy.diff(logL,a)
#d2logLda2 = sy.diff(dlogLda,a)

sy.pprint(dlogLda)
print(tex(dlogLda))
#sy.pprint(sy.simplify(d2logLda2))
print("========= dlogLdy =========")

dlogLdy = sy.diff(logL,y)
sy.pprint(dlogLdy)
sy.pprint(tex(dlogLdy))
print("========= yhat =========")

yhat = sy.solve(sy.Eq(dlogLdy,0),y)
sy.pprint(yhat)
