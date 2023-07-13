import sympy as sy

a_, C, y, b = sy.symbols("alphaprime C y beta",positive=True)
x = sy.symbols("xprime", nonnegative=True)
alpha = sy.symbols("alpha")
beta = sy.symbols("beta")
xm = sy.symbols(r"\xm")
X = sy.Symbol("x")

a = a_ + 3

def rep(expr,replace_b_with=beta):
    #expr = expr.replace(C,C_)
    expr = expr.replace(a_,alpha-3)
    expr = expr.replace(b, beta)
    expr = expr.replace(beta,replace_b_with)
    expr = expr.replace(y, xm)
    return expr

def prep(expr,replace_b_with=beta):
    expr = rep(expr,replace_b_with=beta)
    expr = expr.powsimp(expr,force=True)
    expr = sy.simplify(expr)
    #expr = expr.factor(expr)
    #expr = expr.powsimp(expr,force=True)
    #return expr
    expr = expr.powsimp(expr,force=True)
    return sy.simplify(expr)

def pw_dist(left, right):
    p = sy.Function('p')(X)
    pw = sy.Piecewise(
                (left, X <= xm),
                (right, X > xm),
            )
    return sy.Eq(p, pw)


print("\n\n\n            exp pareto=======\n")

#C_ = alpha*(alpha-1)/y/(1+(alpha-1)*sy.exp(alpha))
#C_ = a*(a-1)/y/(1+(a-1)*sy.exp(a))

left = C*sy.exp(-b*(x/y-1))
right = C*(y/x)**a
massleft = sy.integrate(left, (x,0,y))
massright = sy.integrate(right, (x,y,sy.oo))
mass = (massleft + massright)
C_ = sy.solve(sy.Eq(mass, 1),C)[0]
print("======== C ========")
sy.pprint(prep(C_))

sy.pprint(left)

xleft = sy.integrate(left*x, (x,0,y))
xright = sy.integrate(right*x, (x,y,sy.oo))
x2left = sy.integrate(left*x**2, (x,0,y))
x2right = sy.integrate(right*x**2, (x,y,sy.oo))

Pcrit = sy.integrate(left, (x,0,y))
CDFleft = prep(sy.integrate(left, (x, 0, X)))
CDFright = prep(Pcrit + sy.integrate(right, (x, y, X)))
print("============ mean x ========")
mean = prep(xleft+xright)
sy.pprint(mean)
print(sy.latex(mean))

print("============  <x^2> ========")
x2 = prep(x2left+x2right)
sy.pprint(x2)
print(x2)
print(sy.latex(x2))


print("\n<Pcrit> ========")
sy.pprint(prep(Pcrit))

print("\nCDF ========")
print("left=")
sy.pprint(prep(CDFleft))
print("right=")
sy.pprint(prep(CDFright))


sy.pprint(CDFleft)
print(CDFleft)
sy.pprint(CDFright)
print(CDFright)

p = pw_dist(CDFleft,CDFright)

sy.pprint(p)
print(sy.latex(p))
