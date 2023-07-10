import sympy as sy

a_, C, y = sy.symbols("alphaprime C y",positive=True)
x, b = sy.symbols("xprime beta", nonnegative=True)
alpha = sy.symbols("alpha")
beta = sy.symbols("beta")
xm = sy.symbols(r"\xm")
X = sy.Symbol("x")

a = a_ + 3

left = C * (2-(x/y)**b)
right = C * (y/x)**a

#C_ = (alpha - 1)*(beta+1)/(2*alpha*beta - beta + alpha)/y
C_ = (a - 1)*(b+1)/(2*a*b - b + a)/y

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


xleft = sy.integrate(left*x, (x,0,y))
xright = sy.integrate(right*x, (x,y,sy.oo))
x2left = sy.integrate(left*x**2, (x,0,y))
x2right = sy.integrate(right*x**2, (x,y,sy.oo))

Pcrit = sy.integrate(left, (x,0,y))
CDFleft = prep(sy.integrate(left, (x, 0, X)))
CDFright = prep(Pcrit + sy.integrate(right, (x, y, X)))

print("\n\n\n=======general nonpath pareto=======\n")
print("\nmean x ========")
mean = prep(xleft+xright)
sy.pprint(mean)

print("\n<x^2> ========")
x2 = prep(x2left+x2right)
sy.pprint(x2)

print("\n<Pcrit> ========")
sy.pprint(prep(Pcrit))

print("\nCDF ========")
sy.pprint(pw_dist(CDFleft,CDFright))

print("\n\n\n=======forced alg pareto=======\n")
print("============ mean x ========")
mean = prep(xleft+xright,replace_b_with=alpha)
#sy.pprint(xleft+xright)
sy.pprint(mean)

print("============  <x^2> ========")
x2 = prep(x2left+x2right,replace_b_with=alpha)
sy.pprint(x2)

print("\n<Pcrit> ========")
sy.pprint(prep(Pcrit,replace_b_with=alpha))

print("\nCDF ========")
print("left=")
sy.pprint(prep(CDFleft,replace_b_with=alpha))
print("right=")
sy.pprint(prep(CDFright,replace_b_with=alpha))


print("\n\n\n            uni pareto=======\n")
print("============ mean x ========")
mean = prep(xleft+xright,replace_b_with=0)
#sy.pprint(xleft+xright)
sy.pprint(mean)

print("============  <x^2> ========")
x2 = prep(x2left+x2right,replace_b_with=0)
sy.pprint(x2)

print("\n<Pcrit> ========")
sy.pprint(prep(Pcrit,replace_b_with=0))

print("\nCDF ========")
print("left=")
sy.pprint(prep(CDFleft,replace_b_with=0))
print("right=")
sy.pprint(prep(CDFright,replace_b_with=0))






print("\n\n\n            exp pareto=======\n")

#C_ = alpha*(alpha-1)/y/(1+(alpha-1)*sy.exp(alpha))
C_ = a*(a-1)/y/(1+(a-1)*sy.exp(a))

left = C*sy.exp(-a*(x/y-1))
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

print("============  <x^2> ========")
x2 = prep(x2left+x2right)
sy.pprint(x2)


print("\n<Pcrit> ========")
sy.pprint(prep(Pcrit))

print("\nCDF ========")
print("left=")
sy.pprint(prep(CDFleft))
print("right=")
sy.pprint(prep(CDFright))


