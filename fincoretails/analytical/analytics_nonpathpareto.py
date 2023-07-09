import sympy as sy
import numpy as np


a = sy.Symbol("alpha",positive=True)
b = sy.Symbol("beta",real=True)
xm = sy.Symbol("x_min",positive=True)
x = sy.Symbol("x",nonnegative=True)
X = sy.Symbol("X",nonnegative=True)
C = sy.Symbol("C",nonnegative=True)

_C = (a-1)*(b+1) / xm / (2*a*b-b+a)
sy.pprint(_C)

left = C*(2-(x/xm)**b)
right = C*(xm/x)**a

Pcrit = sy.integrate(left.replace(C,_C),(x,0,xm),conds="none")
Pcrit = sy.simplify(Pcrit)
sy.pprint(Pcrit)

#Pright = Pcrit + sy.integrate(right,(x,xm,sy.oo),conds="none")
Pright = sy.integrate(right,(x,xm,X),conds="none")
#sy.pprint(Pright)
Pright = sy.simplify(Pright)
sy.pprint(Pright)

p = sy.Piecewise(
        ( left, x <= xm),
        ( right, x > xm),
    )

meanl = sy.integrate(left*x, (x,0,xm),conds="none")
meanr = sy.integrate(right*x, (x,xm,sy.oo),conds="none")
mean = meanl+meanr

sy.pprint(meanl)
sy.pprint(meanr)
sy.pprint(sy.simplify(mean.replace(C,_C)))
sy.pprint(sy.simplify(mean))

x2l = sy.integrate(left*x**2, (x,0,xm),conds="none")
x2r = sy.integrate(right*x**2, (x,xm,sy.oo),conds="none")
x2 = x2l+x2r
sy.pprint(x2l)
sy.pprint(x2r)
sy.pprint(sy.simplify(x2.replace(C,_C)))
sy.pprint(sy.simplify(x2))
