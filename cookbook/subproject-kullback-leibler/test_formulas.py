from numpy import sqrt as Sqrt
from numpy import log as Log
import numpy as np

b = 2
x0 = 1
y = 1.01
x = 2*y
pL = 0.001

orig = ((1 + b)*x0**b*y - 2*x0*y**b + (x0**b*Sqrt((-1 + b)*y)*\
            Sqrt(-4*y + 4*x0*(y/x0)**b + (-1 + b)*pL*y*Log(x/y)))/(Sqrt(pL)*Sqrt(Log(x/y))))/\
                (2*x0**b*y - 2*x0*y**b)

better = 1/(1 - (x0/y)**(-1 + b)) + ((-1 + b)*\
         ((1 + b)/(-1 + b) + Sqrt(1 + (-4 + 4*(y/x0)**(-1 + b))/((-1 + b)*pL*Log(x/y)))))/\
        (2.*(1 - (y/x0)**(-1 + b)))

print(orig, better)
