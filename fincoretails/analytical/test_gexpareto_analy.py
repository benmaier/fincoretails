import numpy as np
Log = np.log
Sqrt = np.sqrt
E = np.exp(1)

b = 1
n = 10
nL = 1
x = 10
y = 4

a1 = 1 - (b*(1 - Sqrt(1 + (4*(-1 + E**b)*n)/(nL*b*Log(x/y)))))/(2.*(-1 + E**b))


a2 =(-2*nL*Log(x/y) + 2*E**b*nL*Log(x/y) - nL*b*Log(x/y) + Sqrt(nL)*Sqrt(b)*Sqrt(Log(x/y))*Sqrt(-4*n + 4*E**b*n + nL*b*Log(x/y)))/\
       (2.*(-(nL*Log(x/y)) + E**b*nL*Log(x/y)))

print(a1, a2)
