import numpy as np
Sqrt = np.sqrt
Log = np.log
b = 1
x = 5
y = 2
n = 10
nL = 4

print((Sqrt((1 + b)*(1 + b + (4*n)/(nL*Log(x/y)))) - Sqrt((1 + b)*(4*n + nL*(1 + b)*Log(x/y)))/Sqrt(nL*Log(x/y)))/2.)
