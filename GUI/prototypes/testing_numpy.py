import numpy as np
import sympy as smp

x = smp.symbols('x')
eq = 3*x+5

b = smp.solveset(eq, x, domain=smp.Reals)
c = np.array(smp.solve(eq, x), dtype='float64')
d = np.array([1.0000])
print(type(d[0]))
print(type(b))
print(b)
print(type(c[0]))
print(c)

x = np.array([1,2,3])
y = np.array([1,0,0])
print(x+y)
