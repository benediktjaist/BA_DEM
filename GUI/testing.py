import sympy as smp

x = smp.symbols('x')
b = 7.25 - 2.5*x
c = smp.sqrt(1 ** 2 - (x - 1) ** 2) + 1
eq = b-c

solution = smp.solveset(eq, x, domain=smp.Reals)

if solution:
    print("The equation has real solution")
else:
    print("The equation has no real solutions")




