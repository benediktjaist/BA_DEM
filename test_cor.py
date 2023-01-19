import math

def RootByBisection(a, b, tol, maxiter, coefficient_of_restitution):

        if coefficient_of_restitution < 0.001 :
            coefficient_of_restitution = 0.001

        if coefficient_of_restitution > 0.999 :
            return 0.0
        k=0
        gamma = 0.5 * (a + b)

        while b - a > tol and k <= maxiter:
            coefficient_of_restitution_trial = coeff_of_rest_diff(gamma, coefficient_of_restitution)

            if coeff_of_rest_diff(a, coefficient_of_restitution) * coefficient_of_restitution_trial < 0:
                b = gamma

            elif coefficient_of_restitution_trial == 0:
                return gamma

            else:
                a = gamma

            gamma = 0.5 * (a + b)
            k += 1

        return gamma


def coeff_of_rest_diff(gamma, desired_coefficient_of_restit):

    if gamma <= 1.0/math.sqrt(2.0) :
        return math.exp(-gamma/math.sqrt(1.0-gamma*gamma)*(math.pi-math.atan(2.0*gamma*math.sqrt(1.0-gamma*gamma)/(-2.0*gamma*gamma+1.0))))-desired_coefficient_of_restit
    elif gamma < 1.0 :
        return math.exp(-gamma/math.sqrt(1.0-gamma*gamma)*math.atan(2.0*gamma*math.sqrt(1.0-gamma*gamma)/(2.0*gamma*gamma-1.0)))-desired_coefficient_of_restit
    elif gamma == 1.0 :
        return 0.135335283 - desired_coefficient_of_restit
    else:
        return math.exp(-gamma/math.sqrt(gamma*gamma-1.0)*math.log((gamma/math.sqrt(gamma*gamma-1.0)+1.0)/(gamma/math.sqrt(gamma*gamma-1.0)-1.0)))-desired_coefficient_of_restit


def GammaForHertzThornton(e):
    if e < 0.001:
        e = 0.001

    if e > 0.999:
        return 0.0

    h1  = -6.918798
    h2  = -16.41105
    h3  =  146.8049
    h4  = -796.4559
    h5  =  2928.711
    h6  = -7206.864
    h7  =  11494.29
    h8  = -11342.18
    h9  =  6276.757
    h10 = -1489.915

    alpha = e*(h1+e*(h2+e*(h3+e*(h4+e*(h5+e*(h6+e*(h7+e*(h8+e*(h9+e*h10)))))))))

    return math.sqrt(1.0/(1.0 - (1.0+e)*(1.0+e) * math.exp(alpha)) - 1.0)



cor   = [i/100 for i in range(1,100)]
gamma_hmd = [GammaForHertzThornton(i) for i in cor] 
gamma_lsd = [RootByBisection(0.0, 16.0, 0.0001, 300, i) for i in cor] 



import matplotlib.pyplot as plt
plt.plot(cor, gamma_hmd, label="hmd")
plt.plot(cor, gamma_lsd, label="lsd")
plt.legend()
plt.grid()
plt.show()