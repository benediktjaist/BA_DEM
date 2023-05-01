import math
import matplotlib.pyplot as plt

# e stands for epsilon = COR
## --- LSD Analytical Solution for gamma() -------------------------------------

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

# function below from Thornton

def coeff_of_rest_diff(gamma, desired_coefficient_of_restit):

    if gamma <= 1.0/math.sqrt(2.0) :
        return math.exp(-gamma/math.sqrt(1.0-gamma*gamma)*(math.pi-math.atan(2.0*gamma*math.sqrt(1.0-gamma*gamma)/(-2.0*gamma*gamma+1.0))))-desired_coefficient_of_restit
    elif gamma < 1.0 :
        return math.exp(-gamma/math.sqrt(1.0-gamma*gamma)*math.atan(2.0*gamma*math.sqrt(1.0-gamma*gamma)/(2.0*gamma*gamma-1.0)))-desired_coefficient_of_restit
    elif gamma == 1.0 :
        return 0.135335283 - desired_coefficient_of_restit
    else:
        return math.exp(-gamma/math.sqrt(gamma*gamma-1.0)*math.log((gamma/math.sqrt(gamma*gamma-1.0)+1.0)/(gamma/math.sqrt(gamma*gamma-1.0)-1.0)))-desired_coefficient_of_restit

## --------------------------------------------------------------------------------


## ------------- HMD fitting curves by Thornton -----------------------------------
def GammaForHertzThornton(e):
    if e < 0.001:
        e = 0.001

    if e > 0.999:
        return 0.0

    h1 = -6.918798
    h2 = -16.41105
    h3 = 146.8049
    h4 = -796.4559
    h5 = 2928.711
    h6 = -7206.864
    h7 = 11494.29
    h8 = -11342.18
    h9 = 6276.757
    h10 = -1489.915

    alpha = e * (h1 + e * (h2 + e * (h3 + e * (h4 + e * (h5 + e * (h6 + e * (h7 + e *(h8 + e * (h9 + e * h10)))))))))

    return math.sqrt(1.0/(1.0 - (1.0+e)*(1.0+e) * math.exp(alpha)) - 1.0)
## --------------------------------------------------------------------------------


## ------------- LSD fitting curves by Thornton -----------------------------------
def GammaForLSDbyThorntonFittings(e):

    if e < 0.001:
        e = 0.001

    if e > 0.999:
        return 0.0

    h_1 = 0.2446517
    h_2 = -0.5433478
    h_3 = 0.9280126
    h_4 = -1.5897793
    h_5 = 1.2102729
    h_6 = 3.3815393
    h_7 = 6.3814014
    h_8 = -34.482428
    h_9 = 25.672467
    h_10 = 94.396267

    beta = e - 0.5

    xi = (h_1 + beta * (h_2 + beta * (h_3 + beta * (h_4 + beta * (h_5 + beta * (
        h_6 + beta * (h_7 + beta * (h_8 + beta * (h_9 + beta * (beta * h_10))))))))))


    zeta = e * (1 - e)**2 * xi

    return zeta

'''
cor = [i/100 for i in range(1,100)]
gamma_hmd_fitted = [GammaForHertzThornton(i) for i in cor]
gamma_lsd_analytical = [RootByBisection(0.0, 16.0, 0.0001, 300, i) for i in cor]
gamma_lsd_fitted = [GammaForLSDbyThorntonFittings(i) for i in cor]

orange = (227 / 255, 114 / 255, 34 / 255)
#plt.plot(cor, gamma_lsd_analytical, color=(0 / 255, 101 / 255, 189 / 255), label="Zeta Analytical")

plt.plot(cor, gamma_hmd_fitted, color=orange, label="Zeta Fitted for HMD")
#plt.plot(cor, gamma_lsd_analytical, color=(0 / 255, 101 / 255, 189 / 255), label="Zeta Analytical LSD")
plt.legend()
#plt.grid()
plt.ylabel('Dashpot Coefficient $\zeta$ [-]')
plt.xlabel('Coefficient of Restitution $\epsilon$ [-]')
plt.savefig('C:/Users/Jaist/Desktop/plots/' + 'zeta_cor_hmd_lsd' + '.pdf')
plt.show()


label="Zeta Fitted by Thornton "
'''