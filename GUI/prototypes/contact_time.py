import numpy as np
def nthroot(a,n):
    return np.power(a,(1/n))


def contact_time_lsd(damp_coeff, m_eq, kn_eq):
    beta = damp_coeff/m_eq
    omega_0 = np.sqrt(kn_eq/m_eq)
    omega = np.sqrt(omega_0**2 - beta**2)

    if beta < (omega_0/np.sqrt(2)):
        t_c = 1 / omega * (np.pi - np.arctan((2 * beta * omega) / (omega ** 2 - beta ** 2)))

    else:
        t_c = 1 / omega * (np.arctan((2 * beta * omega) / (omega ** 2 - beta ** 2)))

    return t_c

def contact_time_lsd_attracting(damp_coeff, m_eq, kn_eq):
    beta = damp_coeff / m_eq
    omega_0 = np.sqrt(kn_eq / m_eq)
    omega = np.sqrt(omega_0 ** 2 - beta ** 2)

    t_c = np.pi/(omega)

    return t_c

def contact_time_hmd(gamma, r1, r2, m1, m2, poisson1, poisson2, e1, e2, v1, v2):
        r_eq = (r1*r2)/(r1+r2)
        m_eq = (m1*m2)/(m1+m2)
        e_eq = 1/((1-poisson1**2)/e1 + (1-poisson2**2)/e2)
        gamma = 16/15 * e_eq * np.sqrt(r_eq/m_eq)
        v0 = v1-v2

        t_c = 2.94328 * nthroot(1/(v0 * gamma**2), 5)

        return t_c




