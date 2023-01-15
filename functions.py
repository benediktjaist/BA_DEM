# -*- coding: utf-8 -*-

# -- imports --
import numpy as np

def get_rgb(colour):
    colour_rgb = {"Black": (0, 0, 0), "White": (255, 255, 255), "Red": (255, 0, 0), "Lime": (0, 255, 0),
                  "Blue": (0, 0, 255),
                  "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255), "Silver": (192, 192, 192),
                  "Gray": (128, 128, 128),
                  "Maroon": (128, 0, 0), "Olive": (128, 128, 0), "Green": (0, 128, 0), "Purple": (128, 0, 128),
                  "Teal": (0, 128, 128), "Navy": (0, 0, 128)}
    rgb = colour_rgb[str(colour)]
    return rgb

colour_list = ["Black", "White", "Red", "Lime", "Blue",
               "Yellow", "Cyan", "Magenta", "Silver", "Gray",
               "Maroon", "Olive", "Green", "Purple", "Teal", "Navy"]


def calculate_damp_coeff(epsilon_n):
    h_1 =float(0.2446517)
    h_2 =float(-0.5433478)
    h_3 =float(0.9280126)
    h_4 =float(-1.5897793)
    h_5 =float(1.2102729)
    h_6 =float(3.3815393)
    h_7 =float(6.3814014)
    h_8 =float(-34.482428)
    h_9 =float(25.672467)
    h_10 =float(94.396267)

    beta = float(epsilon_n - 0.5)

    gamma = epsilon_n * (1-epsilon_n)**2 * (h_1 + beta*(h_2 + beta * (h_3 + beta *
                    (h_4 + beta * (h_5 + beta * (h_6 + beta * (h_7 + beta * (h_8 + beta * (h_9 + beta * h_10)))))))))

    return gamma



def predict_position(particle, dt):
    # integration of motion with verlocity verlet (predict)
    pred_vel05 = particle.velocity + 0.5 * dt * particle.acceleration
    pred_posi = particle.position + dt * pred_vel05
    # update position
    particle.pred_position = np.around(pred_posi, decimals=4)


def compute_norm_cij(pi, pj):
    cij = pi.pred_position - pj.pred_position
    norm_cij = np.sqrt(cij[0] ** 2 + cij[1] ** 2)  # =norm_cji
    return norm_cij


def compute_normal_ij(pi, pj, norm_cij):
    normal_ij = (pj.pred_position - pi.pred_position) / norm_cij
    return normal_ij


def compute_normal_ji(pi, pj, norm_cij):
    normal_ji = (pi.pred_position - pj.pred_position) / norm_cij
    return normal_ji


def update_position(particle, dt):
    new_vel_05 = particle.velocity + 0.5 * dt * particle.acceleration
    particle.position = np.around(particle.position + dt * new_vel_05, decimals=4)
    new_force = np.around(particle.force, decimals=4)
    particle.acceleration = np.around(new_force * (1 / particle.mass), decimals=4)
    particle.velocity = np.around(new_vel_05 + 0.5 * dt * particle.acceleration, decimals=4)


def calculate_energies(pi,pj, interpenetration, interpenetration_vel, damp_coeff, elstiffnesn_eq):
    energy_i = 0.5 * pi.mass * np.linalg.norm(pi.velocity ** 2)
    energy_j = 0.5 * pj.mass * np.linalg.norm(pj.velocity ** 2)
    energy_el = 0.5 * np.sqrt(2) / 2 * interpenetration ** 2 * elstiffnesn_eq  # mit * np.sqrt(2)/2 und alter elstiffnes_eq klappts
    energy_damp = 0.5 * damp_coeff * np.linalg.norm(interpenetration_vel) * interpenetration
    energy = energy_i + energy_j + energy_el + energy_damp
    return energy, energy_el, energy_i, energy_j, energy_damp


def plot_paras():
    en = []  # gesamtenergie
    en_damp = []  # energie des dämpfers
    en_dissipated = 0  # historische variable
    en_dissipated_t = []
    en_el = []  # energie der feder
    en_i = []  # ekin
    en_j = []  # ekin
    t_points = []
    interpen_calc = []
    interpen_pred = []
    cont_forces = []
    acc = []
    interpen_acc = []
    interpen_vel = []
    en.append(energy)
    t_points.append(t)
    interpen_pred.append(interpenetration_max)
    interpen_calc.append(interpenetration)
    en_i.append(energy_i)
    en_j.append(energy_j)
    en_el.append(energy_el)
    en_damp.append(energy_damp)
    cont_forces.append(np.linalg.norm(pi.force))
    acc.append(np.linalg.norm(pi.acceleration))
    interpen_acc.append(np.linalg.norm(interpenetration_acc))
    interpen_vel.append(np.linalg.norm(interpenetration_vel))
    en_dissipated += energy_damp
    en_dissipated_t.append(en_dissipated)
'''
# hilfsvektoren von center of Particle zu point of contact
r_ijc = (pi.radius - (pj.elstiffnesn / (pi.elstiffnesn + pj.elstiffnesn)) * interpenetration) * normal_ij
r_jic = (pj.radius - (pi.elstiffnesn / (pi.elstiffnesn + pj.elstiffnesn)) * interpenetration) * normal_ji

# position of the contact point
p_ijc = pi.pred_position + r_ijc  # ortsvektor/point of contact from p1
p_jic = pj.pred_position + r_jic  # point of contact from p2 ==p_ijc

# velocity at the contact point
v_ij = (np.cross(pj.rotation, r_jic) + pj.velocity) - (np.cross(pi.rotation, r_ijc) + pi.velocity)

# decomposition in the local reference frame defined at the contact point
v_ij_n = (np.dot(v_ij, normal_ij) * normal_ij)
v_ij_t = v_ij - v_ij_n

# tangential unit vector should be tangential_ij
if np.linalg.norm(v_ij_t) != 0:
    t_ij = v_ij_t / np.linalg.norm(v_ij_t)
else:
    t_ij = np.array([1, 1, 0])
    t_ij[0] = v_ij_n[1]
    t_ij[1] = v_ij_n[0]
    t_ij[2] = 0
    t_ij = t_ij / np.linalg.norm(t_ij)
    '''
