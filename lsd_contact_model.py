# -*- coding: utf-8 -*-
import numpy as np


def compute_forces_lsd(normal_ij, t_ij, interpenetration, interpenetration_vel):

    m_eq = (pi.mass*pj.mass)/(pi.mass+pj.mass)          # equivalent mass of the contact
    chi = 0.5                                           # should be taken from literature

    damp_coeff = 2 * chi * np.sqrt(m_eq*elstiffnesn)               #elstiffnes from which Particle?
                      #no cohesive kontakt --> equation only for interpenetration > 0
    rel_vel_normal = -np.dot((p2.velocity-p1.velocity), normal_ij)
    f_nd = damp_coeff * rel_vel_normal
    f_ne = interpenetration * pi.elstiffnesn * normal_ij

    f_n = f_ne + f_nd
    f_ij = f_n * normal_ij + f_t * t_ij
    return f_ij


class particle:
    list_of_particles = []

    def __init__(self, position, velocity, acceleration, rotation, force, radius, elstiffnesn, mass, pred_position):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.rotation = rotation
        self.force = force
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        self.pred_position = pred_position

        particle.list_of_particles.append(self)

# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation[0,0,w], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
p1 = particle(np.array([395,0,0]), np.array([1,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]),1,1000,500,np.array([395,0,0]))
p2 = particle(np.array([400,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]),1,1000,500,np.array([400,0,0]))
damp_coeff = 400
dt = 0.01
simtime = 100
for t in np.arange(0, simtime, dt):
    for n_particle in particle.list_of_particles:
        pred_vel05 = n_particle.velocity + 0.5 * dt * n_particle.acceleration
        pred_posi = n_particle.position + dt * pred_vel05
        n_particle.pred_position = np.around(pred_posi, decimals=4)
    # partile pairs for contact
    for index, pi in enumerate(particle.list_of_particles):
        for pj in particle.list_of_particles[index + 1:]:
            cij = pi.pred_position - pj.pred_position
            norm_cij = np.sqrt(cij[0] ** 2 + cij[1] ** 2)  # =norm_cji
            normal_ij = (pj.pred_position - pi.pred_position) / norm_cij
            normal_ji = (pi.pred_position - pj.pred_position) / norm_cij
            # contact detection
            if norm_cij < pi.radius + pj.radius:
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position - pi.pred_position), normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                v = 1/(pi.radius+pj.radius)*(pi.velocity-pj.velocity)*(pi.radius-pj.radius)
                print("contact",interpenetration, interpenetration_vel, v)
                i_acc = np.linalg.norm(interpenetration_acc)
                if np.linalg.norm(pi.force) != 0:
                    if i_acc == 0:
                        pi.force = [0, 0, 0]
                        pj.force = [0, 0, 0]
                    else:
                        pi.force = - interpenetration * pi.elstiffnesn * normal_ij - interpenetration_vel * damp_coeff * normal_ij
                        pj.force = - interpenetration * pj.elstiffnesn * normal_ji - interpenetration_vel * damp_coeff * normal_ij
                else:
                    pi.force = - interpenetration * pi.elstiffnesn * normal_ij - interpenetration_vel * damp_coeff * normal_ij
                    pj.force = - interpenetration * pj.elstiffnesn * normal_ji - interpenetration_vel * damp_coeff * normal_ij

            else:
                interpenetration = 0
                interpenetration_vel = 0
                interpenetration_acc = 0
                pi.force = [0, 0, 0]
                pj.force = [0, 0, 0]



            rel_vel = np.linalg.norm(interpenetration_vel)
            m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            omega = np.sqrt((pi.elstiffnesn)/m_eq)         # wie bestimmt man systemsteifigkeit für k(pi) =/= k(pj)
            psi = 0/(2*m_eq)                # = 0 für lin. elastisch und kann später mit coeff of restitution bestimmt werden
            interpenetration_max = (rel_vel/omega)*np.exp(-(psi/omega)*np.arctan(omega/psi))
            print(interpenetration_max)
            # particle doesnt reach the max. interpenetration
            i_acc = np.linalg.norm(interpenetration_acc)
            print(i_acc)







            new_vel_05 = pi.velocity + 0.5 * dt * pi.acceleration
            pi.position = np.around(pi.position + dt * new_vel_05, decimals=4)
            new_force = np.around(pi.force, decimals=4)
            pi.acceleration = np.around(new_force * (1 / pi.mass), decimals=4)
            pi.velocity = np.around(new_vel_05 + 0.5 * dt * pi.acceleration, decimals=4)

            new_vel_05 = pj.velocity + 0.5 * dt * pj.acceleration
            pj.position = np.around(pj.position + dt * new_vel_05, decimals=4)
            new_force = np.around(pj.force, decimals=4)
            pj.acceleration = np.around(new_force / pj.mass, decimals=4)  # n_particle.mass
            pj.velocity = np.around(new_vel_05 + 0.5 * dt * pj.acceleration, decimals=4)

            print(pi.position, pi.velocity, pi.acceleration, pi.force)
            print(pj.position, pj.velocity, pj.acceleration, pj.force)
            print('----')

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
    tangential_ij = v_ij_t / np.linalg.norm(v_ij_t)
else:
    t_ij = np.array([1, 1, 0])
    t_ij[0] = v_ij_n[1]
    t_ij[1] = v_ij_n[0]
    t_ij[2] = 0
    tangential_ij = t_ij / np.linalg.norm(t_ij)

# tangential_ji brauche ich auch noch

# for
# f_ij = f_ij_n + f_ij_t
# f_ij = f_n* normal_ij + f_t*tangential_ij

# contact forces
# forces = 0 if norm(ci,cj)>2r
# does not work apparently
'''