# -*- coding: utf-8 -*-
# -- imports --
import numpy as np
from Particle import Particle
from Boundary import Boundary
import COR as cor
from typing import List
import time
from PyQt6.QtCore import QObject, pyqtSignal


class System(QObject):
    iterationChanged = pyqtSignal(int)
    total_iterationsChanged = pyqtSignal(int)
    remaining_timeChanged = pyqtSignal(str)


    def __init__(self, particles: List[Particle], boundaries: List[Boundary], dt: float, simtime: float, mu: float, coeff_of_restitution: float, gravity = False, contact_model = "LSD"):
        super().__init__()
        self.particles = particles
        self.boundaries = boundaries
        self.dt = dt
        self.simtime = simtime
        self.mu = mu
        self.coeff_of_restitution = coeff_of_restitution
        self.gravity = gravity
        self.crit_dt = self.__calculate_critical_time_step()
        self.total_iterations = int(simtime/dt)
        self.elapsed_time = 0
        self.remaining_time = 0
        # plotting examples
        self.plot_time_steps = []
        self.interpenetrations = []
        self.interpenetrationvels = []
        self.total_energy = []
        self.velo = []
        self.momente = []
        self.vi = []
        self.vj = []
        self.vjrotvel = []
        self.contact_model = contact_model
        self.plot_forces = []
        #self.damping_force = [] ?
        #self.elastic_force = []

        if dt > self.crit_dt:
            print("WARNING: dt > crit_dt. Setting dt to", np.round(self.crit_dt, 4))
            self.dt = self.crit_dt

    def __calculate_critical_time_step(self):
        crit_steps = []
        for particle in self.particles:
            crit_steps.append(0.3 * 2 * np.sqrt(particle.mass / particle.k_n))  # 0.3 chosen as O'Sullivan and Bray suggest
        return min(crit_steps)

    def run_simulation(self):
        "runs the DEM simulation"
        start_time = time.time()
        if self.contact_model == "LSD":
            gamma = cor.RootByBisection(0.0, 16.0, 0.0001, 300, self.coeff_of_restitution)
        else:
            gamma = cor.GammaForHertzThornton(self.coeff_of_restitution)

        for particle in self.particles:
            particle.interpenetrations_pp = [[]for i in range(len(self.particles)-1)]  # len(self.particles) - 1 because a particle cannot interact with itself
            particle.interpenetrations_pb = [[]for i in self.boundaries]

        for iteration, t in enumerate(np.arange(0, self.simtime, self.dt)):  # timeloop
            self.plot_time_steps.append(t)

            for n_particle in self.particles:  # particle loop
                # velocity-verlet time integration step predict
                n_particle.velocity = n_particle.velocity + 0.5 * self.dt * n_particle.acceleration  # velocity at half time step n+1/2
                n_particle.position = n_particle.position + self.dt * n_particle.velocity

                n_particle.rotation_vel = n_particle.rotation_vel + 0.5 * self.dt * n_particle.rotation_acc
                n_particle.rotation = n_particle.rotation + self.dt * n_particle.rotation_vel

                # update particle energies E_kin, E_rot, E_pot
                n_particle.energy_kin.append(0.5 * n_particle.mass * (np.linalg.norm(n_particle.velocity)) ** 2)
                n_particle.energy_rot.append(0.5 * n_particle.moment_of_inertia * (np.linalg.norm(n_particle.rotation_vel)) ** 2)
                if self.gravity == True:
                    n_particle.energy_pot.append(n_particle.mass * 10 * (832-n_particle.position[1]))
                else:
                    n_particle.energy_pot.append(0)

                # reset pp_torque and pb_torque -- pp_force does not have to be resetted because torque is calculated from force
                # n_particle.pp_force = np.array([0,0,0])
                n_particle.pp_torque = np.array([0, 0, 0])
                n_particle.pb_torque = np.array([0, 0, 0])

            for pi in self.particles:  # initialise pi.pp_forces for multiple particle contact
                if len(pi.pp_force) != (len(self.particles)-1):
                    pi.pp_force = [0 for i in range(len(self.particles)-1)]

            for index, pi in enumerate(self.particles):  # particle particle contact loop
                for pj in self.particles[index + 1:]:
                    cij = pi.position - pj.position
                    norm_cij = np.sqrt(cij[0] ** 2 + cij[1] ** 2)
                    normal_ij = (pj.position - pi.position) / norm_cij
                    normal_ji = (pi.position - pj.position) / norm_cij

                    if norm_cij < pi.radius + pj.radius:
                        interpenetration = pi.radius + pj.radius - np.dot((pj.position - pi.position), normal_ij)

                        pi.interpenetrations_pp[pj.id - 1].append(interpenetration)
                        pj.interpenetrations_pp[pi.id].append(interpenetration)

                        interpenetration_vel = -np.dot((pj.velocity - pi.velocity), normal_ij)
                        interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                        v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)

                        r_ijc = (pi.radius - (
                                pj.k_n / (pi.k_n + pj.k_n)) * interpenetration) * normal_ij
                        r_jic = (pj.radius - (
                                pi.k_n / (pi.k_n + pj.k_n)) * interpenetration) * normal_ji

                        p_ijc = pi.position + r_ijc  # point of contact from p1
                        p_jic = pj.position + r_jic  # point of contact from p2 == p_ijc

                        # relative velocity at the contact point
                        v_ij = np.array( -(np.cross(pj.rotation_vel, r_jic) + pj.velocity) + (
                                    np.cross(pi.rotation_vel, r_ijc) + pi.velocity)) #, dtype=np.float64

                        # force decomposition in the local reference frame defined at the contact point
                        v_ij_n = np.array(np.dot(v_ij, normal_ij) * normal_ij)
                        v_ij_t = np.around(v_ij - v_ij_n, 13)  # np.around in order to avoid wiggle during central impact

                        # compute tangential unit vector t_ij
                        if np.linalg.norm(v_ij_t) != 0:
                            t_ij = v_ij_t / np.linalg.norm(v_ij_t)
                        else:
                            t_ij = np.array([0, 0, 0])

                        m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
                        radius_eq = (pi.radius * pj.radius) / (pi.radius + pj.radius)

                        if self.contact_model == "LSD":  # compute spring and damping constants
                            k_n_eq = (pi.k_n * pj.k_n) / (pi.k_n + pj.k_n)
                            c_n_eq = 2 * gamma * np.sqrt(m_eq * k_n_eq)

                            poisson_eq = 2 * (pi.poisson * pj.poisson)/(pi.poisson + pj.poisson)
                            k_t_eq = 2 * (1 - poisson_eq) / (2 - poisson_eq) * k_n_eq
                            c_t_eq = 2 * gamma * np.sqrt(m_eq * k_t_eq)

                            # k_t_i = 2 * (1 - pi.poisson) / (2 - pi.poisson) * pi.k_n
                            # k_t_j = 2 * (1 - pj.poisson) / (2 - pj.poisson) * pj.k_n
                            # k_t_eq = (k_t_i * k_t_j)/(k_t_i + k_t_j )

                        else:  # HMD
                            E_eq = (1)/((1-pi.poisson**2)/(pi.E) + (1-pj.poisson**2)/(pj.E))
                            G_eq = (1)/((2-pi.poisson)/(pi.E/(2*(1+pi.poisson))) + (2-pj.poisson)/(pj.E/(2*(1+pj.poisson))))

                            k_n_eq = 2 * E_eq * np.sqrt(radius_eq * pi.interpenetrations_pp[pj.id-1][-1])
                            k_t_eq = 8 * G_eq * np.sqrt(radius_eq * pi.interpenetrations_pp[pj.id-1][-1])

                            c_n_eq = 2 * gamma * np.sqrt(m_eq * k_n_eq)
                            c_t_eq = 2 * gamma * np.sqrt(m_eq * k_t_eq)

                        increment_of_t_displacement = np.linalg.norm(v_ij_t) * self.dt  # tangential displacement

                        if self.contact_model == "LSD":
                            max_friction_force = self.mu * np.dot(pi.pp_force[pj.id-1], normal_ij)
                            # tan_force = abs(np.dot(pi.pp_force[pj.id-1], t_ij)) + k_t_eq * increment_of_t_displacement
                            tan_force_trial = abs(np.dot(pi.pp_force[pj.id-1], t_ij)) * t_ij +  k_t_eq * increment_of_t_displacement * t_ij + c_t_eq * v_ij_t  # damping is in the check for sliding included c.f. Santasousana

                            if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                f_t = np.linalg.norm(max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)
                            else:
                                f_t = np.linalg.norm(tan_force_trial)  # * tan_force/np.linalg.norm(tan_force)

                            # tan energy
                            if len(pi.energy_tan) == 0:
                                pi.energy_tan.append((f_t * increment_of_t_displacement) * 0.5)  # damping energy?
                                pj.energy_tan.append((f_t * increment_of_t_displacement) * 0.5)
                            else:
                                pi.energy_tan.append((pi.energy_tan[-1] + f_t * increment_of_t_displacement) * 0.5)
                                pj.energy_tan.append((pj.energy_tan[-1] + f_t * increment_of_t_displacement) * 0.5)

                            f_n_vector = (-interpenetration * k_n_eq * normal_ij - interpenetration_vel * c_n_eq * normal_ij)

                            # self.damping_force.append(interpenetration_vel * c_n_eq) ?
                            # self.elastic_force.append(interpenetration * k_n_eq)

                            if np.dot(f_n_vector, normal_ij) >= 0:  # exclude attracting forces
                                f_n_vector = np.array([0, 0, 0])
                            # else:
                                # print('normal force positive')
                            pi.pp_force[pj.id-1] = np.array(f_n_vector - f_t * t_ij)
                            pj.pp_force[pi.id] = - pi.pp_force[pj.id-1]

                            # torque LSD
                            moment_i = - f_t * np.cross(r_ijc, t_ij) # -t_ij
                            moment_j = f_t * np.cross(r_jic, t_ij)

                            pi.pp_torque = pi.pp_torque + moment_i
                            pj.pp_torque = pj.pp_torque + moment_j

                        else:  # HMD
                            f_n = 2/3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel

                            if np.dot(pi.pp_force[pj.id-1], normal_ij) >= 0:
                                tan_force = abs(np.dot(pi.pp_force[pj.id - 1], t_ij)) + k_t_eq * increment_of_t_displacement
                            else:
                                if len(pi.interpenetrations_pp[pj.id - 1]) >= 2:
                                    k_t_eq_prev_step = 8 * G_eq * np.sqrt(radius_eq * pi.interpenetrations_pp[pj.id - 1][-2])
                                else:
                                    k_t_eq_prev_step = 0
                                tan_force = abs(
                                    np.dot(pi.pp_force[pj.id - 1], t_ij)) * (k_t_eq/(k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                            tan_force_trial = tan_force * t_ij + c_t_eq * v_ij_t
                            max_friction_force = self.mu * np.dot(pi.pp_force[pj.id - 1], normal_ij)
                            if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                f_t = np.linalg.norm(max_friction_force)
                            else:
                                f_t = np.linalg.norm(tan_force_trial)

                            pi.pp_force[pj.id - 1] = np.array(-f_n * normal_ij - f_t * t_ij)
                            pj.pp_force[pi.id] = - pi.pp_force[pj.id - 1]

                            # torque HMD
                            moment_i = - f_t * np.cross(r_ijc, t_ij)  # -t_ij
                            moment_j = f_t * np.cross(r_jic, t_ij)

                            pi.pp_torque = pi.pp_torque + moment_i
                            pj.pp_torque = pj.pp_torque + moment_j


                        if self.contact_model == "LSD":  # particle particle energies
                            pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                    interpenetration - pi.interpenetrations_pp[pj.id - 1][-2]) * 0.5))
                            pj.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                    interpenetration - pi.interpenetrations_pp[pj.id - 1][-2]) * 0.5))
                        else:  # HMD
                            pi.energy_el.append((pi.energy_el[-1] + 2/3 * k_n_eq * interpenetration * (
                                        interpenetration - pi.interpenetrations_pp[pj.id - 1][-2]) * 0.5))
                            pj.energy_el.append((pi.energy_el[-1] + 2/3 * k_n_eq * interpenetration * (
                                    interpenetration - pi.interpenetrations_pp[pj.id - 1][-2]) * 0.5))

                        pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (interpenetration - pi.interpenetrations_pp[pj.id - 1][-2])) * 0.5)
                        pj.energy_damp.append((pj.energy_damp[-1] + c_n_eq * interpenetration_vel * (interpenetration - pi.interpenetrations_pp[pj.id - 1][-2])) * 0.5)

                    else:  # no particle particle contact
                        interpenetration = 0
                        interpenetration_vel = 0
                        interpenetration_acc = 0

                        pi.interpenetrations_pp[pj.id - 1].append(interpenetration)
                        pj.interpenetrations_pp[pi.id].append(interpenetration)

                        pi.pp_force[pj.id-1] = np.array([0, 0, 0])
                        pj.pp_force[pi.id] = np.array([0, 0, 0])
                        pi.pp_torque = pi.pp_torque + np.array([0, 0, 0])
                        pj.pp_torque = pj.pp_torque + np.array([0, 0, 0])

                        pi.energy_el.append(0)
                        pi.energy_damp.append(0)
                        pi.energy_tan.append(0)

                        pj.energy_el.append(0)
                        pj.energy_damp.append(0)
                        pj.energy_tan.append(0)

                        # WARNING
                        # multiple contact energy tracking is missing
                    # maximum interpenetration could be calculated as follows
                    '''
                    rel_vel = np.linalg.norm(pi.velocity - pj.velocity)
                    omega = np.sqrt(k_n_eq / m_eq)  # wie bestimmt man systemsteifigkeit für k(pi) =/= k(pj)
                    psi = self.damp_coeff / (2 * m_eq)  # =0 für lin. elastisch wird später mit COR bestimmt werden

                    interpenetration_max = (rel_vel / omega) * np.exp(-(psi / omega) * np.arctan(omega / psi))
                    '''
                    self.interpenetrationvels.append(interpenetration_vel)
                    self.interpenetrations.append(interpenetration)

            for pi in self.particles:  # contact with boundaries
                # all_bf = []  # no longer needed since pi.pb_forces exist
                all_bt = []
                all_bp_energies = []

                if len(pi.pb_force) != len(self.boundaries):  # force lists for particle contacting multiple boundaries at once
                    pi.pb_force = [0 for i in range(len(self.boundaries))]

                for boundary in self.boundaries:
                    m_eq = pi.mass
                    radius_eq = pi.radius

                    if self.contact_model == "LSD":
                        k_n_eq = pi.k_n
                        c_n_eq = 2 * gamma * np.sqrt(m_eq * k_n_eq)
                        k_t_eq = 2 * (1 - pi.poisson) / (2 - pi.poisson) * pi.k_n
                        c_t_eq = 2 * gamma * np.sqrt(m_eq * k_t_eq)

                    else:  # HMD
                        E_eq = pi.E / (1 - pi.poisson ** 2)
                        G_eq = (E_eq) / (2 * (1 + pi.poisson))

                        if pi.interpenetrations_pb[boundary.id]:
                            k_n_eq = 2 * E_eq * np.sqrt(radius_eq * pi.interpenetrations_pb[boundary.id][-1])
                            k_t_eq = 8 * G_eq * np.sqrt(radius_eq * pi.interpenetrations_pb[boundary.id][-1])
                        else:
                            k_n_eq = 0
                            k_t_eq = 0

                        c_n_eq = 2 * gamma * np.sqrt(m_eq * k_n_eq)
                        c_t_eq = 2 * gamma * np.sqrt(m_eq * k_t_eq)

                    # check if boundary is a vertical line
                    if boundary.start_point[0] == boundary.end_point[0]:

                        # compute the distance between the center of the circle and the vertical line
                        dist = abs(pi.position[0] - boundary.start_point[0])

                        # check if particle intersects with boundary
                        if dist <= pi.radius:
                            y_intercepts = []
                            ip1 = np.sqrt(pi.radius ** 2 - (boundary.start_point[0] - pi.position[0]) ** 2) + pi.position[1]
                            ip2 = - np.sqrt(pi.radius ** 2 - (boundary.start_point[0] - pi.position[0]) ** 2) + pi.position[1]
                            y_intercepts.append(ip1)
                            y_intercepts.append(ip2)

                            # compute the point of contact (poc)
                            y_mid = sum(y_intercepts) / 2
                            poc = np.array([boundary.start_point[0], y_mid, 0])

                            # check if the poc is element of the interval of the boundary
                            ymin = min(boundary.start_point[1], boundary.end_point[1])
                            ymax = max(boundary.start_point[1], boundary.end_point[1])

                            # check if particle edge contact or particle vertex contact

                            if  poc[1] >= ymin and poc[1] <= ymax:  # particle edge contact

                                normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)
                                r_ibc = poc - pi.position

                                interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                                interpenetration_vel = np.dot(pi.velocity, normal_ib)
                                pi.interpenetrations_pb[boundary.id].append(interpenetration)
                                print(pi.interpenetrations_pb)
                                # velocity at the contact point
                                v_ib = (np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                                # decomposition in the local reference frame defined at the contact point
                                v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                                # v_ib_t = v_ib - v_ib_n
                                v_ib_t = np.around(v_ib - v_ib_n, 13)

                                # tangential unit vector should be tangential_ij
                                if np.linalg.norm(v_ib_t) !=0:  # welche grenze?
                                    t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                                else:
                                    t_ib = np.array([0,0,0])
                                increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                                if self.contact_model == "LSD":
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    #tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                           #t_ib)) + k_t_eq * increment_of_t_displacement
                                    tan_force_trial = abs(np.dot(pi.pb_force[boundary.id],
                                                                 t_ib)) * t_ib + k_t_eq * increment_of_t_displacement * t_ib + c_t_eq * v_ib_t

                                    # should damping in the check for sliding be included?

                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(
                                            max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)

                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)  # * tan_force/np.linalg.norm(tan_force)

                                    f_n_vector = (
                                                -interpenetration * k_n_eq * normal_ib - interpenetration_vel * c_n_eq * normal_ib)

                                    if np.dot(f_n_vector, normal_ib) >= 0:  # exclude attracting forces
                                        f_n_vector = np.array([0, 0, 0])
                                    # else:
                                    # print('normal force positive')
                                    pi.pb_force[boundary.id] = np.array(f_n_vector - f_t * t_ib)  # + t_ij

                                    # -- torque
                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                else:  # HMD
                                    f_n = 2 / 3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel

                                    if np.dot(pi.pb_force[boundary.id], normal_ib) >= 0:
                                        tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                               t_ib)) + k_t_eq * increment_of_t_displacement
                                    else:
                                        if len(pi.interpenetrations_pb[boundary.id]) >= 2:
                                            k_t_eq_prev_step = 8 * G_eq * np.sqrt(
                                                radius_eq * pi.interpenetrations_pb[boundary.id][-2])
                                        else:
                                            k_t_eq_prev_step = 0
                                        tan_force = abs(
                                            np.dot(pi.pb_force[boundary.id], t_ib)) * (k_t_eq / (
                                            k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                                    tan_force_trial = tan_force * t_ib + c_t_eq * v_ib_t
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)

                                    pi.pb_force[boundary.id] = np.array(-f_n * normal_ib - f_t * t_ib)

                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                # pb energy if boundary is vertical line and boundary edge contact
                                if self.contact_model == "LSD":
                                    pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                else:  # HMD
                                    pi.energy_el.append((pi.energy_el[-1] + 2 / 3 * k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])) * 0.5)

                            # particle vertex contact for first vertex
                            elif np.linalg.norm(pi.position - boundary.start_point) <= pi.radius:
                                poc = np.array([boundary.start_point[0], boundary.start_point[1], 0])
                                normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)

                                interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                                interpenetration_vel = np.dot(pi.velocity, normal_ib)
                                r_ibc = poc - pi.position
                                pi.interpenetrations_pb[boundary.id].append(interpenetration)

                                # velocity at the contact point
                                v_ib = (np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                                # decomposition in the local reference frame defined at the contact point
                                v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                                # v_ib_t = v_ib - v_ib_n
                                v_ib_t = np.around(v_ib - v_ib_n, 13)

                                # tangential unit vector should be tangential_ij
                                if np.linalg.norm(v_ib_t) != 0:  # welche grenze?
                                    t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                                else:
                                    t_ib = np.array([0, 0, 0])
                                increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                                if self.contact_model == "LSD":
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    #tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                           #t_ib)) + k_t_eq * increment_of_t_displacement
                                    tan_force_trial = abs(np.dot(pi.pb_force[boundary.id],
                                                                 t_ib)) * t_ib + k_t_eq * increment_of_t_displacement * t_ib + c_t_eq * v_ib_t

                                    # should damping in the check for sliding be included?

                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(
                                            max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)
                                        # print('max friction force', max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)  # * tan_force/np.linalg.norm(tan_force)

                                    f_n_vector = (
                                            -interpenetration * k_n_eq * normal_ib - interpenetration_vel * c_n_eq * normal_ib)

                                    if np.dot(f_n_vector, normal_ib) >= 0:
                                        # exclude attracting forces
                                        f_n_vector = np.array([0, 0, 0])

                                    # else:
                                    # print('normal force positive')
                                    pi.pb_force[boundary.id] = np.array(f_n_vector - f_t * t_ib)  # + t_ij

                                    # -- torque
                                    moment = -f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                else:  # HMD
                                    f_n = 2 / 3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel
                                    if np.dot(pi.pb_force[boundary.id], normal_ib) >= 0:
                                        tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                               t_ib)) + k_t_eq * increment_of_t_displacement
                                    else:
                                        if len(pi.interpenetrations_pb[boundary.id]) >= 2:
                                            k_t_eq_prev_step = 8 * G_eq * np.sqrt(
                                                radius_eq * pi.interpenetrations_pb[boundary.id][-2])
                                        else:
                                            k_t_eq_prev_step = 0
                                        tan_force = abs(
                                            np.dot(pi.pb_force[boundary.id], t_ib)) * (k_t_eq / (
                                            k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                                    tan_force_trial = tan_force + c_t_eq * v_ib_t
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)

                                    pi.pb_force[boundary.id] = np.array(-f_n * normal_ib - f_t * t_ib)

                                    #print(pi.pb_force[boundary.id])

                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                # boundary vertical and first vertex contact
                                if self.contact_model == "LSD":
                                    pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                else:  # HMD
                                    pi.energy_el.append((pi.energy_el[-1] + 2 / 3 * k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (
                                        interpenetration - pi.interpenetrations_pb[boundary.id][-2])) * 0.5)


                            # particle vertex contact for second vertex
                            elif np.linalg.norm(pi.position - boundary.end_point) <= pi.radius:
                                poc = np.array([boundary.end_point[0], boundary.end_point[1], 0])
                                normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)

                                interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                                interpenetration_vel = np.dot(pi.velocity, normal_ib)
                                r_ibc = poc - pi.position
                                pi.interpenetrations_pb[boundary.id].append(interpenetration)

                                # velocity at the contact point
                                v_ib = (np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                                # decomposition in the local reference frame defined at the contact point
                                v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                                # v_ib_t = v_ib - v_ib_n
                                v_ib_t = np.around(v_ib - v_ib_n, 13)

                                # tangential unit vector should be tangential_ij
                                if np.linalg.norm(v_ib_t) != 0:  # welche grenze?
                                    t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                                else:
                                    t_ib = np.array([0, 0, 0])
                                increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                                if self.contact_model == "LSD":
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                           t_ib)) + k_t_eq * increment_of_t_displacement
                                    tan_force_trial = abs(np.dot(pi.pb_force[boundary.id],
                                                                 t_ib)) * t_ib + k_t_eq * increment_of_t_displacement * t_ib + c_t_eq * v_ib_t

                                    # should damping in the check for sliding be included?

                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(
                                            max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)  # * tan_force/np.linalg.norm(tan_force)

                                    f_n_vector = (
                                            -interpenetration * k_n_eq * normal_ib - interpenetration_vel * c_n_eq * normal_ib)

                                    if np.dot(f_n_vector, normal_ib) >= 0:
                                        # exclude attracting forces
                                        f_n_vector = np.array([0, 0, 0])

                                    # else:
                                    # print('normal force positive')
                                    pi.pb_force[boundary.id] = np.array(f_n_vector - f_t * t_ib)  # + t_ij

                                    # -- torque
                                    moment = -f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                else:  # HMD
                                    f_n = 2 / 3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel
                                    #print(pi.pb_force)
                                    if np.dot(pi.pb_force[boundary.id], normal_ib) >= 0:
                                        tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                               t_ib)) + k_t_eq * increment_of_t_displacement
                                    else:
                                        if len(pi.interpenetrations_pb[boundary.id]) >= 2:
                                            k_t_eq_prev_step = 8 * G_eq * np.sqrt(
                                                radius_eq * pi.interpenetrations_pb[boundary.id][-2])
                                        else:
                                            k_t_eq_prev_step = 0
                                        tan_force = abs(
                                            np.dot(pi.pb_force[boundary.id], t_ib)) * (k_t_eq / (
                                            k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                                    tan_force_trial = tan_force + c_t_eq * v_ib_t
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)

                                    pi.pb_force[boundary.id] = np.array(-f_n * normal_ib - f_t * t_ib)

                                    #print(pi.pb_force[boundary.id])

                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                # pb energy if boundary is vertical line and second vertex contact
                                if self.contact_model == "LSD":
                                    pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                else:  # HMD
                                    pi.energy_el.append((pi.energy_el[-1] + 2 / 3 * k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])) * 0.5)

                        # no contact
                        else:
                            pi.pb_force[boundary.id] = np.array([0, 0, 0])
                            pi.energy_el.append(0)
                            pi.energy_damp.append(0)
                            pi.interpenetrations_pb[boundary.id].append(0)

                    # boundary can be expressed as linear function
                    else:
                        m = (boundary.end_point[1] - boundary.start_point[1])/(boundary.end_point[0] - boundary.start_point[0])
                        tt = boundary.start_point[1] - m * boundary.start_point[0]
                        A = m**2 + 1
                        B = 2*(m * tt - m * pi.position[1] - pi.position[0])
                        C = pi.position[1]**2 - pi.radius**2 + pi.position[0]**2 - 2 * tt * pi.position[1] + tt**2
                        D = B**2 - 4 * A * C

                        if D >= 0:

                            x1 = (- B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                            x2 = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

                            x_mid = (x1 + x2)/2
                            poc = np.array([x_mid, m * x_mid + tt, 0])
                            # check if the poc is element of the interval of the boundary
                            xmin = min(boundary.start_point[0], boundary.end_point[0])
                            xmax = max(boundary.start_point[0], boundary.end_point[0])


                            if poc[0] >= xmin and poc[0] <= xmax:
                                normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)

                                interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                                interpenetration_vel = np.dot(pi.velocity, normal_ib)
                                r_ibc = poc - pi.position
                                pi.interpenetrations_pb[boundary.id].append(interpenetration)

                                # velocity at the contact point
                                v_ib = (np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                                # decomposition in the local reference frame defined at the contact point
                                v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                                # v_ib_t = v_ib - v_ib_n
                                v_ib_t = np.around(v_ib - v_ib_n, 13)

                                # tangential unit vector should be tangential_ij
                                if np.linalg.norm(v_ib_t) != 0:  # welche grenze?
                                    t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                                else:
                                    t_ib = np.array([0, 0, 0])
                                increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                                if self.contact_model == "LSD":
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    #tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                           #t_ib)) + k_t_eq * increment_of_t_displacement
                                    tan_force_trial = abs(np.dot(pi.pb_force[boundary.id],
                                                                 t_ib)) * t_ib + k_t_eq * increment_of_t_displacement * t_ib + c_t_eq * v_ib_t

                                    # should damping in the check for sliding be included?

                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(
                                            max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)

                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)  # * tan_force/np.linalg.norm(tan_force)

                                    f_n_vector = (
                                            -interpenetration * k_n_eq * normal_ib - interpenetration_vel * c_n_eq * normal_ib)

                                    if np.dot(f_n_vector, normal_ib) >= 0:
                                        # exclude attracting forces
                                        f_n_vector = np.array([0, 0, 0])

                                    # else:
                                    # print('normal force positive')
                                    pi.pb_force[boundary.id] = np.array(f_n_vector - f_t * t_ib)

                                    # -- torque
                                    moment = -f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                else:  # HMD
                                    f_n = 2 / 3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel
                                    #print(pi.pb_force)
                                    if np.dot(pi.pb_force[boundary.id], normal_ib) >= 0:
                                        tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                               t_ib)) + k_t_eq * increment_of_t_displacement
                                    else:
                                        if len(pi.interpenetrations_pb[boundary.id]) >= 2:
                                            k_t_eq_prev_step = 8 * G_eq * np.sqrt(
                                                radius_eq * pi.interpenetrations_pb[boundary.id][-2])
                                        else:
                                            k_t_eq_prev_step = 0
                                        tan_force = abs(
                                            np.dot(pi.pb_force[boundary.id], t_ib)) * (k_t_eq / (
                                            k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                                    tan_force_trial = tan_force + c_t_eq * v_ib_t
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)

                                    pi.pb_force[boundary.id] = np.array(-f_n * normal_ib - f_t * t_ib)

                                    #print(pi.pb_force[boundary.id])

                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                # pb contact (linear equation) and line contact
                                if self.contact_model == "LSD":
                                    pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                else:  # HMD
                                    pi.energy_el.append((pi.energy_el[-1] + 2 / 3 * k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (
                                        interpenetration - pi.interpenetrations_pb[boundary.id][-2])) * 0.5)

                            # check if the particle intersects with the first boundary point
                            elif np.linalg.norm(pi.position - boundary.start_point) <= pi.radius:
                                poc = np.array([boundary.start_point[0], boundary.start_point[1], 0])
                                normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)

                                interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                                interpenetration_vel = np.dot(pi.velocity, normal_ib)
                                r_ibc = poc - pi.position
                                pi.interpenetrations_pb[boundary.id].append(interpenetration)

                                # velocity at the contact point
                                v_ib = (np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                                # decomposition in the local reference frame defined at the contact point
                                v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                                # v_ib_t = v_ib - v_ib_n
                                v_ib_t = np.around(v_ib - v_ib_n, 13)

                                # tangential unit vector should be tangential_ij
                                if np.linalg.norm(v_ib_t) != 0:  # welche grenze?
                                    t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                                else:
                                    t_ib = np.array([0, 0, 0])
                                increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                                if self.contact_model == "LSD":
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                           t_ib)) + k_t_eq * increment_of_t_displacement
                                    tan_force_trial = abs(np.dot(pi.pb_force[boundary.id],
                                                                 t_ib)) * t_ib + k_t_eq * increment_of_t_displacement * t_ib+ c_t_eq * v_ib_t

                                    # should damping in the check for sliding be included?

                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(
                                            max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)
                                    else:
                                        f_t = np.linalg.norm(tan_force)  # * tan_force/np.linalg.norm(tan_force)

                                    f_n_vector = (
                                            -interpenetration * k_n_eq * normal_ib - interpenetration_vel * c_n_eq * normal_ib)

                                    if np.dot(f_n_vector, normal_ib) >= 0:
                                        # exclude attracting forces
                                        f_n_vector = np.array([0, 0, 0])
                                    # else:
                                    # print('normal force positive')
                                    pi.pb_force[boundary.id] = np.array(f_n_vector - f_t * t_ib)  # + t_ij

                                    # -- torque
                                    moment = -f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                else:  # HMD
                                    f_n = 2 / 3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel
                                    if np.dot(pi.pb_force[boundary.id], normal_ib) >= 0:
                                        tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                               t_ib)) + k_t_eq * increment_of_t_displacement
                                    else:
                                        if len(pi.interpenetrations_pb[boundary.id]) >= 2:
                                            k_t_eq_prev_step = 8 * G_eq * np.sqrt(
                                                radius_eq * pi.interpenetrations_pb[boundary.id][-2])
                                        else:
                                            k_t_eq_prev_step = 0
                                        tan_force = abs(
                                            np.dot(pi.pb_force[boundary.id], t_ib)) * (k_t_eq / (
                                            k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                                    tan_force_trial = tan_force + c_t_eq * v_ib_t
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)

                                    pi.pb_force[boundary.id] = np.array(-f_n * normal_ib - f_t * t_ib)

                                    # torque
                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                # particle boundary energies
                                if self.contact_model == "LSD":
                                    pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                else:  # HMD
                                    pi.energy_el.append((pi.energy_el[-1] + 2 / 3 * k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])) * 0.5)

                            # particle boundary contact with second vertex
                            elif np.linalg.norm(pi.position - boundary.end_point) <= pi.radius:
                                poc = np.array([boundary.end_point[0], boundary.end_point[1], 0])
                                normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)

                                interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                                interpenetration_vel = np.dot(pi.velocity, normal_ib)
                                r_ibc = poc - pi.position
                                pi.interpenetrations_pb[boundary.id].append(interpenetration)

                                # velocity at the contact point
                                v_ib = (np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                                # decomposition in the local reference frame defined at the contact point
                                v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                                # v_ib_t = v_ib - v_ib_n
                                v_ib_t = np.around(v_ib - v_ib_n, 13)

                                # tangential unit vector should be tangential_ij
                                if np.linalg.norm(v_ib_t) != 0:  # welche grenze?
                                    t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                                else:
                                    t_ib = np.array([0, 0, 0])
                                increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                                if self.contact_model == "LSD":
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                           t_ib)) + k_t_eq * increment_of_t_displacement
                                    tan_force_trial = abs(np.dot(pi.pb_force[boundary.id],
                                                                 t_ib)) * t_ib + k_t_eq * increment_of_t_displacement * t_ib + c_t_eq * v_ib_t
                                    # tan_force = tan_force + c_t_eq * v_ib_t
                                    # should damping in the check for sliding be included?

                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(
                                            max_friction_force)  # * tan_force_trial/np.linalg.norm(tan_force_trial)
                                        # #print('max friction force', max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force)  # * tan_force/np.linalg.norm(tan_force)

                                    f_n_vector = (
                                            -interpenetration * k_n_eq * normal_ib - interpenetration_vel * c_n_eq * normal_ib)

                                    if np.dot(f_n_vector, normal_ib) >= 0:
                                        # exclude attracting forces
                                        f_n_vector = np.array([0, 0, 0])
                                    # else:
                                    # print('normal force positive')
                                    pi.pb_force[boundary.id] = np.array(f_n_vector - f_t * t_ib)  # + t_ij

                                    # -- torque
                                    moment = -f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                else:  # HMD
                                    f_n = 2 / 3 * k_n_eq * interpenetration + c_n_eq * interpenetration_vel

                                    if np.dot(pi.pb_force[boundary.id], normal_ib) >= 0:
                                        tan_force = abs(np.dot(pi.pb_force[boundary.id],
                                                               t_ib)) + k_t_eq * increment_of_t_displacement
                                    else:
                                        if len(pi.interpenetrations_pb[boundary.id]) >= 2:
                                            k_t_eq_prev_step = 8 * G_eq * np.sqrt(
                                                radius_eq * pi.interpenetrations_pb[boundary.id][-2])
                                        else:
                                            k_t_eq_prev_step = 0
                                        tan_force = abs(
                                            np.dot(pi.pb_force[boundary.id], t_ib)) * (k_t_eq / (
                                            k_t_eq_prev_step) + k_t_eq) * increment_of_t_displacement

                                    tan_force_trial = tan_force + c_t_eq * v_ib_t
                                    max_friction_force = self.mu * np.dot(pi.pb_force[boundary.id], normal_ib)
                                    if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force_trial):
                                        f_t = np.linalg.norm(max_friction_force)
                                    else:
                                        f_t = np.linalg.norm(tan_force_trial)

                                    pi.pb_force[boundary.id] = np.array(-f_n * normal_ib - f_t * t_ib)

                                    # torque
                                    moment = - f_t * np.cross(r_ibc, t_ib)
                                    all_bt.append(moment)

                                # pb contact with second vertex (linear equation)
                                if self.contact_model == "LSD":
                                    pi.energy_el.append((pi.energy_el[-1] + k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                else:  # HMD
                                    pi.energy_el.append((pi.energy_el[-1] + 2 / 3 * k_n_eq * interpenetration * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])))

                                pi.energy_damp.append((pi.energy_damp[-1] + c_n_eq * interpenetration_vel * (
                                            interpenetration - pi.interpenetrations_pb[boundary.id][-2])) * 0.5)

                            else:
                                pi.pb_force[boundary.id] = np.array([0, 0, 0])
                        # no contact
                        else:

                            pi.pb_force[boundary.id] = np.array([0, 0, 0])
                            pi.energy_el.append(0)
                            pi.energy_damp.append(0)
                            pi.interpenetrations_pb[boundary.id].append(0)

                if len(all_bt) != 0:
                    pi.pb_torque = sum(all_bt)

            for particle in self.particles:
                particle.force = sum(particle.pp_force) + sum(particle.pb_force) #.values()
                particle.torque = particle.pp_torque + particle.pb_torque
                if particle.id == 0:
                    self.plot_forces.append(particle.force)
                if self.gravity == True:
                    gravity = np.array([0, 10, 0])
                else:
                    gravity = np.array([0, 0, 0])

                particle.acceleration = gravity + np.array(particle.force) * (1 / particle.mass)
                particle.rotation_acc = np.array(particle.torque) / particle.moment_of_inertia

                particle.velocity = particle.velocity + 0.5 * self.dt * particle.acceleration
                particle.rotation_vel = particle.rotation_vel + 0.5 * self.dt * particle.rotation_acc

                particle.historic_positions.append(particle.position)
                particle.historic_rotations.append(particle.rotation)

            # Progress Tracker
            self.elapsed_time = time.time() - start_time
            remaining_time = (self.total_iterations - iteration - 1) * self.elapsed_time / (iteration + 1)
            hours = float(remaining_time // 3600)
            minutes = float((remaining_time % 3600) // 60)
            seconds = float(remaining_time % 60)
            self.remaining_time = f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"
            self.iterationChanged.emit(iteration + 1)
            self.total_iterationsChanged.emit(self.total_iterations)
            self.remaining_timeChanged.emit(self.remaining_time)


        caluclation_time = time.time() - start_time
        calc_hours = float(caluclation_time // 3600)
        calc_minutes = float((caluclation_time % 3600) // 60)
        calc_seconds = float(caluclation_time % 60)
        print('calculation time: ', f"{calc_hours:02.0f}:{calc_minutes:02.0f}:{calc_seconds:02.0f}", '[hh:mm:ss]')
        print('calculation time [s]: ', caluclation_time)
        plot_forces = []
        for force in self.plot_forces:
            plot_forces.append(np.linalg.norm(force))



        # total energy for each time step
        '''
        print('time steps', len(self.plot_time_steps))
        print('pi kin',len(self.particles[0].energy_kin))
        print('pi rot',len(self.particles[0].energy_rot))
        print('pi el',len(self.particles[0].energy_damp))
        print('pi damp',len(self.particles[0].energy_el))
        print('pi e tan', len(self.particles[0].energy_tan))
        print('pi e pot', len(self.particles[0].energy_pot))
        #print('time steps', len(self.plot_time_steps))
        print('pj kin',len(self.particles[1].energy_kin))
        print('pj rot',len(self.particles[1].energy_rot))
        print('pj el',len(self.particles[1].energy_damp))
        print('pj damp',len(self.particles[1].energy_el))
        print('pj tan', len(self.particles[1].energy_tan))
        print('p pot', len(self.particles[1].energy_pot))
        
        for step in range(0, len(self.plot_time_steps)):
            total_en = 0
            for particle in self.particles:
                total_en += particle.energy_kin[step] + particle.energy_rot[step] + particle.energy_el[step] + particle.energy_damp[step] + particle.energy_pot[step] #+ particle.energy_tan[step]
            self.total_energy.append(total_en)

        mpl.use("Qt5Agg")
        
        ## plotting energies

        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig, ax1 = plt.subplots(1)
        # fig.tight_layout()
        # fig.canvas.set_window_title('Developement of v_ij and the Torque between two Particles over Time(Oblique Impact)')
        #ax1.title.set_text('Energy Distribution Oblique Elastic Impact')
        # Plot the data

        ax1.plot(self.plot_time_steps, self.total_energy, color=(0 / 255, 0 / 255, 0 / 255), label='total energy')
        #ax1.plot(self.plot_time_steps, self.damping_force, color=(0 / 255, 0 / 255, 0 / 255), label='Damping Force')
        #ax1.plot(self.plot_time_steps, self.elastic_force, color=(0 / 255, 101 / 255, 189 / 255), label='Elastic Force')
        ax1.plot(self.plot_time_steps, self.particles[0].energy_kin, color=(0 / 255, 101 / 255, 189 / 255), label='pi e_kin')
        ax1.plot(self.plot_time_steps, self.particles[1].energy_kin, color=(227 / 255, 114 / 255, 34 / 255), label='pj e_kin')

        #ax1.plot(self.plot_time_steps, self.particles[0].energy_rot, color=(196 / 255, 101 / 255, 27 / 255), label='pi e_rot')
        #ax1.plot(self.plot_time_steps, self.particles[1].energy_rot, color=(255 / 255, 220 / 255, 0 / 255), label='pj e_rot')

        ax1.plot(self.plot_time_steps, self.particles[0].energy_el, color=(162 / 255, 173 / 255, 0 / 255), label='pi e_el')
        ax1.plot(self.plot_time_steps, self.particles[1].energy_el, color=(88 / 255, 88 / 255, 90 / 255), label='pj e_el')

        #ax1.plot(self.plot_time_steps, self.particles[0].energy_tan, color=(128 / 255, 0 / 255, 128 / 255), label='pi e_tan')
        #ax1.plot(self.plot_time_steps, self.particles[1].energy_tan, color=(165 / 255, 42 / 255, 42 / 255), label='pj e_tan')


        ax1.plot(self.plot_time_steps, self.particles[0].energy_damp, color=(196 / 255, 101 / 255, 27 / 255), label='pi e_damp')
        ax1.plot(self.plot_time_steps, self.particles[1].energy_damp, color=(255 / 255, 220 / 255, 0 / 255), label='pj e_damp')

        # Set the axis labels and title
        ax1.set_xlabel('Simulation Time [s]')
        #ax2.set_xlabel('Simulation Time [s]')
        #ax3.set_xlabel('Simulation Time [s]')
        ax1.set_ylabel('Energy [J]')
        #ax1.set_ylabel('Norm of Damping Force [N]')
        #ax2.set_ylabel('Kinetic Energy [J]')
        #ax3.set_ylabel('Interaction Energy [J]')

        # ax.set_title('Penetration Over Time Steps')
        ax1.legend()
        #ax2.legend()
        #ax3.legend()

        ax1.grid()
        #ax2.grid()
        #ax3.grid()

        ax1.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))
        #ax2.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))
        #ax3.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))

        # Save the plot as a PDF file in a directory
        # plt.savefig('C:/Users/Jaist/Desktop/plots/'+str(self.dt)+'.svg')
        # plt.savefig('C:/Users/Jaist/Desktop/plots/' + str(self.dt) + '.png')
        plt.show()
        #plt.savefig('C:/Users/Jaist/Desktop/plots/' + 'Energy_oblique'+ self.contact_model+'dt'+str(self.dt)+'cor'+str(self.coeff_of_restitution)+'.pdf')
        plt.savefig(
            'C:/Users/Jaist/Desktop/plots/' + 'Energy_central_dampened_stiff.pdf')
        '''











        """
        ## plotting examples
        # Create a figure and axis object
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.tight_layout()
        #fig.canvas.set_window_title('Developement of v_ij and the Torque between two Particles over Time(Oblique Impact)')
        ax1.title.set_text('Developement of v_ij and the Torque between two Particles over Time(Oblique Impact)')
        # Plot the data

        #ax.plot(self.plot_time_steps, self.interpenetrations, color=(0 / 255, 101 / 255, 189 / 255), label='interpenetration')
        #ax.plot(self.plot_time_steps, self.interpenetrations, color=(227 / 255, 114 / 255, 34 / 255), label='maximum predicted interpenetration')
        ax1.plot(self.plot_time_steps, self.velo, color=(0 / 255, 101 / 255, 189 / 255), label='v_ij')
        ax2.plot(self.plot_time_steps, self.momente, color=(227 / 255, 114 / 255, 34 / 255), label='momente')
        ax3.plot(self.plot_time_steps, self.vi, label = 'vi')
        ax3.plot(self.plot_time_steps, self.vj, label = 'vj')
        ax3.plot(self.plot_time_steps, self.vjrotvel, label = 'vjrotvelo')


        # Set the axis labels and title
        ax1.set_xlabel('Simulation Time')
        ax2.set_xlabel('Simulation Time')
        ax3.set_xlabel('Simulation Time')
        ax1.set_ylabel('sgn(v_ij_y) * norm(v_ij)')
        ax2.set_ylabel('Torque')
        ax3.set_ylabel('Geschwindigkeiten am Kontaktpunkt')

        # ax.set_title('Penetration Over Time Steps')
        ax1.legend()
        ax2.legend()
        ax3.legend()

        ax1.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))
        ax2.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))
        ax3.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))

        # Save the plot as a PDF file in a directory
        #plt.savefig('C:/Users/Jaist/Desktop/plots/'+str(self.dt)+'.svg')
        #plt.savefig('C:/Users/Jaist/Desktop/plots/' + str(self.dt) + '.png')
        plt.savefig('C:/Users/Jaist/Desktop/plots/' + str(self.dt) + 'fixed_tangential.pdf')

        # Display the plot
        plt.show()

        #print("E0 ", self.tot_kin_energy[0])
        #print("E-1 ", self.tot_kin_energy[-1])
       """


    ## for interpenetration and interpenetration vel -----------------------------------------------------------
    """
    mpl.use("Qt5Agg")

    fig, ax = plt.subplots()
    ax.plot(self.plot_time_steps, self.interpenetrations, color=(0 / 255, 101 / 255, 189 / 255), label="Interpenetrations" )
    ax.plot(self.plot_time_steps, self.interpenetrationvels, color=(227 / 255, 114 / 255, 34 / 255),
            label="Interpenetration Velocities")
    #ax.invert_xaxis()
    #ax.scatter(t_incrs, rel_errors, marker='x', color=(227 / 255, 114 / 255, 34 / 255), zorder=5)
    # ax.plot(t_incrs, rel_errors, color=(0 / 255, 101 / 255, 189 / 255))
    # ax.plot(self.plot_time_steps, self.interpenetrations, color=(227 / 255, 114 / 255, 34 / 255), label='maximum predicted interpenetration')

    # Set the axis labels and title
    ax.set_ylabel('Interpenetration [m], Interpenetration-Velocities [m/s]')
    ax.set_xlabel('Simulation Time [s]')
    ax.legend(loc="upper right")
    # plt.savefig('C:/Users/Jaist/Desktop/plots/'+'dt error'+'.svg')
    plt.savefig('C:/Users/Jaist/Desktop/plots/' + 'LSD_forces' + '.pdf')

    plt.show()
    """