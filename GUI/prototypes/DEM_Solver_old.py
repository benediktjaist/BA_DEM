# -*- coding: utf-8 -*-
# -- imports --
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as mcolors
from Particle import Particle
from Boundary import Boundary
# import functions as fn
import test_cor as cor
from typing import List
import sympy as smp
import time
from PyQt6.QtCore import QObject, pyqtSignal



class System(QObject):
    iterationChanged = pyqtSignal(int)
    total_iterationsChanged = pyqtSignal(int)
    remaining_timeChanged = pyqtSignal(str)


    def __init__(self, particles: List[Particle], boundaries: List[Boundary], dt: float, simtime: float, mu: float, coeff_of_restitution: float, gravity = False):
        super().__init__()
        self.particles = particles
        self.boundaries = boundaries
        self.dt = dt
        self.simtime = simtime
        self.mu = mu
        self.coeff_of_restitution = coeff_of_restitution
        self.gravity = gravity
        self.damp_coeff = self.__calculate_damping_coefficient()
        self.crit_dt = self.__calculate_critical_time_step()
        self.total_iterations = int(simtime/dt)
        self.elapsed_time = 0
        self.remaining_time = 0
        # plotting examples
        self.plot_time_steps = []
        self.interpenetration = []
        self.tot_kin_energy = []
        self.plot_interpenetrations = []

        if dt > self.crit_dt:
            print("WARNING: dt > crit_dt. Setting dt to", np.round(self.crit_dt, 4))
            self.dt = self.crit_dt

    def __calculate_damping_coefficient(self):
        return cor.RootByBisection(0.0, 16.0, 0.0001, 300, self.coeff_of_restitution)

    def __calculate_critical_time_step(self):
        crit_steps = []
        for particle in self.particles:
            crit_steps.append(0.3 * 2 * np.sqrt(particle.mass / particle.k_n))
        return min(crit_steps)

    def run_simulation(self):
        start_time = time.time()
        for iteration, t in enumerate(np.arange(0, self.simtime, self.dt)):
            for n_particle in self.particles:
                pred_vel05 = n_particle.velocity + 0.5 * self.dt * n_particle.acceleration
                pred_posi = n_particle.position + self.dt * pred_vel05
                n_particle.position = pred_posi
                n_particle.velocity = pred_vel05

                pred_rot_vel05 = n_particle.rotation_vel + 0.5 * self.dt * n_particle.rotation_acc
                pred_rota = n_particle.rotation + self.dt * pred_rot_vel05
                n_particle.rotation = pred_rota
                n_particle.rotation_vel = pred_rot_vel05

                # update ekin, erot, epot
                n_particle.energy_kin.append(0.5 * n_particle.mass * (np.linalg.norm(n_particle.velocity)) ** 2)
                n_particle.energy_rot.append(0.5 * n_particle.moment_of_inertia * (np.linalg.norm(n_particle.rotation_vel)) ** 2)
                # n_particle.energy_pot.append(0)

                # setze kräfte und momente zu null um aus den vergangenen loops nichts zu übernehmen
                n_particle.pp_force = np.array([0,0,0])
                n_particle.pp_torque = np.array([0,0,0])
                n_particle.force = np.array([0,0,0])
                n_particle.torque = np.array([0,0,0])


            for index, pi in enumerate(self.particles):
                if len(self.particles) == 1:
                    pi.acceleration = np.array(pi.force) * (1 / pi.mass)
                    pi.rotation_acc = np.array(pi.torque) / pi.moment_of_inertia
                    pi.velocity = pi.velocity + 0.5 * self.dt * pi.acceleration
                    pi.rotation_vel = pi.rotation_vel + 0.5 * self.dt * pi.rotation_acc


                for pj in self.particles[index + 1:]:
                    cij = pi.position - pj.position
                    norm_cij = np.sqrt(cij[0] ** 2 + cij[1] ** 2)
                    normal_ij = (pj.position - pi.position) / norm_cij
                    normal_ji = (pi.position - pj.position) / norm_cij

                    elstiffnesn_eq = (pi.k_n * pj.k_n) / (pi.k_n + pj.k_n)
                    m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
                    radius_eq = (pi.radius * pj.radius) / (pi.radius + pj.radius)
                    k_t = elstiffnesn_eq * 0.8

                    if norm_cij < pi.radius + pj.radius:
                        interpenetration = pi.radius + pj.radius - np.dot((pj.position - pi.position), normal_ij)
                        interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                        interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                        v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)

                        i_acc = np.linalg.norm(interpenetration_acc)

                        r_ijc = (pi.radius - (
                                pj.k_n / (pi.k_n + pj.k_n)) * interpenetration) * normal_ij
                        r_jic = (pj.radius - (
                                pi.k_n / (pi.k_n + pj.k_n)) * interpenetration) * normal_ji

                        p_ijc = pi.position + r_ijc  # ortsvektor/point of contact from p1
                        p_jic = pj.position + r_jic  # point of contact from p2 ==p_ijc
                        # velocity at the contact point
                        v_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
                        # decomposition in the local reference frame defined at the contact point
                        v_ij_n = (np.dot(v_ij, normal_ij) * normal_ij)
                        v_ij_t = v_ij - v_ij_n

                        # tangential unit vector should be tangential_ij
                        if np.linalg.norm(v_ij_t) != 0:  # welche grenze?
                            t_ij = v_ij_t / np.linalg.norm(v_ij_t)
                        else:
                            t_ij = np.array([0,0,0])

                        increment_of_t_displacement = np.linalg.norm(v_ij_t) * self.dt

                        max_friction_force = self.mu*np.dot(pi.force, normal_ij)
                        tan_force = np.dot(pi.force, t_ij) + k_t * increment_of_t_displacement

                        if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force):
                            f_t = np.linalg.norm(max_friction_force)
                        else:
                            f_t = np.linalg.norm(tan_force)

                        pi.pp_force = pi.pp_force + np.array(-interpenetration * elstiffnesn_eq * normal_ij - interpenetration_vel * self.damp_coeff * normal_ij - f_t * t_ij)  # nicht normal_ij!! # k_t * f_t oder mu* np.array(pi.force) geht nicht
                        pj.pp_force = - pi.pp_force

                        # -- torque
                        # moment = f_t * np.linalg.norm(r_ijc)
                        moment = f_t * np.cross(r_ijc, t_ij)

                        pi.pp_torque = moment
                        pj.pp_torque = - pi.pp_torque

                        # energy
                        pi.energy_el.append(0.5 * (0.5 * interpenetration ** 2 * elstiffnesn_eq))
                        pi.energy_damp.append(
                            0.5 * (0.5 * self.damp_coeff * np.linalg.norm(interpenetration_vel) * interpenetration))

                        pj.energy_el.append(0.5 * (0.5 * interpenetration ** 2 * elstiffnesn_eq))
                        pi.energy_damp.append(
                            0.5 * (0.5 * self.damp_coeff * np.linalg.norm(interpenetration_vel) * interpenetration))



                    else:
                        interpenetration = 0
                        interpenetration_vel = 0
                        interpenetration_acc = 0
                        pi.pp_force = pi.pp_force + np.array([0, 0, 0])
                        pj.pp_force = pj.pp_force + np.array([0, 0, 0])
                        pi.pp_torque = pi.pp_torque + np.array([0, 0, 0])
                        pj.pp_torque = pj.pp_torque + np.array([0, 0, 0])

                        pi.energy_el.append(0)
                        pi.energy_damp.append(0)

                        pj.energy_el.append(0)
                        pj.energy_damp.append(0)

                        # WARNING
                        # multiple contact energy tracking is missing
                    #'''
                    rel_vel = np.linalg.norm(pi.velocity - pj.velocity)
                    omega = np.sqrt(elstiffnesn_eq / m_eq)  # wie bestimmt man systemsteifigkeit für k(pi) =/= k(pj)
                    psi = self.damp_coeff / (2 * m_eq)  # =0 für lin. elastisch wird später mit COR bestimmt werden

                    interpenetration_max = (rel_vel / omega) * np.exp(-(psi / omega) * np.arctan(omega / psi))
                    #'''
                    # interpenetration_max = 5
                    # plotting test
                    self.plot_interpenetrations.append(interpenetration_max)
                    self.plot_time_steps.append(t)
                    self.tot_kin_energy.append((pi.energy_kin[-1]+pj.energy_kin[-1]))
                    self.interpenetration.append(interpenetration)
            # contact with boundaries
            for pi in self.particles:

                all_bf = []
                all_bt = []

                all_bp_energies = []

                for boundary in self.boundaries:
                    # check if boundary can be written as linear equation
                    if boundary.start_point[0] == boundary.end_point[0]:

                        # compute the distance between the center of the circle and the vertical line
                        dist = abs(pi.position[0] - boundary.start_point[0])

                        # check if particle intersects with boundary
                        if dist <= pi.radius:
                            # solve for intersection points (insert x vals in two branches of circle equation)
                            y_intercepts = []
                            ip1 = np.sqrt(pi.radius ** 2 - (boundary.start_point[0] - pi.position[0]) ** 2) + pi.position[1]
                            ip2 = - np.sqrt(pi.radius ** 2 - (boundary.start_point[0] - pi.position[0]) ** 2) + pi.position[1]
                            y_intercepts.append(ip1)
                            y_intercepts.append(ip2)

                            # compute the point of contact (poc)
                            y_mid = sum(y_intercepts) / 2
                            poc = np.array([boundary.start_point[0], y_mid, 0])
                            print('poc vertikal ', poc)
                            normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)
                            elstiffnesn_eq = pi.k_n
                            m_eq = pi.mass
                            radius_eq = pi.radius
                            k_t = elstiffnesn_eq * 0.8

                            interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                            interpenetration_vel = -np.dot(pi.velocity, normal_ib)
                            r_ibc = poc - pi.position

                            # velocity at the contact point
                            v_ib = -(np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                            # decomposition in the local reference frame defined at the contact point
                            v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                            v_ib_t = v_ib - v_ib_n

                            # tangential unit vector should be tangential_ij
                            if np.linalg.norm(v_ib_t) !=0:  # welche grenze?
                                t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                            else:
                                t_ib = np.array([0,0,0])
                            increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                            max_friction_force = self.mu * np.dot(pi.force, normal_ib)
                            tan_force = np.dot(pi.force, t_ib) + k_t * increment_of_t_displacement

                            if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force):
                                f_t = np.linalg.norm(max_friction_force)
                            else:
                                f_t = np.linalg.norm(tan_force)

                            all_bf.append(np.array(-interpenetration * elstiffnesn_eq * normal_ib - interpenetration_vel * self.damp_coeff * normal_ib - f_t * t_ib))  # nicht normal_ij!! # k_t * f_t oder mu* np.array(pi.force) geht nicht

                            # -- torque
                            moment = f_t * np.linalg.norm(r_ibc)
                            all_bt.append(np.array([0, 0, moment]))

                        else:
                            all_bf.append(np.array([0, 0, 0]))
                            all_bt.append(np.array([0, 0, 0]))

                    # boundary can be expressed as linear function
                    else:
                        '''
                        # compute the circle equations
                        x = smp.symbols('x')
                        y1 = smp.sqrt(pi.radius ** 2 - (x - pi.position[0]) ** 2) + pi.position[1]
                        y2 = -smp.sqrt(pi.radius ** 2 - (x - pi.position[0]) ** 2) + pi.position[1]
                        y3 = boundary.get_lin_eq()

                        solution1 = smp.solveset(y1 - y3, x, domain=smp.Reals)
                        solution2 = smp.solveset(y2 - y3, x, domain=smp.Reals)

                        if solution1 or solution2:
                            # solve for intersection points
                            ip1 = smp.solve(y1 - y3, x)
                            ip2 = smp.solve(y2 - y3, x)
                            nullstellen = np.array((ip1 + ip2), dtype='float64')

                            # delete duplicates
                            nullstellen = [x for i, x in enumerate(nullstellen) if x not in nullstellen[:i]]

                            # compute point of contact (poc)
                            x_mid = sum(nullstellen) / 2
                            poc = np.array([x_mid, y3.evalf(subs={x: x_mid}), 0], dtype='float64')
                            '''

                        m = (boundary.end_point[1] - boundary.start_point[1])/(boundary.end_point[0] - boundary.start_point[0])
                        t = boundary.start_point[1] - m * boundary.start_point[0]
                        A = m**2 + 1
                        B = 2*(m * t - m * pi.position[1] - pi.position[0])
                        C = pi.position[1]**2 - pi.radius**2 + pi.position[0]**2 - 2 * t * pi.position[1] + t**2
                        D = B**2 - 4 * A * C

                        if D >= 0:
                            x1 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                            x2 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

                            x_mid = (x1 + x2)/2
                            poc = np.array([x_mid, m * x_mid + t, 0])
                            print('poc ', poc)

                            normal_ib = (poc - pi.position) / np.linalg.norm(poc - pi.position)
                            # AttributeError: 'Float' object has no attribute 'sqrt'
                            elstiffnesn_eq = pi.k_n
                            m_eq = pi.mass
                            radius_eq = pi.radius
                            k_t = elstiffnesn_eq * 0.8

                            interpenetration = pi.radius - np.dot((poc - pi.position), normal_ib)
                            interpenetration_vel = -np.dot(-pi.velocity, normal_ib) #VZ

                            r_ibc = poc - pi.position

                            # velocity at the contact point
                            v_ib = -(np.cross(pi.rotation_vel, r_ibc) + pi.velocity)
                            # decomposition in the local reference frame defined at the contact point
                            v_ib_n = (np.dot(v_ib, normal_ib) * normal_ib)
                            v_ib_t = v_ib - v_ib_n

                            # tangential unit vector should be tangential_ij
                            if np.linalg.norm(v_ib_t) != 0:
                                t_ib = v_ib_t / np.linalg.norm(v_ib_t)
                            else:
                                t_ib = np.array([0,0,0])

                            increment_of_t_displacement = np.linalg.norm(v_ib_t) * self.dt

                            max_friction_force = self.mu * np.dot(pi.force, normal_ib)
                            tan_force = np.dot(pi.force, t_ib) + k_t * increment_of_t_displacement

                            if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force):
                                f_t = np.linalg.norm(max_friction_force)
                            else:
                                f_t = np.linalg.norm(tan_force)

                            all_bf.append(
                                -interpenetration * elstiffnesn_eq * normal_ib - interpenetration_vel * self.damp_coeff * normal_ib - f_t * t_ib)  # nicht normal_ij!! # k_t * f_t oder mu* np.array(pi.force) geht nicht

                            # -- torque
                            moment = f_t * np.cross(r_ibc, t_ib)
                            all_bt.append(moment)


                pi.pb_force = sum(all_bf)

                pi.pb_torque = sum(all_bt)


                # spring + dashpot energy from all boundaries



            for particle in self.particles:
                particle.force = particle.pp_force + particle.pb_force
                particle.torque = particle.pp_torque + particle.pb_torque

                if self.gravity == True:
                    gravity = np.array([0, 10, 0])
                else:
                    gravity = np.array([0, 0, 0])
                particle.acceleration = gravity + np.array(particle.force) * (1 / particle.mass) # to change gravity
                particle.rotation_acc = np.array(particle.torque) / particle.moment_of_inertia
                particle.velocity = particle.velocity + 0.5 * self.dt * particle.acceleration
                particle.rotation_vel = particle.rotation_vel + 0.5 * self.dt * particle.rotation_acc
                #self.position_tracker.update(time_step_count)
                particle.historic_positions.append(particle.position)
                particle.historic_rotations.append(particle.rotation)
                # print(particle.historic_positions[-1])
                # particle.energy.append(0.5*particle.mass*(np.linalg.norm(particle.velocity))**2
                                       # + 0.5*particle.moment_of_inertia*(np.linalg.norm(particle.rotation_vel))**2)
                print('rotation ', particle.rotation)
                print('rotation velocity ', particle.rotation_vel)
                print('position ', particle.position)
                print('force ', particle.force)

            # Progress Tracker
            self.elapsed_time = time.time() - start_time
            remaining_time = (self.total_iterations - iteration - 1) * self.elapsed_time / (iteration + 1)
            hours = float(remaining_time // 3600)
            minutes = float((remaining_time % 3600) // 60)
            seconds = float(remaining_time % 60)
            # print(f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
            self.remaining_time = f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"
            # print('4567')
            self.iterationChanged.emit(iteration + 1)
            self.total_iterationsChanged.emit(self.total_iterations)
            self.remaining_timeChanged.emit(self.remaining_time)


'''
        ## plotting examples
        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Plot the data

        ax.plot(self.plot_time_steps, self.interpenetration, color=(0 / 255, 101 / 255, 189 / 255), label='interpenetration')
        ax.plot(self.plot_time_steps, self.plot_interpenetrations, color=(227 / 255, 114 / 255, 34 / 255), label='maximum predicted interpenetration')

        # Set the axis labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Interpenetration')
        # ax.set_title('Penetration Over Time Steps')
        ax.legend()
        ax.set_xlim((min(self.plot_time_steps), max(self.plot_time_steps) + self.dt))

        # Save the plot as a PDF file in a directory
        plt.savefig('C:/Users/Jaist/Desktop/plots/'+str(self.dt)+'.svg')
        plt.savefig('C:/Users/Jaist/Desktop/plots/' + str(self.dt) + '.png')
        plt.savefig('C:/Users/Jaist/Desktop/plots/' + str(self.dt) + '.pdf')


        # Display the plot
        plt.show()

        print("E0 ", self.tot_kin_energy[0])
        print("E-1 ", self.tot_kin_energy[-1])
'''