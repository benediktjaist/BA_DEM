# -*- coding: utf-8 -*-
# -- imports --
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as mcolors
from classes import Particle
from classes import Boundary
import functions as fn
import test_cor as cor
from typing import List


class PositionTracker:
    # particle_position_tracker is a list of lists
    # with the first index corresponding to time and the second index corresponding to particle id
    def __init__(self, particles: List[Particle], simtime: float, dt: float):
        self.particles = particles
        self.simtime = simtime
        self.dt = dt
        self.positions = [[] for _ in range(int(self.simtime / self.dt) + 1)]
        self.rotations = [[] for _ in range(int(self.simtime / self.dt) + 1)]
        for i in range(len(self.positions)):
            for particle in particles:
                self.positions[i].append([])
                self.rotations[i].append([])

    def update(self, t: int):
        for i, particle in enumerate(self.particles):
            self.positions[t][i].append(particle.position)
            self.rotations[t][i].append(particle.rotation)


class System:
    def __init__(self, particles: List[Particle], dt: float, simtime: float, mu: float, coeff_of_restitution: float):
        self.particles = particles
        self.dt = dt
        self.simtime = simtime
        self.mu = mu
        self.coeff_of_restitution = coeff_of_restitution
        self.damp_coeff = self.__calculate_damping_coefficient()
        self.crit_dt = self.__calculate_critical_time_step()
        self.position_tracker = PositionTracker(self.particles, self.simtime, self.dt)

        if dt > self.crit_dt:
            print("WARNING: dt > crit_dt. Setting dt to", np.round(self.crit_dt, 4))
            self.dt = self.crit_dt

    def __calculate_damping_coefficient(self):
        return cor.RootByBisection(0.0, 16.0, 0.0001, 300, self.coeff_of_restitution)

    def __calculate_critical_time_step(self):
        crit_steps = []
        for particle in self.particles:
            crit_steps.append(0.3 * 2 * np.sqrt(particle.mass / particle.elstiffnesn))
        return min(crit_steps)

    def get_positions(self):
        return self.position_tracker.positions

    def run_simulation(self):
        for t in np.arange(0, self.simtime, self.dt):
            for n_particle in self.particles:
                pred_vel05 = n_particle.velocity + 0.5 * self.dt * n_particle.acceleration
                pred_posi = n_particle.position + self.dt * pred_vel05
                n_particle.position = pred_posi
                n_particle.velocity = pred_vel05

                pred_rot_vel05 = n_particle.rotation_vel + 0.5 * self.dt * n_particle.rotation_acc
                pred_rota = n_particle.rotation + self.dt * pred_rot_vel05
                n_particle.rotation = pred_rota
                n_particle.rotation_vel = pred_rot_vel05

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

                    elstiffnesn_eq = (pi.elstiffnesn * pj.elstiffnesn) / (pi.elstiffnesn + pj.elstiffnesn)
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
                                    pj.elstiffnesn / (pi.elstiffnesn + pj.elstiffnesn)) * interpenetration) * normal_ij
                        r_jic = (pj.radius - (
                                    pi.elstiffnesn / (pi.elstiffnesn + pj.elstiffnesn)) * interpenetration) * normal_ji

                        p_ijc = pi.position + r_ijc  # ortsvektor/point of contact from p1
                        p_jic = pj.position + r_jic  # point of contact from p2 ==p_ijc
                        # velocity at the contact point
                        v_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
                        # decomposition in the local reference frame defined at the contact point
                        v_ij_n = (np.dot(v_ij, normal_ij) * normal_ij)
                        v_ij_t = v_ij - v_ij_n

                        # tangential unit vector should be tangential_ij
                        if np.linalg.norm(v_ij_t) >= 0.001:  # welche grenze?
                            t_ij = v_ij_t / np.linalg.norm(v_ij_t)
                        else:
                            t_ij = 0

                        increment_of_t_displacement = np.linalg.norm(v_ij_t) * self.dt

                        max_friction_force = self.mu*np.dot(pi.force, normal_ij)
                        tan_force = np.dot(pi.force, t_ij) + k_t * increment_of_t_displacement

                        if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force):
                            f_t = np.linalg.norm(max_friction_force)
                        else:
                            f_t = np.linalg.norm(tan_force)

                        pi.force = np.array(-interpenetration * elstiffnesn_eq * normal_ij - interpenetration_vel * self.damp_coeff * normal_ij - f_t * t_ij)  # nicht normal_ij!! # k_t * f_t oder mu* np.array(pi.force) geht nicht
                        pj.force = - pi.force

                        # -- torque
                        moment = f_t * np.linalg.norm(r_ijc)
                        pi.torque = np.array([0, 0, moment])
                        pj.torque = pi.torque

                    else:
                        interpenetration = 0
                        interpenetration_vel = 0
                        interpenetration_acc = 0
                        pi.force = [0, 0, 0]
                        pj.force = [0, 0, 0]
                        pi.torque = 0
                        pj.torque = 0

                    rel_vel = np.linalg.norm(pi.velocity - pj.velocity)
                    omega = np.sqrt(elstiffnesn_eq / m_eq)
                    psi = self.damp_coeff / (2 * m_eq)

                    interpenetration_max = (rel_vel / omega) * np.exp(-(psi / omega) * np.arctan(omega / psi))

                    pi.acceleration = np.array([0, 0, 0]) + np.array(pi.force) * (1 / pi.mass) # to change gravity
                    pi.rotation_acc = np.array(pi.torque) / pi.moment_of_inertia
                    pi.velocity = pi.velocity + 0.5 * self.dt * pi.acceleration
                    pi.rotation_vel = pi.rotation_vel + 0.5 * self.dt * pi.rotation_acc

                    pj.acceleration = np.array(pj.force) * (1 / pj.mass)
                    pj.rotation_acc = np.array(pj.torque) / pj.moment_of_inertia
                    pj.velocity = pj.velocity + 0.5 * self.dt * pj.acceleration
                    pj.rotation_vel = pj.rotation_vel + 0.5 * self.dt * pj.rotation_acc
                    self.position_tracker.update(int(t / self.dt))
                    # important assumption: simtime is integer divisible by dt
                    print(pi.position)
