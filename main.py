# -*- coding: utf-8 -*-
# -- imports --
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
from classes import Particle
from classes import Boundary
import functions as fn
import os


# -- initialising pygame
pg.init()
screen = pg.display.set_mode((800, 800))   # creating display
pg.display.set_caption('simulation of two particles')
animationTimer = pg.time.Clock()

# -- simulation --
# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation_vel[0,0,0], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
p1 = Particle(np.array([200, 100, 0]), np.array([100, 100, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 2000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
p2 = Particle(np.array([500, 500, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 200, 10000, 100000, np.array([400, 400, 0]), np.array([0, 0, 0]))
# b1 = Boundary(np.array([50, 700, 0]), np.array([700, 700, 0]), np.array([0,0]))
# -- simulation parameters
coeff_of_restitution = 0.6
damp_coeff = fn.calculate_damp_coeff(coeff_of_restitution)
mu = 0.3 # Reibkoeffizient
k_t = 1000

print("Dämpfung: ", damp_coeff)
dt = 0.01
simtime = 4  # number max steps for simulation

for t in np.arange(0, simtime, dt):

    for n_particle in Particle.all_particles:
        fn.predict_position(n_particle, dt)

    for index, pi in enumerate(Particle.all_particles):
        if len(Particle.all_particles) == 1:
            fn.update_position_single_particle(pi, dt)
        for pj in Particle.all_particles[index + 1:]:
            norm_cij = fn.compute_norm_cij(pi, pj)
            normal_ij = fn.compute_normal_ij(pi, pj, norm_cij)
            normal_ji = fn.compute_normal_ji(pi, pj, norm_cij)

            # Quellenangabe wäre gut
            elstiffnesn_eq = (pi.elstiffnesn * pj.elstiffnesn) / (pi.elstiffnesn + pj.elstiffnesn)
            m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            radius_eq = (pi.radius*pj.radius) / (pi.radius+pj.radius)

            if norm_cij < pi.radius + pj.radius:
                # interpenetration
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position - pi.pred_position), normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)
                # print("contact", "penetra", interpenetration, interpenetration_vel, v)
                i_acc = np.linalg.norm(interpenetration_acc)

                # -- kinematik für t_ij im lokalen Koordinatensystem
                # hilfsvektoren von center of Particle zu point of contact
                r_ijc = (pi.radius - (
                            pj.elstiffnesn / (pi.elstiffnesn + pj.elstiffnesn)) * interpenetration) * normal_ij
                r_jic = (pj.radius - (
                            pi.elstiffnesn / (pi.elstiffnesn + pj.elstiffnesn)) * interpenetration) * normal_ji
                # position of the contact point
                p_ijc = pi.pred_position + r_ijc  # ortsvektor/point of contact from p1
                p_jic = pj.pred_position + r_jic  # point of contact from p2 ==p_ijc
                # velocity at the contact point
                v_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
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


                print(t_ij)

                # -- compute increment of tangential displacement (for updating F_t)
                # translation of point of contact
                u_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (
                        np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
                print("uij: ", u_ij)

                # increment of tangential displacement
                increment_of_t_displacement = np.linalg.norm(u_ij * t_ij)

                # -- forces
                f_t = min(mu*np.dot(pi.force, normal_ij), k_t * increment_of_t_displacement)
                pi.force = interpenetration * elstiffnesn_eq * normal_ji + interpenetration_vel * damp_coeff * normal_ji  #+ k_t * f_t# nicht normal_ij!!
                pj.force = - pi.force

            else:
                interpenetration = 0
                interpenetration_vel = 0
                interpenetration_acc = 0
                pi.force = [0, 0, 0]
                pj.force = [0, 0, 0]



            rel_vel = np.linalg.norm(pi.velocity - pj.velocity)
            omega = np.sqrt((elstiffnesn_eq) / m_eq)  # wie bestimmt man systemsteifigkeit für k(pi) =/= k(pj)
            psi = damp_coeff / (2 * m_eq)  # = 0 für lin. elastisch und kann später mit coeff of restitution bestimmt werden

            interpenetration_max = (rel_vel / omega) * np.exp(-(psi / omega) * np.arctan(omega / psi))
            # print("max_penetr: ", interpenetration_max)
            # i_acc = np.linalg.norm(interpenetration_acc)
            # print("i_acc :", i_acc)


            ##############################################
            # bisher haben partikel bei schrägen stößen beschleunigt, da die gesamte Kraft pi.force im Schwerpunkt
            # angesetzt wurde, anstatt nur die normale komponente zu nehmen
            #############################################
            fn.update_position(pi, dt, normal_ij)
            fn.update_position(pj, dt, normal_ji)
            #fn.update_position_single_particle(pi, dt)
            #fn.update_position_single_particle(pj, dt)
            fn.calculate_energies(pi, pj, interpenetration, interpenetration_vel, damp_coeff, elstiffnesn_eq)
            print(pi.position, pi.velocity, pi.acceleration, pi.force)
            print(pj.position, pj.velocity, pj.acceleration, pj.force)

            print('--------------------------')



    # contact particles - boundaries
    for n_particle in Particle.all_particles:
        for n_boundary in Boundary.all_boundaries:
            distance = fn.compute_normal_distance(n_boundary.point_1, n_boundary.point_2, n_particle.position)
            print(distance)
            point_of_contact = n_boundary.point_1 + np.dot((n_boundary.point_2-n_boundary.point_2),
                                                           (n_particle.position - n_boundary.point_1))
            #print(point_of_contact)
            n_boundary.point_of_contact = point_of_contact

            if n_particle.radius > distance:
                pass


# -- drawing section for pygame
    same_colour = False  # set False for different colours of the particles
    if same_colour == True:
        screen.fill((100, 100, 200))
        for n_particle in Particle.all_particles:
            pg.draw.circle(screen, (255, 0, 0), (n_particle.position[0], n_particle.position[1]), n_particle.radius)
    else:
        screen.fill((100, 100, 200))
        for indexc, n_particle in enumerate(Particle.all_particles):
            chosen_c = fn.colour_list[indexc]  # choosing the colour
            chosen_c_rgb = fn.get_rgb(chosen_c)  # turning colour name to rgb
            pg.draw.circle(screen, chosen_c_rgb, (n_particle.position[0], n_particle.position[1]), n_particle.radius)

    for n_boundary in Boundary.all_boundaries:
        pg.draw.line(screen, (255, 255, 255), (n_boundary.point_1[0], n_boundary.point_1[1]),
                     (n_boundary.point_2[0], n_boundary.point_2[1]), width=5)
        pg.draw.circle(screen, (255, 255, 255), (n_boundary.point_1[0], n_boundary.point_1[1]), radius=5)
        pg.draw.circle(screen, (0, 0, 0), (n_boundary.point_2[0], n_boundary.point_2[1]), radius=5)
        if np.all(n_boundary.point_of_contact) == 0:
            pass
        else:
            print("particle hits at: ", n_boundary.point_of_contact)
    #draw.line(screen, get_rgb("Green"), (200, 200), (600, 600), width=5)
    # limit to 30 fps
    animationTimer.tick(30)

    pg.display.update()
pg.quit()

