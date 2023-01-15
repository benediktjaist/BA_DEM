# -*- coding: utf-8 -*-
# -- imports --
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
from classes import Particle
import functions as fn
import os


# -- initialising pygame
pg.init()
screen = pg.display.set_mode((800, 800))   # creating display
pg.display.set_caption('simulation of two particles')
animationTimer = pg.time.Clock()

# -- simulation --
# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation[0,0,w], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
p1 = Particle(np.array([300, 300, 0]), np.array([70, 70, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 1000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
p2 = Particle(np.array([400, 400, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 1000, 50, np.array([400, 400, 0]), np.array([0, 0, 0]))

# -- simulation parameters
coeff_of_restitution = 0.5
damp_coeff = fn.calculate_damp_coeff(coeff_of_restitution)

print("Dämpfung: ", damp_coeff)
dt = 0.01
simtime = 2  # number max steps for simulation

for t in np.arange(0, simtime, dt):

    for n_particle in Particle.all_particles:
        fn.predict_position(n_particle, dt)

    for index, pi in enumerate(Particle.all_particles):
        for pj in Particle.all_particles[index + 1:]:
            norm_cij = fn.compute_norm_cij(pi, pj)
            normal_ij = fn.compute_normal_ij(pi, pj, norm_cij)
            normal_ji = fn.compute_normal_ji(pi, pj, norm_cij)

            # Quellenangabe wäre gut
            elstiffnesn_eq = (pi.elstiffnesn * pj.elstiffnesn) / (pi.elstiffnesn + pj.elstiffnesn)
            m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            radius_eq = (pi.radius*pj.radius) / (pi.radius+pj.radius)

            if norm_cij < pi.radius + pj.radius:
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position - pi.pred_position), normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)
                # print("contact", "penetra", interpenetration, interpenetration_vel, v)
                i_acc = np.linalg.norm(interpenetration_acc)

                pi.force = - interpenetration * elstiffnesn_eq * normal_ij - interpenetration_vel * damp_coeff * normal_ij
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

            fn.update_position(pi, dt)
            fn.update_position(pj, dt)
            fn.calculate_energies(pi, pj, interpenetration, interpenetration_vel, damp_coeff, elstiffnesn_eq)

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
    #draw.line(screen, get_rgb("Green"), (200, 200), (600, 600), width=5)
    # limit to 30 fps
    animationTimer.tick(30)

    pg.display.update()
pg.quit()

