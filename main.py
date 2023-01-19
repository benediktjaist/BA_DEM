# -*- coding: utf-8 -*-
# -- imports --
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from classes import Particle
from classes import Boundary
import functions as fn
import os


# -- initialising pygame
pg.init()
screen = pg.display.set_mode((1000, 800))   # creating display
pg.display.set_caption('simulation of two particles')
animationTimer = pg.time.Clock()

# -- simulation --
# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation_vel[0,0,0], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
#p1 = Particle(np.array([500, 500, 0]), np.array([100, 100, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              #np.array([0, 0, 0]), 50, 2000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
#p4 = Particle(np.array([600, 650, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              #np.array([0, 0, 0]), 50, 2000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
zentral_1 = Particle(np.array([450, 430, 0]), np.array([100, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 2000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
zentral_2 = Particle(np.array([600, 400, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 2000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
#p2 = Particle(np.array([650, 500, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              #np.array([0, 0, 0]), 100, 10000, 100000, np.array([400, 400, 0]), np.array([0, 0, 0]))
#p3 = Particle(np.array([350, 500, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              #np.array([0, 0, 0]), 100, 10000, 100000, np.array([400, 400, 0]), np.array([0, 0, 0]))
# b1 = Boundary(np.array([50, 700, 0]), np.array([700, 700, 0]), np.array([0,0]))
# -- simulation parameters
coeff_of_restitution = 1
print(type(zentral_2.force))
damp_coeff = fn.calculate_damp_coeff(coeff_of_restitution)
mu = 0.3 # Reibkoeffizient
k_t = 1000
crit_steps = []
for particle in Particle.all_particles:
    crit_steps.append(0.3 * 2 * np.sqrt(particle.mass/particle.elstiffnesn))
crit_dt = min(crit_steps)
print("required time step < ", crit_dt)
print("Dämpfung: ", damp_coeff)
dt = 0.01
simtime = 2  # number max steps for simulation
if dt > crit_dt:
    print()
    print("IMPORTANT WARNING\ndt < crit_dt is not satisfied\nchange dt to: ", np.round(crit_dt, 4))
    exit()

# -- plotting
en = []  # gesamtenergie
en_damp = []  # energie des dämpfers
en_dissipated = 0  # historische variable
en_dissipated_t = []
en_el = []  # energie der feder
en_i = []  # ekin
en_j = []  # ekin
t_points = []

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


                #print(t_ij)

                # -- compute increment of tangential displacement (for updating F_t)
                # translation of point of contact
                u_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (
                        np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
                #print("uij: ", u_ij)

                # increment of tangential displacement
                increment_of_t_displacement = np.linalg.norm(u_ij * t_ij)

                # -- forces
                f_t = min(mu*np.dot(pi.force, normal_ij), np.dot(pi.force, t_ij) + k_t * increment_of_t_displacement)
                pi.force = np.array(-interpenetration * elstiffnesn_eq * normal_ij - interpenetration_vel * damp_coeff * normal_ij - f_t * t_ij)  # nicht normal_ij!! # k_t * f_t oder mu* np.array(pi.force) geht nicht
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
            energy, energy_el, energy_i, energy_j, energy_damp = fn.calculate_energies(pi, pj, interpenetration, interpenetration_vel, damp_coeff, elstiffnesn_eq)
            #print(pi.position, pi.velocity, pi.acceleration, pi.force)
            #print(pj.position, pj.velocity, pj.acceleration, pj.force)
            #print('--------------------------')

            en.append(energy)
            t_points.append(t)

            en_i.append(energy_i)
            en_j.append(energy_j)
            en_el.append(energy_el)
            en_damp.append(energy_damp)
            en_dissipated += energy_damp
            en_dissipated_t.append(en_dissipated)



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


sum_ekin = []
for i in range(len(en_i)):
    sum_ekin.append(en_i[i]+en_j[i])


plt.title('physical behaviour')

fig, ax = plt.subplots()
ax.plot(t_points, en, color='tab:pink', label='total energy')
ax.plot(t_points, en_i, color='tab:blue', label='ekin_i')
ax.plot(t_points, en_j, color='tab:blue', label='ekin_j')
ax.plot(t_points, en_el, color='tab:red', label='e_spring')
ax.legend()

at1 = AnchoredText(
    "en_bf: "+str(round(en[0])), prop=dict(size=10), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at1)
at2 = AnchoredText(
    "en_af: "+str(round(en[-1])), prop=dict(size=10), frameon=True, loc='upper right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at2)


file_name = "oblique, elastic, tangential force limited"
plt.savefig("C:/Users/Jaist/Documents/GitHub/BA_DEM/plots_1701/" + file_name + ".png")

print("Gesamtenergie vor Stoß: ", en[0])
print("Gesamtenergie nach Stoß: ", en[-1])

