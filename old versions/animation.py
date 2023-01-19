import os

import pygame
from pygame import *
import sys  # wofür?
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle
from boundary_class import boundary
import random
import itertools as it


###############################################################################

# -- defs
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

# initialising pygame
pygame.init()

# creating display
screen = pygame.display.set_mode((800, 800))
display.set_caption('simulation of two particles')

# animation
animationTimer = time.Clock()

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

# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation[0,0,w], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
zentral_1 = particle(np.array([450, 400, 0]), np.array([100, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 10000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
zentral_2 = particle(np.array([600, 400, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 10000, 50, np.array([300, 300, 0]), np.array([0, 0, 0]))
# p3 = particle(np.array([200, 200, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
# np.array([0, 0, 0]), 50, 100, 50000000, np.array([200, 200, 0]))

# p3 = particle(np.array([200,600]), np.array([5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))
# p4 = particle(np.array([600,600]), np.array([-5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))

damp_coeff = 5
# initialization
dt = 0.01
simtime = 2  # number max steps for simulation

###############################  timeloop  #################################
for t in np.arange(0, simtime, dt):
    ############################  loop of Particle

    for n_particle in particle.list_of_particles:
        # integration of motion with verlocity verlet (predict)
        pred_vel05 = n_particle.velocity + 0.5 * dt * n_particle.acceleration
        pred_posi = n_particle.position + dt * pred_vel05
        # update position
        n_particle.pred_position = np.around(pred_posi, decimals=4)

    # contact detection with pred_posi
    for index, pi in enumerate(particle.list_of_particles):
        for pj in particle.list_of_particles[index + 1:]:
            cij = pi.pred_position - pj.pred_position
            norm_cij = np.sqrt(cij[0] ** 2 + cij[1] ** 2)  # =norm_cji
            normal_ij = (pj.pred_position - pi.pred_position) / norm_cij
            normal_ji = (pi.pred_position - pj.pred_position) / norm_cij
            elstiffnesn_eq = (pi.elstiffnesn * pj.elstiffnesn) / (
                        pi.elstiffnesn + pj.elstiffnesn)  # Quellenangabe wäre gut
            m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            radius_eq = (pi.radius*pj.radius) / (pi.radius+pj.radius)
            # elstiffnesn_eq = elstiffnesn_eq*radius_eq*np.sqrt(2)
            # contact detection
            if norm_cij < pi.radius + pj.radius:
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position - pi.pred_position), normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)
                print("contact", "penetra", interpenetration, interpenetration_vel, v)
                i_acc = np.linalg.norm(interpenetration_acc)



                if np.linalg.norm(interpenetration_vel) <= 0.01:    # i_acc<=0 and interpenetration_vel = negativ aber VZ geht durch norm verloren
                    pi.force = [0, 0, 0]
                    pj.force = [0, 0, 0]
                else:
                    pi.force = - interpenetration * elstiffnesn_eq * normal_ij - interpenetration_vel * damp_coeff * normal_ij
                    pj.force = - pi.force  # - interpenetration * pj.elstiffnesn * normal_ji - interpenetration_vel * damp_coeff * normal_ji

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
            print("max_penetr: ", interpenetration_max)
            # particle doesn't reach the max. interpenetration
            i_acc = np.linalg.norm(interpenetration_acc)
            print("i_acc :", i_acc)

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

            energy_i = 0.5 * pi.mass * np.linalg.norm(pi.velocity ** 2)
            energy_j = 0.5 * pj.mass * np.linalg.norm(pj.velocity ** 2)
            energy_el = 0.5 * np.sqrt(2)/2 * interpenetration ** 2 * elstiffnesn_eq # mit * np.sqrt(2)/2 und alter elstiffnes_eq klappts
            energy_damp = 0.5 * damp_coeff * np.linalg.norm(interpenetration_vel) * interpenetration
            energy = energy_i + energy_j + energy_el + energy_damp
            print(pi.position, pi.velocity, pi.acceleration, pi.force)
            print(pj.position, pj.velocity, pj.acceleration, pj.force)
            print(energy)
            print('----')

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




    # '''
    #### drawing section
    same_colour = False  # set False for different colours of the particles
    if same_colour == True:
        screen.fill((100, 100, 200))
        for n_particle in particle.list_of_particles:
            draw.circle(screen, (255, 0, 0), (n_particle.position[0], n_particle.position[1]), n_particle.radius)
    else:
        screen.fill((100, 100, 200))
        for indexc, n_particle in enumerate(particle.list_of_particles):
            chosen_c = colour_list[indexc]  # choosing the colour
            chosen_c_rgb = get_rgb(chosen_c)  # turning colour name to rgb
            draw.circle(screen, chosen_c_rgb, (n_particle.position[0], n_particle.position[1]), n_particle.radius)
    #draw.line(screen, get_rgb("Green"), (200, 200), (600, 600), width=5)
    # limit to 30 fps
    animationTimer.tick(30)

    display.update()
### end of drawing section
######################################   timeloop  ##########################
sum_ekin = []
for i in range(len(en_i)):
    sum_ekin.append(en_i[i]+en_j[i])

interpen_pred_max = max(interpen_pred)
for ind, val in enumerate(interpen_pred):
    interpen_pred[ind] = interpen_pred_max
interpen_error = round((max(interpen_calc) - interpen_pred_max) / max(interpen_calc), 2)
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('physical behaviour with attracting forces')

ax1.plot(t_points, en_el, color='tab:red', label='energy in spring')
ax1.plot(t_points, en_damp, color='tab:green', label='energy in dashpot')
ax1.plot(t_points, en_i, color='tab:gray', label='ekin pi')
ax1.plot(t_points, en_j, color='tab:gray', label='ekin pj')
ax1.plot(t_points, en, color='tab:pink', label='total energy')
ax1.plot(t_points, sum_ekin, color='tab:blue', label='total kinetic energy')
#ax1.plot(t_points, en_dissipated_t, color='tab:orange', label='dissipated energy')
# ax1.plot(t_points, en_damp, color='tab:blue', label='energy damp')
ax1.set(xlabel='time', ylabel='energy of the system', title='energy dissipation')
ax1.grid()
ax1.legend(loc='upper right', shadow=True, fontsize='medium')
Penetration = False
if Penetration == True:
    ax2.plot(t_points, interpen_calc, color='tab:blue', label='interpen calc')
    ax2.plot(t_points, interpen_pred, color='tab:orange', label='interpen pred')
    ax2.set(xlabel='time', ylabel='interpenetration', title='interpenetration')
    ax2.annotate("error of interpenetration: " + str(interpen_error), xy=(1, 1),  # xycoords='data',
                xytext=(0.8, 0.95), textcoords='axes fraction',
                # arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top')
else:
    ax2.plot(t_points, cont_forces, color='tab:blue', label='contact forces')
    ax2.set(xlabel='time', ylabel='interpenetration', title='contact forces')

ax2.grid()
ax2.legend(loc='best', shadow=True, fontsize='medium')

ax3.grid()
ax3.plot(t_points, interpen_vel, color='tab:blue', label='interpen vel')
ax3.plot(t_points, interpen_acc, color='tab:green', label='interpen acc')
ax3.plot(t_points, acc, color='tab:red', label='particle acceleration')
ax3.legend(loc='upper right', shadow=True, fontsize='medium')
ax3.set(xlabel='time', ylabel='force', title='contact')
# ax4.grid()
# ax4.plot(t_points, cont_forces, color='tab:blue', label='contact force')
ymax = max(acc)
xpos = np.where(acc == ymax)
xpos = int(xpos[0])
xmax = t_points[xpos]
ax3.annotate(
    'max= '+str(round(ymax, 2)),
    xy=(xmax, ymax), xycoords='data',
    xytext=(-50, 30), textcoords='offset points',
    arrowprops=dict(arrowstyle="->"))
fig.tight_layout()
fig.set_figwidth(10)
fig.set_figheight(10)
fig.savefig("no_attracting_forces_rigid_dampened_correct.png")
plt.show()
pygame.quit()

# script_dir = os.path.dirname(__file__)
# results_dir = os.path.join(script_dir, 'Results')
file_name = "new_damping"
fig.savefig("C:/Users/Jaist/Documents/GitHub/BA_DEM/plots_1301/" + file_name + ".png")

print("Gesamtenergie vor Stoß: ", en[0])
print("Gesamtenergie nach Stoß: ", en[-1])
