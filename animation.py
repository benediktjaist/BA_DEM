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
display.set_caption('this should be an animation')

# animation
animationTimer = time.Clock()

# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation[0,0,w], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
p1 = particle(np.array([300, 300, 0]), np.array([50, 50, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 1000, 50, np.array([300, 300, 0]))
p2 = particle(np.array([400, 400, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              np.array([0, 0, 0]), 50, 1000, 50000, np.array([400, 400, 0]))
#p3 = particle(np.array([200, 200, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
              #np.array([0, 0, 0]), 50, 100, 50000000, np.array([200, 200, 0]))

#p3 = particle(np.array([200,600]), np.array([5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))
#p4 = particle(np.array([600,600]), np.array([-5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))

damp_coeff = 320
# initialization
dt = 0.01
simtime = 40  # number max steps for simulation
# ohne Dämpfung erreiche ich max penetration
# je stärker die dämpfung, desto weiter dringe ich über max penetration ein
# check ob critical dampened auch für subcritical dampened beispiele: c=200 m=50 k =1000 vi= 50,50
# in diesem fall separieren particles auch noch für overdampened mit c=400 obwohl c_crit=320
# mit c_crit und mj=1000*mi verbleibt ca ein Zehntel der max Energy

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
            elstiffnesn_eq = (pi.elstiffnesn * pj.elstiffnesn) / (pi.elstiffnesn + pj.elstiffnesn)      # Quelle dafür wäre gut
            # contact detection
            if norm_cij < pi.radius + pj.radius:
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position - pi.pred_position), normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)
                print("contact", "penetra", interpenetration, interpenetration_vel, v)
                i_acc = np.linalg.norm(interpenetration_acc)
                if np.linalg.norm(pi.force) != 0:
                    if i_acc <= 0.001:
                        pi.force = [0, 0, 0]
                        pj.force = [0, 0, 0]
                    else:
                        pi.force = - interpenetration * pi.elstiffnesn * normal_ij - interpenetration_vel * damp_coeff * normal_ij
                        pj.force = - pi.force #- interpenetration * pj.elstiffnesn * normal_ji - interpenetration_vel * damp_coeff * normal_ji
                else:
                    pi.force = - interpenetration * pi.elstiffnesn * normal_ij - interpenetration_vel * damp_coeff * normal_ij
                    pj.force = -pi.force #- interpenetration * pj.elstiffnesn * normal_ji - interpenetration_vel * damp_coeff * normal_ji

            else:
                interpenetration = 0
                interpenetration_vel = 0
                interpenetration_acc = 0
                pi.force = [0, 0, 0]
                pj.force = [0, 0, 0]

            rel_vel = np.linalg.norm(pi.velocity-pj.velocity)
            m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            omega = np.sqrt((pi.elstiffnesn) / m_eq)  # wie bestimmt man systemsteifigkeit für k(pi) =/= k(pj)
            psi = damp_coeff / (2 * m_eq)  # = 0 für lin. elastisch und kann später mit coeff of restitution bestimmt werden
            interpenetration_max = (rel_vel / omega) * np.exp(-(psi / omega) * np.arctan(omega / psi))
            print("max_penetr: ", interpenetration_max)
            # particle doesnt reach the max. interpenetration
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

            energy_i = 0.5*pi.mass*np.linalg.norm(pi.velocity**2)
            energy_j = 0.5*pi.mass*np.linalg.norm(pj.velocity**2)
            energy = energy_i + energy_j
            print(pi.position, pi.velocity, pi.acceleration, pi.force)
            print(pj.position, pj.velocity, pj.acceleration, pj.force)
            print('----')
            print("Energy:", energy)

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

    # limit to 30 fps
    animationTimer.tick(30)

    display.update()
### end of drawing section
# '''
######################################   timeloop  ##########################

pygame.quit()
sys.exit()