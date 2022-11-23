# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:30:46 2022

@author: Jaist
"""
###############################################################################
import pygame
from pygame import *
import sys #wofür?
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
    colour_rgb = {"Black": (0,0,0), "White": (255,255,255), "Red": (255,0,0), "Lime": (0,255,0), "Blue": (0,0,255),
                        "Yellow": (255,255,0), "Cyan": (0,255,255), "Magenta": (255,0,255), "Silver": (192,192,192), "Gray": (128,128,128),
                  "Maroon": (128,0,0), "Olive": (128,128,0), "Green": (0,128,0), "Purple": (128,0,128), "Teal": (0,128,128), "Navy": (0,0,128)}
    rgb = colour_rgb[str(colour)]
    return rgb


colour_list = ["Black", "White", "Red", "Lime", "Blue",
               "Yellow", "Cyan", "Magenta", "Silver", "Gray",
               "Maroon", "Olive", "Green", "Purple", "Teal", "Navy"]
 # '''
# initialising pygame
pygame.init()

# creating display
screen = pygame.display.set_mode((800, 800))
display.set_caption('this should be an animation')

# animation
animationTimer = time.Clock()
 # '''

'''
# labeling and drawing particles
font = pygame.font.SysFont(None, 100)
text = font.render("p"+str(n_particle), True, (255, 255, 0))

colourlist=[]
for n_particle in Particle.list_of_particles:
    colour = (np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255))
    colourlist.append(colour)
'''

# position[x,y, z=0], velocity[dx,dy, dz = 0], acceleration[ddx,ddy, ddz = 0], rotation[0,0,w], force[fx,fy, 0], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
p1 = particle(np.array([300,300,0]), np.array([8,8,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]),50,100,500,np.array([300,300,0]))
p2 = particle(np.array([400,400,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]),50,100,500,np.array([400,400,0]))


#p3 = Particle(np.array([200,600]), np.array([5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))
#p4 = Particle(np.array([600,600]), np.array([-5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))


# initialization
dt = 0.01
simtime = 10 #number max steps for simulation


###############################  timeloop  #################################
for t in np.arange(0, simtime, dt):
    ############################  loop of Particle

    for n_particle in particle.list_of_particles:  
        # integration of motion with verlocity verlet (predict)
        pred_vel05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
        pred_posi = n_particle.position + dt*pred_vel05
        # update position
        n_particle.pred_position = np.around(pred_posi, decimals=4)

    # contact detection with pred_posi
    for index, pi in enumerate(particle.list_of_particles):
        for pj in particle.list_of_particles[index + 1:]:
            cij = pi.pred_position-pj.pred_position
            norm_cij = np.sqrt(cij[0]**2+cij[1]**2)  # =norm_cji
            normal_ij = (pj.pred_position-pi.pred_position)/(norm_cij)
            normal_ji = (pi.pred_position-pj.pred_position)/(norm_cij)

            if norm_cij < pi.radius + pj.radius:
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position-pi.pred_position),normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
            else:
                interpenetration = 0
                interpenetration_vel = 0

            # hilfsvektoren von center of Particle zu point of contact
            r_ijc = (pi.radius-(pj.elstiffnesn/(pi.elstiffnesn + pj.elstiffnesn))*interpenetration)*normal_ij
            r_jic = (pj.radius-(pi.elstiffnesn/(pi.elstiffnesn + pj.elstiffnesn))*interpenetration)*normal_ji

            # position of the contact point
            p_ijc = pi.pred_position + r_ijc # ortsvektor/point of contact from p1
            p_jic = pj.pred_position + r_jic # point of contact from p2 ==p_ijc

            # velocity at the contact point
            v_ij = (np.cross(pj.rotation, r_jic) + pj.velocity) - (np.cross(pi.rotation, r_ijc) + pi.velocity)

            # decomposition in the local reference frame defined at the contact point
            v_ij_n = (np.dot(v_ij, normal_ij) * normal_ij)
            v_ij_t = v_ij - v_ij_n

            # tangential unit vector should be tangential_ij
            if np.linalg.norm(v_ij_t) != 0:
                t_ij = v_ij_t/np.linalg.norm(v_ij_t)
            else:
                t_ij = np.array([1,1,0])
                t_ij[0] = v_ij_n[1]
                t_ij[1] = v_ij_n[0]
                t_ij[2] = 0
                t_ij = t_ij/np.linalg.norm(t_ij)

            # tangential_ji brauche ich auch noch

            # for
            # f_ij = f_ij_n + f_ij_t
            # f_ij = f_n* normal_ij + f_t*t_ij


            # contact forces
            # forces = 0 if norm(ci,cj)>2r
            # does not work apparently
            #if norm_cij < pi.radius + pj.radius:
            pi.force = -interpenetration * pi.elstiffnesn * normal_ij
            pj.force = -interpenetration * pj.elstiffnesn * normal_ji
            #else:
            #pi.force = 0
            #pj.force = 0


            '''
            ######### LS+D ############
            f_ne = 0
            f_nd = 0
           
           
            m_eq = (p1.mass*p2.mass)/(p1.mass+p2.mass)          #equivalent mass of the contact 
           
            chi = 0.5                           # should be taken from literature
           
            elstiffnesn = 10
           
            damp_coeff = 2 * chi * np.sqrt(m_eq*elstiffnesn)               #elstiffnes from which Particle?
           
            f_ne = p1.elstiffnesn * interpenetration   #no cohesive kontakt --> equation only for interpenetration > 0
           
            rel_vel_normal = -np.dot((p2.velocity-p1.velocity), normal_ij)
           
            f_nd = damp_coeff * rel_vel_normal
           
            f_n = f_ne + f_nd
           
            print(f_ne)
            print(f_nd)
           
            '''
            new_vel_05 = pi.velocity + 0.5 * dt * pi.acceleration
            pi.position = np.around(pi.position + dt * new_vel_05, decimals=4)
            new_force = np.around(pi.force, decimals=4)
            pi.acceleration = np.around(new_force * (1/pi.mass), decimals=4)
            pi.velocity = np.around(new_vel_05 + 0.5 * dt * pi.acceleration, decimals=4)

            new_vel_05 = pj.velocity + 0.5 * dt * pj.acceleration
            pj.position = np.around(pj.position + dt * new_vel_05, decimals=4)
            new_force = np.around(pj.force, decimals=4)
            pj.acceleration = np.around(new_force / pj.mass , decimals=4) # n_particle.mass
            pj.velocity = np.around(new_vel_05 + 0.5 * dt * pj.acceleration, decimals=4)


            print(interpenetration_vel)
            print(t_ij)
            print(pi.position, pi.velocity, pi.acceleration, pi.force)
            print(pj.position, pj.velocity, pj.acceleration, pj.force)
            print('----')


    #'''
    #### drawing section
    same_colour = False      # set False for different colours of the particles
    if same_colour == True:
        screen.fill((100,100,200))
        for n_particle in particle.list_of_particles:
            draw.circle(screen, (255,0,0), (n_particle.position[0], n_particle.position[1]), n_particle.radius)
    else:
        screen.fill((100, 100, 200))
        for indexc, n_particle in enumerate(particle.list_of_particles):
            chosen_c = colour_list[indexc]  # choosing the colour
            chosen_c_rgb = get_rgb(chosen_c)    # turning colour name to rgb
            draw.circle(screen, chosen_c_rgb, (n_particle.position[0], n_particle.position[1]), n_particle.radius)

    # limit to 30 fps
    animationTimer.tick(30)
    
    display.update()
   ### end of drawing section
   #'''
######################################   timeloop  ##########################
print(p1.velocity)
print(p2.velocity)
pygame.quit() 
sys.exit()

