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

# initialising pygame
pygame.init()

# creating display
screen = pygame.display.set_mode((800, 800))
display.set_caption('this should be an animation')

# animation
animationTimer = time.Clock()
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
p1 = particle(np.array([200,200,0]), np.array([10,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]),50,10,5,np.array([0,0,0]))
p2 = particle(np.array([450,200,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]),50,10,5,np.array([0,0,0]))


#p3 = Particle(np.array([200,600]), np.array([5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))
#p4 = Particle(np.array([600,600]), np.array([-5,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),50,10,5,np.array([0,0]))


# initialization
dt = 0.1
simsteps = 20 #number max steps for simulation


###############################  timeloop  #################################
for t in np.arange(0, simsteps, dt):
    ############################  loop of Particle

    for n_particle in particle.list_of_particles:  
        # integration of motion with verlocity verlet (predict)
        pred_vel05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
        pred_posi = n_particle.position + dt*pred_vel05
        # update position
        n_particle.pred_position = pred_posi

    # contact detection with pred_posi
    for index, pi in enumerate(particle.list_of_particles):
        for pj in particle.list_of_particles[index + 1:]:
            cij = pi.pred_position-pj.pred_position
            norm_cij = np.sqrt(cij[0]**2+cij[1]**2)  # =norm_cji
            normal_ij = (pj.pred_position-pi.pred_position)/(norm_cij)
            normal_ji = (pi.pred_position-pj.pred_position)/(norm_cij)

            if norm_cij < pi.radius + pj.radius:
                interpenetration = pi.radius + pj.radius - np.dot((pj.pred_position-pi.pred_position),normal_ij)

            else:
                interpenetration = 0

            # hilfsvektoren von center of Particle zu point of contact
            r_ijc = (pi.radius-(pj.elstiffnesn/(pi.elstiffnesn + pj.elstiffnesn))*interpenetration)*normal_ij
            r_jic = (pj.radius-(pi.elstiffnesn/(pi.elstiffnesn + pj.elstiffnesn))*interpenetration)*normal_ji

            # position of the contact point
            p_ijc = pi.pred_position + r_ijc # ortsvektor/point of contact from p1
            p_jic = pj.pred_position + r_jic # point of contact from p2 ==p_ijc

            # velocity at the contact point
            v_ij = (np.cross(pj.rotation, r_jic) + pj.velocity) - (np.cross(pi.rotation, r_ijc) + pi.velocity)

            # decomposition in the local reference frame defined at the contact point
            v_ij_n = (np.dot(v_ij, normal_ij) * normal_ij)  # äußeres skalarprodukt: skalar * vektor --> fehler mit np.multiply()
            v_ij_t = v_ij - v_ij_n

            # tangential unit vector
            t_ij = v_ij/np.linalg.norm(v_ij_t)

            # for
            # f_ij = f_ij_n + f_ij_t
            # f_ij = f_n* normal_ij + f_t*t_ij


            # contact forces
            pi.force = interpenetration * pi.elstiffnesn
            pj.force = -interpenetration * pj.elstiffnesn


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
            pi.position = pi.position + dt * new_vel_05
            new_force = pi.force
            pi.acceleration = new_force / pi.mass
            pi.velocity = new_vel_05 + 0.5 * dt * pi.acceleration

            new_vel_05 = pj.velocity + 0.5 * dt * pj.acceleration
            pj.position = pj.position + dt * new_vel_05
            new_force = pj.force
            pj.acceleration = new_force / pj.mass  # n_particle.mass
            pj.velocity = new_vel_05 + 0.5 * dt * pj.acceleration



    #### drawing section
    screen.fill((100,100,200))
    for n_particle in particle.list_of_particles:
        draw.circle(screen, (255,0,0), (n_particle.position[0], n_particle.position[1]), n_particle.radius)

    show_score(600,600)


    # limit to 30 fps
    animationTimer.tick(30)
    
    display.update()
   ### end of drawing section
   
######################################   timeloop  ##########################
pygame.quit() 
sys.exit()

