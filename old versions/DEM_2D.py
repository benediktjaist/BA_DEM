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
'''
# initialising pygame
pygame.init()

# creating display
screen = pygame.display.set_mode((800, 800))
display.set_caption('this should be an animation')

# animation
animationTimer = time.Clock()
'''
#########################
'''
# labeling and drawing particles
font = pygame.font.SysFont(None, 100)
text = font.render("p"+str(n_particle), True, (255, 255, 0))

colourlist=[]
for n_particle in Particle.list_of_particles:
    colour = (np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255))
    colourlist.append(colour)
'''

# position[x,y], velocity[dx,dy], acceleration[ddx,ddy], force[fx,fy], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit aktueller Position)):
p1 = particle(np.array([200,400]), np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),100,100,5,np.array([200,400]))
p2 = particle(np.array([500,400]), np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),100,100,5,np.array([500,400]))
#p3 = Particle(np.array([700,700]), np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0]),100,100,5,np.array([700,700]))

liste = it.combinations(particle.list_of_particles, 2)

# initialization
dt = 1
simsteps = 3 #number max steps for simulation

###############################  timeloop  #################################
for t in np.arange(0, simsteps, dt):
    ############################  loop of Particle
    for n_particle in particle.list_of_particles:  
        # integration of motion with verlocity verlet (predict)
        pred_vel05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
        # print(pred_vel05)
        pred_posi = n_particle.position + dt*pred_vel05
        # print(pred_posi)
        n_particle.pred_position = pred_posi

        # contact detection with pred_posi
        for combinations in liste:                # combinations sind tupel
            # combinations[0] ist i
            # combinations[1] ist j
            
            #print(combinations[0].position)
            
           # exit()
            cij=combinations[0].pred_position-combinations[1].pred_position
            norm_cij = np.sqrt(cij[0]**2+cij[1]**2) # =norm_cji
            normal_ij = (combinations[1].pred_position-combinations[0].pred_position)/(norm_cij)
            normal_ji = (combinations[0].pred_position-combinations[1].pred_position)/(norm_cij)
    
            if norm_cij < combinations[0].radius + combinations[1].radius:
                print("contact")
                interpenetration = combinations[0].radius + combinations[0].radius - np.dot((combinations[1].pred_position-combinations[0].pred_position),normal_ij)
                print("interpenetration is:",interpenetration)   
                
            else:
                print("no contact")
                interpenetration = 0

                
            # hilfsvektoren von center of Particle zu point of contact
            r_ijc=(combinations[0].radius-(combinations[1].elstiffnesn/(combinations[0].elstiffnesn + combinations[1].elstiffnesn))*interpenetration)*normal_ij 
            r_jic=(combinations[1].radius-(combinations[0].elstiffnesn/(combinations[0].elstiffnesn + combinations[1].elstiffnesn))*interpenetration)*normal_ji
    
            # position of the contact point
            p_ijc = combinations[0].pred_position + r_ijc # ortsvektor/point of contact from p1
            p_jic = combinations[1].pred_position + r_jic # point of contact from p2 ==p_ijc
    
            # velocity at the contact point
            v_ij =(np.cross(combinations[1].rotation, r_jic) + combinations[1].velocity) - (np.cross(combinations[0].rotation, r_ijc) + combinations[0].velocity)
    
            # decomposition in the local reference frame defined at the contact point
            v_ij_n = (np.dot(v_ij, normal_ij)* normal_ij) #äußeres skalarprodukt: skalar * vektor --> fehler mit np.multiply()
            v_ij_t = v_ij - v_ij_n
        
            # tangential unit vector
           # t_ij = v_ij/np.linalg.norm(v_ij_t)
    
            #for
            #f_ij = f_ij_n + f_ij_t
            #f_ij = f_n* normal_ij + f_t*t_ij
           
            #contact forces
            combinations[0].force = interpenetration * combinations[0].elstiffnesn
            combinations[1].force = -interpenetration * combinations[1].elstiffnesn
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

        #die zuordnung von den partikeln i&j aus combinations muss bekannt sein, damit die partikel, die interagieren, auch geupdatet werden können
        # integration of motion with verlocity verlet (update)
        
        # for n_particle in Particle.list_of_particles:
            new_vel_05i = combinations[0].velocity + 0.5*dt*combinations[0].acceleration
            new_posi = combinations[0].position + dt*new_vel_05i
            new_forcei = combinations[0].force
            new_acci = new_forcei/combinations[0].mass  # n_particle.mass
            new_veli = new_vel_05i + 0.5*dt*new_acci
        
            combinations[0].position = new_posi
            combinations[0].velocity = new_veli
            combinations[0].acceleration = new_acci
        
            new_vel_05j = combinations[1].velocity + 0.5*dt*combinations[1].acceleration
            new_posj = combinations[1].position + dt*new_vel_05j
            new_forcej = combinations[1].force
            new_accj = new_forcei/combinations[1].mass  # n_particle.mass
            new_velj = new_vel_05j + 0.5*dt*new_accj
        
            combinations[1].position = new_posj
            combinations[1].velocity = new_velj
            combinations[1].acceleration = new_accj
        

        print("pos{}:".format(n_particle),n_particle.position, "vel{}:".format(n_particle),n_particle.velocity, "acc{}:".format(n_particle),n_particle.acceleration)
        print()
    print("-------------")
    '''
    #### drawing section
    screen.fill((100,100,200))
    draw.circle(screen, (255,0,0), n_particle.position, n_particle.radius)
        
    # limit to 30 fps
    animationTimer.tick(30)
        
    display.update()
    ### end of drawing section
######################################   timeloop  ##########################
pygame.quit() 
sys.exit()
'''
