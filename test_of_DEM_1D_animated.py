# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:36:53 2022

@author: Jaist
"""
# importing pygame module
import pygame
from pygame import *
import sys #wofür?
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle
from boundary_class import boundary
 
# initialising pygame
pygame.init()
 
#initialisation of DEM+Animation
t=0
dt = 0.1
simsteps = 20 #number max steps for simulation
screen = pygame.display.set_mode((800, 800))
display.set_caption('this should be an animation')
animationTimer = time.Clock()
# creating a running loop for animation

#while t <= simtime:
    #creating a loop to check events that are occurring
    #for event in pygame.event.get():
       #if event.type == pygame.QUIT:
           # pygame.quit()
           # sys.exit()
            
# update positions    
#generating particles
# position, velocity, acceleration,rotation, force, radius, elstiffnesn, mass, pred_posi(initialisiert mit 0)):
p1 = particle(200 , 5,0,0,0,100,5,5,0)
p2 = particle(500 , 0,0,0,0,100,5,5,0) 
    
#timeloop for DEM
for t in np.arange(0, simsteps, dt):
    
    #loop of particle
    
    
    for n_particle in particle.list_of_particles:  
            
        #integration of motion with verlocity verlet (predict)
        pred_vel05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
        # print(pred_vel05)
        pred_posi = n_particle.position + dt*pred_vel05
        # print(pred_posi)
        n_particle.pred_posi = pred_posi
            
    print("pred pos1 for contact search:",p1.pred_posi)
    print("pred pos2 for contact search:",p2.pred_posi) 
    
    #contact detection with pred_posi
    if abs(p1.pred_posi-p2.pred_posi) < p1.radius + p2.radius:
        print("contact")
        interpenetration = p1.radius + p2.radius - (p2.pred_posi-p1.pred_posi)
        print("the interpenetration is:",interpenetration)
    else:
        print("no contact")
        interpenetration = 0
        

    #contact forces
    for n_particle in particle.list_of_particles:
        n_particle.force = interpenetration * n_particle.elstiffnesn
           
   #print(n_particle.force, "force was updated")
           
    for n_particle in particle.list_of_particles:
        #integration of motion with verlocity verlet (update)
        new_vel_05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
        new_pos = n_particle.position + dt*new_vel_05
        new_force = n_particle.force
        new_acc = new_force/n_particle.mass  #n_particle.mass
        new_vel = new_vel_05 + 0.5*dt*new_acc
           
        n_particle.position = new_pos
        n_particle.velocity = new_vel
        n_particle.acceleration = new_acc
           
    print("pos1:",p1.position, "vel1:",p1.velocity, "acc1:",p1.acceleration)
    print("pos2:",p2.position, "vel2:",p2.velocity, "acc2:",p2.acceleration)
    
    #draw objects
    screen.fill((100,100,200))
       
    #draw.line(screen, (255,0,0), (0,10), (100,50))
    #draw particles
    draw.circle(screen, (255,0,0), (p1.position,500), p1.radius)
    draw.circle(screen, (255,0,0), (p2.position,500), p2.radius)
   
    # limit to 30 fps
    animationTimer.tick(30)
   
    display.update()
        
        
pygame.quit() 
sys.exit()

''' die kinematik der kontaktkräfte wurde noch nicht ordentlich berücksichtigt
    deshalb beschleuniggen die particles
    if event.type == pygame.KEYDOWN:
        pygame.quit() 
        sys.exit()
    
    ''' 
    
      
        

