# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:53:47 2022

@author: Jaist
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle
from boundary_class import boundary

#position, velocity, acceleration, force, radius, elstiffnesn, mass):
p1 = particle(0,1,0,4,1,10,4)
p2 = particle(3,0,0,4,1,10,5)

#boundaries --> define a vector with p1 & p2 
b1 = boundary((0,0), (0,10), 1, 1000)
b2 = boundary((0,3), (0,5), 2, 4)

#initialization
dt = 1
simtime = 2 #time of simulation

#timeloop
for t in range(0,simtime,dt):
    #loop of particle
   for n_particle in particle.list_of_particles:  # for p in range (1,len(particle.list_of_particles)):
        
        #integration of motion with verlocity verlet (predict)
        
        
        
        #contact detection
        if abs(p1.position-p2.position) < p1.radius + p2.radius:
            print("contact")
            interpenetration = p1.radius + p2.radius - (p2.position-p1.position)
            print("the interpenetration is:",interpenetration)
        else:
            print("no contact")
            interpenetration = 0
        #contact forces
        for n_particle in particle.list_of_particles:
            n_particle.force = interpenetration * n_particle.elstiffnesn
            print(n_particle.force)
        
        for n_particle in particle.list_of_particles:
        #integration of motion with verlocity verlet (update)
            new_vel_05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
            new_pos = n_particle.position + dt*new_vel_05
            new_force = n_particle.force
            new_acc = new_force/n_particle.mass  #n_particle.mass
            new_vel = new_vel_05 + 0.5*dt*new_acc
            print(new_vel)
      
        
        
        