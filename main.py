# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:53:47 2022

@author: Jaist
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle

#position, velocity, acceleration, force, radius, elstiffnesn, mass):
p1 = particle(0,0,0,0,1,10,4)
p2 = particle(5,0,0,0,6,10,5)

#initialization
dt = 0.5
simtime = 10 #time of simulation
nofp=2
velocity = 1
acc=1
force=1
#list_of_particles=[]
#list_of_particles.append(p1)
#list_of_particles.append(p2)


print(particle.list_of_particles)

for n_particle in particle.list_of_particles:
    print(n_particle.position)

#timeloop
for t in range(simtime):
    #loop of particle
    for p in range (1, nofp):
        
        #integration of motion with verlocity verlet #predict
        
        
        
        
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
        
        
        for n_particle in particle.list_of_particles:
        #integration of motion with verlocity verlet #update
            new_vel_05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
            new_pos = n_particle.position + dt*new_vel_05
            new_force = n_particle.force
            new_acc = new_force/n_particle.mass  #n_particle.mass
            new_vel = new_vel_05 + 0.5*dt*new_acc
            print(new_vel)
      
        
        
        