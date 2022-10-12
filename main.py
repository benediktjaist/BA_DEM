# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:53:47 2022

@author: Jaist
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle

p1 = particle(0,1,10,4)
p2 = particle(5,6,10,5)

#initialization
dt = 0.5
simtime = 10 #time of simulation
np=1
velocity = 1
acc=1
force=0
list_of_particles=[]
list_of_particles.append(p1)
list_of_particles.append(p2)

#timeloop
for t in range(simtime):
    #loop of particle
    for p in range (1, np):
        #contact detection
        if abs(p1.position-p2.position) < p1.radius + p2.radius:
            print("contact")
            interpenetration = p1.radius + p2.radius - (p2.position-p1.position)
            print("the interpenetration is: ",interpenetration)
        else:
            print("no contact")
            interpenetration = 0
        #contact forces
        f_n_e = interpenetration * p1.elstiffnesn
        
        #integration of motion with verlocity verlet
        new_vel_05 = velocity + 0.5*dt*acc
        new_pos = p1.position + dt*new_vel_05
        new_force = force
        new_acc = new_force/p1.mass
        new_vel = new_vel_05 + 0.5*dt*new_acc