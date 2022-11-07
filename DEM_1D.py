# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:30:46 2022

@author: Jaist
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle, system
from boundary_class import boundary


#position, velocity, acceleration, force, radius, elstiffnesn, mass, pred_posi(initialisiert mit 0)):
p1 = particle(0 , 1,0,0,1,100,5,0,0)
p2 = particle(4 , 0,0,0,1,100,5,0,0)

list_of_particles = [p1,p2]

#s = sytem(list_of_particles)

#initialization
dt = 1
simtime = 6 #time of simulation

#timeloop
for t in np.arange(0, simtime, dt):
   #loop of Particle
   for n_particle in particle.list_of_particles:  #p1.particles
        
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
   print()









