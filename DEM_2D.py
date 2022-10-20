# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:30:46 2022

@author: Jaist
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from particle_class import particle
from boundary_class import boundary



'''
# creating an array with float type
#b = np.array([0,0], dtype='f')
#print(b[0])
print(p1.position[0])
print(type(p1.position[0]))
'''
#int=0
#float=0.

#position[x,y], velocity[dx,dy], acceleration[ddx,ddy], force[fx,fy], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
p1 = particle(np.array([3,0]) , np.array([1,0]),np.array([0,0]),np.array([0,0]),1,100,5,np.array([0,0]))
p2 = particle(np.array([4,0]) , np.array([0,0]),np.array([0,0]),np.array([0,0]),1,100,5,np.array([0,0]))


#initialization
dt = 0.3
simtime = 6 #time of simulation

#timeloop
for t in np.arange(0, simtime, dt):
   #loop of particle
   for n_particle in particle.list_of_particles:  
        
        #integration of motion with verlocity verlet (predict)
        pred_vel05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
       # print(pred_vel05)
        pred_posi = n_particle.position + dt*pred_vel05
       # print(pred_posi)
        n_particle.pred_posi = pred_posi
      #  print(pred_vel05) #testing with arrays
  # exit()
   print("pred pos1 for contact search:",p1.pred_posi)
   print("pred pos2 for contact search:",p2.pred_posi) 

   #contact detection with pred_posi
   cij=p1.pred_posi-p2.pred_posi
   norm_cij = np.sqrt(cij[0]**2+cij[1]**2)
   normal_ij = (p2.pred_posi-p1.pred_posi)/(norm_cij)


   if norm_cij < p1.radius + p2.radius:
       print("contact")
       interpenetration = p1.radius + p2.radius - np.dot((p2.pred_posi-p1.pred_posi),normal_ij)
       print("the interpenetration is:",interpenetration)
   else:
       print("no contact")
       interpenetration = 0
       
   exit()
       
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


