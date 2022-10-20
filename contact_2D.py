# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:18:26 2022

@author: Jaist
"""

import numpy as np

class particle_test:
   # list_of_particles=[]
    
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
    
   
        
       # particle.list_of_particles.append(self)

p1 =particle_test(np.array([0,3]),1)    #i
p2 =particle_test(np.array([1.5,2]),1)  #j


cij=p1.position-p2.position
norm_cij = np.sqrt(cij[0]**2+cij[1]**2)
normal_ij = (p2.position-p1.position)/(norm_cij)


if norm_cij < p1.radius + p2.radius:
   print("contact")
   interpenetration = p1.radius + p2.radius - np.dot((p2.position-p1.position),normal_ij)
   print("the interpenetration is:",interpenetration)
else:
   print("no contact")
   interpenetration = 0
