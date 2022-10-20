# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:18:26 2022

@author: Jaist
"""

import numpy as np

class particle_test:
   # list_of_particles=[]
    
    def __init__(self, position, velocity,  rotation, radius, elstiffnesn):
        self.position = position
        self.velocity = velocity
        self.rotation = rotation
        self.radius = radius
        self.elstiffnesn = elstiffnesn
    
   
        
       # particle.list_of_particles.append(self)

#position[x,y], velocity[dx,dy], acceleration[ddx,ddy], force[fx,fy], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
#p1 = particle(np.array([3,0]) , np.array([1,0]),np.array([0,0]),np.array([0,0]),1,100,5,np.array([0,0]))
#p2 = particle(np.array([4,0]) , np.array([0,0]),np.array([0,0]),np.array([0,0]),1,100,5,np.array([0,0]))

#posiiton, radius, stiffnes, rotation
p1 =particle_test(np.array([0,1]),np.array([0,1]),np.array([0,1]),1,10)    #i
p2 =particle_test(np.array([1,0]),np.array([0,1]),np.array([0,1]),1,20)  #j


cij=p1.position-p2.position
norm_cij = np.sqrt(cij[0]**2+cij[1]**2) # =norm_cji
normal_ij = (p2.position-p1.position)/(norm_cij)
normal_ji = (p1.position-p2.position)/(norm_cij)


if norm_cij < p1.radius + p2.radius:
   print("contact")
   interpenetration = p1.radius + p2.radius - np.dot((p2.position-p1.position),normal_ij)
   print("the interpenetration is:",interpenetration)
   
  
else:
   print("no contact")
   interpenetration = 0

#hilfsvektoren von center of particle zu point of contact
r_ijc=(p1.radius-(p2.elstiffnesn/(p1.elstiffnesn+p2.elstiffnesn))*interpenetration)*normal_ij 
r_jic=(p2.radius-(p1.elstiffnesn/(p1.elstiffnesn+p2.elstiffnesn))*interpenetration)*normal_ji
print(r_ijc)
print(r_jic)

#position of the contact point
p_ijc = p1.position + r_ijc #ortsvektor/point of contact from p1
p_jic = p2.position + r_jic #point of contact from p2 ==p_ijc
print(p_ijc)
#print(p_jic)

#velocity at the contact point
v_ij =(np.cross(p2.rotation, r_jic) + p2.velocity) - (np.cross(p1.rotation, r_ijc) + p1.velocity)

#decomposition in the local reference frame defined at the contact point
print((np.dot(v_ij, normal_ij)))
v_ij_n = np.dot((np.dot(v_ij, normal_ij), normal_ij)) #äußeres skalarprodukt: skalar * vektor --> fehler
v_ij_t = v_ij - v_ij_n


#tangential unit vector
t_ij = v_ij/np.linalg.norm(v_ij_t)

#forces
#f_ij = f_ij_n + f_ij_t
f_ij = f_n* normal_ij + f_t*t_ij

'''
x=np.linalg.norm(p1.position-p_ijc)
print(x) #x sollte radius-0.5*interpenetration = 0.9013878188659974 sein es kommt aber z raus
#y=p1.radius-0.5*interpenetration
#z=np.linalg.norm(p1.position)+0.5*interpenetration
'''