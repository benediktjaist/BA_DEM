# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:18:26 2022

@author: Jaist
"""

import numpy as np
import itertools as it
class particle_test:
    list_of_particles=[]
    
    def __init__(self, position, velocity,  rotation, radius, elstiffnesn, mass):
        self.position = position
        self.velocity = velocity
        self.rotation = rotation
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        
        particle_test.list_of_particles.append(self)

#position[x,y], velocity[dx,dy], acceleration[ddx,ddy], force[fx,fy], radius, elstiffnesn, mass, pred_posi[x,y](initialisiert mit 0)):
#p1 = Particle(np.array([3,0]) , np.array([1,0]),np.array([0,0]),np.array([0,0]),1,100,5,np.array([0,0]))
#p2 = Particle(np.array([4,0]) , np.array([0,0]),np.array([0,0]),np.array([0,0]),1,100,5,np.array([0,0]))

#posiiton, radius, stiffnes, rotation
p1 =particle_test(np.array([0,1]),np.array([1,0]),np.array([0,0]),1,10,5)    #i
p2 =particle_test(np.array([1,0]),np.array([0,0]),np.array([0,0]),1,10,5)   #j
p3 =particle_test(np.array([5,0]),np.array([0,0]),np.array([0,0]),1,10,5)  


#x = it.permutations(particle_test.list_of_particles,2) #keine permutationen, da die reihenfolge nicht relevant ist und über actio = reactio berücksichtigt werden soll

liste = it.combinations(particle_test.list_of_particles, 2)



for combinations in liste:              #combinations sind tupel
    positioni = combinations[0].position #combinations[0] ist i
    positionj = combinations[1].position #combinations[1] ist j
    print(positioni)
    print(positionj)
    print()
    
    cij=combinations[0].position-combinations[1].position
    norm_cij = np.sqrt(cij[0]**2+cij[1]**2) # =norm_cji
    normal_ij = (combinations[1].position-combinations[0].position)/(norm_cij)
    normal_ji = (combinations[0].position-combinations[1].position)/(norm_cij)

    if norm_cij < combinations[0].radius + combinations[1].radius:
        print("contact")
        interpenetration = combinations[0].radius + combinations[0].radius - np.dot((combinations[1].position-combinations[0].position),normal_ij)
        print("interpenetration is:",interpenetration)   
    else:
        print("no contact")
        interpenetration = 0

    #hilfsvektoren von center of Particle zu point of contact
    r_ijc=(combinations[0].radius-(combinations[1].elstiffnesn/(combinations[0].elstiffnesn + combinations[1].elstiffnesn))*interpenetration)*normal_ij 
    r_jic=(combinations[1].radius-(combinations[0].elstiffnesn/(combinations[0].elstiffnesn + combinations[1].elstiffnesn))*interpenetration)*normal_ji

    #position of the contact point
    p_ijc = combinations[0].position + r_ijc #ortsvektor/point of contact from p1
    p_jic = combinations[1].position + r_jic #point of contact from p2 ==p_ijc
    print('position of contact point is:', p_ijc)
    print()
    #print(p_jic)

    #velocity at the contact point
    v_ij =(np.cross(combinations[1].rotation, r_jic) + combinations[1].velocity) - (np.cross(combinations[0].rotation, r_ijc) + combinations[0].velocity)

    #decomposition in the local reference frame defined at the contact point
    v_ij_n = (np.dot(v_ij, normal_ij)* normal_ij) #äußeres skalarprodukt: skalar * vektor --> fehler mit np.multiply()
    v_ij_t = v_ij - v_ij_n

    print(v_ij_n)
    print(v_ij_t)

    #tangential unit vector
    t_ij = v_ij/np.linalg.norm(v_ij_t) #hier passt die länge nicht =/=1 im validierungsbeispiel

    print(t_ij)
    print('----------------------------')
   
    
#forces
#f_ij = f_ij_n + f_ij_t
#f_ij = f_n* normal_ij + f_t*t_ij

'''
x=np.linalg.norm(p1.position-p_ijc)
print(x) #x sollte radius-0.5*interpenetration = 0.9013878188659974 sein es kommt aber z raus
#y=p1.radius-0.5*interpenetration
#z=np.linalg.norm(p1.position)+0.5*interpenetration
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



















