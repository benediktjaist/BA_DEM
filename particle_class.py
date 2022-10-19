# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:40:25 2022

@author: Jaist
"""

class particle:
    list_of_particles=[]
    
    def __init__(self, position, velocity, acceleration, force, radius, elstiffnesn, mass, pred_posi):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.force = force
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        self.pred_posi = pred_posi
        
        particle.list_of_particles.append(self)
        
  #  def get_position(self):
    #    return self.position
    
   # def set_position(self, position):
    #    if position < 0:
     #       print("try a position > 0")
     #   else:
     #       self.position = position
  #  position = property(get_position, set_position)
       
        # self.__class__. 
        
#p = particle(0, 0, 0, 0, 0, 0, 5)

#print(p.position)

