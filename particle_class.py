# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:40:25 2022

@author: Jaist
"""

class particle:
    list_of_particles=[]
    
    
    def __init__(self, position, radius, elstiffnesn, mass):
        self.position = position
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        
        particle.list_of_particles.append(self)
        
       
        # self.__class__. 