# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:44:08 2022

@author: Jaist
"""

class boundary:
    list_of_boundaries=[]
    #point1 & point2 in x-z-plane as tupel (x,Z)
    def __init__(self, point1, point2, elstiffnesn, mass):
        self.point1 = point1
        self.point2 = point2
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        
        boundary.list_of_boundaries.append(self)
        
       
        # self.__class__. 