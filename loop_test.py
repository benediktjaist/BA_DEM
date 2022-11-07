# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:42:54 2022

@author: Jaist
"""
import itertools as it

list =[1,2,3]

for i in range(0,len(list)):                
    pi = list[i]
    for j in range(i+1,len(list)):
        pj =list[i+1]
        print(pi,pj)
        
print("----------------------------")
liste = it.combinations(list, 2)

for i in liste:
    print(i)
        