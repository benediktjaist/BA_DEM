# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:59:28 2022

@author: Jaist
"""

# importing pygame module
import pygame
from pygame import *
 
# importing sys module
import sys
 
# initialising pygame
pygame.init()
 
# creating display
screen = pygame.display.set_mode((800, 800))
display.set_caption('this should be an animation')

#animation
animationTimer = time.Clock()
x = 50
y = 50

#endAnimation = False
# creating a running loop
#if endAnimation == True:
while True:
    
    # creating a loop to check events that are occurring
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            print("A key has been pressed")
            pygame.quit()
            sys.exit()
            
    #update position
    x += 1
    #draw objects
    screen.fill((100,100,200))
    #rectangle
    draw.rect(screen, (255,0,0), (10,10,100,100), 4)
    #filled rectangle
    draw.rect(screen, (255,0,0), (x,y,100,100) )
    draw.line(screen, (255,0,0), (0,10), (100,50))
    if x == 60:
        endAnimation = True
    
    # limit to 30 fps
    animationTimer.tick(30)
    
    display.update()