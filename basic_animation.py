# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:37:38 2022

@author: Jaist
"""

from pygame import*

init()
# screen
screen = display.set_mode((800,800))  #tupel (width heigth)
display.set_caption('basic graphics')

runProgramm = True
while runProgramm == True:
    #pygame event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            endProgramm = False
    #draw objects
    screen.fill((100,100,200))
    #rectangle
    draw.rect(screen, (255,0,0), (10,10,100,100), 4)
    #fill rectangle
    draw.rect(screen, (255,0,0), (200,10,100,100) )
    display.update()

