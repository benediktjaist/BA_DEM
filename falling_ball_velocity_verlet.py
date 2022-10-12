# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:30:46 2022

@author: Jaist
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


#parameter
g=9.81

#initial conditions
positions=[0] #array benutzen
velocities=[0]



iterations =6
dt =1
timeline=[0]


for i in range(iterations):   #range(start opt def 0,stop req, increment opt def1)
  #determine the new positions & velocities
  new_vel_05 = velocities[i] + 0.5*g*dt
  new_pos = positions[i] + new_vel_05*dt #+ 0.5*g*dt*dt
  #new_acc = ....
  new_vel = new_vel_05 + 0.5*g*dt
  
  #save the new positions & velocities
  positions.append(new_pos)
  velocities.append(new_vel)
  
  timeline.append(i)

print(positions)
print(timeline)

#plot results 
#fig = plt.figure()
#plt.xlim([-4, 4])
#plt.ylim([-1,1])
#camera = Camera(fig)
#for i in range(len(positions)):    
 #   plt.plot(timeline[i],positions[i], color="green", linewidth=1.0, linestyle="-")
 #   text = "T = " + str(round(dt*i,2))
 #   plt.text(5, 200, text, ha='left',wrap=False)
 #   camera.snap()

#animation = camera.animate()
#animation.save('falling_ball_3.gif', writer = 'pillow', fps=1)


fig, ax = plt.subplots()


line, = ax.plot(timeline, positions)


def animate(i):
    line.set_ydata(positions) # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)


ani.save('falling_ball_5.gif', writer = 'pillow', fps=20)

plt.show()








