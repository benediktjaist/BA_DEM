# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:30:46 2022

@author: Jaist
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class system:
    def __init__(self, number_of_particles=2): #max number of particles equals 2
        self.number_of_particles = number_of_particles
        #save particle positions
        #save forces
    def create_particle
    

class particle:
    def __init__(self, position, radius, elstiffnesn, mass):
        self.position = position
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
    
        
        
        
p1 = particle(0,1,10,4)
p2 = particle(5,6,10,5)

print(p1.position)
print(p2.position)


    
#contact detection
if abs(p1.position-p2.position) < p1.radius + p2.radius:
    print("contact")
    interpenetration = p1.radius + p2.radius - (p2.position-p1.position)
    print("the interpenetration is: ",interpenetration)
else:
    print("no contact")
    
    
#contact forces
f_n_e = interpenetration * p1.elstiffnesn

print(f_n_e)
    
quit()


#parameter
g=9.81

#initial conditions
positions=[0] #array benutzen
velocities=[0]
acc=[9,81]



iterations =6
dt =1
timeline=[0]


for i in range(iterations):   #range(start opt def 0,stop req, increment opt def1)
  #determine the new positions & velocities
  
  new_vel_05 = velocities[i] + 0.5*dt*acc
  new_pos = positions[i] + dt*new_vel_05
  new_force = force
  new_acc = new_force/m
  new_vel = new_vel_05 + 0.5*dt*new_acc
  
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








