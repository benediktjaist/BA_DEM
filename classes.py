# -*- coding: utf-8 -*-
import numpy as np


class Particle:
    all_particles = []

    def __init__(self, position, velocity, acceleration, rotation_vel, force, radius, elstiffnesn, mass, pred_position,
                 interpenetration_vel):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.rotation_vel = rotation_vel
        self.force = force
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        self.pred_position = pred_position
        self.interpenetration_vel = interpenetration_vel

        Particle.all_particles.append(self)


class Boundary:
    all_boundaries = []

    def __init__(self, point_1, point_2, point_of_contact):
        self.point_1 = point_1
        self.point_2 = point_2
        self.point_of_contact = point_of_contact

        Boundary.all_boundaries.append(self)


#  def get_position(self):
#    return self.position

# def set_position(self, position):
#    if position < 0:
#       print("try a position > 0")
#   else:
#       self.position = position
#  position = property(get_position, set_position)

# self.__class__.

# p = Particle(0, 0, 0, 0, 0, 0, 5)

# print(p.position)


class system:
    def __init__(self, listOfParticles):
        self.list_of_particles = listOfParticles
        '''
        #timeloop
        for t in np.arange(0, simtime, dt):
           #loop of Particle
           for n_particle in Particle.list_of_particles:  

                #integration of motion with verlocity verlet (predict)
                pred_vel05 = n_particle.velocity + 0.5*dt*n_particle.acceleration
               # print(pred_vel05)
                pred_posi = n_particle.position + dt*pred_vel05
               # print(pred_posi)
                n_particle.pred_posi = pred_posi
              #  print(pred_vel05) #testing with arrays
              '''

