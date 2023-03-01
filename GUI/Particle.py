# -*- coding: utf-8 -*-
import numpy as np


class Particle:
    # all_particles = []

    def __init__(self, position, velocity, acceleration, force, rotation, rotation_vel, rotation_acc, torque, radius, elstiffnesn,
                 mass, pred_position, interpenetration_vel):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.force = force
        self.rotation = rotation
        self.rotation_vel = rotation_vel
        self.rotation_acc = rotation_acc
        self.torque = torque
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        self.pred_position = pred_position
        self.interpenetration_vel = interpenetration_vel
        self.moment_of_inertia = 0.5 * self.mass * self.radius * self.radius
        # Particle.all_particles.append(self)
