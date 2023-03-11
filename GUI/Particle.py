# -*- coding: utf-8 -*-
import numpy as np


class Particle:

    def __init__(self, position, velocity, acceleration, force, rotation, rotation_vel, rotation_acc, torque, radius, elstiffnesn,
                 mass, pred_position, interpenetration_vel, pp_force=np.array([0, 0, 0]), pp_torque=np.array([0, 0, 0]),
                 pb_force=np.array([0, 0, 0]), pb_torque=np.array([0, 0, 0])):
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
        self.historic_positions = []
        self.historic_rotations = []
        self.pp_force = pp_force
        self.pp_torque = pp_torque
        self.pb_force = pb_force
        self.pb_torque = pb_torque
        self.energy_kin = []
        self.energy_rot = []
        self.energy_pot = []
        self.energy_el = []
        self.energy_damp = []

    def __str__(self):
        return f"Position: {self.position}\nVelocity: {self.velocity}\nAcceleration: {self.acceleration}\n" \
               f"Force: {self.force}\nRotation: {self.rotation}\nRotation Velocity: {self.rotation_vel}\n" \
               f"Rotation Acceleration: {self.rotation_acc}\nTorque: {self.torque}\nRadius: {self.radius}\n" \
               f"Elastic Stiffness: {self.elstiffnesn}\nMass: {self.mass}\n" \
               f"Predicted Position: {self.pred_position}\nInterpenetration Velocity: {self.interpenetration_vel}"


teilchen = [[1,1,1], [2,2,2], [2,2,2]]
time = [1,2,3]

simtime = 10
dt = 1
for t in np.arange(0, simtime, dt):
    for i in range(0, 10):
        pass
    for i in range(0,10):
        pass



