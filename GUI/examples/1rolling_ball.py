import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

radius1 = 50
elstiffnessn1 = 2000
mass1 = 100

position = np.array([100, 449, 0])
velocity = np.array([0, 0, 0])


particles_to_import = [Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0.5]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, poisson=0.3, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))
            ]

boundaries_to_import = [Boundary((50,500), (600,500))]

gravitation_to_import = True

cor_to_import = 1

dt_to_import = 0.01
simtime_to_import = 10

mu_to_import = 1




