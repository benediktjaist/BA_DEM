import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

radius1 = 50
elstiffnessn1 = 2000
mass1 = 100

position1 = np.array([425,315,0])
velocity1 = np.array([0,50,0])

position2 = np.array([370,400,0])
velocity2 = np.array([0,0,0])

position3 = np.array([480,400,0])
velocity3 = np.array([0,0,0])

particles_to_import = [Particle(position=position1, velocity=velocity1, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, poisson=0.3, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),

            Particle(position=position2, velocity=velocity2, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, poisson=0.3, mass=mass1,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),

Particle(position=position3, velocity=velocity3, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, poisson=0.3, mass=mass1,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))

            ]

boundaries_to_import = []

gravitation_to_import = False

cor_to_import = 1
dt_to_import = 0.001
simtime_to_import = 4
mu_to_import = 1