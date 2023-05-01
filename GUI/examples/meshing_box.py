import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

position1 = np.array([400,400,0])
velocity1 = np.array([50,0,0])
radius1 = 50
elstiffnessn1 = 2000
mass1 = 100

position2 = np.array([550,400,0])
velocity2 = np.array([0,0,0])


particles_to_import = []

boundaries_to_import = [Boundary((1228, 50), (50, 50)),
                            Boundary((50, 50), (50, 782)),
                            Boundary((50, 782), (1228, 782)),
                            Boundary((1228, 782), (1228, 50))]

gravitation_to_import = False

cor_to_import = 1

dt_to_import = 0.01

simtime_to_import = 2

mu_to_import = 1

'''[Particle(position=position1, velocity=velocity1, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, poisson=0.3, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),

            Particle(position=position2, velocity=velocity2, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, poisson=0.3, mass=mass1,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))
            ]'''