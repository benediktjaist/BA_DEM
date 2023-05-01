import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

position1 = np.array([510,400,0])
velocity1 = np.array([0,0,0])
radius1 = 50
elstiffnessn1 = 2000
mass1 = 100

position2 = np.array([400,400,0])
velocity2 = np.array([50,0,0])


teilchen = [Particle(position=position1, velocity=velocity1, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]), poisson=0.3),

            Particle(position=position2, velocity=velocity2, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, mass=mass1,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]), poisson=0.3)
            ]

grenzen = []

gravitation = False

kor = 1
tinkr = 0.1
simzeit = 1.1
mue = 0.2


