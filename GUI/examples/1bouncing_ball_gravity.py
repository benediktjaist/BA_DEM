import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

radius1 = 100
elstiffnessn1 = 2000
mass1 = 1000

position1 = np.array([400,100,0])
velocity1 = np.array([0,0,0])

teilchen = [Particle(position=position1, velocity=velocity1, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1, elstiffnesn=elstiffnessn1, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),


            ]

grenzen = [Boundary((100, 800), (800, 800))
           ]

gravitation = True

kor = 1
tinkr = 0.001
simzeit = 60
mue = 1