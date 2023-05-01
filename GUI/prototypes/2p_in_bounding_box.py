import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

radius1 = 50
elstiffnessn1 = 2000
mass1 = 100

position1 = np.array([200,220,0])
velocity1 = np.array([0,-50,0])

position2 = np.array([350,700,0])
velocity2 = np.array([0,50,0])


teilchen = [Particle(position=position1, velocity=velocity1, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1,k_n=elstiffnessn1, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),

            Particle(position=position2, velocity=velocity2, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius1, k_n=elstiffnessn1, mass=mass1,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))#, pp_force=np.array([0, 0, 0]), pp_torque=np.array([0, 0, 0]),
                 #pb_force={'moiiiin':np.array([0,0,0])}, pb_torque=np.array([0, 0, 0]))
            ]

grenzen = [Boundary((800, 100), (100, 100)),
           Boundary((100, 100), (100, 800)),
           Boundary((100, 800), (800, 800)),
           Boundary((800, 800), (800, 100))]

gravitation = False

kor = 1
tinkr = 0.001
simzeit = 4
mue = 1
