import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

position = np.array([111,111,0])
position2 = np.array([222,222,0])
velocity = np.array([50,-50,0])
radius = 50
elstiffnessn = 2000
mass = 100

position3 = np.array([700,500,0])
position2_obl = np.array([700,580,0])
velocity2 = np.array([0,0,0])

kugeln = [Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),
             Particle(position=position2, velocity=velocity, acceleration=np.array([0, 0, 0]),
                      force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                      rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                      torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                      pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))
             ]
p1 = Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))


