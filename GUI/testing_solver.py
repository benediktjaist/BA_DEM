import numpy as np

from Particle import Particle
from Boundary import Boundary
from dem_solver import PositionTracker, System
from video_creator import VideoCreator

position = np.array([500,300,0])
velocity = np.array([50,50,0])
radius = 50
elstiffnessn = 2000
mass = 100

position2 = np.array([700,500,0])
position2_obl = np.array([700,580,0])
velocity2 = np.array([0,0,0])

dt = 0.0001
simtime = 30

particle1 = [Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),]

particle2 = [Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),
            Particle(position=position2_obl, velocity=velocity2, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))
            ]

particle3 = [Particle(position=np.array([200,200,0]), velocity=np.array([-50,50,0]), acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),
            Particle(position=np.array([350,700,0]), velocity=np.array([0,-50,0]), acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),
Particle(position=np.array([700,200,0]), velocity=np.array([-50,-50,0]), acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))
            ]


boundary0 = []
boundary1 = [Boundary((100,100),(650,700))]
boundary4 = [Boundary((100,100),(800,100)),Boundary((100,100),(100,800)),
             Boundary((100,800),(800,800)), Boundary((800,800),(800,100))]

system = System(particles=particle3, boundaries=boundary4, dt=dt, simtime=simtime, mu=0.7,
                coeff_of_restitution=1)
system.run_simulation()

vid_creator = VideoCreator(particles=particle3, boundaries=boundary4, dt=dt, simtime=simtime)
vid_creator.animate()

for particle in particle1:
    print('anf ', particle.energy[0])
    print('end' , particle.energy[-1])