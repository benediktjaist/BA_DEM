import numpy as np
from Particle import Particle
from Boundary import Boundary
from DEM_Solver import System
from Video_Creator import VideoCreator

position = np.array([600,200,0])
velocity = np.array([50,-50,0])
radius = 50
elstiffnessn = 2000
mass = 100

position2 = np.array([700,500,0])
position2_obl = np.array([700,580,0])
velocity2 = np.array([0,0,0])

dt = 0.001
simtime = 3

particle1 = [Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),]



boundary0 = []
boundary1 = [Boundary((700,100),(700,700))]
boundary2 = [Boundary((700,100),(700,800)), Boundary((100,100),(700,100))]

boundary4 = [Boundary((100,100),(800,100)),Boundary((100,100),(100,800)),
             Boundary((100,800),(800,800)), Boundary((800,800),(800,100))]

system = System(particles=particle1, boundaries=boundary2, dt=dt, simtime=simtime, mu=0.7,
                coeff_of_restitution=1)
system.run_simulation()

vid_creator = VideoCreator(particles=particle1, boundaries=boundary2, dt=dt, simtime=simtime)
vid_creator.animate()

for particle in particle1:
    print('anf ', particle.energy[0])
    print('end', particle.energy[-1])
