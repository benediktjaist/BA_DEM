import numpy as np
from GUI.Particle import Particle
from GUI.Boundary import Boundary

radius = [50, 30,50,30]


elstiffnessn1 = 2000
mass1 = 100
positionx = [i for i in np.linspace(200,700,4)]
positiony = [i for i in np.linspace(200,700,4)]

velocityx = [i for i in np.linspace(-90,90,4)]
velocityy =[i for i in np.linspace(-90,90,4)]

particles_to_import = []
for i in range(4):
    for j in range(4):
        particles_to_import.append(Particle(position=np.array([positionx[i], positiony[j],0]), velocity=np.array([velocityx[i], velocityy[j],0]), acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius[i], k_n=elstiffnessn1, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]), poisson=0.3))

boundaries_to_import = [Boundary((800, 100), (100, 100)),
           Boundary((100, 100), (100, 800)),
           Boundary((100, 800), (800, 800)),
           Boundary((800, 800), (800, 100))]

gravitation_to_import = True

cor_to_import = 1
dt_to_import = 0.001
simtime_to_import = 5
mu_to_import = 0.1
