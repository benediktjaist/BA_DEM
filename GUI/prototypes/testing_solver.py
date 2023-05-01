import numpy as np
from Particle import Particle
from Boundary import Boundary
from DEM_Solver_old import System
from Video_Creator import VideoCreator
from Plot_Creator import PlotCreator

position1 = np.array([600,600,0])
velocity1 = np.array([0,0,0])
radius1 = 50
elstiffnessn1 = 2000
mass1 = 100

position2 = np.array([400,400,0])
velocity2 = np.array([50,50,0])


teilchen = [Particle(position=position1, velocity=velocity1, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius1, elstiffnesn=elstiffnessn1, mass=mass1,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0])),

            Particle(position=position2, velocity=velocity2, acceleration=np.array([0, 0, 0]),
                     force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                     rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                     torque=np.array([0, 0, 0]), radius=radius1, elstiffnesn=elstiffnessn1, mass=mass1,
                     pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))
            ]

grenzen = []

gravitation = False

kor = 1
tinkr = 0.01
simzeit = 4
mue = 1

# video = VideoCreator(particles=teilchen, boundaries=grenzen, dt=tinkr, simtime=simzeit,
                     # video_name='test2')
# video.animate()

solver = System(particles=teilchen, boundaries=grenzen, dt=tinkr, simtime=simzeit, mu=mue,
                coeff_of_restitution=kor
                )
solver.run_simulation()

'''
for particle in teilchen:
    #print('ekin:', particle.energy_kin)
    #print('erot:', particle.energy_kin)
    #print('eel:', particle.energy_kin)
    #print('edamp:', particle.energy_kin)
    print(len(particle.energy_kin), len(particle.energy_el))
    # print(particle.energy_el)

for particle in teilchen:
    print('anf ', particle.energy_kin[0])
    print('end', particle.energy_kin[-1])

plotter = PlotCreator(particles=teilchen, dt=tinkr, simtime=simzeit)
plotter.plot_energy_kin()
'''