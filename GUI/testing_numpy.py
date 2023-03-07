import numpy as np
import sympy as smp

x = smp.symbols('x')
eq = 3*x+5

b = smp.solveset(eq, x, domain=smp.Reals)
c = np.array(smp.solve(eq, x), dtype='float64')
d = np.array([1.0000])
print(type(d[0]))
print(type(b))
print(b)
print(type(c[0]))
print(c)

x = np.array([1,2,3])
y = np.array([1,0,0])
print(x+y)

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
