import matplotlib.pyplot as plt
import numpy as np
'''
ekin_0 = 125000
all_ekin_1 = {
    "dt=0.1": 188824.1848791771,
    "dt=0.08": 169820.3511203604,
    "dt=0.06": 156505.76636462117,
    "dt=0.04": 145592.65823320305,
    "dt=0.02": 134486.64811516166,
    "dt=0.01": 129571.97366191601,
    "dt=0.005": 127241.91724739953,
    "dt=0.001": 125440.90767829573,
    "dt=0.0001": 125043.9223990139,
    "dt=0.00001": 125004.39067905648,
    "dt=0.000001": 125000.43905301516,
    }
ekin_1 = {
    "dt=0.1": 188824.1848791771,
    "dt=0.01": 129571.97366191601,
    "dt=0.001": 125440.90767829573,
    "dt=0.0001": 125043.9223990139,
    "dt=0.00001": 125004.39067905648,
    "dt=0.000001": 125000.43905301516,
    }

rel_errors = []
for value in ekin_1.values():
    rel_err = np.linalg.norm((ekin_0 - value))/ekin_0
    rel_errors.append(rel_err)

t_incrs = []

for key in ekin_1.keys():
    dt = float(key[3:])
    t_incrs.append(dt)
print(t_incrs)
print(rel_errors)



fig, ax = plt.subplots()

ax.loglog(t_incrs, rel_errors, color=(0 / 255, 101 / 255, 189 / 255), zorder=0)
ax.invert_xaxis()
ax.scatter(t_incrs, rel_errors, marker='x', color=(227 / 255, 114 / 255, 34 / 255), zorder=5)
#ax.plot(t_incrs, rel_errors, color=(0 / 255, 101 / 255, 189 / 255))
#ax.plot(self.plot_time_steps, self.plot_interpenetrations, color=(227 / 255, 114 / 255, 34 / 255), label='maximum predicted interpenetration')


# Set the axis labels and title
ax.set_xlabel('Time Step')
ax.set_ylabel('Relative Error')

#plt.savefig('C:/Users/Jaist/Desktop/plots/'+'dt error'+'.svg')
plt.savefig('C:/Users/Jaist/Desktop/plots/' +'dt error'+ '.pdf')

plt.show()
'''
## calculation time for different numbers of particles in bounding box

number_of_particles = [5, 10, 15, 20, 30, 40]
simulation_time = [5.94, 14.52, 28.59, 46.08, 121.01, 211.76]

fig, ax = plt.subplots()
ax.plot(number_of_particles, simulation_time, color=(0 / 255, 101 / 255, 189 / 255), zorder=0)

#ax.scatter(t_incrs, rel_errors, marker='x', color=(227 / 255, 114 / 255, 34 / 255), zorder=5)
#ax.plot(t_incrs, rel_errors, color=(0 / 255, 101 / 255, 189 / 255))
#ax.plot(self.plot_time_steps, self.plot_interpenetrations, color=(227 / 255, 114 / 255, 34 / 255), label='maximum predicted interpenetration')


# Set the axis labels and title
ax.set_xlabel('Number of Particles in the Simulation')
ax.set_ylabel('Required Computation Time [s] for 10s Realtime')

#plt.savefig('C:/Users/Jaist/Desktop/plots/'+'dt error'+'.svg')
plt.savefig('C:/Users/Jaist/Desktop/plots/' +'computational_time_p_in_box'+ '.pdf')

plt.show()
