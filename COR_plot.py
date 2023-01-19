import matplotlib.pyplot as plt
import functions as fn
import numpy as np

all_cor = np.linspace(0, 1, 100)
all_damp_coeffs = []

for cor in all_cor:
    erg = fn.calculate_damp_coeff(cor)
    all_damp_coeffs.append(erg)

plt.title('computing damping coefficient from COR like thornton2012')
plt.plot(all_cor, all_damp_coeffs, color='tab:blue') #, label='test')
# plt.legend()
plt.grid()
plt.xlabel('coefficient of restitution')
plt.ylabel('damping coefficient')

file_name = "LSD-damping_coeff-COR-2"
plt.savefig("C:/Users/Jaist/Documents/GitHub/BA_DEM/plots_1301/" + file_name + ".png")

'''
acc = np.array([1, 1, 1])
acc_conv = np.ndarray.tolist(acc)

mass = 1.0 / 10
force = [index * mass for index, objekt in enumerate(acc_conv)]
print(force)
print(type(force))
newforce = np.array(force)
print(newforce)
print(type(newforce))

    new_vel_05 = particle.velocity + 0.5 * dt * particle.acceleration
    particle.position = particle.position + dt * new_vel_05
    new_force = particle.force
    print(type(new_force))
    # -- np.array() * float geht nicht, nur * int
    #new_force = np.ndarray.tolist(new_force)
    beschl = [index * (1 / particle.mass) for index, objekt in enumerate(new_force)]
    particle.acceleration = np.array(beschl)
    # --
    # particle.acceleration = new_force * (1 / particle.mass)
    particle.velocity = new_vel_05 + 0.5 * dt * particle.acceleration
'''
a = np.array([1,2,3])
b = -a
print(b)
print(type(b))