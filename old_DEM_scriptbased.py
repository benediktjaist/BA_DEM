# -*- coding: utf-8 -*-
# -- imports --
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as mcolors
# from classes import Particle
from classes import Boundary
import functions as fn
import test_cor as cor
import os


class Particle:
    all_particles = []

    def __init__(self, position, velocity, acceleration, force, rotation, rotation_vel, rotation_acc, torque, radius, elstiffnesn,
                 mass, pred_position, interpenetration_vel):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.force = force
        self.rotation = rotation
        self.rotation_vel = rotation_vel
        self.rotation_acc = rotation_acc
        self.torque = torque
        self.radius = radius
        self.elstiffnesn = elstiffnesn
        self.mass = mass
        self.pred_position = pred_position
        self.interpenetration_vel = interpenetration_vel
        self.moment_of_inertia = 0.5 * self.mass * self.radius * self.radius
        Particle.all_particles.append(self)



# -- initialising pygame
pg.init()
screen = pg.display.set_mode((1000, 800))   # creating display
pg.display.set_caption('simulation of two particles')
animationTimer = pg.time.Clock()

# -- simulation --
# Particle(position, velocity, acceleration, force, rotation_vel, rotation_acc, torque, radius, k_n,
#                  mass, pred_position, interpenetration_vel):
# pred_position [x,y] (initialisiert mit 0)):
#parallel_1 = Particle(position=np.array([450, 500, 0]), velocity=np.array([100, 0, 0]),
                     # acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                     # mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#parallel_2 = Particle(position=np.array([600, 500, 0]), velocity=np.array([0, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#schief_1 = Particle(position=np.array([500, 500, 0]), velocity=np.array([25, 25, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=200000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#schief_2 = Particle(position=np.array([580, 580, 0]), velocity=np.array([0, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=200000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
rotation_1 = Particle(position=np.array([500, 500, 0]), velocity=np.array([50, 50, 0]),
                      acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, elstiffnesn=2000,
                      mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
rotation_2 = Particle(position=np.array([500, 620, 0]), velocity=np.array([0, 0, 0]),
                      acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, elstiffnesn=2000,
                      mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#rotation_plot = Particle(position=np.array([500, 500, 0]), velocity=np.array([50, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 1.5]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#cortesting_1 = Particle(position=np.array([500, 500, 0]), velocity=np.array([25, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#cortesting_2 = Particle(position=np.array([620, 500, 0]), velocity=np.array([0, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#rotation_rigid1 = Particle(position=np.array([590, 350, 0]), velocity=np.array([0, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                      #mass=50, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
#rotation_rigid2 = Particle(position=np.array([590, 500, 0]), velocity=np.array([0, 0, 0]),
                      #acceleration=np.array([0, 0, 0]), force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]), rotation_vel=np.array([0, 0, 0]),
                      #rotation_acc=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), radius=50, k_n=2000,
                      #mass=500000, pred_position=np.array([300, 300, 0]), interpenetration_vel=np.array([0, 0, 0]))
# b1 = Boundary(np.array([50, 700, 0]), np.array([700, 700, 0]), np.array([0,0]))
# -- simulation parameters
coeff_of_restitution = 1
# print(type(zentral_2.force))
# damp_coeff = fn.calculate_damp_coeff(coeff_of_restitution)
damp_coeff = cor.RootByBisection(0.0, 16.0, 0.0001, 300, coeff_of_restitution)
mu = 0.7 # Reibkoeffizient
# k_t = 0.8*cortesting_1.k_n
crit_steps = []
for particle in Particle.all_particles:
    crit_steps.append(0.3 * 2 * np.sqrt(particle.mass / particle.k_n))
crit_dt = min(crit_steps)
print("required time step < ", crit_dt)
print("Dämpfung: ", damp_coeff)
dt = 0.01
simtime = 2  # number max steps for simulation
if dt > crit_dt:
    print()
    print("IMPORTANT WARNING\ndt < crit_dt is not satisfied\nchange dt to: ", np.round(crit_dt, 4))
    exit()

# -- plotting
en = []  # gesamtenergie
en_damp = []  # energie des dämpfers
en_dissipated = 0  # historische variable
en_dissipated_t = []
en_el = []  # energie der feder
en_i = []  # ekin
en_j = []  # ekin
en_rot_i = []  # erot
en_rot_j = [] # erot
t_points = []

for t in np.arange(0, simtime, dt):

    for n_particle in Particle.all_particles:
        # fn.predict_position(n_particle, dt)
        # Predict Position
        pred_vel05 = n_particle.velocity + 0.5 * dt * n_particle.acceleration
        pred_posi = n_particle.position + dt * pred_vel05
        n_particle.position = pred_posi
        n_particle.velocity = pred_vel05  # updating half step velocity to use it for the second velocity update

        pred_rot_vel05 = n_particle.rotation_vel + 0.5 * dt * n_particle.rotation_acc
        pred_rota = n_particle.rotation + dt * pred_rot_vel05
        n_particle.rotation = pred_rota
        n_particle.rotation_vel = pred_rot_vel05  # updating half step velocity for the second velocity update

        # Initialisierung?
        # pj.acceleration = np.array(pj.force) / pj.mass
        # pj.rotation_acc = np.array(pj.torque) / pj.moment_of_inertia


    for index, pi in enumerate(Particle.all_particles):
        if len(Particle.all_particles) == 1:
            # first vel update in step predict
            # pos update in predict position
            pi.acceleration = np.array(pi.force) * (1 / pi.mass)
            pi.rotation_acc = np.array(pi.torque) / pi.moment_of_inertia
            pi.velocity = pi.velocity + 0.5 * dt * pi.acceleration
            pi.rotation_vel = pi.rotation_vel + 0.5 * dt * pi.rotation_acc
            energy_i = 0.5 * pi.mass * np.linalg.norm(pi.velocity) ** 2
            energy_j = 0
            energy_el = 0
            energy_damp = 0
            energy = energy_i + energy_j + energy_el + energy_damp
            energy_rot_i = 0.5 * pi.moment_of_inertia * pi.rotation_vel[2] ** 2
            energy_rot_j = 0

            en.append(energy)
            t_points.append(t)

            en_i.append(energy_i)
            en_j.append(energy_j)
            en_el.append(energy_el)
            en_damp.append(energy_damp)
            en_dissipated += energy_damp
            en_dissipated_t.append(en_dissipated)
            en_rot_i.append(energy_rot_i)
            en_rot_j.append(energy_rot_j)


        for pj in Particle.all_particles[index + 1:]:

            cij = pi.position - pj.position
            norm_cij = np.sqrt(cij[0] ** 2 + cij[1] ** 2)  # =norm_cji

            normal_ij = (pj.position - pi.position) / norm_cij

            normal_ji = (pi.position - pj.position) / norm_cij

            # Quellenangabe wäre gut
            elstiffnesn_eq = (pi.k_n * pj.k_n) / (pi.k_n + pj.k_n)
            m_eq = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            radius_eq = (pi.radius*pj.radius) / (pi.radius+pj.radius)
            k_t = elstiffnesn_eq * 0.8

            # ------------------------------------------------------------------------------------------------------(22)
            if norm_cij < pi.radius + pj.radius:
                # interpenetration
                interpenetration = pi.radius + pj.radius - np.dot((pj.position - pi.position), normal_ij)
                interpenetration_vel = -(pj.velocity - pi.velocity) * normal_ij
                interpenetration_acc = -(pj.acceleration - pi.acceleration) * normal_ij
                v = 1 / (pi.radius + pj.radius) * (pi.velocity - pj.velocity) * (pi.radius - pj.radius)

                i_acc = np.linalg.norm(interpenetration_acc)

                # -- kinematik für t_ij im lokalen Koordinatensystem
                # hilfsvektoren von center of Particle zu point of contact
                r_ijc = (pi.radius - (
                        pj.k_n / (pi.k_n + pj.k_n)) * interpenetration) * normal_ij
                r_jic = (pj.radius - (
                        pi.k_n / (pi.k_n + pj.k_n)) * interpenetration) * normal_ji
                # position of the contact point
                p_ijc = pi.position + r_ijc  # ortsvektor/point of contact from p1
                p_jic = pj.position + r_jic  # point of contact from p2 ==p_ijc
                # velocity at the contact point
                v_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
                # decomposition in the local reference frame defined at the contact point
                v_ij_n = (np.dot(v_ij, normal_ij) * normal_ij)
                v_ij_t = v_ij - v_ij_n

                # tangential unit vector should be tangential_ij
                if np.linalg.norm(v_ij_t) >= 0.001:  # welche grenze?
                    t_ij = v_ij_t / np.linalg.norm(v_ij_t)
                else:
                    t_ij = 0

                # -- compute increment of tangential displacement (for updating F_t)
                # translation of point of contact

                # u_ij = (np.cross(pj.rotation_vel, r_jic) + pj.velocity) - (
                        # np.cross(pi.rotation_vel, r_ijc) + pi.velocity)
                # print("uij: ", u_ij) uij funktioniert nicht 31.01.

                # increment of tangential displacement
                # increment_of_t_displacement = np.linalg.norm(u_ij * t_ij) uij funktioniert nicht
                increment_of_t_displacement = np.linalg.norm(v_ij_t) * dt
                print("incr of t displ" , increment_of_t_displacement)

                # -- forces
                # f_t = min(mu*np.dot(pi.force, normal_ij), np.dot(pi.force, t_ij) + k_t * increment_of_t_displacement) # aus t
                # truth value of an array is ambigous --> np.linalg.norm
                max_friction_force = mu*np.dot(pi.force, normal_ij)
                tan_force = np.dot(pi.force, t_ij) + k_t * increment_of_t_displacement
                print("reibkraft: ", max_friction_force)
                print("andere t kraft: ", tan_force)
                if np.linalg.norm(max_friction_force) < np.linalg.norm(tan_force):
                    f_t = np.linalg.norm(max_friction_force) # stimmt die norm hier?
                else:
                    f_t = np.linalg.norm(tan_force)
                # + f_t aus drehmoment
                pi.force = np.array(-interpenetration * elstiffnesn_eq * normal_ij - interpenetration_vel * damp_coeff * normal_ij - f_t * t_ij)  # nicht normal_ij!! # k_t * f_t oder mu* np.array(pi.force) geht nicht
                pj.force = - pi.force
                print("ft: ", f_t, "r_ijc :", r_ijc)
                print("ft alternativ ", - np.dot(pi.force, t_ij))
                print("normalkraft: ", interpenetration * elstiffnesn_eq * normal_ij)
                print("tangentialkraft: ", f_t * t_ij)

                # -- torque
                moment = f_t * np.linalg.norm(r_ijc)
                pi.torque = np.array([0, 0, moment])
                pj.torque = pi.torque
                print("pi moment ", pi.torque)
                print("pj moment ", pj.torque)

            else:
                interpenetration = 0
                interpenetration_vel = 0
                interpenetration_acc = 0
                pi.force = [0, 0, 0]
                pj.force = [0, 0, 0]
                pi.torque = 0
                pj.torque = 0
            # ------------------------------------------------------------------------------------------------------(22)

            rel_vel = np.linalg.norm(pi.velocity - pj.velocity)
            omega = np.sqrt(elstiffnesn_eq / m_eq)  # wie bestimmt man systemsteifigkeit für k(pi) =/= k(pj)
            psi = damp_coeff / (2 * m_eq)  # =0 für lin. elastisch wird später mit COR bestimmt werden

            interpenetration_max = (rel_vel / omega) * np.exp(-(psi / omega) * np.arctan(omega / psi))


            # ------------------------------------------------------------------------------------------------------(23)
            # new_force = np.dot(np.dot(pi.force, normal_ij),
                               # normal_ij)  # nur die normale Komponente = new_force das ist falsch

            pi.acceleration = np.array([0, 10, 0]) + np.array(pi.force) * (1 / pi.mass)
            pi.rotation_acc = np.array(pi.torque) / pi.moment_of_inertia
            # ------------------------------------------------------------------------------------------------------(23)

            # ------------------------------------------------------------------------------------------------------(24)
            pi.velocity = pi.velocity + 0.5 * dt * pi.acceleration
            pi.rotation_vel = pi.rotation_vel + 0.5 * dt * pi.rotation_acc
            # ------------------------------------------------------------------------------------------------------(24)

            pj.acceleration = np.array(pj.force) * (1 / pj.mass)
            pj.rotation_acc = np.array(pj.torque) / pj.moment_of_inertia
            pj.velocity = pj.velocity + 0.5 * dt * pj.acceleration
            pj.rotation_vel = pj.rotation_vel + 0.5 * dt * pj.rotation_acc


            energy, energy_el, energy_i, energy_j, energy_damp = fn.calculate_energies(pi, pj, interpenetration, interpenetration_vel, damp_coeff, elstiffnesn_eq)
            energy_rot_i = 0.5 * pi.moment_of_inertia * pi.rotation_vel[2]**2
            energy_rot_j = 0.5 * pj.moment_of_inertia * pj.rotation_vel[2] ** 2
            print(pi.position, pi.velocity, pi.acceleration, pi.force)
            print(pj.position, pj.velocity, pj.acceleration, pj.force)
            #print("rot i: ", pi.rotation_vel, pi.rotation_acc, pi.torque)
            #print("rot j: ", pj.rotation_vel, pj.rotation_acc, pj.torque)
            #print("ROTATION: ", pi.rotation)
            print('--------------------------')

            en.append(energy)
            t_points.append(t)

            en_i.append(energy_i)
            en_j.append(energy_j)
            en_el.append(energy_el)
            en_damp.append(energy_damp)
            en_dissipated += energy_damp
            en_dissipated_t.append(en_dissipated)
            en_rot_i.append(energy_rot_i)
            en_rot_j.append(energy_rot_j)

            # I = 1/2 * m * r**2
            # E_rot = 1/2 * I * omega**2


    # contact particles - boundaries
    for n_particle in Particle.all_particles:
        for n_boundary in Boundary.all_boundaries:
            distance = fn.compute_normal_distance(n_boundary.point_1, n_boundary.point_2, n_particle.position)
            print(distance)
            point_of_contact = n_boundary.point_1 + np.dot((n_boundary.point_2-n_boundary.point_2),
                                                           (n_particle.position - n_boundary.point_1))
            #print(point_of_contact)
            n_boundary.point_of_contact = point_of_contact

            if n_particle.radius > distance:
                pass


# -- drawing section for pygame
    same_colour = False  # set False for different colours of the particles
    if same_colour == True:
        screen.fill((100, 100, 200))
        for n_particle in Particle.all_particles:
            pg.draw.circle(screen, (255, 0, 0), (n_particle.position[0], n_particle.position[1]), n_particle.radius)
    else:
        screen.fill((100, 100, 200))
        for indexc, n_particle in enumerate(Particle.all_particles):
            chosen_c = fn.colour_list[indexc]  # choosing the colour
            chosen_c_rgb = fn.get_rgb(chosen_c)  # turning colour name to rgb
            pg.draw.circle(screen, chosen_c_rgb, (n_particle.position[0], n_particle.position[1]), n_particle.radius)
            pg.draw.line(screen, (139, 0, 0), (n_particle.position[0], n_particle.position[1]),
                         (n_particle.position[0]+n_particle.radius*np.cos(n_particle.rotation[2]),
                          n_particle.position[1]+n_particle.radius*np.sin(n_particle.rotation[2])), 3)

            # pg.draw.line(screen, (255, 255, 255), (p_ijc[0], p_ijc[1]),
                         # (n_particle.position[0], n_particle.position[1]), width=5)


    for n_boundary in Boundary.all_boundaries:
        pg.draw.line(screen, (255, 255, 255), (n_boundary.point_1[0], n_boundary.point_1[1]),
                     (n_boundary.point_2[0], n_boundary.point_2[1]), width=5)
        pg.draw.circle(screen, (255, 255, 255), (n_boundary.point_1[0], n_boundary.point_1[1]), radius=5)
        pg.draw.circle(screen, (0, 0, 0), (n_boundary.point_2[0], n_boundary.point_2[1]), radius=5)
        if np.all(n_boundary.point_of_contact) == 0:
            pass
        else:
            print("particle hits at: ", n_boundary.point_of_contact)
    #draw.line(screen, get_rgb("Green"), (200, 200), (600, 600), width=5)
    # limit to 30 fps
    animationTimer.tick(30)

    pg.display.update()
pg.quit()


sum_ekin = []
for i in range(len(en_i)):
    sum_ekin.append(en_i[i]+en_j[i])


plt.title('physical behaviour')

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_figheight(10)
fig.set_figwidth(15)
ax1.plot(t_points, en, color='pink', label='total energy')
ax1.plot(t_points, en_i, color='blue', label='ekin_i')
ax1.plot(t_points, en_j, color='green', label='ekin_j')
ax1.plot(t_points, en_el, color='red', label='e_spring')
ax2.plot(t_points, en_rot_i, color=mcolors.CSS4_COLORS["deepskyblue"], label='e_rot_i')
#ax2.plot(t_points, en_rot_j, color=mcolors.CSS4_COLORS["lime"], label='e_rot_j')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)


at1 = AnchoredText(
    "en_bf: "+str(round(en[0])), prop=dict(size=10), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
at2 = AnchoredText(
    "en_af: "+str(round(en[-1])), prop=dict(size=10), frameon=True, loc='upper right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at2)

file_name = "test"
plt.savefig("C:/Users/Jaist/Documents/GitHub/BA_DEM/plots_3101/" + file_name + ".png")

print("Gesamtenergie vor Stoß: ", en[0])
print("Rotationsenergie vor Stoß: ", en_rot_i[0] + en_rot_j[0])
print("Gesamtenergie nach Stoß: ", en[-1])
print("Rotationsenergie nach Stoß: ", en_rot_i[-1] + en_rot_j[-1])
print("energiegewinn: ", en[-1]-en[0])
print("diff v_end und v_min i: ", en_i[-1]-min(en_i))
print("diff v_end und v_min j: ", en_j[-1]-min(en_j))



