import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from Colours import get_colour_tuple as col

class PlotCreator:
    def __init__(self, particles, dt, simtime, directory, plot_name): # plot_name, plot_dir
        self.particles = particles
        self.dt = dt
        self.simtime = simtime
        self.plot_name = plot_name
        self.plot_dir = directory #"C:/Users/Jaist/Documents/GitHub/BA_DEM/GUI/examples/plots"
        self.time_array = np.arange(0, self.simtime, self.dt)
        self.total_energy = []



    def plot(self):
        # data
        if len(self.particles) == 1:
            for step in range(0, len(self.time_array)):
                total_en = 0
                for particle in self.particles:
                    total_en += particle.energy_kin[step] + particle.energy_rot[step] + particle.energy_el[step] + \
                                particle.energy_damp[step] #+ particle.energy_pot[step]  # + particle.energy_tan[step]
                self.total_energy.append(total_en)

            mpl.use("Qt5Agg")

            ## plotting energies

            # fig, (ax1, ax2, ax3) = plt.subplots(3)
            fig, ax1 = plt.subplots(1)
            # fig.tight_layout()

            ax1.plot(self.time_array, self.total_energy, color=(0 / 255, 0 / 255, 0 / 255), label='total energy')
            # ax1.plot(self.time_array, self.damping_force, color=(0 / 255, 0 / 255, 0 / 255), label='Damping Force')
            # ax1.plot(self.time_array, self.elastic_force, color=(0 / 255, 101 / 255, 189 / 255), label='Elastic Force')
            ax1.plot(self.time_array, self.particles[0].energy_kin, color=(0 / 255, 101 / 255, 189 / 255),
                     label='pi e_kin')
            #ax1.plot(self.time_array, self.particles[1].energy_kin, color=(227 / 255, 114 / 255, 34 / 255),
                     #label='pj e_kin')

            ax1.plot(self.time_array, self.particles[0].energy_rot, color=(196 / 255, 101 / 255, 27 / 255), label='pi e_rot')
            # ax1.plot(self.time_array, self.particles[1].energy_rot, color=(255 / 255, 220 / 255, 0 / 255), label='pj e_rot')

            ax1.plot(self.time_array, self.particles[0].energy_el, color=(162 / 255, 173 / 255, 0 / 255),
                     label='pi e_el')
            #ax1.plot(self.time_array, self.particles[1].energy_el, color=(88 / 255, 88 / 255, 90 / 255),
                     #label='pj e_el')

            # ax1.plot(self.time_array, self.particles[0].energy_tan, color=(128 / 255, 0 / 255, 128 / 255), label='pi e_tan')
            # ax1.plot(self.time_array, self.particles[1].energy_tan, color=(165 / 255, 42 / 255, 42 / 255), label='pj e_tan')

            ax1.plot(self.time_array, self.particles[0].energy_damp, color=(255 / 255, 220 / 255, 0 / 255),
                     label='pi e_damp')
            #ax1.plot(self.time_array, self.particles[1].energy_damp, color=(255 / 255, 220 / 255, 0 / 255),
                     #label='pj e_damp')
            #(196 / 255, 101 / 255, 27 / 255)
            # Set the axis labels and title
            ax1.set_xlabel('Simulation Time [s]')
            ax1.set_ylabel('Energy [J]')
            # ax.set_title('Penetration Over Time Steps')
            ax1.legend()
            ax1.grid()
            ax1.set_xlim((min(self.time_array), max(self.time_array) + self.dt))
            plt.show()

            plt.savefig(self.plot_dir +'/' + self.plot_name + '.pdf')

        elif len(self.particles) == 2:
            for step in range(0, len(self.time_array)):
                total_en = 0
                for particle in self.particles:
                    total_en += particle.energy_kin[step] + particle.energy_rot[step] + particle.energy_el[step] + \
                                particle.energy_damp[step] + particle.energy_pot[step]  # + particle.energy_tan[step]
                self.total_energy.append(total_en)

            mpl.use("Qt5Agg")

            ## plotting energies

            # fig, (ax1, ax2, ax3) = plt.subplots(3)
            fig, ax1 = plt.subplots(1)
            # fig.tight_layout()
            # fig.canvas.set_window_title('Developement of v_ij and the Torque between two Particles over Time(Oblique Impact)')
            # ax1.title.set_text('Energy Distribution Oblique Elastic Impact')
            # Plot the data

            ax1.plot(self.time_array, self.total_energy, color=(0 / 255, 0 / 255, 0 / 255), label='total energy')
            # ax1.plot(self.plot_time_steps, self.damping_force, color=(0 / 255, 0 / 255, 0 / 255), label='Damping Force')
            # ax1.plot(self.plot_time_steps, self.elastic_force, color=(0 / 255, 101 / 255, 189 / 255), label='Elastic Force')
            ax1.plot(self.time_array, self.particles[0].energy_kin, color=(0 / 255, 101 / 255, 189 / 255),
                     label='pi e_kin')
            ax1.plot(self.time_array, self.particles[1].energy_kin, color=(227 / 255, 114 / 255, 34 / 255),
                     label='pj e_kin')

            # ax1.plot(self.plot_time_steps, self.particles[0].energy_rot, color=(196 / 255, 101 / 255, 27 / 255), label='pi e_rot')
            # ax1.plot(self.plot_time_steps, self.particles[1].energy_rot, color=(255 / 255, 220 / 255, 0 / 255), label='pj e_rot')

            ax1.plot(self.time_array, self.particles[0].energy_el, color=(162 / 255, 173 / 255, 0 / 255),
                     label='pi e_el')
            ax1.plot(self.time_array, self.particles[1].energy_el, color=(88 / 255, 88 / 255, 90 / 255),
                     label='pj e_el')

            # ax1.plot(self.plot_time_steps, self.particles[0].energy_tan, color=(128 / 255, 0 / 255, 128 / 255), label='pi e_tan')
            # ax1.plot(self.plot_time_steps, self.particles[1].energy_tan, color=(165 / 255, 42 / 255, 42 / 255), label='pj e_tan')

            ax1.plot(self.time_array, self.particles[0].energy_damp, color=(196 / 255, 101 / 255, 27 / 255),
                     label='pi e_damp')
            ax1.plot(self.time_array, self.particles[1].energy_damp, color=(255 / 255, 220 / 255, 0 / 255),
                     label='pj e_damp')

            # Set the axis labels and title
            ax1.set_xlabel('Simulation Time [s]')
            ax1.set_ylabel('Energy [J]')
            # ax.set_title('Penetration Over Time Steps')
            ax1.legend()
            ax1.grid()
            ax1.set_xlim((min(self.time_array), max(self.time_array) + self.dt))
            plt.show()

            plt.savefig(self.plot_dir +'/' + self.plot_name + '.pdf')



'''
    def plot_energy_kin(self):
        # Plot individual particle energies over time
        fig, ax = plt.subplots()
        for particle in self.particles:
            ax.plot(self.time, particle.energy_kin, label=f"Particle {i + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Kinetic Energy")
        ax.legend()

        # Sum up particle energies to get total energy over time
        total_energy = np.sum(self.energy_kin, axis=0)

        # Plot total energy over time
        fig, ax = plt.subplots()
        ax.plot(self.time, total_energy)
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Kinetic Energy")
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "energy_kin.pdf")
        plt.savefig(save_path)
        plt.show()
'''