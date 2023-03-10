import os

import numpy as np
import matplotlib.pyplot as plt
from Colours import get_colour_tuple as col

class PlotCreator:
    def __init__(self, particles, dt, simtime): # plot_name, plot_dir
        self.particles = particles
        self.dt = dt
        self.simtime = simtime
        self.plot_name = None
        self.plot_dir = "C:/Users/Jaist/Documents/GitHub/BA_DEM/GUI/examples/plots"
        self.time = np.arange(0, self.simtime, self.dt)

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
    def plot(self):
        # data

        print(self.time)

        # Create plot
        fig, ax = plt.subplots()
        ax.plot(self.time, self.particles[0].energy_kin)
        ax.plot(self.time, self.particles[0].energy_rotation)
        ax.plot(self.time, self.particles[0].energy_el)
        ax.plot(self.time, self.particles[0].energy_damp)
        # Customize plot
        ax.set_title('Energy Tracking')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        # Show plot
        plt.show()
'''


