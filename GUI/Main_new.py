import sys

import numpy as np
# main script von hier laufen lassen
from PyQt6 import QtWidgets, QtGui
from MainWindow_test import Ui_MainWindow
from SecondWindow_parser import SecondWindow
from drawing_boundaries import boundary_creator
from dem_solver import PositionTracker, System

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.second_window = None
        self.particles = None
        self.boundaries = None
        self.secondWindow.clicked.connect(self.second_window_calling)
        # self.Particles.clicked.connect()
        self.Boundaries.clicked.connect(self.create_boundaries) # bei self.methode brauch ich keine klammern ausser externe variablen
        self.pushButton_5.clicked.connect(self.run_simulation)

    def second_window_calling(self):
        self.second_window = SecondWindow(parent=self)
        self.second_window.list_of_particles.connect(self.receive_particles)
        self.second_window.show()

    def receive_particles(self, particle):
        self.particles = particle
        print(self.particles)
        print(self.particles[0])
        print(type(self.particles[0].position))

    def create_boundaries(self):
        self.boundaries = boundary_creator()
        print(self.boundaries)
        print(len(self.boundaries))

    def run_simulation(self):
        system = System(particles=self.particles, boundaries=self.boundaries, dt=0.01, simtime=2, mu=0.7, coeff_of_restitution=1)  # system properties
        system.run_simulation()
        # positions = system.get_positions()




# this ensures that the game can only be called
# here and not through an import
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()







