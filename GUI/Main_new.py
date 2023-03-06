import sys

import numpy as np
# main script von hier laufen lassen
from PyQt6 import QtWidgets, QtGui
from MainWindow_test import Ui_MainWindow
from SecondWindow_parser import SecondWindow
from drawing_boundaries import boundary_creator


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

    def second_window_calling(self):
        self.second_window = SecondWindow(parent=self)
        self.second_window.list_of_particles.connect(self.receive_particles)
        self.second_window.show()

    def receive_particles(self, particle):
        self.particles = particle
        print(self.particles)

    def create_boundaries(self):
        self.boundaries = boundary_creator()
        print(self.boundaries)
        print(len(self.boundaries))


# this ensures that the game can only be called
# here and not through an import
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()







