import numpy as np
# main script von hier laufen lassen
from PyQt6 import QtWidgets, QtGui, QtCore
from SecondWindow import Ui_MainWindow
from Particle import Particle


class SecondWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    list_of_particles = QtCore.pyqtSignal(list) # wahrscheinlich Datentyp, zur Not leer lassen, zur änderung von datentypen z.ip2.

    def __init__(self, *args, obj=None, **kwargs):
        super(SecondWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.a = None
        self.back.clicked.connect(self.second_close)
        self.Add_Particle.clicked.connect(self.add_particle)
        self.counter = 1
        self.particles = []
        self.current_Particle.display(1)


    # hier müssen die funktionen rein
    def second_close(self):
        #self.a = 123
        #self.list_of_particles.emit(self.a)
        self.list_of_particles.emit(self.particles)
        self.close()

    def add_particle(self):
        particle = self.create_particle()
        if particle is not None:
            self.particles.append(particle)
            self.counter += 1
            self.current_Particle.display(self.counter)

    def create_particle(self):
        position_str = self.position.text().split(',')
        velocity_str = self.velocity.text().split(',')
        rot_vel = float(self.rot_vel.text())
        rotation = float(self.rotation.text())
        radius = float(self.radius.text())
        elstiffnessn = float(self.stiffnes.text())
        mass = float(self.mass.text())
        poisson = float(self.poisson.text())

        try:
            position = np.array([float(position_str[0]), float(position_str[1]), 0])
            velocity = np.array([float(velocity_str[0]), float(velocity_str[1]), 0])
        except ValueError:
            # If the position or velocity strings cannot be converted to arrays, show an error message
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid position or velocity.")
            return None

        particle = Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, float(rotation)]),
                            rotation_vel=np.array([0, 0, float(rot_vel)]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, k_n=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]), poisson=poisson)
        print(particle)
        return particle



