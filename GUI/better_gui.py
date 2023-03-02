from PyQt5 import QtWidgets
import sys
from Particle import Particle
import numpy as np
import random


class NumParticlesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Number of Particles')
        self.setGeometry(2500, 1300, 200, 100)  # (position x, position y, size x, size y)

        # Initialize widgets
        self.num_particles_label = QtWidgets.QLabel('Enter number of particles:')
        self.num_particles_edit = QtWidgets.QLineEdit('2')
        self.ok_button = QtWidgets.QPushButton('OK')
        self.cancel_button = QtWidgets.QPushButton('Cancel')

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.num_particles_label)
        layout.addWidget(self.num_particles_edit)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_input(self):
        """
        Displays the dialog and returns the result.
        """
        result = self.exec_()
        return result


class ParticleCreator(QtWidgets.QDialog):
    def __init__(self, default_position=None, default_velocity=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Particle Creator')

        # Initialize widgets
        self.position_label = QtWidgets.QLabel('Enter position (format: x,y):')
        self.position_edit = QtWidgets.QLineEdit()
        self.velocity_label = QtWidgets.QLabel('Enter velocity (format: x,y):')
        self.velocity_edit = QtWidgets.QLineEdit()
        self.radius_label = QtWidgets.QLabel('Enter radius:')
        self.radius_edit = QtWidgets.QLineEdit('50')
        self.elstiffness_label = QtWidgets.QLabel('Enter elstiffnessn:')
        self.elstiffness_edit = QtWidgets.QLineEdit('2000')
        self.mass_label = QtWidgets.QLabel('Enter mass:')
        self.mass_edit = QtWidgets.QLineEdit('100')
        self.create_button = QtWidgets.QPushButton('Create')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.setGeometry(2500, 1300, 400, 300)  # (position x, position y, size x, size y)

        # Set default values for position and velocity
        if default_position is not None:
            self.position_edit.setText(f"{default_position[0]},{default_position[1]}")
        if default_velocity is not None:
            self.velocity_edit.setText(f"{default_velocity[0]},{default_velocity[1]}")

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.position_label)
        layout.addWidget(self.position_edit)
        layout.addWidget(self.velocity_label)
        layout.addWidget(self.velocity_edit)
        layout.addWidget(self.radius_label)
        layout.addWidget(self.radius_edit)
        layout.addWidget(self.elstiffness_label)
        layout.addWidget(self.elstiffness_edit)
        layout.addWidget(self.mass_label)
        layout.addWidget(self.mass_edit)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect signals
        self.create_button.clicked.connect(self.create_particle)
        self.cancel_button.clicked.connect(self.reject)

        self.created_particle = None

    def create_particle(self):
        """
        Create a particle object and clear the input fields.
        """
        position = np.array(list(map(float, self.position_edit.text().split(','))) + [0])
        velocity = np.array(list(map(float, self.velocity_edit.text().split(','))) + [0])
        radius = float(self.radius_edit.text())
        elstiffnessn = float(self.elstiffness_edit.text())
        mass = float(self.mass_edit.text())
        particle = Particle(position=position, velocity=velocity, acceleration=np.array([0, 0, 0]),
                            force=np.array([0, 0, 0]), rotation=np.array([0, 0, 0]),
                            rotation_vel=np.array([0, 0, 0]), rotation_acc=np.array([0, 0, 0]),
                            torque=np.array([0, 0, 0]), radius=radius, elstiffnesn=elstiffnessn, mass=mass,
                            pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]))

        self.position_edit.clear()
        self.velocity_edit.clear()
        self.radius_edit.clear()
        self.elstiffness_edit.clear()
        self.mass_edit.clear()

        self.created_particle = particle

        self.accept()

    def get_particle_input(self):
        """
        Displays the dialog and returns the result and the created particle object.
        """
        result = self.exec_()
        return result, self.created_particle


def particle_creator():
    app = QtWidgets.QApplication(sys.argv)

    # Get number of particles
    num_particles_dialog = NumParticlesDialog()
    result = num_particles_dialog.get_input()

    if result == QtWidgets.QDialog.Accepted:
        # Create empty list to hold particle objects
        all_particles = []

        # Ask user for attributes of each particle
        for i in range(int(num_particles_dialog.num_particles_edit.text())):
            if i == 0:
                # Set different default values for first particle
                default_position = (500, 500)
                default_velocity = (50, 50)
            elif i == 1:
                # Set different default values for second particle
                default_position = (500, 620)
                default_velocity = (0, 0)
            else:
                # Set default values for all other particles
                default_position = (500, 500)
                default_velocity = (50, 50)

                # Randomize default values for all other particles
                default_position = (random.randint(0, 1000), random.randint(0, 1000))
                default_velocity = (random.uniform(-100, 100), random.uniform(-100, 100))

            particle_dialog = ParticleCreator(default_position=default_position, default_velocity=default_velocity)
            particle_dialog.position_label.setText(f'Enter position for particle {i + 1} (format: x,y):')
            particle_dialog.velocity_label.setText(f'Enter velocity for particle {i + 1} (format: x,y):')

            result, particle = particle_dialog.get_particle_input()

            if result == QtWidgets.QDialog.Accepted:
                all_particles.append(particle)
            else:
                break

        # Return list of particle objects
        return all_particles

    else:
        # Return None if user cancels input
        return None


# this ensures that the GUI can only be called
# here and not through an import for e.g. test purposes
if __name__ == "__main__":
    particles = particle_creator()
    if particles:
        print(particles[0].position)
        print(type(particles[0].position))


