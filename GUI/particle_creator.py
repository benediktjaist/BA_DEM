from PyQt5 import QtWidgets, QtGui
import sys
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity


class ParticleCreator(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Particle Creator')

        # Initialize widgets
        self.num_particles_label = QtWidgets.QLabel('Enter number of particles (default is 2):')
        self.num_particles_edit = QtWidgets.QLineEdit()
        self.position_label = QtWidgets.QLabel('Enter position (format: x y):')
        self.position_edit = QtWidgets.QLineEdit()
        self.velocity_label = QtWidgets.QLabel('Enter velocity (format: x y):')
        self.velocity_edit = QtWidgets.QLineEdit()
        self.create_button = QtWidgets.QPushButton('Create')
        self.cancel_button = QtWidgets.QPushButton('Cancel')

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.num_particles_label)
        layout.addWidget(self.num_particles_edit)
        layout.addWidget(self.position_label)
        layout.addWidget(self.position_edit)
        layout.addWidget(self.velocity_label)
        layout.addWidget(self.velocity_edit)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect signals
        self.create_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_input(self):
        """
        Displays the dialog and returns the input values.
        """
        result = self.exec_()
        num_particles = int(self.num_particles_edit.text() or 2)
        position = tuple(map(float, self.position_edit.text().split()))
        velocity = tuple(map(float, self.velocity_edit.text().split()))
        return result, num_particles, position, velocity

def particle_creator():
    app = QtWidgets.QApplication(sys.argv)

    # Create ParticleCreator dialog
    dialog = ParticleCreator()

    # Get user input
    result, num_particles, position, velocity = dialog.get_input()

    if result == QtWidgets.QDialog.Accepted:
        # Create empty list to hold particle objects
        all_particles = []

        # Ask user for attributes of each particle
        for i in range(num_particles):
            particle_dialog = ParticleCreator()
            particle_dialog.num_particles_edit.setText('1')
            particle_dialog.position_edit.setText(f'{position[0]} {position[1]}')
            particle_dialog.velocity_edit.setText(f'{velocity[0]} {velocity[1]}')

            result, num_particles, position, velocity = particle_dialog.get_input()

            if result == QtWidgets.QDialog.Accepted:
                particle = Particle(position, velocity)
                all_particles.append(particle)
            else:
                break

        # Return list of particle objects
        return all_particles

    else:
        # Return None if user cancels input
        return None


if __name__ == "__main__":
    particles = particle_creator()
    print(particles)

