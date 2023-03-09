import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QInputDialog, \
    QLineEdit, QTextEdit, QDialog

all_particles = []


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Particle Creator')

        # Create welcome message
        welcome_label = QLabel('Welcome to the Particle Creator!')
        welcome_label.setStyleSheet('font-size: 20px;')

        # Create button to input number of particles
        num_particles_button = QPushButton('Input Number of Particles', self)
        num_particles_button.clicked.connect(self.show_num_particles_dialog)

        # Create button to input properties of each particle
        particle_properties_button = QPushButton('Input Particle Properties', self)
        particle_properties_button.clicked.connect(self.show_particle_properties_dialog)

        # Add widgets to layout
        vbox = QVBoxLayout()
        vbox.addWidget(welcome_label)
        vbox.addWidget(num_particles_button)
        vbox.addWidget(particle_properties_button)

        self.setLayout(vbox)

    def create_particles(self, dialog):
        global all_particles
        all_particles = []

        for i in range(self.num_particles):
            position_str = dialog.layout().itemAt(i).widget().itemAt(2).widget().text()
            velocity_str = dialog.layout().itemAt(i).widget().itemAt(4).widget().text()

            position = tuple(map(int, position_str.split(',')))
            velocity = tuple(map(int, velocity_str.split(',')))

            particle = Particle(position, velocity)
            all_particles.append(particle)

        # Print list of Particle objects to console
        for particle in all_particles:
            print("Position: {}, Velocity: {}".format(particle.position, particle.velocity))

    def show_num_particles_dialog(self):
        num_particles, ok = QInputDialog.getInt(self, 'Input Number of Particles', 'Enter number of particles:', 2, 1)
        if ok:
            self.num_particles = num_particles

    def show_particle_properties_dialog(self):
        wanted_particles = []
        particle_properties_dialog = QDialog(self)
        particle_properties_dialog.setWindowTitle('Particle Properties')

        # Create layout for dialog box
        dialog_vbox = QVBoxLayout()

        # Create list to store particle properties
        particle_properties = []

        # Get the number of particles from the num_particles attribute
        num_particles = self.num_particles

        # Loop through each particle and create input fields for its properties
        for i in range(num_particles):
            # Create label for particle number
            particle_num_label = QLabel('Particle ' + str(i + 1))

            # Create input fields for particle position and velocity
            position_label = QLabel('Position (x, y):')
            position_edit = QLineEdit('0, 0')
            velocity_label = QLabel('Velocity (x, y):')
            velocity_edit = QLineEdit('0, 0')

            # Add input fields to layout
            particle_layout = QVBoxLayout()
            particle_layout.addWidget(particle_num_label)
            particle_layout.addWidget(position_label)
            particle_layout.addWidget(position_edit)
            particle_layout.addWidget(velocity_label)
            particle_layout.addWidget(velocity_edit)

            dialog_vbox.addLayout(particle_layout)

            # Store input values for particle position and velocity in list
            particle_properties.append((position_edit, velocity_edit))

        # Create button to create particles
        create_particles_button = QPushButton('Create Particles', particle_properties_dialog)
        create_particles_button.clicked.connect(lambda: self.create_particles(particle_properties_dialog))

        # Add button to layout
        dialog_vbox.addWidget(create_particles_button)

        # Set layout for dialog box
        particle_properties_dialog.setLayout(dialog_vbox)

        # Show dialog box and wait for it to be closed
        if particle_properties_dialog.exec_() == QDialog.Accepted:
            #self.particle_properties = particle_properties

            # Create Particle objects with specified properties
            for i in range(self.num_particles):
                position_str = particle_properties[i][0].text()
                velocity_str = particle_properties[i][1].text()

                position = tuple(map(int, position_str.split(',')))
                velocity = tuple(map(int, velocity_str.split(',')))

                Particle(position, velocity)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())





