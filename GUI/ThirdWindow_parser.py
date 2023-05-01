import numpy as np
# main script von hier laufen lassen
from PyQt6 import QtWidgets, QtGui, QtCore
from ThirdWindow import Ui_MainWindow
from Particle import Particle
import random


class ThirdWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    list_of_particle_prop_limits = QtCore.pyqtSignal(list)

    def __init__(self, *args, obj=None, **kwargs):
        super(ThirdWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.a = None
        self.back.clicked.connect(self.third_close)
        self.confirm_part_props.clicked.connect(self.get_particle_prop_limits)
        self.particle_prop_limits = []


    def third_close(self):
        self.list_of_particle_prop_limits.emit(self.particle_prop_limits)
        self.close()

    def get_particle_prop_limits(self):
        max_number_of_particles = int(self.num_particles_box.value())
        dist_factor = float(self.dist_factor_box.value())
        range_x_vel = (int(self.vel_x_min_box.value()), int(self.vel_x_max_box.value()))
        range_y_vel = (int(self.vel_y_min_box.value()), int(self.vel_y_max_box.value()))
        range_rot_vel = (int(self.rot_vel_min_box.value()), int(self.rot_vel_max_box.value()))
        range_rot = (int(self.rot_min_box.value()), int(self.rot_max_box.value()))
        range_radius = (int(self.rad_min_box.value()), int(self.rad_max_box.value()))
        range_mass = (int(self.mass_min_box.value()), int(self.mass_max_box.value()))
        range_young = (int(self.young_min.value()), int(self.young_max.value()))
        range_poisson = (int(self.poisson_min_box.value()), int(self.poisson_max_box.value()))

        self.particle_prop_limits.append(max_number_of_particles)
        self.particle_prop_limits.append(dist_factor)
        self.particle_prop_limits.append(range_radius)
        self.particle_prop_limits.append(range_x_vel)
        self.particle_prop_limits.append(range_y_vel)
        self.particle_prop_limits.append(range_rot_vel)
        self.particle_prop_limits.append(range_rot)
        self.particle_prop_limits.append(range_mass)
        self.particle_prop_limits.append(range_young)
        self.particle_prop_limits.append(range_poisson)

'''
# Define rectangle properties
rect_top = 50
rect_left = 50
rect_bottom = 782
rect_right = 1228
while current_circles < max_circles:
x = random.randint(rect_left + max_radius, rect_right - max_radius)
y = random.randint(rect_top + max_radius, rect_bottom - max_radius)
radius = random.randint(min_radius, max_radius)

# Check for intersections with all previously generated circles
intersect = False
for circle in circles:
    distance = ((x - circle[0]) ** 2 + (y - circle[1]) ** 2) ** 0.5
    if distance < circle[2] + radius + min_distance:
        intersect = True
        break

# Add circle if no intersections
if not intersect:
    circles.append((x, y, radius))
    current_circles += 1
for i in range(int(self.num_particles_box.text())):

position_str = self.position.text().split(',')
random_number = random.randint(1, 10)
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
                    pred_position=np.array([0, 0, 0]), interpenetration_vel=np.array([0, 0, 0]),
                    poisson=poisson)
print(particle)
self.particles.append(particle)
'''



