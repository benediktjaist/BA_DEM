import importlib
import sys
import os
import importlib.util
import traceback
import numpy as np
# main script von hier laufen lassen
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import QMessageBox

from MainWindow_test import Ui_MainWindow
from SecondWindow_parser import SecondWindow
from drawing_boundaries import boundary_creator
from DEM_Solver import System
from Preview_Assembly import Assembly
from Video_Creator import VideoCreator


def normalize_path(path):
    return path.replace("\\", "/")


def remove_trailing_zeros(number):
    formatted_number = '{:.16g}'.format(number)
    return float(formatted_number.rstrip('0').rstrip('.'))


sys._excepthook = sys.excepthook


def exception_hook(exctype, value, traceback):
    #print(exctype, value, traceback)
    #sys.exit()
    tb = ''.join(traceback.format_tb(traceback))
    message = f'{exctype.__name__}: {value}\n{tb}'
    print(message)
    sys.exit()


sys.excepthook = exception_hook


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.showMaximized()
        self.second_window = None
        self.particles = []
        self.boundaries = []
        self.gravity = False
        self.cor = 1.0
        self.dt = 0.001
        self.simtime = 1
        self.mu = 0.5
        self.vid_dir = "C:/Users/Jaist/Desktop/ba_videos"
        self.vid_name = "test2.mp4"
        self.fps = 50
        self.secondWindow.clicked.connect(self.second_window_calling)
        # self.Particles.clicked.connect()
        self.Boundaries.clicked.connect(self.create_boundaries) # bei self.methode brauch ich keine klammern ausser externe variablen
        self.Simulation.clicked.connect(self.run_simulation)
        self.Del_Particles.clicked.connect(self.delete_particles)
        self.Del_Boundaries.clicked.connect(self.delete_boundaries)
        self.Assembly.clicked.connect(self.show_assembly)
        self.gravity_checkBox.stateChanged.connect(self.update_gravity)
        self.cor_SpinBox.valueChanged.connect(self.update_cor)
        self.dt_SpinBox.valueChanged.connect(self.update_dt)
        self.simtime_SpinBox.valueChanged.connect(self.update_simtime)
        self.mu_SpinBox.valueChanged.connect(self.update_mu)
        self.vid_dir_edit.textChanged.connect(self.update_vid_dir)
        self.vid_name_edit.textChanged.connect(self.update_vid_name)
        self.fps_spinBox.valueChanged.connect(self.update_fps)
        self.Video.clicked.connect(self.create_video)
        self.import_button.clicked.connect(self.import_assembly)
        self.import_comboBox.addItems(self.get_file_names())
        self.import_change_path.clicked.connect(self.import_comboBox.clear)
        self.import_change_path.clicked.connect(lambda: self.import_comboBox.addItems(self.get_file_names()))

    def import_assembly(self):
        # Deleting particles and boundaries from a previous import
        self.particles = []
        self.boundaries = []
        ort = self.import_path.text()
        module_path = ort + "/" + str(self.import_comboBox.currentText())
        # print(module_path)
        module_name = str(self.import_comboBox.currentText())[:-3]
        # print(module_name)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # change attributes
        for id, teil in enumerate(module.teilchen):
            self.particles.append(teil)
            teil.id = id
        for id, grenze in enumerate(module.grenzen):
            self.boundaries.append(grenze)
            grenze.id = id
        self.gravity = module.gravitation
        self.cor = module.kor
        self.dt = module.tinkr
        self.simtime = module.simzeit
        self.mu = module.mue
        self.message_import_finished()
        # change displayed attributes
        self.cor_SpinBox.setValue(module.kor)
        self.dt_SpinBox.setValue(module.tinkr)
        self.simtime_SpinBox.setValue(module.simzeit)
        self.mu_SpinBox.setValue(module.mue)
        if self.gravity == True:
            self.gravity_checkBox.setChecked(True)
        else:
            self.gravity_checkBox.setChecked(False)



    def get_file_names(self):
        # Set the directory containing the files
        directory = self.import_path.text()
        # Get a list of file names in the directory
        file_names = os.listdir(directory)
        # Filter the list to only include files ending in .py
        file_names = [f for f in file_names if f.endswith('.py')]
        return file_names



    def update_fps(self, value):
        self.fps = round(value)

    def update_vid_name(self, text):
        self.vid_name = text

    def update_vid_dir(self, text):
        self.vid_dir = text

    def update_mu(self, value):
        self.mu = round(value, 2)
        print(self.mu)

    def update_simtime(self, value):
        self.simtime = round(value, 2)
        print(self.simtime)

    def update_dt(self, value):
        self.dt = round(value, 6)
        print(self.dt)

    def update_cor(self, value):
        self.cor = round(value, 2)
        print(self.cor)

    def update_gravity(self, state):
        self.gravity = state == 2
        print(self.gravity)

    def second_window_calling(self):
        self.second_window = SecondWindow(parent=self)
        self.second_window.list_of_particles.connect(self.receive_particles)
        self.second_window.show()

    def receive_particles(self, particle):
        self.particles = particle
        for id, particlee in enumerate(self.particles):
            particlee.id = id
        print(self.particles)
        print(self.particles[0])
        print(type(self.particles[0].position))

    def delete_particles(self):
        self.particles = []
        self.message_deleted_particles()

    def create_boundaries(self):
        self.boundaries = boundary_creator()
        for id, boundary in enumerate(self.boundaries):
            boundary.id = id
            #boundary.id = f'B{id + 1}'
        self.message_created_boundaries()

    def delete_boundaries(self):
        self.boundaries = []
        self.message_deleted_boundaries()

    def show_assembly(self):
        assembly = Assembly(particles=self.particles, boundaries=self.boundaries, gravity=self.gravity)
        assembly.show()

    def run_simulation(self):
        system = System(particles=self.particles, boundaries=self.boundaries, dt=self.dt, simtime=self.simtime,
                        mu=self.mu, coeff_of_restitution=self.cor, gravity=self.gravity)
        # Progress Tracker
        system.iterationChanged.connect(self.sim_progressBar.setValue)
        system.total_iterationsChanged.connect(self.sim_progressBar.setMaximum)
        system.remaining_timeChanged.connect(self.sim_dur_lcdNumber.display)
        self.sim_progressBar.setMaximum(system.total_iterations)

        system.run_simulation()
        self.message_sim_finished()
        if len(self.boundaries) > 0:
            what = 'p'+str(len(self.particles))+'b'+str(len(self.boundaries))
        else:
            what = 'p'+str(len(self.particles))
        if self.cor == 1:
            how = 'elastic'
        else:
            how = 'damp'
        text = what+'_'+how+'_'+'dt'+str(remove_trailing_zeros(self.dt))+'.mp4'
        self.vid_name_edit.setText(text)

    def create_video(self):
        print(self.vid_name)
        video = VideoCreator(particles=self.particles, boundaries=self.boundaries, dt=self.dt, simtime=self.simtime,
                             video_dir=self.vid_dir, video_name=self.vid_name)
        # Progress Tracker
        video.vid_iterationChanged.connect(self.vid_progressBar.setValue)
        video.vid_total_iterationsChanged.connect(self.vid_progressBar.setMaximum)
        video.vid_remaining_timeChanged.connect(self.vid_dur_lcdNumber.display)
        self.sim_progressBar.setMaximum(video.total_iterations)
        video.animate()
        self.message_video_finished()

    def message_sim_finished(self):
        msg = QMessageBox()
        msg.setWindowTitle("DEM Solver")
        msg.setText("Now the Simulation is finished")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def message_video_finished(self):
        msg = QMessageBox()
        msg.setWindowTitle("Video Creator")
        msg.setText("Now the Video of the Particles is finished")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def message_import_finished(self):
        msg = QMessageBox()
        msg.setWindowTitle("Assembly Import")
        msg.setText("Import finished successfully")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def message_deleted_particles(self):
        msg = QMessageBox()
        msg.setWindowTitle("Deleting Particles")
        msg.setText("deleting the particles was successful")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def message_deleted_boundaries(self):
        msg = QMessageBox()
        msg.setWindowTitle("Deleting Boundaries")
        msg.setText("deleting the boundaries was successful")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def message_created_boundaries(self):
        msg = QMessageBox()
        msg.setWindowTitle("Creating Boundaries")
        msg.setText("creating the boundaries was successful")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()


# this ensures that the game can only be called
# here and not through an import
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()







