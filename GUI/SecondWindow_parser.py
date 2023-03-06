import numpy as np
# main script von hier laufen lassen
from PyQt6 import QtWidgets, QtGui, QtCore
from SecondWindow import Ui_MainWindow


class SecondWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    list_of_particles = QtCore.pyqtSignal(int) # wahrscheinlich Datentyp, zur Not leer lassen, zur änderung von datentypen z.ip2.

    def __init__(self, *args, obj=None, **kwargs):
        super(SecondWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.a = None
        self.back.clicked.connect(self.second_close)


    # hier müssen die funktionen rein
    def second_close(self):
        self.a = 123
        self.list_of_particles.emit(self.a)
        self.close()
