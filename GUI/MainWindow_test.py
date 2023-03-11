# Form implementation generated from reading ui file 'MainWindow_test.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1064, 787)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Help = QtWidgets.QPushButton(parent=self.centralwidget)
        self.Help.setGeometry(QtCore.QRect(730, 380, 210, 51))
        self.Help.setObjectName("Help")
        self.TUM_Logo = QtWidgets.QLabel(parent=self.centralwidget)
        self.TUM_Logo.setEnabled(True)
        self.TUM_Logo.setGeometry(QtCore.QRect(700, 30, 381, 131))
        self.TUM_Logo.setPixmap(QtGui.QPixmap("TUM_Logo_extern_mt_EN_RGB_p.png"))
        self.TUM_Logo.setScaledContents(True)
        self.TUM_Logo.setObjectName("TUM_Logo")
        self.sim_progressBar = QtWidgets.QProgressBar(parent=self.centralwidget)
        self.sim_progressBar.setGeometry(QtCore.QRect(660, 250, 118, 23))
        self.sim_progressBar.setProperty("value", 0)
        self.sim_progressBar.setObjectName("sim_progressBar")
        self.sim_progress_lab = QtWidgets.QLabel(parent=self.centralwidget)
        self.sim_progress_lab.setGeometry(QtCore.QRect(520, 250, 111, 20))
        self.sim_progress_lab.setObjectName("sim_progress_lab")
        self.sim_dur_lab = QtWidgets.QLabel(parent=self.centralwidget)
        self.sim_dur_lab.setGeometry(QtCore.QRect(511, 310, 201, 20))
        self.sim_dur_lab.setObjectName("sim_dur_lab")
        self.sim_dur_lcdNumber = QtWidgets.QLCDNumber(parent=self.centralwidget)
        self.sim_dur_lcdNumber.setGeometry(QtCore.QRect(730, 310, 81, 23))
        self.sim_dur_lcdNumber.setStyleSheet("font: 8pt \"MS Shell Dlg 2\"; \n"
"color: rgb(0, 101, 189);\n"
"background-color: white;")
        self.sim_dur_lcdNumber.setDigitCount(8)
        self.sim_dur_lcdNumber.setObjectName("sim_dur_lcdNumber")
        self.layoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(60, 580, 481, 102))
        self.layoutWidget.setObjectName("layoutWidget")
        self.plotLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.plotLayout.setContentsMargins(0, 0, 0, 0)
        self.plotLayout.setObjectName("plotLayout")
        self.plot_name_edit = QtWidgets.QLineEdit(parent=self.layoutWidget)
        self.plot_name_edit.setObjectName("plot_name_edit")
        self.plotLayout.addWidget(self.plot_name_edit, 1, 1, 1, 1)
        self.plot_dir_lab = QtWidgets.QLabel(parent=self.layoutWidget)
        self.plot_dir_lab.setObjectName("plot_dir_lab")
        self.plotLayout.addWidget(self.plot_dir_lab, 0, 0, 1, 1)
        self.plot_name_lab = QtWidgets.QLabel(parent=self.layoutWidget)
        self.plot_name_lab.setObjectName("plot_name_lab")
        self.plotLayout.addWidget(self.plot_name_lab, 1, 0, 1, 1)
        self.create_plot = QtWidgets.QPushButton(parent=self.layoutWidget)
        self.create_plot.setObjectName("create_plot")
        self.plotLayout.addWidget(self.create_plot, 2, 1, 1, 1)
        self.plot_dir_edit = QtWidgets.QLineEdit(parent=self.layoutWidget)
        self.plot_dir_edit.setObjectName("plot_dir_edit")
        self.plotLayout.addWidget(self.plot_dir_edit, 0, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(60, 470, 421, 81))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.videoLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.videoLayout.setContentsMargins(0, 0, 0, 0)
        self.videoLayout.setObjectName("videoLayout")
        self.vid_dir_lab = QtWidgets.QLabel(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vid_dir_lab.sizePolicy().hasHeightForWidth())
        self.vid_dir_lab.setSizePolicy(sizePolicy)
        self.vid_dir_lab.setObjectName("vid_dir_lab")
        self.videoLayout.addWidget(self.vid_dir_lab, 0, 0, 1, 1)
        self.vid_name_lab = QtWidgets.QLabel(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vid_name_lab.sizePolicy().hasHeightForWidth())
        self.vid_name_lab.setSizePolicy(sizePolicy)
        self.vid_name_lab.setObjectName("vid_name_lab")
        self.videoLayout.addWidget(self.vid_name_lab, 1, 0, 1, 1)
        self.fps_spinBox = QtWidgets.QSpinBox(parent=self.layoutWidget1)
        self.fps_spinBox.setMinimum(30)
        self.fps_spinBox.setMaximum(120)
        self.fps_spinBox.setSingleStep(10)
        self.fps_spinBox.setProperty("value", 50)
        self.fps_spinBox.setObjectName("fps_spinBox")
        self.videoLayout.addWidget(self.fps_spinBox, 2, 1, 1, 1)
        self.vid_dir_edit = QtWidgets.QLineEdit(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vid_dir_edit.sizePolicy().hasHeightForWidth())
        self.vid_dir_edit.setSizePolicy(sizePolicy)
        self.vid_dir_edit.setObjectName("vid_dir_edit")
        self.videoLayout.addWidget(self.vid_dir_edit, 0, 1, 1, 4)
        self.vid_name_edit = QtWidgets.QLineEdit(parent=self.layoutWidget1)
        self.vid_name_edit.setObjectName("vid_name_edit")
        self.videoLayout.addWidget(self.vid_name_edit, 1, 1, 1, 4)
        self.Video = QtWidgets.QPushButton(parent=self.layoutWidget1)
        self.Video.setObjectName("Video")
        self.videoLayout.addWidget(self.Video, 2, 2, 1, 3)
        self.label = QtWidgets.QLabel(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.videoLayout.addWidget(self.label, 2, 0, 1, 1)
        self.layoutWidget2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(60, 330, 421, 112))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.cor_SpinBox = QtWidgets.QDoubleSpinBox(parent=self.layoutWidget2)
        self.cor_SpinBox.setMinimum(0.0)
        self.cor_SpinBox.setMaximum(1.0)
        self.cor_SpinBox.setSingleStep(0.1)
        self.cor_SpinBox.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
        self.cor_SpinBox.setProperty("value", 1.0)
        self.cor_SpinBox.setObjectName("cor_SpinBox")
        self.gridLayout.addWidget(self.cor_SpinBox, 0, 1, 1, 1)
        self.simtime_SpinBox = QtWidgets.QDoubleSpinBox(parent=self.layoutWidget2)
        self.simtime_SpinBox.setMinimum(1.0)
        self.simtime_SpinBox.setMaximum(100.0)
        self.simtime_SpinBox.setObjectName("simtime_SpinBox")
        self.gridLayout.addWidget(self.simtime_SpinBox, 2, 1, 2, 1)
        self.dt_lab = QtWidgets.QLabel(parent=self.layoutWidget2)
        self.dt_lab.setObjectName("dt_lab")
        self.gridLayout.addWidget(self.dt_lab, 1, 0, 1, 1)
        self.dt_SpinBox = QtWidgets.QDoubleSpinBox(parent=self.layoutWidget2)
        self.dt_SpinBox.setDecimals(6)
        self.dt_SpinBox.setMinimum(1e-06)
        self.dt_SpinBox.setMaximum(1.0)
        self.dt_SpinBox.setSingleStep(0.0001)
        self.dt_SpinBox.setProperty("value", 0.001)
        self.dt_SpinBox.setObjectName("dt_SpinBox")
        self.gridLayout.addWidget(self.dt_SpinBox, 1, 1, 1, 1)
        self.mu_SpinBox = QtWidgets.QDoubleSpinBox(parent=self.layoutWidget2)
        self.mu_SpinBox.setDecimals(1)
        self.mu_SpinBox.setMaximum(1.0)
        self.mu_SpinBox.setSingleStep(0.1)
        self.mu_SpinBox.setProperty("value", 0.5)
        self.mu_SpinBox.setObjectName("mu_SpinBox")
        self.gridLayout.addWidget(self.mu_SpinBox, 4, 1, 1, 1)
        self.simtime_lab = QtWidgets.QLabel(parent=self.layoutWidget2)
        self.simtime_lab.setObjectName("simtime_lab")
        self.gridLayout.addWidget(self.simtime_lab, 2, 0, 1, 1)
        self.gravity_checkBox = QtWidgets.QCheckBox(parent=self.layoutWidget2)
        self.gravity_checkBox.setObjectName("gravity_checkBox")
        self.gridLayout.addWidget(self.gravity_checkBox, 2, 2, 1, 1)
        self.cor_lab = QtWidgets.QLabel(parent=self.layoutWidget2)
        self.cor_lab.setObjectName("cor_lab")
        self.gridLayout.addWidget(self.cor_lab, 0, 0, 1, 1)
        self.Simulation = QtWidgets.QPushButton(parent=self.layoutWidget2)
        self.Simulation.setObjectName("Simulation")
        self.gridLayout.addWidget(self.Simulation, 4, 2, 1, 1)
        self.mu_lab = QtWidgets.QLabel(parent=self.layoutWidget2)
        self.mu_lab.setObjectName("mu_lab")
        self.gridLayout.addWidget(self.mu_lab, 3, 0, 2, 1)
        self.layoutWidget3 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(60, 210, 218, 89))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.createLayout = QtWidgets.QGridLayout(self.layoutWidget3)
        self.createLayout.setContentsMargins(0, 0, 0, 0)
        self.createLayout.setObjectName("createLayout")
        self.secondWindow = QtWidgets.QPushButton(parent=self.layoutWidget3)
        self.secondWindow.setObjectName("secondWindow")
        self.createLayout.addWidget(self.secondWindow, 0, 0, 1, 1)
        self.Del_Particles = QtWidgets.QPushButton(parent=self.layoutWidget3)
        self.Del_Particles.setObjectName("Del_Particles")
        self.createLayout.addWidget(self.Del_Particles, 0, 1, 1, 1)
        self.Boundaries = QtWidgets.QPushButton(parent=self.layoutWidget3)
        self.Boundaries.setObjectName("Boundaries")
        self.createLayout.addWidget(self.Boundaries, 1, 0, 1, 1)
        self.Del_Boundaries = QtWidgets.QPushButton(parent=self.layoutWidget3)
        self.Del_Boundaries.setObjectName("Del_Boundaries")
        self.createLayout.addWidget(self.Del_Boundaries, 1, 1, 1, 1)
        self.Assembly = QtWidgets.QPushButton(parent=self.layoutWidget3)
        self.Assembly.setObjectName("Assembly")
        self.createLayout.addWidget(self.Assembly, 2, 0, 1, 2)
        self.layoutWidget4 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget4.setGeometry(QtCore.QRect(61, 41, 521, 85))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget4)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.import_text = QtWidgets.QLabel(parent=self.layoutWidget4)
        self.import_text.setObjectName("import_text")
        self.gridLayout_2.addWidget(self.import_text, 1, 0, 1, 1)
        self.import_path = QtWidgets.QLineEdit(parent=self.layoutWidget4)
        self.import_path.setObjectName("import_path")
        self.gridLayout_2.addWidget(self.import_path, 1, 1, 1, 2)
        self.import_button = QtWidgets.QPushButton(parent=self.layoutWidget4)
        self.import_button.setObjectName("import_button")
        self.gridLayout_2.addWidget(self.import_button, 3, 0, 1, 3)
        self.import_change_path = QtWidgets.QPushButton(parent=self.layoutWidget4)
        self.import_change_path.setObjectName("import_change_path")
        self.gridLayout_2.addWidget(self.import_change_path, 2, 0, 1, 1)
        self.import_comboBox = QtWidgets.QComboBox(parent=self.layoutWidget4)
        self.import_comboBox.setObjectName("import_comboBox")
        self.gridLayout_2.addWidget(self.import_comboBox, 2, 1, 1, 2)
        self.vid_progress_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.vid_progress_label.setGeometry(QtCore.QRect(560, 480, 141, 16))
        self.vid_progress_label.setObjectName("vid_progress_label")
        self.vid_progressBar = QtWidgets.QProgressBar(parent=self.centralwidget)
        self.vid_progressBar.setGeometry(QtCore.QRect(730, 480, 118, 23))
        self.vid_progressBar.setProperty("value", 0)
        self.vid_progressBar.setObjectName("vid_progressBar")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(550, 530, 171, 16))
        self.label_3.setObjectName("label_3")
        self.vid_dur_lcdNumber = QtWidgets.QLCDNumber(parent=self.centralwidget)
        self.vid_dur_lcdNumber.setGeometry(QtCore.QRect(750, 530, 81, 23))
        self.vid_dur_lcdNumber.setStyleSheet("font: 8pt \"MS Shell Dlg 2\"; \n"
"color: rgb(0, 101, 189);\n"
"background-color: white;")
        self.vid_dur_lcdNumber.setDigitCount(8)
        self.vid_dur_lcdNumber.setObjectName("vid_dur_lcdNumber")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Help.setText(_translate("MainWindow", "Help"))
        self.sim_progress_lab.setText(_translate("MainWindow", "Simulation Progress:"))
        self.sim_dur_lab.setText(_translate("MainWindow", "Remaining Simulation Time: [s]"))
        self.plot_dir_lab.setText(_translate("MainWindow", "file directory for plot"))
        self.plot_name_lab.setText(_translate("MainWindow", "file name for plot"))
        self.create_plot.setText(_translate("MainWindow", "Create Energy Plot"))
        self.plot_dir_edit.setText(_translate("MainWindow", "C:/Users/Jaist/Documents/GitHub/BA_DEM/GUI/examples/plots"))
        self.vid_dir_lab.setText(_translate("MainWindow", "file directory for video"))
        self.vid_name_lab.setText(_translate("MainWindow", "file name for video"))
        self.vid_dir_edit.setText(_translate("MainWindow", "C:/Users/Jaist/Desktop/ba_videos"))
        self.vid_name_edit.setText(_translate("MainWindow", "test.mp4"))
        self.Video.setText(_translate("MainWindow", "Create Video with current Settings"))
        self.label.setText(_translate("MainWindow", "video FPS:"))
        self.dt_lab.setText(_translate("MainWindow", "time increment [s]"))
        self.simtime_lab.setText(_translate("MainWindow", "Simulation Time [s]"))
        self.gravity_checkBox.setText(_translate("MainWindow", "turn gravity on"))
        self.cor_lab.setText(_translate("MainWindow", "coefficient of restitution"))
        self.Simulation.setText(_translate("MainWindow", "Run Simulation"))
        self.mu_lab.setText(_translate("MainWindow", "friction coefficient"))
        self.secondWindow.setText(_translate("MainWindow", "Create Particles"))
        self.Del_Particles.setText(_translate("MainWindow", "Delete Particles"))
        self.Boundaries.setText(_translate("MainWindow", "Create Boundaries"))
        self.Del_Boundaries.setText(_translate("MainWindow", "Delete Boundaries"))
        self.Assembly.setText(_translate("MainWindow", "Preview Assembly"))
        self.import_text.setText(_translate("MainWindow", "Absolute Path for file:"))
        self.import_path.setText(_translate("MainWindow", "C:/Users/Jaist/Documents/GitHub/BA_DEM/GUI/examples"))
        self.import_button.setText(_translate("MainWindow", "Import Configuration"))
        self.import_change_path.setText(_translate("MainWindow", "Change Path"))
        self.vid_progress_label.setText(_translate("MainWindow", "Video  Creation Progress:"))
        self.label_3.setText(_translate("MainWindow", "Remaining Rendering Time: [s]"))
