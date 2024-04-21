import sys

import pyvista
from PySide6.QtCore import QCoreApplication, QUrl, QObject, Slot, QStringEncoder, Qt
from PySide6.QtGui import QGuiApplication, QFont, QAction
from PySide6.QtWidgets import QWidget, QCheckBox, QToolBar, QListView, QVBoxLayout, QPushButton, QFileDialog, QApplication, QLabel, QHBoxLayout
from PySide6 import QtWidgets
from superqt import QLabeledRangeSlider, QLabeledSlider
import pyvista as pv
from pyvistaqt import BackgroundPlotter, QtInteractor, MainWindow
from typing import Any
import simple_path


SOFTWARE_VERSION = "1.0.0"

class MainPanelWidget(QWidget):
    def __init__(self, plotter):
        super().__init__()

        self.layout = QVBoxLayout(self)

        self.plotter = plotter

        self.title = QLabel(self)
        self.title.setText("Flight path generator")
        self.title.setFont(QFont('Arial', 24, QFont.Bold))
        self.layout.addWidget(self.title)

        # settings section
        self.label_settings = QLabel(self)
        self.label_settings.setText("Display settings")
        self.layout.addWidget(self.label_settings)
        hbox_layout = QHBoxLayout()
        # Add toggle show bounds option
        self.show_bounds_checkbox = QCheckBox("Show Bounds", self)
        self.show_bounds_checkbox.setChecked(False)
        self.show_bounds_checkbox.stateChanged.connect(self.toggle_show_bounds)
        hbox_layout.addWidget(self.show_bounds_checkbox)
        self.layout.addLayout(hbox_layout)

        self.file_label = QLabel(self)
        self.file_label.setText("Select file")
        self.layout.addWidget(self.file_label)
        # Add file dialog button
        self.file_dialog_button = QPushButton('Select File', self)
        self.file_dialog_button.clicked.connect(self.showFileDialog)
        self.layout.addWidget(self.file_dialog_button)

        self.label_slider1 = QLabel(self)
        self.label_slider1.setText("Waypoint Offset")
        self.layout.addWidget(self.label_slider1)

        self.labeled_slider1 = QLabeledSlider(self)
        self.layout.addWidget(self.labeled_slider1)

        self.label_slider2 = QLabel(self)
        self.label_slider2.setText("Overlap Amount")
        self.layout.addWidget(self.label_slider2)

        self.labeled_slider2 = QLabeledSlider(self)
        self.layout.addWidget(self.labeled_slider2)

        self.label_rangeSlider1 = QLabel(self)
        self.label_rangeSlider1.setText("Altitude Range")
        self.layout.addWidget(self.label_rangeSlider1)
        self.range_slider_one = QLabeledRangeSlider(Qt.Horizontal, self)
        self.range_slider_one.setRange(0, 10)
        self.range_slider_one.setValue([2, 8])
        self.layout.addWidget(self.range_slider_one)

        self.label_rangeSlider2 = QLabel(self)
        self.label_rangeSlider2.setText("Scan Range")
        self.layout.addWidget(self.label_rangeSlider2)
        self.range_slider_two = QLabeledRangeSlider(Qt.Horizontal, self)
        self.range_slider_two.setRange(0, 10)
        self.range_slider_two.setValue([2, 8])
        self.layout.addWidget(self.range_slider_two)

        self.submit_btn = QPushButton('Generate', self)
        self.submit_btn.clicked.connect(self.generate)
        self.layout.addWidget(self.submit_btn)

        self.setLayout(self.layout)


    def toggle_show_bounds(self):
        if self.show_bounds_checkbox.isChecked():
            self.plotter.show_bounds()
        else:
            self.plotter.remove_bounds_axes()

    def showFileDialog(self):
        self.file_dialog = QFileDialog(self)
        self.file_dialog.setNameFilters(["OBJ files(*.obj)"])
        self.file_dialog.setFileMode(QFileDialog.AnyFile)
        if self.file_dialog.exec_() == QFileDialog.Accepted:
            file_name = self.file_dialog.selectedFiles()[0]
            self.file_label.setText(file_name.split("/")[-1])
            self.plotter.clear_actors()
            hull = pyvista.read(file_name)
            # self.plotter.add_mesh(hull, cmap="terrain", lighting=True, smooth_shading=True, split_sharp_edges=True)
            self.plotter.add_mesh_clip_box(hull, cmap="terrain", lighting=True)
            self.plotter.reset_camera()

    def generate(self):
        path = simple_path.SimplePath(self.plotter.box_clipped_meshes[0])
        path.generate_path()

class ApplicationMainWindow(MainWindow):
    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle("Flight path generator v"+SOFTWARE_VERSION)

        # create the frame
        self.frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        self.plotter.show_axes()
        layout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.mainPanelWidgetInstance = MainPanelWidget(self.plotter)
        self.mainPanelWidgetInstance.setMaximumWidth(int(self.width()*0.66))
        layout.addWidget(self.mainPanelWidgetInstance)

        self.frame.setLayout(layout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        if show:
            self.show()

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere, show_edges=True)
        self.plotter.reset_camera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ApplicationMainWindow()
    window.showMaximized()

    ex = app.exec()
    sys.exit(ex)