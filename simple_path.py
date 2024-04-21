import sys

import pyvista
from PySide6.QtCore import QCoreApplication, QUrl, QObject, Slot, QStringEncoder, Qt
from PySide6.QtGui import QGuiApplication, QFont, QAction
from PySide6.QtWidgets import QWidget, QCheckBox, QToolBar, QListView, QVBoxLayout, QPushButton, QFileDialog, QApplication, QLabel, QHBoxLayout
from PySide6 import QtWidgets
from superqt import QLabeledRangeSlider, QLabeledSlider
import pyvista as pv
from pyvistaqt import BackgroundPlotter, QtInteractor, MainWindow


"""
The simple path algo is a test algorithm developed while waiting for the final version of the more complex algorithm.
The algorithm slices the given mesh in the z axis with a given amount of slices along the z axis.
Once the slices have been created it will create distributed points along the path of each point at any given slice.
The points will have an offset from the path of the slices along the z axis by a given amount. 
This offset will act as the distance between the drone and the hull of the ship its scanning.
Each point represents a picture being taken of the ship hull.
"""
class SimplePath():
    def __init__(self, mesh):
        super().__init__()
        self.path_offset = 1
        self.line_width = 4
        self.mesh = mesh

    def generate_path(self):
        slices = self.mesh.slice_along_axis(n=64, axis=2)
        print("is_poly_data: "+str(slices.is_all_polydata))
        print("blocks: "+str(slices.n_blocks))
        slices.plot(line_width=self.line_width)
