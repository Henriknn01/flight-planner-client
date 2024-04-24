from PySide6.QtGui import QAction
from PySide6 import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow, MultiPlotter

from widgets import InteractorWidget, MainPanelWidget

SOFTWARE_VERSION = "1.0.0"


class ApplicationMainWindow(MainWindow):
    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle("Flight path generator v"+SOFTWARE_VERSION)

        # create the frame
        self.frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout()

        self.plotter = MultiPlotter(ncols=2, show=False)  # QtInteractor(self.frame)
        self.plot1 = InteractorWidget(self.plotter[0, 0], "Model preview - select scan section")
        layout.addWidget(self.plot1)

        self.plot2 = InteractorWidget(self.plotter[0, 1], "Generated Flight Path")
        layout.addWidget(self.plot2)

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
        self.plotter[0, 0].add_mesh(sphere, show_edges=True)
        self.plotter[0, 0].reset_camera()
