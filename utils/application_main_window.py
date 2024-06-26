from PySide6.QtCore import SIGNAL
from PySide6.QtGui import QAction, QFont
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QLabel, QFileDialog
from pyvistaqt import MainWindow
from pages import GeneratePathPage, SelectReferencePointsPage
from widgets import MainPanelWidget

SOFTWARE_VERSION = "1.0.0"


class ApplicationMainWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        layout = QtWidgets.QVBoxLayout()

        # Set the title of the application window
        self.setWindowTitle("Flight path generator v"+SOFTWARE_VERSION)

        # Pag heading + styling
        self.title = QLabel(self)
        self.title.setText("Flight path generator")
        self.title.setFont(QFont('Arial', 44, QFont.Bold))
        self.title.setContentsMargins(0, 20, 0, 20)
        layout.addWidget(self.title)

        # Create the frame
        self.frame = QtWidgets.QFrame()
        self.stackedView = QtWidgets.QStackedWidget()

        # Set up pages
        self.selectReferencePointsPage = SelectReferencePointsPage()
        self.selectReferencePointsPage.mainPanelWidgetInstance.page_changed.connect(self.page_change)
        self.generatePage = GeneratePathPage()
        self.generatePage.mainPanelWidgetInstance.page_changed.connect(self.page_change)
        self.selectReferencePointsPage.mainPanelWidgetInstance.model_uploaded.connect(self.generatePage.mainPanelWidgetInstance.set_plotter_model)
        self.selectReferencePointsPage.mainPanelWidgetInstance.reference_points_changed.connect(self.generatePage.mainPanelWidgetInstance.set_reference_points)

        # Add widgets to view
        self.stackedView.addWidget(self.generatePage)
        self.stackedView.addWidget(self.selectReferencePointsPage)

        # Set the current widget displayed to the user
        self.stackedView.setCurrentWidget(self.selectReferencePointsPage)

        layout.addWidget(self.stackedView)

        self.frame.setLayout(layout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        export = fileMenu.addAction('Export to simulator', self.export_to_sim_file)
        export_gps = fileMenu.addAction('Export GPS data', self.export_to_gps_file)
        export_kml = fileMenu.addAction('Export to KML', self.export_to_kml_file)
        exitButton = QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

    @QtCore.Slot(int)
    def page_change(self, page):
        print("page changed to " + str(page))
        self.stackedView.setCurrentIndex(page)

    def export_to_sim_file(self):
        fileName = QFileDialog.getSaveFileName(self, 'Save path to simulator file', '/', selectedFilter='*.txt')
        if fileName:
            self.generatePage.mainPanelWidgetInstance.algo.export_to_sim_file(fileName)

    def export_to_gps_file(self):
        fileName = QFileDialog.getSaveFileName(self, 'Save path to gps file', '/', selectedFilter='*.txt')
        if fileName:
            self.generatePage.mainPanelWidgetInstance.algo.export_to_gps_file(fileName)

    def export_to_kml_file(self):
        fileName = QFileDialog.getSaveFileName(self, 'Save path to kml file', '/', selectedFilter='*.txt')
        if fileName:
            self.generatePage.mainPanelWidgetInstance.algo.export_to_kml_file(fileName)

    def handleDataReceived(self, data):
        print(data)
