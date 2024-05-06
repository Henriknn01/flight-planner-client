from PySide6.QtGui import QAction, QFont
from PySide6 import QtWidgets
from PySide6.QtWidgets import QLabel
import pyvista as pv
from pyvistaqt import MainWindow
from utils.page import GeneratePathPage

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
        self.generatePage = GeneratePathPage()

        # Add widgets to view
        self.stackedView.addWidget(self.generatePage)

        # Set the current widget displayed to the user
        self.stackedView.setCurrentWidget(self.generatePage)

        layout.addWidget(self.stackedView)

        self.frame.setLayout(layout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)
