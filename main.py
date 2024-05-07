import sys
from PySide6.QtWidgets import QApplication
from utils.application_main_window import ApplicationMainWindow
import random
import logging
import time


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ApplicationMainWindow()
    window.showMaximized()
    ex = app.exec()
    sys.exit(ex)
