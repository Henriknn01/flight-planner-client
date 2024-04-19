import os
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QUrl, QObject, Slot, QStringEncoder, Qt
from PySide6.QtGui import QGuiApplication, QFont
from PySide6.QtQuick import QQuickView
from PySide6.QtDataVisualization import qDefaultSurfaceFormat
from PySide6.QtWidgets import QWidget, QListView, QVBoxLayout, QSlider, QPushButton, QFileDialog, QApplication, QLabel, QHBoxLayout
from superqt import QLabeledRangeSlider, QLabeledSlider

SOFTWARE_VERSION = "1.0.0"


class SettingSection(QWidget):
    def __init__(self, name: str):
        super().__init__()
        self.sectionVisible = True

        self.layout = QVBoxLayout(self)

        self.toggleSection = QPushButton(name, self)
        self.toggleSection.clicked.connect(self.toggleSectionFunc)
        self.layout.addWidget(self.toggleSection)

        self.sectionParentWidget = QWidget(self)
        self.sectionWidgets = QVBoxLayout(self.sectionParentWidget)
        self.layout.addWidget(self.sectionParentWidget)

        self.initSectionHeight = self.sectionParentWidget.height() + 10

    def addWidget(self, widget: QWidget):
        self.sectionWidgets.addWidget(widget)
        self.initSectionHeight += widget.height()

    def toggleSectionFunc(self):
        if self.sectionVisible:
            self.sectionParentWidget.setFixedHeight(0)
            self.sectionVisible = False
        else:
            self.sectionParentWidget.setFixedHeight(self.initSectionHeight)
            self.sectionVisible = True


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        self.title = QLabel(self)
        self.title.setText("Flight path generator v"+ SOFTWARE_VERSION)
        self.title.setFont(QFont('Arial', 24, QFont.Bold))
        self.layout.addWidget(self.title)

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
        self.layout.addWidget(self.submit_btn)

        self.setLayout(self.layout)

    def showFileDialog(self):
        self.file_dialog = QFileDialog(self)
        self.file_dialog.setFileMode(QFileDialog.AnyFile)
        if self.file_dialog.exec_() == QFileDialog.Accepted:
            file_name = self.file_dialog.selectedFiles()[0]
            self.file_label.setText(file_name.split("/")[-1])
            print("Selected file: ", file_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWidget()
    widget.resize(400, 600)
    widget.setWindowTitle("Flight path generator v"+SOFTWARE_VERSION)
    widget.show()
    ex = app.exec()
    sys.exit(ex)