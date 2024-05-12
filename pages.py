from PySide6.QtWidgets import QWidget, QHBoxLayout
from pyvistaqt import MultiPlotter
from widgets import InteractorWidget, MainPanelWidget, SelectReferencePointsWidget


class GeneratePathPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = MultiPlotter(ncols=2, show=False)  # QtInteractor(self.frame)
        self.plot1 = InteractorWidget(self.plotter[0, 0], "Select Scan Section")
        layout.addWidget(self.plot1)

        self.plot2 = InteractorWidget(self.plotter[0, 1], "Generated Flight Path")
        layout.addWidget(self.plot2)

        # self.signal_close.connect(self.plotter.close)

        self.mainPanelWidgetInstance = MainPanelWidget(self.plotter)
        self.mainPanelWidgetInstance.setMaximumWidth(int(self.width()*0.33))
        self.mainPanelWidgetInstance.setMinimumWidth(300)
        layout.addWidget(self.mainPanelWidgetInstance)

        self.setLayout(layout)


class SelectReferencePointsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = MultiPlotter(ncols=1, show=False)  # QtInteractor(self.frame)
        self.plot = InteractorWidget(self.plotter[0, 0], "Please upload the ship hull model")
        layout.addWidget(self.plot)

        self.mainPanelWidgetInstance = SelectReferencePointsWidget(self.plotter)
        self.mainPanelWidgetInstance.setMaximumWidth(int(self.width()))
        self.mainPanelWidgetInstance.setMinimumWidth(300)
        layout.addWidget(self.mainPanelWidgetInstance)

        self.setLayout(layout)

