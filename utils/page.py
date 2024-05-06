from PySide6.QtWidgets import QWidget, QHBoxLayout
from pyvistaqt import MultiPlotter
from widgets import InteractorWidget, MainPanelWidget


class GeneratePathPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = MultiPlotter(ncols=2, show=False)  # QtInteractor(self.frame)
        self.plot1 = InteractorWidget(self.plotter[0, 0], "Model preview - select scan section")
        layout.addWidget(self.plot1)

        self.plot2 = InteractorWidget(self.plotter[0, 1], "Generated Flight Path")
        layout.addWidget(self.plot2)

        # self.signal_close.connect(self.plotter.close)

        self.mainPanelWidgetInstance = MainPanelWidget(self.plotter)
        self.mainPanelWidgetInstance.setMaximumWidth(int(self.width()*0.33))
        layout.addWidget(self.mainPanelWidgetInstance)

        self.setLayout(layout)


class SelectReferencePointsPage(QWidget):
    def __init__(self):
        super().__init__()

