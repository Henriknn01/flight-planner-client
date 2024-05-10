import pyvista
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QWidget, QCheckBox, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QLineEdit
from superqt import QLabeledRangeSlider, QLabeledSlider, QLabeledDoubleSlider
from utils.simple_path import SimplePath
from utils.algo import SliceSurfaceAlgo


class InteractorWidget(QWidget):
    def __init__(self, plotter, top_text=None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        # The class assumes that the plotter is a multi plotter with 2 cols
        self.plotter = plotter
        self.original_mesh = None

        if top_text:
            self.plotter.add_text(top_text)

        self.plotter.show_axes()

        # self.layout.addWidget(self.plotter.main_menu)
        self.layout.addWidget(self.plotter.default_camera_tool_bar)
        self.layout.addWidget(self.plotter.interactor)

        self.setLayout(self.layout)


class ReferencePointWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        hbox_layout = QHBoxLayout()

        self.label_ref = QLabel(self)
        self.label_ref.setText("Virtual Coordinates")
        self.layout.addWidget(self.label_ref)

        self.label_ref_x = QLabel(self)
        self.label_ref_x.setText("x: ")
        self.x_r = QLineEdit(self)
        self.x_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_x)
        hbox_layout.addWidget(self.x_r)

        self.label_ref_y = QLabel(self)
        self.label_ref_y.setText("y: ")
        self.y_r = QLineEdit(self)
        self.y_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_y)
        hbox_layout.addWidget(self.y_r)

        self.label_ref_z = QLabel(self)
        self.label_ref_z.setText("z: ")
        self.z_r = QLineEdit(self)
        self.z_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_z)
        hbox_layout.addWidget(self.z_r)

        self.layout.addLayout(hbox_layout)

        self.label_ref = QLabel(self)
        self.label_ref.setText("GPS Coordinates")
        self.layout.addWidget(self.label_ref)

        hbox_layout = QHBoxLayout()

        self.label_ref_n = QLabel(self)
        self.label_ref_n.setText("N: ")
        self.n_r = QLineEdit(self)
        self.n_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_n)
        hbox_layout.addWidget(self.n_r)

        self.label_ref_e = QLabel(self)
        self.label_ref_e.setText("E: ")
        self.e_r = QLineEdit(self)
        self.e_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_e)
        hbox_layout.addWidget(self.e_r)

        self.layout.addLayout(hbox_layout)

        self.setLayout(self.layout)

    def set_xyz(self, x, y, z):
        self.x_r.setText(str(x))
        self.y_r.setText(str(y))
        self.z_r.setText(str(z))


class PanelWidget(QWidget):
    page_changed = Signal(int)
    model_uploaded = Signal(object)

    def __init__(self):
        super().__init__()
        self.original_mesh = None

    def change_page(self, page):
        self.page_changed.emit(page)

    def set_model(self, model):
        self.file_dialog = QFileDialog(self)
        self.file_dialog.setNameFilters(["OBJ files(*.obj)"])
        self.file_dialog.setFileMode(QFileDialog.AnyFile)
        if self.file_dialog.exec_() == QFileDialog.Accepted:
            file_name = self.file_dialog.selectedFiles()[0]
            self.file_label.setText(file_name.split("/")[-1])
            hull = pyvista.read(file_name)
            self.original_mesh = hull
            self.model_uploaded.emit(hull)

class SelectReferencePointsWidget(PanelWidget):
    def __init__(self, plotter):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = plotter

        self.selected_point = []

        self.ref1_widget = ReferencePointWidget()
        self.ref2_widget = ReferencePointWidget()
        self.ref3_widget = ReferencePointWidget()

        self.file_label = QLabel(self)
        self.file_label.setText("Select file")
        self.file_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.layout.addWidget(self.file_label)
        # Add file dialog button
        self.file_dialog_button = QPushButton('Select File', self)
        self.file_dialog_button.clicked.connect(self.set_model)
        self.layout.addWidget(self.file_dialog_button)

        hbox_layout = QHBoxLayout()

        self.label_ref = QLabel(self)
        self.label_ref.setText("Currently Selected Point")
        self.layout.addWidget(self.label_ref)

        self.label_ref_x = QLabel(self)
        self.label_ref_x.setText("x: ")
        self.x_r = QLineEdit(self)
        self.x_r.setDisabled(True)
        self.x_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_x)
        hbox_layout.addWidget(self.x_r)

        self.label_ref_y = QLabel(self)
        self.label_ref_y.setText("y: ")
        self.y_r = QLineEdit(self)
        self.y_r.setDisabled(True)
        self.y_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_y)
        hbox_layout.addWidget(self.y_r)

        self.label_ref_z = QLabel(self)
        self.label_ref_z.setText("z: ")
        self.z_r = QLineEdit(self)
        self.z_r.setDisabled(True)
        self.z_r.setValidator(QDoubleValidator())
        hbox_layout.addWidget(self.label_ref_z)
        hbox_layout.addWidget(self.z_r)

        self.layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()
        self.copy_to_ref1 = QPushButton('Copy to Ref 1', self)
        self.copy_to_ref1.clicked.connect(lambda: self.ref1_widget.set_xyz(float(self.x_r.text()), float(self.y_r.text()), float(self.z_r.text())))
        hbox_layout.addWidget(self.copy_to_ref1)

        self.copy_to_ref2 = QPushButton('Copy to Ref 2', self)
        self.copy_to_ref2.clicked.connect(lambda: self.ref2_widget.set_xyz(float(self.x_r.text()), float(self.y_r.text()), float(self.z_r.text())))
        hbox_layout.addWidget(self.copy_to_ref2)

        self.copy_to_ref3 = QPushButton('Copy to Ref 3', self)
        self.copy_to_ref3.clicked.connect(lambda: self.ref3_widget.set_xyz(float(self.x_r.text()), float(self.y_r.text()), float(self.z_r.text())))
        hbox_layout.addWidget(self.copy_to_ref3)

        self.layout.addLayout(hbox_layout)

        self.label_ref1 = QLabel(self)
        self.label_ref1.setText("Reference Point 1:")
        self.label_ref1.setFont(QFont('Arial', 18, QFont.Bold))
        self.layout.addWidget(self.label_ref1)

        self.layout.addWidget(self.ref1_widget)

        self.label_ref2 = QLabel(self)
        self.label_ref2.setText("Reference Point 2:")
        self.label_ref2.setFont(QFont('Arial', 18, QFont.Bold))
        self.layout.addWidget(self.label_ref2)

        self.layout.addWidget(self.ref2_widget)

        self.label_ref3 = QLabel(self)
        self.label_ref3.setText("Reference Point 3:")
        self.label_ref3.setFont(QFont('Arial', 18, QFont.Bold))
        self.layout.addWidget(self.label_ref3)

        self.layout.addWidget(self.ref3_widget)

        self.submit_btn = QPushButton('Next', self)
        self.submit_btn.clicked.connect(self.next)
        self.layout.addWidget(self.submit_btn)

        self.model_uploaded.connect(self.set_plotter_model)

        self.setLayout(self.layout)

    def next(self):
        self.change_page(0)

    def pick_callback(self, point):
        self.selected_point = point
        self.x_r.setText(str(self.selected_point[0]))
        self.y_r.setText(str(self.selected_point[1]))
        self.z_r.setText(str(self.selected_point[2]))

    @Slot(object)
    def set_plotter_model(self, hull):
        self.plotter[0, 0].clear_actors()
        self.plotter[0, 0].add_mesh(hull, cmap="terrain", lighting=True, show_vertices=True, pickable=True)
        self.plotter[0, 0].enable_surface_point_picking(callback=self.pick_callback, show_point=True)
        self.plotter[0, 0].reset_camera()


class MainPanelWidget(PanelWidget):
    def __init__(self, plotter):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = plotter

        # settings section
        """
        self.file_label = QLabel(self)
        self.file_label.setText("Select file")
        self.file_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.layout.addWidget(self.file_label)
        # Add file dialog button
        self.file_dialog_button = QPushButton('Select File', self)
        self.file_dialog_button.clicked.connect(self.showFileDialog)
        self.layout.addWidget(self.file_dialog_button)
        
        """

        self.label_camera = QLabel(self)
        self.label_camera.setText("Drone Path Settings")
        self.label_camera.setFont(QFont('Arial', 22, QFont.Bold))
        self.layout.addWidget(self.label_camera)

        self.label_slider1 = QLabel(self)
        self.label_slider1.setText("Waypoint Offset")
        self.layout.addWidget(self.label_slider1)

        self.labeled_slider1 = QLabeledDoubleSlider(self)
        self.labeled_slider1.setRange(0.5, 10)
        self.layout.addWidget(self.labeled_slider1)

        self.label_slider2 = QLabel(self)
        self.label_slider2.setText("Overlap Amount")
        self.layout.addWidget(self.label_slider2)

        self.labeled_slider2 = QLabeledDoubleSlider(self)
        self.labeled_slider2.setRange(0, 0.5)
        self.layout.addWidget(self.labeled_slider2)

        self.label_rangeSlider1 = QLabel(self)
        self.label_rangeSlider1.setText("Altitude Range (in meters)")
        self.layout.addWidget(self.label_rangeSlider1)

        self.range_slider_one = QLabeledRangeSlider(Qt.Horizontal, self)
        self.range_slider_one.setRange(0, 60)
        self.range_slider_one.setValue([1, 30])
        self.layout.addWidget(self.range_slider_one)

        self.label_camera = QLabel(self)
        self.label_camera.setText("Camera Settings")
        self.label_camera.setFont(QFont('Arial', 22, QFont.Bold))
        self.layout.addWidget(self.label_camera)

        self.label_camera_range = QLabel(self)
        self.label_camera_range.setText("Camera max tilt range")
        self.layout.addWidget(self.label_camera_range)

        self.range_slider_camera = QLabeledRangeSlider(Qt.Horizontal, self)
        self.range_slider_camera.setRange(-180, 180)
        self.range_slider_camera.setValue([-90, 180])
        self.layout.addWidget(self.range_slider_camera)

        self.label_camera_fov = QLabel(self)
        self.label_camera_fov.setText("Camera FOV")
        self.layout.addWidget(self.label_camera_fov)

        self.slider_camera_fov = QLabeledSlider(Qt.Horizontal, self)
        self.slider_camera_fov.setRange(0, 180)
        self.slider_camera_fov.setValue(90)
        self.layout.addWidget(self.slider_camera_fov)

        hbox_layout = QHBoxLayout()
        self.camera_h_resolution_label = QLabel(self)
        self.camera_h_resolution_label.setText("Horizontal Resolution")
        hbox_layout.addWidget(self.camera_h_resolution_label)


        self.camera_v_resolution_label = QLabel(self)
        self.camera_v_resolution_label.setText("Vertical Resolution")
        hbox_layout.addWidget(self.camera_v_resolution_label)

        self.layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()

        self.camera_h_resolution = QLineEdit(self)
        self.camera_h_resolution.setValidator(QIntValidator())
        self.camera_h_resolution.setText("1920")
        hbox_layout.addWidget(self.camera_h_resolution)

        self.camera_v_resolution = QLineEdit(self)
        self.camera_v_resolution.setValidator(QIntValidator())
        self.camera_v_resolution.setText("1080")
        hbox_layout.addWidget(self.camera_v_resolution)

        self.layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()

        self.back_btn = QPushButton('Back', self)
        self.back_btn.clicked.connect(self.back)
        hbox_layout.addWidget(self.back_btn)
        self.submit_btn = QPushButton('Generate', self)
        self.submit_btn.clicked.connect(self.generate)
        hbox_layout.addWidget(self.submit_btn)

        self.layout.addLayout(hbox_layout)

        self.model_uploaded.connect(self.set_plotter_model)

        self.setLayout(self.layout)

        self.algoOutput = None

    def back(self):
        self.change_page(1)

    @Slot(object)
    def set_plotter_model(self, hull):
        self.original_mesh = hull
        self.plotter[0, 0].clear_actors()
        self.plotter[0, 0].add_text("Select Scan Section")
        self.plotter[0, 0].add_mesh_clip_box(hull, cmap="terrain", lighting=True)
        # self.plotter[0, 0].add_callback(self.generate)
        self.plotter[0, 0].reset_camera()

    def generate(self):
        if self.original_mesh is not None:
            p = SliceSurfaceAlgo(self.plotter[0, 0].box_clipped_meshes[0].extract_surface(), self.original_mesh, self.plotter[0, 1])
            p.generate_path()
