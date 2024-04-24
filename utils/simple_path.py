import pyvista
"""
The simple path algo is a test algorithm developed while waiting for the final version of the more complex algorithm.
The algorithm slices the given mesh in the z axis with a given amount of slices along the z axis.
Once the slices have been created it will create distributed points along the path of each point at any given slice.
The points will have an offset from the path of the slices along the z axis by a given amount. 
This offset will act as the distance between the drone and the hull of the ship its scanning.
Each point represents a picture being taken of the ship hull.
"""


class SimplePath:
    def __init__(self, mesh):
        super().__init__()
        self.line_width = 4
        self.mesh = mesh

    def generate_path(self, plotter, n_v_slices=3, n_h_slices=3, offset_x=1.2, offset_y=1.1, offset_z=0.95):
        plotter.clear_actors()
        plotter.add_text("Generated Flight Path")
        multi_block = pyvista.MultiBlock()
        slices = self.mesh.slice_along_axis(n=n_h_slices, axis=1, generate_triangles=True)
        slices = slices.slice_along_axis(n=n_v_slices, axis=2, generate_triangles=True)
        multi_block.append(slices)
        multi_block.append(self.mesh)
        for name in slices.keys():
            block = slices[name]
            for b in block:
                b.points[:, 0] *= offset_x
                b.points[:, 1] *= offset_y
                b.points[:, 2] *= offset_z
        plotter.add_mesh(slices, style="points", point_size=5, render_points_as_spheres=True, color="red")
        plotter.add_mesh(self.mesh)
        # plotter.show()
