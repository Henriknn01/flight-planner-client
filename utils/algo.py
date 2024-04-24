import heapq
import math
import pyvista as pv
import numpy as np
import networkx as nx
import logging
import scipy


"""

# Setup 3d models for use in a multiblock for easy handlingd
multi_block = pv.MultiBlock()
hull = pv.read('hullobj.obj')
ship = pv.read("ship.obj")
multi_block.append(hull, 'Hull')
multi_block.append(ship, 'Ship')
bounds = multi_block["Hull"].bounds

# setting up bounds of the 3d model
x_min, x_max, y_min, y_max, z_min, z_max = bounds



# filename to export the cam positions
export_file = "sliced.txt"

# generate spacing of the drone based on camera specs, this gives optimal covrage of the pictures
horizontal_spacing, vertical_spacing = calculate_spacing_by_cam_spec(camera_specs, drone_distance, overlap_amount)
logger.debug(f"Spacing is {horizontal_spacing}, {vertical_spacing}")

point_cloud = find_optimal_location_for_picture(multi_block, horizontal_spacing, vertical_spacing,
                                                drone_distance, min_height, max_height, min_distance,
                                                camera_specs, scan_height)

startpos = get_start_pos_based_on_model(multi_block['Hull'], 5)




#optional_nodes = get_optional_nodes_from_model(bounds)


point_cloud = [startpos] + point_cloud
tsp_path_with_rotation = tsp_with_weight_calculation(point_cloud, 5, 1, multi_block)


export_cameras_to_file(tsp_path_with_rotation, "C:\\Users\\Gardh\\Downloads\\Drone build\\sliced.txt")
"""


class SliceSurfaceAlgo:
    def __init__(self, mesh, original_mesh, plotter):
        super().__init__()
        # Configure the basic settings for logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Get the root logger
        self.logger = logging.getLogger()

        # Mesh
        self.mesh = mesh
        self.original_mesh = original_mesh

        self.plotter = plotter

        # Settings for camera specs of the drone, this decides what points to eliminate based on tilt and fov
        self.camera_specs = {
            "fov": 40,
            "camera_range": 30,
            "max_tilt_up": 150,
            "max_tilt_down": -90,
            "h_resolution": 1920,
            "v_resolution": 1080
        }

        # Distance from the ship hull, this determines how many pictures is needed, and the resolution
        self.drone_distance = 2
        self.overlap_amount = 0.1  # overlap makes the points slightly closer
        self.min_distance = 1  # distance from the ship, if inside the box of 2 meters it will be deleted

        # Max height for the drone to fly
        self.max_height = 21
        self.min_height = -10

        # Max heigt for the drone to inspect
        self.scan_height = 6
        self.scan_low = 0

    def calculate_look_at_euler_angles(self, camera_position, target_position):
        direction = np.array(camera_position) - np.array(target_position)
        direction /= np.linalg.norm(direction)

        # Blender's coordinate system: Z up, Y forward, X right
        z = np.array([0, 0, 1])
        x = np.cross(z, direction)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1, 0, 0])
        x = x.astype(np.float64)
        x /= np.linalg.norm(x)
        y = np.cross(direction, x)

        # Create rotation matrix
        rot_matrix = np.column_stack([x, y, direction])

        # Calculate Euler angles
        pitch = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
        roll = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1] ** 2 + rot_matrix[2, 2] ** 2))
        yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

        # Convert from radians to degrees and adjust range from 0 to 360
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        yaw = np.degrees(yaw)

        return pitch, yaw, roll

    def find_center_of_bulbus_hull(self, multi_block, y_max, z_min):
        # so plan is, do ray cast, in a straight line down middle, find the shortest ray, thats the center of bulb
        # then find the local minimum upwards thats the end of bulb
        # then find where its higher then local minimum downwards thats the end of the other side
        # thus we find center in y direction aswell.
        bulbus_hull_rays = []
        best_distance = 10000
        best_point = None
        for ray_z_level in np.arange(self.mesh.center[2], z_min, -0.3):
            ray_start = [0, y_max + 2, ray_z_level]
            ray_end = [0, self.mesh.center[1], ray_z_level]
            intersections = self.mesh.ray_trace(ray_start, ray_end)[0]
            bulb_ray = pv.Line(ray_start, ray_end)
            bulbus_hull_rays.append(bulb_ray)
            if intersections.size > 0:
                projected_point = intersections[0]
                distance = self.euclidean_distance(projected_point, ray_start)
                if distance < best_distance:
                    best_distance = distance
                    best_point = projected_point

        if self.logger.isEnabledFor(logging.DEBUG):
            try:
                bulbplotter = pv.Plotter()
                bulbplotter.add_mesh(self.original_mesh, color='lightgrey')
                # plotter.add_mesh(laser_cylinder, color='red', opacity=0.5)
                bulbplotter.add_mesh(bulbus_hull_rays[0], color="blue", line_width=5, label="Ray Segments", opacity=0.5)
                for ray in bulbus_hull_rays[1:]:
                    bulbplotter.add_mesh(ray, color="blue", line_width=5, opacity=0.5)
                bulbplotter.add_points(pv.PolyData(best_point), color='red', point_size=10, opacity=0.5)
                bulbplotter.show()
            except:
                pass
        return best_point

    def find_optimal_location_for_picture(self, multi_block, horizontal_spacing, vertical_spacing, distance, z_min_filter,
                                          z_max_filter, min_distance, camera_specs, scan_height):
        x_min, x_max, y_min, y_max, z_min, z_max = self.mesh.bounds
        z_res = (self.mesh.length / (horizontal_spacing / 0.8))
        v_res = ((abs(x_min) + x_max) / (vertical_spacing / 2.2))
        self.logger.debug(f"z_res: {z_res}, v_res: {v_res}")

        bulb_z_height = self.find_center_of_bulbus_hull(multi_block, y_max, z_min)

        laser_cylinder = pv.CylinderStructured(center=self.mesh.center, direction=(0, 1, 0),
                                               radius=((x_max - x_min) / 2) * 2, height=self.mesh.length,
                                               theta_resolution=int(v_res), z_resolution=int(z_res))
        wave_breaker_laser = pv.Sphere(center=(self.mesh.center[0], y_max - 2, bulb_z_height[2]),
                                       direction=(1, 0, 0), radius=10, theta_resolution=5, phi_resolution=5)

        projected_points = []
        cameras = []
        plot_points = []
        displaced_points = []
        tilt_delete = []
        bound_delete = []
        filter_delete = []
        rays = []

        laser_points = {
            "cylinder": laser_cylinder.points,
            "wave_breaker": wave_breaker_laser.points,
            "bulbus_additonals": [[0, y_max + 2, bulb_z_height[2]], [0, y_max + 2, self.mesh.center[2]]]
        }

        # Project each grid point onto the mesh surface
        for model in laser_points:
            for point in laser_points[model]:

                if model == "cylinder":
                    if point[2] > scan_height:
                        continue
                    else:
                        ray_start = point
                        ray_end = [self.mesh.center[0], point[1], self.mesh.center[2]]

                elif model == "wave_breaker":
                    if point[1] < y_max - 1.5:
                        continue
                    else:
                        ray_start = point
                        ray_end = [self.mesh.center[0], y_max - 2.5, bulb_z_height[2]]

                elif model == "bulbus_additonals":
                    ray_start = point
                    ray_end = [self.mesh.center[0], y_max - 2.5, bulb_z_height[2]]

                intersections = self.mesh.ray_trace(ray_start, ray_end)[0]
                ray = pv.Line(ray_start, ray_end)
                rays.append(ray)
                if intersections.size > 0:
                    projected_point = intersections[0]  # Take the first intersection
                    plot_points.append(projected_point)

                    # Find the closest point on the mesh and get the point normal
                    closest_point_id = self.mesh.find_closest_point(projected_point)
                    point_normal = self.mesh.point_normals[closest_point_id]

                    # Displace the projected point based on the normal and distance
                    displaced_point = projected_point + point_normal * distance
                    displaced_points.append(displaced_point)
                    if displaced_point[2] < z_min_filter or displaced_point[2] > z_max_filter:
                        filter_delete.append(displaced_point)
                        continue
                    elif self.check_for_camerapos_to_close_to_3dmodel(displaced_point, self.original_mesh, min_distance):
                        bound_delete.append(displaced_point)
                        continue
                    else:
                        pitch, yaw, roll = self.calculate_look_at_euler_angles(displaced_point, projected_point)
                        if pitch > camera_specs["max_tilt_up"] or pitch < camera_specs["max_tilt_down"]:
                            tilt_delete.append(displaced_point)
                            continue
                        projected_points.append(displaced_point)
                        cameras.append([displaced_point, [pitch, yaw, roll]])

        # PyVista plotting
        if self.logger.isEnabledFor(logging.DEBUG):
            try:
                plotter = pv.Plotter()
                plotter.add_mesh(self.original_mesh, color='lightgrey')
                plotter.add_mesh(rays[0], color="blue", line_width=5, label="Ray Segments", opacity=0.5)
                for ray in rays[1:]:
                    plotter.add_mesh(ray, color="blue", line_width=5, opacity=0.5)
                plotter.show()
            except:
                pass

            try:
                pointplot = pv.Plotter()
                pointplot.add_mesh(self.original_mesh, color='lightgrey')
                pointplot.add_points(pv.PolyData(displaced_points), color='red', point_size=5, opacity=0.5)
                pointplot.add_points(pv.PolyData(plot_points), color='green', point_size=5, opacity=0.5)
                pointplot.show()
            except:
                pass

            try:
                deleted_plotter = pv.Plotter()
                deleted_plotter.add_mesh(self.original_mesh, color='lightgrey')
                deleted_plotter.add_points(pv.PolyData(bound_delete), color='red', point_size=5, opacity=0.5)
                if len(filter_delete) > 0:
                    deleted_plotter.add_points(pv.PolyData(filter_delete), color='orange', point_size=5, opacity=0.5)
                deleted_plotter.add_points(pv.PolyData(tilt_delete), color='green', point_size=5, opacity=0.5)
                deleted_plotter.show()
            except:
                pass

        return cameras

    def check_for_camerapos_to_close_to_3dmodel(self, point, mesh, min_distance):
        # This checks for collision using raytrace in a star formation, This is a fast method to check for any intersections
        # where we shoot a ray out in most directions and checks for a intersect in the min distance value
        # It returns true if there is an intersections, and false if there is no intersection
        ray_points = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
                      [0, -1, -1], [0, 1, 1], [1, 0, 1], [-1, 0, -1], [1, 1, 0], [-1, -1, 0]]
        # ray points gives us the direction of travel for the ray
        for i in range(0, len(ray_points)):
            ray_direction = ray_points[i]
            # loops over all directions

            ray_start = point
            ray_end = [
                point[0] + min_distance * ray_direction[0],
                point[1] + min_distance * ray_direction[1],
                point[2] + min_distance * ray_direction[2]
            ]
            # scale all the directions with the distance

            intersections = mesh.ray_trace(ray_start, ray_end)[0]
            # checks for intersections

            if intersections.size > 0:
                return True
        return False

    def calculate_spacing_by_cam_spec(self, camera_specs, distance, overlap_amount):
        # Calculates the horizontal and vertical distances based on the camera specs, this gives us the
        # Cylynder stucture points to raytrace from
        fov = camera_specs["fov"]
        d = distance

        # Calculate horizontal and vertical distances using tangent function
        horizontal_distance = d * math.tan(fov / 2 * math.pi / 180)
        vertical_distance = horizontal_distance * camera_specs["v_resolution"] / camera_specs["h_resolution"]

        return horizontal_distance - overlap_amount, vertical_distance - overlap_amount

    def export_cameras_to_file(self, cameras, file_path):
        # Exports the list with pos and angles to a txt file
        with open(file_path, 'w') as file:
            for position, euler_angles in cameras:
                file.write(
                    f"PIC,{position[0]},{position[1]},{position[2]},{euler_angles[0]},{euler_angles[1]},{euler_angles[2]}\n")

    def euclidean_distance(self, point1, point2):
        # do normal lingalg to get distance between points
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    def find_closest_node(self, target, nodes_with_rotation):
        """Find the closest node to the given target point."""
        closest_index = None
        min_distance = float('inf')
        for index, (position, _) in enumerate(nodes_with_rotation):
            distance = self.euclidean_distance(target, position)
            if distance < min_distance:
                min_distance = distance
                closest_index = index
        return closest_index

    def tsp_with_weight_calculation(self, points_with_rotation, x_penalty, z_factor, multi_block, start_point=None):
        visited = set()  # Keep track of visited nodes
        path_indices = []
        collisions = []
        # Determine the starting node based on the provided start_point
        if start_point:
            start_index = self.find_closest_node(start_point[0], points_with_rotation)
        else:
            start_index = 0  # Default to the first node if no start point is provided

        path_indices.append(start_index)
        visited.add(start_index)
        rays = []
        while len(visited) < len(points_with_rotation):
            last_index = path_indices[-1]
            shortest_distance = float('inf')
            next_index = None
            for i, (position, _) in enumerate(points_with_rotation):
                if i not in visited:
                    distance = self.euclidean_distance(points_with_rotation[last_index][0], position)

                    # Check for Y-axis crossing
                    if (points_with_rotation[last_index][0][0] < 0 < position[0]) or (
                            position[0] < 0 < points_with_rotation[last_index][0][0]):
                        distance += x_penalty  # Add penalty for crossing the Y-axis
                    z_axis_move = abs(points_with_rotation[last_index][0][2] - points_with_rotation[i][0][2])
                    distance += z_axis_move * z_factor
                    if distance < shortest_distance:
                        shortest_distance = distance
                        next_index = i

            path_indices.append(next_index)
            visited.add(next_index)

            point, collision = self.check_for_collisions(points_with_rotation[last_index][0],
                                                    points_with_rotation[next_index][0], multi_block)
            if collision:
                print(
                    f"collision betwween: {last_index} and {next_index}, which will be inserted as {len(path_indices)}")
                insert_point = len(path_indices) - 1
                collisions.append([insert_point, point])

        path = [points_with_rotation[index] for index in path_indices]
        i = 0
        for coll in collisions:
            path.insert((coll[0]) + i, coll[1])
            print(f"added: {coll[1]} at location {coll[0] + i}")
            i += 1

        try:
            first_elements = [sublist[0] for sublist in path if sublist]
            for i in range(len(first_elements) - 1):
                ray = pv.Line(first_elements[i], first_elements[i + 1])
                rays.append(ray)
            self.plotter.clear_actors()
            self.plotter.add_text("Generated Flight Path")
            self.plotter.add_mesh(self.original_mesh, color='lightgrey')
            self.plotter.add_points(pv.PolyData(first_elements), color='red', point_size=5, opacity=0.5)
            for ray1 in rays:
                self.plotter.add_mesh(ray1, color="blue", line_width=5, opacity=0.5)
        except:
            pass

        return path

    def check_for_collisions(self, point1, point2, multi_block):
        # Chjeck if there are any intersections between two points using ray tracing
        # Returns true if there is an intersections, and false if not.
        intersections = self.original_mesh.ray_trace(point1, point2)[0]
        if len(intersections) > 0:
            xm = (point1[0] + point2[0]) / 2
            ym = (point1[1] + point2[1]) / 2
            zm = (point1[2] + point2[2]) / 2

            p1 = np.array([xm, ym, zm])
            p2 = np.array(self.original_mesh.center)
            # Calculate the direction vector from p2 to p1
            direction_vector = p1 - p2

            # Normalize the direction vector
            norm = np.linalg.norm(direction_vector)
            if norm == 0:
                raise ValueError("The two points are identical, cannot determine direction.")
            normalized_vector = direction_vector / norm

            # Displace the point p1 by delta in the direction of the normalized vector
            displaced_point = p1 + 5 * normalized_vector
            displaced_point = [displaced_point, [0, 0, 180]]
            return displaced_point, True
        return None, False

    def get_start_pos_based_on_model(self, mesh, distance):
        # Calculate starting position based on the models bounds
        # the position is offset byu a set distance infron of the model in the y axis
        bounds = mesh.bounds
        # Defined bounding box
        target_point = [0, bounds[3] + distance, bounds[4]]
        # set point where starting point should be
        startpos = [target_point, [90, 0, 180]]
        # Startpos loccation and a start rotation is set
        return startpos

    def get_optional_nodes_from_model(self, bounds):
        rotation = [90, 0, 0]  # Use a list for rotation
        # Create linear spaces for x and y dimensions
        # x_min, x_max, y_min, y_max, z_min, z_max = bounds
        x = np.linspace(bounds[0], bounds[1], 3)
        y = np.linspace(bounds[2], bounds[3], 10)

        # Z coordinate is constant at the bottom of the bounding box
        z = bounds[4] - 1.5

        # Create a meshgrid from x and y, and repeat z for each point
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z)

        # Stack the coordinates to create a 3D array of points
        points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

        # Create an array of rotations, repeated for each point
        rotations = np.tile(rotation, (points.shape[0], 1))

        # Combine points and rotations in the required nested list format
        output = []
        for point, rotation in zip(points, rotations):
            output.append([point.tolist(), rotation.tolist()])

        return output

    # implement method that outputs depth images to specified folder
    # follow documentation: https://docs.pyvista.org/version/stable/examples/02-plot/image_depth.html
    def get_depth_map(self, cpos):
        return True

    def generate_path(self):
        bounds = self.mesh.bounds

        # setting up bounds of the 3d model
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        # filename to export the cam positions
        # export_file = "sliced.txt"

        # generate spacing of the drone based on camera specs, this gives optimal covrage of the pictures
        horizontal_spacing, vertical_spacing = self.calculate_spacing_by_cam_spec(self.camera_specs, self.drone_distance, self.overlap_amount)
        self.logger.debug(f"Spacing is {horizontal_spacing}, {vertical_spacing}")

        point_cloud = self.find_optimal_location_for_picture(self.mesh, horizontal_spacing, vertical_spacing,
                                                             self.drone_distance, self.min_height, self.max_height,
                                                             self.min_distance, self.camera_specs, self.scan_height)

        startpos = self.get_start_pos_based_on_model(self.mesh, 5)

        # optional_nodes = get_optional_nodes_from_model(bounds)

        point_cloud = [startpos] + point_cloud
        tsp_path_with_rotation = self.tsp_with_weight_calculation(point_cloud, 5, 1, self.mesh)

        # self.export_cameras_to_file(tsp_path_with_rotation, "C:\\Users\\Gardh\\Downloads\\Drone build\\sliced.txt")
