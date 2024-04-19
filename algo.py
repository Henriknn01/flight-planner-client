import math
import pyvista as pv
import numpy as np
import networkx as nx
import logging
import os

def calculate_look_at_euler_angles(camera_position, target_position):
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


def orientation_to_vector(orientation):
    pitch, yaw, _ = np.radians(orientation)
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return np.array([x, y, z])


def find_bulb_center(multi_block, y_max, z_min):
    # so plan is, do ray cast, in a straight line down middle, find the shortest ray, thats the center of bulb
    # then find the local minimum upwards thats the end of bulb
    # then find where its higher then local minimum downwards thats the end of the other side
    # thus we find center in y direction aswell.
    bulbus_hull_rays = []
    best_distance = 10000
    best_point = None
    for ray_z_level in np.arange(multi_block["Hull"].center[2], z_min, -0.3):
        ray_start = [0, y_max + 2, ray_z_level]
        ray_end = [0, multi_block["Hull"].center[1], ray_z_level]
        intersections = multi_block["Hull"].ray_trace(ray_start, ray_end)[0]
        bulb_ray = pv.Line(ray_start, ray_end)
        bulbus_hull_rays.append(bulb_ray)
        if intersections.size > 0:
            projected_point = intersections[0]
            distance = euclidean_distance(projected_point, ray_start)
            if distance < best_distance:
                best_distance = distance
                best_point = projected_point

    if logger.isEnabledFor(logging.DEBUG):
        try:
            bulbplotter = pv.Plotter()
            bulbplotter.add_mesh(multi_block["Ship"], color='lightgrey')
            # plotter.add_mesh(laser_cylinder, color='red', opacity=0.5)
            bulbplotter.add_mesh(bulbus_hull_rays[0], color="blue", line_width=5, label="Ray Segments", opacity=0.5)
            for ray in bulbus_hull_rays[1:]:
                bulbplotter.add_mesh(ray, color="blue", line_width=5, opacity=0.5)
            bulbplotter.add_points(pv.PolyData(best_point), color='red', point_size=10, opacity=0.5)
            bulbplotter.show()
        except:
            pass
    return best_point


def project_points_to_surface_cylynder_style(multi_block, horizontal_spacing, vertical_spacing, distance, z_min_filter,
                                             z_max_filter, min_distance, camera_specs, scan_height):
    x_min, x_max, y_min, y_max, z_min, z_max = multi_block["Hull"].bounds
    z_res = (multi_block["Hull"].length / (horizontal_spacing / 0.8))
    v_res = ((abs(x_min) + x_max) / (vertical_spacing / 2.2))
    logger.debug(f"z_res: {z_res}, v_res: {v_res}")

    bulb_z_height = find_bulb_center(multi_block, y_max, z_min)

    laser_cylinder = pv.CylinderStructured(center=multi_block["Hull"].center, direction=(0, 1, 0),
                                           radius=((x_max - x_min) / 2) * 2, height=multi_block["Hull"].length,
                                           theta_resolution=int(v_res), z_resolution=int(z_res))
    wave_breaker_laser = pv.Sphere(center=(multi_block["Hull"].center[0], y_max - 2, bulb_z_height[2]),
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
        "bulbus_additonals": [[0, y_max + 2, bulb_z_height[2]], [0, y_max + 2, multi_block["Hull"].center[2]]]
    }

    # Project each grid point onto the mesh surface
    for model in laser_points:
        for point in laser_points[model]:

            if model == "cylinder":
                if point[2] > scan_height:
                    continue
                else:
                    ray_start = point
                    ray_end = [multi_block["Hull"].center[0], point[1], multi_block["Hull"].center[2]]

            elif model == "wave_breaker":
                if point[1] < y_max - 1.5:
                    continue
                else:
                    ray_start = point
                    ray_end = [multi_block["Hull"].center[0], y_max - 2.5, bulb_z_height[2]]

            elif model == "bulbus_additonals":
                ray_start = point
                ray_end = [multi_block["Hull"].center[0], y_max - 2.5, bulb_z_height[2]]

            intersections = multi_block["Hull"].ray_trace(ray_start, ray_end)[0]
            ray = pv.Line(ray_start, ray_end)
            rays.append(ray)
            if intersections.size > 0:
                projected_point = intersections[0]  # Take the first intersection
                plot_points.append(projected_point)

                # Find the closest point on the mesh and get the point normal
                closest_point_id = multi_block["Hull"].find_closest_point(projected_point)
                point_normal = multi_block["Hull"].point_normals[closest_point_id]

                # Displace the projected point based on the normal and distance
                displaced_point = projected_point + point_normal * distance
                displaced_points.append(displaced_point)
                if displaced_point[2] < z_min_filter or displaced_point[2] > z_max_filter:
                    filter_delete.append(displaced_point)
                    continue
                elif point_inside_bounding_box(displaced_point, multi_block["Ship"], min_distance):
                    bound_delete.append(displaced_point)
                    continue
                else:
                    pitch, yaw, roll = calculate_look_at_euler_angles(displaced_point, projected_point)
                    if pitch > camera_specs["max_tilt_up"] or pitch < camera_specs["max_tilt_down"]:
                        tilt_delete.append(displaced_point)
                        continue
                    projected_points.append(displaced_point)
                    cameras.append([displaced_point, [pitch, yaw, roll]])

    # PyVista plotting
    if logger.isEnabledFor(logging.DEBUG):
        try:
            plotter = pv.Plotter()
            plotter.add_mesh(multi_block["Ship"], color='lightgrey')
            plotter.add_mesh(rays[0], color="blue", line_width=5, label="Ray Segments", opacity=0.5)
            for ray in rays[1:]:
                plotter.add_mesh(ray, color="blue", line_width=5, opacity=0.5)
            plotter.show()
        except:
            pass

        try:
            pointplot = pv.Plotter()
            pointplot.add_mesh(multi_block["Ship"], color='lightgrey')
            pointplot.add_points(pv.PolyData(displaced_points), color='red', point_size=5, opacity=0.5)
            pointplot.add_points(pv.PolyData(plot_points), color='green', point_size=5, opacity=0.5)
            pointplot.show()
        except:
            pass

        try:
            deleted_plotter = pv.Plotter()
            deleted_plotter.add_mesh(multi_block["Ship"], color='lightgrey')
            deleted_plotter.add_points(pv.PolyData(bound_delete), color='red', point_size=5, opacity=0.5)
            if len(filter_delete) > 0:
                deleted_plotter.add_points(pv.PolyData(filter_delete), color='orange', point_size=5, opacity=0.5)
            deleted_plotter.add_points(pv.PolyData(tilt_delete), color='green', point_size=5, opacity=0.5)
            deleted_plotter.show()
        except:
            pass

    return cameras


def point_inside_bounding_box(point, mesh, min_distance):
    ray_points = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
                  [0, -1, -1], [0, 1, 1], [1, 0, 1], [-1, 0, -1], [1, 1, 0], [-1, -1, 0]]
    for i in range(0, len(ray_points)):
        ray_direction = ray_points[i]

        ray_start = point
        ray_end = [
            point[0] + min_distance * ray_direction[0],
            point[1] + min_distance * ray_direction[1],
            point[2] + min_distance * ray_direction[2]
        ]

        intersections = mesh.ray_trace(ray_start, ray_end)[0]

        if intersections.size > 0:
            return True
    return False


def calculate_spacing_by_cam_spec(camera_specs, distance, overlap_amount):
    fov = camera_specs["fov"]
    d = distance

    # Calculate horizontal and vertical distances using tangent function
    horizontal_distance = d * math.tan(fov / 2 * math.pi / 180)
    vertical_distance = horizontal_distance * camera_specs["v_resolution"] / camera_specs["h_resolution"]

    return horizontal_distance - overlap_amount, vertical_distance - overlap_amount


def export_cameras_to_file(cameras, file_path):
    with open(file_path, 'w') as file:
        for position, euler_angles in cameras:
            file.write(
                f"PIC,{position[0]},{position[1]},{position[2]},{euler_angles[0]},{euler_angles[1]},{euler_angles[2]}\n")


def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def tsp_with_weight_calculation(points_with_rotation, x_penalty, y_factor):
    points = [sublist[0] for sublist in points_with_rotation]
    n = len(points)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            base_distance = np.linalg.norm(points[i] - points[j])
            if check_for_collisions(points[i], points[j]):
                base_distance += 100000
            if np.sign(points[i][0]) != np.sign(points[j][0]):
                base_distance += x_penalty
            y_diff = np.abs(points[i][2] - points[j][2])
            base_distance += y_factor * y_diff
            dist_mat[i, j] = dist_mat[j, i] = base_distance

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_mat[i][j])
    tsp_path = list(nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=False))

    tsp_path_with_rotation = [points_with_rotation[index] for index in tsp_path]
    return tsp_path_with_rotation




def check_for_collisions(point1, point2):
    intersections = multi_block["Ship"].ray_trace(point1, point2)[0]
    if len(intersections) > 0:
        return True
    return False

def get_start_pos_based_on_model(mesh, distance):
    bounds = mesh.bounds
    target_point = [0, bounds[3] + distance, bounds[4]]
    startpos = [target_point, [90, 0, 180]]
    return startpos


# Configure the basic settings for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the root logger
logger = logging.getLogger()

# Setup 3d models for use in a multiblock for easy handlingd
multi_block = pv.MultiBlock()
hull = pv.read('hullobj.obj')
ship = pv.read("ship.obj")
multi_block.append(hull, 'Hull')
multi_block.append(ship, 'Ship')
bounds = multi_block["Hull"].bounds

# setting up bounds of the 3d model
x_min, x_max, y_min, y_max, z_min, z_max = bounds

# Settings for camera specs of the drone, this decides what points to eliminate based on tilt and fov
camera_specs = {
    "fov": 40,
    "camera_range": 30,
    "max_tilt_up": 150,
    "max_tilt_down": -90,
    "h_resolution": 1920,
    "v_resolution": 1080
}

# Distance from the ship hull, this determines how many pictures is needed, and the resolution
drone_distance = 2
overlap_amount = 0.0  # overlap makes the points slightly closer
min_distance = 1  # distance from the ship, if inside the box of 2 meters it will be deleted

# Max height for the drone to fly
max_height = 21
min_height = 2.2

# Max heigt for the drone to inspect
scan_height = 6
scan_low = 0

# filename to export the cam positions
export_file = "sliced.txt"

# generate spacing of the drone based on camera specs, this gives optimal covrage of the pictures
horizontal_spacing, vertical_spacing = calculate_spacing_by_cam_spec(camera_specs, drone_distance, overlap_amount)
logger.debug(f"Spacing is {horizontal_spacing}, {vertical_spacing}")

point_cloud = project_points_to_surface_cylynder_style(multi_block, horizontal_spacing, vertical_spacing,
                                                       drone_distance, min_height, max_height, min_distance,
                                                       camera_specs, scan_height)

startpos = get_start_pos_based_on_model(multi_block['Hull'], 5)

point_cloud = [startpos] + point_cloud
tsp_path_with_rotation = tsp_with_weight_calculation(point_cloud, 20, 7)


export_cameras_to_file(tsp_path_with_rotation, os.path.dirname(os.path.realpath(__file__)) + "/sliced.txt")