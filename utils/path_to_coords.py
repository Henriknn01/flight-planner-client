from typing import Tuple
from math import sin, cos, sqrt, asin, radians, acos, degrees, atan2
import numpy as np
from scipy.optimize import minimize, least_squares

import numpy

"""
Translate a path to coordinates

Use reference points on the ship hull paired with coordinates to translate the path coordinates into real world coordinates.
"""


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        # set initial values for x, y, and z and set up variables
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # set values
        self.set_x(x)
        self.set_y(y)
        self.set_z(z)

    def __str__(self):
        return f"[x: {self.x}, y: {self.y}, z: {self.z}]"

    def __add__(self, other: 'Vector3') -> 'Vector3':
        """ Adds two vectors together. """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        """ Subtracts a vector from another vector. """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: (int | float)) -> 'Vector3':
        """ Multiplies a vector by a scalar. """
        if not isinstance(scalar, (int | float)):
            raise TypeError(f"{scalar} is not an int or float")
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: (int | float)) -> 'Vector3':
        """ Divides a vector by a scalar. """
        if not isinstance(scalar, (int | float)):
            raise TypeError(f"{scalar} is not an int or float")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __eq___(self, other: 'Vector3') -> bool:
        """
        Checks if two vectors are equal.
        :returns: true if they are equal, false otherwise.
        """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: 'Vector3') -> bool:
        """
        Checks if two vectors are not equal.
        :returns: true if they are not equal, false otherwise.
        """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return self.x != other.x or self.y != other.y or self.z != other.z

    def set_x(self, x: float):
        """
        Set the x coordinate of the point.
        :returns: the new x coordinate of the point.
        """
        self.x = x

    def get_x(self) -> float:
        """
        Get the x coordinate of the point.
        :returns: the x coordinate of the point.
        """
        return self.x

    def set_y(self, y: float):
        """
        Set the y coordinate of the point.
        :returns: the new y coordinate of the point.
        """
        self.y = y

    def get_y(self) -> float:
        """
        Get the y coordinate of the point.
        :returns: the y coordinate of the point.
        """
        return self.y

    def set_z(self, z: float):
        """
        Set the z coordinate of the point.
        :returns: the x, y and z coordinates with the new z coordinate.
        """
        self.z = z

    def get_z(self) -> float:
        """
        Get the z coordinate of the point.
        :returns: the z coordinate of the point.
        """
        return self.z

    def set(self, x: float, y: float, z: float):
        """
        Set x, y, z of the point.
        :returns: the new x, y and z coordinates of the point.
        """
        self.set_x(x)
        self.set_y(y)
        self.set_z(z)

    def get(self) -> Tuple[float, float, float]:
        """
        Get the x, y, z of the point.
        :returns: x, y and z coordinates.
        """
        return self.x, self.y, self.z

    def get_distance(self, other: 'Vector3') -> float:
        """
        :returns: the distance between this vector and another vector.
        """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def get_magnitude(self) -> float:
        """
        :returns: the magnitude of the vector.
        """
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def get_angle(self, other: 'Vector3') -> float:
        """
        :returns: the angle between this vector and another vector.
        """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return degrees(atan2(other.get_z() - self.get_z(), other.get_x() - self.get_x()))


class GPSCoordinate:
    def __init__(self, lat: float, long: float):
        # set initial values for lat and long and set up variables
        self.lat = 0.0
        self.long = 0.0

        # set the values equal to the input
        self.set_lat(lat)
        self.set_long(long)

    def __str__(self):
        return f"[lat: {self.lat}, long: {self.long}]"

    def __sub__(self, other: 'GPSCoordinate') -> 'GPSCoordinate':
        if not isinstance(other, GPSCoordinate):
            raise TypeError(f"{other} is not a GPSCoordinate object")
        return GPSCoordinate(self.lat - other.lat, self.long - other.long)

    def __add__(self, other: 'GPSCoordinate') -> 'GPSCoordinate':
        if not isinstance(other, GPSCoordinate):
            raise TypeError(f"{other} is not a GPSCoordinate object")
        return GPSCoordinate(self.lat + other.lat, self.long + other.long)

    def set_lat(self, lat: float):
        """Set the latitude of the point."""
        if lat > 90 or lat < -90:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        self.lat = lat

    def get_lat(self) -> float:
        """
        Get the latitude of the point.
        :returns: the latitude of the point.
        """
        return self.lat

    def set_long(self, long: float):
        """Set the longitude of the point."""
        if long > 180 or long < -180:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        self.long = long

    def get_long(self) -> float:
        """
        Get the longitude of the point.
        :returns: the longitude of the point.
        """
        return self.long

    def set(self, lat: float, long: float):
        """Set the latitude and longitude of the point."""
        self.set_lat(lat)
        self.set_long(long)

    def get(self) -> Tuple[float, float]:
        """
        Get the latitude and longitude of the point.
        :returns: the latitude and longitude of the point.
        """
        return self.lat, self.long

    def get_distance(self, gps_coordinate: 'GPSCoordinate', unit_modifier: float = 1.0) -> float:
        """
        Get the distance between two GPS coordinates in km using the haversine formula.
        Change unit_modifier to change between different distance units, example: 1000 to return distance in meters.

        :param gps_coordinate: GPS coordinate
        :param unit_modifier: modifier for return unit, 1 for km, 1000 for meters.

        :returns: Distance between two coordinates in kilometers (if other unit is not specified by unit_modifier).
        """
        if not isinstance(gps_coordinate, GPSCoordinate):
            raise TypeError("Invalid type for gps_coordinate. Expected GPSCoordinate.")
        R = 6371.0 * unit_modifier  # earths approximate radius in km * unit_modifier
        lat1 = radians(self.get_lat())  # θ₁
        long1 = radians(self.get_long())  # φ₁
        lat2 = radians(gps_coordinate.get_lat())  # θ₂
        long2 = radians(gps_coordinate.get_long())  # φ₂
        # get distance using the haversine formula
        distance = 2*R*asin(sqrt((sin((lat2 - lat1)/2))**2) + cos(lat1) * cos(lat2) * (sin((long2 - long1)/2))**2)
        return distance


class BasePoint:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, lat: float = 0.0, long: float = 0.0, altitude: float = 0.0):
        super().__init__()
        self.position = Vector3(x, y, z)  # virtual position of the point.
        self.gps_position = GPSCoordinate(lat, long)  # gnss position of the point.
        self.altitude = altitude  # altitude of the point.

    def __str__(self):
        return f"{self.position}, {self.gps_position}, alt: {self.altitude}"

    def set_x(self, x: float):
        self.position.set_x(x)

    def set_y(self, y: float):
        self.position.set_y(y)

    def set_z(self, z: float):
        self.position.set_z(z)

    def set_lat(self, lat: float):
        self.gps_position.set_lat(lat)

    def set_long(self, long: float):
        self.gps_position.set_long(long)

    def set_altitude(self, altitude: float):
        if altitude < 0:
            raise ValueError("Altitude must be greater than or equal to zero.")
        self.altitude = altitude

    @staticmethod
    def gps_to_cartesian(lat: float, long: float, altitude: float = 0.0) -> Tuple[float, float, float]:
        """
        Convert GPS coordinates to cartesian coordinates.
        AI Disclaimer: This function is mostly AI generated.

        :returns: Cartesian coordinates of the GPS coordinates.
        """
        if not isinstance(lat, float) or not isinstance(long, float) or not isinstance(altitude, float):
            raise TypeError(f"Paramaters must be of type float")
        lat, long = np.deg2rad(lat), np.deg2rad(long)
        R = 6371.0
        x = (R + altitude) * cos(lat) * cos(long)
        y = (R + altitude) * cos(lat) * sin(long)
        z = (R + altitude) * sin(lat)
        return x, y, z

    @staticmethod
    def trilateration(x, y, z, dist):
        """
        Trilateration algorithm to find unknown point with known distances to reference points.
        AI Disclaimer: This function is mostly AI generated.

        :returns: Lat and long coordinates of the unknown point.
        """

        def residuals(v):
            return np.sqrt((x - v[0]) ** 2 + (y - v[1]) ** 2 + (z - v[2]) ** 2) - dist

        res = least_squares(residuals, (0.0, 0.0, 0.0))
        return res.x


class ReferencePoint(BasePoint):
    def __init__(self, x, y, z, lat, long, altitude):
        super().__init__(x, y, z, lat, long, altitude)

    def get_conversion_factor(self, other: 'ReferencePoint') -> float:
        """
        Gets the conversion factor between virtual distances and real distances.
        It gets the conversion factor by getting the distance between two points in both km and virtual distance.
        The distance in km is then divided by the virtual distance to get the conversion factor.

        :param other: A ReferencePoint object to base the conversion factor of.
        :return: the conversion factor between virtual distances and real distances.
        """
        if not isinstance(other, ReferencePoint):
            raise TypeError(f"{other} is not a ReferencePoint object")
        return self.gps_position.get_distance(other.gps_position) / self.position.get_distance(other.position)


class PathPoint(BasePoint):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, lat=0.0, long=0.0, altitude=0.0)
        # TODO: add direction/orientation

    def get_altitude(self, ref1: ReferencePoint, ref2: ReferencePoint) -> float:
        # TODO: implement mehtod that takes into account the hull offset from the ground and the height of the ship.
        # Note: This is just a dummy method and wil return false altitude
        # This method does not take into account if the bottom of the ship has a negative y value,
        # if so then all altitudes will be wrong.
        conversion_factor = ref1.get_conversion_factor(ref2)*1000  # *1000 to convert units from km to m
        return abs(self.position.get_y())*conversion_factor

    def get_gps_pos_from_reference_points(self, ref1: ReferencePoint, ref2: ReferencePoint, ref3: ReferencePoint) -> GPSCoordinate:
        """
        Get a GPS coordinate from three reference points.

        AI Disclaimer: This function contains AI generated code.

        :param ref1: Reference point 1
        :param ref2: Reference point 2
        :param ref3: Reference point 3
        :return: GPS coordinate of the PathPoint derived from reference points.
        """
        if not isinstance(ref1, ReferencePoint) or not isinstance(ref2, ReferencePoint) or not isinstance(ref3, ReferencePoint):
            raise TypeError(f"One or more of the reference points are not a ReferencePoint object")

        conversion_factor = ref1.get_conversion_factor(ref2)  # create conversion factor

        R = 6371.0  # Earths radius

        # Map virtual distances to real distances
        distance_to_ref1 = self.position.get_distance(ref1.position) * conversion_factor
        distance_to_ref2 = self.position.get_distance(ref2.position) * conversion_factor
        distance_to_ref3 = self.position.get_distance(ref3.position) * conversion_factor

        # List containing the distances from the unknown point to the reference points
        distances = [distance_to_ref1, distance_to_ref2, distance_to_ref3]

        # List containing the gps coordinates of the reference points
        gps_positions = [ref1.gps_position.get(), ref2.gps_position.get(), ref3.gps_position.get()]

        # Translate the gps coordinates to cartesian coordinates
        cartesian = np.array([self.gps_to_cartesian(*pos) for pos in gps_positions])

        # Use trilateration to find the position of the unknown point
        x, y, z = self.trilateration(*cartesian.T, dist=distances)

        lat = np.arcsin(z / R)  # convert back to latitude
        long = np.arctan2(y, x)  # convert back to longitude
        lat, long = np.rad2deg((lat, long))  # Convert from radians to degrees
        return GPSCoordinate(lat, long)
