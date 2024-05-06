from typing import Tuple
from math import sin, cos, sqrt, asin, radians
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
        """ Checks if two vectors are equal. Returns true if they are equal, false otherwise. """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: 'Vector3') -> bool:
        """ Checks if two vectors are not equal. Returns true if they are not equal, false otherwise. """
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return self.x != other.x or self.y != other.y or self.z != other.z

    def set_x(self, x: float):
        """Set the x coordinate of the point. Returns the new x coordinate of the point."""
        self.x = x

    def get_x(self) -> float:
        """Get the x coordinate of the point. Returns the x coordinate of the point."""
        return self.x

    def set_y(self, y: float):
        """Set the y coordinate of the point. Returns the new y coordinate of the point."""
        self.y = y

    def get_y(self) -> float:
        """Get the y coordinate of the point. Returns the y coordinate of the point."""
        return self.y

    def set_z(self, z: float):
        """Set the z coordinate of the point. Returns the x, y and z coordinates with the new z coordinate."""
        self.z = z

    def get_z(self) -> float:
        """Get the z coordinate of the point. Returns the z coordinate of the point."""
        return self.z

    def set(self, x: float, y: float, z: float):
        """Set x, y, z of the point. Returns the new x, y and z coordinates of the point."""
        self.set_x(x)
        self.set_y(y)
        self.set_z(z)

    def get(self) -> Tuple[float, float, float]:
        """Get the x, y, z of the point. Return x, y and z coordinates."""
        return self.x, self.y, self.z

    def get_distance(self, other: 'Vector3') -> float:
        """ Returns the distance between this vector and another vector."""
        if not isinstance(other, Vector3):
            raise TypeError(f"{other} is not a Vector3 object")
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


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

    def set_lat(self, lat: float):
        """Set the latitude of the point."""
        if lat > 90 or lat < -90:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        self.lat = lat

    def get_lat(self) -> float:
        """Get the latitude of the point. Returns the latitude of the point."""
        return self.lat

    def set_long(self, long: float):
        """Set the longitude of the point."""
        if long > 180 or long < -180:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        self.long = long

    def get_long(self) -> float:
        """Get the longitude of the point. Returns the longitude of the point."""
        return self.long

    def set(self, lat: float, long: float):
        """Set the latitude and longitude of the point."""
        self.set_lat(lat)
        self.set_long(long)

    def get(self) -> Tuple[float, float]:
        """Get the latitude and longitude of the point. Returns the latitude and longitude of the point."""
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


class ReferencePoint(BasePoint):
    def __init__(self, x, y, z, lat, long, altitude):
        super().__init__(x, y, z, lat, long, altitude)


class PathPoint(BasePoint):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, lat=0.0, long=0.0, altitude=0.0)

    def get_lat_long_from_reference_points(self, ref1: ReferencePoint, ref2: ReferencePoint, ref3: ReferencePoint):
        # Needs to map virtual distance to real world distance, then use triangulation to find the gps coordinate of the position.
        pass
