"""
Electrodes defines any type of sources and receivers required for a survey.
"""
# Copyright 2018-2021 The emg3d Developers.
#
# This file is part of emg3d.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

from copy import deepcopy

import numpy as np
from scipy.special import sindg, cosdg

__all__ = ['Electrode', 'Point', 'Dipole']


# List of electrodes
ELECTRODE_LIST = {}


def register_electrode(func):
    """Decorator to register sources and receivers.

    Adds them to both, the list ``emg3d.electrodes.ELECTRODE_LIST``, and
    ``emg3d.electrodes.__all__``.

    """
    ELECTRODE_LIST[func.__name__] = func
    __all__.append(func.__name__)
    return func


# BASE ELECTRODE TYPES
class Electrode:
    """TODO

    Points must be in the form of::

        [[x0, y0, z0], [...], [xN, yN, zN]]: (x, 3)

    Coordinates can be different, it is what the given class uses.
    If not provided, it is set to coordinates.

    """

    _serialize = {'coordinates'}

    def __init__(self, points, coordinates=None):
        """Initiate an electrode."""

        points = np.atleast_2d(points)

        if not (points.ndim == 2 and points.shape[1] == 3):
            raise ValueError(
                "`points` must be of shape (x, 3), provided: "
                f"{points.shape}"
            )

        self._points = np.asarray(points, dtype=float)

        if coordinates is None:
            self._coordinates = points
        else:
            self._coordinates = coordinates

    def __eq__(self, electrode):
        """Compare two electrodes."""

        # Check if same Type.
        equal = self.__class__.__name__ == electrode.__class__.__name__

        # Check input.
        if equal:
            for name in self._serialize:
                equal *= np.allclose(getattr(self, name),
                                     getattr(electrode, name))

        return bool(equal)

    def copy(self):
        """Return a copy of the Survey."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        out = {
            '__class__': self.__class__.__name__,
            **{prop: getattr(self, prop) for prop in self._serialize},
        }
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        return cls(**{k: v for k, v in inp.items() if k != '__class__'})

    @property
    def points(self):
        return self._points

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def xtype(self):
        if not hasattr(self, '_xtype'):
            if 'Current' in self.__class__.__name__:
                self._xtype = 'current'
            elif ('Flux' in self.__class__.__name__ or
                  'Loop' in self.__class__.__name__):
                self._xtype = 'flux'
            elif 'Magnetic' in self.__class__.__name__:
                self._xtype = 'magnetic'
            else:  # Default
                self._xtype = 'electric'
        return self._xtype

    @property
    def center(self):
        if not hasattr(self, '_center'):
            self._center = self.points.mean(axis=0)
        return self._center

    @property
    def length(self):
        if not hasattr(self, '_length'):
            self._length = np.linalg.norm(
                    np.diff(self.points, axis=0), axis=1).sum()
        return self._length


class Point(Electrode):
    """TODO"""

    def __init__(self, coordinates):
        """Initiate an electric point."""

        if len(coordinates) != 5:
            raise ValueError(
                "Point coordinates are wrong defined. They must be "
                "defined as (x, y, z, azimuth, elevation)."
                f"Provided coordinates: {coordinates}."
            )
        coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()
        super().__init__(points=coordinates[:3], coordinates=coordinates)

    def __repr__(self):
        s0 = f"{self.__class__.__name__}: \n"
        s1 = (f"    x={self.center[0]:,.1f} m, "
              f"y={self.center[1]:,.1f} m, z={self.center[2]:,.1f} m, ")
        s2 = f"θ={self.azimuth:.1f}°, φ={self.elevation:.1f}°"
        return s0 + s1 + s2 if len(s1+s2) < 80 else s0 + s1 + "\n    " + s2

    @property
    def azimuth(self):
        return self._coordinates[3]

    @property
    def elevation(self):
        return self._coordinates[4]


class Dipole(Electrode):
    """TODO

    These are dipoles, this format (x, y, z, azimuth, elevation), length
    is changed to ([x0, y0, z0], [x1, y1, z1])

    TODO either (x, y, z, azimuth, elevation), length or
                (x0, x1, y0, y1, z0, z1) or
                ([x0, y0, z0], [x1, y1, z1])

    """

    def __init__(self, coordinates, length):
        """Initiate an electric dipole."""

        coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()

        is_point = coordinates.size == 5
        is_points_a = coordinates.ndim == 2 and coordinates.shape[1] == 3
        is_points_b = coordinates.ndim == 1 and coordinates.shape[0] == 6

        if not is_point and not is_points_a and not is_points_b:
            raise ValueError(
                "`coordinates` must be of shape (3,), (5,) (6,), or (2, 3), "
                f"provided: {coordinates.shape}"
            )

        # Check size => finite or point dipole?
        if coordinates.size == 5:

            self._serialize = {'length'} | self._serialize

            # Get lengths in each direction.
            if length is None:
                length = 1.0

            # Get the two separate electrodes.
            if self.xtype == 'magnetic':
                # square loop of area corresponding to length
                points = _point_to_square_loop(coordinates, length)
            else:
                points = _point_to_dipole(coordinates, length)

            self._length = length

        elif coordinates.size == 6:

            if length is not None:
                raise ValueError("length must be zero for this format")

            if coordinates.ndim == 1:
                points = np.array([coordinates[::2], coordinates[1::2]])
                coordinates = coordinates

            else:
                points = coordinates
                coordinates = None

            if self.xtype == 'magnetic':
                azimuth, elevation, length = _dipole_to_point(points)
                center = tuple(np.sum(points, 0)/2)
                coo = (*center, azimuth, elevation)
                points = _point_to_square_loop(coo, length)

            # Ensure the two poles are distinct.
            if np.allclose(points[0, :], points[1, :]):
                raise ValueError(
                    "The two poles are identical, use the format "
                    "(x, y, z, azimuth, elevation) instead. "
                    f"Provided coordinates: {coordinates}."
                )

            if length is not None:
                raise ValueError("No length with this format")

        else:
            raise ValueError(
                "Dipole coordinates are wrong defined. They must be "
                "defined either as a point, (x, y, z, azimuth, elevation), or "
                "as two poles, (x0, x1, y0, y1, z0, z1) or "
                "[(x0, y0, z0), (x1, y1, z1)] , all floats. "
                f"Provided coordinates: {coordinates}."
            )

        super().__init__(points=points, coordinates=coordinates)

    def __repr__(self):
        s0 = (f"{self.__class__.__name__}: "
              f"{self._repr_add if hasattr(self, '_repr_add') else ''}\n")
        if self._coordinates.size == 5:
            s1 = (f"    center={{{self.center[0]:,.1f}; "
                  f"{self.center[1]:,.1f}; {self.center[2]:,.1f}}} m; ")
            s2 = (f"θ={self.azimuth:.1f}°, φ={self.elevation:.1f}°; "
                  f"l={self.length:,.1f} m")
        else:
            s1 = (f"    e1={{{self.points[0, 0]:,.1f}; "
                  f"{self.points[0, 1]:,.1f}; {self.points[0, 2]:,.1f}}} m; ")
            s2 = (f"e2={{{self.points[1, 0]:,.1f}; "
                  f"{self.points[1, 1]:,.1f}; {self.points[1, 2]:,.1f}}} m")
        return s0 + s1 + s2 if len(s1+s2) < 80 else s0 + s1 + "\n    " + s2

    @property
    def azimuth(self):
        if not hasattr(self, '_azimuth'):
            if self._points.shape[0] == 5:
                self._azimuth = self._coordinates[3]
                self._elevation = self._coordinates[4]
            else:
                out = _dipole_to_point(self._points)
                self._azimuth, self._elevation, _ = out
        return self._azimuth

    @property
    def elevation(self):
        if not hasattr(self, '_elevation'):
            _ = self.azimuth  # Sets azimuth and elevation
        return self._elevation


class Wire(Electrode):
    """TODO

    Electric Wire.

    For both TxElectricLoop and TxElectricWire
    - ONLY accepts coordinates of shape=(x, 3), ndim=2
    """

    def __init__(self, coordinates):
        """Initiate an electric wire."""

        super().__init__(points=coordinates)

    def __repr__(self):
        s0 = (f"{self.__class__.__name__}: "
              f"{self._repr_add if hasattr(self, '_repr_add') else ''}\n")
        s1 = (f"    center={{{self.center[0]:,.1f}; "
              f"{self.center[1]:,.1f}; {self.center[2]:,.1f}}} m; ")
        s2 = (f"n={self.segments_n}; l={self.length:,.1f} m")
        return s0 + s1 + s2 if len(s1+s2) < 80 else s0 + s1 + "\n    " + s2

    @property
    def segment_lengths(self):
        if not hasattr(self, '_segment_lengths'):
            self._segment_lengths = np.linalg.norm(
                    np.diff(self.points, axis=0), axis=1)
        return self._segment_lengths

    @property
    def segments_n(self):
        return len(self.segment_lengths)


# SOURCES
@register_electrode
class TxElectricDipole(Dipole):
    """TODO"""

    _serialize = {'strength'} | Dipole._serialize

    def __init__(self, coordinates, strength=0.0, length=None):
        """Initiate an electric dipole source."""

        self._strength = strength
        super().__init__(coordinates=coordinates, length=length)

        self._repr_add = f"{self.moment():,.1f} Am"

    @property
    def strength(self):
        return self._strength

    def moment(self, smu0=None):
        return self.length*self.strength if self.strength != 0 else 1.0


@register_electrode
class TxMagneticDipole(Dipole):
    """TODO

    Approximated by a square loop perpendicular to dipole.

    Length is taken as area of the perpendicular loop.

    """

    _serialize = {'strength'} | Dipole._serialize

    def __init__(self, coordinates, strength=0.0, length=None):
        """Initiate a magnetic source."""

        self._strength = strength
        super().__init__(coordinates=coordinates, length=length)

        # Currently only works for mu_r=1
        self._repr_add = f"i w mu {self.moment(1):,.1f} A"

    @property
    def strength(self):
        return self._strength

    def moment(self, smu0=None):
        moment = self.length*self.strength if self.strength != 0 else 1.0
        return -moment/smu0


@register_electrode
class TxElectricWire(Wire):
    """TODO"""

    _serialize = {'strength'} | Wire._serialize

    # - has length, area (NotImplemented) attributes
    # - ensures no point coincides

    def __init__(self, coordinates, strength=0.0):
        """Initiate an electric wire source."""

        if len(np.unique(coordinates, axis=0)) != coordinates.shape[0]:
            raise ValueError(
                "All provided points must be unique. "
                f"Provided coordinates:\n{coordinates}."
            )

        self._strength = strength
        super().__init__(coordinates=coordinates)

        self._repr_add = f"{self.moment():,.1f} Am"

    @property
    def strength(self):
        return self._strength

    def moment(self, smu0=None):
        return self.length*self.strength if self.strength != 0 else 1.0


@register_electrode
class TxElectricLoop(Wire):
    """TODO"""

    _serialize = {'strength'} | Wire._serialize

    # - has length, area (NotImplemented) attributes
    # - ensures no point coincides except first and last
    # - factor ?
    def __init__(self, coordinates, strength=0.0):
        """Initiate an electric loop source."""

        if not np.allclose(coordinates[0, :], coordinates[-1, :]):
            raise ValueError(
                "First and last point must be coincident. "
                f"Provided coordinates:\n{coordinates}."
            )

        if len(np.unique(coordinates, axis=0)) != coordinates.shape[0]-1:
            raise ValueError(
                "All provided points except first/last must be unique. "
                f"Provided coordinates:\n{coordinates}."
            )

        self._strength = strength
        super().__init__(coordinates=coordinates)

        # TODO adjust for loops etc, +/- iw check
        raise NotImplementedError

    @property
    def strength(self):
        return self._strength

    # TODO adjust for loops etc, +/- iw check
    def moment(self, smu0=None):
        # Take center, and compute triangles to the points.
        raise NotImplementedError


# RECEIVERS
@register_electrode
class RxElectricPoint(Point):
    """TODO

    (x, y, z, azimuth, elevation)

    """

    def __init__(self, coordinates):
        """Initiate an electric point receiver."""
        super().__init__(coordinates)


@register_electrode
class RxMagneticPoint(Point):
    """TODO"""

    def __init__(self, coordinates):
        """Initiate a magnetic point receiver."""
        super().__init__(coordinates)


@register_electrode
class RxCurrentPoint(Point):
    """TODO"""

    def __init__(self, coordinates):
        """Initiate an electric current density point receiver."""

        # self.factor = NotImplemented
        # super().__init__(coordinates)

        raise NotImplementedError(
            "Electric current density receiver not yet fully implemented"
        )


@register_electrode
class RxFluxPoint(Point):
    """TODO"""

    def __init__(self, coordinates):
        """Initiate a magnetic flux density point receiver."""

        # self.factor = NotImplemented
        # super().__init__(coordinates)

        raise NotImplementedError(
            "Magnetic flux density receiver not yet fully implemented"
        )


# CONVERSIONS
def _point_to_dipole(point, length, deg=True):
    """Return coordinates of dipole points defined by center, angles, length.

    Spherical to Cartesian.

    Parameters
    ----------
    point : tuple
        Point coordinates in the form of (x, y, z, azimuth, elevation).

    length : float
        Dipole length (m).

    deg : bool, default: True
        Angles are in degrees if True, radians if False.


    Returns
    -------
    dipole : ndarray
        Coordinates of shape (2, 3): [[x0, y0, z0], [x1, y1, z1]].

    """

    # Get coordinates relative to centrum.
    xyz = _rotation(point[3], point[4], deg=deg)*length/2

    # Add half a dipole on both sides of the center.
    return point[:3] + np.array([-xyz, xyz])


def _dipole_to_point(dipole, deg=True):
    """Return azimuth and elevation for given electrode pair.

    Parameters
    ----------
    dipole : ndarray
        Dipole coordinates of shape (2, 3): [[x0, y0, z0], [x1, y1, z1]].

    deg : bool, default: True
        Return angles in degrees if True, radians if False.


    Returns
    -------
    azimuth : float
        Anticlockwise angle from x-axis towards y-axis.

    elevation : float
        Anticlockwise (upwards) angle from the xy-plane towards z-axis.

    length : float, default: 1.0
        Dipole length (m).

    """
    # Get distances between coordinates.
    dx, dy, dz = np.diff(dipole.T).squeeze()
    length = np.linalg.norm([dx, dy, dz])

    # Get angles from complex planes.
    azimuth = np.angle(dx + 1j*dy, deg=deg)  # same as:  np.arctan2(dy, dx)
    elevation = np.angle(np.sqrt(dx**2+dy**2) + 1j*dz, deg=deg)  # (dz, dxy)

    return azimuth, elevation, length


def _point_to_square_loop(source, area):
    """Return points of area m^2 perp to dipole.

    Parameters
    ----------
    source : tuple
        Source coordinates in the form of (x, y, z, azimuth, elevation).

    area : float
        Area of the square loop (m^2).


    Returns
    -------
    out : ndarray
        Array of shape (3, 5), corresponding to the x/y/z-coordinates for the
        five points describing a closed rectangle perpendicular to the dipole,
        of side-length length.

    """
    half_diag = np.sqrt(area/2)
    xyz_hor = _rotation(source[3]+90.0, 0.0)*half_diag
    xyz_ver = _rotation(source[3], source[4]+90.0)*half_diag
    points = source[:3] + np.stack(
            [xyz_hor, xyz_ver, -xyz_hor, -xyz_ver, xyz_hor])
    return points


def _rotation(azimuth, elevation, deg=True):
    """Rotation factors for RHS coordinate system with positive z upwards.

    The rotation factors multiplied with the length yield the corresponding
    Cartesian coordinates. (The rotation factors correspond to the rotation of
    a unit radius of length 1.)

    Definition of spherical coordinates:
    - azimuth θ: anticlockwise from x-axis towards y-axis, (-180°, +180°].
    - elevation φ: anticlockwise (upwards) from the xy-plane towards z-axis
      [-90°, +90°].
    - radius (m).

    Definition Cartesian coordinates:

    - x is Easting;
    - y is Northing;
    - z is positive upwards (RHS).


    Parameters
    ----------
    azimuth : float
        Anticlockwise angle from x-axis towards y-axis.

    elevation : float
        Anticlockwise (upwards) angle from the xy-plane towards z-axis.

    deg : bool, default: True
        Angles are in degrees if True, radians if False.


    Returns
    -------
    rot : ndarray
        Rotation factors (x, y, z).

    """
    if deg:
        cos, sin = cosdg, sindg
    else:
        cos, sin = np.cos, np.sin

    return np.array([cos(azimuth)*cos(elevation),
                     sin(azimuth)*cos(elevation),
                     sin(elevation)])
