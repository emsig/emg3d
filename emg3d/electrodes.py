"""
Electrodes define any type of sources and receivers used in a survey.
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

from emg3d import fields, utils

__all__ = ['Electrode', 'Point', 'Dipole', 'Source', 'TxElectricDipole',
           'TxMagneticDipole', 'TxElectricWire', 'RxElectricPoint',
           'RxMagneticPoint', 'rotation', 'point_to_dipole', 'dipole_to_point',
           'point_to_square_loop']


# BASE ELECTRODE TYPES
class Electrode:
    """Electrode is the BaseClass on which any source or receiver is built.

    .. note::

        Use any of the Tx*/Rx* classes to create sources and receivers, not
        this class.


    Parameters
    ----------
    coordinates : ndarray
        Electrode locations of shape (n, 3), where n is the number of
        electrodes: ``[[x1, y1, z1], [...], [xn, yn, zn]]``.

    """

    _serialize = {'coordinates'}

    def __init__(self, coordinates):
        """Initiate an electrode."""

        # Cast and check dimension and shape.
        self._points = np.asarray(np.atleast_2d(coordinates), dtype=float)
        if not (self._points.ndim == 2 and self._points.shape[1] == 3):
            raise ValueError(
                "`coordinates` must be of shape (x, 3), provided: "
                f"{coordinates}"
            )

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
        """Store the necessary information of the Electrode in a dict.

        Parameters
        ----------
        copy : bool, default: False
            If True, returns a deep copy of the dict.


        Returns
        -------
        out : dict
            Dictionary containing all information to re-create the Electrode.

        """
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
        """Convert dictionary into its class instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from the classes' ``to_dict``.

        Returns
        -------
        electrode : {Tx*, Rx*}
            A source or receiver instance.

        """
        return cls(**{k: v for k, v in inp.items() if k != '__class__'})

    @property
    def points(self):
        """Electrode locations (n, 3)."""
        return self._points

    @property
    def coordinates(self):
        """Electrode coordinate as accepted by its class."""
        if hasattr(self, '_coordinates'):
            return self._coordinates
        else:
            return self._points

    @property
    def xtype(self):
        """Flag of the type of electrodes.

        In reality, all electrodes are electric. But we do idealize some loops
        as theoretical "magnetic dipoles". ``xtype`` is a flag for this.

        """
        if not hasattr(self, '_xtype'):
            if 'Magnetic' in self.__class__.__name__:
                self._xtype = 'magnetic'
            else:  # Default
                self._xtype = 'electric'
        return self._xtype

    @property
    def center(self):
        """Center point of all electrodes."""
        if not hasattr(self, '_center'):
            self._center = self.points.mean(axis=0)
        return self._center

    @property
    def length(self):
        """Total length of all dipoles formed by the electrodes."""
        if not hasattr(self, '_length'):
            self._length = np.linalg.norm(
                    np.diff(self.points, axis=0), axis=1).sum()
        return self._length


class Point(Electrode):
    """A point electrode is defined by its center, azimuth, and elevation.

    .. note::

        Use any of the Tx*/Rx* classes to create sources and receivers, not
        this class.


    Parameters
    ----------
    coordinates : array_like
        Point defined as (x, y, z, azimuth, elevation)

    """

    def __init__(self, coordinates):
        """Initiate an electric point."""

        if len(coordinates) != 5:
            raise ValueError(
                "Point coordinates are wrong defined. They must be "
                "defined as (x, y, z, azimuth, elevation)."
                f"Provided coordinates: {coordinates}."
            )

        # Cast and store input coordinates.
        self._coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()

        # Provide center to `Electrode`.
        super().__init__(coordinates[:3])

    def __repr__(self):
        """Simple representation."""
        s0 = f"{self.__class__.__name__}: \n"
        s1 = (f"    x={self.center[0]:,.1f} m, "
              f"y={self.center[1]:,.1f} m, z={self.center[2]:,.1f} m, ")
        s2 = f"θ={self.azimuth:.1f}°, φ={self.elevation:.1f}°"
        return s0 + s1 + s2 if len(s1+s2) < 80 else s0 + s1 + "\n    " + s2

    @property
    def azimuth(self):
        """Anticlockwise rotation (°) from x-axis towards y-axis."""
        return self._coordinates[3]

    @property
    def elevation(self):
        """Anticlockwise (upwards) rotation (°) from the xy-plane."""
        return self._coordinates[4]


class Dipole(Electrode):
    """A dipole consists of two electrodes in a straight line.

    .. note::

        Use any of the Tx*/Rx* classes to create sources and receivers, not
        this class.


    Parameters
    ----------
    coordinates : array_like
        Dipole coordinates. Three formats are accepted:

        - [[x1, y1, z1], [x2, y2, z2]];
        - (x1, x2, y1, y2, z1, z2);
        - (x, y, z, azimuth, elevation); this format takes also the ``length``
          parameter.

    length : float, default: 1.0
        Length of the dipole (m). This parameter is only used if the provided
        coordinates are in the format (x, y, z, azimuth, elevation).

    """

    def __init__(self, coordinates, length=1.0):
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

            # Get the two separate electrodes.
            if self.xtype == 'magnetic':
                # square loop of area corresponding to length
                points = point_to_square_loop(coordinates, length)
            else:
                points = point_to_dipole(coordinates, length)

            self._length = length
            self._coordinates = coordinates

        elif coordinates.size == 6:

            if coordinates.ndim == 1:
                points = np.array([coordinates[::2], coordinates[1::2]])
                self._coordinates = coordinates

            else:
                points = coordinates

            if self.xtype == 'magnetic':
                azimuth, elevation, length = dipole_to_point(points)
                center = tuple(np.sum(points, 0)/2)
                coo = (*center, azimuth, elevation)
                points = point_to_square_loop(coo, length)

            # Ensure the two poles are distinct.
            if np.allclose(points[0, :], points[1, :]):
                raise ValueError(
                    "The two poles are identical, use the format "
                    "(x, y, z, azimuth, elevation) instead. "
                    f"Provided coordinates: {self._coordinates}."
                )

        else:
            raise ValueError(
                "Dipole coordinates are wrong defined. They must be "
                "defined either as a point, (x, y, z, azimuth, elevation), or "
                "as two poles, (x1, x2, y1, y2, z1, z2) or "
                "[(x1, y1, z1), (x2, y2, z2)] , all floats. "
                f"Provided coordinates: {self._coordinates}."
            )

        super().__init__(points)

    def __repr__(self):
        """Simple representation."""
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
        """Anticlockwise rotation (°) from x-axis towards y-axis."""
        if not hasattr(self, '_azimuth'):
            if self._points.shape[0] == 5:
                self._azimuth = self._coordinates[3]
                self._elevation = self._coordinates[4]
            else:
                out = dipole_to_point(self._points)
                self._azimuth, self._elevation, _ = out
        return self._azimuth

    @property
    def elevation(self):
        """Anticlockwise (upwards) rotation (°) from the xy-plane."""
        if not hasattr(self, '_elevation'):
            _ = self.azimuth  # Sets azimuth and elevation
        return self._elevation


class Wire(Electrode):
    """TODO

    .. note::

        Use any of the Tx*/Rx* classes to create sources and receivers, not
        this class.

    Electric Wire.

    For both TxElectricLoop and TxElectricWire
    - ONLY accepts coordinates of shape=(x, 3), ndim=2
    """

    def __init__(self, coordinates):
        """Initiate an electric wire."""
        self._coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()

        super().__init__(coordinates)

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
class Source(Electrode):
    """TODO

    .. note::

        Use any of the Tx* classes to create sources, not this class.


    """

    _serialize = {'strength'} | Electrode._serialize

    def __init__(self, coordinates, strength, **kwargs):
        """Initiate an electric  source."""

        if abs(strength) == 0.0:
            raise ValueError(
                f"Source strength cannot be zero. Provided: {strength}."
            )

        self._strength = strength
        self._repr_add = f"{self.strength:,.1f} A"

        super().__init__(coordinates=coordinates, **kwargs)

    @property
    def strength(self):
        return self._strength

    def get_field(self, grid, frequency):
        return fields.get_source_field(grid, self, frequency)


@utils.known_class
class TxElectricDipole(Source, Dipole):
    """TODO"""

    _serialize = Source._serialize | Dipole._serialize

    def __init__(self, coordinates, strength=1.0, length=1.0):
        """Initiate an electric dipole source."""

        super().__init__(coordinates=coordinates, strength=strength,
                         length=length)


@utils.known_class
class TxMagneticDipole(Source, Dipole):
    """TODO

    Approximated by a square loop perpendicular to dipole.

    Length is taken as area of the perpendicular loop.

    """

    _serialize = Source._serialize | Dipole._serialize

    def __init__(self, coordinates, strength=1.0, length=1.0):
        """Initiate a magnetic source."""

        super().__init__(coordinates=coordinates, strength=strength,
                         length=length)


@utils.known_class
class TxElectricWire(Source, Wire):
    """TODO"""

    _serialize = Source._serialize | Wire._serialize

    def __init__(self, coordinates, strength=1.0):
        """Initiate an electric wire source."""

        super().__init__(coordinates=coordinates, strength=strength)


# RECEIVERS
@utils.known_class
class RxElectricPoint(Point):
    """TODO

    (x, y, z, azimuth, elevation)

    """

    def __init__(self, coordinates):
        """Initiate an electric point receiver."""
        super().__init__(coordinates)


@utils.known_class
class RxMagneticPoint(Point):
    """TODO"""

    def __init__(self, coordinates):
        """Initiate a magnetic point receiver."""
        super().__init__(coordinates)


# CONVERSIONS
def point_to_dipole(point, length, deg=True):
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
        Coordinates of shape (2, 3): [[x1, y1, z1], [x2, y2, z2]].

    """

    # Get coordinates relative to centrum.
    xyz = rotation(point[3], point[4], deg=deg)*length/2

    # Add half a dipole on both sides of the center.
    return point[:3] + np.array([-xyz, xyz])


def dipole_to_point(dipole, deg=True):
    """Return azimuth and elevation for given electrode pair.

    Cartesian to spherical.


    Parameters
    ----------
    dipole : ndarray
        Dipole coordinates of shape (2, 3): [[x1, y1, z1], [x2, y2, z2]].

    deg : bool, default: True
        Return angles in degrees if True, radians if False.


    Returns
    -------
    azimuth : float
        Anticlockwise angle from x-axis towards y-axis.

    elevation : float
        Anticlockwise (upwards) angle from the xy-plane towards z-axis.

    length : float
        Dipole length (m).

    """
    # Get distances between coordinates.
    dx, dy, dz = np.diff(dipole.T).squeeze()
    length = np.linalg.norm([dx, dy, dz])

    # Get angles from complex planes.
    azimuth = np.angle(dx + 1j*dy, deg=deg)  # same as:  np.arctan2(dy, dx)
    elevation = np.angle(np.sqrt(dx**2+dy**2) + 1j*dz, deg=deg)  # (dz, dxy)

    return azimuth, elevation, length


def point_to_square_loop(source, area):
    """Return points of a loop of area perpendicular to source dipole.


    Parameters
    ----------
    source : tuple
        Source dipole coordinates in the form of (x, y, z, azimuth, elevation).

    area : float
        Area of the square loop (m^2).


    Returns
    -------
    out : ndarray
        Array of shape (5, 3), corresponding to the x/y/z-coordinates for the
        five points describing a closed rectangle perpendicular to the dipole,
        of side-length length.

    """
    half_diag = np.sqrt(area/2)
    xyz_hor = rotation(source[3]+90.0, 0.0)*half_diag
    xyz_ver = rotation(source[3], source[4]+90.0)*half_diag
    points = source[:3] + np.stack(
            [xyz_hor, xyz_ver, -xyz_hor, -xyz_ver, xyz_hor])
    return points


def rotation(azimuth, elevation, deg=True):
    """Rotation factors for RHS coordinate system with positive z upwards.


    The rotation factors multiplied with the length yield the corresponding
    Cartesian coordinates. (The rotation factors correspond to the rotation of
    a unit radius of length 1.)

    Definition of spherical coordinates:

    - azimuth θ: anticlockwise from x-axis towards y-axis, (-180°, +180°].
    - elevation φ: anticlockwise (upwards) from the xy-plane towards z-axis
      [-90°, +90°].
    - radius (m).

    Definition of Cartesian coordinates:

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
