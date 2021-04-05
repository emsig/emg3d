"""
Electrodes define any type of sources and receivers used in a survey.
"""
# Copyright 2018-2021 The EMSiG community.
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

__all__ = ['Wire', 'Point', 'Dipole', 'Source', 'TxElectricDipole',
           'TxMagneticDipole', 'TxElectricWire', 'RxElectricPoint',
           'RxMagneticPoint', 'txrx_coordinates_to_dict', 'txrx_lists_to_dict',
           'rotation', 'point_to_dipole', 'dipole_to_point',
           'point_to_square_loop']


# BASE ELECTRODE TYPES
class Wire:
    """A wire consists of an arbitrary number of electrodes.

    .. note::

        Use any of the Tx*/Rx* classes to create sources and receivers, not
        this class.


    Parameters
    ----------
    coordinates : array_like
        Electrode locations of shape (n, 3), where n is the number of
        electrodes: ``[[x1, y1, z1], [...], [xn, yn, zn]]``.

    """

    # Attributes which are stored/required with to_dict/from_dict.
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

    def __repr__(self):
        """Simple representation."""
        s0 = (f"{self.__class__.__name__}: "
              f"{self._repr_add if hasattr(self, '_repr_add') else ''}\n")
        s1 = (f"    center={{{self.center[0]:,.1f}; "
              f"{self.center[1]:,.1f}; {self.center[2]:,.1f}}} m; ")
        s2 = (f"n={self.segment_n}; l={self.length:,.1f} m")
        return s0 + s1 + s2 if len(s1+s2) < 80 else s0 + s1 + "\n    " + s2

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
        as theoretical "magnetic dipoles" (TxMagneticDipole, RxMagneticPoint).
        ``xtype`` is a flag for this.

        """
        if not hasattr(self, '_xtype'):
            if 'Magnetic' in self.__class__.__name__:
                self._xtype = 'magnetic'
            else:  # Default
                self._xtype = 'electric'
        return self._xtype

    @property
    def center(self):
        """Center point of all unique electrodes."""
        if not hasattr(self, '_center'):
            self._center = np.unique(self.points, axis=0).mean(axis=0)
        return self._center

    @property
    def length(self):
        """Total length of all dipole segments formed by the electrodes."""
        if not hasattr(self, '_length'):
            lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
            self._segment_lengths = lengths
            self._length = lengths.sum()
        return self._length

    @property
    def segment_lengths(self):
        """Length of each individual dipole segment in the wire."""
        if not hasattr(self, '_segment_lengths'):
            _ = self.length  # Sets length and segment_lengths
        return self._segment_lengths

    @property
    def segment_n(self):
        """Number of dipole segments in the wire."""
        return len(self.segment_lengths)

    @property
    def _prefix(self):
        """Prefix used for collecting Tx/Rx in Surveys."""
        name = self.__class__.__name__
        return name[:2] + ''.join(c for c in name if c.isupper())[1:]


class Point(Wire):
    """A point electrode is defined by its center, azimuth, and elevation.

    A ``Point`` is a special case of a ``Wire`` that consists of only one
    electrode, and has therefore no length (infinitesimal small dipole). It is
    principally used by receivers to sample the field at a given point.

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
        s0 = (f"{self.__class__.__name__}: "
              f"{self._repr_add if hasattr(self, '_repr_add') else ''}\n")
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


class Dipole(Wire):
    """A dipole consists of two electrodes in a straight line.

    A ``Dipole`` is a special case of a ``Wire`` that consists of exactly two
    electrodes. A dipole has therefore an azimuth and an elevation. It
    corresponds to one segment in a wire.

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

        # Cast coordinates.
        coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()

        # Check which format was provided.
        is_point = coordinates.shape == (5, )
        is_flat = coordinates.shape == (6, )
        is_dipole = coordinates.shape == (2, 3)

        # Store depending on format.
        if is_point:

            # Add length to attributes which have to be serialized.
            self._serialize = {'length'} | self._serialize

            # If magnetic, get the loop which area corresponds to length.
            if self.xtype == 'magnetic':
                points = point_to_square_loop(coordinates, length)

            # If electric, get the dipole.
            else:
                points = point_to_dipole(coordinates, length)

            # Store length and original input coordinates.
            self._length = length
            self._coordinates = coordinates

        elif is_flat or is_dipole:

            if is_flat:
                # Re-arrange for points.
                points = np.array([coordinates[::2], coordinates[1::2]])

                # Store original input.
                self._coordinates = coordinates

            else:
                # Input is already in the format for Electrode.
                points = coordinates

            # If magnetic, get the loop which area corresponds to its length.
            if self.xtype == 'magnetic':
                azimuth, elevation, length = dipole_to_point(points)
                center = tuple(np.sum(points, 0)/2)
                coo = (*center, azimuth, elevation)
                points = point_to_square_loop(coo, length)

                # Store original input.
                self._coordinates = coordinates

            # Ensure the two poles are distinct.
            if np.allclose(points[0, :], points[1, :]):
                raise ValueError(
                    "The two electrodes are identical, use the format "
                    "(x, y, z, azimuth, elevation) instead. "
                    f"Provided coordinates: {coordinates}."
                )

        else:
            raise ValueError(
                "Coordinates are wrong defined. They must be defined either "
                "as a point, (x, y, z, azimuth, elevation), or as two points, "
                "(x1, x2, y1, y2, z1, z2) or [[x1, y1, z1], [x2, y2, z2]]. "
                f"Provided coordinates: {coordinates}."
            )

        super().__init__(points)

    def __repr__(self):
        """Simple representation."""

        s0 = (f"{self.__class__.__name__}: "
              f"{self._repr_add if hasattr(self, '_repr_add') else ''}\n")

        # Point dipole.
        if self.coordinates.size == 5:
            s1 = (f"    center={{{self.center[0]:,.1f}; "
                  f"{self.center[1]:,.1f}; {self.center[2]:,.1f}}} m; ")
            s2 = (f"θ={self.azimuth:.1f}°, φ={self.elevation:.1f}°; "
                  f"l={self.length:,.1f} m")

        # Finite dipole.
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
            if len(self.coordinates) == 5:
                out = self._coordinates[3:]
            else:
                out = dipole_to_point(self._points)[:2]
            self._azimuth, self._elevation = out
        return self._azimuth

    @property
    def elevation(self):
        """Anticlockwise (upwards) rotation (°) from the xy-plane."""
        if not hasattr(self, '_elevation'):
            _ = self.azimuth  # Sets azimuth and elevation
        return self._elevation


# SOURCES
class Source(Wire):
    """A source adds strength to a Wire instance.

    .. note::

        Use any of the Tx* classes to create sources, not this class.


    Parameters
    ----------
    strength : {float, complex}
        Source strength (A).

    """

    # Add strength to attributes which have to be serialized.
    _serialize = {'strength'} | Wire._serialize

    def __init__(self, strength, **kwargs):
        """Initiate an electric source."""

        # Store strength, add a repr-addition.
        self._strength = strength
        self._repr_add = f"{self.strength:,.1f} A;"

        super().__init__(**kwargs)

    @property
    def strength(self):
        """Source strength (A)."""
        return self._strength

    def get_field(self, grid, frequency):
        """Returns source field for given grid and frequency."""
        return fields.get_source_field(grid, self, frequency)


@utils.known_class
class TxElectricDipole(Source, Dipole):
    """Electric dipole source, two electrodes connected by a wire.


    Parameters
    ----------
    coordinates : array_like
        Dipole coordinates. Three formats are accepted:

        - [[x1, y1, z1], [x2, y2, z2]];
        - (x1, x2, y1, y2, z1, z2);
        - (x, y, z, azimuth, elevation); this format takes also the ``length``
          parameter.

    strength : float, default: 1.0
        Source strength (A).

    length : float, default: 1.0
        Length of the dipole (m). This parameter is only used if the provided
        coordinates are in the format (x, y, z, azimuth, elevation).

    """

    def __init__(self, coordinates, strength=1.0, length=1.0):
        """Initiate an electric dipole source."""

        super().__init__(
                coordinates=coordinates, strength=strength, length=length)


@utils.known_class
class TxMagneticDipole(Source, Dipole):
    """Magnetic dipole source using a square loop perpendicular to the dipole.

    The magnetic dipole source simulates a magnetic dipole with an electric
    square loop perpendicular and at the center of the dipole. The area of the
    loop corresponds to the dipole length, to represent the same strength.


    Parameters
    ----------
    coordinates : array_like
        Dipole coordinates. Three formats are accepted:

        - [[x1, y1, z1], [x2, y2, z2]];
        - (x1, x2, y1, y2, z1, z2);
        - (x, y, z, azimuth, elevation); this format takes also the ``length``
          parameter.

    strength : float, default: 1.0
        Source strength (A).

    length : float, default: 1.0
        Length of the dipole (m). This parameter is only used if the provided
        coordinates are in the format (x, y, z, azimuth, elevation).

    """

    def __init__(self, coordinates, strength=1.0, length=1.0):
        """Initiate a magnetic source."""

        super().__init__(
                coordinates=coordinates, strength=strength, length=length)


@utils.known_class
class TxElectricWire(Source, Wire):
    """Electric wire source consisting of a series of dipoles.


    Parameters
    ----------
    coordinates : array_like
        Electrode locations of shape (n, 3), where n is the number of
        electrodes: ``[[x1, y1, z1], [...], [xn, yn, zn]]``.

    strength : float, default: 1.0
        Source strength (A).

    """

    def __init__(self, coordinates, strength=1.0):
        """Initiate an electric wire source."""

        super().__init__(coordinates=coordinates, strength=strength)


# RECEIVERS
class Receiver(Wire):
    """A receiver can be positioned absolutely or relative to source..

    .. note::

        Use any of the Rx* classes to create receivers, not this class.


    Parameters
    ----------
    relative : bool
        If False, the coordinates are absolute coordinates. If True, the
        coordinates define the offset from the source center.

        Note that ``relative=True`` makes only sense in combination with
        sources, such as is the case in a :class:`emg3d.surveys.Survey`.

    """

    # Add relative to attributes which have to be serialized.
    _serialize = {'relative'} | Wire._serialize

    def __init__(self, relative, **kwargs):
        """Initiate a receiver."""

        # Store relative, add a repr-addition.
        self._relative = relative
        self._repr_add = f"{['absolute', 'relative'][self.relative]};"

        super().__init__(**kwargs)

    @property
    def relative(self):
        """True if coordinates are relative to source, False if absolute."""
        return self._relative

    def points_abs(self, source):
        """Returns points as absolute positions."""
        if self.relative:
            return source.center + self.points
        else:
            return self.points


@utils.known_class
class RxElectricPoint(Receiver, Point):
    """Electric point receiver (point sampling the field).


    Parameters
    ----------
    coordinates : array_like
        Point defined as (x, y, z, azimuth, elevation)

    relative : bool, default: False
        If False, the coordinates are absolute coordinates. If True, the
        coordinates define the offset from the source center.

        Note that ``relative=True`` makes only sense in combination with
        sources, such as is the case in a :class:`emg3d.surveys.Survey`.

    """

    def __init__(self, coordinates, relative=False):
        """Initiate an electric point receiver."""

        super().__init__(coordinates=coordinates, relative=relative)


@utils.known_class
class RxMagneticPoint(Receiver, Point):
    """Magnetic point receiver (point sampling the field).


    Parameters
    ----------
    coordinates : array_like
        Point defined as (x, y, z, azimuth, elevation)

    relative : bool, default: False
        If False, the coordinates are absolute coordinates. If True, the
        coordinates define the offset from the source center.

        Note that ``relative=True`` makes only sense in combination with
        sources, such as is the case in a :class:`emg3d.surveys.Survey`.

    """

    def __init__(self, coordinates, relative=False):
        """Initiate a magnetic point receiver."""

        super().__init__(coordinates=coordinates, relative=relative)


# CONVENIENCE FUNCTIONS, ROTATIONS
def txrx_coordinates_to_dict(TxRx, coordinates, **kwargs):
    """Create dict of TxRx instances with provided coordinates.

    Source and receiver dictionaries to input into a
    :class:`emg3d.surveys.Survey` can be created in many ways. This is a helper
    function to create a dict from a tuple of coordinates.


    Parameters
    ----------
    TxRx : {Tx*, Rx*)
        Any of the available sources or receivers, e.g.,
        :class:`emg3d.electrodes.TxElectricDipole`.

    coordinates : tuple
        Tuple containing the input coordinates for the defined TxRx class.
        Each element of the tuple must either have length ``1`` or ``n``.

    **kwargs :
        Other parameters passed through to TxRx; again, each must be of size
        ``1`` or ``n``.


    Returns
    -------
    out : dict
        Dict where the keys consist of a TxRx-prefix followed by a number, and
        the values contain the corresponding TxRx instances.


    Examples
    --------

    .. ipython::

       In [1]: import emg3d
          ...: import numpy as np

       In [2]: # Create 10 electric dipole sources from x=2000:2000:10,000, of
          ...: # strength 100 A.
          ...: offsets = np.arange(1, 6)*2000
          ...: sources = emg3d.electrodes.txrx_coordinates_to_dict(
          ...:                 emg3d.TxElectricDipole,
          ...:                 (offsets, 0, 0, 0, 0), strength=100)
          ...: sources  # QC the source dict

    """

    # Get max dimension.
    nd = max([np.array(n, ndmin=1).size for n in coordinates])

    # Expand coordinates.
    coo = np.array([nd*[val, ] if np.array(val).size == 1 else
                    val for val in coordinates], dtype=np.float64)

    # Expand kwargs.
    inp = {}
    for i in range(nd):
        inp[i] = {}
        for k, v in kwargs.items():
            inp[i][k] = v if np.array(v).size == 1 else v[i]

    # Return TxRx-dict.
    return txrx_lists_to_dict([TxRx(coo[:, i], **inp[i]) for i in range(nd)])


def txrx_lists_to_dict(txrx):
    """Create dict from provided list of Tx/Rx instances.

    Source and receiver dictionaries to input into a
    :class:`emg3d.surveys.Survey` can be created in many ways. This is a helper
    function to create a dict from a list of source or receiver instances, or
    from a list of lists and dicts of source or receiver instances.


    Parameters
    ----------
    txrx : {list, dict)
        A list or dict of Tx*/Rx* instances. If it is a dict, it is returned
        unaltered.

        A list can also consist of other lists and dicts of Tx*/Rx* instances.


    Returns
    -------
    out : dict
        Dict where the keys consist of a TxRx-specific prefix followed by a
        number, and the values contain the corresponding TxRx instances.


    Examples
    --------

    .. ipython::

       In [1]: import emg3d
          ...: import numpy as np

       In [2]: # Create two electric, fixed receivers.
          ...: electric = [emg3d.RxElectricPoint((x, 0, 0, 0, 0))
          ...:             for x in [1000, 1100]]

       In [3]: # Create three magnetic, fixed receivers.
          ...: magnetic = emg3d.electrodes.txrx_coordinates_to_dict(
          ...:                 emg3d.RxMagneticPoint,
          ...:                 ([950, 1050, 1150], 0, 0, 0, 90))

       In [4]: # Create a streamer receiver, flying 5 m behind the source.
          ...: streamer = emg3d.RxElectricPoint((5, 0, 0, 0, 0), relative=True)

       In [5]: # Collect all receivers.
          ...: receivers = emg3d.electrodes.txrx_lists_to_dict(
          ...:                 [[streamer, ], electric, magnetic])
          ...: receivers  # QC our collected receivers

    """

    # If input is a dict, return it unaltered.
    if isinstance(txrx, dict):
        return txrx

    # Undocumented, but possible: A single Tx*/Rx* instance.
    elif hasattr(txrx, '_prefix'):
        txrx = [txrx, ]

    # If it is a list of lists/dicts, collect them.
    elif isinstance(txrx[0], (list, dict)):

        # Add all lists dict to new list.
        new_txrx = list()
        for trx in txrx:
            # If dict, cast it to list.
            if isinstance(trx, dict):
                trx = list(trx.values())
            new_txrx += trx

        # Overwrite original list with new flat list.
        txrx = new_txrx

    # else, it has to be a list of Tx/Rx instances.

    # Return TxRx-dict.
    nx = len(txrx)
    return {f"{trx._prefix}-{i+1:0{len(str(nx))}d}": trx
            for i, trx in enumerate(txrx)}


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
