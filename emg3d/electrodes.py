"""
A survey stores a set of sources, receivers, and the measured data.
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
# TODO from dataclasses import dataclass

import numpy as np

from emg3d import maps

__all__ = ['Electrode', 'Point', 'Dipole', ]


# List of electrodes
ELECTRODE_LIST = {}


def register_electrode(func):
    ELECTRODE_LIST[func.__name__] = func
    __all__.append(func.__name__)
    return func


class Electrode:

    _serialize = {'coordinates', }

    def __init__(self, points, coordinates=None):
        """

        Points must be in the form of
            [[x0, y0, z0], [...], [xN, yN, zN]]: (x, 3)

        Coordinates can be different, it is what the given class uses.
        If not provided, it is set to coordinates.

        """

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
        inp.pop('__class__', None)
        return cls(**inp)

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


class Point(Electrode):

    def __init__(self, coordinates):

        coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()
        super().__init__(points=coordinates[:3], coordinates=coordinates)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"x={self.center[0]:,.1f}m, "
                f"y={self.center[1]:,.1f}m, "
                f"z={self.center[2]:,.1f}m, "
                f"θ={self.azimuth:.1f}°, "
                f"φ={self.dip:.1f}°)")

    @property
    def center(self):
        return self._coordinates[:3]

    @property
    def azimuth(self):
        return self._coordinates[3]

    @property
    def dip(self):
        return self._coordinates[4]


# Create own Source class, adjust for multiple inheritance
#
# class Source(Electrode):
#     _serialize = {*Electrode._serialize, 'strength'}


class Dipole(Electrode):

    _serialize = {*Electrode._serialize, 'strength'}

    def __init__(self, coordinates, strength, length):

        coordinates = np.asarray(coordinates, dtype=np.float64).squeeze()

        # TODO either (x, y, z, azimuth, dip), length or
        #             (x0, x1, y0, y1, z0, z1) or
        #             ([x0, y0, z0], [x1, y1, z1])

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

            # Get lengths in each direction.
            if length is None:
                length = 1.0

            # Get the two separate electrodes.
            points = maps.get_points(*coordinates, length)

        elif coordinates.size == 6:
            if coordinates.ndim == 1:
                points = np.array([coordinates[::2], coordinates[1::2]])

            else:
                points = coordinates
                coordinates = None

            # Ensure the two poles are distinct.
            if np.allclose(points[0, :], points[1, :]):
                raise ValueError(
                    "The two poles are identical, use the format "
                    "(x, y, z, azimuth, dip) instead. "
                    f"Provided coordinates: {coordinates}."
                )

            if length is not None:
                raise ValueError("No length with this format")

        else:
            raise ValueError(
                "Dipole coordinates are wrong defined. They must be "
                "defined either as a point, (x, y, z, azimuth, dip), or "
                "as two poles, (x0, x1, y0, y1, z0, z1) or "
                "[(x0, y0, z0), (x1, y1, z1)] , all floats. "
                f"Provided coordinates: {coordinates}."
            )

        self._strength = float(strength)

        super().__init__(points=points, coordinates=coordinates)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"x={self.center[0]:,.1f}m, "
                f"y={self.center[1]:,.1f}m, "
                f"z={self.center[2]:,.1f}m, "
                f"θ={self.azimuth:.1f}°, "
                f"φ={self.dip:.1f}°"
                f"; {self.length}m; {self.strength})")

    @property
    def strength(self):
        return self._strength

    @property
    def center(self):
        if not hasattr(self, '_center'):
            self._center = tuple(np.sum(self._points, 0)/2)
        return self._center

    @property
    def azimuth(self):
        if not hasattr(self, '_azimuth'):
            self._azimuth, self._dip = maps.get_angles(self._points)
        return self._azimuth

    @property
    def dip(self):
        if not hasattr(self, '_dip'):
            self._azimuth, self._dip = maps.get_angles(self._points)
        return self._dip

    @property
    def length(self):
        if not hasattr(self, '_length'):
            self._length = np.linalg.norm(self.points[1, :]-self.points[0, :])
        return self._length


class Wire(Electrode):
    # For both TxElectricLoop and TxElectricWire
    #
    # - ONLY accepts coordinates of shape=(x, 3), ndim=2
    #
    # def __repr__(self):
    #
    # @property
    # def center(self):
    #
    # @property
    # def length(self):
    #
    # @property
    # def area(self):
    #     NotImplemented
    pass


@register_electrode
class TxElectricDipole(Dipole):

    def __init__(self, coordinates, strength=1.0, length=None):

        super().__init__(coordinates, strength, length)


@register_electrode
class TxMagneticDipole(Dipole):
    pass


@register_electrode
class TxElectricWire(Wire):
    # - has length, area (NotImplemented) attributes
    # - ensures no point coincides
    pass


@register_electrode
class TxElectricLoop(Wire):
    # - has length, area (NotImplemented) attributes
    # - ensures no point coincides except first and last
    # - factor ?
    pass


@register_electrode
class RxElectricPoint(Point):

    def __init__(self, coordinates):
        """
        (x, y, z, azimuth, dip)
        """
        super().__init__(coordinates)


@register_electrode
class RxMagneticPoint(Point):

    def __init__(self, coordinates):
        super().__init__(coordinates)


@register_electrode
class RxCurrentPoint(Point):

    def __init__(self, coordinates):
        self.factor = NotImplemented
        super().__init__(coordinates)


@register_electrode
class RxFluxPoint(Point):

    def __init__(self, coordinates, **kwargs):
        self.factor = NotImplemented
        super().__init__(coordinates)
