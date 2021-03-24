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

    def __init__(self, coordinates):
        """

        Coordinates must be in the form of
            [[x0, y0, z0], [...], [xN, yN, zN]]: (x, 3)

        """

        coordinates = np.atleast_2d(coordinates)

        if not (coordinates.ndim == 2 and coordinates.shape[1] == 3):
            raise ValueError(
                "`coordinates` must be of shape (x, 3), provided: "
                f"{coordinates.shape}"
            )

        self._coordinates = np.asarray(coordinates, dtype=float)

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
    def coordinates(self):
        return self._coordinates

    @property
    def xtype(self):
        if not hasattr(self, '_xtype'):
            if 'Current' in self.__class__.__name__:
                self._xtype = 'current'
            elif ('Current' in self.__class__.__name__ or
                  'Loop' in self.__class__.__name__):
                self._xtype = 'flux'
            elif 'Magnetic' in self.__class__.__name__:
                self._xtype = 'magnetic'
            else:  # Default
                self._xtype = 'electric'
        return self._xtype


class Point(Electrode):

    _serialize = {*Electrode._serialize, 'azimuth', 'dip'}

    def __init__(self, coordinates, azimuth, dip):

        self._azimuth = float(azimuth)
        self._dip = float(dip)

        super().__init__(coordinates)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"x={self.center[0]:,.1f}m, "
                f"y={self.center[1]:,.1f}m, "
                f"z={self.center[2]:,.1f}m, "
                f"θ={self.azimuth:.1f}°, "
                f"φ={self.dip:.1f}°)")

    @property
    def center(self):
        if not hasattr(self, '_center'):
            self._center = tuple(self._coordinates[0, :])
        return self._center

    @property
    def azimuth(self):
        return self._azimuth

    @property
    def dip(self):
        return self._dip

# Create own Source class, adjust for multiple inheritance
#
# class Source(Electrode):
#     _serialize = {*Electrode._serialize, 'strength'}


class Dipole(Electrode):

    _serialize = {*Electrode._serialize, 'strength'}

    def __init__(self, coordinates, strength, length):

        coordinates = np.squeeze(coordinates)

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
            coordinates = maps._get_electrodes(*coordinates, length)

        elif coordinates.size == 6:
            if coordinates.ndim == 1:
                coordinates = np.array([coordinates[::2], coordinates[1::2]])

            # Ensure the two poles are distinct.
            if np.allclose(coordinates[0, :], coordinates[1, :]):
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

        super().__init__(coordinates)

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
            self._center = tuple(np.sum(self._coordinates, 0)/2)
        return self._center

    @property
    def azimuth(self):
        if not hasattr(self, '_azimuth'):
            self._azimuth, self._dip = maps._get_angles(self._coordinates)
        return self._azimuth

    @property
    def dip(self):
        if not hasattr(self, '_dip'):
            self._azimuth, self._dip = maps._get_angles(self._coordinates)
        return self._dip

    @property
    def length(self):
        if not hasattr(self, '_length'):
            self._length = np.linalg.norm(
                self.coordinates[1, :]-self.coordinates[0, :])
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
class RxPointElectricField(Point):

    def __init__(self, coordinates, **kwargs):
        """
        (x, y, z, azimuth, dip)
        """
        out = _check_point_coordinates(coordinates, **kwargs)

        super().__init__(*out)


@register_electrode
class RxPointMagneticField(Point):

    def __init__(self, coordinates, **kwargs):

        out = _check_point_coordinates(coordinates, **kwargs)

        super().__init__(*out)


@register_electrode
class RxPointElectricCurrentDensity(Point):

    def __init__(self, coordinates, **kwargs):

        self.factor = NotImplemented

        out = _check_point_coordinates(coordinates, **kwargs)

        super().__init__(*out)


@register_electrode
class RxPointMagneticFluxDensity(Point):

    def __init__(self, coordinates, **kwargs):

        self.factor = NotImplemented

        out = _check_point_coordinates(coordinates, **kwargs)

        super().__init__(*out)


def _check_point_coordinates(coordinates, azimuth=None, dip=None):
    coordinates = np.squeeze(coordinates)
    wrong = azimuth is None and dip is not None
    wrong *= azimuth is not None and dip is None
    if wrong:
        raise ValueError("Either both or None")

    elif azimuth is None:
        azimuth, dip = coordinates[3:]
        coordinates = coordinates[:3]

    return coordinates, azimuth, dip
