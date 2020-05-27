"""

:mod:`survey` -- Surveys
========================

A survey combines a set of sources and receivers.

"""
# Copyright 2018-2020 The emg3d Developers.
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


import numpy as np
# from typing import Tuple
from dataclasses import dataclass  # , field

# from emg3d import maps, models

__all__ = ['Survey', 'DipoleSource', 'DipoleReceiver']


class Survey:
    """Create a survey with sources and receivers.

    Survey takes care to join same receivers
    Survey should check for
    - source-receiver combinations,
    - report missing sources
    - list nfreq per source and all the rest


    - Need to get frequencies by receivers and by source v
    - Need to get sources and receivers by frequency     ^

    Parameters
    ----------
    ?

    """

    def __init__(self):
        """Initiate a new Survey instance."""

        # TODO We should be able to provide source and receiver list straight
        #      to the survey, I think.

        # Initiate dictionaries.
        self._receivers = {}
        self._sources = {}
        self._frequencies = {}
        self._data = {}

    def copy(self):
        """Return a copy of the Survey."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Survey in a dict."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Survey` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Survey.to_dict`.
            The dictionary needs the keys TODO.

        Returns
        -------
        obj : :class:`Survey` instance

        """
        raise NotImplementedError

    def add_receivers(self, name, coordinates):
        """Add receiver(s) to the survey.

        Parameters
        ----------
        name : str or list
            Receiver name(s).

        coordinates : tuple
            `(x, y, z, azimuth, dip)` [m, m, m, °, °]; dimensions must be
            compatible (also with `name`).

        """

        # TODO: Add recursion for receiver list.

        # Warn about duplicate names.
        if name in self._receivers:
            print(f"* WARNING :: Overwriting existing receiver <{name}>:\n"
                  f"             - Old: {self._receivers[name].coordinates}\n"
                  f"             - New: {coordinates}")

        # Create a receiver dipole.
        self._receivers[name] = DipoleReceiver(name, coordinates)

    def remove_receivers(self, names):
        # TODO
        # Delete from receiver-list
        # Delete corresponding data for all sources and frequencies
        pass

    def add_sources(self, names, coordinates):
        """Add source(s) to the survey.

        Parameters
        ----------
        names : str or list
            Source name(s).

        coordinates : tuple
            Coordinates in one of the two following formats:

            - `(x, y, z, azimuth, dip)` [m, m, m, °, °];
            - `(x0, x1, y0, y1, z0, z1)` [m, m, m, m, m, m].

            Dimensions must be compatible (also with `name`).

        """

        if isinstance(names, list):
            # TODO: Add recursion for source list.
            for name in names:
                self.add_sources(name, coordinates)

        else:
            # Warn about duplicate names.
            if names in self._sources:
                old_coords = self._sources[names].coordinates
                print(f"* WARNING :: Overwriting existing source <{names}>:\n"
                      f"             - Old: {old_coords}\n"
                      f"             - New: {coordinates}")

            # Create a receiver dipole.
            self._sources[names] = DipoleSource(names, coordinates)

    def remove_sources(self):
        pass

    def add_data(self, source, receiver, frequency, data=None):
        """Add receiver(s) to the survey.

        Parameters
        ----------
        source, receiver : str
            Source and receiver name(s).

        frequencies : float or array
            Frequencies [Hz].

        data : None or array (nsrc, nfreq, nrec)
            Measured data.

        """

        # TODO: Add recursion for receiver list.
        pass

        # # Warn about duplicate names.
        # print(self._receivers)
        # if name in self._receivers:
        #     print(f"* WARNING :: Overwriting existing receiver <{name}>.")
        #
        # else:
        #     # Create a receiver dipole.
        #     self._receivers[name] = DipoleReceiver(name, coordinates)
        #     # (nsrc, nfreq)
        #     # frequencies=frequencies,
        #     # sources=sources,
        #     # data=data,

        # Only name should be required
        # coordinates are optionally, name is enough if once defined
        # frequency and source are optionally
        # data only if frequency and source provided

        # # Move to add
        # self._names = dict()
        # for name in names:
        #     self._names[name] = dict()

    # Todo
    def validate(self):
        # verify sources and receivers, that all exist, and that there are no
        # unused sources or receivers.
        pass

    def shape(self):
        # Return nsrc x nrec x nfreq
        pass

    def size(self):
        # Return actual data size (NOT equals nsrc x nrec x nfreq)
        pass

    def data(self):
        # Store the actual data in an array
        pass

    # TODO: Get measured data
    # get e-field => needs survey
    # get h-field => needs survey


# # Sources and Receivers # #

@dataclass(order=True, unsafe_hash=True)
class PointDipole:
    """Infinitesimal small point dipole.

    Defined by its coordinates (xco, yco, zco), its azimuth (azm), and its dip.

    Not meant to be used directly. Use :class:`DipoleSource` and
    :class:`DipoleReceiver` instead.

    """
    __slots__ = ['name', 'xco', 'yco', 'zco', 'azm', 'dip']
    name: str
    xco: float
    yco: float
    zco: float
    azm: float
    dip: float


class Dipole(PointDipole):
    """Finite length dipole or point dipole.

    Expansion of the basic :class:`PointDipole` to allow for finite length
    dipoles, and to provide coordinate inputs in the form of
    (x, y, z, azimuth, dip) or (x0, x1, y0, y1, z0, z1).

    Adds attributes `is_finite`, `electrode1`, `electrode2`, `length`, and
    `coordinates` to the class.

    The length of a point dipole is set to unity (and the electrode positions
    computed accordingly).

    Not meant to be used directly. Use :class:`DipoleSource` and
    :class:`DipoleReceiver` instead.

    """

    def __init__(self, name, coordinates):
        """Check coordinates and kwargs."""

        # Check coordinates.
        try:
            # Conversion to float-array fails if there are lists and tuples
            # within the tuple, or similar.
            # This should catch many wrong inputs, hopefully.
            coords = np.array(coordinates, dtype=float)

            # Check size => finite or point dipole?
            if coords.size == 5:
                self.is_finite = False

            elif coords.size == 6:
                self.is_finite = True

                # Ensure the two poles are distinct.
                if np.allclose(coords[::2], coords[1::2]):
                    raise ValueError

            else:
                raise ValueError

        except ValueError:
            print("* ERROR   :: Dipole coordinates are wrong defined.\n"
                  "             They must be defined either as a point,\n"
                  "             (x, y, z, azimuth, dip), or as two poles,\n"
                  "             (x0, x1, y0, y1, z0, z1), all floats.\n"
                  "             In the latter, pole0 != pole1.\n"
                  f"             Provided coordinates: {coordinates}.")
            raise ValueError("Dipole coordinates")

        # Get xco, yco, zco, azm, and dip.
        if self.is_finite:

            # Get the two separate electrodes.
            self.electrode1 = tuple(coords[::2])
            self.electrode2 = tuple(coords[1::2])

            # Compute center.
            xco, yco, zco = np.sum(coords.reshape(3, -1), 1)/2

            # Get lengths in each direction.
            dx, dy, dz = np.diff(coords.reshape(3, -1)).ravel()

            # Length of bipole.
            self.length = np.linalg.norm([dx, dy, dz], axis=0)

            # Horizontal deviation from x-axis.
            azm = np.round(np.rad2deg(np.arctan2(dy, dx)), 5)

            # Vertical deviation from xy-plane down.
            dip = np.round(np.rad2deg(np.pi/2-np.arccos(dz/self.length)), 5)

            # (Very small angles set to zero, as, e.g., sin(pi/2) != exact 0)

        else:
            # Get coordinates, angles, and set length.
            xco, yco, zco, azm, dip = tuple(coords)
            self.length = 1.0

            # Get lengths in each direction (total length is 1).
            # (Set very small angles to zero, as, e.g., sin(pi/2) != exact 0)
            dx = np.round(np.cos(np.deg2rad(azm))*np.cos(np.deg2rad(dip)), 5)
            dy = np.round(np.sin(np.deg2rad(azm))*np.cos(np.deg2rad(dip)), 5)
            dz = np.round(np.sin(np.deg2rad(dip)), 5)

            # Get the two separate electrodes.
            self.electrode1 = (xco-dx/2, yco-dy/2, zco-dz/2)
            self.electrode2 = (xco+dx/2, yco+dy/2, zco+dz/2)

        super().__init__(name, xco, yco, zco, azm, dip)

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.name}, "
                f"{{{self.xco}m; {self.yco}m; {self.zco}m}}, θ={self.azm}°, "
                f"φ={self.dip}°, l={self.length}m)")

    @property
    def coordinates(self):
        """Return coordinates.

        In the format (x, y, z, azimuth, dip) or (x0, x1, y0, y1, z0, z1).

        This format is used in many other routines.
        """
        if self.is_finite:
            return (self.electrode1[0], self.electrode2[0],
                    self.electrode1[1], self.electrode2[1],
                    self.electrode1[2], self.electrode2[2])
        else:
            return (self.xco, self.yco, self.zco, self.azm, self.dip)


class DipoleSource(Dipole):
    """Dipole source.


    Parameters
    ----------
    name : str
        Dipole name.

    coordinates : tuple of floats
        Source coordinates, one of the following:

        - (x0, x1, y0, y1, z0, z1): finite length dipole,
        - (x, y, z, azimuth, dip): point dipole.

        The coordinates x, y, and z are in meters (m), the azimuth and dip in
        degree (°).

        Angles (coordinate system is right-handed with positive z up;
        East-North-Depth):

        - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
        - +/-dip (°): vertical deviation from xy-plane down/up-wards.

    strength : float
        Source strength. If `strength=0` (default), the source is normalized to
        a moment (hence source length and source strength) of 1 A m.

    """

    def __init__(self, name, coordinates, **kwargs):

        # Get source strength.
        self.strength = kwargs.pop('strength', 0.0)

        # Warn if any kwargs left; TODO: respect verbosity.
        if kwargs:
            print(f"* WARNING :: Remaining kwargs: {kwargs}")

        super().__init__(name, coordinates)

    def __repr__(self):
        """Add strength to repr."""
        return (f"{super().__repr__()[:-1]}, I={self.strength}A)")


class DipoleReceiver(Dipole):
    """Dipole receiver.


    Parameters
    ----------
    name : str
        Dipole name.

    coordinates : tuple of floats
        Receiver coordinates, one of the following:

        - (x0, x1, y0, y1, z0, z1): finite length dipole,
        - (x, y, z, azimuth, dip): point dipole.

        The coordinates x, y, and z are in meters (m), the azimuth and dip in
        degree (°).

        Angles (coordinate system is right-handed with positive z up;
        East-North-Depth):

        - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
        - +/-dip (°): vertical deviation from xy-plane down/up-wards.

    """

    def __init__(self, name, coordinates, **kwargs):

        # Warn if any kwargs left; TODO: respect verbosity.
        if kwargs:
            print(f"* WARNING :: Remaining kwargs: {kwargs}")

        super().__init__(name, coordinates)
