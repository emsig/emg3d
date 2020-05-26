"""

:mod:`survey` -- Survey, sources, and receivers
===============================================

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
from dataclasses import dataclass, field

# from emg3d import maps, models

# __all__ = []


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


@dataclass(eq=False)
class Dipole:
    """Electric dipole, source or receiver, finite or point.

    Parameters
    ----------
    name : str
        Dipole name.

    x, y, z : float
        Coordinates of the center of the dipole (m).

    azimuth, dip : float
        Angles (coordinate system is either left-handed with positive z down or
        right-handed with positive z up; East-North-Depth):

        - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
        - +/-dip (°): vertical deviation from xy-plane down/up-wards.

    length : float
        Length of (finite or point) dipole (m)

    is_finite : bool
        If True, dipole is a finite length dipole; if False, dipole is a point
        dipole.

    TODO to self: Use validators as soon as dataclasses have them.

    """
    name: str
    xco: float = field(default=0.0, metadata={'units': 'm'})
    yco: float = field(default=0.0, metadata={'units': 'm'})
    zco: float = field(default=0.0, metadata={'units': 'm'})
    azm: float = field(default=0.0, metadata={'units': '°'})
    dip: float = field(default=0.0, metadata={'units': '°'})
    length: float = field(default=1.0, metadata={'units': 'm'})
    is_finite: bool = field(default=False, repr=False)


# TODO:
# - Combine PointDipole and FiniteDipole again
#   - Convert to dataclass; very simple, NO @property
#   - No kwargs
# NO, BETTER: Make `Dipole`-> `PointDipole`
#             Dipole(PointDipole)
# - Write simple DipoleSource and DipoleReceiver wrappers.
#   - Move docstrings to this one

class PointDipole(Dipole):
    """Electric infinitesimal small dipole, source or receiver."""

    def __init__(self, name, coordinates, **kwargs):
        """Check coordinates and kwargs."""

        # Warn if any kwargs left; TODO: respect verbosity.
        if kwargs:
            print(f"* WARNING :: Remaining kwargs: {kwargs}")

        # Check coordinates.
        try:
            # Conversion to float-array fails if there are lists and tuples
            # within the tuple, or similar.
            # This should catch many wrong inputs, hopefully.
            coords = np.array(coordinates, dtype=float)

            # Check size:
            if coords.size != 5:
                raise ValueError

        except ValueError:
            print("* ERROR   :: Point-dipole coordinates are wrong defined.\n"
                  "             They must be defined in the following way:\n"
                  "             (x, y, z, azimuth, dip), all floats.\n"
                  f"             Provided coordinates: {coordinates}.")
            raise ValueError("Point-dipole coordinates")

        super().__init__(name, *coordinates, length=1.0, is_finite=False)

    @property
    def coordinates(self):
        """In the format (x, y, z, azimuth, dip)."""
        return (self.xco, self.yco, self.zco, self.azm, self.dip)


class FiniteDipole(Dipole):
    """Electric finite length dipole, source or receiver."""

    def __init__(self, name, coordinates, **kwargs):
        """Check coordinates and kwargs."""

        # Check coordinates.
        try:
            # Conversion to float-array fails if there are lists and tuples
            # within the tuple, or similar.
            # This should catch many wrong inputs, hopefully.
            coords = np.array(coordinates, dtype=float)

            # Check size:
            if coords.size != 6:
                raise ValueError

        except ValueError:
            print("* ERROR   :: Finite-dipole coordinates are wrong defined.\n"
                  "             They must be defined in the following way:\n"
                  "             (x0, x1, y0, y1, z0, z1), all floats.\n"
                  f"             Provided coordinates: {coordinates}.")
            raise ValueError("Finite-dipole coordinates")

        # Add electrodes.
        self.electrode1 = tuple(coords[::2])
        self.electrode2 = tuple(coords[1::2])

        # Compute center.
        center = tuple(np.sum(coords.reshape(3, -1), 1)/2)

        # Get lengths in each direction
        dx, dy, dz = np.diff(coords.reshape(3, -1)).ravel()

        # Length of bipole
        length = np.linalg.norm([dx, dy, dz], axis=0)

        # Horizontal deviation from x-axis
        azm = np.arctan2(dy, dx)

        # Vertical deviation from xy-plane down
        dip = np.pi/2-np.arccos(dz/length)

        super().__init__(name, *center, azm, dip, length, True)

    @property
    def coordinates(self):
        """In the format (x0, x1, y0, y1, z0, z1)."""
        return (self.electrode1[0], self.electrode2[0],
                self.electrode1[1], self.electrode2[1],
                self.electrode1[2], self.electrode2[2])


class DipoleSource(Dipole):
    """
    Maybe this should be a dataclass.
    """

    def __init__(self, name, coordinates, **kwargs):

        super().__init__(name, coordinates, **kwargs)


class DipoleReceiver(PointDipole):
    """
    Maybe this should be a dataclass, and only data should be a dataclass.
    """
    def __init__(self, name, coordinates, **kwargs):

        # `Dipole` permits len in [5, 6].
        # However, receivers are only implemented as points so far.

        if len(coordinates) == 6:
            print("* ERROR   :: Receivers are only implemented for point "
                  "dipoles\n"
                  "             at the moment, but finite length coordinates\n"
                  "             (x1, x2, y1, y2, z1, z2) were provided.")
            # TODO don't fail with an error, convert to point dipoles!
            raise ValueError("coordinates")

        super().__init__(name, coordinates, **kwargs)
