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
from copy import deepcopy
# from typing import Tuple
from dataclasses import dataclass  # , field

# from emg3d import maps, models

__all__ = ['Survey', 'Dipole']


class Survey:
    """Create a survey with sources and receivers.

    A survey contains sources and receivers, which, on their own, do not say
    anything about data. The data added must then be provided with the info
    at which receiver it was measured, from which source comes the signal, and
    what is the source frequency.



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
        # Maybe these should be _dicts, with @property getters.
        self.receivers = {}
        self.sources = {}
        self.frequencies = {}

        # Data - Create a class
        # stored as an array
        # provides different dicts (views) (basically any combination):
        # - by src-rec-freq
        # - by src-freq-rec
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

    def add_receivers(self, names, coordinates):
        """Add receiver(s) to the survey.

        It accepts point dipoles and finite length dipoles. However, obtaining
        receiver responses is currently only implemented for point dipoles, so
        that representation is used. TODO: Throw a warning when that happens.

        Parameters
        ----------
        name : str or list
            Receiver name(s).

        coordinates : tuple
            `(x, y, z, azimuth, dip)` [m, m, m, °, °]; dimensions must be
            compatible (also with `name`).

        """

        # Recursion for multiple receivers.
        if isinstance(names, list):

            nl = len(names)
            coords = np.array([nl*[val, ] if np.array(val).size == 1 else
                              val for val in coordinates], dtype=float)

            # Loop over names.
            for i, name in enumerate(names):
                self.add_receivers(name, coords[:, i])

        else:
            # Warn about duplicate names.
            if names in self.receivers:
                old_coords = self.receivers[names].coordinates
                print(f"* WARNING :: Overwriting existing receiver <{names}>:"
                      f"\n             - Old: {old_coords}\n"
                      f"             - New: {coordinates}")

            # Create a receiver dipole.
            self.receivers[names] = Dipole(names, coordinates)

    def remove_receivers(self, names):
        # TODO
        # Delete from receiver-list
        # Delete corresponding data for all sources and frequencies
        pass

    def add_sources(self, names, coordinates):
        """Add source(s) to the survey.

        # strength : float
        #     Source strength. If `strength=0` (default), the source is
        #     normalized to a moment (hence source length and source strength)
        #     of 1 A m.

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

        # Recursion for multiple sources.
        if isinstance(names, list):

            nl = len(names)
            coords = np.array([nl*[val, ] if np.array(val).size == 1 else
                              val for val in coordinates], dtype=float)

            # Loop over names.
            for i, name in enumerate(names):
                self.add_sources(name, coords[:, i])

        else:
            # Warn about duplicate names.
            if names in self.sources:
                old_coords = self.sources[names].coordinates
                print(f"* WARNING :: Overwriting existing source <{names}>:\n"
                      f"             - Old: {old_coords}\n"
                      f"             - New: {coordinates}")

            # Create a receiver dipole.
            self.sources[names] = Dipole(names, coordinates)

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

        pass

        # # Warn about duplicate names.
        # print(self.receivers)
        # if name in self.receivers:
        #     print(f"* WARNING :: Overwriting existing receiver <{name}>.")
        #
        # else:
        #     # Create a receiver dipole.
        #     self.receivers[name] = Dipole(name, coordinates)
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

    Not meant to be used directly. Use :class:`Dipole` instead.

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

    For *point dipoles*, this gives it a length of unity (1 m), takes its
    coordinates as center, and computes the two electrode positions.

    For *finite length dipoles* it sets the coordinates to its center and
    computes its length, azimuth, and dip.

    So in the end finite length dipoles and point dipoles have the exactly
    same signature. They can only be distinguished by the attribute
    `is_finite`.


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

    **kwargs : Optional solver options:
        Currently, any other key will be added as attributes to the dipole.

        This is for early development and will change in the future to avoid
        abuse. It also raises a warning if it is an unknown key to keep
        remembering me of that.

    """
    # These are the only kwargs, that do not raise a Warning.
    # These are also the only ones which are (de-)serialized.
    # This is subject to change, and holds during development.
    accepted_keys = ['strength', ]

    def __init__(self, name, coordinates, **kwargs):
        """Check coordinates and kwargs."""

        # Add additional info to the dipole.
        for key, value in kwargs.items():
            # TODO: respect verbosity
            if key not in self.accepted_keys:
                print(f"* WARNING :: Unknown kwargs {{{key}: {value}}}")
            setattr(self, key, value)

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
                f"{{{self.xco:,.1f}m; {self.yco:,.1f}m; {self.zco:,.1f}m}}, "
                f"θ={self.azm:.1f}°, φ={self.dip:.1f}°, "
                f"l={self.length:,.1f}m)")

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

    def copy(self):
        """Return a copy of the Dipole."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Dipole in a dict."""
        out = {'name': self.name, 'coordinates': self.coordinates,
               '__class__': self.__class__.__name__}

        # Add accepted kwargs.
        for key in self.accepted_keys:
            if hasattr(self, key):
                out[key] = getattr(self, key)

        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Dipole` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Dipole.to_dict`.
            The dictionary needs the keys `name` and `coordinates`.

        Returns
        -------
        obj : :class:`Dipole` instance

        """
        try:
            _ = inp.pop('__class__', '')
            name = inp.pop('name')
            coordinates = inp.pop('coordinates')
            return cls(name=name, coordinates=coordinates, **inp)
        except KeyError as e:
            print(f"* ERROR   :: Variable {e} missing in `inp`.")
            raise
