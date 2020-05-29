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
from dataclasses import dataclass

__all__ = ['Survey', 'Dipole']


class Survey:
    """Create a survey with sources and receivers.

    A survey contains sources and receivers, which, on their own, do not say
    anything about data. The data added must then be provided with the info
    at which receiver it was measured, from which source comes the signal, and
    what is the source frequency.


    Parameters
    ----------
    ?

    """

    def __init__(self, name, sources=None, receivers=None, frequencies=None):
        """Initiate a new Survey instance."""

        # Survey name.
        self.name = name

        # Initiate sources.
        if sources is None:
            self.sources = {}
        else:
            self.sources = {
                    k: Dipole.from_dict(v) for k, v in sources.items()}

        # Initiate receivers.
        if receivers is None:
            self.receivers = {}
        else:
            self.receivers = {
                    k: Dipole.from_dict(v) for k, v in receivers.items()}

        # Initiate frequencies. (TODO: dict or list?)
        self.frequencies = {}

        # Data
        # TODO: Create its own class; store actual data as a 1D array.
        #       The dicts should only contain indexes as int16.
        self._data = {}

    def __repr__(self):
        return (f"{self.__class__.__name__}: {self.name}\n"
                f"> {self.shape[0]} sources; {self.shape[1]} receivers; "
                f"{self.shape[2]} frequencies; {self.size} data points.")

    def copy(self):
        """Return a copy of the Survey."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Survey in a dict."""

        out = {'name': self.name, '__class__': self.__class__.__name__}

        # Add sources.
        out['sources'] = {}
        for key, value in self.sources.items():
            out['sources'][key] = value.to_dict()

        # Add receivers.
        out['receivers'] = {}
        for key, value in self.receivers.items():
            out['receivers'][key] = value.to_dict()

        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Survey` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Survey.to_dict`.
            The dictionary needs the keys `name`, `sources`, `receivers`.

        Returns
        -------
        obj : :class:`Survey` instance

        """
        try:
            name = inp.pop('name')
            sources = inp.pop('sources')
            receivers = inp.pop('receivers')
            return cls(name=name, sources=sources, receivers=receivers)
        except KeyError as e:
            print(f"* ERROR   :: Variable {e} missing in `inp`.")
            raise

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
                new_coords = tuple(np.array(coordinates, dtype=float))
                print(f"* WARNING :: Overwriting existing receiver <{names}>:"
                      f"\n             - Old: {old_coords}\n"
                      f"             - New: {new_coords}")

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
                new_coords = tuple(np.array(coordinates, dtype=float))
                print(f"* WARNING :: Overwriting existing source <{names}>:\n"
                      f"             - Old: {old_coords}\n"
                      f"             - New: {new_coords}")

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

        # TODO : expand dims for source, receiver, frequency
        #
        # TODO : different inputs; (nsrc, nrec, nfreq), (nsrc, nfreq), etc

        if source not in self._data:
            self._data[source] = {}

        if receiver not in self._data[source]:
            self._data[source][receiver] = {}

        # Warn about overwriting
        if float(frequency) in self._data[source][receiver]:
            old_data = self._data[source][receiver][frequency]
            if data is None:
                new_data = 'None'
            else:
                new_data = float(data)  # TODO change to complex
            print(f"* WARNING :: {source}-{receiver}-{frequency}: Overwriting "
                  f"{old_data} with {new_data}.")

        if data is None:
            self._data[source][receiver][float(frequency)] = data
        else:  # TODO change to complex
            self._data[source][receiver][float(frequency)] = float(data)

        # Add frequency.
        if float(frequency) not in self.frequencies:
            self.frequencies[float(frequency)] = {}

    # Todo
    def validate(self):
        # verify sources and receivers, that all exist, and that there are no
        # unused sources or receivers.
        pass

    @property
    def shape(self):
        """Return nsrc x nrec x nfreq.

        Note that not all source-receiver-frequency pairs do actually have
        data. Check `size` to see how many data points there are.
        """
        return (len(self.sources), len(self.receivers), len(self.frequencies))

    @property
    def size(self):
        """Return actual data size (does NOT equal nsrc x nrec x nfreq)."""
        # TODO: Again, data should move to a 1D array, and then we can simply
        #       return self.data.size.
        return sum([1 for k1 in self.data for k2 in self.data[k1]
                    for k3 in self.data[k1][k2]])

    @property
    def data(self):
        """Data in the default format, [source][receiver][frequency]."""
        # TODO (see also `sdata()`).
        # - Currently, it is the actual, complex data.
        #   Change to index with unsigned int16 (0-65,535)
        # Currently probably both slow and heavy on RAM.

        return self._data

    def sdata(self, sort='srf'):
        """Return data in arbitrarily nested dictionaries.

        Parameters
        ----------
        sort : str
            String consists of three characters for [s]ource, [r]eceiver, and
            [f]requency.

        Returns
        -------
        data : dict
            Data in a nested dict following the three letters of the
            sort-string: data[character1][character2][character3].

            E.g., if `sort='sfr'`, the returned data is of the form
            `data[source][frequency][receiver]`.

        """
        # TODO (see also `data()`).
        # - store them at first use, delete them if anything changes in the
        #   data or survey.
        # Currently probably both slow and heavy on RAM.

        if sort == 'srf':  # Default: source-receiver-frequency.
            return self._data

        elif sort == 'rsf':  # receiver-source-frequency.
            return self._reorder_dict_abc(self.data, self.receivers, 'ab')

        elif sort == 'sfr':  # source-frequency-receiver.
            return self._reorder_dict_abc(self.data, self.frequencies, 'bc')

        elif sort == 'rfs':  # receiver-frequency-source.
            out = self._reorder_dict_abc(self.data, self.receivers, 'ab')
            return self._reorder_dict_abc(out, self.frequencies, 'bc')

        elif sort == 'fsr':  # frequency-source-receiver.
            out = self._reorder_dict_abc(self.data, self.frequencies, 'bc')
            return self._reorder_dict_abc(out, self.frequencies, 'ab')

        elif sort == 'frs':  # frequency-receiver-source.
            out = self._reorder_dict_abc(self.data, self.receivers, 'ab')
            out = self._reorder_dict_abc(out, self.frequencies, 'bc')
            return self._reorder_dict_abc(out, self.frequencies, 'ab')

        else:
            return NotImplemented

    def _reorder_dict_abc(self, data, keys, swap_type):
        """Re-arrange the levels of a nested dict."""

        if swap_type == 'ab':
            # Swaps out first and second level of ordered dict.
            out = {k2:  # 2nd level keys are new 1st level keys.
                   {k1: data[k1][k2] for k1 in data if k2 in data[k1]}
                   for k2 in keys if np.any([k2 in data[k1] for k1 in data])}

        elif swap_type == 'bc':
            # 'bc' is the same as a 'ab' for each key on the top level.
            out = {}
            for ka in data.keys():
                out[ka] = self._reorder_dict_abc(data[ka], keys, 'ab')

        return out


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
