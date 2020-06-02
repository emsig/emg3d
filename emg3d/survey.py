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
import xarray as xr
from copy import deepcopy
from dataclasses import dataclass

__all__ = ['Survey', 'Dipole', 'PointDipole']


class Survey:
    """Create a survey with sources and receivers.

    A survey contains all the sources with their frequencies, receivers, and
    corresponding data.


    .. todo::

        - Implement source strength (if `strength=0` (default), the source is
          normalized to a moment of 1 A m).
        - Reciprocity flag.
        - Add more data than just `data`: `noise`, `active`, etc.

    .. note::

        This survey is optimised for a node-based, marine CSEM survey. It
        stores the date in an array of size (nsrc, nrec, nfreq), it is
        therefore most compact if each receiver has measurements for each
        source and each frequency. NaN's are placed where there is no data. The
        survey is therefore not good for a survey-type like a streamer-based
        CSEM survey, as there would be a huge matrix required with mostly
        NaN's. If this is required either a new survey-class has to be created,
        or this one has to be adjusted. Probably with a :class:`Dataset` where
        each source is a new :class:`DataArray`.


    Parameters
    ----------
    name : str
        Name of the survey

    sources, receivers : tuple, list, or dict
        Sources and receivers.

        - Tuples: Coordinates in one of the two following formats:

          - `(name, x, y, z, azimuth, dip)` [str, m, m, m, °, °];
          - `(name, x0, x1, y0, y1, z0, z1)` [str, m, m, m, m, m, m].

          Dimensions will be expanded (hence, if `n` dipoles, each parameters
          must have length 1 or `n`; only `name` must be of length `n`).
        - List: A list of :class:`Dipole`-instances.
        - Dict: A dict of de-serialized :class:`Dipole`-instances; mainly used
          for loading from file.

        Warning: names are not checked for uniqueness. Hence, if the same name
        is provided more than once it will be overwritten.

    frequencies : ndarray
        Source frequencies (Hz).

    data : ndarray or None
        The observed data (dtype=complex); must have shape (nsrc, nrec, nfreq).
        If None, it will be initiated with NaN's.


    """
    # Currently, `survey.ds` contains an :class:`xarray.Dataset`, where
    # `survey.data` is a shortcut to the :class:`xarray.DataArray`
    # `survey.ds.data`. As such, the `Survey`-Class has an xarray-dataset as
    # one of its attributes. Probably there would be a cleaner way to simply
    # use xarray instead of a dedicated `Survey`-Class by utilizing, e.g.,
    # :func:`xarray.register_dataset_accessor`.

    def __init__(self, name, sources, receivers, frequencies, data=None):
        """Initiate a new Survey instance."""

        # Survey name.
        self.name = name

        # Initiate sources.
        self._sources = self._dipole_info_to_dict(sources, 'sources')

        # Initiate receivers.
        self._receivers = self._dipole_info_to_dict(receivers, 'receivers')

        # Initiate frequencies.
        self._frequencies = np.array(frequencies, dtype=float, ndmin=1)

        # Initialize NaN-data if not provided.
        if data is None:
            data = np.ones((len(self._sources), len(self._receivers),
                            self._frequencies.size), dtype=complex)*np.nan

        # Initialize xarray dataset.
        self._ds = xr.Dataset(
            {'data': xr.DataArray(data, dims=('src', 'rec', 'freq'))},
            coords={'src': list(self.sources.keys()),
                    'rec': list(self.receivers.keys()),
                    'freq': list(self.frequencies)},
        )
        self._ds.src.attrs['long_name'] = 'Source dipole'
        self._ds.src.attrs['dipoles'] = self.sources
        self._ds.rec.attrs['long_name'] = 'Receiver dipole'
        self._ds.rec.attrs['dipoles'] = self.receivers
        self._ds.freq.attrs['long_name'] = 'Source frequency'
        self._ds.freq.attrs['units'] = 'Hz'

    def __repr__(self):
        return (f"{self.__class__.__name__}: {self.name}\n\n"
                f"{self.ds.__repr__()}")

    def _repr_html_(self):
        return (f"<h4>{self.__class__.__name__}: {self.name}</h4><br>"
                f"{self.ds._repr_html_()}")

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

        # Add frequencies.
        out['frequencies'] = self.frequencies

        # Add data.
        out['data'] = self.data.values

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
            The dictionary needs the keys `name`, `sources`, `receivers`
            `frequencies`, and `data`.

        Returns
        -------
        obj : :class:`Survey` instance

        """
        try:
            return cls(name=inp['name'], sources=inp['sources'],
                       receivers=inp['receivers'],
                       frequencies=inp['frequencies'], data=inp['data'])

        except KeyError as e:
            print(f"* ERROR   :: Variable {e} missing in `inp`.")
            raise

    @property
    def shape(self):
        """Return nsrc x nrec x nfreq.

        Note that not all source-receiver-frequency pairs do actually have
        data. Check `size` to see how many data points there are.
        """
        return self.ds.data.shape

    @property
    def size(self):
        """Return actual data size (does NOT equal nsrc x nrec x nfreq)."""
        return int(self.ds.data.count())

    @property
    def ds(self):
        """Dataset, an :class:`xarray.Dataset` instance.."""
        return self._ds

    @property
    def data(self):
        """Observed data, an :class:`xarray.DataArray` instance.."""
        return self.ds.data

    @property
    def sources(self):
        """Source dict containing all source dipoles."""
        return self._sources

    @property
    def receivers(self):
        """Receiver dict containing all receiver dipoles."""
        return self._receivers

    @property
    def frequencies(self):
        """Frequency array."""
        return self._frequencies

    def _dipole_info_to_dict(self, inp, name):
        """Create dict with provided source/receiver information."""

        # Create dict depending if `inp` is list, tuple, or dict.
        if isinstance(inp, list):  # List of Dipoles
            out = {d.name: d for d in inp}

        elif isinstance(inp, tuple):  # Tuple with names, coordinates

            # Get names.
            names = inp[0]
            if not isinstance(names, list):
                names = [names]
            nl = len(names)

            # Expand coordinates.
            coo = np.array([nl*[val, ] if np.array(val).size == 1 else
                           val for val in inp[1:]], dtype=float)

            # Create Dipole-dict.
            out = {names[i]: Dipole(names[i], coo[:, i]) for i in range(nl)}

        elif isinstance(inp, dict):  # Dict of de-serialized Dipoles.
            out = {k: Dipole.from_dict(v) for k, v in inp.items()}

        else:
            print(f"* ERROR   :: Input format of <{name}> not recognized: "
                  f"{type(inp)}.")
            raise ValueError("Dipoles")

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
