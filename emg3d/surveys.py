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

    Underlying the survey-class is an xarray, which is basically a regular
    ndarray with axis labels and more.

    This class was developed with a node-based, marine CSEM survey layout in
    mind. It is therefore optimised for that setup, and mostly tested with that
    setup. This means for a number of receivers which measure for all source
    positions. The general layout of the data for such a survey is (L, M, N),
    where `L` is the number of sources, `M` the number of receivers, and `N`
    the number of frequencies::

                             f1
            Rx1 Rx2  .  RxM /   f2
           ┌───┬───┬───┬───┐   /   .
       Tx1 │   │   │   │   │──┐   /   fN
           ├───┼───┼───┼───┤  │──┐   /
       Tx2 │   │   │   │   │──┤  │──┐
           ├───┼───┼───┼───┤  │──┤  │
        .  │   │   │   │   │──┤  │──┤
           ├───┼───┼───┼───┤  │──┤  │
       TxL │   │   │   │   │──┤  │──┤
           └───┴───┴───┴───┘  │──┤  │
              └───┴───┴───┴───┘  │──┤
                 └───┴───┴───┴───┘  │
                    └───┴───┴───┴───┘

    However, the class can also be used for a CSEM streamer-style survey
    layout (by setting `fixed=True`), where there is a moving source with one
    or several receivers at a fixed offset. The layout of the data is then also
    (L, M, N), but here `L` is the number of locations of the only source, `M`
    is the number of receiver-offsets, and `N` is the number of frequencies::

                                        f1
                Offs1     .   OffsM    /   .
              ┌─────────┬───┬─────────┐   /   fN
       TxPos1 │ Rx1-TP1 │ . │ RxM-TP1 │──┐   /
              ├─────────┼───┼─────────┤  │──┐
       TxPos2 │ Rx1-TP2 │ . │ RxM-TP2 │──┤  │
              ├─────────┼───┼─────────┤  │──┤
        .     │ .       │ . │ .       │──┤  │
              ├─────────┼───┼─────────┤  │──┤
       TxPosL │ Rx1-TPL │ . │ RxM-TPL │──┤  │
              └─────────┴───┴─────────┘  │──┤
                 └─────────┴───┴─────────┘  │
                    └─────────┴───┴─────────┘

    This means that even though there is only one source, there are actually
    `L` source dipoles, as each position is treated as a different dipole. The
    number of receiver dipoles in this case is `LxM`.


    .. todo::

        - Implement source strength (if `strength=0` (default), the source is
          normalized to a moment of 1 A m).
        - Reciprocity flag.
        - For data, add noise floor and error.
        - Add an example of the different usages to the gallery.
        - Return receiver coordinates as list for any source.
        - Include logging/verbosity; check with CLI.


    Parameters
    ----------
    name : str
        Name of the survey

    sources, receivers : tuple, list, or dict
        Sources and receivers.

        - Tuples: Coordinates in one of the two following formats:

          - `(x, y, z, azimuth, dip)` [m, m, m, °, °];
          - `(x0, x1, y0, y1, z0, z1)` [m, m, m, m, m, m].

          Dimensions will be expanded (hence, if `n` dipoles, each parameter
          must have length 1 or `n`). These dipoles will be named sequential
          with `Tx###` and `Rx###`.

          The tuple can additionally contain an additional element at the end
          (after `dip` or `z1`), `electric`, a boolean of length 1 or `n`, that
          indicates if the dipoles are electric or magnetic.

        - List: A list of :class:`Dipole`-instances. The names of all dipoles
          in the list must be unique.

        - (Dict: A dict of de-serialized :class:`Dipole`-instances; mainly used
          for loading from file.)

    frequencies : ndarray
        Source frequencies (Hz).

    data : ndarray or None
        The observed data (dtype=complex); must have shape (nsrc, nrec, nfreq)
        or, if `fixed=True`, (nsrc, noff, nfreq).
        If None, it will be initiated with NaN's.

    fixed : bool
        Node-based CSEM survey (`fixed=False`; default) or streamer-type CSEM
        survey (`fixed=True`). In the streamer-type survey, the number of
        `receivers` supplied must be a multiple of the source positions.
        In this case, the receivers are grouped into offsets.


    """
    # Currently, `surveys.ds` contains an :class:`xarray.Dataset`, where
    # `surveys.data` is a shortcut to the :class:`xarray.DataArray`
    # `surveys.ds.data`. As such, the `Survey`-Class has an xarray-dataset as
    # one of its attributes. Probably there would be a cleaner way to simply
    # use xarray instead of a dedicated `Survey`-Class by utilizing, e.g.,
    # :func:`xarray.register_dataset_accessor`.

    def __init__(self, name, sources, receivers, frequencies, data=None,
                 fixed=0):
        """Initiate a new Survey instance."""

        # Store survey name and fixed.
        self.name = name
        self.fixed = fixed

        # Initiate sources.
        self._sources = self._dipole_info_to_dict(sources, 'source')

        # Initiate receivers.
        self._receivers = self._dipole_info_to_dict(receivers, 'receiver')

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
        self._ds.src.attrs['src-dipoles'] = self.sources
        self._ds.rec.attrs['long_name'] = 'Receiver dipole'
        self._ds.rec.attrs['rec-dipoles'] = self.receivers
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
        out['sources'] = {k: v.to_dict() for k, v in self.sources.items()}

        # Add receivers.
        if self.fixed:
            rec = {}
            for key, value in self.receivers.items():
                rec[key] = {k: v.to_dict() for k, v in value.items()}
        else:
            rec = {k: v.to_dict() for k, v in self.receivers.items()}
        out['receivers'] = rec

        # Add frequencies.
        out['frequencies'] = self.frequencies

        # Add data.
        out['data'] = self.data.values

        # Fixed.
        out['fixed'] = int(self.fixed)

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
            `frequencies`, `data`, and `fixed`.

        Returns
        -------
        obj : :class:`Survey` instance

        """
        try:
            return cls(name=inp['name'], sources=inp['sources'],
                       receivers=inp['receivers'],
                       frequencies=inp['frequencies'], data=inp['data'],
                       fixed=bool(inp['fixed']))

        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.")

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
    def rec_coords(self):
        """Return receiver coordinates.

        The returned format is `[x, y, z, azm, dip]`, a list of 5 tuples. If
        `fixed=True` it returns a dict with the offsets as keys, and for each
        offset it returns the corresponding receiver coordinates as just
        outlined.
        """

        # Get receiver coordinates depending if fixed or not.
        if self.fixed:
            coords = {}
            for src in self.sources.keys():
                coords[src] = tuple(
                        np.array([[self.receivers[off][src].xco,
                                   self.receivers[off][src].yco,
                                   self.receivers[off][src].zco,
                                   self.receivers[off][src].azm,
                                   self.receivers[off][src].dip]
                                  for off in self.receivers.keys()]).T)
        else:
            coords = tuple(np.array([[r.xco, r.yco, r.zco, r.azm, r.dip] for r
                                     in self.receivers.values()]).T)

        return coords

    @property
    def frequencies(self):
        """Frequency array."""
        return self._frequencies

    def _dipole_info_to_dict(self, inp, name):
        """Create dict with provided source/receiver information."""

        # Create dict depending if `inp` is list, tuple, or dict.
        if isinstance(inp, list):  # List of Dipoles

            if self.fixed and name == 'receiver':  # Streamer-type receivers.

                # Get dimensions.
                nd = len(inp)
                ns = len(self.sources)  # Number of source position.
                nr = nd//ns             # Number of receivers / source.
                dnr = len(str(nr-1))    # Max number of digits; rec.

                # Create name lists.
                rec_names = [f"{i:0{dnr}d}" for i in range(nr)]
                src_names = list(self.sources.keys())

                # Ensure receivers are multiples of source positions.
                if nd % ns != 0:
                    raise ValueError(
                            "For fixed surveys, the number of receivers\n"
                            "must be a multiple of number of sources.\n"
                            f"Provided: #src: {ns}; #rec: {nd}.")

                # Assemble dict.
                out = {'Off'+name: {} for name in rec_names}
                for i, key in enumerate(out.keys()):
                    for ii, src_name in enumerate(src_names):
                        out[key][src_name] = inp[ii + i*ns]

            else:

                out = {d.name: d for d in inp}

                # Ensure that all names were unique:
                if len(out) != len(inp):
                    raise ValueError(
                            f"There are duplicate {name} names.\n"
                            f"Provided {name}s: {len(inp)}; "
                            f"unique names: {len(out)}.")

        elif isinstance(inp, tuple):  # Tuple with coordinates

            # See if last tuple element is boolean, hence el/mag-flag.
            if isinstance(inp[-1], (list, tuple, np.ndarray)):
                provided_elmag = isinstance(inp[-1][0], bool)
            else:
                provided_elmag = isinstance(inp[-1], bool)

            # Get max dimension.
            nd = max([np.array(n, ndmin=1).size for n in inp])

            # Expand coordinates.
            coo = np.array([nd*[val, ] if np.array(val).size == 1 else
                           val for val in inp], dtype=float)

            # Extract el/mag flag or set to ones (electric) if not provided.
            if provided_elmag:
                elmag = coo[-1, :]
                coo = coo[:-1, :]
            else:
                elmag = np.ones(nd)

            # Create dipole names (number-strings).
            prefix = 'Tx' if name == 'source' else 'Rx'
            dnd = len(str(nd-1))  # Max number of digits.
            names = [f"{prefix}{i:0{dnd}d}" for i in range(nd)]

            # Create Dipole-dict.
            if self.fixed and name == 'receiver':  # Streamer-type receivers.

                # Get dimensions.
                ns = len(self.sources)  # Number of source position.
                nr = nd//ns             # Number of receivers / source.
                dnr = len(str(nr-1))    # Max number of digits; rec.

                # Create name lists.
                rec_names = [f"{i:0{dnr}d}" for i in range(nr)]
                src_names = list(self.sources.keys())

                # Ensure receivers are multiples of source positions.
                if nd % ns != 0:
                    raise ValueError(
                            "For fixed surveys, the number of receivers\n"
                            "must be a multiple of number of sources.\n"
                            f"Provided: #src: {ns}; #rec: {nd}.")

                # Assemble dict.
                out = {'Off'+rec_name: {} for rec_name in rec_names}
                for i, key in enumerate(out.keys()):
                    for ii, src_name in enumerate(src_names):
                        iii = ii + i*ns
                        out[key][src_name] = Dipole(
                                names[iii], coo[:, iii], elmag[iii])

            else:  # Default node-type src-rec comb. and src for streamer-type.
                out = {names[i]: Dipole(names[i], coo[:, i], elmag[i])
                       for i in range(nd)}

        elif isinstance(inp, dict):  # Dict of de-serialized Dipoles.
            if self.fixed and name == 'receiver':
                out = {}
                for k, v in inp.items():
                    out[k] = {k2: Dipole.from_dict(v2) for k2, v2 in v.items()}
            else:
                out = {k: Dipole.from_dict(v) for k, v in inp.items()}

        else:
            raise ValueError(
                    f"Input format of <{name}s> not recognized: {type(inp)}.")

        return out


# # Sources and Receivers # #
@dataclass(order=True, unsafe_hash=True)
class PointDipole:
    """Infinitesimal small electric or magnetic point dipole.

    Defined by its coordinates (xco, yco, zco), its azimuth (azm), its dip, and
    its type (electric).

    Not meant to be used directly. Use :class:`Dipole` instead.

    """
    __slots__ = ['name', 'xco', 'yco', 'zco', 'azm', 'dip', 'electric']
    name: str
    xco: float
    yco: float
    zco: float
    azm: float
    dip: float
    electric: bool


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

    electric : bool
        Electric dipole if True, magnetic dipole otherwise. Default is True.

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

    def __init__(self, name, coordinates, electric=True, **kwargs):
        """Check coordinates and kwargs."""

        # Add additional info to the dipole.
        for key, value in kwargs.items():
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
            raise ValueError(
                    "Dipole coordinates are wrong defined. They must be\n"
                    "defined either as a point, (x, y, z, azimuth, dip), or\n"
                    "as two poles, (x0, x1, y0, y1, z0, z1), all floats.\n"
                    "In the latter, pole0 != pole1.\n"
                    f"Provided coordinates: {coordinates}.")

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

        super().__init__(name, xco, yco, zco, azm, dip, bool(electric))

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.name}, "
                f"{['H', 'E'][self.electric]}, "
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
               'electric': self.electric, '__class__': self.__class__.__name__}

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
            Dictionary as obtained from :func:`Dipole.to_dict`. The dictionary
            needs the keys `name`, `coordinates`, and `electric`.

        Returns
        -------
        obj : :class:`Dipole` instance

        """
        try:
            kwargs = {k: v for k, v in inp.items() if k in cls.accepted_keys}
            return cls(name=inp['name'], coordinates=inp['coordinates'],
                       electric=inp['electric'], **kwargs)
        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.")
