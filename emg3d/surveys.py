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

import numpy as np

try:
    import xarray
except ImportError:
    xarray = None

from emg3d import electrodes, utils, io

__all__ = ['Survey', ]


@utils.known_class
class Survey:
    """Create a survey with sources, receivers, and data.

    A survey contains all data with the corresponding information about
    sources, receivers, and frequencies. The data is a 3D-array of shape
    (nsrc, nrec, nfreq).

    Underlying the survey-class is an xarray, which is basically a regular
    ndarray with axis labels and more. The module `xarray` is a soft
    dependency, and has to be installed manually to use the `Survey`
    functionality.

    The data is stored in an ndarray of the form `nsrc`x`nrec`x`nfreq` in the
    following way::

                             f1
            Rx1 Rx2  .  RxR /   f2
           ┌───┬───┬───┬───┐   /   .
       Tx1 │   │   │   │   │──┐   /   fF
           ├───┼───┼───┼───┤  │──┐   /
       Tx2 │   │   │   │   │──┤  │──┐
           ├───┼───┼───┼───┤  │──┤  │
        .  │   │   │   │   │──┤  │──┤
           ├───┼───┼───┼───┤  │──┤  │
       TxS │   │   │   │   │──┤  │──┤
           └───┴───┴───┴───┘  │──┤  │
              └───┴───┴───┴───┘  │──┤
                 └───┴───┴───┴───┘  │
                    └───┴───┴───┴───┘

    Receivers have a switch ``relative``, which is False by default and means
    that the coordinates are absolute values. If the switch is set to True, the
    coordinates are relative to the source. As such, the above layout can also
    be used for a moving source at positions Tx1, Tx2, ..., where the receivers
    Rx1, Rx2, ... have a constant offset from the source.


    Parameters
    ----------
    sources, receivers : tuple or dict
        Sources and receivers.

        - Tuples: Coordinates in one of the two following formats:

          - `(x, y, z, azimuth, elevation)` [m, m, m, °, °];
          - `(x0, x1, y0, y1, z0, z1)` [m, m, m, m, m, m].

          Dimensions will be expanded (hence, if `n` dipoles, each parameter
          must have length 1 or `n`). These dipoles will be named sequential
          with `Tx###` and `Rx###`.

          The tuple can additionally contain an additional element at the end
          (after `elevation` or `z1`), `electric`, a boolean of length 1 or
          `n`, that indicates if the dipoles are electric or magnetic.

        - Dictionary: A dict where the values are :class:`Dipole`-instances,
          de-serialized or not.

    frequencies : ndarray or dict
        Source frequencies (Hz).

        - ndarray : (or tuple, list): Frequencies will be stored in a dict with
          keys assigned starting with 'f0', 'f1', and so on.

        - dict: keys can be arbitrary names, values must be floats.

    data : ndarray, optional
        The observed data (dtype=np.complex128); must have shape (nsrc, nrec,
        nfreq). Alternatively, it can be a dict containing many datasets, in
        which one could also store, for instance, standard-deviations for each
        source-receiver-frequency pair.

        If None, it will be initiated with NaN's.

    noise_floor, relative_error : float or ndarray, optional
        Noise floor and relative error of the data. Default to None.
        They can be arrays of a shape which can be broadcasted to the data
        shape, e.g., (nsrc, 1, 1) or (1, nrec, nfreq), or have the dimension of
        data.
        See :attr:`Survey.standard_deviation` for more info.

    name : str, optional
        Name of the survey.

    date : str, optional
        Acquisition date.

    info : str, optional
        Survey info or any other info (e.g., what was the intent of the survey,
        what were the acquisition conditions, problems encountered).

    """

    def __init__(self, sources, receivers, frequencies, data=None, **kwargs):
        """Initiate a new Survey instance."""

        # Initiate sources, receivers, and frequencies.
        self._sources = _dipole_info_to_dict(sources, 'source')
        self._receivers = _dipole_info_to_dict(receivers, 'receiver')
        out = _frequency_info_to_dict(frequencies)
        self._frequencies, self._freq_dkeys, self._freq_array = out

        # Initialize xarray dataset.
        self._initiate_dataset(data)

        # Get the optional keywords related to standard deviation.
        self.noise_floor = kwargs.pop('noise_floor', None)
        self.relative_error = kwargs.pop('relative_error', None)

        # Get the optional info.
        self.name = kwargs.pop('name', None)
        self.date = kwargs.pop('date', None)
        self.info = kwargs.pop('info', None)

        # Ensure no kwargs left.
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

    @utils._requires('xarray')
    def _initiate_dataset(self, data):
        """Initiate Dataset; wrapped in fct to check if xarray is installed."""

        # Shape of DataArrays.
        shape = (len(self._sources), len(self._receivers),
                 len(self._frequencies))

        # Initialize NaN-data if not provided.
        if data is None:
            data = {'observed': np.full(shape, np.nan+1j*np.nan)}

        elif not isinstance(data, dict):
            data = {'observed': np.atleast_3d(data)}

        # Ensure we have 'observed' data:
        if 'observed' not in data.keys():
            data['observed'] = np.full(shape, np.nan+1j*np.nan)

        # Create Dataset.
        dims = ('src', 'rec', 'freq')
        self._data = xarray.Dataset(
            {k: xarray.DataArray(v, dims=dims) for k, v in data.items()},
            coords={'src': list(self.sources.keys()),
                    'rec': list(self.receivers.keys()),
                    'freq': list(self.frequencies)},
        )

        # Add attributes.
        self._data.src.attrs['Sources'] = "".join(
                f"{k}: {s.__repr__()};\n" for k, s in self.sources.items())
        self._data.rec.attrs['Receivers'] = "".join(
                f"{k}: {d.__repr__()};\n" for k, d in self.receivers.items())
        self._data.freq.attrs['Frequencies'] = "".join(
                f"{k}: {f};\n" for k, f in self.frequencies.items())
        self._data.freq.attrs['units'] = 'Hz'

    def __repr__(self):
        name = f"  Name: {self.name}\n" if self.name else ""
        date = f"  Date: {self.date}\n" if self.date else ""
        info = f"  Info: {self.info}\n" if self.info else ""
        return (f"{self.__class__.__name__}\n{name}{date}{info}\n"
                f"{self.data.__repr__()}")

    def _repr_html_(self):
        name = f"Name: {self.name}<br>" if self.name else ""
        date = f"Date: {self.date}<br>" if self.date else ""
        info = f"Info: {self.info}<br>" if self.info else ""
        return (f"<h4>{self.__class__.__name__}</h4><br>{name}{date}{info}"
                f"{self.data._repr_html_()}")

    def copy(self):
        """Return a copy of the Survey."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Survey in a dict."""

        out = {'__class__': self.__class__.__name__}

        # Add sources.
        out['sources'] = {k: v.to_dict() for k, v in self.sources.items()}

        # Add receivers.
        rec = {k: v.to_dict() for k, v in self.receivers.items()}
        out['receivers'] = rec

        # Add frequencies.
        out['frequencies'] = self.frequencies

        # Add data.
        out['data'] = {}
        for key in self.data.keys():
            out['data'][key] = self.data[key].data

        # Add `noise_floor` and `relative error`.
        out['noise_floor'] = self.data.noise_floor
        out['relative_error'] = self.data.relative_error

        # Add info.
        out['name'] = self.name
        out['date'] = self.date
        out['info'] = self.info

        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    @utils._requires('xarray')
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Survey` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Survey.to_dict`.
            The dictionary needs the keys `sources`, `receivers` `frequencies`,
            and `data`.

        Returns
        -------
        obj : :class:`Survey` instance

        """
        try:
            # Optional parameters.
            opt = ['noise_floor', 'relative_error', 'name', 'info', 'date']

            # Initiate survey.
            out = cls(sources=inp['sources'],
                      receivers=inp['receivers'],
                      frequencies=inp['frequencies'],
                      data=inp['data'],
                      **{k: inp[k] if k in inp.keys() else None for k in opt})

            return out

        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.") from e

    def to_file(self, fname, name='survey', **kwargs):
        """Store Survey to a file.

        Parameters
        ----------
        fname : str
            File name inclusive ending, which defines the used data format.
            Implemented are currently:

            - `.h5` (default): Uses `h5py` to store inputs to a hierarchical,
              compressed binary hdf5 file. Recommended file format, but
              requires the module `h5py`. Default format if ending is not
              provided or not recognized.
            - `.npz`: Uses `numpy` to store inputs to a flat, compressed binary
              file. Default format if `h5py` is not installed.
            - `.json`: Uses `json` to store inputs to a hierarchical, plain
              text file.

        name : str
            Name under which the survey is stored within the file.

        kwargs : Keyword arguments, optional
            Passed through to :func:`io.save`.

        """
        kwargs[name] = self                # Add survey to dict.
        return io.save(fname, **kwargs)

    @classmethod
    @utils._requires('xarray')
    def from_file(cls, fname, name='survey', **kwargs):
        """Load Survey from a file.

        Parameters
        ----------
        fname : str
            File name including extension. Used backend depends on the file
            extensions:

            - '.npz': numpy-binary
            - '.h5': h5py-binary (needs `h5py`)
            - '.json': json

        name : str
            Name under which the survey is stored within the file.

        kwargs : Keyword arguments, optional
            Passed through to :func:`io.load`.


        Returns
        -------
        survey : :class:`Survey`
            The survey that was stored in the file.

        """
        out = io.load(fname, **kwargs)
        if 'verb' in kwargs and kwargs['verb'] < 0:
            return out[0][name], out[1]
        else:
            return out[name]

    def select(self, sources=None, receivers=None, frequencies=None):
        """Return a survey with selectod sources, receivers, and frequencies.


        Parameters
        ----------
        sources, receivers, frequencies : list
            Lists containing the wanted sources, receivers, and frequencies.


        Returns
        -------
        subsurvey : :class:`emg3d.surveys.Survey`
            Survey with selected data.

        """

        # Get a dict of the survey
        survey = self.to_dict()
        selection = {}

        # Select sources.
        if sources is not None:
            if isinstance(sources, str):
                sources = [sources, ]
            survey['sources'] = {s: survey['sources'][s] for s in sources}
            selection['src'] = sources

        # Select receivers.
        if receivers is not None:
            if isinstance(receivers, str):
                receivers = [receivers, ]
            survey['receivers'] = {
                    r: survey['receivers'][r] for r in receivers}
            selection['rec'] = receivers

        # Select frequencies.
        if frequencies is not None:
            if isinstance(frequencies, str):
                frequencies = [frequencies, ]
            survey['frequencies'] = {
                    r: survey['frequencies'][r] for r in frequencies}
            selection['freq'] = frequencies

        # Replace data with selected data.
        for key in survey['data'].keys():
            survey['data'][key] = self.data[key].sel(**selection)

        # Return reduced survey.
        return Survey.from_dict(survey)

    @property
    def shape(self):
        """Return nsrc x nrec x nfreq.

        Note that not all source-receiver-frequency pairs do actually have
        data. Check `size` to see how many data points there are.
        """
        return self.data.observed.shape

    @property
    def size(self):
        """Return actual data size (does NOT equal nsrc x nrec x nfreq)."""
        return int(self.data.observed.count())

    @property
    def data(self):
        """Data, a :class:`xarray.DataSet` instance.

        Contains the :class:`xarray.DataArray` element `.observed`, but other
        data can be added. E.g., :class:`emg3d.simulations.Simulation` adds the
        `synthetic` array.
        """
        return self._data

    @property
    def sources(self):
        """Source dict containing all source dipoles."""
        return self._sources

    @property
    def receivers(self):
        """Receiver dict containing all receiver dipoles."""
        return self._receivers

    @property
    def src_coords(self):
        """Return source coordinates.

        The returned format is `(x, y, z, azimuth, elevation)`, a tuple of 5
        tuples.
        """

        return tuple(
            np.array([[s.center[0], s.center[1], s.center[2], s.azimuth,
                       s.elevation] for s in self.sources.values()]).T
        )

    @property
    def rec_coords(self):
        """Return receiver coordinates as `(x, y, z, azimuth, elevation)`."""
        return tuple(
            np.array([[r.center[0], r.center[1], r.center[2], r.azimuth,
                       r.elevation] for r in self.receivers.values()]).T
        )

    @property
    def rec_types(self):
        """Return receiver flags if electric, as tuple."""
        return tuple([r.xtype == 'electric' for r in self.receivers.values()])

    @property
    def frequencies(self):
        """Frequency dict containing all frequencies."""
        return self._frequencies

    @property
    def freq_array(self):
        """Return frequencies as tuple."""
        return tuple(self._freq_array)

    def _freq_key(self, frequency):
        """Return key of `frequency`, where frequency is str (key) or float."""
        if isinstance(frequency, str):
            return frequency
        else:
            return self._freq_dkeys[frequency]

    @property
    def standard_deviation(self):
        r"""Return the standard deviation of the data.

        The standard deviation can be set by providing an array of the same
        dimension as the data itself:

        .. code-block:: python

            survey.standard_deviation = ndarray  # (nsrc, nrec, nfreq)

        Alternatively, one can set the `noise_floor` :math:`\epsilon_\text{nf}`
        and the `relative_error` :math:`\epsilon_\text{r}`:

        .. code-block:: python

            survey.noise_floor = float or ndarray      # (> 0 or None)
            survey.relative error = float or ndarray   # (> 0 or None)

        They must be either floats, or three-dimensional arrays of shape
        ``([nsrc or 1], [nrec or 1], [nfreq or 1])``; dimensions of one will be
        broadcasted to the corresponding size. E.g., for a dataset of arbitrary
        amount of sources and receivers with three frequencies you can define
        a purely frequency-dependent relative error via
        ``relative_error=np.array([err_f1, err_f2, err_f3])[None, None, :]``.

        The standard deviation :math:`\varsigma_i` of observation :math:`d_i`
        is then given in terms of the noise floor
        :math:`\epsilon_{\text{nf};i}` and the relative error
        :math:`\epsilon_{\text{re};i}` by

        .. math::
            :label: std

            \varsigma_i = \sqrt{
                \epsilon_{\text{nf}; i}^2 +
                \left(\epsilon_{\text{re}; i}|d_i|\right)^2 } \, .

        Note that a set standard deviation is prioritized over potentially also
        defined noise floor and relative error. To use the noise floor and the
        relative error after defining standard deviation directly you would
        have to reset it like

        .. code-block:: python

            survey.standard_deviation = None

        after which Equation :eq:`std` would be used again.

        """
        # If `std` was set, return it, else compute it from noise_floor and
        # relative_error.
        if 'standard_deviation' in self._data.keys():
            return self.data['standard_deviation']

        elif self.noise_floor is not None or self.relative_error is not None:

            # Initiate std (xarray of same type as the observed data)
            std = self.data.observed.copy(data=np.zeros(self.shape))

            # Add noise floor if given.
            if self.noise_floor == 'data._noise_floor':
                std += self.data._noise_floor**2
            elif self.noise_floor is not None:
                std += self.noise_floor**2

            # Add relative error if given.
            if self.relative_error == 'data._relative_error':
                std += np.abs(self.data._relative_error*self.data.observed)**2
            elif self.relative_error is not None:
                std += np.abs(self.relative_error*self.data.observed)**2

            # Return.
            return np.sqrt(std)

        else:
            # If nothing is defined, return None
            return None

    @standard_deviation.setter
    def standard_deviation(self, standard_deviation):
        """Update standard deviation."""
        # If None it means basically to delete it; otherwise set it.
        if standard_deviation is None and 'standard_deviation' in self.data:
            del self._data['standard_deviation']
        elif standard_deviation is not None:
            # Ensure all values are bigger than zero.
            if np.any(standard_deviation <= 0.0):
                raise ValueError(
                    "All values of `standard_deviation` must be bigger "
                    "than zero."
                )
            self._data['standard_deviation'] = self.data.observed.copy(
                    data=standard_deviation)

    @property
    def noise_floor(self):
        r"""Return the noise floor of the data.

        See :attr:`emg3d.surveys.Survey.standard_deviation` for more info.

        """
        return self.data.noise_floor

    @noise_floor.setter
    def noise_floor(self, noise_floor):
        """Update noise floor.

        See :attr:`Survey.standard_deviation` for more info.
        """
        if noise_floor is not None and not isinstance(noise_floor, str):

            # Cast
            # noise_floor = np.array(noise_floor, dtype=float, ndmin=1)
            noise_floor = np.asarray(noise_floor)

            # Ensure all values are bigger than zero.
            if np.any(noise_floor <= 0.0):
                raise ValueError(
                    "All values of `noise_floor` must be bigger than zero."
                )

            # Store relative error.
            if noise_floor.size == 1:
                # If one value it is stored as attribute.
                noise_floor = float(noise_floor)
            else:
                # If more than one value it is stored as data array;
                # broadcasting it if necessary.
                self.data['_noise_floor'] = self.data.observed.copy(
                        data=np.ones(self.shape)*noise_floor)
                noise_floor = 'data._noise_floor'

        self._data.attrs['noise_floor'] = noise_floor

    @property
    def relative_error(self):
        r"""Return the relative error of the data.

        See :attr:`emg3d.surveys.Survey.standard_deviation` for more info.

        """
        return self.data.relative_error

    @relative_error.setter
    def relative_error(self, relative_error):
        """Update relative error.

        See :attr:`Survey.standard_deviation` for more info.
        """
        if relative_error is not None and not isinstance(relative_error, str):

            # Cast
            # relative_error = np.array(relative_error, dtype=float, ndmin=1)
            relative_error = np.asarray(relative_error)

            # Ensure all values are bigger than zero.
            if np.any(relative_error <= 0.0):
                raise ValueError(
                    "All values of `relative_error` must be bigger than zero."
                )

            # Store relative error.
            if relative_error.size == 1:
                # If one value it is stored as attribute.
                relative_error = float(relative_error)
            else:
                # If more than one value it is stored as data array;
                # broadcasting it if necessary.
                self.data['_relative_error'] = self.data.observed.copy(
                        data=np.ones(self.shape)*relative_error)
                relative_error = 'data._relative_error'

        self._data.attrs['relative_error'] = relative_error


def _dipole_info_to_dict(inp, name):
    """Create dict with provided source/receiver information."""

    # Create dict depending if `inp` is tuple or dict.
    if isinstance(inp, tuple):  # Tuple with coordinates

        # See if last tuple element is boolean, hence el/mag-flag.
        if isinstance(inp[-1], (list, tuple, np.ndarray)):
            provided_elmag = isinstance(inp[-1][0], (bool, np.bool_))
        else:
            provided_elmag = isinstance(inp[-1], (bool, np.bool_))

        # Get max dimension.
        nd = max([np.array(n, ndmin=1).size for n in inp])

        # Expand coordinates.
        coo = np.array([nd*[val, ] if np.array(val).size == 1 else
                        val for val in inp], dtype=np.float64)

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
        if name == 'source':
            TxRx = [electrodes.TxMagneticDipole, electrodes.TxElectricDipole]
        else:
            TxRx = [electrodes.RxMagneticPoint, electrodes.RxElectricPoint]
        out = {names[i]: TxRx[int(elmag[i])](coo[:, i]) for i in range(nd)}

    elif isinstance(inp, dict):
        if isinstance(inp[list(inp)[0]], dict):  # Dict of de-ser. Dipoles.
            out = {k: getattr(electrodes, v['__class__']).from_dict(v)
                   for k, v in inp.items()}
        else:  # Assumed dict of dipoles.
            out = inp

    else:
        raise TypeError(
            f"Input format of <{name}s> not recognized: {type(inp)}."
        )

    return out


def _frequency_info_to_dict(frequencies, naming="f"):
    """Create dicts with provided frequency information.

    For easier access three items are stored:
    - dict {name: freq}: default and used for xarray;
    - dict {freq: name}: reverse for flexibility to use the float;
    - array (frequencies): the frequencies as an array.

    """

    if isinstance(frequencies, dict):
        name_freq = frequencies
        freq_name = {float(v): k for k, v in frequencies.items()}
        freqs = np.array([float(v) for v in frequencies.values()])
    else:
        freqs = np.array(frequencies, dtype=np.float64, ndmin=1)

        if freqs.size != np.unique(freqs).size:
            raise ValueError(f"Contains non-unique frequencies: {freqs}.")

        if naming:
            dnd = len(str(freqs.size-1))  # Max number of digits.
            name_freq, freq_name = {}, {}
            for i, x in enumerate(freqs):
                name = f"{naming}{i:0{dnd}d}"
                name_freq[name] = float(x)
                freq_name[float(x)] = name
        else:
            name_freq = {str(x): float(x) for i, x in enumerate(freqs)}
            freq_name = {float(x): str(x) for i, x in enumerate(freqs)}

    return name_freq, freq_name, freqs
