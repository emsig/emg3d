"""
A survey stores a set of sources and their frequencies, receivers, and the
measured data.
"""
# Copyright 2018-2021 The emsig community.
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

__all__ = ['Survey', 'random_noise', 'txrx_coordinates_to_dict',
           'txrx_lists_to_dict', 'frequencies_to_dict']


@utils._known_class
class Survey:
    """Create a survey containing sources, receivers, and data.

    A survey contains the acquisition information such as source types,
    positions, and frequencies and receiver types and positions. A survey
    contains also any acquired or synthetic data and their expected relative
    error and noise floor.

    The data is stored in an 3D ndarray of dimension ``nsrc x nrec x nfreq``.
    Underlying the survey-class is an :class:`xarray.Dataset`, where each
    individual data set (e.g., acquired data or synthetic data) is stored as a
    :class:`xarray.DataArray`. The module xarray is a soft dependency of emg3d,
    and has to be installed manually to use the survey functionality.

    Receivers have a switch ``relative``, which is False by default and means
    that the coordinates are absolute values (grid-based acquisition). If the
    switch is set to True, the coordinates are relative to the source. This can
    be used to model streamer-based acquisitions such as marine streamers or
    airborne surveys. The two acquisition types can also be mixed in a survey.

    .. note::

        The package ``xarray`` has to be installed in order to use ``Survey``:
        ``pip install xarray`` or ``conda install -c conda-forge xarray``.


    Parameters
    ----------
    sources, receivers : {Tx*, Rx*, list, dict)
        Any of the available sources or receivers, e.g.,
        :class:`emg3d.electrodes.TxElectricDipole`, or a list or dict of
        Tx*/Rx* instances. If it is a dict, it is used as is, including the
        provided keys. In all other cases keys are assigned to the values.

        It can also be a list containing a combination of the above (lists,
        dicts, and instances).

        Receivers can be set to ``None``, if one is only interested in forward
        modelling the entire fields. In this case, the related data object and
        the noise floor and relative error have no meaning. Also, in
        conjunction with a :class:`emg3d.simulations.Simulation`, the misfit
        and the gradient will be zero.

    frequencies : {array_like, dict}
        Source frequencies (Hz).

        - array_like: Frequencies will be stored in a dict with keys assigned
          starting with ``'f-1'``, ``'f-2'``, and so on.

        - dict: Keys can be arbitrary names, values must be floats.

    data : ndarray, default: None
        The observed data (dtype=np.complex128); must have shape (nsrc, nrec,
        nfreq). Alternatively, it can be a dict containing many datasets, in
        which one could also store, for instance, standard-deviations for each
        source-receiver-frequency pair.

        If None, it will be initiated with NaN's.

    noise_floor, relative_error : {float, ndarray}, default: None
        Noise floor and relative error of the data. They can be arrays of a
        shape which can be broadcasted to the data shape, e.g., (nsrc, 1, 1) or
        (1, nrec, nfreq), or have the dimension of data.
        See :attr:`Survey.standard_deviation` for more info.

    name : str, default: None
        Name of the survey.

    date : str, default: None
        Acquisition date.

    info : str, default: None
        Survey info or any other info (e.g., what was the intent of the survey,
        what were the acquisition conditions, problems encountered).

    """

    def __init__(self, sources, receivers, frequencies, data=None, **kwargs):
        """Initiate a new Survey instance."""

        # Store sources, receivers, and frequencies.
        self._sources = txrx_lists_to_dict(sources)
        if receivers is None:
            self._receivers = {}
        else:
            self._receivers = txrx_lists_to_dict(receivers)
        self._frequencies = frequencies_to_dict(frequencies)

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

    def __repr__(self):
        """Simple representation."""
        name = f" «{self.name}»" if self.name else ""
        date = f" {self.date}" if self.date else ""
        info = f"{self.info}\n" if self.info else ""
        return (f":: {self.__class__.__name__}{name} ::{date}\n{info}\n"
                f"{self.data.__repr__()}")

    def _repr_html_(self):
        """HTML representation with fancy xarray display."""
        name = f" «{self.name}»" if self.name else ""
        date = f" <tt>{self.date}</tt>" if self.date else ""
        info = f"{self.info}<br>" if self.info else ""
        return (f"<h3>{self.__class__.__name__}{name}{date}</h3>{info}"
                f"{self.data._repr_html_()}")

    def copy(self):
        """Return a copy of the Survey."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Survey in a dict.

        Parameters
        ----------
        copy : bool, default: False
            If True, returns a deep copy of the dict.


        Returns
        -------
        out : dict
            Dictionary containing all information to re-create the Survey.

        """
        out = {
            '__class__': self.__class__.__name__,
            'sources': {k: v.to_dict() for k, v in self.sources.items()},
            'receivers': {k: v.to_dict() for k, v in self.receivers.items()},
            'frequencies': self.frequencies,
            'data': {k: v.data for k, v in self.data.items()},
            'noise_floor': self.data.noise_floor,
            'relative_error': self.data.relative_error,
            'name': self.name,
            'date': self.date,
            'info': self.info,
        }
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    @utils._requires('xarray')
    def from_dict(cls, inp):
        """Convert dictionary into :class:`emg3d.surveys.Survey` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`emg3d.surveys.Survey.to_dict`.
            The dictionary needs the keys `sources`, `receivers`, and
            `frequencies`.

        Returns
        -------
        survey : Survey
            A :class:`emg3d.surveys.Survey` instance.

        """
        inp = {k: v for k, v in inp.items() if k != '__class__'}
        inp['sources'] = {k: getattr(electrodes, v['__class__']).from_dict(v)
                          for k, v in inp['sources'].items()}
        inp['receivers'] = {k: getattr(electrodes, v['__class__']).from_dict(v)
                            for k, v in inp['receivers'].items()}
        return cls(**inp)

    def to_file(self, fname, name='survey', **kwargs):
        """Store Survey to a file.

        Parameters
        ----------
        fname : str
            Absolute or relative file name including ending, which defines the
            used data format. See :func:`emg3d.io.save` for the options.

        name : str, default: 'survey'
            Name with which the survey is stored in the file.

        kwargs : Keyword arguments, optional
            Passed through to :func:`emg3d.io.save`.

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
            Absolute or relative file name including extension.

        name : str, default: 'survey'
            Name under which the survey is stored within the file.

        kwargs : Keyword arguments, optional
            Passed through to :func:`io.load`.


        Returns
        -------
        survey : Survey
            A :class:`emg3d.surveys.Survey` instance.

        info : str, returned if verb<0
            Info-string.

        """
        out = io.load(fname, **kwargs)
        if kwargs.get('verb', 0) < 0:
            return out[0][name], out[1]
        else:
            return out[name]

    # DATA
    @utils._requires('xarray')
    def _initiate_dataset(self, data):
        """Initiate Dataset."""

        # Get shape of DataArrays.
        shape = (len(self._sources),
                 len(self._receivers),
                 len(self._frequencies))

        # Initialize data and ensure there is 'observed'.
        if data is None:
            data = {'observed': np.full(shape, np.nan+1j*np.nan)}
        elif not isinstance(data, dict):
            data = {'observed': np.atleast_3d(data)}
        elif 'observed' not in data.keys():
            data['observed'] = np.full(shape, np.nan+1j*np.nan)

        # Create Dataset, add all data as DataArrays.
        dims = ('src', 'rec', 'freq')
        self._data = xarray.Dataset(
            {k: xarray.DataArray(v, dims=dims) for k, v in data.items()},
            coords={'src': list(self.sources.keys()),
                    'rec': list(self.receivers.keys()),
                    'freq': list(self.frequencies)},
        )

        # Add attributes.
        self._data.src.attrs['Sources'] = "".join(
                f"{k}: {s.__repr__()};\n" for k, s in self.sources.items()
                )[:-2]+'.'
        self._data.rec.attrs['Receivers'] = "".join(
                f"{k}: {d.__repr__()};\n" for k, d in self.receivers.items()
                )[:-2]+'.'
        self._data.freq.attrs['Frequencies'] = "".join(
                f"{k}: {f} Hz;\n" for k, f in self.frequencies.items()
                )[:-2]+'.'

    @property
    def data(self):
        """Data, a :class:`xarray.Dataset` instance."""
        return self._data

    def select(self, sources=None, receivers=None, frequencies=None):
        """Return a Survey with selected sources, receivers, and frequencies.


        Parameters
        ----------
        sources, receivers, frequencies : list, default: None
            Lists containing the wanted sources, receivers, and frequencies.
            If None, all are selected.


        Returns
        -------
        survey : Survey
            A :class:`emg3d.surveys.Survey` instance.

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
                    f: survey['frequencies'][f] for f in frequencies}
            selection['freq'] = frequencies

        # Replace data with selected data.
        for key in survey['data'].keys():
            survey['data'][key] = self.data[key].sel(**selection)

        # Return new, reduced survey.
        return Survey.from_dict(survey)

    @property
    def shape(self):
        """Shape of data (nsrc, nrec, nfreq)."""
        return self.data.observed.shape

    @property
    def size(self):
        """Size of data (nsrc x nrec x nfreq)."""
        return int(self.data.observed.size)

    @property
    def count(self):
        """Count of observed data."""
        return int(self.data.observed.count())

    # SOURCES, RECEIVERS, FREQUENCIES
    @property
    def sources(self):
        """Source dict containing all sources."""
        return self._sources

    @property
    def receivers(self):
        """Receiver dict containing all receivers."""
        return self._receivers

    def source_coordinates(self):
        """Return source center coordinates as ndarray [x, y, z]."""
        return np.array([s.center for s in self.sources.values()]).T

    def receiver_coordinates(self, source=None):
        """Return receiver center coordinates as ndarray [x, y, z].

        For relative receivers, all positions are listed one after the other
        for each source position. Alternatively, a source-name (string) can be
        provided, in which case only the position for this source is added.
        """
        coords = []

        # Loop over receivers.
        for v in self.receivers.values():

            # If relative, loop over sources and add all positions.
            if v.relative and source is None:
                for s in self.sources.values():
                    coords.append(v.center_abs(s))

            # If relative with a provided source, add this position.
            elif v.relative:
                coords.append(v.center_abs(self.sources[source]))

            # If absolute, add.
            else:
                coords.append(v.center)

        return np.array(coords).T

    @property
    def frequencies(self):
        """Frequency dict containing all frequencies."""
        return self._frequencies

    # STANDARD DEVIATION and NOISE
    @property
    def standard_deviation(self):
        r"""Return standard deviation of the data.

        The standard deviation can be set by providing an array of the same
        dimension as the data itself::

            survey.standard_deviation = ndarray  # (nsrc, nrec, nfreq)

        Alternatively, one can set the ``noise_floor``
        :math:`\epsilon_\text{n}` and the ``relative_error``
        :math:`\epsilon_\text{r}`::

            survey.noise_floor = {float, ndarray}      # (> 0 or None)
            survey.relative error = {float, ndarray}   # (> 0 or None)

        They must be either floats, or three-dimensional arrays of shape
        ``({1;nsrc}, {1;nrec}, {1;nfreq})``; dimensions of one will be
        broadcasted to the corresponding size. E.g., for a dataset of arbitrary
        amount of sources and receivers with three frequencies you can define a
        purely frequency-dependent relative error via::

            relative_error = np.array([err_f1, err_f2, err_f3])[None, None, :]

        The standard deviation :math:`\varsigma_i` of observation :math:`d_i`
        is then given in terms of the noise floor :math:`\epsilon_{\text{n};i}`
        and the relative error :math:`\epsilon_{\text{r};i}` by

        .. math::
            :label: std

            \varsigma_i = \sqrt{\epsilon_{\text{n}; i}^2 +
                          \left(\epsilon_{\text{r}; i}|d_i|\right)^2 } \, .

        Note that a set standard deviation is prioritized over potentially also
        defined noise floor and relative error. To use the noise floor and the
        relative error after defining standard deviation directly you would
        have to reset it like

        .. code-block:: python

            survey.standard_deviation = None

        after which Equation :eq:`std` would be used again.

        """
        # If `standard_deviation` was set, return it.
        if 'standard_deviation' in self._data.keys():
            return self.data['standard_deviation']

        # Compute it if not already set.
        elif self.noise_floor is not None or self.relative_error is not None:

            # Initiate std (xarray of same type as the observed data)
            std = self.data.observed.copy(data=np.zeros(self.shape))

            # Add noise floor if given.
            if self.noise_floor is not None:
                std += self.noise_floor**2

            # Add relative error if given.
            if self.relative_error is not None:
                std += np.abs(self.relative_error*self.data.observed)**2

            return np.sqrt(std)

        # If nothing is defined, return None
        else:
            return None

    @standard_deviation.setter
    def standard_deviation(self, standard_deviation):
        """Update standard deviation."""

        # Update standard_deviation.
        if standard_deviation is not None:

            # Ensure all values are bigger than zero.
            if np.any(standard_deviation <= 0.0):
                raise ValueError(
                    "All values of `standard_deviation` must be bigger "
                    f"than zero. Provided: {standard_deviation}."
                )

            self._data['standard_deviation'] = self.data.observed.copy(
                    data=standard_deviation)

        # If None: assure no standard_deviation in data.
        elif 'standard_deviation' in self.data:
            del self._data['standard_deviation']

    @property
    def noise_floor(self):
        r"""Return noise floor of the data.

        See :attr:`emg3d.surveys.Survey.standard_deviation` for more info.

        """
        if isinstance(self.data.noise_floor, str):
            return self.data._noise_floor.data
        else:
            return self.data.noise_floor

    @noise_floor.setter
    def noise_floor(self, noise_floor):
        """Update noise floor."""
        self._set_nf_re('noise_floor', noise_floor)

    @property
    def relative_error(self):
        r"""Return relative error of the data.

        See :attr:`emg3d.surveys.Survey.standard_deviation` for more info.

        """
        if isinstance(self.data.relative_error, str):
            return self.data._relative_error.data
        else:
            return self.data.relative_error

    @relative_error.setter
    def relative_error(self, relative_error):
        """Update relative error."""
        self._set_nf_re('relative_error', relative_error)

    def add_noise(self, min_offset=0.0, min_amplitude='half_nf',
                  add_to='observed', **kwargs):
        """Add random noise to the data defined by ``add_to``.

        The noise is generated with :func:`emg3d.surveys.random_noise`, consult
        that function to see how it is actually generated (``kwargs`` are
        passed through).

        This function takes further care of removing data which is too close to
        the source (``min_offset``) or has a too low signal
        (``min_amplitude``).


        Parameters
        ----------

        min_offset : float, default: 0.0
            Data points in ``data.observed`` where the offset < min_offset are
            set to NaN.

        min_amplitude : {float, str, None}, default: 'half_nf'
            Data points in ``data.observed`` where abs(data) < min_amplitude
            are set to NaN. If ``'half_nf'``, the ``noise_floor`` divided by
            two is used as ``min_amplitude``. Set to ``None`` to include all
            data.

        add_to : str, default: 'observed'
            Data to which to add the noise. By default it is added to the
            observed data. If a name is provided that is not an existing
            dataset it will create a dataset of zeroes. You can use that to
            obtain the pure noise.

        """
        # If a new data set is defined as output, initiate it.
        if add_to not in self.data.keys():
            self.data[add_to] = self.data.observed.copy(
                    data=np.zeros(self.shape, dtype=complex))

        # Set data below minimum amplitude to NaN.
        if min_amplitude == 'half_nf':
            min_amplitude = self.noise_floor
            if min_amplitude is not None:
                min_amplitude /= 2.0
        if min_amplitude is not None:
            cut_amp = abs(self.data.observed.data) < min_amplitude
            self.data[add_to].data[cut_amp] = np.nan + 1j*np.nan

        # Set offsets below minimum offset to NaN.
        if min_offset > 0.0:
            for ks, s in self.sources.items():
                for kr, r in self.receivers.items():
                    if np.linalg.norm(r.center_abs(s) - s.center) < min_offset:
                        self.data[add_to].loc[ks, kr, :] = np.nan + 1j*np.nan

        # Add noise if noise_floor and/or relative_error given.
        if self.standard_deviation is not None:
            noise = random_noise(self.standard_deviation.data, **kwargs)
            self.data[add_to].data += noise

    def _set_nf_re(self, name, value):
        """Update noise_floor or relative_error."""

        if value is not None and not isinstance(value, str):

            # Cast
            value = np.asarray(value)

            # Ensure all values are bigger than zero.
            if np.any(value <= 0.0):
                raise ValueError(
                    f"All values of `{name}` must be bigger than zero. "
                    f"Provided: {value}."
                )

            # If one value it is stored as attribute.
            if value.size == 1:
                value = float(value)

            # If more than one value it is stored as data array;
            # broadcasting it if necessary.
            else:
                self.data['_'+name] = self.data.observed.copy(
                        data=np.ones(self.shape)*value)
                value = 'data._'+name  # str-flag on attrs.

        self._data.attrs[name] = value


def random_noise(standard_deviation, mean_noise=0.0, ntype='white_noise'):
    r"""Return random noise for given inputs.

    Different methods are implemented to create random noise for
    frequency-domain CSEM data. All methods generate random noise in the
    following way

    .. math::

        d^\text{noise} =
        \varsigma \left[(1 + \text{i})\,u + \mathcal{R} \right] \, .

    where :math:`\varsigma` is the standard deviation (see
    :attr:`emg3d.surveys.Survey.standard_deviation`), :math:`u` is the mean
    value of the randomly distributed noise, and :math:`\mathcal{R}` are the
    random realizations of the noise.

    Currently there are three methods (``ntype``) implemented.

    1. ``white_noise``

       Random uniform phases with constant amplitudes. This is the default
       implementation, and corresponds to white noise in the time-domain: a
       flat amplitude spectrum for all frequencies, with random phases:

       .. math::

           \mathcal{R}_\text{wn} = \exp[\text{i}\,\mathcal{U}(0, 2\pi)] \, ,

       where :math:`\mathcal{U}(0, 2\pi)` is the uniform distribution and its
       range.

    2. Random Gaussian noise.

       In the following, :math:`\mathcal{N}(0, 1)` is the standard normal
       distribution of zero mean and unit standard deviation.

       a. ``gaussian_correlated``

          Same realization added to real and imaginary part.

          .. math::

              \mathcal{R}_\text{gc} = (1+\text{i})\,\mathcal{N}(0, 1) \, .


       b. ``gaussian_uncorrelated``

          Independent realizations added to real and imaginary part.

          .. math::

              \mathcal{R}_\text{gu} =
              \mathcal{N}(0, 1) + \text{i}\,\mathcal{N}(0, 1) \, .


    There are, of course, other possibilities. One could, e.g., make the
    non-zero mean itself random.

    See the example `random_noise_f_domain.html
    <https://empymod.emsig.xyz/en/latest/gallery/educational/random_noise_f_domain.html>`_
    for more details about random noise in the frequency domain.


    Parameters
    ----------
    standard_deviation : ndarray
        Standard deviations of the data.

    mean_noise : float, default: 0.0
        Mean value of the random noise (as fraction of standard_deviation).

    ntype : str, default: white_noise
        What type of noise. Options:

        - ``'white_noise'``:
          random uniform phases with constant amplitude.

        - ``'gaussian_correlated'``:
          Same Gaussian random realizations added to Real and Imaginary part.

        - ``'gaussian_uncorrelated'``:
          Independent Gaussian random realizations added to Real and Imaginary
          part.


    Returns
    -------
    noise : ndarray
        Noise, a complex-valued ndarray of the same shape as
        standard_deviation.

    """
    shape = standard_deviation.shape

    # Initiate Random Generator.
    rng = np.random.default_rng()

    # Random Gaussian noise independently for Real and Imaginary part.
    if ntype == 'gaussian_uncorrelated':
        noise = rng.standard_normal(shape) + 1j*rng.standard_normal(shape)

    # Random Gaussian noise; same for Real and Imaginary part.
    elif ntype == 'gaussian_correlated':
        noise = rng.standard_normal(shape)*(1+1j)

    # Random Uniform phases with constant amplitude (white noise); default.
    else:
        noise = np.exp(1j * rng.uniform(0, 2*np.pi, shape))

    # Return noise.
    return standard_deviation * ((1+1j)*mean_noise + noise)


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
          ...: sources = emg3d.surveys.txrx_coordinates_to_dict(
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
    txrx : {Tx*, Rx*, list, dict)
        Any of the available sources or receivers, e.g.,
        :class:`emg3d.electrodes.TxElectricDipole`, or a list or dict of
        Tx*/Rx* instances. If it is a dict, it is returned unaltered.

        It can also be a list containing a combination of the above (lists,
        dicts, and instances).


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
          ...: magnetic = emg3d.surveys.txrx_coordinates_to_dict(
          ...:                 emg3d.RxMagneticPoint,
          ...:                 ([950, 1050, 1150], 0, 0, 0, 90))

       In [4]: # Create a streamer receiver, flying 5 m behind the source.
          ...: streamer = emg3d.RxElectricPoint((5, 0, 0, 0, 0), relative=True)

       In [5]: # Collect all receivers.
          ...: receivers = emg3d.surveys.txrx_lists_to_dict(
          ...:                 [[streamer, ], electric, magnetic])
          ...: receivers  # QC our collected receivers

    """

    # If input is a dict, return it unaltered.
    if isinstance(txrx, dict):
        return txrx

    # A single Tx*/Rx* instance.
    elif hasattr(txrx, '_prefix'):
        txrx = [txrx, ]

    # If it is a list and contains other lists or dicts, collect them.
    elif any(isinstance(el, (list, tuple, dict)) for el in txrx):

        # Add all lists and dicts to new list.
        new_txrx = list()
        for trx in txrx:

            # A single Tx*/Rx* instance.
            if hasattr(trx, '_prefix'):
                trx = [trx, ]

            # If dict, cast it to list.
            elif isinstance(trx, dict):
                trx = list(trx.values())

            new_txrx += trx

        # Overwrite original list with new flat list.
        txrx = new_txrx

    # else, it has to be a list/tuple of Tx/Rx instances.

    # Return TxRx-dict.
    nx = len(txrx)
    return {f"{trx._prefix}-{i+1:0{len(str(nx))}d}": trx
            for i, trx in enumerate(txrx)}


def frequencies_to_dict(frequencies):
    """Create dict from provided frequencies.

    Parameters
    ----------
    frequencies : {array_like, dict}
        Source frequencies (Hz).

        - array_like: Frequencies will be stored in a dict with keys assigned
          starting with ``'f-1'``, ``'f-2'``, and so on.

        - dict: returned unaltered.


    Returns
    -------
    out : dict
        Frequency dict.

    """

    if not isinstance(frequencies, dict):

        # Cast.
        freqs = np.array(frequencies, dtype=np.float64, ndmin=1)

        # Ensure frequencies are unique.
        if freqs.size != np.unique(freqs).size:
            raise ValueError(f"Contains non-unique frequencies: {freqs}.")

        # Store in dict.
        frequencies = {f"f-{i+1:0{len(str(freqs.size))}d}": freq
                       for i, freq in enumerate(freqs)}

    return frequencies
