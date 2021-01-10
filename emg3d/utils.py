"""
Utility functions for the multigrid solver.
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

import copy
import importlib
from timeit import default_timer
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import PchipInterpolator as Pchip
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

try:
    import scooby
    from scooby import Report as ScoobyReport
except ImportError:
    scooby = None

    class ScoobyReport:
        pass

try:
    import empymod
except ImportError:
    empymod = None

# Version: We take care of it here instead of in __init__, so we can use it
# within the package itself (logs).
try:
    # - Released versions just tags:       0.8.0
    # - GitHub commits add .dev#+hash:     0.8.1.dev4+g2785721
    # - Uncommitted changes add timestamp: 0.8.1.dev4+g2785721.d20191022
    from emg3d.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. emg3d should be installed
    # properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')

__all__ = ['Fourier', 'Time', 'Report', 'EMArray']


# SOFT DEPENDENCIES CHECK
def _requires(*args, **kwargs):
    """Decorator to wrap functions with extra dependencies.

    This function is taken from `pysal` (in `lib/common.py`); see
    https://github.com/pysal/pysal (released under the 'BSD 3-Clause "New" or
    "Revised" License').


    Parameters
    ---------
    args : list
        Strings containing the modules to import.

    verbose : bool
        If True (default) print a warning message on import failure.


    Returns
    -------
    out : func
        Original function if all modules are importable, otherwise returns a
        function that passes.
    """
    def simport(modname):
        """Safely import a module without raising an error."""
        try:
            return True, importlib.import_module(modname)
        except ImportError:
            return False, None

    v = kwargs.pop('verbose', True)
    wanted = copy.deepcopy(args)

    def inner(function):
        available = [simport(arg)[0] for arg in args]
        if all(available):
            return function
        else:
            def passer(*args, **kwargs):
                if v:
                    missing = [arg for i, arg in enumerate(wanted)
                               if not available[i]]
                    print("=> This feature of `emg3d` requires the following, "
                          f"missing soft dependencies: {missing}.")
                else:
                    pass
            return passer

    return inner


# EMArray
class EMArray(np.ndarray):
    r"""Create an EM-ndarray: add *amplitude* <amp> and *phase* <pha> methods.

    Parameters
    ----------
    data : array
        Data to which to add `.amp` and `.pha` attributes.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod.utils import EMArray
    >>> emvalues = EMArray(np.array([1+1j, 1-4j, -1+2j]))
    >>> print(f"Amplitude         : {emvalues.amp()}")
    Amplitude         : [1.41421356 4.12310563 2.23606798]
    >>> print(f"Phase (rad)       : {emvalues.pha()}")
    Phase (rad)       : [ 0.78539816 -1.32581766 -4.24874137]
    >>> print(f"Phase (deg)       : {emvalues.pha(deg=True)}")
    Phase (deg)       : [  45.          -75.96375653 -243.43494882]
    >>> print(f"Phase (deg; lead) : {emvalues.pha(deg=True, lag=False)}")
    Phase (deg; lead) : [-45.          75.96375653 243.43494882]

    """

    def __new__(cls, data):
        r"""Create a new EMArray."""
        return np.asarray(data).view(cls)

    def amp(self):
        """Amplitude of the electromagnetic field."""
        return np.abs(self.view())

    def pha(self, deg=False, unwrap=True, lag=True):
        """Phase of the electromagnetic field.

        Parameters
        ----------
        deg : bool
            If True the returned phase is in degrees, else in radians.
            Default is False (radians).

        unwrap : bool
            If True the returned phase is unwrapped.
            Default is True (unwrapped).

        lag : bool
            If True the returned phase is lag, else lead defined.
            Default is True (lag defined).

        """
        # Get phase, lead or lag defined.
        if lag:
            pha = np.angle(self.view())
        else:
            pha = np.angle(np.conj(self.view()))

        # Unwrap if `unwrap`.
        # np.unwrap removes the EMArray class;
        # for consistency, we wrap it in EMArray again.
        if unwrap and self.size > 1:
            pha = EMArray(np.unwrap(pha))

        # Convert to degrees if `deg`.
        if deg:
            pha *= 180/np.pi

        return pha


# TIME DOMAIN
@_requires('empymod')
class Fourier:
    r"""Time-domain CSEM computation.

    Class to carry out time-domain modelling with the frequency-domain code
    `emg3d`. Instances of the class take care of computing the required
    frequencies, the interpolation from coarse, limited-band frequencies to the
    required frequencies, and carrying out the actual transform.

    Everything related to the Fourier transform is done by utilising the
    capabilities of the 1D modeller :mod:`empymod`. The input parameters
    `time`, `signal`, `ft`, and `ftarg` are passed to the function
    :func:`empymod.utils.check_time` to obtain the required frequencies. The
    actual transform is subsequently carried out by calling
    :func:`empymod.model.tem`. See these functions for more details about the
    exact implementations of the Fourier transforms and its parameters.
    Note that also the `verb`-argument follows the definition in `empymod`.

    The mapping from computed frequencies to the frequencies required for the
    Fourier transform is done in three steps:

    - Data for :math:`f>f_\mathrm{max}` is set to 0+0j.
    - Data for :math:`f<f_\mathrm{min}` is interpolated by adding an additional
      data point at a frequency of 1e-100 Hz. The data for this point is
      ``data.real[0]+0j``, hence the real part of the lowest computed
      frequency and zero imaginary part. Interpolation is carried out using
      PCHIP :func:`scipy.interpolate.pchip_interpolate`.
    - Data for :math:`f_\mathrm{min}\le f \le f_\mathrm{max}` is computed
      with cubic spline interpolation (on a log-scale)
      :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

    Note that `fmin` and `fmax` should be chosen wide enough such that the
    mapping for :math:`f>f_\mathrm{max}` :math:`f<f_\mathrm{min}` does not
    matter that much.


    Parameters
    ----------

    time : ndarray
        Desired times (s).

    fmin, fmax : float
        Minimum and maximum frequencies (Hz) to compute:

          - Data for freq > fmax is set to 0+0j.
          - Data for freq < fmin is interpolated, using an extra data-point at
            f = 1e-100 Hz, with value data.real[0]+0j. (Hence zero imaginary
            part, and the lowest computed real value.)

    signal : {0, 1, -1}, optional
        Source signal, default is 0:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    ft : {'sin', 'cos', 'fftlog'}, optional
        Flag to choose either the Digital Linear Filter method (Sine- or
        Cosine-Filter) or the FFTLog for the Fourier transform.
        Defaults to 'sin'.

    ftarg : dict, optional
        Depends on the value for `ft`:

            - If `ft='dlf'`:

                - `dlf`: string of filter name in :mod:`empymod.filters` or the
                  filter method itself. (Default:
                  :func:`empymod.filters.key_201_CosSin_2012`)
                - `pts_per_dec`: points per decade; (default: -1)

                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF

            - If `ft='fftlog'`:

                - `pts_per_dec`: sampels per decade (default: 10)
                - `add_dec`: additional decades [left, right]
                  (default: [-2, 1])
                - `q`: exponent of power law bias (default: 0); -1 <= q <= 1


    freq_inp : array
        Frequencies to use for computation. Mutually exclusive with
        `every_x_freq`.

    every_x_freq : int
        Every `every_x_freq`-th frequency of the required frequency-range is
        used for computation. Mutually exclusive with `freq_calc`.


    """

    def __init__(self, time, fmin, fmax, signal=0, ft='dlf', ftarg=None,
                 **kwargs):
        """Initialize a Fourier instance."""

        # Store the input parameters.
        self._time = time
        self._fmin = fmin
        self._fmax = fmax
        self._signal = signal
        self._ft = ft
        if ftarg is None:
            self._ftarg = {}
        else:
            self._ftarg = ftarg

        # Get kwargs.
        self._freq_inp = kwargs.pop('freq_inp', None)
        self._every_x_freq = kwargs.pop('every_x_freq', None)
        self.verb = kwargs.pop('verb', 3)

        # Ensure no kwargs left.
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

        # Ensure freq_inp and every_x_freq are not both set.
        self._check_coarse_inputs(keep_freq_inp=True)

        # Get required frequencies.
        self._check_time()

    def __repr__(self):
        """Simple representation."""
        return (f"Fourier: {self._ft}; {self.time.min()}-{self.time.max()} s; "
                f"{self.fmin}-{self.fmax} Hz")

    # PURE PROPERTIES
    @property
    def freq_req(self):
        """Frequencies required to carry out the Fourier transform."""
        return self._freq_req

    @property
    def freq_coarse(self):
        """Coarse frequency range, can be different from `freq_req`."""
        if self.every_x_freq is None and self.freq_inp is None:
            # If none of {every_x_freq, freq_inp} given, then
            # freq_coarse = freq_req.
            return self.freq_req

        elif self.every_x_freq is None:
            # If freq_inp given, then freq_coarse = freq_inp.
            return self.freq_inp

        else:
            # If every_x_freq given, get subset of freq_req.
            return self.freq_req[::self.every_x_freq]

    @property
    def freq_calc_i(self):
        """Indices of `freq_coarse` which have to be computed."""
        ind = (self.freq_coarse >= self.fmin) & (self.freq_coarse <= self.fmax)
        return ind

    @property
    def freq_calc(self):
        """Frequencies at which the model has to be computed."""
        return self.freq_coarse[self.freq_calc_i]

    @property
    def freq_extrapolate_i(self):
        """Indices of the frequencies to extrapolate."""
        return self.freq_req < self.fmin

    @property
    def freq_extrapolate(self):
        """These are the frequencies to extrapolate.

        In fact, it is dow via interpolation, using an extra data-point at f =
        1e-100 Hz, with value data.real[0]+0j. (Hence zero imaginary part, and
        the lowest computed real value.)
        """
        return self.freq_req[self.freq_extrapolate_i]

    @property
    def freq_interpolate_i(self):
        """Indices of the frequencies to interpolate.

        If freq_req is equal freq_coarse, then this is eual to freq_calc_i.
        """
        return (self.freq_req >= self.fmin) & (self.freq_req <= self.fmax)

    @property
    def freq_interpolate(self):
        """These are the frequencies to interpolate.

        If freq_req is equal freq_coarse, then this is eual to freq_calc.
        """
        return self.freq_req[self.freq_interpolate_i]

    @property
    def ft(self):
        """Type of Fourier transform.
        Set via ``fourier_arguments(ft, ftarg)``.
        """
        return self._ft

    @property
    def ftarg(self):
        """Fourier transform arguments.
        Set via ``fourier_arguments(ft, ftarg)``.
        """
        return self._ftarg

    # PROPERTIES WITH SETTERS
    @property
    def time(self):
        """Desired times (s)."""
        return self._time

    @time.setter
    def time(self, time):
        """Update desired times (s)."""
        self._time = time
        self._check_time()

    @property
    def fmax(self):
        """Maximum frequency (Hz) to compute."""
        return self._fmax

    @fmax.setter
    def fmax(self, fmax):
        """Update maximum frequency (Hz) to compute."""
        self._fmax = fmax
        self._print_freq_calc()

    @property
    def fmin(self):
        """Minimum frequency (Hz) to compute."""
        return self._fmin

    @fmin.setter
    def fmin(self, fmin):
        """Update minimum frequency (Hz) to compute."""
        self._fmin = fmin
        self._print_freq_calc()

    @property
    def signal(self):
        """Signal in time domain {0, 1, -1}."""
        return self._signal

    @signal.setter
    def signal(self, signal):
        """Update signal in time domain {0, 1, -1}."""
        self._signal = signal

    @property
    def freq_inp(self):
        """If set, freq_coarse is set to freq_inp."""
        return self._freq_inp

    @freq_inp.setter
    def freq_inp(self, freq_inp):
        """Update freq_inp. Erases every_x_freq if set."""
        self._freq_inp = freq_inp
        self._check_coarse_inputs(keep_freq_inp=True)

    @property
    def every_x_freq(self):
        """If set, freq_coarse is every_x_freq-frequency of freq_req."""
        return self._every_x_freq

    @every_x_freq.setter
    def every_x_freq(self, every_x_freq):
        """Update every_x_freq. Erases freq_inp if set."""
        self._every_x_freq = every_x_freq
        self._check_coarse_inputs(keep_freq_inp=False)

    # OTHER STUFF
    def fourier_arguments(self, ft, ftarg):
        """Set Fourier type and its arguments."""
        self._ft = ft
        self._ftarg = ftarg
        self._check_time()

    def interpolate(self, fdata):
        """Interpolate from computed data to required data.

        Parameters
        ----------

        fdata : ndarray
            Frequency-domain data corresponding to `freq_calc`.

        Returns
        -------
        full_data : ndarray
            Frequency-domain data corresponding to `freq_req`.

        """

        # Pre-allocate result.
        out = np.zeros(self.freq_req.size, dtype=np.complex128)

        # 1. Interpolate between fmin and fmax.

        # If freq_coarse is not exactly freq_req, we use cubic spline to
        # interpolate from fmin to fmax.
        if self.freq_coarse.size != self.freq_req.size:

            int_real = Spline(np.log(self.freq_calc),
                              fdata.real)(np.log(self.freq_interpolate))
            int_imag = Spline(np.log(self.freq_calc),
                              fdata.imag)(np.log(self.freq_interpolate))

            out[self.freq_interpolate_i] = int_real + 1j*int_imag

        else:  # If they are the same, just fill in the data.
            out[self.freq_interpolate_i] = fdata

        # 2. Extrapolate from freq_req.min to fmin using PCHIP.

        # 2.a Extend freq_req/data by adding a point at 1e-100 Hz with
        # - same real part as lowest computed frequency and
        # - zero imaginary part.
        freq_ext = np.r_[1e-100, self.freq_calc]
        data_ext = np.r_[fdata[0].real-1e-100j, fdata]

        # 2.b Actual 'extrapolation' (now an interpolation).
        ext_real = Pchip(freq_ext, data_ext.real)(self.freq_extrapolate)
        ext_imag = Pchip(freq_ext, data_ext.imag)(self.freq_extrapolate)

        out[self.freq_extrapolate_i] = ext_real + 1j*ext_imag

        return out

    def freq2time(self, fdata, off):
        """Compute corresponding time-domain signal.

        Carry out the actual Fourier transform.

        Parameters
        ----------

        fdata : ndarray
            Frequency-domain data corresponding to `freq_calc`.

        off : float
            Corresponding offset (m).

        Returns
        -------
        tdata : ndarray
            Time-domain data corresponding to Fourier.time.

        """
        # Interpolate the computed data at the required frequencies.
        inp_data = self.interpolate(fdata)

        # Carry out the Fourier transform.
        tdata, _ = empymod.model.tem(
                inp_data[:, None], np.array(off), freq=self.freq_req,
                time=self.time, signal=self.signal, ft=self.ft,
                ftarg=self.ftarg)

        return np.squeeze(tdata)

    # PRIVATE ROUTINES
    def _check_time(self):
        """Get required frequencies for given times and ft/ftarg."""

        # Get freq via empymod.
        _, freq, ft, ftarg = empymod.utils.check_time(
            self.time, self.signal, self.ft, self.ftarg, self.verb)

        # Store required frequencies and check ft, ftarg.
        self._freq_req = freq
        self._ft = ft
        self._ftarg = ftarg

        # Print frequency information (if verbose).
        if self.verb > 2:
            self._print_freq_ftarg()
            self._print_freq_calc()

    def _check_coarse_inputs(self, keep_freq_inp=True):
        """Parameters `freq_inp` and `every_x_freq` are mutually exclusive."""

        # If they are both set, reset one depending on `keep_freq_inp`.
        if self._freq_inp is not None and self._every_x_freq is not None:
            print("\n* WARNING :: `freq_inp` and `every_x_freq` are mutually "
                  "exclusive.\n             Re-setting ", end="")

            if keep_freq_inp:  # Keep freq_inp.
                print("`every_x_freq=None`.\n")
                self._every_x_freq = None

            else:              # Keep every_x_freq.
                print("`freq_inp=None`.\n")
                self._freq_inp = None

    # PRINTING ROUTINES
    def _print_freq_ftarg(self):
        """Print required frequency range."""
        if self.verb > 2:
            empymod.utils._prnt_min_max_val(
                    self.freq_req, "   Req. freq  [Hz] : ", self.verb)

    def _print_freq_calc(self):
        """Print actually computed frequency range."""
        if self.verb > 2:
            empymod.utils._prnt_min_max_val(
                    self.freq_calc, "   Calc. freq [Hz] : ", self.verb)


# TIMING AND REPORTING
class Time:
    """Class for timing (now; runtime)."""

    def __init__(self):
        """Initialize time zero (t0) with current time stamp."""
        self._t0 = default_timer()

    def __repr__(self):
        """Simple representation."""
        return f"Runtime : {self.runtime}"

    @property
    def t0(self):
        """Return time zero of this class instance."""
        return self._t0

    @property
    def now(self):
        """Return string of current time."""
        return datetime.now().strftime("%H:%M:%S")

    @property
    def runtime(self):
        """Return string of runtime since time zero."""
        return str(timedelta(seconds=np.round(self.elapsed)))

    @property
    def elapsed(self):
        """Return runtime in seconds since time zero."""
        return default_timer() - self._t0


@_requires('scooby')
class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use `scooby` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), `numpy`, `scipy`, `emg3d`,
    `numba`, `sys.version`, and time/date.

    Additionally shown are, if they can be imported, `IPython` and
    `matplotlib`. It also shows MKL information, if available.

    All modules provided in `add_pckg` are also shown.

    .. note::

        The package `scooby` has to be installed in order to use `Report`:
        ``pip install scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from emg3d import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['numpy', 'scipy', 'numba', 'emg3d']

        # Optional packages.
        optional = ['empymod', 'xarray', 'discretize', 'h5py', 'matplotlib',
                    'tqdm', 'IPython']

        super().__init__(additional=add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)


# MISC
def _process_map(fn, *iterables, max_workers, **kwargs):
    """Imitate tqdm.contrib.concurrent.process_map without tqdm.

    emg3d.simulation uses `process_map` from `tqdm` to run jobs asynchronously.
    However, `tqdm` is a soft dependency. In case it is not installed we simply
    use `concurrent.futures.ProcessPoolExecutor`, from the standard library,
    and imitate the behaviour of process_map (basically a
    `ProcessPoolExecutor.map`, returned as a list, and wrapped in a context
    manager).

    """
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, *iterables))
