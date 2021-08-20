"""
Functionalities related to time-domain modelling using a frequency-domain code.
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

import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator as Pchip
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

try:
    import empymod
except ImportError:
    empymod = None

from emg3d import utils


__all__ = ['Fourier', ]


@utils._requires('empymod')
class Fourier:
    r"""Time-domain CSEM computation.

    Class to carry out time-domain modelling with the frequency-domain code
    ``emg3d`` following [WeMS21]_. Instances of the class take care of
    computing the required frequencies, the interpolation from coarse,
    limited-band frequencies to the required frequencies, and carrying out the
    actual transform.

    Everything related to the Fourier transform is done by utilising the
    capabilities of the 1D modeller :mod:`empymod`. The input parameters
    ``time``, ``signal``, ``ft``, and ``ftarg`` are passed to the function
    :func:`empymod.utils.check_time` to obtain the required frequencies. The
    actual transform is subsequently carried out by calling
    :func:`empymod.model.tem`. See these functions for more details about the
    exact implementations of the Fourier transforms and its parameters. Note
    that also the ``verb``-argument follows the definition in ``empymod``.

    The mapping from computed frequencies to the frequencies required for the
    Fourier transform is done in three steps:

    - Data for :math:`f>f_\mathrm{max}` is set to 0+0j.
    - Data for :math:`f<f_\mathrm{min}` is interpolated by adding an additional
      data point at a frequency of 1e-100 Hz. The data for this point is
      ``data.real[0]+0j``, hence the real part of the lowest computed
      frequency and zero imaginary part. Interpolation is carried out using
      PCHIP from :func:`scipy.interpolate.pchip_interpolate`.
    - Data for :math:`f_\mathrm{min}\le f \le f_\mathrm{max}` is computed
      with cubic spline interpolation (on a log-scale) using
      :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

    .. note::

        The package ``empymod`` has to be installed in order to use
        ``Fourier``:
        ``pip install empymod`` or ``conda install -c conda-forge empymod``.


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

    signal : {-1, 0, 1}, default: 0
        Source signal:

        - -1 : Switch-off time-domain response
        - 0 : Impulse time-domain response
        - +1 : Switch-on time-domain response

    ft : {'sin', 'cos', 'fftlog'}, default: 'sin'
        Flag to choose either the Digital Linear Filter method (Sine- or
        Cosine-Filter) or the FFTLog for the Fourier transform.

    ftarg : dict, default depends on ``ft``
        Fourier transform arguments.

        - If ``ft='dlf'``:

           - ``dlf``: string of filter name in :mod:`empymod.filters` or the
             filter method itself; default: ``'key_201_CosSin_2012'``.
           - ``pts_per_dec``: points per decade; default: -1.

              - If 0: Standard DLF;
              - If < 0: Lagged Convolution DLF;
              - If > 0: Splined DLF.

        - If ``ft='fftlog'``:

           - ``pts_per_dec``: samples per decade; default: 10.
           - ``add_dec``: additional decades [left, right]; default: [-2, 1].
           - ``q``: exponent of power law bias, -1 <= q <= 1 ; default: 0.

    input_freq : ndarray, default: None
        Frequencies to use for computation. Mutually exclusive with
        ``every_x_freq``.

    every_x_freq : int, default: None
        Every ``every_x_freq``-th frequency of the required frequency-range is
        used for computation. Mutually exclusive with ``input_freq``.


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
        self._ftarg = {} if ftarg is None else ftarg
        self._input_freq = kwargs.pop('input_freq', None)
        self._every_x_freq = kwargs.pop('every_x_freq', None)
        self.verb = kwargs.pop('verb', 3)

        # Ensure no kwargs left.
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

        # Ensure input_freq and every_x_freq are not both set.
        self._check_coarse_inputs(keep_inp_freq=True)

        # Get required frequencies.
        self._check_time()

    def __repr__(self):
        """Simple representation."""
        return (f"{self.__class__.__name__}: {self._ft}; "
                f"{self.time.min()}-{self.time.max()} s; "
                f"{self.fmin}-{self.fmax} Hz")

    # PURE PROPERTIES
    @property
    def freq_required(self):
        """Frequencies required to carry out the Fourier transform."""
        return self._freq_req

    @property
    def freq_coarse(self):
        """Coarse frequency range, can be different from `freq_required`."""
        # If none of {every_x_freq, input_freq} given, then
        # freq_coarse = freq_required.
        if self.every_x_freq is None and self.input_freq is None:
            return self.freq_required

        # If input_freq given, then freq_coarse = input_freq.
        elif self.every_x_freq is None:
            return self.input_freq

        # If every_x_freq given, get subset of freq_required.
        else:
            return self.freq_required[::self.every_x_freq]

    @property
    def ifreq_compute(self):
        """Indices of `freq_coarse` which have to be computed."""
        return ((self.freq_coarse >= self.fmin) &
                (self.freq_coarse <= self.fmax))

    @property
    def freq_compute(self):
        """Frequencies at which the model has to be computed."""
        return self.freq_coarse[self.ifreq_compute]

    @property
    def ifreq_extrapolate(self):
        """Indices of the frequencies to extrapolate."""
        return self.freq_required < self.fmin

    @property
    def freq_extrapolate(self):
        """These are the frequencies to extrapolate.

        In the end it is done via interpolation, using an extra data-point at
        f = 1e-100 Hz, with value data.real[0]+0j. (Hence zero imaginary part,
        and the lowest computed real value.)
        """
        return self.freq_required[self.ifreq_extrapolate]

    @property
    def ifreq_interpolate(self):
        """Indices of the frequencies to interpolate."""
        return ((self.freq_required >= self.fmin) &
                (self.freq_required <= self.fmax))

    @property
    def freq_interpolate(self):
        """These are the frequencies to interpolate.

        If ``freq_required`` is equal ``freq_coarse``, then this is equal to
        ``freq_compute``.
        """
        return self.freq_required[self.ifreq_interpolate]

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
        """Signal in time domain {-1, 0, 1}."""
        return self._signal

    @signal.setter
    def signal(self, signal):
        """Update signal in time domain {-1, 0, 1}."""
        self._signal = signal

    @property
    def input_freq(self):
        """If set, freq_coarse is set to input_freq."""
        return self._input_freq

    @input_freq.setter
    def input_freq(self, input_freq):
        """Update input_freq. Erases every_x_freq if set."""
        self._input_freq = input_freq
        self._check_coarse_inputs(keep_inp_freq=True)

    @property
    def every_x_freq(self):
        """If set, freq_coarse is every_x_freq-frequency of freq_required."""
        return self._every_x_freq

    @every_x_freq.setter
    def every_x_freq(self, every_x_freq):
        """Update every_x_freq. Erases input_freq if set."""
        self._every_x_freq = every_x_freq
        self._check_coarse_inputs(keep_inp_freq=False)

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
            Frequency-domain data corresponding to ``freq_compute``.

        Returns
        -------
        full_data : ndarray
            Frequency-domain data corresponding to ``freq_required``.

        """

        # Pre-allocate result.
        out = np.zeros(self.freq_required.size, dtype=np.complex128)

        # 1. Interpolate between fmin and fmax.

        # If freq_coarse is not exactly freq_required, we use cubic spline to
        # interpolate from fmin to fmax.
        if self.freq_coarse.size != self.freq_required.size:

            int_real = Spline(np.log(self.freq_compute),
                              fdata.real)(np.log(self.freq_interpolate))
            int_imag = Spline(np.log(self.freq_compute),
                              fdata.imag)(np.log(self.freq_interpolate))

            out[self.ifreq_interpolate] = int_real + 1j*int_imag

        # If they are the same, just fill in the data.
        else:
            out[self.ifreq_interpolate] = fdata

        # 2. Extrapolate from freq_required.min to fmin using PCHIP.

        # 2.a Extend freq_required/data by adding a point at 1e-100 Hz with
        # - same real part as lowest computed frequency and
        # - zero imaginary part.
        freq_ext = np.r_[1e-100, self.freq_compute]
        data_ext = np.r_[fdata[0].real-1e-100j, fdata]

        # 2.b Actual 'extrapolation' (now an interpolation).
        ext_real = Pchip(freq_ext, data_ext.real)(self.freq_extrapolate)
        ext_imag = Pchip(freq_ext, data_ext.imag)(self.freq_extrapolate)

        out[self.ifreq_extrapolate] = ext_real + 1j*ext_imag

        return out

    def freq2time(self, fdata, off):
        """Compute corresponding time-domain signal.

        Carry out the actual Fourier transform.

        Parameters
        ----------

        fdata : ndarray
            Frequency-domain data corresponding to ``Fourier.freq_compute``.

        off : float
            Corresponding offset (m).

        Returns
        -------
        tdata : ndarray
            Time-domain data corresponding to ``Fourier.time``.

        """
        # Interpolate the computed data at the required frequencies.
        inp_data = self.interpolate(fdata)

        # Carry out the Fourier transform.
        tdata, _ = empymod.model.tem(
                inp_data[:, None], np.array(off), freq=self.freq_required,
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

    def _check_coarse_inputs(self, keep_inp_freq=True):
        """Parameters `input_freq` & `every_x_freq` are mutually exclusive."""

        # If they are both set, reset one depending on `keep_inp_freq`.
        if self._input_freq is not None and self._every_x_freq is not None:
            msg = ("emg3d: `input_freq` and `every_x_freq` are mutually "
                   "exclusive. Re-setting ")

            if keep_inp_freq:  # Keep input_freq.
                msg += "`every_x_freq=None`."
                self._every_x_freq = None

            else:              # Keep every_x_freq.
                msg += "`input_freq=None`."
                self._input_freq = None

            # Warn.
            warnings.warn(msg, UserWarning)

    # PRINTING ROUTINES
    def _print_freq_ftarg(self):
        """Print required frequency range."""
        if self.verb > 2:
            empymod.utils._prnt_min_max_val(
                    self.freq_required, "   Req. freq  [Hz] : ", self.verb)

    def _print_freq_calc(self):
        """Print actually computed frequency range."""
        if self.verb > 2:
            empymod.utils._prnt_min_max_val(
                    self.freq_compute, "   Calc. freq [Hz] : ", self.verb)
