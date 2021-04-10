"""
Utility functions for the multigrid solver.
"""
# Copyright 2018-2021 The EMSiG community.
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
import warnings
import importlib
from timeit import default_timer
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    from scooby import Report as ScoobyReport
except ImportError:
    class ScoobyReport:
        pass

try:
    import tqdm
    import tqdm.contrib.concurrent
except ImportError:
    tqdm = None

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

__all__ = ['Time', 'Report', 'EMArray']


# List of known classes for (de-)serialization
KNOWN_CLASSES = {}


def known_class(func):
    """Decorator to register class as known for I/O."""
    KNOWN_CLASSES[func.__name__] = func
    return func


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
                    # Print is always shown and simpler, warn for the CLI logs.
                    msg = ("This feature of `emg3d` requires the following,"
                           f" missing soft dependencies: {missing}.")
                    print(f"* WARNING :: {msg}")
                    warnings.warn(msg, UserWarning)
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
    """Dispatch processes in parallel or not, using tqdm or not.

    :class:``emg3d.simulations.Simulation`` uses ``process_map`` from ``tqdm``
    to run jobs asynchronously. However, ``tqdm`` is a soft dependency. In case
    it is not installed we use ``concurrent.futures.ProcessPoolExecutor``
    directly, from the standard library, and imitate the behaviour of
    process_map (basically a ``ProcessPoolExecutor.map``, returned as a list,
    and wrapped in a context manager). If max_workers is smaller than two then
    we we avoid parallel execution.

    """
    # Parallel
    if max_workers > 1 and tqdm is None:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, *iterables))

    # Parallel with tqdm
    elif max_workers > 1:
        kwargs['max_workers'] = max_workers
        return tqdm.contrib.concurrent.process_map(fn, *iterables, **kwargs)

    # Sequential
    elif tqdm is None:
        return list(map(fn, *iterables))

    # Sequential with tqdm
    else:
        return list(tqdm.auto.tqdm(
            iterable=map(fn, *iterables), total=len(iterables[0]), **kwargs))
