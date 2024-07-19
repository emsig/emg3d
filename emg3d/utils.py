"""
Utility functions for the multigrid solver.
"""
# Copyright 2018 The emsig community.
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
from time import perf_counter
from datetime import datetime, timedelta

import numpy as np
from empymod import EMArray
from scooby import Report as ScoobyReport

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

__all__ = ['Report', 'EMArray', 'Timer']

OPTIONAL = ['xarray', 'discretize', 'h5py', 'matplotlib', 'tqdm', 'IPython']


def __dir__():
    return __all__


# Set emg3d-warnings to always.
warnings.filterwarnings('always', 'emg3d: ', category=UserWarning)


# PRIVATE UTILS
_KNOWN_CLASSES = {}  # List of known classes for (de-)serialization


def _known_class(func):
    """Decorator to register class as known for I/O."""
    _KNOWN_CLASSES[func.__name__] = func
    return func


def _requires(*args, **kwargs):
    """Decorator to wrap functions with extra dependencies.

    This function is taken from `pysal` (in `lib/common.py`); see
    https://github.com/pysal/pysal (released under the 'BSD 3-Clause "New" or
    "Revised" License').


    Parameters
    ---------
    args : list
        Strings containing the modules to import.

    requires_verbose : bool
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

    def inner(function):
        available = [simport(arg)[0] for arg in args]
        if all(available):
            return function
        else:
            verbose = kwargs.pop('requires_verbose', True)
            wanted = copy.deepcopy(args)

            def passer(*args, **kwargs):
                if verbose:
                    missing = [arg for i, arg in enumerate(wanted)
                               if not available[i]]

                    # Warn.
                    msg = (
                        "emg3d: This feature requires the missing "
                        f"soft dependencies {missing}."
                    )
                    warnings.warn(msg, UserWarning)
            return passer

    return inner


# PUBLIC UTILS
class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use ``scooby`` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``, ``emg3d``,
    ``numba``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``,
    ``matplotlib``, and all soft dependencies of ``emg3d``. It also shows MKL
    information, if available.

    All modules provided in ``add_pckg`` are also shown.


    Parameters
    ----------
    add_pckg : {package, str}, default: None
        Package or list of packages to add to output information (must be
        imported beforehand or provided as string).

    ncol : int, default: 3
        Number of package-columns in html table (no effect in text-version).

    text_width : int, default: 80
        The text width for non-HTML display modes

    sort : bool, default: False
        Sort the packages when the report is shown

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['numpy', 'scipy', 'numba', 'emg3d', 'empymod']

        # Optional packages.
        optional = OPTIONAL

        super().__init__(additional=add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)


class Timer:
    """Class for timing (now; runtime)."""

    def __init__(self):
        """Initiate timer with a performance counter."""
        self._t0 = perf_counter()

    def __repr__(self):
        """Simple representation."""
        return f"Runtime : {self.runtime}"

    @property
    def t0(self):
        """Return time zero of this class instance."""
        return self._t0

    @property
    def now(self):
        """Return current time as hh:mm:ss string."""
        return datetime.now().strftime("%H:%M:%S")

    @property
    def runtime(self):
        """Return elapsed time as hh:mm:ss string."""
        return str(timedelta(seconds=np.round(self.elapsed)))

    @property
    def elapsed(self):
        """Return elapsed time in seconds."""
        return perf_counter() - self._t0
