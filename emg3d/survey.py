"""

:mod:`survey` -- Survey, sources, and receivers
===============================================

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


# import numpy as np

# from emg3d import maps, models

# __all__ = []


class Survey():
    r"""Create a survey with sources and receivers.


    Parameters
    ----------

    """

    def __init__(self, sources, receivers):
        """Initiate a new Survey instance."""
        pass

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

    # sources
    #

    # Add/remove source

    # receivers
    #

    # Add/remove receiver

    # frequencies
    # => collects all possible frequencies

    # shape: nsrc x nrec x nfreq
    # nrec, nsrc, nfreq
    # size: actual data, which can be less than nsrc*nrec*nfreq


class Dipole():
    # location
    pass


class DipoleSource(Dipole):
    # which receivers
    # what frequencies
    # nfreq
    # nrec
    pass


class DipoleReceiver(Dipole):
    # which sources
    # what frequencies
    # measured data
    # nsrc
    # nfreq
    # get e-field
    # get h-field
    pass
