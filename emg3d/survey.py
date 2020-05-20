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
# from typing import Tuple
# from dataclasses import dataclass

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

    # data

    # verify sources and receivers, that all exist, and that there are no
    # unused sources or receivers.


class Dipole():

    def __init__(self, names, coordinates=None, **kwargs):
        # Make it possible for either names and coordinates, or names as dict
        # with name as keys and coordinates as values

        # expand coordinates

        # Move to add
        self._names = dict()
        for name in names:
            self._names[name] = dict()

        # Move to add
        if len(coordinates) == 5:
            print("point dipole")
            self.is_finite = False
        elif len(coordinates) == 6:
            print("finite length dipole")
            self.is_finite = True
        else:
            print("* ERROR   :: Dipole coordinates are wrong defined.\n"
                  "             They must describe either a point,\n"
                  "             (x, y, z, azimuth, dip), or a finite dipole,\n"
                  "             (x1, x2, y1, y2, z1, z2)\n."
                  f"            Provided coordinates: {coordinates}.")
            raise ValueError("coordinates")

    def names(self):
        return self._names


class DipoleSource(Dipole):

    def __init__(self, names, coordinates=None, **kwargs):
        # location

        # Connect receivers
        # Only name is required

        super().__init__(names, coordinates, **kwargs)

    # what frequencies
    # nfreq
    # nrec

    # Method verify, to check that all connected receivers exist

    # __repr__ method to print info

    # to_dict/from_dict


class DipoleReceiver(Dipole):

    def __init__(self, names=None, coordinates=None, **kwargs):
        # location

        # `Dipole` permits len in [5, 6].
        # However, receivers are only implemented as points so far.

        if len(coordinates) == 6:
            print("* ERROR   :: Receivers are only implemented for point "
                  "dipoles\n"
                  "             at the moment, but finite length coordinates\n"
                  "             (x1, x2, y1, y2, z1, z2) were provided.")
            # TODO don't fail with an error, convert to point dipoles!
            raise ValueError("coordinates")

        # Connect sources and frequencies
        # Only name is required

        super().__init__(names, coordinates, **kwargs)

    # what frequencies
    # measured data
    # nsrc
    # nfreq
    # get e-field => needs survey
    # get h-field => needs survey

    # Method verify, to check that all connected sources exist with the
    # appropriate frequencies.

    # __repr__ method to print info

    # to_dict/from_dict


# @dataclass
# class NewReceiver:
#     name: str
#     coordinates: Tuple[float, float, float]
#     azimuth: float
#     dip: float

# @dataclass
# class NewSource:
#     receivers: List[NewReceiver]
