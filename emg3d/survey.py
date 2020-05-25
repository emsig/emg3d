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


import numpy as np
# from typing import Tuple
# from dataclasses import dataclass

# from emg3d import maps, models

# __all__ = []


class Survey:
    """Create a survey with sources and receivers.

    Survey takes care to join same receivers
    Survey should check for
    - source-receiver combinations,
    - report missing sources
    - list nfreq per source and all the rest


    Parameters
    ----------

    """

    def __init__(self):
        """Initiate a new Survey instance."""
        self.names = {}


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

    def add_receivers(self, name, coordinates, frequency, source, data=None):

        # Only name should be required
        # coordinates are optionally, name is enough if once defined
        # frequency and source are optionally
        # data only if frequency and source provided

        # # Move to add
        # self._names = dict()
        # for name in names:
        #     self._names[name] = dict()

        self.names[name] = DipoleReceiver(
            name=name,
            coordinates=coordinates,
            frequency=frequency,
            source=source,
            data=data,
        )

    def remove_receivers(self):
        pass


    def add_sources(self):
        pass

    def remove_sources(self):
        pass

    # Todo
    def validate(self):
        # verify sources and receivers, that all exist, and that there are no
        # unused sources or receivers.
        pass


    def shape(self):
        # Return nsrc x nrec x nfreq
        pass

    def size(self):
        # Return actual data size (NOT equals nsrc x nrec x nfreq)
        pass

    def data(self):
        # Store the actual data in an array
        pass


class Dipole:
    """
    Base dipole (source or receiver).
    """

    def __init__(self, name, coordinates, **kwargs):

        # expand coordinates

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

        self.name = name
        self.coordinates = coordinates

        if kwargs:
            print(f"¡ Error ! Remaining kwargs: {kwargs}")

    @property
    def length(self):
        if not hasattr(self, '_length'):
            if self.is_finite:
                self._length = np.linalg.norm(
                        np.diff(np.array(self.coordinates).reshape(-1, 2)))
            else:
                self._length = 1.0

        return self._length

    @property
    def center(self):
        """Center of the dipole."""
        pass

    @property
    def dip(self):
        """Implement that there is always a dip, not only for point dipoles."""
        if not hasattr(self, '_dip'):
            if self.is_finite:
                print('TODO calculate dip')
                raise NotImplementedError
            else:
                self._dip = self.coordinates[4]

        return self._dip

    @property
    def azimuth(self):
        """Implement that there is always a azimuth, not only for point
        dipoles."""
        if not hasattr(self, '_azimuth'):
            if self.is_finite:
                print('TODO calculate azimuth')
                raise NotImplementedError
            else:
                self._azimuth = self.coordinates[3]

        return self._azimuth


class DipoleSource(Dipole):
    """
    """

    def __init__(self, name, coordinates, **kwargs):

        super().__init__(name, coordinates, **kwargs)

    # __repr__ method to print info

    # to_dict/from_dict


class DipoleReceiver(Dipole):
    """
    Maybe this shouldn't be a dataclass, and only data should be a dataclass.

    - Frequencies could be a list, matching the data size.
    - Source names could be a list, matching the data size.
    - If frequencies and source names are lists, then data should have dimension
      (nsrc, nfreq).

    - names/coords should be able to be a list too (nrec, nsrc, nfreq)

    Todos:
    - Add frequencies
    - Add sources

    """
    def __init__(self, name, coordinates, **kwargs):

        # `Dipole` permits len in [5, 6].
        # However, receivers are only implemented as points so far.

        if len(coordinates) == 6:
            print("* ERROR   :: Receivers are only implemented for point "
                  "dipoles\n"
                  "             at the moment, but finite length coordinates\n"
                  "             (x1, x2, y1, y2, z1, z2) were provided.")
            # TODO don't fail with an error, convert to point dipoles!
            raise ValueError("coordinates")

        # Cast frequencies
        self._frequency = kwargs.pop('frequency')
        self._source = kwargs.pop('source')
        self.data = kwargs.pop('data')

        super().__init__(name, coordinates, **kwargs)


    def source(self):
        return self._source

    def n_src(self):
        return len(self.source)

    def frequency(self):
        return self._frequency

    def n_freq(self):
        return self.frequency.size

    # measured data
    # get e-field => needs survey
    # get h-field => needs survey

    # __repr__ method to print info

    # to_dict/from_dict
