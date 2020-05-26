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
from typing import Tuple
from dataclasses import dataclass, field

# from emg3d import maps, models

# __all__ = []


class Survey:
    """Create a survey with sources and receivers.

    Survey takes care to join same receivers
    Survey should check for
    - source-receiver combinations,
    - report missing sources
    - list nfreq per source and all the rest


    - Need to get frequencies by receivers and by source v
    - Need to get sources and receivers by frequency     ^

    Parameters
    ----------
    ?

    """

    def __init__(self):
        """Initiate a new Survey instance."""

        # TODO We should be able to provide source and receiver list straight
        #      to the survey, I think.

        # Initiate dictionaries.
        self._receivers = {}
        self._sources = {}
        self._frequencies = {}
        self._data = {}

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

    def add_receivers(self, name, coordinates):
        """Add receiver(s) to the survey.

        Parameters
        ----------
        name : str or list
            Receiver name(s).

        coordinates : tuple
            `(x, y, z, azimuth, dip)` [m, m, m, °, °]; dimensions must be
            compatible (also with `name`).

        """

        # TODO: Add recursion for receiver list.

        # Warn about duplicate names.
        if name in self._receivers:
            print(f"* WARNING :: Overwriting existing receiver <{name}>:\n"
                  f"             - Old: {self._receivers[name].coordinates}\n"
                  f"             - New: {coordinates}")

        # Create a receiver dipole.
        self._receivers[name] = DipoleReceiver(name, coordinates)

    def remove_receivers(self, names):
        # TODO
        # Delete from receiver-list
        # Delete corresponding data for all sources and frequencies
        pass


    def add_sources(self, names, coordinates):
        """Add source(s) to the survey.

        Parameters
        ----------
        names : str or list
            Source name(s).

        coordinates : tuple
            Coordinates in one of the two following formats:

            - `(x, y, z, azimuth, dip)` [m, m, m, °, °];
            - `(x0, x1, y0, y1, z0, z1)` [m, m, m, m, m, m].

            Dimensions must be compatible (also with `name`).

        """

        if isinstance(names, list):
            # TODO: Add recursion for source list.
            for name in names:
                self.add_sources(name, coordinates)

        else:
            # Warn about duplicate names.
            if names in self._sources:
                old_coords = self._sources[names].coordinates
                print(f"* WARNING :: Overwriting existing source <{names}>:\n"
                      f"             - Old: {old_coords}\n"
                      f"             - New: {coordinates}")

            # Create a receiver dipole.
            self._sources[names] = DipoleSource(names, coordinates)

    def remove_sources(self):
        pass

    def add_data(self, source, receiver, frequency, data=None):
        """Add receiver(s) to the survey.

        Parameters
        ----------
        source, receiver : str
            Source and receiver name(s).

        frequencies : float or array
            Frequencies [Hz].

        data : None or array (nsrc, nfreq, nrec)
            Measured data.

        """

        # TODO: Add recursion for receiver list.



        # Warn about duplicate names.
        print(self._receivers)
        if name in self._receivers:
            print(f"* WARNING :: Overwriting existing receiver <{name}>.")

        else:
            # Create a receiver dipole.
            self._receivers[name] = DipoleReceiver(name, coordinates)
            # (nsrc, nfreq)
            # frequencies=frequencies,
            # sources=sources,
            # data=data,


        # Only name should be required
        # coordinates are optionally, name is enough if once defined
        # frequency and source are optionally
        # data only if frequency and source provided

        # # Move to add
        # self._names = dict()
        # for name in names:
        #     self._names[name] = dict()

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

    # TODO: Get measured data
    # get e-field => needs survey
    # get h-field => needs survey


# Should it be a dataclass? In the future they might support validators.
@dataclass(eq=False)
class Dipole:
    """
    Base dipole (source or receiver).

    Maybe this should be a dataclass.
    """

    name: str

    # Coordinates is either Tuple[5*float] or Tuple[6*float]; probably not
    # ideal: (x, y, z, azimuth, dip) OR (x0, x1, y0, y1, z0, z1)
    coordinates: Tuple[float]

    is_finite: bool = field(init=False)

    def __post_init__(self, **kwargs):

        # Check coordinates.
        try:
            # Conversion to float-array fails if there are lists and tuples
            # within the tuple, or similar.
            # This should catch many wrong inputs, hopefully.
            test = np.array(self.coordinates, dtype=float)

            # Check size:
            if test.size == 5:
                print("point dipole")
                self.is_finite = False
            elif test.size == 6:
                print("finite length dipole")
                self.is_finite = True
            else:
                raise ValueError

        except ValueError:
            print("* ERROR   :: Dipole coordinates are wrong defined.\n"
                  "             They must describe either a point,\n"
                  "             (x, y, z, azimuth, dip), or a finite dipole,\n"
                  "             (x1, x2, y1, y2, z1, z2).\n"
                  f"             Provided coordinates: {self.coordinates}.")
            raise ValueError("Dipole coordinates")

        # # Store name and coordinates
        # self.name = name
        # self.coordinates = coordinates

        # Currently warn if kwargs left.
        # TODO: Change to Error.
        if kwargs:
            print(f"* WARNING :: Remaining kwargs: {kwargs}")

    # def __repr__(self):
    #     """Simple representation."""
    #     return (f"{self.__class__.__name__}: {self.coordinates}")

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
        # TODO
        pass

    @property
    def dip(self):
        """Implement that there is always a dip, not only for point dipoles."""
        if not hasattr(self, '_dip'):
            if self.is_finite:
                print('TODO calculate dip')
                # TODO
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
                # TODO
                raise NotImplementedError
            else:
                self._azimuth = self.coordinates[3]

        return self._azimuth


class DipoleSource(Dipole):
    """
    Maybe this should be a dataclass.
    """

    def __init__(self, name, coordinates, **kwargs):

        super().__init__(name, coordinates, **kwargs)


class DipoleReceiver(Dipole):
    """
    Maybe this should be a dataclass, and only data should be a dataclass.
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

        super().__init__(name, coordinates, **kwargs)
