"""

:mod:`simulation` -- Compute EM responses for a survey
======================================================

In its heart, `emg3d` is a multigrid solver for 3D electromagnetic diffusion
with tri-axial electrical anisotropy. However, it contains most functionalities
to also act as a modeller. The simulation module combines all these things
by combining surveys with computational meshes and fields and providing
high-level, specialised modelling routines.

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


class Simulation():
    r"""Create a simulation with survey, meshes, and fields.


    Parameters
    ----------

    """

    def __init__(self, sources, receivers, **kwargs):
        """Initiate a new Simulation instance."""
        # solver options
        # min_amp to consider
        # min_offset, max_offset
        pass

    def copy(self):
        """Return a copy of the Simulation."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Simulation in a dict."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Simulation` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Simulation.to_dict`.
            The dictionary needs the keys TODO.

        Returns
        -------
        obj : :class:`Simulation` instance

        """
        raise NotImplementedError

    # connected survey

    # connected model(s)

    # resulting meshes

    # compute E-Source (H-source)  [forward]
    # get E-field (H-field)        [interpolation]


def model_marine_csem():
    # takes a model; fills up water if req., adds air
    # takes a survey -> deduces computational domain from that
    # takes gridding parameters
    pass
