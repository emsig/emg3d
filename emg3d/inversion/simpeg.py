"""
Thin wrappers to use emg3d as a forward modelling kernel within the package
*Simulation and Parameter Estimation in Geophysics* `SimPEG
<https://simpeg.xyz>`_.

It deals mainly with converting the data and model from the emg3d format to the
SimPEG format and back, and creating the correct classes and functions as
expected by a SimPEG inversion.
"""
# Copyright 2024 The emsig community.
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

from emg3d import utils  # electrodes, meshes, models, simulations, surveys

try:
    import simpeg
    # import discretize
    # from simpeg.electromagnetics import frequency_domain as simpeg_fd
    # Add simpeg to the emg3d.Report().
    utils.OPTIONAL.extend(['simpeg',])
except ImportError:
    simpeg = None


__all__ = []


def __dir__():
    return __all__


if simpeg is not None:
    print("NOTE: SimPEG(emg3d) is in development.")
