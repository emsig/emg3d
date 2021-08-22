"""
Functionalities related to optimization (minimization, inversion), such as the
misfit function and its gradient.

DEPRECATED, will be removed from v1.4.0 onwards.
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


def misfit(simulation):
    """Deprecated, moved directly to `emg3d.Simulation.misfit`."""
    msg = ("emg3d: `optimize` is deprecated and will be removed in v1.4.0."
           "`optimize.misfit` is embedded directly in Simulation.misfit`.")
    warnings.warn(msg, FutureWarning)
    return simulation.misfit


def gradient(simulation):
    """Deprecated, moved directly to `emg3d.Simulations.gradient`."""
    msg = ("emg3d: `optimize` is deprecated and will be removed in v1.4.0."
           "`optimize.gradient` is embedded directly in Simulation.gradient`.")
    warnings.warn(msg, FutureWarning)
    return simulation.gradient
