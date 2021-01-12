"""
Electromagnetic modeller in the diffusive limit (low frequencies) for 3D media
with tri-axial electrical anisotropy. The matrix-free multigrid solver can be
used as main solver or as preconditioner for one of the Krylov subspace methods
implemented in :mod:`scipy.sparse.linalg`, and the governing equations are
discretized on a staggered Yee grid. The code is written completely in Python
using the ``numpy``/``scipy``-stack, where the most time-consuming parts are
sped-up through jitted ``numba``-functions.
"""
# Copyright 2018-2021 The emg3d Developers.
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

# Import modules
from emg3d import io
from emg3d import maps
from emg3d import utils
from emg3d import solver
from emg3d import fields
from emg3d import meshes
from emg3d import models
from emg3d import surveys
from emg3d import optimize
from emg3d import simulations

# Import most important functions and classes
from emg3d.solver import solve
from emg3d.utils import Report
from emg3d.io import save, load

# # For top-namespace
from emg3d import core  # noqa
from emg3d.models import Model  # noqa
from emg3d.utils import Fourier  # noqa
from emg3d.surveys import Survey  # noqa
from emg3d.simulations import Simulation  # noqa
from emg3d.meshes import TensorMesh, construct_mesh  # noqa
from emg3d.fields import (Field, get_source_field, get_receiver,  # noqa
                          get_receiver_response, get_h_field)

__all__ = ['solve', 'solver', 'utils', 'io', 'fields', 'maps', 'meshes',
           'models', 'Report', 'save', 'load', 'surveys', 'simulations',
           'optimize']

# Version defined in utils, so we can easier use it within the package itself.
__version__ = utils.__version__
