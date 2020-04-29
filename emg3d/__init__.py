"""
Electromagnetic modeller in the diffusive limit (low frequencies) for 3D media
with tri-axial electrical anisotropy. The matrix-free multigrid solver can be
used as main solver or as preconditioner for one of the Krylov subspace methods
implemented in :mod:`scipy.sparse.linalg`, and the governing equations are
discretized on a staggered Yee grid. The code is written completely in Python
using the ``numpy``/``scipy``-stack, where the most time-consuming parts are
sped-up through jitted ``numba``-functions.
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

# Import modules
from emg3d.utils import io
from emg3d.utils import maps
from emg3d.utils import misc
from emg3d.utils import fields
from emg3d.utils import meshes
from emg3d.utils import models
from emg3d.optimize import gradient

# Import most important functions and classes
from emg3d.solver import solve
from emg3d.utils.misc import Report
from emg3d.utils.fields import Field
from emg3d.utils.models import Model
from emg3d.utils.io import save, load
from emg3d.utils.meshes import TensorMesh

# # For top-namespace
from emg3d.utils.misc import Fourier  # noqa
from emg3d.utils.maps import grid2grid  # noqa
from emg3d.utils.meshes import get_hx_h0  # noqa
from emg3d.utils.fields import (
        get_source_field, get_receiver, get_h_field)  # noqa

__all__ = ['solve', 'fields', 'io', 'maps', 'meshes', 'models', 'utils',
           'optimize', 'Field', 'Model', 'TensorMesh', 'Report', 'save',
           'load']

# Version defined in misc, so we can easier use it within the package itself.
__version__ = misc.__version__
