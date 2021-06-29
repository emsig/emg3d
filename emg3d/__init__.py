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

# Import most important functions and classes
from emg3d.electrodes import (
        TxElectricPoint, TxElectricDipole, TxMagneticDipole, TxElectricWire,
        RxElectricPoint, RxMagneticPoint,
)
from emg3d.fields import Field, get_source_field, get_magnetic_field
from emg3d.io import save, load
from emg3d.meshes import TensorMesh, construct_mesh
from emg3d.models import Model
from emg3d.simulations import Simulation
from emg3d.solver import solve, solve_source
from emg3d.surveys import Survey
from emg3d.time import Fourier
from emg3d.utils import Report, __version__
