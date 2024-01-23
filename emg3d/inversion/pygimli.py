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

import numpy as np

try:
    import pygimli
except ImportError:
    pygimli = None

from emg3d import models, utils

__all__ = ['Inversion']

utils.OPTIONAL.extend(['pygimli', 'pgcore'])


def __dir__():
    return __all__


@utils._requires('pygimli')
class Jacobian(pygimli.Matrix):
    """Create a Jacobian operator to use emg3d as kernel within pyGIMLi."""

    def __init__(self, sim):
        super().__init__()
        self.sim = sim

    def cols(self):
        """ToDo"""
        # sim.model.size corresponds to the number of cells
        return self.sim.model.size

    def rows(self):
        """ToDo"""
        # sim.survey.count corresponds to the number of non-NaN data points.
        # Multiply by 2 for real + imaginary parts.
        return self.sim.survey.count * 2

    def mult(self, x):
        """J * x """
        # - Input x has size of the model; later, we want to generalize that
        #   to allow for anisotropic models etc.
        # - Output has size of non-NaN data [Re, Im]
        jvec = self.sim.jvec(np.reshape(x, self.sim.model.shape, order='F'))
        data = jvec[self.sim._finite_data]
        return np.hstack((data.real, data.imag))

    def transMult(self, x):
        """J.T * x = (x * J.T)^T """
        # - Input has size of non-NaN data [Re, Im]
        # - Output has size of the model
        data = np.ones(
                self.sim.survey.shape,
                dtype=self.sim.data.observed.dtype
        )*np.nan
        x = np.asarray(x)
        data[self.sim._finite_data] = x[:x.size//2] + 1j*x[x.size//2:]
        return self.sim.jtvec(data).ravel('F')

    def save(self, *args):
        """ToDo"""
        pass


@utils._requires('pygimli')
class Kernel(pygimli.Modelling):
    """Create a forward operator to use emg3d as kernel within pyGIMLi."""

    def __init__(self, sim):
        """Initialize the pyGIMLi(emg3d)-wrapper."""

        # Initiate first pygimli.Modelling, which will do its magic.
        super().__init__()

        # Check current limitations; PURELY for development
        # IMPROVE (implement fully or convert to checks, do not do assert!)
        assert sim.model.case == 'isotropic'
        assert sim.model.map.name == 'Conductivity'

        # Add a bool to the simulation which selects all data
        # which are finite (this should go into emg3d.survey directly!)
        sim._finite_data = np.isfinite(sim.data.observed.data)

        # Store the simulation
        self.sim = sim

        # Translate discretize TensorMesh to pygimli-Grid
        self.mesh_ = pygimli.createGrid(
            x=sim.model.grid.nodes_x,
            y=sim.model.grid.nodes_y,
            z=sim.model.grid.nodes_z,
        )

        # Set marker -> water is 1, subsurface is 0
        # JUST TO DEVELOP, this SHOULD NOT be hard-coded in the wrapper
        self.mesh_.setCellMarkers(pygimli.z(self.mesh_.cellCenters()) > 0)

        # Set the mesh properly
        self.setMesh(self.mesh_)

        # Define J and setJacobian
        self.J = Jacobian(sim)
        self.setJacobian(self.J)

        # Store obs-data and obs-error
        cplx_data = sim.data.observed.data[sim._finite_data]
        self.obs_data = np.hstack([cplx_data.real, cplx_data.imag])

        abs_errors = sim.survey.standard_deviation.data[sim._finite_data]
        self.obs_errors = np.hstack(
                [abs_errors, abs_errors]
        ) / abs(self.obs_data)
        # To completely ignore big errors
        # => Test if it is actually necessary or not
        self.obs_errors[self.obs_errors > 0.5] = 1e8

    def response(self, model):
        """Create synthetic data for provided model."""

        # Clean emg3d-simulation, so things are recomputed
        self.sim.clean('computed')

        # Replace model
        self.sim.model = models.Model(
            grid=self.sim.model.grid,
            property_x=model,
            mapping='Conductivity'
        )

        # Compute forward model and set initial residuals.
        _ = self.sim.misfit

        # Return the responses
        data = self.sim.data.synthetic.data[self.sim._finite_data]
        return np.hstack((data.real, data.imag))

    def createStartModel(self, dataVals):   # NOT SURE ABOUT THIS
        """Create a start model...????"""
        # Use the model in the simulation as starting model
        # => make this more flexibel!
        return self.sim.model.property_x.ravel('F')

    def createJacobian(self, model):
        """Dummy to prevent pygimli.Modelling from doing it the hard way."""
        pass  # do nothing


@utils._requires('pygimli')
class Inversion(pygimli.Inversion):

    def __init__(self, fop, inv=None, **kwargs):
        super().__init__(fop=Kernel(fop), inv=inv,  **kwargs)

    def run(self, dataVals=None, errorVals=None, **kwargs):
        super().run(
            dataVals=self.fop.obs_data if dataVals is None else dataVals,
            errorVals=self.fop.obs_errors if errorVals is None else errorVals,
            **kwargs
        )
