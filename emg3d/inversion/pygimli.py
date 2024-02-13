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
    """Create a Jacobian operator to use emg3d within a pyGIMLi inversion.

    This never builds the actual Jacobian, but provides functions to compute
    the Jacobian times a model vector (Jm) and the Jacobian transposed times
    a data vector (Jᵀd).


    Parameters
    ----------
    simulation : Simulation
        The emg3d simulation a :class:`emg3d.simulations.Simulation` instance.

    """

    def __init__(self, simulation):
        """Initiate a new Jacobian instance."""
        super().__init__()

        # Store the simulation.
        self.simulation = simulation

    def cols(self):
        """The number of columns corresponds to the model size."""
        return self.simulation.model.size

    def rows(self):
        """The number of rows corresponds to the data size.

        To be precise, the data size times two, to account for real and
        imaginary parts.
        """
        return self.simulation.survey.count * 2

    def mult(self, x):
        """Multiply Jacobian with a vector, Jm."""
        self.simulation._count_jvec += 1

        # Compute jvec.
        jvec = self.simulation.jvec(
            vector=np.reshape(x, self.simulation.model.shape, order='F')
        )

        # Get non-NaN data.
        data = jvec[self.simulation.survey.isfinite]

        # Return real and imaginary parts stacked.
        return np.hstack((data.real, data.imag))

    def transMult(self, x):
        """Multiply  Jacobian transposed with a vector, Jᵀd = (dJᵀ)ᵀ."""
        self.simulation._count_jtvec += 1

        # Cast finite [Re, Im] data from pyGIMLi into the emg3d format.
        data = np.ones(
                self.simulation.survey.shape,
                dtype=self.simulation.data.observed.dtype
        )*np.nan
        x = np.asarray(x)
        xl = x.size//2
        data[self.simulation.survey.isfinite] = x[:xl] + 1j*x[xl:]

        # Return jtvec as a 1D array.
        return self.simulation.jtvec(data).ravel('F')

    def save(self, *args):
        """There is no save for this pseudo-Jacobian."""
        pass


@utils._requires('pygimli')
class Kernel(pygimli.Modelling):
    """Create a forward operator to use emg3d within a pyGIMLi inversion.


    Parameters
    ----------
    simulation : Simulation
        The emg3d simulation a :class:`emg3d.simulations.Simulation` instance.

    """

    def __init__(self, simulation):
        """Initialize the pyGIMLi(emg3d)-wrapper."""

        # Initiate first pygimli.Modelling, which will do its magic.
        super().__init__()

        # Check limitation: Only isotropic so far.
        mcase = simulation.model.case
        if mcase != 'isotropic':
            raise NotImplementedError(
                f"pyGIMLi(emg3d) not implemented for {mcase} case."
            )

        # Store the simulation.
        self.simulation = simulation

        # HEEEEEEEERE TODO

        # Define J and setJacobian
        self.J = Jacobian(simulation)  # TODO do we have to store sim again?
        self.setJacobian(self.J)

    def response(self, model):
        """Create synthetic data for provided model."""

        # Clean emg3d-simulation, so things are recomputed
        self.simulation.clean('computed')

        # Replace model
        self.simulation.model = models.Model(
            grid=self.simulation.model.grid,
            property_x=model,
            mapping='Conductivity'
        )

        # Compute forward model and set initial residuals.
        self.simulation._count_forward += 1
        _ = self.simulation.misfit

        # Return the responses
        data = self.simulation.data.synthetic.data[
                self.simulation.survey.isfinite
        ]
        return np.hstack((data.real, data.imag))

    def createStartModel(self, dataVals=None):
        """Returns the model from the provided simulation."""
        return self.simulation.model.property_x.ravel('F')

    def createJacobian(self, model):
        """Dummy to prevent pygimli.Modelling from doing it the hard way."""
        pass  # do nothing


def post_step(n, inv):
    """TODO"""

    kc = (inv.fop.simulation._count_forward
          + inv.fop.simulation._count_jvec
          + inv.fop.simulation._count_jtvec)
    cglsit = max(0, inv.fop.simulation._count_jvec-1)
    phi = inv.inv.getPhi()
    if not hasattr(inv, 'lastphi'):
        lastphi = ""
    else:
        lastphi = f"; Δϕ = {(1-phi/inv.lastphi)*100:.2f}%"
    inv.lastphi = phi
    pygimli.info(
        f"{n}: "
        f"χ² = {inv.inv.chi2():7.2f}; "
        f"λ = {inv.inv.getLambda()}; "
        f"#CGLS {cglsit:2d} ({kc:2d} solves); "
        f"ϕ = {inv.inv.getPhiD():.2f} + {inv.inv.getPhiM():.2f} = "
        f"{phi:.2f}{lastphi}"
    )

    # Reset counters
    inv.fop.simulation._count_forward = 0
    inv.fop.simulation._count_jvec = 0
    inv.fop.simulation._count_jtvec = 0

    # TODO: save data, model, and everything to re-start inversion.

    # inv.chi2History
    # inv.modelHistory


@utils._requires('pygimli')
class Inversion(pygimli.Inversion):

    def __init__(self, fop, inv=None, **kwargs):
        fop._count_forward = 0
        fop._count_jvec = 0
        fop._count_jtvec = 0
        super().__init__(fop=Kernel(fop), inv=inv,  **kwargs)

        # Translate discretize TensorMesh to pygimli-Grid.
        # => TODO move to Kernel __init__, where it belongs
        # Maybe that could be a dummy mesh of sizes 1
        self.inv_mesh = pygimli.createGrid(
            x=self.fop.simulation.model.grid.nodes_x,
            y=self.fop.simulation.model.grid.nodes_y,
            z=self.fop.simulation.model.grid.nodes_z,
        )

        self.setPostStep(post_step)

    def run(self, dataVals=None, errorVals=None, **kwargs):

        pygimli.info("pyGIMLi(emg3d) START")
        itime = utils.Timer()  # Timer.

        # Set the mesh.
        # => TODO move to Kernel __init__, where it belongs
        self.fop.setMesh(self.inv_mesh)

        # Take data from the survey if not provided.
        if dataVals is None:
            finite_data = self.fop.simulation.survey.finite_data()
            dataVals = np.hstack([finite_data.real, finite_data.imag])

        # Take the error from the survey if not provided.
        if errorVals is None:
            # TODO - IS THIS CORRECT?
            std_dev_full = self.fop.simulation.survey.standard_deviation
            std_dev = std_dev_full.data[self.fop.simulation.survey.isfinite]
            errorVals = np.hstack([std_dev, std_dev]) / abs(dataVals)

            # TODO does it make any difference, is it needed?
            # To completely ignore big errors
            # => Test if it is actually necessary or not
            errorVals[errorVals > 0.5] = 1e8

        # Run the inversion.
        out = super().run(dataVals=dataVals, errorVals=errorVals, **kwargs)

        # Print passed time and exit.
        pygimli.info(f"pyGIMLi(emg3d) END :: {itime.runtime}")

        return out
