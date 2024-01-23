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

__all__ = ['Kernel', 'Inversion', 'Jacobian']

# Add pygimli and pgcore to the emg3d.Report().
utils.OPTIONAL.extend(['pygimli', 'pgcore'])


def __dir__():
    return __all__

# TODO: Create functions
# - convert_model(from/to)
# - convert_data(from/to)


@utils._requires('pygimli')
class Jacobian(pygimli.Matrix):
    """Create a Jacobian operator for emg3d which is understood by pyGIMLi.

    This never builds the actual Jacobian, but provides functions to compute
    the

    - Jacobian times a model vector ``Jm``
      (``jvec`` in emg3d, ``mult`` in pyGIMLi), and the
    - Jacobian transposed times a data vector ``Jᵀd``
      (``jtvec`` in emg3d, ``transMult`` in pyGIMLi).


    Parameters
    ----------
    simulation : Simulation
        The simulation; a :class:`emg3d.simulations.Simulation` instance.

    mesh : Mesh
        The mesh; :func:`pygimli.meshtools.grid.createGrid` instance.

    """

    def __init__(self, simulation, mesh):
        """Initiate a new Jacobian instance."""
        super().__init__()

        # Store pointers to the emg3d-simulation and the pyGIMLi-mesh.
        self.simulation = simulation
        self.mesh = mesh

        # Store n-cols and n-rows.
        self._cols = simulation.model.size
        self._rows = simulation.survey.count * 2

    def cols(self):
        """The number of columns corresponds to the model size."""
        return self._cols

    def rows(self):
        """The number of rows corresponds to 2x(data size), for [Re; Im]."""
        return self._rows

    def mult(self, x):
        """Multiply the Jacobian with a vector, Jm."""

        self.simulation._count_jvec += 1

        # Resort x to represent the model.
        model = x[self.mesh().cellMarkers()]

        # Compute jvec.
        jvec = self.simulation.jvec(
            vector=np.reshape(model, self.simulation.model.shape, order='F')
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

        # Compute jtvec.
        jtvec = self.simulation.jtvec(data).ravel('F')

        # Resort jtvec according to regions.
        out = np.empty(jtvec.size)
        out[self.mesh().cellMarkers()] = jtvec
        return out

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

    def __init__(self, simulation, markers=None, pgthreads=2):
        """Initialize the pyGIMLi(emg3d)-wrapper."""

        pygimli.setThreadCount(pgthreads)

        # TODO move this to the Simulation class!
        simulation._count_forward = 0
        simulation._count_jvec = 0
        simulation._count_jtvec = 0

        # Check current limitation 1: isotropic.
        mcase = simulation.model.case
        if mcase != 'isotropic':
            raise NotImplementedError(
                f"pyGIMLi(emg3d) not implemented for {mcase} case."
            )

        # Check current limitation 2: conductivity.
        mname = simulation.model.map.name
        if mname != 'Conductivity':
            raise NotImplementedError(
                f"pyGIMLi(emg3d) not implemented for {mname} mapping."
            )

        # Initiate first pygimli.Modelling, which will do its magic.
        super().__init__()

        # Store the simulation.
        self.simulation = simulation

        # Translate discretize TensorMesh to pygimli-Grid.
        mesh = pygimli.createGrid(
            x=simulation.model.grid.nodes_x,
            y=simulation.model.grid.nodes_y,
            z=simulation.model.grid.nodes_z,
        )

        if markers is not None:
            mesh.setCellMarkers(markers)

        self.setMesh(mesh)

        # Store J and set it
        self.J = Jacobian(self.simulation, self.mesh)
        self.setJacobian(self.J)

    def response(self, model):
        """Create synthetic data for provided model."""

        # Clean emg3d-simulation, so things are recomputed
        self.simulation.clean('computed')

        # Replace model
        self.simulation.model = models.Model(
            grid=self.simulation.model.grid,
            # Resort inversion model array to represent the actual model.
            property_x=model[self.mesh().cellMarkers()],
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


def _post_step(n, inv):
    """TODO"""

    # TODO: save data, model, and everything to re-start inversion.

    # inv.chi2History
    # inv.modelHistory

    sim = inv.fop.simulation

    kc = sim._count_forward + sim._count_jvec + sim._count_jtvec
    sim.survey.data[f"it{n}"] = sim.survey.data.synthetic
    cglsit = max(0, sim._count_jvec-1)
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
        f"ϕ = {inv.inv.getPhiD():.2f} + {inv.inv.getPhiM():.2f}·λ = "
        f"{phi:.2f}{lastphi}"
    )

    # Reset counters
    sim._count_forward = 0
    sim._count_jvec = 0
    sim._count_jtvec = 0


@utils._requires('pygimli')
class Inversion(pygimli.Inversion):
    """TODO"""
    def __init__(self, fop=None, inv=None, **kwargs):
        super().__init__(fop=fop, inv=inv, **kwargs)
        self._postStep = _post_step

    def run(self, dataVals=None, errorVals=None, **kwargs):
        timer = utils.Timer()
        pygimli.info(":: pyGIMLi(emg3d) START ::")

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

        # Run the inversion
        out = super().run(dataVals=dataVals, errorVals=errorVals, **kwargs)

        # Print passed time and exit
        pygimli.info(f":: pyGIMLi(emg3d) END   :: runtime = {timer.runtime}")

        return out
