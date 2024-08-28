"""
Thin wrappers to use emg3d as a forward modelling kernel within the
*Geophysical Inversion & Modelling Library* `pyGIMLi <https://pygimli.org>`_.

It deals mainly with converting the data and model from the emg3d format to the
pyGIMLi format and back, and creating the correct classes and functions as
expected by a pyGIMLi inversion.
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
import numpy as np

from emg3d import io, utils, _multiprocessing

try:
    import pygimli
    # Add pygimli and pgcore to the emg3d.Report().
    utils.OPTIONAL.extend(['pygimli', 'pgcore'])
except ImportError:
    pygimli = None

__all__ = ['Kernel', 'Inversion']


def __dir__():
    return __all__


class Kernel(pygimli.Modelling if pygimli else object):
    """Create a forward operator of emg3d to use within a pyGIMLi inversion.


    Parameters
    ----------
    simulation : Simulation
        The simulation; a :class:`emg3d.simulations.Simulation` instance.

    markers : ndarray of dtype int, default: None
        An ndarray of integers of the same shape as the model. All cells with
        the same number belong to the same region with this number, which can
        subsequently be defined through
        :func:`pygimli.frameworks.modelling.Modelling.setRegionProperties`.

    pgthreads : int, default: 2
        Number of threads for pyGIMLi (sets ``OPENBLAS_NUM_THREADS``). This is
        by default a small number, as the important parallelization in
        pyGIMLi(emg3d) happens over sources and frequencies in emg3d. This is
        controlled in the parameter ``max_workers`` when creating the
        simulation.

    """

    @utils._requires('pygimli')
    def __init__(self, simulation, markers=None, pgthreads=2):
        """Initialize a pyGIMLi(emg3d)-wrapper."""
        super().__init__()

        # Set pyGIMLi threads.
        pygimli.setThreadCount(pgthreads)

        # Check current limitations.
        checks = {
            'case': (simulation.model.case, 'isotropic'),
            'mapping': (simulation.model.map.name, 'Conductivity'),
        }
        for k, v in checks.items():
            if v[0] != v[1]:
                msg = f"pyGIMLi(emg3d) is not implemented for {v[0]} {k}."
                raise NotImplementedError(msg)

        # Store the simulation.
        self.simulation = simulation

        # Translate discretize TensorMesh to pygimli-Grid.
        mesh = pygimli.createGrid(
            x=simulation.model.grid.nodes_x,
            y=simulation.model.grid.nodes_y,
            z=simulation.model.grid.nodes_z,
        )

        # Set markers.
        if markers is not None:
            self.markers = markers.ravel('F')
        else:
            self.markers = np.arange(simulation.model.size, dtype=int)
        mesh.setCellMarkers(self.markers)
        # Store original props; required if a region is set to ``background``.
        self._model = simulation.model.copy()
        # Store volumes; required if a region is set to ``single``.
        self._volumes = simulation.model.grid.cell_volumes
        # Set mesh.
        self.setMesh(mesh)
        self._fullmodel = None

        # Create J, store and set it.
        self.J = self.Jacobian(
            simulation=self.simulation,
            data2gimli=self.data2gimli,
            data2emg3d=self.data2emg3d,
            model2gimli=self.model2gimli,
            model2emg3d=self.model2emg3d,
        )
        self.setJacobian(self.J)

    def response(self, model):
        """Create synthetic data for provided pyGIMLi model."""

        # Clean emg3d-simulation, so things are recomputed
        self.simulation.clean('computed')

        # Replace model
        self.simulation.model.property_x = self.model2emg3d(model)

        # Compute forward model and set initial residuals.
        _ = self.simulation.misfit

        # Return the responses as pyGIMLi array
        return self.data2gimli(self.simulation.data.synthetic.data)

    def createStartModel(self, dataVals=None):
        """Returns the model from the provided simulation."""
        return self.model2gimli(self.simulation.model.property_x)

    def createJacobian(self, model):
        """Dummy to prevent pyGIMLi from doing it the hard way."""

    def data2gimli(self, data):
        """Convert an emg3d data-xarray to a pyGIMLi data array."""
        out = data[self.simulation.survey.isfinite]
        if np.iscomplexobj(out):
            return np.hstack((out.real, out.imag))
        else:  # For standard deviation
            return np.hstack((out, out))

    def data2emg3d(self, data):
        """Convert a pyGIMLi data array to an emg3d data-xarray."""
        out = np.ones(
                self.simulation.survey.shape,
                dtype=self.simulation.data.observed.dtype
        )*np.nan
        data = np.asarray(data)
        ind = data.size//2
        out[self.simulation.survey.isfinite] = data[:ind] + 1j*data[ind:]
        return out

    def model2gimli(self, model):
        """Convert an emg3d Model property to a pyGIMLi model array.

        This function deals with the regions defined in pyGIMLi.
        """
        model = model.ravel('F')
        out = np.empty(self.simulation.model.size)

        # If the inversion model is smaller than the model, we have to
        # take care of the regions.
        if self.fullmodel:
            out[self.mesh().cellMarkers()] = model

        else:
            i = 0

            for n, v in sorted(self.regionProperties().items()):
                ni = self.markers == n
                if v['background'] or v['fix']:
                    ii = 0
                elif v['single']:
                    ii = 1
                    # TODO: Should av. happen on log-scale (not for jtvec)
                    out[i] = np.average(model[ni], weights=self._volumes[ni])
                else:
                    ii = np.sum(ni)
                    out[i:i+ii] = model[ni]
                i += ii

            out = out[:i]

        return out

    def model2emg3d(self, model, jvec=False):
        """Convert a pyGIMLi model array to an emg3d Model property.

        This function deals with the regions defined in pyGIMLi.
        """

        # If the inversion model is smaller than the model, we have to
        # take care of the regions.
        if self.fullmodel:

            out = np.asarray(model[self.mesh().cellMarkers()])

        else:

            # TODO: default for jvec: zeros, ones, something else?
            out = np.zeros(self.simulation.model.size)
            i = 0

            # TODO:: What about dsigma dm?

            for n, v in sorted(self.regionProperties().items()):
                ni = self.markers == n
                # TODO:: Zeros for background/fixed?
                if v['background']:
                    ii = 0
                    if not jvec:
                        out[ni] = self._model.property_x.ravel('F')[ni]
                elif v['fix']:
                    ii = 0
                    if not jvec:
                        out[ni] = v['startModel']
                elif v['single']:
                    ii = 1
                    out[ni] = model[i]
                    # TODO:: Necessary to normalize?
                    # if jvec:
                    #     out[ni] *= self._volumes[ni]/self._volumes[ni].sum()
                else:
                    ii = np.sum(ni)
                    out[ni] = model[i:ii+i]
                i += ii

        return out.reshape(self.simulation.model.shape, order='F')

    @property
    def fullmodel(self):
        """Flag if the full model is used for the inversion or not."""
        if self._fullmodel is None:
            self._fullmodel = True
            if self.regionProperties():
                keys = ['background', 'fix', 'single']
                for v in self.regionProperties().values():
                    if np.any([v[k] is True for k in keys]):
                        self._fullmodel = False
                        break

        return self._fullmodel

    class Jacobian(pygimli.Matrix if pygimli else object):
        """Return Jacobian operator for pyGIMLi(emg3d)."""

        def __init__(self, simulation,
                     data2gimli, data2emg3d, model2gimli, model2emg3d):
            """Initiate a new Jacobian instance."""
            super().__init__()
            self.simulation = simulation
            self.data2gimli = data2gimli
            self.data2emg3d = data2emg3d
            self.model2gimli = model2gimli
            self.model2emg3d = model2emg3d

        def cols(self):
            """The number of columns corresponds to the model size."""
            if not hasattr(self, '_cols'):
                gmodel = self.model2gimli(self.simulation.model.property_x)
                self._cols = len(gmodel)
            return self._cols

        def rows(self):
            """The number of rows corresponds to 2x data-size (Re; Im)."""
            if not hasattr(self, '_rows'):
                self._rows = self.simulation.survey.count * 2
            return self._rows

        def mult(self, x):
            """Multiply the Jacobian with a vector, Jm."""
            jvec = self.simulation.jvec(vector=self.model2emg3d(x, jvec=True))
            return self.data2gimli(jvec)

        def transMult(self, x):
            """Multiply  Jacobian transposed with a vector, Jᵀd = (dJᵀ)ᵀ."""
            jtvec = self.simulation.jtvec(self.data2emg3d(x))
            return self.model2gimli(jtvec)

        def save(self, *args):
            """There is no save for this pseudo-Jacobian."""


class Inversion(pygimli.Inversion if pygimli else object):
    """Thin wrapper, adding verbosity and taking care of data format."""

    @utils._requires('pygimli')
    def __init__(self, fop=None, inv=None, **kwargs):
        """Initialize an Inversion instance."""
        super().__init__(fop=fop, inv=inv, **kwargs)
        self._postStep = _post_step

    def run(self, dataVals=None, errorVals=None, **kwargs):
        """Run the inversion."""

        # Reset counter, start timer, print message.
        _multiprocessing.process_map.count = 0
        timer = utils.Timer()
        self.fop.simulation.timer = timer
        pygimli.info(":: pyGIMLi(emg3d) START ::")

        # Take data from the survey if not provided.
        if dataVals is None:
            dataVals = self.fop.data2gimli(
                    self.fop.simulation.data.observed.data)

        # Take the error from the survey if not provided.
        if errorVals is None:
            std_dev = self.fop.data2gimli(
                    self.fop.simulation.survey.standard_deviation.data)
            errorVals = std_dev / abs(dataVals)

        # Reset full-model flag.
        self.fop._fullmodel = None

        # Run the inversion
        out = super().run(dataVals=dataVals, errorVals=errorVals, **kwargs)

        # Store last model in the simulation
        self.fop.simulation.model.property_x = self.fop.model2emg3d(out)

        # Store last iteration as inversion result
        survey = self.fop.simulation.survey
        n = self.inv.iter()
        survey.data["inv"] = self.fop.simulation.data[f"it{n}"].copy()
        survey.data["inv"].data[np.invert(survey.isfinite)] *= np.nan

        # Print passed time and exit
        pygimli.info(f":: pyGIMLi(emg3d) END   :: runtime = {timer.runtime}")


def _post_step(n, inv):
    """Print some values for each iteration."""
    sim = inv.fop.simulation

    # Print info
    phi = inv.inv.getPhi()
    if n == 0:
        pygimli.info(
            f"{71*'='}\n{39*' '}"
            " it        χ²   F(m)       λ         ϕᵈ         ϕᵐ"
            f"   ϕ=ϕᵈ+λϕᵐ   Δϕ (%)\n{39*' '}{71*'-'}"
        )
        deltaphi = 0
        sim.invinfo = {}

    else:
        deltaphi = (1-phi/inv.lastphi)*100
    inv.lastphi = phi
    pygimli.info(
        f"{n:3d}{_round_format(inv.inv.chi2())}"
        f"{_multiprocessing.process_map.count:7d}{inv.inv.getLambda():8.3}"
        f" {_round_format(inv.inv.getPhiD())}"
        f" {_round_format(inv.inv.getPhiM())} {_round_format(phi)}"
        f"{deltaphi:9.2f}"
    )

    # Store data TODO store everything required to reproduce or restart inv
    sim.survey.data[f"it{n}"] = sim.survey.data.synthetic
    # TODO store data and model as gimli vectors; requires region info
    model = inv.fop.simulation.model.copy()
    model.property_x = inv.fop.model2emg3d(inv.model)
    sim.invinfo[n] = {
        'model': model,
        'chi2': inv.inv.chi2(),
        'phi': phi,
        'phi_d': inv.inv.getPhiD(),
        'phi_m': inv.inv.getPhiM(),
        'phi_delta': deltaphi,
        'count': _multiprocessing.process_map.count,
        'lambda': inv.inv.getLambda(),
        'time': sim.timer.elapsed,
    }
    if sim.name:
        io.save(
            f"{sim.name}.h5",
            simulation=sim.to_dict(what='plain'),
            invinfo=sim.invinfo,
            verb=0,
        )

    # Reset counter
    _multiprocessing.process_map.count = 0


def _round_format(x, ndigit=1, threshold=1e6, lower='10.1f', upper='10.3'):
    """Rounding numbers for info display."""
    return f"{np.round(x, ndigit):{lower if x < threshold else upper}}"
