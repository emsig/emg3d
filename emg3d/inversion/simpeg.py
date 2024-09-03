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
import numpy as np

from emg3d import electrodes, io, meshes, models, utils, _multiprocessing


try:
    import simpeg
    import discretize
    from simpeg.electromagnetics import frequency_domain as simpeg_fd
    # Add simpeg to the emg3d.Report().
    utils.OPTIONAL.extend(['simpeg', 'pydiso'])
except ImportError:
    simpeg = None


__all__ = ['FDEMSimulation', 'Inversion']


def __dir__():
    return __all__


class FDEMSimulation(
        simpeg_fd.simulation.BaseFDEMSimulation if simpeg else object):
    """Create a forward operator of emg3d to use within a SimPEG inversion.

    TODO - check limitation!

    .. note::

        Currently only isotropic models are implemented, with unit relative
        electric permittivity and unit relative magnetic permeability.


    Parameters
    ----------
    simulation : Simulation
        The simulation; a :class:`emg3d.simulations.Simulation` instance.

    All other parameters will be forwarded to
    :class:`simpeg.electromagnetic.frequency_domain.simulation.BaseFDEMSimulation`.

    """

    @utils._requires("simpeg")
    def __init__(
        self, simulation, active_indices, imap=simpeg.maps.ExpMap, **kwargs
    ):
        """Initialize Simulation using emg3d as solver."""

        # Store simulation
        self.simulation = simulation
        self.grid = simulation.model.grid

        # Create conductivity map
        self.inds_active = active_indices.ravel('F')
        conductivity_map = simpeg.maps.InjectActiveCells(
            self.grid,
            self.inds_active,
            simulation.model.property_x.ravel('F')[~self.inds_active],
        ) * imap(nP=int(self.inds_active.sum()))

        # Instantiate SimPEG Simulation
        super().__init__(
            mesh=self.grid,
            survey=self.survey2simpeg(simulation.survey),
            sigmaMap=conductivity_map,
            **kwargs
        )

        # Replace once https://github.com/simpeg/simpeg/pull/1523 is released
        data = Data(
            self.survey,
            dobs=self.data2simpeg(simulation.data.observed.data),
            standard_deviation=self.data2simpeg(
                simulation.survey.standard_deviation.data
            ),
        )

        self.m0 = simulation.model.property_x.ravel('F')[self.inds_active]
        self.m0 *= np.log(1.)

        # Replace once https://github.com/simpeg/simpeg/pull/1524 is released
        self.dmis = L2DataMisfit(data=data, simulation=self)

    @property
    def _di(self):
        """Return tuple of indices linking emg3d xarray to SimPEG array."""

        if not hasattr(self, '_data_indices'):
            size = self.simulation.survey.size
            shape = self.simulation.survey.shape

            # Get boolean in the right order for SimPEG
            dmap = self.simulation.survey.isfinite.reshape(
                (shape[0], -1), order='F').ravel()

            # Create indices tuple for the booleans
            self._data_indices = (
                np.arange(size)[dmap] // shape[1] // shape[2],  # src
                np.arange(size)[dmap] % shape[1],               # rec
                np.arange(size)[dmap] // shape[1] % shape[2]    # freq
            )

        return self._data_indices

    def data2simpeg(self, data):
        """Convert an emg3d data-xarray to a SimPEG data array."""
        return data[self._di[0], self._di[1], self._di[2]]

    def data2emg3d(self, data):
        """Convert a SimPEG data array to an emg3d data-xarray."""

        # Dummy array
        if not hasattr(self, '_data_array'):
            self._data_array = np.ones(
                self.simulation.survey.shape,
                self.simulation.survey.data.observed.dtype
             )*np.nan

        # Put data on dummy.
        self._data_array[self._di[0], self._di[1], self._di[2]] = data
        return self._data_array

    @property
    def model2emg3d(self):
        """emg3d conductivity model; obtained from SimPEG conductivities."""
        return models.Model(
            meshes.TensorMesh(self.mesh.h, self.mesh.origin),
            property_x=self.sigma.reshape(self.mesh.shape_cells, order='F'),
            # property_y=None,  Not yet implemented
            # property_z=None,   "
            # mu_r=None,         "
            # epsilon_r=None,    "
            mapping='Conductivity',
        )

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector.

        Parameters
        ----------
        TODO
        m : numpy.ndarray
            Inversion model (nP,)
        v : numpy.ndarray
            Vector which we take sensitivity product with (nP,)
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            Fields object

        Returns
        -------
        TODO
        """
        if self.verbose:
            print("Compute Jvec")

        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = discretize.utils.mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        dsig_dm_v = (self.sigmaDeriv @ v).reshape(
                self.simulation.model.shape, order='F')
        j_vec = f.jvec(vector=dsig_dm_v)

        # Map emg3d-data-array to SimPEG-data-vector
        return self.data2simpeg(j_vec)

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transpose times a vector

        Parameters
        ----------
        TODO
        m : numpy.ndarray
            Inversion model (nP,)
        v : numpy.ndarray
            Vector which we take adjoint product with (ndata,)
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            Fields object

        Returns
        -------
        TODO
        """
        if self.verbose:
            print("Compute Jtvec")

        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = discretize.utils.mkvc(np.dot(J.T, self.data2emg3d(v)))

            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """Compute adjoint sensitivity matrix (J^T) and vector (v) product."""
        if v is not None:
            return self.sigmaDeriv.T @ f.jtvec(self.data2emg3d(v)).ravel('F')
        else:
            raise NotImplementedError

    def getJ(self, m, f=None):
        """Generate full sensitivity matrix (not implemented)."""
        if self._Jmatrix is None:
            raise NotImplementedError
        return self._Jmatrix

    def dpred(self, m=None, f=None):
        r"""Return the predicted (modelled) data for a given model.

        The fields, f, (if provided) will be used for the predicted data
        instead of recalculating the fields (which may be expensive!).

        .. math::

            d_\text{pred} = P(f(m))

        Where P is a projection of the fields onto the data space.

        Parameters
        ----------
        TODO
        m : numpy.ndarray
            Model
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            Fields object

        Returns
        -------
        TODO
        """

        if self.verbose:
            print("Compute predicted")

        if f is None:
            f = self.fields(m=m)

        # Map emg3d-data-array to SimPEG-data-vector
        return self.data2simpeg(f.data.synthetic.data)

    def fields(self, m=None):
        """Return the electric fields for a given model.

        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: f, the fields
        """

        if self.verbose:
            print("Compute fields")

        if m is not None:

            # Store model and update emg3d equivalent.
            self.model = m
            self.simulation.model = self.model2emg3d

            # Clean emg3d-Simulation from old computed data.
            self.simulation.clean('computed')

        # Compute forward model and set initial residuals.
        _ = self.simulation.misfit

        return self.simulation

    # TODO Should we re-define `residual` here, and maybe other stuff?

    def survey2simpeg(self, survey):
        """Return SimPEG survey from provided emg3d survey.


        - A SimPEG survey consists of a list of source-frequency pairs with
          associated receiver lists:

          .. code-block:: console

              [
                [source_1, frequency_1, rec_list],
                [source_2, frequency_1, rec_list],
                ...
                [source_n, frequency_1, rec_list],
                [source_1, frequency_2, rec_list],
                ...
                [source_n, frequency_n, rec_list],
              ]

        Frequencies and receiver lists can be different for different sources.
        Data is not part of the survey, it is handled in a separate data class.

        - An emg3d survey consists of a dictionary each for sources, receivers,
          and frequencies. It contains the corresponding data in an xarray of
          dimension ``nsrc x nrec x nfreq``. The xarray can store any amount of
          data set for the survey. Source-receiver-frequency pair which do not
          exist in the survey are marked with a NaN in the xarray.


        .. note::

            If the survey contains observed data, then only the src-rec-freq
            combinations with non-NaN values are added to the SimPEG survey.


        Parameters
        ----------
        survey : Survey
            emg3d survey instance.


        Returns
        -------
        simpeg_survey : Survey
            SimPEG survey instance.

        """

        # Check if survey contains any non-NaN data.
        data = survey.data.observed
        check = False
        if survey.count:
            check = True
        else:
            raise ValueError

        # Start source and data lists
        src_list = []

        # 1. Loop over sources
        for sname, src in survey.sources.items():

            # If source has no data, skip it.
            sdata = data.loc[sname, :, :]
            if check and not np.any(np.isfinite(sdata.data)):
                continue

            # 2. Loop over frequencies
            for sfreq, freq in survey.frequencies.items():

                # If frequency has no data, skip it.
                fdata = sdata.loc[:, sfreq]
                if check and not np.any(np.isfinite(fdata.data)):
                    continue

                # Start receiver list
                rec_list = []

                # 3. Loop over non-NaN receivers
                for srec, rec in survey.receivers.items():

                    # If receiver has no data, skip it.
                    rdata = fdata.loc[srec].data
                    if check and not np.isfinite(rdata):
                        continue

                    # Add this receiver to receiver list
                    if isinstance(rec, electrodes.RxElectricPoint):
                        rfunc = simpeg_fd.receivers.PointElectricField
                    elif isinstance(rec, electrodes.RxMagneticPoint):
                        rfunc = simpeg_fd.receivers.PointMagneticField
                    else:
                        raise NotImplementedError(
                            f"Receiver type {rec} not implemented."
                        )

                    trec = rfunc(
                        locations=rec.center, component='complex',
                        orientation=electrodes.rotation(
                            rec.azimuth, rec.elevation),
                    )

                    rec_list.append(trec)

                # Add this source-frequency to source list
                if isinstance(src, electrodes.TxElectricWire):
                    tsrc = simpeg_fd.sources.LineCurrent(
                        location=src.points, receiver_list=rec_list,
                        frequency=freq, current=src.strength,
                    )
                elif isinstance(src, electrodes.TxElectricDipole):
                    # Replace once https://github.com/simpeg/simpeg/pull/1525
                    # is released
                    tsrc = ElectricDipole(
                        receiver_list=rec_list, frequency=freq,
                        location=src.center, strength=src.strength,
                        orientation=electrodes.rotation(
                            src.azimuth, src.elevation),
                    )
                else:
                    raise NotImplementedError(
                        f"Source type {src} not implemented."
                    )

                src_list.append(tsrc)

        return simpeg_fd.survey.Survey(src_list)


class Inversion(simpeg.inversion.BaseInversion if simpeg else object):
    """Thin wrapper, adding verbosity and taking care of data format."""

    @utils._requires('simpeg')
    def __init__(
        self, simulation, maxIter, regularization_opts, optimization_opts,
        directiveList, **kwargs
    ):
        """Initialize an Inversion instance."""

        # TODO # Add own directive
        self.simulation = simulation

        reg = simpeg.regularization.WeightedLeastSquares(
            simulation.grid,
            active_cells=simulation.inds_active,
            reference_model=simulation.m0,  # TODO option from kwargs
            **regularization_opts
        )

        # Replace once https://github.com/simpeg/simpeg/pull/1517 is released
        opt = InexactGaussNewton(maxIter=maxIter, **optimization_opts)

        inv_prob = simpeg.inverse_problem.BaseInvProblem(
                simulation.dmis, reg, opt)

        self.save = Directive(simulation)
        directiveList.append(self.save)

        super().__init__(
                invProb=inv_prob, directiveList=directiveList, **kwargs)

    def run(self, m0=None):
        """Run the inversion."""

        # Reset counter, start timer, print message.
        _multiprocessing.process_map.count = 0
        timer = utils.Timer()
        self.simulation.simulation.timer = timer
        print(":: SimPEG(emg3d) START ::")

        # Take start model from Simulation if not provided.
        if m0 is None:
            m0 = self.simulation.m0

        self.save._store(0)

        # Run the inversion
        _ = super().run(m0)

        # Print passed time and exit
        print(f":: SimPEG(emg3d) END   :: runtime = {timer.runtime}")


class Directive(simpeg.directives.InversionDirective if simpeg else object):
    """Print some values for each iteration."""

    def __init__(self, simulation, **kwargs):
        simulation.simulation.invinfo = {}
        self.sim = simulation
        super().__init__(**kwargs)

    @utils._requires('simpeg')
    def endIter(self):

        self._store(self.opt.iter)

        if self.sim.simulation.name:
            io.save(
                f"{self.sim.simulation.name}.h5",
                simulation=self.sim.simulation.to_dict(what='plain'),
                invinfo=self.sim.simulation.invinfo,
                verb=0,
            )

        # Reset counter
        _multiprocessing.process_map.count = 0

    def _store(self, n):
        sim = self.sim.simulation
        sim.survey.data[f"it{n}"] = sim.survey.data.synthetic

        if n > 0:
            sim.invinfo[n] = {
                'model': self.sim.model2emg3d,
                'phi': self.opt.f,
                'phi_d': self.invProb.phi_d,
                'phi_m': self.invProb.phi_m,
                'beta': self.invProb.beta,
                'count': _multiprocessing.process_map.count,
                'time': sim.timer.elapsed,
                # 'chi2': ,
                # 'phi_delta': ,
            }
        else:
            sim.invinfo[n] = {
                'model': sim.model,
                'count': _multiprocessing.process_map.count,
                'time': sim.timer.elapsed,
            }


# ########################################################################### #
# The following are Monkey-Patches for SimPEG PRs:                            #
# - #1525: Add ElectricDipole                                                 #
# - #1524: Ensure misfit is purely real valued                                #
# - #1523: Fix validate_ndarray_with_shape                                    #
# - #1517: Pass `rtol` to SciPy solvers for SciPy>=1.12                       #
# Leave the patches until the connesponding PRs are merged AND released.      #
# ########################################################################### #


# Remove once https://github.com/simpeg/simpeg/pull/1525 is released
class ElectricDipole(simpeg_fd.sources.BaseFDEMSrc):
    """Point electric dipole source."""

    def __init__(
        self,
        receiver_list,
        frequency,
        location=None,
        strength=1.0,
        orientation="x",
        **kwargs,
    ):
        if location is None:
            location = np.r_[0.0, 0.0, 0.0]

        super().__init__(
            receiver_list=receiver_list,
            frequency=frequency,
            location=location,
            **kwargs,
        )

        self.strength = strength
        self.orientation = orientation

    @property
    def location(self):
        """Location of the dipole """
        return self._location

    @location.setter
    def location(self, vec):
        self._location = simpeg.utils.validate_location_property(
                "location", vec, 3)

    @property
    def strength(self):
        """Strength of the electric dipole (:math:`Am`) """
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = simpeg.utils.validate_float(
                "strength", value, min_val=0)

    @property
    def orientation(self):
        """Orientation of the dipole as a normalized vector """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = simpeg.utils.validate_direction(
                "orientation", var, dim=3)


# Remove once https://github.com/simpeg/simpeg/pull/1524 is released
class L2DataMisfit(simpeg.data_misfit.L2DataMisfit):
    r"""Least-squares data misfit."""

    @simpeg.utils.timeIt
    def __call__(self, m, f=None):
        R = self.W * self.residual(m, f=f)
        return np.vdot(R, R).real


# Remove once https://github.com/simpeg/simpeg/pull/1523 is released
class Data(simpeg.data.Data):
    """Data container."""

    @property
    def dobs(self):
        return self._dobs

    @dobs.setter
    def dobs(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self._dobs = simpeg.utils.validate_ndarray_with_shape(
            "dobs", value, shape=(self.survey.nD,), dtype=(float, complex)
        )


# Remove once https://github.com/simpeg/simpeg/pull/1517 is released
class InexactGaussNewton(simpeg.optimization.InexactGaussNewton):
    """Minimizes using CG as the inexact solver of """

    @simpeg.utils.timeIt
    def findSearchDirection(self):
        inp = {"rtol": self.tolCG, "maxiter": self.maxIterCG}
        Hinv = simpeg.optimization.SolverICG(self.H, M=self.approxHinv, **inp)
        p = Hinv * (-self.g)
        return p
