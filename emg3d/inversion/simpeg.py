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

from emg3d import electrodes, meshes, models, surveys, utils

try:
    import simpeg
    import discretize
    from simpeg.electromagnetics import frequency_domain as simpeg_fd
    # Add simpeg to the emg3d.Report().
    utils.OPTIONAL.extend(['simpeg',])
except ImportError:
    simpeg = None


__all__ = ['Kernel', ]


def __dir__():
    return __all__


class Kernel(
        simpeg_fd.simulation.BaseFDEMSimulation if simpeg else object):
    """3D simulation of electromagnetic fields using emg3d as a solver.

    .. note::

        Currently only isotropic models are implemented, with unit relative
        electric permittivity and unit relative magnetic permeability.


    Parameters
    ----------
    simulation_opts : dict
        Input parameters forward to ``emg3d.Simulation``. See the emg3d
        documentation for all the possibilities.

        By default, `gridding='same'`, which is different from the default in
        emg3d. However, any `gridding` and `gridding_opts` can be provided. In
        that case one can also provide a `model`, which is used as the
        reference model for the automatic gridding routine.

    """

    @utils._requires("simpeg")
    def __init__(self, simulation, **kwargs):
        """Initialize Simulation using emg3d as solver."""

        # Store simulation
        self.simulation = simulation

        # Create SimPEG survey.
        survey = survey2simpeg(simulation.survey)

        # Get the data map
        # TODO - can we simplify survey2emg3d?
        _, dmap = survey2emg3d(survey)

        # Add reverse map to emg3d-data (is saved with survey).
        ind = np.full(simulation.survey.shape, -1)
        ind[dmap] = np.arange(survey.nD)
        esurvey = simulation.survey
        esurvey.data['indices'] = esurvey.data.observed.copy(data=ind)

        # Store emg3d-survey and data map.
        self._emg3d_survey = esurvey
        self._dmap_simpeg_emg3d = dmap

        # Create emg3d data dummy; can be re-used.
        self._emg3d_array = np.full(esurvey.shape, np.nan+1j*np.nan)

        super().__init__(mesh=simulation.model.grid, survey=survey, **kwargs)

    @property
    def emg3d_model(self):
        """emg3d conductivity model; obtained from SimPEG conductivities."""
        # TODO improve
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
        return j_vec[self._dmap_simpeg_emg3d]

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

            # Put v onto emg3d data-array.
            self._emg3d_array[self._dmap_simpeg_emg3d] = v

            J = self.getJ(m, f=f)
            Jtv = discretize.utils.mkvc(np.dot(J.T, self._emg3d_array))

            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m=m)

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        Full J matrix can be computed by setting v=None (not implemented yet).
        """

        if v is not None:
            # Put v onto emg3d data-array.
            self._emg3d_array[self._dmap_simpeg_emg3d] = v

            # Get gradient with `v` as residual.
            jt_sigma_vec = f.jtvec(self._emg3d_array)

            return self.sigmaDeriv.T @ jt_sigma_vec.ravel('F')

        else:

            raise NotImplementedError

    def getJ(self, m, f=None):
        """Generate full sensitivity matrix."""

        if self._Jmatrix is not None:
            return self._Jmatrix

        else:
            if self.verbose:
                print("Calculating J and storing")
            self.model = m
            if f is None:
                f = self.fields(m)
            self._Jmatrix = (self._Jtvec(m, v=None, f=f)).T

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
        data_complex = simpeg.data.Data(
            survey=self.survey,
            dobs=f.data.synthetic.data[self._dmap_simpeg_emg3d]
        )
        data = []
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                # Hear would come Re/Im [used now] or Amp/Pha or
                # log(Re)/Log(im) etc
                data.append(data_complex[src, rx])
        return np.hstack(data)

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
            self.simulation.model = self.emg3d_model

            # Clean emg3d-Simulation from old computed data.
            self.simulation.clean('computed')

        # Compute forward model and set initial residuals.
        _ = self.simulation.misfit

        return self.simulation


@utils._requires("simpeg")
def survey2emg3d(survey):
    """Return emg3d survey from provided SimPEG survey.


    - A SimPEG survey consists of a list of source-frequency pairs with
      associated receiver lists:

          [[source_1, frequency, rec_list],
           [source_2, frequency, rec_list],
           ...
          ]

      Frequencies and receiver lists can be different for different sources.
      Data is not part of the survey, it is handled in a separate data class.

    - An emg3d survey consists of a dictionary each for sources, receivers, and
      frequencies. It contains the corresponding data in an xarray of dimension
      ``nsrc x nrec x nfreq``. The xarray can store any amount of data set for
      the survey. Source-receiver-frequency pair which do not exist in the
      survey are marked with a NaN in the xarray.


    See Also
    --------
    :func:`survey2simpeg` : Opposite way, from emg3d to SimPEG.


    Parameters
    ----------
    survey : Survey
        SimPEG survey instance.


    Returns
    -------
    emg3d_survey : Survey
        emg3d survey instance, containing the data set `indices`.

    data_map : tuple
        Indices to map SimPEG-data to emg3d data and vice-versa.

        To put SimPEG data array on, e.g., the emg3d synthetic xarray:

           emg3d_survey.data.synthetic.data[dmap] = simpeg_array

        To obtain SimPEG data array from, e.g., the emg3d synthetic xarray:

           simpeg_array = emg3d_survey.data.synthetic.data[dmap]

    """

    # Allocate lists to create data to/from dicts.
    src_list = []
    freq_list = []
    rec_list = []
    data_dict = {}
    rec_uid = {}
    indices = np.zeros((survey.nD, 3), dtype=int)

    # Counter for SimPEG data object (lists the data continuously).
    ind = 0

    # Loop over sources.
    for src in survey.source_list:

        # Create emg3d source.
        if isinstance(src, simpeg_fd.sources.LineCurrent):
            source = electrodes.TxElectricWire(
                src.location,
                strength=src.current
            )
        elif isinstance(src, simpeg_fd.sources.ElectricDipole):
            azimuth, elevation = _vector2angles(src.orientation)
            source = electrodes.TxElectricDipole(
                (*np.squeeze(src.location), azimuth, elevation),
                strength=src.strength, length=1.0
            )
        else:
            raise NotImplementedError(f"Source type {src} not implemented")

        # New frequency: add.
        if src.frequency not in freq_list:
            f_ind = len(freq_list)
            freq_list.append(src.frequency)

        # Existing source: get index.
        else:
            f_ind = freq_list.index(src.frequency)

        # New source: add.
        if source not in src_list:
            s_ind = len(src_list)
            data_dict[s_ind] = {f_ind: {}}
            src_list.append(source)

        # Existing source: get index.
        else:
            s_ind = src_list.index(source)

            # If new frequency for existing source, add:
            if f_ind not in data_dict[s_ind].keys():
                data_dict[s_ind][f_ind] = {}

        # Loop over receiver lists.
        rec_types = [electrodes.RxElectricPoint, electrodes.RxMagneticPoint]
        for rec in src.receiver_list:

            # If this SimPEG receiver was already processed, store it.
            if rec._uid in rec_uid.keys():
                li = len(rec_uid[rec._uid])
                indices[ind:ind+li, 0] = s_ind
                indices[ind:ind+li, 1] = rec_uid[rec._uid]
                indices[ind:ind+li, 2] = f_ind
                ind += li
                continue
            else:
                rec_uid[rec._uid] = []

            if rec.projField not in ['e', 'h']:
                raise NotImplementedError(
                    "Only projField = {'e'; 'h'} implemented."
                )

            # Get type, component.
            rec_type = rec_types[rec.projField == 'h']
            component = rec.component

            # Loop over receivers.
            for i in range(rec.locations[:, 0].size):

                # Create emg3d receiver.
                azimuth, elevation = _vector2angles(rec.orientation)
                receiver = rec_type(
                    (*rec.locations[i, :], azimuth, elevation),
                    data_type=component,
                )
                # New receiver: add.
                if receiver not in rec_list:
                    r_ind = len(rec_list)
                    data_dict[s_ind][f_ind][r_ind] = ind
                    rec_list.append(receiver)

                # Existing receiver: get index.
                else:
                    r_ind = rec_list.index(receiver)

                    # If new receiver for existing src-freq, add:
                    existing = data_dict[s_ind][f_ind].keys()
                    if r_ind not in existing:
                        data_dict[s_ind][f_ind][r_ind] = ind

                    # Else, throw an error.
                    else:
                        raise ValueError(
                            "Duplicate source-receiver-frequency."
                        )

                # Store receiver index, in case the entire receiver
                # is used several times.
                rec_uid[rec._uid].append(r_ind)

                # Store the SimPEG<->emg3d mapping for this receiver
                indices[ind, :] = [s_ind, r_ind, f_ind]
                ind += 1

    # Create and store survey.
    emg3d_survey = surveys.Survey(
        name='Survey created by SimPEG',
        sources=surveys.txrx_lists_to_dict(src_list),
        receivers=surveys.txrx_lists_to_dict(rec_list),
        frequencies=freq_list,
        noise_floor=1.,       # We deal with std in SimPEG.
        relative_error=None,  # "    "   "
    )

    # Store data-mapping SimPEG <-> emg3d
    data_map = tuple(indices.T)

    # Add reverse map to emg3d-data (is saved with survey).
    ind = np.full(emg3d_survey.shape, -1)
    ind[data_map] = np.arange(survey.nD)
    emg3d_survey.data['indices'] = emg3d_survey.data.observed.copy(data=ind)

    return emg3d_survey, data_map


@utils._requires("simpeg")
def survey2simpeg(survey):
    """Return SimPEG survey from provided emg3d survey.


    - A SimPEG survey consists of a list of source-frequency pairs with
      associated receiver lists:

          [[source_1, frequency, rec_list],
           [source_2, frequency, rec_list],
           ...
          ]

      Frequencies and receiver lists can be different for different sources.
      Data is not part of the survey, it is handled in a separate data class.

    - An emg3d survey consists of a dictionary each for sources, receivers, and
      frequencies. It contains the corresponding data in an xarray of dimension
      ``nsrc x nrec x nfreq``. The xarray can store any amount of data set for
      the survey. Source-receiver-frequency pair which do not exist in the
      survey are marked with a NaN in the xarray.


    .. note::

        If the survey contains observed data, then only the src-rec-freq
        combinations with non-NaN values are added to the SimPEG survey.


    See Also
    --------
    :func:`survey2emg3d` : Opposite way, from SimPEG to emg3d.


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
    if survey.isfinite.sum():
        check = True

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
                    orientation=_angles2vector(rec.azimuth, rec.elevation),
                )

                rec_list.append(trec)

            # Add this source-frequency to source list
            if isinstance(src, electrodes.TxElectricWire):
                tsrc = simpeg_fd.sources.LineCurrent(
                    location=src.points, receiver_list=rec_list,
                    frequency=freq, current=src.strength,
                )
            elif isinstance(src, electrodes.TxElectricDipole):
                tsrc = simpeg_fd.sources.ElectricDipole(
                    receiver_list=rec_list, frequency=freq,
                    location=src.center, strength=src.strength,
                    orientation=_angles2vector(src.azimuth, src.elevation),
                )
            else:
                raise NotImplementedError(
                    f"Source type {src} not implemented."
                )

            src_list.append(tsrc)

    return simpeg_fd.survey.Survey(src_list)


def _angles2vector(azimuth, elevation):
    """Convert azimuth and elevation to a SimPEG-orientation vector."""
    return electrodes.rotation(azimuth, elevation, deg=True)


def _vector2angles(orientation):
    """Convert a SimPEG-orientation vector to azimuth and elevation."""
    if isinstance(orientation, str):
        azimuth = 0.0
        elevation = 0.0
        if orientation == "y":
            azimuth = 90.0
        elif orientation == "z":
            elevation = 90.0
    else:
        x, y, z = orientation
        azimuth = np.angle(complex(x, y), deg=True)
        elevation = np.angle(complex(np.linalg.norm([x, y]), z), deg=True)

    return azimuth, elevation
