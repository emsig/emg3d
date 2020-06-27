"""

:mod:`gradient` -- Compute adjoint-state method gradient
========================================================

Functions to compute the gradient of the misfit function using the
adjoint-state method, see [PlMu08]_.

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
import numpy as np
from concurrent import futures
from scipy import signal, interpolate

from emg3d import maps, fields


# TODO - here or in simulation? TODO
def resfield_parallel(data, grids, efields, wdata, nproc=4):
    """Call resfield concurrent over source positions and frequencies.

    Computes the residual field and the data misfit in parallel. See
    :func:`resfield` for more info.


    Parameters
    ----------
    data : xarray
        CSEM data.

    grids : :class:`emg3d.utils.TensorMesh` or dict
        If `TensorMesh`-instance, the same grid is used for all sources
        and frequencies. Else a dict can be provided.

    efields : dict
        Containing the electric-fields for all sources and frequencies.

    wdata : dict or None
        Parameter-dict for data weighting.

    nproc : int
        Maximum number of processes that can be used. Default is four.


    Returns
    -------
    ResidualField : :class:`emg3d.utils.SourceField`-instance
        Total residual field for all sources and frequencies.

    data_misfit : float
        The total, summed data misfit for all sources and frequencies.

    """
    # Get data weighting instance if wdata is not None.
    W = wdata if wdata is None else DataWeighting(**wdata)

    # Initiate dictionary to store output from concurrent runs.
    out = {}

    # Context manager for futures.
    with futures.ProcessPoolExecutor(nproc) as executor:

        # Loop over source positions.
        for src in data.src_name.values:

            # This source
            src_coord = (data.loc[src, :, :, :].srcx.values,
                         data.loc[src, :, :, :].srcy.values,
                         data.loc[src, :, :, :].srcz.values)

            # Sub-dictionary for this source.
            out[src] = {}

            # Loop over frequencies.
            for freq in data.frequency.values:

                # There seems to be an issue with xarray/NetCDF/HDF5 and
                # concurrent.futures.
                cdat = data.loc[src, :, :, freq]
                rec = (cdat.recx.values, cdat.recy.values, cdat.recz.values)
                inpdat = {}
                for comp in cdat.component.values:
                    inpdat[comp] = cdat.loc[:, comp].values

                # Spawn threads.
                out[src][freq] = executor.submit(
                    resfield,
                    data=inpdat,
                    grid=grids[src][freq],
                    rec=rec,
                    src=src_coord,
                    freq=freq,
                    field=efields[src][freq],
                    W=W,
                )

    # Get residual fields from concurrent output.
    resfields = {f: {k: v.result()[0] for k, v in s.items()}
                 for f, s in out.items()}

    # Sum-up data misfits from concurrent output. [PlMu08]_, Equation (1).
    data_misfit = np.sum([v2.result()[1] for v1 in out.values()
                          for v2 in v1.values()])

    return resfields, data_misfit


def resfield(data, grid, rec, src, freq, field, W):
    """Return residual field for this src-freq-pair and all receivers.

    The residual at each receiver location constitutes a source, which are
    summed up for all receivers to generate the residual source field for the
    back-propagation in the adjoint-state method.

    .. todo::

        - Currently, the source is hardwired to the 'Ex'-component; generalize!


    Parameters
    ----------
    data : dict
        The keys are the components ('ex', 'ey', ...), and the values are
        ndarrays containing the complex responses for all receivers.

    grid : :class:`emg3d.utils.TensorMesh`-instance
        The computational grid for this source and frequency.

    rec : tuple
        (rec-x, rec-y, rec-z)

    src : tuple
        (src-x, src-y, src-z)

    freq : float
        Frequency (Hz).

    field : :class:`emg3d.utils.Field`-instance
        The electric field (on grid) for this source and frequency.

    W : :class:`DataWeight` instance or None
        Data weighting.


    Returns
    -------
    ResidualField : :class:`emg3d.utils.SourceField`-instance
        The residual field for this source and this frequency.

    data_misfit : float
        The summed data misfit for this source and this frequency.

    """
    # Extract the field at the receivers.
    respx, respy, respz = fields.get_receiver(grid, field, rec)
    data_syn = {'ex': respx, 'ey': respy, 'ez': respz}

    # Subtract the observed data => residual.
    data_res = {}
    data_wres = {}
    for comp in data.keys():
        data_res[comp] = data_syn[comp] - data[comp]

        data_wres[comp] = data_syn[comp] - data[comp]
        if W is not None:
            data_wres[comp] *= W.weights(data[comp], rec, src, freq)

    # Initiate empty field
    ResidualField = fields.SourceField(grid, freq=freq)

    # Loop over receivers, input as source.
    for i in range(rec[0].size):

        # Strength: in get_source_field the strength is multiplied with iwmu;
        # so we undo this here.
        # TODO Ey, Ez
        strength = data_wres['ex'][i].conj()/ResidualField.smu0

        ThisSField = fields.get_source_field(
            grid=grid,
            src=[rec[0][i], rec[1][i], rec[2][i], 0, 0],  # TODO azimuth, dip
            freq=freq,
            strength=strength,
        )

        # If strength is zero (very unlikely), get_source_field would return a
        # normalized field for a unit source. However, in this case we do not
        # want that.
        if strength != 0:
            ResidualField += ThisSField

    # Get compute data misfit.
    data_misfit = 0.0
    for comp in data.keys():
        d_res = data_res[comp]
        d_wres = data_wres[comp]
        data_misfit += np.abs(np.sum(d_res.conj()*d_wres))/2

    # Return.
    return ResidualField, data_misfit


def gradient(ffields, bfields, grids, mesh, wdepth):
    r"""Compute the discrete gradient using the adjoint-state method.

    The discrete gradient for a single source at a single frequency is given by
    Equation (10) of [PlMu08]_:

    .. math::

        -\sum_{k,l,m}\mathbf{\bar{\lambda}}^E_x
               \frac{\partial S}{\partial \textbf{p}} \textbf{E}_x
        -\sum_{k,l,m}\mathbf{\bar{\lambda}}^E_y
               \frac{\partial S}{\partial \textbf{p}} \textbf{E}_y
        -\sum_{k,l,m}\mathbf{\bar{\lambda}}^E_z
               \frac{\partial S}{\partial \textbf{p}} \textbf{E}_z  ,

    where the grid notation (:math:`\{k, l, m\}` and its :math:`\{+1/2\}`
    equivalents) have been ommitted for brevity, except in the :math:`\sum`
    symbols.


    Parameters
    ----------
    ffields, bfields : dict
        Dictionary (over sources and frequencies) containing the forward and
        backward electric fields.

    grids : dict
        Dictionary containing the grids corresponding to the provided fields.

    mesh : TensorMesh
        Model grid, on which the gradient is computed; a
        ``emg3d.utils.TensorMesh`` instance.

    wdepth : dict or None
        Parameter-dict for depth weighting.


    Returns
    -------
    grad : ndarray
        Current gradient; has shape mesh.vnC (same as model properties).

    """
    # Get depth weighting (preconditioner) instance if wdepth is not None.
    D = wdepth if wdepth is None else DepthWeighting(mesh, **wdepth)

    # Pre-allocate the gradient on the mesh.
    grad_model = np.zeros(mesh.vnC, order='F')

    # Initiate the preconditioner-dict.
    precond_dict = {}

    # Loop over sources.
    for src in ffields.keys():

        # Loop over frequencies.
        for freq in ffields[src].keys():

            # Get depth weights.
            if D is not None:
                if freq in precond_dict.keys():
                    precond = precond_dict[freq]
                else:
                    precond = D.weights(freq).reshape(mesh.vnC, order='F')
                    precond_dict[freq] = precond

            # Multiply forward field with backward field; take real part.
            efield = -np.real(bfields[src][freq] * ffields[src][freq] *
                              ffields[src][freq].smu0)

            # Pre-allocate the gradient for the computational grid.
            grad = np.zeros(grids[src][freq].vnC, order='F')

            # Map the field to cell centers times volume.
            maps.avg_field2cell_volume(
                    grad, grids[src][freq].vol.reshape(
                        grids[src][freq].vnC, order='F'),
                    efield.fx, efield.fy, efield.fz)

            # Bring the gradient back from the computation grid to the model
            # grid.
            tgrad = maps.grid2grid(
                        grids[src][freq], grad, mesh, method='cubic')

            # Add this src-freq gradient to the total gradient.
            if D is not None:
                grad_model += precond*tgrad
            else:
                grad_model += tgrad

    return grad_model


# WEIGHTING CLASSES - TODO - these have to go to `optimize/weighting.py` TODO
class DataWeighting:
    r"""Data Weighting; Plessix and Mulder, equation 18.

    .. math::

        W(x_s, x_r, \omega) =
        \frac{||x_s-x_r||^{\gamma_d}}
        {\omega^{\beta_f}||E_1^0(x_s, x_r, \omega)||^{\beta_d}}


    .. todo::

        - Currently, low amplitudes are switched-off, because it is divided by
          data. Test if small offsets should also be switched off.
        - Include other data weighting functions.


    Parameters
    ----------
    gamma_d : float
        Offset weighting exponent.

    beta_d : float
        Data weighting exponent.

    beta_f : float
        Frequency weighting exponent.

    """

    def __init__(self, gamma_d=0.5, beta_d=1.0, beta_f=0.25):
        """Initialize new DataWeighting instance."""

        # Store values
        self.gamma_d = gamma_d
        self.beta_d = beta_d
        self.beta_f = beta_f

    def weights(self, data, rec, src, freq, min_amp=1e-25):
        """[PlMu08]_, equation 18.


        Parameters
        ----------
        data : ndarray
            CSEM data.

        rec, src : tupples
            Receiver and source coordinates.

        freq : float
            Frequency (Hz)

        min_amp : float
            Data with amplitudes below min_amp get zero data weighting (data
            weighting divides by the amplitude, which is a problem if the
            amplitude is zero).


        Returns
        -------
        data_weight : ndarray
            Data weights (size of number of receivers).

        """

        # Mute
        mute = np.abs(data) < 1e-15

        # Get offsets.
        locs = np.stack([rec[0]-src[0], rec[1]-src[1], rec[2]-src[2]])
        offsets = np.linalg.norm(locs, axis=0)

        # Compute the data weight.
        data_weight = offsets**self.gamma_d / (2*np.pi*freq)**self.beta_f
        data_weight /= np.sqrt(np.real(data*data.conj()))**self.beta_d

        # Set small amplitudes to zero.
        data_weight[mute] = 0

        return data_weight


class DepthWeighting:
    r"""Depth Weighting; Plessix and Mulder, equation 17.

    Very crude, needs improvement.

    .. math::

        D = \left[ z^{-\gamma_m} \exp\left(-\beta_m \frac{z-z_b}{\delta}
            \right)\right] \qquad (text{for} \quad z > z_b)

    Adds a Tukey-window style taper at the bottom.

    .. todo::

        - Include other depth weighting functions.
        - "Make it 3D":

            - `depths` should depend on x and y. At the moment, they don't.
            - `z_ref` should depend on x and y. At the moment, it doesn't.


    Parameters
    ----------
    grid : :class:`emg3d.utils.TensorMesh`-instance
        The computational grid of the resistivity model.

    z_ref : float
        Reference depth; normally sea-bottom in the marine case or surface
        in the land case.

    res_ref : float
        Reference resistivity for skin depth.

    gamma_m : float
        Linear model weighting exponent.

    beta_m : float
        Exponential model weighting exponent.

    epsilon_r : float
        Regularization term.

    taper : list of two floats
        Tukey-taper: [n_skind-start, n_skind-end]
        Default is [2, 3]

    """

    def __init__(self, grid, z_ref, res_ref, gamma_m=0.0, beta_m=1.0,
                 epsilon_r=1.0, taper=None):
        """Initialize new DepthWeighting instance."""

        # Store values
        self.z_ref = z_ref
        self.res_ref = res_ref
        self.gamma_m = gamma_m
        self.beta_m = beta_m
        self.epsilon_r = epsilon_r
        self.vnC = grid.vnC
        self.depths = grid.vectorCCz

        # Default: Taper off from 2*skind to 3*skind
        if taper is None:
            self.taper = [2, 3]
        else:
            self.taper = taper

    def weights(self, freq):
        """[PlMu08]_, equation 17.

        Depth weighting, used as a preconditioner, with a Tukey-style taper at
        the bottom; frequency-dependent.

        Parameters
        ----------
        freq : float
            Frequency (Hz)

        Returns
        -------
        depth_weight : ndarray
            Depth weights (size of number of receivers).

        """

        # Take skin-depth as reference depth.
        skind = 503.3*np.sqrt(self.res_ref/freq)

        # Depth weighting, Plessix and Mulder, equation 17.
        # There is a typo in the equation.
        z = -(self.depths-self.z_ref)/skind

        # Depth dependency.
        precond = 1/(z**-self.gamma_m * np.exp(-self.beta_m*z) +
                     self.epsilon_r)

        # Reset to zero everything above seafloor.
        precond[self.depths >= self.z_ref] = 0

        # Use a Tukey-window to fade-out in depth.
        # Let [ne, ns] = taper, then the Tukey window goes
        # from z_ref-ne*sd to z_ref-ns*sd.
        # - alpha is defined by taper:
        alpha = np.diff(self.taper)/self.taper[1]
        # - create a 101 pt filter:
        win = signal.tukey(101, alpha=alpha)
        # - we are only interested in the half below z_ref:
        win[50:] = 1
        # - and map it to the actual data:
        end = self.z_ref - self.taper[1]*skind
        x = end + np.arange(101)*self.taper[1]*skind/50
        f = interpolate.interp1d(x, win, kind='cubic',
                                 bounds_error=False, fill_value=(0.0, 1.0))
        # - apply it:
        precond *= f(self.depths)

        # Normalize and return.
        precond /= precond.max()
        return np.repeat(precond, self.vnC[0]*self.vnC[1])
