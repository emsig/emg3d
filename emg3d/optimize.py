"""

Inversion
=========

Functionalities related to optimization (inversion), e.g., misfit function,
gradient of the misfit function, or data- and depth-weighting.

Currently it follows the implementation of [PlMu08]_, using the adjoint-state
technique for the gradient.

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
import scipy.linalg as sl

from emg3d import maps

__all__ = ['gradient', 'misfit', 'data_weighting']


def misfit(simulation):
    r"""Return the misfit function.

    The weighted least-squares functional, often called objective function or
    misfit function, as implemented in `emg3d`, is given by Equation 1 in
    [PlMu08]_,

    .. math::
        :label: misfit

            J(\textbf{p}) = \frac{1}{2} \sum_f\sum_s\sum_r
                \left\{
                \left\lVert
                    W_{s,r,f}^e \Delta^e
                \right\rVert^2
                + \left\lVert
                    W_{s,r,f}^h \Delta^h
                \right\rVert^2
                \right\}
            + R(\textbf{p}) \ ,

    where :math:`\Delta^{\{e;h\}}` are the residuals between the observed and
    synthetic data,

    .. math::
        :label: misfit_e

            \Delta^e = \textbf{e}_{s,r,f}[\sigma(\textbf{p})]
                       -\textbf{e}_{s,r,f}^\text{obs} \ ,

    and

    .. math::
        :label: misfit_h

            \Delta^h = \textbf{h}_{s,r,f}[\sigma(\textbf{p})]
                       -\textbf{h}_{s,r,f}^\text{obs} \ .

    Here, :math:`f, s, r` stand for frequency, source, and receiver,
    respectively; :math:`W^{\{e;h\}}` are the weighting functions for the
    electric and magnetic data residual,
    :math:`\{\textbf{e};\textbf{h}\}^\text{obs}` are the observed electric and
    magnetic data, and :math:`\{\textbf{e};\textbf{h}\}` are the synthetic
    electric and magnetic data, computed for a given conductivity
    :math:`\sigma`, which depends on the model parameters :math:`\textbf{p}`.
    Finally, :math:`R(\textbf{p})` is a regularization term.


    .. note::

        This is an early implementation of the misfit function. Currently not
        yet implemented are:

        - Magnetic data;
        - Regularization term.


    Parameters
    ----------
    simulation : :class:`emg3d.simulations.Simulation`
        The simulation.


    Returns
    -------
    misfit : float
        Value of the misfit function.

    """

    # Ensure all fields have been computed.
    test_efield = sum([1 if simulation._dict_efield[src][freq] is None else 0
                       for src, freq in simulation._srcfreq])
    if test_efield:
        simulation.compute()

    # Compute the residual
    residual = simulation.data.synthetic - simulation.data.observed
    simulation.data['residual'] = residual

    # Get weighted residual.
    wresidual = data_weighting(simulation)
    simulation.data['wresidual'] = wresidual

    # Compute misfit
    misfit = (residual.data.conj() * wresidual.data).real.sum()/2

    return misfit


def gradient(simulation):
    r"""Compute the discrete gradient using the adjoint-state method.

    The discrete gradient for a single source at a single frequency is given by
    Equation (10) in [PlMu08]_,

    .. math::

        \nabla_p J(\textbf{p}) =
        -\sum_{k,l,m}\mathbf{\bar{\lambda}}^E_x
               \frac{\partial S}{\partial \textbf{p}} \textbf{E}_x
        -\sum_{k,l,m}\mathbf{\bar{\lambda}}^E_y
               \frac{\partial S}{\partial \textbf{p}} \textbf{E}_y
        -\sum_{k,l,m}\mathbf{\bar{\lambda}}^E_z
               \frac{\partial S}{\partial \textbf{p}} \textbf{E}_z \ ,

    where the grid notation (:math:`\{k, l, m\}` and its :math:`\{+1/2\}`
    equivalents) have been omitted for brevity (except for the sum symbols).


    .. note::

        The gradient is currently implemented only for electric sources and
        receivers and only for isotropic models.


    Parameters
    ----------
    simulation : :class:`emg3d.simulations.Simulation`
        The simulation.


    Returns
    -------
    grad : ndarray
        Adjoint-state gradient (same shape as simulation.model).

    """

    # Check limitation 2: So far only isotropic models.
    if simulation.model.case != 0:
        raise NotImplementedError(
                "Gradient only implemented for isotropic models.")

    # Ensure misfit has been computed (and therefore the electric fields).
    _ = simulation.misfit

    # Compute back-propagating electric fields.
    simulation._bcompute()

    # Pre-allocate the gradient on the mesh.
    grad_model = np.zeros(simulation.grid.vnC, order='F')

    # Loop over source-frequency pairs.
    for src, freq in simulation._srcfreq:

        # Multiply forward field with backward field; take real part.
        efield = -np.real(
                simulation._dict_bfield[src][freq] *
                simulation._dict_efield[src][freq] *
                simulation._dict_efield[src][freq].smu0)

        # Pre-allocate the gradient for the computational grid.
        vnC = simulation._dict_grid[src][freq].vnC
        grad_x = np.zeros(vnC, order='F')
        grad_y = np.zeros(vnC, order='F')
        grad_z = np.zeros(vnC, order='F')

        # => TEST what is faster / more accurate.
        #
        # Here, we do
        #   1. edges2cellaverages (Ex[comp] -> CC[comp])
        #   2. grid2grid          (CC[comp] -> CC[model])
        #
        # How about the other way around?
        #   1. grid2grid          (Ex[comp] -> Ex[model])
        #   1. edges2cellaverages (Ex[model] -> CC[model])

        # Map the field to cell centers times volume.
        vol = simulation._dict_grid[src][freq].vol.reshape(vnC, order='F')
        maps.edges2cellaverages(ex=efield.fx, ey=efield.fy, ez=efield.fz,
                                vol=vol,
                                out_x=grad_x, out_y=grad_y, out_z=grad_z)
        grad = grad_x + grad_y + grad_z

        # Bring the gradient back from the computation grid to the model grid.
        tgrad = maps.grid2grid(
                    simulation._dict_grid[src][freq],
                    -grad, simulation.grid, method='cubic')

        # => Frequency-dependent depth-weighting should go here.

        # Add this src-freq gradient to the total gradient.
        grad_model += tgrad

    # => Frequency-independent depth-weighting should go here.

    # Apply derivative of property-map
    # (in case the property is something else than conductivity).
    simulation.model.map.derivative(grad_model, simulation.model.property_x)

    return grad_model


def data_weighting(simulation):
    r"""Return weighted residual.

    Returns the weighted residual as given in Equation 18 of [PlMu08]_,

    .. math::
        :label: data-weighting

        W(\textbf{x}_s, \textbf{x}_r, \omega) =
        \frac{\lVert\textbf{x}_s-\textbf{x}_r\rVert^{\gamma_d}}
        {\omega^{\beta_f}
         \lVert E^\text{ref}(\textbf{x}_s, \textbf{x}_r, \omega)
         \rVert^{\beta_d}}\ .


    Parameters
    ----------
    simulation : :class:`emg3d.simulations.Simulation`
        The simulation. The parameters for data weighting are set in the
        call to `Simulation` through the parameter `data_weight_opts`.


    Returns
    -------
    wresidual : DataArray
        The weighted residual (:math:`W^e \Delta^e`).

    """
    # Get relevant parameters.
    gamma_d = simulation.data_weight_opts.get('gamma_d', 0.5)
    beta_d = simulation.data_weight_opts.get('beta_d', 1.0)
    beta_f = simulation.data_weight_opts.get('beta_f', 0.25)
    min_off = simulation.data_weight_opts.get('min_off', 1000.0)
    noise_floor = simulation.data_weight_opts.get('noise_floor', 1e-15)
    refname = simulation.data_weight_opts.get('reference', 'reference')

    # (A) MUTING.

    # Get all small amplitudes.
    mute = np.abs(simulation.data.observed.data) < noise_floor

    # Add all near offsets.
    offsets = sl.norm(
                np.array(simulation.survey.rec_coords[:3])[:, None, :] -
                np.array(simulation.survey.src_coords[:3])[:, :, None],
                axis=0,
                check_finite=False,
            )
    mute += (offsets < min_off)[:, :, None]

    # (B) WEIGHTS.

    # (B.1) First term: Offset weighting (f-indep.).
    off_weight = offsets**gamma_d

    # (B.2) Second term: Frequency weighting (src-freq-indep.).
    omega = 2*np.pi*simulation.survey.frequencies
    data_weight = off_weight[:, :, None]/(omega**beta_f)[None, None, :]

    # (B.3) Third term: Amplitude weighting.
    if beta_d != 0.0:  # Because of the warn-print check if required.
        ref_data = simulation.data.get(refname, None)
        if ref_data is None:
            if simulation.verb >= 0:
                print(f"Reference data '{refname}' not found, "
                      "using 'synthetic'.")
            ref_data = simulation.data.synthetic
        data_weight /= np.sqrt(np.real(ref_data.conj()*ref_data))**beta_d

    # (C) APPLY.
    wresidual = simulation.data.residual * data_weight
    wresidual.data[mute] = 0.0

    return wresidual
