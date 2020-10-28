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

from emg3d import maps

__all__ = ['gradient', 'misfit']


def misfit(simulation):
    r"""Return the misfit function.

    The data misfit or weighted least-squares functional using an :math:`l_2`
    norm is given by

    .. math::
        :label: misfit

            \phi = \frac{1}{2} \sum_f\sum_s\sum_r
                \left\{
                \left\lVert
                    W_{s,r,f} \left(
                       \textbf{d}_{s,r,f}^\text{pred}
                       -\textbf{d}_{s,r,f}^\text{obs}
                    \right) \right\rVert^2
                \right\}
            + R \ .

    Here, :math:`f, s, r` stand for frequency, source, and receiver,
    respectively; :math:`\textbf{d}^\text{obs}` are the observed electric and
    magnetic data, and :math:`\textbf{d}^\text{pred}` are the synthetic
    electric and magnetic data. Finally, :math:`R` is a regularization term.

    The data weight of observation :math:`d_i` is given by :math:`W_i =
    \varsigma^{-1}_i`, where :math:`\varsigma_i` is the standard deviation of
    the observation (see :attr:`emg3d.surveys.Survey.standard_deviation`).

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
    std = simulation.survey.standard_deviation
    # Raise warning if not set-up properly.
    if std is None:
        raise ValueError(
            "Either `noise_floor` or `relative_error` or both must\n"
            "be provided (>0) to compute the `standard_deviation`.\n"
            "It can also be set directly (same shape as data).\n"
            "The standard deviation is required to compute the misfit.")

    # Ensure all fields have been computed.
    test_efield = sum([1 if simulation._dict_efield[src][freq] is None else 0
                       for src, freq in simulation._srcfreq])
    if test_efield:
        simulation.compute()

    # Compute the residual
    residual = simulation.data.synthetic - simulation.data.observed
    simulation.data['residual'] = residual

    # Get weighted residual.
    if 'weights' not in simulation.data.keys():
        simulation.data['weights'] = 1/std**2
    weights = simulation.data['weights']

    # Compute misfit
    misfit = np.sum(weights*(residual.data.conj()*residual.data).real)/2

    return misfit.data


def gradient(simulation):
    r"""Compute the discrete gradient using the adjoint-state method.

    The discrete gradient for a single source at a single frequency is given by
    Equation (10) in [PlMu08]_,

    .. math::

        \nabla_p \phi(\textbf{p}) =
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
        receivers; only for isotropic models; and not for electric permittivity
        nor magnetic permeability.


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

    # Apply derivative-chain of property-map
    # (in case the property is something else than conductivity).
    simulation.model.map.derivative_chain(
            grad_model, simulation.model.property_x)

    return grad_model
