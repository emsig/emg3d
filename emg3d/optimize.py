"""
Functionalities related to optimization (minimization, inversion), such as the
misfit function and its gradient.
"""
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

import numpy as np

from emg3d import maps, fields

__all__ = ['misfit', 'gradient']


def misfit(simulation):
    r"""Misfit or cost function.

    The data misfit or weighted least-squares functional using an :math:`l_2`
    norm is given by

    .. math::
        :label: misfit

            \phi = \frac{1}{2} \sum_s\sum_r\sum_f
                \left\lVert
                    W_{s,r,f} \left(
                       \textbf{d}_{s,r,f}^\text{pred}
                       -\textbf{d}_{s,r,f}^\text{obs}
                    \right) \right\rVert^2 \, ,

    where :math:`s, r, f` stand for source, receiver, and frequency,
    respectively; :math:`\textbf{d}^\text{obs}` are the observed electric and
    magnetic data, and :math:`\textbf{d}^\text{pred}` are the synthetic
    electric and magnetic data. As of now the misfit does not include any
    regularization term.

    The data weight of observation :math:`d_i` is given by :math:`W_i =
    \varsigma^{-1}_i`, where :math:`\varsigma_i` is the standard deviation of
    the observation, see :attr:`emg3d.surveys.Survey.standard_deviation`.

    .. note::

        You can easily implement your own misfit function (to include, e.g., a
        regularization term) by monkey patching this misfit function with your
        own::

            def my_misfit_function(simulation):
                '''Returns the misfit as a float.'''

                # Computing the misfit...

                return misfit

            # Monkey patch optimize.misfit:
            emg3d.optimize.misfit = my_misfit_function

            # And now all the regular stuff, initiate a Simulation etc
            simulation = emg3d.Simulation(survey, grid, model)
            simulation.misfit
            # => will return your misfit
            #   (will also be used for the adjoint-state gradient).


    Parameters
    ----------
    simulation : Simulation
        The simulation; a :class:`emg3d.simulations.Simulation` instance.


    Returns
    -------
    misfit : float
        Value of the misfit function.

    """

    # Check if electric fields have already been computed.
    test_efield = sum([1 if simulation._dict_efield[src][freq] is None else 0
                       for src, freq in simulation._srcfreq])
    if test_efield:
        simulation.compute()

    # Check if weights are stored already.
    # (weights are currently simply 1/std^2; but might change in the future).
    if 'weights' not in simulation.data.keys():

        # Get standard deviation, raise warning if not set.
        std = simulation.survey.standard_deviation
        if std is None:
            raise ValueError(
                "Either `noise_floor` or `relative_error` or both must "
                "be provided (>0) to compute the `standard_deviation`. "
                "It can also be set directly (same shape as data). "
                "The standard deviation is required to compute the misfit."
            )

        # Store weights
        simulation.data['weights'] = std**-2

    # Calculate and store residual.
    residual = simulation.data.synthetic - simulation.data.observed
    simulation.data['residual'] = residual

    # Get weights, calculate misfit.
    weights = simulation.data['weights']
    misfit = np.sum(weights*(residual.conj()*residual)).real/2

    return misfit.data


def gradient(simulation):
    r"""Compute the discrete gradient using the adjoint-state method.

    The discrete adjoint-state gradient for a single source at a single
    frequency is given by Equation (10) in [PlMu08]_,

    .. math::

        \nabla_p \phi(\textbf{p}) =
            -&\sum_{k,l,m}\mathbf{\bar{\lambda}}_{x; k+\frac{1}{2}, l, m}
                \frac{\partial S_{k+\frac{1}{2}, l, m}}{\partial \textbf{p}}
                 \textbf{E}_{x; k+\frac{1}{2}, l, m}\\
            -&\sum_{k,l,m}\mathbf{\bar{\lambda}}_{y; k, l+\frac{1}{2}, m}
                \frac{\partial S_{k, l+\frac{1}{2}, m}}{\partial \textbf{p}}
                 \textbf{E}_{y; k, l+\frac{1}{2}, m}\\
            -&\sum_{k,l,m}\mathbf{\bar{\lambda}}_{z; k, l, m+\frac{1}{2}}
                \frac{\partial S_{k, l, m+\frac{1}{2}}}{\partial \textbf{p}}
                 \textbf{E}_{z; k, l, m+\frac{1}{2}}\, ,



    where :math:`\textbf{E}` is the electric (forward) field and
    :math:`\mathbf{\lambda}` is the back-propagated residual field (from
    electric and magnetic receivers); :math:`\bar{~}` denotes conjugate.
    The :math:`\partial S`-part takes care of the volume-averaged model
    parameters.


    .. note::

        The currently implemented gradient is only for isotropic models without
        relative electric permittivity nor relative magnetic permeability.


    Parameters
    ----------
    simulation : Simulation
        The simulation; a :class:`emg3d.simulations.Simulation` instance.


    Returns
    -------
    grad : ndarray
        Adjoint-state gradient (same shape as ``simulation.model``).

    """

    # Check limitation 1: So far only isotropic models.
    if simulation.model.case != 'isotropic':
        raise NotImplementedError(
            "Gradient only implemented for isotropic models."
        )

    # Check limitation 2: No epsilon_r, mu_r.
    var = (simulation.model.epsilon_r, simulation.model.mu_r)
    for v, n in zip(var, ('el. permittivity', 'magn. permeability')):
        if v is not None and not np.allclose(v, 1.0):
            raise NotImplementedError(f"Gradient not implemented for {n}.")

    # Ensure misfit has been computed (and therefore the electric fields).
    _ = simulation.misfit

    # Compute back-propagating electric fields.
    simulation._bcompute()

    # Pre-allocate the gradient on the mesh.
    gradient_model = np.zeros(simulation.model.grid.shape_cells, order='F')

    # Loop over source-frequency pairs.
    for src, freq in simulation._srcfreq:

        # Multiply forward field with backward field; take real part.
        # This is the actual Equation (10), with:
        #   del S / del p = iwu0 V sigma / sigma,
        # where lambda and E are already volume averaged.
        efield = simulation._dict_efield[src][freq]  # Forward electric field
        bfield = simulation._dict_bfield[src][freq]  # Conj. backprop. field
        gfield = fields.Field(
            grid=efield.grid,
            data=-np.real(bfield.field * efield.smu0 * efield.field),
            dtype=float,
        )

        # Pre-allocate the gradient for the computational grid.
        shape = gfield.grid.shape_cells
        grad_x = np.zeros(shape, order='F')
        grad_y = np.zeros(shape, order='F')
        grad_z = np.zeros(shape, order='F')

        # Map the field to cell centers times volume.
        cell_volumes = gfield.grid.cell_volumes.reshape(shape, order='F')
        maps.interp_edges_to_vol_averages(
                ex=gfield.fx, ey=gfield.fy, ez=gfield.fz,
                volumes=cell_volumes,
                ox=grad_x, oy=grad_y, oz=grad_z)
        grad = grad_x + grad_y + grad_z

        # Bring the gradient back from the computation grid to the model grid.
        this_gradient = maps.interpolate(
                    gfield.grid, -grad, simulation.model.grid, method='cubic')

        # => Frequency-dependent depth-weighting should go here.

        # Add this src-freq gradient to the total gradient.
        gradient_model += this_gradient

    # => Frequency-independent depth-weighting should go here.

    # Apply derivative-chain of property-map
    # (only relevant if `mapping` is something else than conductivity).
    simulation.model.map.derivative_chain(
            gradient_model, simulation.model.property_x)

    return gradient_model
