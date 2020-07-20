"""

:mod:`gradient` -- Gradient of misfit
=====================================

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
# from concurrent import futures

from emg3d import maps
# from emg3d.optimize import weights

__all__ = ['adjointstate']


def adjointstate(simulation):
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
    # D = None  # TODO
    # wdepth if wdepth is None else weights.DepthWeighting(mesh, **wdepth)

    # Pre-allocate the gradient on the mesh.
    grad_model = np.zeros(simulation.grid.vnC, order='F')

    # Initiate the preconditioner-dict.
    # precond_dict = {}

    # Loop over sources.
    for src in simulation.survey.sources.keys():

        # Loop over frequencies.
        for freq in simulation.survey.frequencies:

            # Get depth weights.
            # if D is not None:
            #     if freq in precond_dict.keys():
            #         precond = precond_dict[freq]
            #     else:
            #         precond = D.weights(freq).reshape(mesh.vnC, order='F')
            #         precond_dict[freq] = precond

            # Multiply forward field with backward field; take real part.
            efield = -np.real(
                    simulation._dict_bfield[src][freq] *
                    simulation._dict_efield[src][freq] *
                    simulation._dict_efield[src][freq].smu0)

            # Pre-allocate the gradient for the computational grid.
            grad = np.zeros(simulation._dict_grid[src][freq].vnC, order='F')

            # TODO v TEST v TODO
            #
            # Here, we do
            #   1. avg_field2cell (Ex[comp] -> CC[comp])
            #   2. grid2grid      (CC[comp] -> CC[model])
            #
            # Not better the other way around?
            #   1. grid2grid      (Ex[comp] -> Ex[model])
            #   1. avg_field2cell (Ex[model] -> CC[model])
            #
            # TODO ^ TEST ^ TODO

            # Map the field to cell centers times volume.
            maps.avg_field2cell_volume(
                    grad, simulation._dict_grid[src][freq].vol.reshape(
                        simulation._dict_grid[src][freq].vnC, order='F'),
                    efield.fx, efield.fy, efield.fz)

            # Bring the gradient back from the computation grid to the model
            # grid.
            tgrad = maps.grid2grid(
                        simulation._dict_grid[src][freq],
                        grad, simulation.grid, method='cubic')

            # TODO generalize (chain rule of mapping)
            simulation.model.map.derivative(tgrad, simulation.model.property_x)

            # Add this src-freq gradient to the total gradient.
            # if D is not None:
            #     grad_model += precond*tgrad
            # else:
            grad_model += tgrad

    return grad_model
