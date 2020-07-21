"""

:mod:`optimize` -- Inversion
============================

Functionalities related to optimization (inversion), e.g., misfit function,
gradient of the misfit function, or data- and depth-weighting.

Currently it follows the adjoint-state method as described in [PlMu08]_.

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

__all__ = ['gradient', 'data_misfit']


def gradient(simulation):
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
            grad_x = np.zeros(simulation._dict_grid[src][freq].vnC, order='F')
            grad_y = np.zeros(simulation._dict_grid[src][freq].vnC, order='F')
            grad_z = np.zeros(simulation._dict_grid[src][freq].vnC, order='F')

            # TODO v TEST v TODO
            #
            # Here, we do
            #   1. edges2cellaverages (Ex[comp] -> CC[comp])
            #   2. grid2grid      (CC[comp] -> CC[model])
            #
            # Not better the other way around?
            #   1. grid2grid      (Ex[comp] -> Ex[model])
            #   1. edges2cellaverages (Ex[model] -> CC[model])
            #
            # TODO ^ TEST ^ TODO

            # Map the field to cell centers times volume.
            vnC = simulation._dict_grid[src][freq].vnC
            vol = simulation._dict_grid[src][freq].vol.reshape(vnC, order='F')
            maps.edges2cellaverages(efield.fx, efield.fy, efield.fz,
                                    vol, grad_x, grad_y, grad_z)
            grad = grad_x + grad_y + grad_z

            # Bring the gradient back from the computation grid to the model
            # grid.
            tgrad = maps.grid2grid(
                        simulation._dict_grid[src][freq],
                        -grad, simulation.grid, method='cubic')

            # TODO generalize (chain rule of mapping)
            simulation.model.map.derivative(tgrad, simulation.model.property_x)

            # Add this src-freq gradient to the total gradient.
            # if D is not None:
            #     grad_model += precond*tgrad
            # else:
            grad_model += tgrad

    return grad_model


def data_misfit(simulation):
    r"""Return the misfit between observed and synthetic data.

    The weighted least-squares functional, as implemented in `emg3d`, is
    given by Equation 1 [PlMu08]_,

    .. math::
        :label:misfit

            J(\textbf{p}) = \frac{1}{2} \sum_f\sum_s\sum_r
                \left\{
                \left\lVert
                    W_{s,r,f}^e\left(\textbf{e}_{s,r,f}[\sigma(\textbf{p})]
                    -\textbf{e}_{s,r,f}^\text{obs}\right)
                \right\rVert^2
                + \left\lVert
                    W_{s,r,f}^h\left(\textbf{h}_{s,r,f}[\sigma(\textbf{p})]
                    -\textbf{h}_{s,r,f}^\text{obs}\right)
                \right\rVert^2
                \right\}
            + R(\textbf{p}) \, .

    """

    # # TODO: - Data weighting;
    # #       - Min_offset;
    # #       - Noise floor.
    # DW = optimize.weights.DataWeighting(**self.data_weight_opts)

    # Store the residual.
    simulation.data['residual'] = (simulation.data.synthetic -
                                   simulation.data.observed)

    # Store a copy for the weighted residual.
    simulation.data['wresidual'] = simulation.data.residual.copy()

    # Compute the weights.
    # for src, freq in simulation._srcfreq:
    #     data = simulation.data.wresidual.loc[src, :, freq].data
    #
    #     # # TODO: Actual weights.
    #     # weig = DW.weights(
    #     #         data,
    #     #         simulation.survey.rec_coords,
    #     #         simulation.survey.sources[src].coords,
    #     #         freq)
    #     # simulation.survey._data['wresidual'].loc[sname, :, freq] *= weig

    data_misfit = np.sum(np.abs(simulation.data.residual.data.conj() *
                                simulation.data.wresidual.data))/2

    return data_misfit
