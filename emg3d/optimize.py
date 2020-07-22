"""

:mod:`optimize` -- Inversion
============================

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
    misfit function, as implemented in `emg3d`, is given by Equation 1 of
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
            + R(\textbf{p}) \, ,

    where :math:`\Delta^{\{e;h\}}` are the residuals between the observed and
    synthetic data,

    .. math::
        :label: misfit_e

            \Delta^e = \textbf{e}_{s,r,f}[\sigma(\textbf{p})]
                       -\textbf{e}_{s,r,f}^\text{obs} \, ,

    and

    .. math::
        :label: misfit_h

            \Delta^h = \textbf{h}_{s,r,f}[\sigma(\textbf{p})]
                       -\textbf{h}_{s,r,f}^\text{obs} \, .

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

    # # So far only Ex is implemented and checked.
    # if sum([(r.azm != 0.0)+(r.dip != 0.0) for r in
    #        simulation.survey.receivers.values()]) > 0:
    #    raise NotImplementedError(
    #            "Gradient only implement for Ex receivers "
    #            "at the moment.")
    # # TODO # # DATA WEIGHTING
    # data_misfit = simulation.data_misfit
    # # Get backwards electric fields (parallel).
    # ????._bcompute()

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


def data_weighting(simulation):
    r"""Return weighted residual.

    Returns the weighted residual as given in Equation 18 of [PlMu08]_,

    .. math::
        :label: data-weighting

        W(\textbf{x}_s, \textbf{x}_r, \omega) =
        \frac{\lVert\textbf{x}_s-\textbf{x}_r\rVert^{\gamma_d}}
        {\omega^{\beta_f}
         \lVert E^\text{ref}(\textbf{x}_s, \textbf{x}_r, \omega)
         \rVert^{\beta_d}}\, .


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
    data_weight = (offsets**gamma_d)[:, :, None]

    # (B.2) Second term: Frequency weighting (src-freq-indep.).
    omega = 2*np.pi*simulation.survey.frequencies
    data_weight /= (omega**beta_f)[None, None, :]

    # (B.3) Third term: Amplitude weighting.
    ref_data = simulation.data.get(refname, None)
    if ref_data is None:
        print(f"Reference data '{refname}' not found, using 'observed'.")
        ref_data = simulation.data.observed
    data_weight /= np.sqrt(np.real(ref_data.conj()*ref_data))**beta_d

    # (C) APPLY.
    wresidual = simulation.data.residual * data_weight
    wresidual.data[mute] = 0.0

    return wresidual


##########

def bfields(self, source, frequency, **kwargs):
    """¿¿¿ Merge with efields or move to gradient. ???"""

    freq = float(frequency)

    # Get solver options and update with kwargs.
    solver_opts = {**self.solver_opts, **kwargs}
    solver_opts['return_info'] = True  # Always return solver info.

    # Compute back-propagating electric field.
    bfield, info = solver.solve(
            self.get_grid(source, freq),
            self.get_model(source, freq),
            self._rfields(source, freq),
            **solver_opts)

    # Store electric field and info.
    if not hasattr(self, '_dict_bfield'):
        self._dict_bfield = self._dict_initiate()
        self._back_info = self._dict_initiate()
    self._dict_bfield[source][freq] = bfield
    self._back_info[source][freq] = info

    # Return electric field.
    return (self._dict_bfield[source][freq],
            self._back_info[source][freq])

def _call_bfields(self, inp):
    return self.bfields(*inp)

def _bcompute(self, **kwargs):
    """¿¿¿ Merge with compute or move to gradient. ???"""
    # TODO TODO

    # Get all source-frequency pairs.
    srcfreq = list(itertools.product(self.survey.sources.keys(),
                                        self.survey.frequencies))

    # Initiate futures-dict to store output.
    disable = self.max_workers >= len(srcfreq)
    out = process_map(
            self._call_bfields, srcfreq,
            max_workers=self.max_workers,
            desc='Compute bfields',
            bar_format='{desc}: {bar}{n_fmt}/{total_fmt}  [{elapsed}]',
            disable=disable)

    # Store electric field and info.
    if not hasattr(self, '_dict_bfield'):
        self._dict_bfield = self._dict_initiate()
        self._back_info = self._dict_initiate()

    # Extract and store.
    i = 0
    warned = False
    for src in self.survey.sources.keys():
        for freq in self.survey.frequencies:
            # Store efield.
            self._dict_bfield[src][freq] = out[i][0]

            # Store solver info.
            info = out[i][1]
            self._back_info[src][freq] = info
            if info['exit'] != 0:
                if not warned:
                    print("Solver warnings:")
                    warned = True
                print(f"- Src {src}; {freq} Hz : {info['exit_message']}")

            i += 1

def _rfields(self, source, frequency):
    """¿¿¿ Merge with sfields or move to optimize/gradient. ???"""

    freq = float(frequency)
    grid = self.get_grid(source, frequency)

    # Initiate empty field
    ResidualField = fields.SourceField(grid, freq=frequency)

    # Loop over receivers, input as source.
    for rname, rec in self.survey.receivers.items():

        # Strength: in get_source_field the strength is multiplied with
        # iwmu; so we undo this here.
        # TODO Ey, Ez
        strength = self.data['wresidual'].loc[
                source, rname, freq].data.conj()
        strength /= ResidualField.smu0
        # ^ WEIGHTED RESIDUAL ^!

        ThisSField = fields.get_source_field(
            grid=grid,
            src=rec.coordinates,
            freq=frequency,
            strength=strength,
        )

        # If strength is zero (very unlikely), get_source_field would
        # return a normalized field for a unit source. However, in this
        # case we do not want that.
        if strength != 0:
            ResidualField += ThisSField

    return ResidualField
