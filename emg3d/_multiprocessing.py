"""
Helper routines to call functions with multiprocessing/concurrent.futures.
"""
# Copyright 2018 The emsig community.
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
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    import tqdm
    import tqdm.contrib.concurrent
except ImportError:
    tqdm = None

from emg3d import io, solver, utils


def process_map(fn, *iterables, max_workers, **kwargs):
    """Dispatch processes in parallel or not, using tqdm or not.

    :class:`emg3d.simulations.Simulation` uses the function
    ``tqdm.contrib.concurrent.process_map`` to run jobs asynchronously.
    However, ``tqdm`` is a soft dependency. In case it is not installed we use
    the class ``concurrent.futures.ProcessPoolExecutor`` directly, from the
    standard library, and imitate the behaviour of process_map (basically a
    ``ProcessPoolExecutor.map``, returned as a list, and wrapped in a context
    manager). If max_workers is smaller than two then we we avoid parallel
    execution.

    """
    process_map.count += 1

    # Parallel
    if max_workers > 1 and tqdm is None:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, *iterables))

    # Parallel with tqdm
    elif max_workers > 1:
        return tqdm.contrib.concurrent.process_map(
                fn, *iterables, max_workers=max_workers, **kwargs)

    # Sequential
    elif tqdm is None:
        return list(map(fn, *iterables))

    # Sequential with tqdm
    else:
        return list(tqdm.auto.tqdm(
            iterable=map(fn, *iterables), total=len(iterables[0]), **kwargs))


# Counter for processing map (used, e.g., for inversions).
process_map.count = 0


def solve(inp):
    """Thin wrapper of `solve` or `solve_source` for a `process_map`.

    Used within a Simulation to call the solver in parallel. This function
    always returns the ``efield`` and the ``info_dict``, independent of the
    provided solver options.


    Parameters
    ----------
    inp : dict, str
        If dict, two formats are recognized:
        - Has keys [model, sfield, efield, solver_opts]:
          Forwarded to `solve`.
        - Has keys [model, grid, source, frequency, efield, solver_opts]
          Forwarded to `solve_source`.

        Consult the corresponding function for details on the input parameters.

        Alternatively the path to the h5-file can be provided as a string
        (file-based computation).

        The ``model`` is interpolated to the grid of the source field (tuple of
        length 4) or to the provided grid (tuple of length 6). Hence, the model
        can be on a different grid (for source and frequency dependent
        gridding).


    Returns
    -------
    efield : Field
        Resulting electric field, as returned from :func:`emg3d.solver.solve`
        or :func:`emg3d.solver.solve_source`.

    info_dict : dict
        Resulting info dictionary, as returned from :func:`emg3d.solver.solve`
        or :func:`emg3d.solver.solve_source`.

    """

    # Four parameters => solve.
    fname = False
    if isinstance(inp, str):
        fname = inp.rsplit('.', 1)[0] + '_out.' + inp.rsplit('.', 1)[1]
        inp = io.load(inp, verb=0)['data']

    # Has keys [model, sfield, efield, solver_opts]
    if 'sfield' in inp.keys():

        # Get input and initiate solver dict.
        solver_input = {**inp['solver_opts'], 'sfield': inp['sfield']}
        inp['grid'] = inp['sfield'].grid

        # Function to compute.
        fct = solver.solve

    # Has keys [model, grid, source, frequency, efield, solver_opts]
    else:

        # Get input and initiate solver dict.
        solver_input = {**inp['solver_opts'], 'source': inp['source'],
                        'frequency': inp['frequency']}

        # Function to compute.
        fct = solver.solve_source

    # Interpolate model to source grid (if different).
    model = inp['model'].interpolate_to_grid(inp['grid'])

    # Add general parameters to input dict.
    solver_input['model'] = model
    solver_input['efield'] = inp['efield']
    solver_input['return_info'] = True
    solver_input['always_return'] = True

    # Return the result.
    efield, info = fct(**solver_input)
    if fname:
        io.save(fname, efield=efield, info=info, verb=0)
        return fname, fname
    else:
        return efield, info


@utils._requires('empymod')
def layered(inp):
    """Returns response or gradient using layered models; for a `process_map`.

    Used within a Simulation to call empymod in parallel for layered models.
    Depending on the input it returns either the forward responses or the
    finite-difference gradient.

    The parameters section describes the content of the input dict.

    Parameters
    ----------
    model : Model
        The model; a :class:`emg3d.models.Model` instance. Must be isotropic or
        VTI.

    src : Tx*
        Any dipole or point source of the available sources, e.g.,
        :class:`emg3d.electrodes.TxElectricDipole`.

    receivers : dict of Rx*
        Receiver dict (:attr:`emg3d.surveys.Survey.receivers`).

    frequencies : dict
        Frequency dict (:attr:`emg3d.surveys.Survey.frequencies`).

    empymod_opts : dict
        Options passed to empymod ({src;rec}pts, {h;f}t, {h;f}targ, xdirect,
        loop, verb).

    observed : DataArray
        Observed data of this source.

    layered_opts : dict
        Options passed to :attr:`emg3d.models.Model.extract_1d`.

    gradient : bool
        If False, the electromagnetic responses are returned; if True, the
        gradient is returned.

        If True, the following things _have_ to be provided: ``observed``,
        ``weights``, and ``residual``; otherwise a zero gradient is returned.

    weights : DataArray, optional
        Data weights corresponding to the data; only required if
        ``gradient=True``.

    residual : DataArray
        Residuals using the current model; only required if ``gradient=True``.


    Returns
    -------
    out : ndarray
        If ``gradient=False``, the output are the electromagnetic responses
        (synthetic data) of shape (nrec, nfreq).

        If ``gradient=True``, the output is the finite-difference gradient of
        shape (3, nx, ny, nz).

    """

    # Extract input.
    model = inp['model']
    src = inp['src']
    receivers = inp['receivers']
    frequencies = np.array([f for f in inp['frequencies'].values()])
    empymod_opts = inp['empymod_opts']
    observed = inp['observed']
    lopts = deepcopy(inp['layered_opts'])
    gradient = inp['gradient']

    # Get method and set to return_imat.
    method = lopts.pop('method')
    lopts['return_imat'] = True

    # Collect rec-independent empymod options.
    empymod_opts = {
        # User input ({src;rec}pts, {h;f}t, {h;f}targ, xdirect, loop, verb).
        **empymod_opts,
        #
        # Source properties, same for all receivers.
        'src': src.coordinates,
        'msrc': src.xtype != 'electric',
        'strength': src.strength,
        #
        # Enforced properties (not implemented).
        'signal': None,
        'epermV': None,
        'mpermV': None,
        'squeeze': True,
    }

    # Create some flags.
    epsilon_r = model.epsilon_r is not None
    mu_r = model.mu_r is not None
    vti = model.case == 'VTI'

    # Pre-allocate output array.
    if gradient:
        # Gradient.
        out = np.zeros((3, *model.shape))

        # Get weights and residual if the gradient is wanted.
        weights = inp.get('weights', None)
        residual = inp.get('residual', None)
        if weights is None or residual is None or observed is None:
            return out

    else:
        # Responses.
        out = np.full((len(receivers), frequencies.size), np.nan+1j*np.nan)

    # Loop over receivers.
    for i, (rkey, rec) in enumerate(receivers.items()):

        # Check observed data, limit to finite values if provided.
        if observed is not None:
            fi = np.isfinite(observed.loc[rkey, :].data)
            if fi.sum() == 0:
                continue
            freqs = frequencies[fi]

        # Generating obs data for all.
        else:
            fi = np.ones(frequencies.size, dtype=bool)
            freqs = frequencies

        # Get 1D model.
        # Note: if method='source', this would be faster outside the loop.
        oned, imat = model.extract_1d(**_get_points(method, src, rec), **lopts)

        # Collect input.
        empymod_inp = {
            **empymod_opts,
            'rec': rec.coordinates,
            'mrec': rec.xtype != 'electric',
            'depth': oned.grid.nodes_z[1:-1],
            'freqtime': freqs,
            'epermH': None if not epsilon_r else oned.epsilon_r[0, 0, :],
            'mpermH': None if not mu_r else oned.mu_r[0, 0, :],
        }

        # Get horizontal and vertical conductivities.
        map2cond = oned.map.backward
        cond_h = map2cond(oned.property_x[0, 0, :])
        cond_v = None if not vti else map2cond(oned.property_z[0, 0, :])

        # Compute gradient.
        if gradient:

            # Get misfit of this src-rec pair.
            obs = observed.loc[rkey, :].data[fi]
            wgt = weights.loc[rkey, :].data[fi]
            res = residual.loc[rkey, :].data[fi]
            misfit = np.sum(wgt*(res.conj()*res)).real/2

            # Get horizontal gradient.
            out[0, ...] += _fd_gradient(cond_h, cond_v, obs, wgt, misfit,
                                        empymod_inp, imat, vertical=False)

            # Get vertical gradient if VTI.
            if vti:
                out[2, ...] += _fd_gradient(cond_h, cond_v, obs, wgt, misfit,
                                            empymod_inp, imat, vertical=True)

        # Compute response.
        else:
            out[i, fi] = _empymod_fwd(cond_h, cond_v, empymod_inp)

    return out


@utils._requires('empymod')
def _empymod_fwd(cond_h, cond_v, empymod_inp):
    """Thin wrapper for empymod.bipole().

    Parameters
    ----------
    cond_h, cond_v : ndarray
        Horizontal and vertical conductivities (S/m). ``cond_v`` can be None,
        in which case an isotropic medium is assumed.

    empymod_inp : dict
        Passed through to :func:`empymod.model.bipole`. Any parameter except
        for ``res`` and ``aniso``.


    Returns
    -------
    resp : EMArray
        Electromagnetic field as returned from :func:`empymod.model.bipole`.


    """
    from empymod import bipole
    aniso = None if cond_v is None else np.sqrt(cond_h/cond_v)
    return bipole(res=1/cond_h, aniso=aniso, **empymod_inp)


def _get_points(method, src, rec):
    """Returns correct method and points for model.extract_1d.

    Parameters
    ----------
    method : str
        All methods accepted by :attr:`emg3d.models.Model.extract_1d` plus
        ``'source'``, ``'receiver'``.

    src, rec : {Tx*, Rx*)
        Any of the available point and dipole sources or receivers, e.g.,
        :class:`emg3d.electrodes.TxElectricDipole`.

    Returns
    -------
    out : dict
        Can be passed directly to :attr:`emg3d.models.Model.extract_1d` for the
        parameters ``method``, ``p0``, and ``p1``.

    """

    # Get default points.
    p0 = src.center[:2]
    p1 = rec.center[:2]

    # If source or receiver, we re-set one point and rename the method
    if method == 'source':
        p1 = p0
        method = 'midpoint'

    elif method == 'receiver':
        p0 = p1
        method = 'midpoint'

    return {'method': method, 'p0': p0, 'p1': p1}


def _fd_gradient(cond_h, cond_v, data, weight, misfit, empymod_inp, imat,
                 vertical):
    """Computes the finite-difference gradient using empymod.

    The finite difference is obtained by adding a relative difference of 0.01 %
    to the layer (currently hard-coded).

    Parameters
    ----------
    cond_h, cond_v : ndarray
        Horizontal and vertical conductivities (S/m). ``cond_v`` can be None,
        in which case an isotropic medium is assumed.

    data : ndarray
        Observed data.

    weight : ndarray
        Weights corresponding to these data.

    misfit : float
        Misfit using the current model.

    empymod_inp : dict
        Passed through to :func:`empymod.model.bipole`. Any parameter except
        for ``res`` and ``aniso``.

    imat : ndarray
        Interpolation matrix as returned by
        :attr:`emg3d.models.Model.extract_1d`.

    vertical : bool
        If True, the gradient for the vertical conductivities is assumed,
        otherwise the gradient for the horizontal conductivities.
        If ``vertical=True``, ``cond_v`` cannot be None (not checked, will fail
        with an AttributeError).


    Returns
    -------
    gradient : ndarray of shape (nx, ny, nz)
        Gradient.


    """
    # Relative difference fixed to 0.01 %; could be made an input parameter.
    rel_diff = 0.0001

    # Loop over layers and compute FD gradient for each.
    grad = np.zeros(cond_h.size)
    for iz in range(cond_h.size):

        # Get 1D model.
        cond_p = cond_h.copy() if not vertical else cond_v.copy()

        # Add relative difference to the layer.
        delta = cond_p[iz] * rel_diff
        cond_p[iz] += delta

        # Call empymod.
        if vertical:
            response = _empymod_fwd(cond_h, cond_p, empymod_inp)
        else:
            response = _empymod_fwd(cond_p, cond_v, empymod_inp)

        # Calculate gradient and add it.
        residual = response - data
        fd_misfit = np.sum(weight*(residual.conj()*residual)).real/2
        grad[iz] = (fd_misfit - misfit)/delta

    # Bring back to full grid and return.
    return imat[..., None] * grad[None, :]
