"""
Helper routines to call functions with multiprocessing/concurrent.futures.
"""
# Copyright 2018-2022 The emsig community.
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
from copy import deepcopy as dc
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
    """TODO :: Work in progress for emg3d(empymod)

    All below is old

    Thin wrapper of `solve` or `solve_source` for a `process_map`.

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

    # Extract input.
    model = inp['model']
    src = inp['src']
    receivers = inp['receivers']
    freqtime = inp['freqtime']
    empymod_opts = inp['empymod_opts']
    observed = inp['observed']
    lopts = dc(inp['layered_opts'])
    gradient = inp['gradient']

    # Get method and set to return_imat.
    method = lopts.pop('method')
    lopts['return_imat'] = True

    # Collect rec-independent empymod options.
    empymod_opts = {
        # User input ({src;rec}pts, {h;f}t, {h;f}targ, xdirect, loop).
        # Contains also verb from simulation class.
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
        weights = inp['weights']
        residual = inp['residual']

    else:
        # Responses.
        out = np.full((len(receivers), len(freqtime)), np.nan+1j*np.nan)

    # Loop over receivers.
    for i, (rkey, rec) in enumerate(receivers.items()):

        # Check observed data, limit to finite values if provided.
        if observed is not None:
            fi = np.isfinite(observed[i, :].data)
            if fi.sum() == 0:
                continue
            freqs = np.array(freqtime)[fi]

        # Skip gradient if no observed data.
        elif observed is None and gradient:
            continue

        # Generating obs data for all.
        else:
            fi = np.ones(len(freqtime), dtype=bool)
            freqs = freqtime

        # Get 1D model.
        # Note: if 'method='source', this would be faster outside the loop.
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
            data = observed.loc[rkey, :].data[fi]
            weight = weights.loc[rkey, :].data[fi]
            residual = residual.loc[rkey, :].data[fi]
            misfit = np.sum(weight*(residual.conj()*residual)).real/2

            # Get horizontal gradient.
            _fd_gradient(out[0, ...], cond_h, cond_v, data, weight, misfit,
                         empymod_inp, imat, vertical=False)

            # Get vertical gradient if VTI.
            if vti:

                _fd_gradient(out[2, ...], cond_h, cond_v, data, weight, misfit,
                             empymod_inp, imat, vertical=True)

        # Compute response.
        else:
            out[i, fi] = _empymod_fwd(cond_h, cond_v, empymod_inp)

    return out


@utils._requires('empymod')
def _empymod_fwd(cond_h, cond_v, empymod_inp):
    # Compute empymod and place in output array.
    from empymod import bipole
    aniso = None if cond_v is None else np.sqrt(cond_h/cond_v)
    return bipole(res=1/cond_h, aniso=aniso, **empymod_inp)


def _get_points(method, src, rec):
    p0 = src.center[:2]
    p1 = rec.center[:2]

    if method == 'source':
        p1 = p0
        method = 'midpoint'
    elif method == 'receiver':
        p0 = p1
        method = 'midpoint'

    return {'method': method, 'p0': p0, 'p1': p1}


def _fd_gradient(gradient, cond_h, cond_v, data, weight, misfit, empymod_inp,
                 imat, vertical=False):

    # Loop over layers and compute FD gradient for each.
    for iz in range(cond_h.size):

        # Get 1D model.
        cond_p = cond_v.copy() if vertical else cond_h.copy()

        # Add a relative error of 0.1 % to the layer.
        delta = cond_p[iz] * 0.0001
        cond_p[iz] += delta

        # Call empymod.
        if vertical:
            response = _empymod_fwd(cond_h, cond_p, empymod_inp)
        else:
            response = _empymod_fwd(cond_p, cond_v, empymod_inp)

        # Calculate gradient and add it.
        residual = response - data
        fd_misfit = np.sum(weight*(residual.conj()*residual)).real/2
        grad = (fd_misfit - misfit)/delta
        # TODO Check the einsum!
        gradient[..., iz] += np.einsum('ji,->ij', imat.T, grad)
