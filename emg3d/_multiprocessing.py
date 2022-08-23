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
def empymod(inp):
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
    from empymod import bipole

    # Extract input.
    src = inp['src']
    receivers = inp['receivers']
    freqtime = inp['freqtime']
    model = inp['model']
    empymod_opts = inp['empymod_opts']
    observed = inp['observed']
    lopts = inp['layered_opts']
    method = lopts.pop('method', 'midpoint')
    lopts['return_imat'] = True

    gradient = inp.get('gradient', False)
    if gradient:
        weights = inp['weights']
        residual = inp['residual']

    # Ensure we can handle the source and receivers.
    fail = False

    point = 'Point' in src.__class__.__name__
    dipole = 'Dipole' in src.__class__.__name__
    if not (point or dipole):
        fail = True

    for r in receivers.values():
        point = 'Point' in r.__class__.__name__
        dipole = 'Dipole' in r.__class__.__name__
        if not (point or dipole):
            fail = True
            break

    # TODO improve!
    if fail:
        raise ValueError("Layered: Only Points and Dipoles supported!")

    # rec-independent empymod options
    empymod_opts = {
        # User input ({src;rec}pts, {h;f}t, {h;f}targ, xdirect, loop).
        # Contains also verb from simulation class.
        **empymod_opts,
        # Source properties, same for all receivers.
        'src': src.coordinates,
        'msrc': src.xtype != 'electric',
        'strength': src.strength,
        # Enforced properties (not implemented).
        'signal': None,
        'epermV': None,
        'mpermV': None,
        'squeeze': True,
    }

    # If method='source', we get the layered model only once.
    if method == 'source':
        p0 = src.center[:2]
        lopts['method'] = 'midpoint'
        # NOTE: IF 'source', and IF all rec_z equal, it would be faster
        # computing it here for all receivers at once.
        oned, imat = model.extract_1d(p0=p0, **lopts)
    elif method == 'receiver':
        lopts['method'] = 'midpoint'
    else:
        p0 = src.center[:2]
        lopts['method'] = method

    # Pre-allocate output array.
    if gradient:
        out = np.zeros((3, *model.shape))
    else:
        out = np.full((len(receivers), len(freqtime)), np.nan+1j*np.nan)

    # Loop over receivers.
    for i, (rkey, rec) in enumerate(receivers.items()):

        # Skip src-rec-freq pairs without data.
        if observed is not None:
            fi = np.isfinite(observed[i, :].data)
            if fi.sum() == 0:
                continue
            freqs = np.array(freqtime)[fi]
        elif observed is None and gradient:
            continue
        else:
            fi = np.ones(len(freqtime), dtype=bool)
            freqs = freqtime

        if method != 'source':
            if method == 'receiver':
                p0 = rec.center[:2]
                p1 = None
            else:
                p1 = rec.center[:2]

            oned, imat = model.extract_1d(p0=p0, p1=p1, **lopts)

        def none_squeeze(inp):
            return None if inp is None else inp[0, 0, :]

        def none_bwd_squeeze(inp):
            if inp is None:
                return None
            else:
                return oned.map.backward(inp[0, 0, :])

        def none_aniso(cond_h, cond_v):
            if cond_v is None:
                return None
            else:
                return np.sqrt(cond_h/cond_v)

        # Collect input.
        empymod_inp = {
            **empymod_opts,
            'rec': rec.coordinates,
            'depth': oned.grid.nodes_z[1:-1],
            'freqtime': freqs,
            'mrec': rec.xtype != 'electric',
            'epermH': none_squeeze(oned.epsilon_r),
            'mpermH': none_squeeze(oned.mu_r),
        }

        cond_h = none_bwd_squeeze(oned.property_x)
        cond_v = none_bwd_squeeze(oned.property_z)

        # TODO simplify, factor out this whole fd-grad shabang.
        if gradient:

            # Get misfit of this src-rec pair.
            obs = observed.loc[rkey, :].data[fi]
            rwgt = weights.loc[rkey, :].data[fi]
            rres = residual.loc[rkey, :].data[fi]
            rmis = np.sum(rwgt*(rres.conj()*rres)).real/2

            for iz in range(model.shape[2]):

                # Get 1D model.
                lcond = cond_h.copy()

                # Add a relative error of 0.1 % to the layer.
                delta = lcond[iz] * 0.0001
                lcond[iz] += delta

                # Compute empymod and place in output array.
                aniso = none_aniso(lcond, cond_v)
                resp = bipole(res=1/lcond, aniso=aniso, **empymod_inp)

                # Get misfit
                lres = resp - obs
                lmis = np.sum(rwgt*(lres.conj()*lres)).real/2

                # Return gradient
                grad = (lmis - rmis)/delta
                out[0, :, :, iz] += np.einsum('ji,->ij', imat.T,  grad)

            if oned.property_z is not None:

                for iz in range(model.shape[2]):

                    # Get 1D model.
                    lcond = cond_v.copy()

                    # Add a relative error of 0.1 % to the layer.
                    delta = lcond[iz] * 0.0001
                    lcond[iz] += delta

                    # Compute empymod and place in output array.
                    aniso = none_aniso(cond_h, lcond)
                    resp = bipole(res=1/cond_h, aniso=aniso, **empymod_inp)

                    # Get misfit
                    lres = resp - obs
                    lmis = np.sum(rwgt*(lres.conj()*lres)).real/2

                    # Return gradient
                    grad = (lmis - rmis)/delta
                    out[2, :, :, iz] += np.einsum('ji,->ij', imat.T,  grad)

        else:
            # Compute empymod and place in output array.
            aniso = none_aniso(cond_h, cond_v)
            out[i, fi] = bipole(res=1/cond_h, aniso=aniso, **empymod_inp)

    return out
