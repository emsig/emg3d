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

try:
    import tqdm
    import tqdm.contrib.concurrent
except ImportError:
    tqdm = None

from emg3d import io, solver


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
