"""
Parser for the configuration file of the command-line interface.
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

import os
import warnings
import configparser
from pathlib import Path


def parse_config_file(args_dict):
    """Read and parse the configuration file and set defaults.

    Parameters
    ----------
    args_dict : dict
        Arguments from terminal, see :func:`emg3d.cli.main`.


    Returns
    -------
    conf : dict
        Configuration-dict.

    """

    # # Read parameter-file into dict # #
    config = args_dict.pop('config')
    configfile = os.path.abspath(config)
    cfg = configparser.ConfigParser(inline_comment_prefixes='#')

    # Check if config-file is actually a file.
    if os.path.isfile(configfile):

        # If it is, read it.
        with open(configfile) as f:
            cfg.read_file(f)

    elif config == '.':

        # If config == '.', no parameter file is given. Workaround to suppress
        # warning when one does not want to provide a config file.
        configfile = config

    # # Check the terminal arguments # #

    # Initiate terminal dict.
    term = {}

    # Add config-file to term.
    term['config_file'] = configfile

    # Get terminal input.
    for key in ['verbosity', 'nproc', 'dry_run', 'clean']:
        term[key] = args_dict.pop(key)

    for key in ['forward', 'misfit', 'gradient']:
        function = args_dict.pop(key)
        if function:
            term['function'] = key
    if 'function' not in term.keys():
        term['function'] = 'forward'

    # Get file names.
    for key in ['path', 'survey', 'model', 'output', 'save', 'load', 'cache']:
        term[key] = args_dict.pop(key)

    # Ensure no keys are left.
    if args_dict:
        raise TypeError(
            f"Unexpected parameter in **args_dict: {list(args_dict.keys())}."
        )

    # Enforce some limits.
    term['verbosity'] = int(min(max(term['verbosity'], -1), 2))  # [-1, 2]
    if term['nproc'] is not None:
        term['nproc'] = int(max(term['nproc'], 1))               # [1, inf]

    # # Check file-paths and files # #

    # Check if parameter-file has a files-section, add it otherwise.
    if 'files' not in cfg.sections():
        cfg.add_section('files')

    # Get file names.
    all_files = dict(cfg.items('files'))

    # First path.
    path = term.pop('path')
    if path is None:
        path = all_files.pop('path', '.')
    path = os.path.abspath(path)

    # Initiate files dict with defaults.
    files = {'save': False, 'load': False, 'cache': False,
             'survey': 'survey', 'model': 'model', 'output': 'emg3d_out'}
    for key, value in files.items():

        config_or_default = all_files.pop(key, value)

        # Get terminal input.
        fname = term.pop(key)

        # If there was no terminal input, get config-file; else, default.
        if fname is None:
            fname = config_or_default

        # Next file if it is not provided.
        if not fname:
            continue

        # Get absolute paths.
        ffile = Path(os.path.join(path, fname))

        # Ensure there is a file ending, if not, fall back to h5.
        if ffile.suffix not in ['.h5', '.json', '.npz']:
            ffile = ffile.with_suffix('.h5')

        if key == 'output':
            logfile = str(ffile.with_suffix('.log'))

        # Store in dict.
        files[key] = str(ffile)

    # If cache, it overwrites save/load:
    cache = files.pop('cache')
    if cache:
        files['load'] = cache
        files['save'] = cache

    # Add log file.
    files['log'] = logfile

    # Ensure no keys are left.
    if all_files:
        raise TypeError(
            f"Unexpected parameter in [files]: {list(all_files.keys())}."
        )

    # # Simulation parameters  # #

    # Initiate dict.
    simulation = {}

    # Check if parameter-file has a simulation-section, add it otherwise.
    if 'simulation' not in cfg.sections():
        cfg.add_section('simulation')
    all_sim = dict(cfg.items('simulation'))

    # Check max_workers.
    key = 'max_workers'
    _ = all_sim.pop(key, None)
    if term['nproc'] is not None:
        simulation[key] = term['nproc']
    elif cfg.has_option('simulation', key):
        simulation[key] = cfg.getint('simulation', key)
    del term['nproc']

    # Check gridding.
    key = 'gridding'
    if cfg.has_option('simulation', key):
        _ = all_sim.pop(key)
        simulation[key] = cfg.get('simulation', key)

    # Check name.
    key = 'name'
    if cfg.has_option('simulation', key):
        _ = all_sim.pop(key)
        simulation[key] = cfg.get('simulation', key)
    else:
        simulation[key] = "emg3d CLI run"

    key = 'file_dir'
    if cfg.has_option('simulation', key):
        _ = all_sim.pop(key)
        simulation[key] = cfg.get('simulation', key)

    key = 'receiver_interpolation'
    if cfg.has_option('simulation', key):
        _ = all_sim.pop(key)
        simulation[key] = cfg.get('simulation', key)
    elif term['function'] == 'gradient':
        # Default is 'cubic' - gradient needs 'linear'
        simulation[key] = 'linear'

    # Check noise parameters => deprecated in ``simulation``, see [noise_opts]!
    noise_kwargs = {}
    keys = ['min_offset', 'mean_noise', 'max_offset']
    for key in keys:
        if cfg.has_option('simulation', key):
            _ = all_sim.pop(key)
            noise_kwargs[key] = cfg.getfloat('simulation', key)
    key = 'ntype'
    if cfg.has_option('simulation', key):
        _ = all_sim.pop(key)
        noise_kwargs[key] = cfg.get('simulation', key)

    # Ensure no keys are left.
    if all_sim:
        raise TypeError(
            f"Unexpected parameter in [simulation]: {list(all_sim.keys())}."
        )

    # # Noise parameters  # #

    if len(noise_kwargs) > 0:
        msg = ("emg3d: `min_offset`, `max_offset`, `mean_noise`, and `ntype` "
               "belong since v1.7.0 in their own section [noise_opts]; "
               "providing them in [simulation] is deprecated and will be "
               "removed in v1.9.0.")
        warnings.warn(msg, FutureWarning)

    # Check if parameter-file has a noise-section, add it otherwise.
    # Note: values in noise_opts overwrite values from simulation.
    if 'noise_opts' in cfg.sections():

        all_noise = dict(cfg.items('noise_opts'))

        keys = ['min_offset', 'max_offset', 'mean_noise']
        for key in keys:
            if cfg.has_option('noise_opts', key):
                _ = all_noise.pop(key)
                noise_kwargs[key] = cfg.getfloat('noise_opts', key)
        key = 'ntype'
        if cfg.has_option('noise_opts', key):
            _ = all_noise.pop(key)
            noise_kwargs[key] = cfg.get('noise_opts', key)

        # Ensure no keys are left.
        if all_noise:
            raise TypeError(
                f"Unexpected parameter in [noise_opts]: "
                f"{list(all_noise.keys())}."
            )

    # # Solver parameters  # #

    # Check if parameter-file has a solver-section, add it otherwise.
    if 'solver_opts' in cfg.sections():

        # Initiate solver-dict.
        solver = {}

        all_solver = dict(cfg.items('solver_opts'))

        # Check for bools.
        for key in ['sslsolver', 'semicoarsening', 'linerelaxation', 'plain']:
            if cfg.has_option('solver_opts', key):
                _ = all_solver.pop(key)
                solver[key] = cfg.getboolean('solver_opts', key)

        # Check for strings.
        for key in ['cycle', ]:
            if cfg.has_option('solver_opts', key):
                _ = all_solver.pop(key)
                solver[key] = cfg.get('solver_opts', key)

        # Check for floats.
        for key in ['tol', ]:
            if cfg.has_option('solver_opts', key):
                _ = all_solver.pop(key)
                solver[key] = float(cfg.get('solver_opts', key))

        # Check for ints.
        int_keys = ['verb', 'maxit', 'nu_init', 'nu_pre', 'nu_coarse',
                    'nu_post', 'clevel']
        for key in int_keys:
            if cfg.has_option('solver_opts', key):
                _ = all_solver.pop(key)
                solver[key] = cfg.getint('solver_opts', key)

        # Ensure no keys are left.
        if all_solver:
            raise TypeError(
                f"Unexpected parameter in [solver_opts]: "
                f"{list(all_solver.keys())}."
            )

        # Add to simulation dict if not empty.
        if solver:
            simulation['solver_opts'] = solver

    # # Data selection parameters  # #

    data = {}
    if 'data' in cfg.sections():
        # Get all parameters.
        all_data = dict(cfg.items('data'))

        for key in ['sources', 'receivers', 'frequencies']:
            value = all_data.pop(key, False)
            if value:
                data[key] = [v.strip() for v in value.split(',')]

        # Check for bools.
        for key in ['remove_empty', ]:
            if cfg.has_option('data', key):
                _ = all_data.pop(key)
                data[key] = cfg.getboolean('data', key)

        # Ensure no keys are left.
        if all_data:
            raise TypeError(
                f"Unexpected parameter in [data]: {list(all_data.keys())}."
            )

    # # Gridding # #

    if 'gridding_opts' in cfg.sections():
        grid = {}

        all_grid = dict(cfg.items('gridding_opts'))

        # Check for lists.
        list_keys = ['properties', 'center', 'cell_number', 'min_width_pps',
                     'expand']
        for key in list_keys:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = [float(v) for v in
                             cfg.get('gridding_opts', key).split(',')]

        # Check for list of lists.
        for key in ['domain', 'distance', 'stretching', 'min_width_limits',
                    'center_on_edge']:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                out = []
                for p in cfg.get('gridding_opts', key).split(';'):
                    if 'none' in p.lower():
                        out.append(None)
                    elif 'true' in p.lower():
                        out.append(True)
                    elif 'false' in p.lower():
                        out.append(False)
                    else:
                        out.append([float(v) for v in p.split(',')])
                if len(out) == 1:
                    out = out[0]
                else:
                    out = {'x': out[0], 'y': out[1], 'z': out[2]}
                grid[key] = out

        # Check for strings.
        for key in ['mapping', 'vector']:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = cfg.get('gridding_opts', key)

        # Check for floats.
        for key in ['frequency', 'seasurface', 'max_buffer', 'lambda_factor']:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = float(cfg.get('gridding_opts', key))

        # Check for ints.
        for key in ['verb', ]:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = cfg.getint('gridding_opts', key)

        # Check for bools.
        for key in ['lambda_from_center', ]:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = cfg.getboolean('gridding_opts', key)

        # Ensure no keys are left.
        if all_grid:
            raise TypeError(
                f"Unexpected parameter in [gridding_opts]: "
                f"{list(all_grid.keys())}"
            )

        # Add to simulation dict if not empty.
        if grid:
            simulation['gridding_opts'] = grid

    # Return.
    out = {'files': files, 'simulation_options': simulation, 'data': data,
           'noise_kwargs': noise_kwargs}
    return out, term
