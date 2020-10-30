"""
Parser for the configuration file of the command-line interface.
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

import os
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

    # Check if configfile is actually a file.
    if os.path.isfile(configfile):

        # If it is, read it.
        with open(configfile) as f:
            cfg.read_file(f)

    elif config == '.':

        # If config == '.', no parameter file is given. Workaround to suppress
        # warning when one does not want to provide a config file.
        configfile = config

    else:

        # Throw a warning that the parameter file was not found.
        configfile = False

    # # Check the terminal arguments # #

    # Initiate terminal dict.
    term = {}

    # Add config-file to term.
    term['config_file'] = configfile

    # Get terminal input.
    for key in ['verbosity', 'nproc', 'dry_run']:
        term[key] = args_dict.pop(key)

    for key in ['forward', 'misfit', 'gradient']:
        function = args_dict.pop(key)
        if function:
            term['function'] = key
    if 'function' not in term.keys():
        term['function'] = 'forward'

    # Get file names.
    for key in ['path', 'survey', 'model', 'output']:
        term[key] = args_dict.pop(key)

    # Ensure no keys are left.
    if args_dict:
        raise TypeError(f"Unexpected key in **args_dict: "
                        f"{list(args_dict.keys())}")

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
    files = {'survey': 'survey', 'model': 'model', 'output': 'emg3d_out'}
    for key, value in files.items():

        config_or_default = all_files.pop(key, value)

        # Get terminal input.
        fname = term.pop(key)

        # If there was no terminal input, get config-file; else, default.
        if fname is None:
            fname = config_or_default

        # Get absolute paths.
        ffile = Path(os.path.join(path, fname))

        # Ensure there is a file ending, if not, fall back to h5p
        if ffile.suffix not in ['.h5', '.json', '.npz']:
            ffile = ffile.with_suffix('.h5')

        # Store in dict.
        files[key] = ffile

    # Ensure files and directory exist:
    for key in ['survey', 'model']:
        files[key] = str(files[key])
    files['log'] = str(files['output'].with_suffix('.log'))
    files['output'] = str(files['output'])

    # Store options (get it with getboolean, and remove from dict).
    files['store_simulation'] = cfg.getboolean(
            'files', 'store_simulation', fallback=False)
    _ = all_files.pop('store_simulation', None)

    # Ensure no keys are left.
    if all_files:
        raise TypeError(f"Unexpected parameter in [files]: "
                        f"{list(all_files.keys())}")

    # # Simulation parameters  # #

    # Initiate dict.
    simulation = {}

    # Check if parameter-file has a simulation-section, add it otherwise.
    if 'simulation' not in cfg.sections():
        cfg.add_section('simulation')

    # Check max_workers.
    key = 'max_workers'
    if term['nproc'] is not None:
        simulation[key] = term['nproc']
    elif cfg.has_option('simulation', key):
        simulation[key] = cfg.getint('simulation', key)
    del term['nproc']

    # Check gridding.
    key = 'gridding'
    if cfg.has_option('simulation', key):
        simulation[key] = cfg.get('simulation', key)

    # Check name.
    key = 'name'
    if cfg.has_option('simulation', key):
        simulation[key] = cfg.get('simulation', key)
    else:
        simulation[key] = "emg3d CLI run"

    key = 'min_offset'
    if cfg.has_option('simulation', key):
        simulation[key] = cfg.getfloat('simulation', key)

    # # Solver parameters  # #

    # Check if parameter-file has a solver-section, add it otherwise.
    if 'solver_opts' in cfg.sections():

        # Initiate solver-dict.
        solver = {}

        # Check for bools.
        for key in ['sslsolver', 'semicoarsening', 'linerelaxation']:
            if cfg.has_option('solver_opts', key):
                solver[key] = cfg.getboolean('solver_opts', key)

        # Check for strings.
        for key in ['cycle', ]:
            if cfg.has_option('solver_opts', key):
                solver[key] = cfg.get('solver_opts', key)

        # Check for floats.
        for key in ['tol', ]:
            if cfg.has_option('solver_opts', key):
                solver[key] = float(cfg.get('solver_opts', key))

        # Check for ints.
        int_keys = ['verb', 'maxit', 'nu_init', 'nu_pre', 'nu_coarse',
                    'nu_post', 'clevel']
        for key in int_keys:
            if cfg.has_option('solver_opts', key):
                solver[key] = cfg.getint('solver_opts', key)

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

        # Ensure no keys are left.
        if all_data:
            raise TypeError(f"Unexpected parameter in [data]: "
                            f"{list(all_data.keys())}")

    # # Gridding # #

    if 'gridding_opts' in cfg.sections():
        grid = {}

        all_grid = dict(cfg.items('gridding_opts'))

        # Lists of floats.
        for key in ['res', 'min_width', 'zval']:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = [float(v) for v in
                             cfg.get('gridding_opts', key).split(',')]

        # Int.
        for key in ['verb', ]:
            if cfg.has_option('gridding_opts', key):
                _ = all_grid.pop(key)
                grid[key] = cfg.getint('gridding_opts', key)

        # Ensure no keys are left.
        if all_grid:
            raise TypeError(f"Unexpected parameter in [gridding_opts]: "
                            f"{list(all_grid.keys())}")

        # Add to simulation dict if not empty.
        if grid:
            simulation['gridding_opts'] = grid

#
# TODO move all docs from parser and example CLI to simulation!
#
#frequency : float                      from simulation (this or average)
#properties : float or list             from model (this and boundary averages)
#center : float                         from simulation (this or average)
#domain : list, optional                from source and receiver in horizontal
#vector : str ('x', 'xz', 'yz', ...)    «optional»
#seasurface : float                     «optional»
#stretching : list                      «optional»
#cell_numbers : list                    «optional»
#min_width_limits : float, list or None «optional»
#min_width_pps : float or int           «optional»
#max_buffer : float                     «optional»
#mapping : str, optional                from model, can be overwritten
#verb : int, optional                   from simulation
#raise_error : bool                     «optional»
# Expand = [water, air]    ¡ADD!        «optional» - NEEDS seasurface

    # # Model/grid expansion # #

    if 'expand' in cfg.sections():
        expand = {}

        all_exp = dict(cfg.items('expand'))

        # Float.
        for key in ['air', 'water', 'seasurface']:
            if cfg.has_option('expand', key):
                _ = all_exp.pop(key)
                expand[key] = cfg.getfloat('expand', key)

        # Ensure no keys are left.
        if all_exp:
            raise TypeError(f"Unexpected parameter in [expand]: "
                            f"{list(all_exp.keys())}")

        # Add to simulation dict if not empty.
        if expand:
            simulation['expand_opts'] = expand

    # Return.
    out = {'files': files, 'simulation_options': simulation, 'data': data}
    return out, term
