"""
Functions that actually call emg3d within the CLI interface.
"""
# Copyright 2018-2021 The emg3d Developers.
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
import sys
import json
import time
import logging

import numpy as np

from emg3d import io, utils, surveys, simulations
from emg3d.cli import parser


def simulation(args_dict):
    """Run `emg3d` invoked by CLI.

    Run and log `emg3d` given the settings stored in the config file, overruled
    by settings passed in `args_dict` (which correspond to command-line
    arguments).

    Results are saved to files according to provided settings.


    Parameters
    ----------
    args_dict : dict
        Arguments from terminal, see :func:`emg3d.cli.main`. Parameters passed
        in `args_dict` overrule parameters in the `config`.

    """

    # Start timer.
    runtime = utils.Time()

    # Parse configuration file.
    cfg, term = parser.parse_config_file(args_dict)
    check_files(cfg, term)  # Check all files and directories exist.
    function, verb = term['function'], term['verbosity']
    dry_run = term.get('dry_run', False)
    verb_io = max(0, verb-1)  # io-verbosity only when debug (verbosity=2).

    # Start this task: start timing.
    logger = initiate_logger(cfg, runtime, verb)

    # Log start info, python and emg3d version, and python path.
    logger.info(f"\n:: emg3d {function} START :: {time.asctime()} :: "
                f"v{utils.__version__}\n")

    # Dump the configuration.
    paramdump = json.dumps(cfg, sort_keys=True, indent=4)
    logger.debug(f"** CONFIGURATION: {term['config_file']}\n{paramdump}\n")

    # Load input.
    sdata, sinfo = io.load(cfg['files']['survey'], verb=-1)
    logger.debug(sinfo)
    mdata, minfo = io.load(cfg['files']['model'], verb=-1)
    logger.debug(minfo)
    min_offset = cfg['simulation_options'].pop('min_offset', 0.0)

    # Select data.
    data = cfg['data']
    if data:

        # Get a dict.
        tdata = sdata['survey']
        tdict = tdata.to_dict()

        # Select sources.
        if 'sources' in data.keys():
            tdata._data = tdata.data.sel(src=data['sources'])
            tdict['sources'] = {
                    k: tdict['sources'][k] for k in data['sources']}

        # Select receivers.
        if 'receivers' in data.keys():
            tdata._data = tdata.data.sel(rec=data['receivers'])
            tdict['receivers'] = {
                    k: tdict['receivers'][k] for k in data['receivers']}

        # Select frequencies.
        if 'frequencies' in data.keys():
            tdata._data = tdata.data.sel(freq=data['frequencies'])
            tdict['frequencies'] = data['frequencies']

        # Replace with selected data.
        for key in tdict['data'].keys():
            tdict['data'][key] = tdata.data[key].data

        # Get new survey from reduced dict.
        sdata['survey'] = surveys.Survey.from_dict(tdict)

    # Create simulation.
    sim = simulations.Simulation(
            survey=sdata['survey'],
            grid=mdata['mesh'],
            model=mdata['model'],
            verb=verb,
            **cfg['simulation_options']
            )

    # Switch-off tqdm if verbosity is zero.
    if verb < 1:
        sim._tqdm_opts['disable'] = True

    # Print simulation info.
    logger.info(f"\n{sim}\n")

    # Print meshes.
    logger.debug(sim.print_grids)

    # Initiate output dict.
    output = {}

    # Compute forward model (all calls).
    if dry_run:
        output['data'] = np.zeros_like(sim.data.synthetic)
    elif function == 'forward':
        sim.compute(observed=True, min_offset=min_offset)
        output['data'] = sim.data.observed
    else:
        sim.compute()
        output['data'] = sim.data.synthetic

    # Compute the misfit.
    if function in ['misfit', 'gradient']:
        if dry_run:
            output['misfit'] = 0.0
        else:
            output['misfit'] = sim.misfit
        output['n_observations'] = sim.survey.size

    # Compute the gradient.
    if function == 'gradient':
        if dry_run:
            output['gradient'] = np.zeros(mdata['mesh'].vnC)
        else:
            output['gradient'] = sim.gradient

    # Add solver exit messages to log.
    if not dry_run:
        ## TODO - This is now double, in Simulation and here.               TODO
        ## TODO - Get Simulation ready for logging and remove from here.    TODO
        infostr = "\n** SOLVER LOGS:\n\n"
        for src, values in sim._dict_efield_info.items():
            for freq, finfo in values.items():
                if finfo['log']:
                    infostr += f"= Src {src}; {freq} Hz : FORWARD =\n"
                    infostr += finfo['log']
                if function == 'gradient':
                    binfo = sim._dict_bfield_info[src][freq]
                    if binfo['log']:
                        infostr += f"= Src {src}; {freq} Hz : BACK    =\n"
                        infostr += binfo['log']
        if len(infostr) > 30:
            logger.debug(infostr)

    # Store output to disk.
    if cfg['files']['store_simulation']:
        output['simulation'] = sim
    oinfo = io.save(cfg['files']['output'], **output, verb=-1)
    logger.debug(oinfo)

    # Goodbye
    logger.info(f"\n:: emg3d {function} END   :: {time.asctime()} :: "
                f"runtime = {runtime.runtime}\n")


def check_files(cfg, term):
    """Ensure all paths and files exist."""

    # First check if config file exists.
    fname = term['config_file']
    if not os.path.isfile(fname) and fname != '.':  # '.' => no config file.
        sys.exit(f"* ERROR   :: Config file not found: {fname}")

    # Check Survey and Model.
    for name in ['Survey', 'Model']:
        fname = cfg['files'][name.lower()]
        if not os.path.isfile(fname):
            sys.exit(f"* ERROR   :: {name} file not found: {fname}")

    # Finally check output directory.
    dname = os.path.split(cfg['files']['log'])[0]
    if not os.path.isdir(dname):
        sys.exit(f"* ERROR   :: Output directory does not exist: {dname}")


def initiate_logger(cfg, runtime, verb):
    """Initiate logger for CLI of emg3d."""

    # Get logger of emg3d.cli.run and add handles.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove the corresponding handlers if they exist already
    # (e.g., consecutive runs in IPython).
    for h in logger.handlers[:]:
        if h.name in ['emg3d_fh', 'emg3d_ch']:
            logger.removeHandler(h)
        h.close()

    # Create file handler; logs everything.
    fh = logging.FileHandler(f"{cfg['files']['log']}", mode='w')
    fh.setLevel(logging.DEBUG)
    fh_format = logging.Formatter('{message}', style='{')
    fh.setFormatter(fh_format)
    fh.set_name('emg3d_fh')  # Add name to easy remove them.
    logger.addHandler(fh)

    # Create console handler.
    ch = logging.StreamHandler()
    ch.setLevel([40, 30, 20, 10][verb+1])
    ch_format = logging.Formatter('{message}', style='{')
    ch.setFormatter(ch_format)
    ch.set_name('emg3d_ch')  # Add name to easy remove them.
    logger.addHandler(ch)

    # Add handlers to Python Warnings. ## TODO test this!                   TODO
    logging.captureWarnings(True)
    logger_warnings = logging.getLogger("py.warnings")
    logger_warnings.setLevel(logging.DEBUG)
    logger_warnings.addHandler(ch)
    logger_warnings.addHandler(fh)

    return logger
