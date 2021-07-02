"""
Functions that actually call emg3d within the CLI interface.
"""
# Copyright 2018-2021 The emsig community.
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

from emg3d import io, utils, simulations
from emg3d.cli import parser


def simulation(args_dict):
    """Run `emg3d` invoked by CLI.

    Run and log ``emg3d`` given the settings stored in the config file,
    overruled by settings passed in ``args_dict`` (which correspond to
    command-line arguments).

    Results are saved to files according to provided settings.


    Parameters
    ----------
    args_dict : dict
        Arguments from terminal, see :func:`emg3d.cli.main`. Parameters passed
        in ``args_dict`` overrule parameters in the ``config``.

    """

    # Start timer.
    runtime = utils.Timer()

    # Parse configuration file.
    cfg, term = parser.parse_config_file(args_dict)
    check_files(cfg, term)  # Check all files and directories exist.
    function, verb = term['function'], term['verbosity']
    dry_run = term.get('dry_run', False)

    # Start this task: start timing.
    logger = initiate_logger(cfg, runtime, verb)

    # Log start info, python and emg3d version, and python path.
    logger.info(f":: emg3d CLI {function} START :: {time.asctime()} :: "
                f"v{utils.__version__}")
    logger.debug(f"{utils.Report()}")

    # Dump the configuration.
    paramdump = json.dumps(cfg, sort_keys=True, indent=4)
    logger.debug("\n    :: CONFIGURATION ::\n")
    logger.debug(f"{term['config_file']}\n{paramdump}")

    min_offset = cfg['simulation_options'].pop('min_offset', 0.0)

    if cfg['files']['load']:

        # Load input.
        logger.info("\n    :: LOAD SIMULATION ::\n")

        sim, sinfo = simulations.Simulation.from_file(
                cfg['files']['load'], verb=-1)
        logger.info(sinfo.split('\n')[0])
        logger.debug(sinfo.split('\n')[1])

    else:

        # Load input.
        logger.info("\n    :: LOAD SURVEY AND MODEL ::\n")
        sdata, sinfo = io.load(cfg['files']['survey'], verb=-1)
        survey = sdata['survey']
        logger.info(sinfo.split('\n')[0])
        logger.debug(sinfo.split('\n')[1])
        model, minfo = io.load(cfg['files']['model'], verb=-1)
        logger.info(minfo.split('\n')[0])
        logger.debug(minfo.split('\n')[1])

        # Select data.
        data = cfg['data']
        if data:
            survey = survey.select(sources=data.get('sources', None),
                                   receivers=data.get('receivers', None),
                                   frequencies=data.get('frequencies', None))

        # Switch-off tqdm if verbosity is zero.
        if verb < 1:
            cfg['simulation_options']['tqdm_opts'] = {'disable': True}

        # Create simulation.
        sim = simulations.Simulation(
                survey=survey,
                model=model['model'],
                verb=-1,  # Only errors.
                **cfg['simulation_options']
        )

    # Print simulation info.
    logger.info("\n    :: SIMULATION ::")
    logger.info(f"\n{sim}\n")

    # Print meshes.
    logger.debug("    :: MESHES ::\n")
    logger.debug(sim.print_grid_info(return_info=True))

    # Initiate output dict, add configuration.
    output = {'configuration': cfg}

    # Compute forward model (all calls).
    logger.info("    :: FORWARD COMPUTATION ::\n")
    if dry_run:
        output['data'] = np.zeros(sim.survey.shape, dtype=complex)
    else:

        if function == 'forward':
            sim.compute(observed=True, min_offset=min_offset)
            output['data'] = sim.data.observed
        else:
            sim.compute()
            output['data'] = sim.data.synthetic

        # Print Solver Logs.
        if verb in [0, 1]:
            sim.print_solver_info('efield', 0)
        logger.debug(sim.print_solver_info('efield', 1, True))

    # Compute the misfit.
    if function in ['misfit', 'gradient']:
        if dry_run:
            output['misfit'] = 0.0
        else:
            output['misfit'] = sim.misfit
        output['n_observations'] = sim.survey.count

    # Compute the gradient.
    if function == 'gradient':
        logger.info("\n    :: BACKWARD COMPUTATION ::\n")

        if dry_run:
            output['gradient'] = np.zeros(sim.model.grid.shape_cells)
        else:
            output['gradient'] = sim.gradient

            # Print Solver Logs.
            if verb in [0, 1]:
                sim.print_solver_info('bfield', 0)
            logger.debug(sim.print_solver_info('bfield', 1, True))

    # Store output to disk.
    logger.info("    :: SAVE RESULTS ::\n")
    if cfg['files']['save']:
        oinfo = sim.to_file(cfg['files']['save'], verb=-1)
        logger.info(oinfo.split('\n')[0])
        logger.debug(oinfo.split('\n')[1])
    oinfo = io.save(cfg['files']['output'], **output, verb=-1)
    logger.info(oinfo.split('\n')[0])
    logger.debug(oinfo.split('\n')[1])

    # Goodbye
    logger.info(f"\n:: emg3d CLI {function} END   :: {time.asctime()} :: "
                f"runtime = {runtime.runtime}")


def check_files(cfg, term):
    """Ensure all paths and files exist."""
    error = ""

    # First check if config file exists.
    fname = term['config_file']
    if not os.path.isfile(fname) and fname != '.':  # '.' => no config file.
        error += f"* ERROR   :: Config file not found: {fname}\n"

    # Check Survey and Model.
    files = {'Survey': 'survey', 'Model': 'model', 'Simulation': 'load'}
    for key, value in files.items():
        ffile = cfg['files'][value]
        if ffile and not os.path.isfile(ffile):
            error += f"* ERROR   :: {key} file not found: {ffile}\n"

    # Finally check output directory.
    dname = os.path.split(cfg['files']['log'])[0]
    if not os.path.isdir(dname):
        error += f"* ERROR   :: Output directory does not exist: {dname}\n"
    if cfg['files']['save']:
        dname = os.path.split(cfg['files']['save'])[0]
        if not os.path.isdir(dname):
            error += f"* ERROR   :: Output directory does not exist: {dname}\n"

    # If any was not found, exit with error.
    if len(error) > 10:
        sys.exit(error[:-1])


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

    # Add handlers to Python Warnings.
    logging.captureWarnings(True)
    logger_warnings = logging.getLogger("py.warnings")
    logger_warnings.setLevel(logging.DEBUG)
    logger_warnings.addHandler(ch)
    logger_warnings.addHandler(fh)

    return logger
