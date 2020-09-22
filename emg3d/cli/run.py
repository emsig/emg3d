"""
Functions that actually call emg3d within the CLI interface.
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

import json
import time
import logging

import numpy as np

from emg3d import io, utils, simulations
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
    function, verb = term['function'], term['verbosity']
    dry_run = term.get('dry_run', False)
    verb_io = max(0, verb-1)  # io-verbosity only when debug (verbosity=2).

    # Start this task: start timing.
    logger = initiate_logger(cfg, runtime, verb)

    # Log start info, python and emg3d version, and python path.
    logger.info(f"\n:: emg3d {function} START :: {time.asctime()} :: "
                f"v{utils.__version__}\n")

    # Dump the configuration.
    if not term['config_file']:
        logger.warning("* WARNING :: CONFIGURATION FILE NOT FOUND.\n")
    paramdump = json.dumps(cfg, sort_keys=True, indent=4)
    logger.debug(f"** CONFIGURATION: {term['config_file']}\n{paramdump}\n")

    # Load input.
    sdata = io.load(cfg['files']['survey'], verb=verb_io)
    mdata = io.load(cfg['files']['model'], verb=verb_io)
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

    # Initiate output dict.
    output = {}

    # Compute forward model (all calls).
    if dry_run:
        output['data'] = np.zeros_like(sim.data.synthetic)
    else:
        sim.compute()
        output['data'] = sim.data.synthetic

    # Compute the misfit.
    if function in ['misfit', 'gradient']:
        if dry_run:
            output['misfit'] = 0.0
        else:
            output['misfit'] = sim.misfit

    # Compute the gradient.
    if function == 'gradient':
        if dry_run:
            output['gradient'] = np.zeros(mdata['mesh'].vnC)
        else:
            output['gradient'] = sim.gradient

    # Add solver exit messages to log.
    if not dry_run:
        infostr = "\nSolver exit messages:\n"
        for src, values in sim._dict_efield_info.items():
            for freq, info in values.items():
                infostr += f"- Src {src}; {freq} Hz : {info['exit_message']}"
                if function == 'gradient':
                    binfo = sim._dict_efield_info[src][freq]['exit_message']
                    infostr += f"; back: : {binfo}\n"
                else:
                    infostr += "\n"
        logger.debug(infostr)

    # Store output to disk.
    if cfg['files'].get('store_simulation', False):
        output['simulation'] = sim
    io.save(cfg['files']['output'], **output, verb=verb_io)

    # Goodbye
    logger.info(f"\n:: emg3d {function} END   :: {time.asctime()} :: "
                f"runtime = {runtime.runtime}\n")


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

    return logger
