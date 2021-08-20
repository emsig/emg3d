"""
Entry point for the command-line interface (CLI).
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
import argparse

from emg3d import utils
from emg3d.cli import run


def main(args=None):
    """Parsing command line inputs of CLI interface."""

    # If not explicitly called, catch arguments.
    if args is None:
        args = sys.argv[1:]

    # Start CLI-arg-parser and define arguments.
    parser = argparse.ArgumentParser(
        description="Multigrid solver for 3D electromagnetic diffusion."
    )

    # arg: Optional parameter-file name.
    parser.add_argument(
        "config",
        nargs="?",
        default="emg3d.cfg",
        type=str,
        help=("name of config file; default is 'emg3d.cfg'; consult "
              "https://emg3d.emsig.xyz/en/stable/manual/cli.html for "
              "its format")
    )

    # arg: Number of processors.
    parser.add_argument(
        "-n", "--nproc",
        type=int,
        default=None,
        help="number of processors"
    )

    # arg: What to run
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        "-f", "--forward",
        action='store_true',
        help="compute synthetic data (default)"
    )
    group1.add_argument(
        "-m", "--misfit",
        action='store_true',
        help="compute synthetic data and their misfit"
    )
    group1.add_argument(
        "-g", "--gradient",
        action='store_true',
        help="compute synthetic data, misfit, and its gradient"
    )

    # arg: Path to files
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="path (abs or rel); file names are relative to path"
    )

    # arg: Survey file name; relative to path
    parser.add_argument(
        "--survey",
        type=str,
        default=None,
        help="input survey file name; default is 'survey.h5'"
    )

    # arg: Model file name; relative to path
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="input model file name; default is 'model.h5'"
    )

    # arg: Output base name; relative to path
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output files base name; default is 'emg3d_out'"
    )

    # arg: Simulation file name to save; relative to path
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="file-name to save simulation; not used by default"
    )

    # arg: Simulation file name to load; relative to path
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="file-name to load simulation; not used by default"
    )

    # arg: Verbosity.
    group3 = parser.add_mutually_exclusive_group()
    group3.add_argument(
        "--verbosity",
        type=int,
        default=0,
        choices=[-1, 0, 1, 2],
        help="set verbosity; default is 0"
    )
    group3.add_argument(
        "-v", "--verbose",
        action="count",
        dest="verbosity",
        help="increase verbosity; can be used multiple times"
    )
    group3.add_argument(
        "-q", "--quiet",
        action="store_const",
        const=-1,
        dest="verbosity",
        help="decrease verbosity"
    )

    # arg: Run without emg3d-computation.
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        default=False,
        help="only display what would have been done"
    )

    # arg: Report
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="only display emg3d report"
    )

    # arg: Version
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="only display emg3d version"
    )

    # Get command line arguments.
    args_dict = vars(parser.parse_args(args))

    # Exits without simulation.
    if args_dict.pop('version'):  # emg3d version info.

        print(f"emg3d v{utils.__version__}")
        return

    elif args_dict.pop('report'):  # emg3d report.
        print(utils.Report())
        return

    elif len(sys.argv) == 1 and not os.path.isfile('emg3d.cfg'):

        # If no arguments provided, and ./emg3d.cfg does not exist, print info.
        print(parser.description)
        version = utils.__version__
        print(f"=> Type `emg3d --help` for more info (emg3d v{version}).")
        return

    # Run simulation with given command line inputs.
    run.simulation(args_dict)


if __name__ == "__main__":
    sys.exit(main())
