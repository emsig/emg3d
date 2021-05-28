"""
7. Command-line interface
=========================

The implemented command-line interface (CLI) can be powerful to use `emg3d`
from any programming environment. It enables, for instance, to use `emg3d` as a
forward modelling kernel in your inversion code running on a server. The
capabilities of the CLI are more limited than of the Python interface, but they
are sufficient to compute the responses of a given `survey` for a provided
`model`, and to compute the gradient of the misfit function.

The biggest difficulty to work with the CLI is, at the moment, the file
formats, or better the lack of their documentation. Surveys, models, and grids
can be stored with `emg3d` to either h5, npz, or json files. The status of the
documentation will improve towards the v1.0.0 release of `emg3d`. The "easiest"
way at the moment is to generate such a file in Python, and reproduce its
structure in your language or program of choice. Please feel free to open an
issue on GitHub if there are questions in this regard. The idea for the future
is to go through https://github.com/softwareunderground/subsurface to interface
to various file formats.

"""
import os
import pooch
import subprocess
# sphinx_gallery_thumbnail_path = '_static/thumbs/cli.png'

# Adjust this path to a folder of your choice.
data_path = os.path.join('..', 'download', '')


###############################################################################
# Note that everything shown in this example is meant to be executed in a
# terminal, nothing is executed in Python. However, as this example gallery is
# generated in Python we have to use a work-around to show the functionality,
# the function ``bash(command)``. You would execute the provided ``command`` in
# your terminal.

def bash(command):
    "Prints the `command`, executes it in bash, and prints the output."""
    # Print command
    print(f"$ {command}")
    # Move to data_path
    command = f"cd {data_path}; " + command
    # Carry out command
    msg = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    # Print output
    print(msg.stdout.decode())


###############################################################################
# We have to fetch the example files. You can also download these manually or
# provide your own survey, model, and parameter files.
model = "GemPy-II.h5"
pooch.retrieve(
    'https://raw.github.com/emsig/data/2021-05-21/emg3d/models/'+model,
    'ea8c23be80522d3ca8f36742c93758370df89188816f50cb4e1b2a6a3012d659',
    fname=model,
    path=data_path,
)
survey = 'GemPy-II-survey-A.h5'
pooch.retrieve(
    'https://raw.github.com/emsig/data/2021-05-21/emg3d/surveys/'+survey,
    '5f2ed0b959a4f80f5378a071e6f729c6b7446898be7689ddc9bbd100a8f5bce7',
    fname=survey,
    path=data_path,
)


###############################################################################
# Getting started - help
# ----------------------
# The best way to get started is, as with any bash command, to have a look at
# its in-built help.

bash("emg3d --help")


###############################################################################
# Configuration file
# ------------------
# The CLI is driven by a configuration file, that is by default named
# ``emg3d.cfg`` and must be in the same directory as you execute the command.
# If it has a different name or a different location you have to provide the
# full or relative path and filename as the first argument to ``emg3d``.
#
# The configuration parameters are described in the documentation, consult
# `Manual -> CLI <https://emg3d.emsig.xyz/en/stable/cli.html>`_.
#
# Let's write a simple example file.
with open(f"{data_path}emg3d.cfg", "w") as f:
    f.write("""[files]
survey = GemPy-II-survey-A.h5
model = GemPy-II.h5
output = emg3d_out.json

[simulation]
name = CLI Test
gridding = single

[solver_opts]
# nothing specified

[gridding_opts]
# nothing specified

[data]
sources = TxED-1
receivers = RxEP-08, RxEP-38
frequencies = f-1""")


###############################################################################
# The file defines the location of the survey and model files, the name and
# format of the output file, and selects a source, two receivers, and a
# frequency to compute.
#
# Compute the forward responses
# -----------------------------
#
# We just compute the forward response here, but the usage for the misfit or
# the gradient is very similar. We use the most verbose version here, to see
# what it does internally.

bash("emg3d --forward -vv")


###############################################################################
# Generated files
# ---------------
#
# The run created the files ``emg3d_out.json`` (name and file-type can be
# changed in the config file), which contains the responses, and the file
# ``emg3d_out.log`` with some information.

bash("ls emg3d* -1")


###############################################################################
# Log
# ---
#
# Let's have a look at the log, which is mostly the same as was printed above
# with the verbosity flag:

bash("cat emg3d_out.log")


###############################################################################
# Result / output
# ---------------
#
# We stored the output as a json-file, just so we can easily cat it. However,
# you might want to use h5 files, as they are compressed. Note that json cannot
# store complex numbers, so the responses are stored as ``[[Re], [Im]]``.

bash("cat emg3d_out.json")


###############################################################################
# Report
# ------
# The report-command prints an extensive list of your python environment of
# components which are important for `emg3d`. If you report an issue it helps a
# lot if you provide this info.

bash("emg3d --report")
