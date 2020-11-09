CLI interface
#############

Command-line interface for certain specific tasks, such as forward modelling
and gradient computation of the misfit function. The command is ``emg3d``,
consult the inbuilt help to get started:

.. code-block:: console

   emg3d --help

The CLI is driven by command-line parameters and a configuration file. The
default configuration file is ``emg3d.cfg``, but another name can be provided
as first positional argument to ``emg3d``. Note that arguments provided in the
command line overwrite the settings in the configuration file.


Format of the config file
-------------------------

The shown values are the defaults. All values are commented out in this
example; remove the comment signs to use them.

::

  # Files
  # -----
  # If the files are provided without ending the suffix `.h5` will be appended.
  # The log has the same name as `output`, but with the suffix `.log`.
  [files]
  # path = .                   # Path (absolute or relative) to the data
  # survey = survey.h5         # Also via `--survey`
  # model = model.h5           # Also via `--model`
  # output = emg3d_out.h5      # Also via `--output`
  # store_simulation = False   # Stores entire simulation in output if True

  # Simulation parameters
  # ---------------------
  # Input parameters for the `Simulation` class, except for `solver_opts`
  # (defined in their own section), but including the parameter `min_offset`
  # for `compute()`.
  [simulation]
  # max_workers = 4    # Also via `-n` or `--nproc`.
  # gridding = single  # One grid for all sources and frequencies.
  # min_offset = 0.0   # Only relevant if `observed=True` (r<r_min set to NaN).

  # Solver options
  # --------------
  # Input parameters for the solver.
  # See https://emg3d.readthedocs.io/en/stable/api/emg3d.solver.solve.html
  # for a list of all parameters. The only parameters that cannot be provided
  # here are grid, model, sfield, efield, and return_info.
  #
  # Note that currently sslsolver, semicoarsening, and linerelaxation only
  # accept True/False through the CLI.
  [solver_opts]
  # sslsolver = True
  # semicoarsening = True
  # linerelaxation = True
  # verb = 0

  # Gridding options
  # ----------------
  # Input parameters for the automatic gridding.
  # See the description of `gridding_opts` and link therein in
  # https://emg3d.readthedocs.io/en/stable/api/emg3d.simulations.Simulation.html
  # for more details.
  #
  # List of lists: lists are comma-separated values, lists are separated by
  # semi-colons.
  #
  # One of the limitation of the CLI is that `vector` has to be a string.
  [gridding_opts]
  # properties =          # list, e.g.: 0.3, 1, 1e5
  # center =              # list, e.g.: 0, 0, 0
  # cell_number =         # list, e.g.: 8, 16, 32, 64, 128
  # min_width_pps =       # list, e.g.: 5, 3, 3
  # expand =              # list, e.g.: 0.3, 1e8
  # domain =              # list of lists, e.g.: -10000, 10000; None; None
  # stretching =          # list of lists, e.g.: None; None; 1.05, 1.5
  # min_width_limits =    # list of lists, e.g.: 10, 100; None; 50
  # mapping =             # string, e.g.: Resistivity
  # vector =              # string, e.g.: xy
  # frequency =           # float, e.g.: 1.0
  # seasurface =          # float, e.g.: 0.0
  # max_buffer =          # float, e.g.: 100000.0
  # lambda_factor =       # float, e.g.: 1.0
  # verb =                # int, e.g.: 0
  # lambda_from_center =  # bool, e.g.: False


  # Data
  # ----
  # Select which sources, receivers, and frequencies of the survey are used. By
  # default all data is used. These are comma-separated lists.
  [data]
  # sources = Tx02, Tx08, Tx14
  # receivers = Rx01, Rx10
  # frequencies = 0.5, 0.75
