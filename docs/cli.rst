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
  # survey = survey.h5         # Also via --survey
  # model = model.h5           # Also via --model
  # output = emg3d_out.h5      # Also via --output
  # store_simulation = False   # Stores entire simulation in output if True

  # Simulation parameters
  # ---------------------
  # Input parameters for the `Simulation` class, except for `solver_opts` and
  # `data_weight_opts` (defined in their own section).
  [simulation]
  # max_workers = 4            # Also via -n or --nproc
  # gridding = same            # Default will change in the future

  # Solver options
  # --------------
  # emg3d-solver parameters.
  # See https://emg3d.readthedocs.io/en/stable/solver.html for a list of all
  # parameters. The only parameters that cannot be provided here are
  # grid, model, sfield, efield, and return_info.
  #
  # Note that currently sslsolver, semicoarsening, and linerelaxation only
  # accept True/False.
  [solver_opts]
  # sslsolver = True
  # semicoarsening = True
  # linerelaxation = True
  # verb = 0

  # Data weighting options
  # ----------------------
  # Data weigthing options.
  # See https://emg3d.readthedocs.io/en/stable/simulation.html for a list of
  # all parameters.
  [data_weight_opts]
  # gamma_d = 0.5
  # beta_d = 1.0
  # beta_f = 0.25
  # noise_floor = 1e-15
  # min_off = 1000
  # reference = reference
