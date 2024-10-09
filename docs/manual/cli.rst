CLI interface
=============

The command-line interface can be used for certain specific tasks, such as
forward modelling and the computation of the gradient. The command is
``emg3d``, consult the inbuilt help in your terminal to get started:

.. code-block:: console

   emg3d --help

The CLI is driven by command-line parameters and a configuration file. The
default configuration filename is ``emg3d.cfg``, but another name can be
provided as first positional argument to ``emg3d``. Note that arguments
provided in the command line overwrite the settings in the configuration file.

For an example see the `CLI example
<https://emsig.xyz/emg3d-gallery/gallery/tutorials/cli.html>`_ in the gallery.

Please note that the CLI is not very well documented. If you plan to use it and
struggle with it please get in touch.


Format of the config file
-------------------------

The shown values are examples. All values are commented out in this example;
remove the comment signs to use them.

``emg3d.cfg``::

  # Files
  # -----
  # If the files are provided without ending the suffix `.h5` will be appended.
  # The log has the same name as `output`, but with the suffix `.log`.
  [files]
  # path = .               # Path (absolute or relative) to the data
  # survey = survey.h5     # Also via `--survey`
  # model = model.h5       # Also via `--model`
  # output = emg3d_out.h5  # Also via `--output`
  #
  # # You can save the entire simulation with the `save` argument.
  # save = my_sim.h5       # Also via `--save`
  #
  # # You can load an existing simulation with the `load` argument.
  # # Note that if you provide a simulation it will ignore
  # # - The survey and model under [files]
  # # - Sections [simulation], [solver_opts], [gridding_opts], and [data]
  # load = my_sim.h5       # Also via `--load`
  #
  # # Shortcut for `load` and `save` if the same filename is used for both.
  # cache = my_sim.h5    # Also via `--cache`

  # Simulation parameters
  # ---------------------
  # Input parameters for the `Simulation` class, except for `solver_opts`
  # (defined in their own section).
  [simulation]
  # max_workers = 4      # Also via `-n` or `--nproc`.
  # gridding = single
  # name = MyTestSimulation
  # file_dir = None      # For file-based comp; absolute or relative path.
  # receiver_interpolation = cubic  # Set it to <linear> for the gradient.
  # layered = False      # Also via `-l` or `--layered`.

  # Solver options
  # --------------
  # Input parameters for the solver.
  # See https://emg3d.emsig.xyz/en/stable/api/emg3d.solver.solve.html
  # for a list of all parameters. The only parameters that cannot be provided
  # here are grid, model, sfield, efield, and return_info.
  #
  # Note that currently sslsolver, semicoarsening, and linerelaxation only
  # accept True/False through the CLI.
  [solver_opts]
  # sslsolver =            # bool
  # semicoarsening =       # bool
  # linerelaxation =       # bool
  # cycle =                # string
  # tol =                  # float
  # tol_gradient =         # float
  # verb =                 # int
  # maxit =                # int
  # nu_init =              # int
  # nu_pre =               # int
  # nu_coarse =            # int
  # nu_post =              # int
  # clevel =               # int
  # plain =                # bool

  # Gridding options
  # ----------------
  # Input parameters for the automatic gridding.
  # See the description of `gridding_opts` and link therein in
  # https://emg3d.emsig.xyz/en/stable/api/emg3d.simulations.Simulation.html
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
  # domain =              # list of lists, e.g.: -10000, 10000; None; None
  # distance =            # list of lists, e.g., None; None; -10000, 10000
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

  # Noise options
  # -------------
  # Only if `--forward`, the noise options are passed to
  # `Simulation.compute(observed=True, **noise_opts)`.
  [noise_opts]
  # add_noise = True      # Set to False to switch noise off.
  # min_offset = 0.0      # off < min_off set to NaN.
  # max_offset = np.inf   # off > max_off set to NaN.
  # mean_noise = 0.0      # Mean of the noise.
  # ntype = white_noise   # Type of the noise.

  # Data
  # ----
  # Select which sources, receivers, and frequencies of the survey are used. By
  # default all data is used. These are comma-separated lists.
  [data]
  # sources = TxED-02, TxMD-08, TxEW-14
  # receivers = RxEP-01, RxMP-10
  # frequencies = f-1, f-3
  # remove_empty = False  # CLI uses False by default.

  # Layered computation
  # -------------------
  # The following parameters are only used if `-l`/`--layered` is set or the
  # simulation section has set `layered` to True.
  [layered]
  # method =               # str
  # radius =               # float
  # factor =               # float
  # minor =                # float
  # merge =                # bool
  # check_foci =           # bool
