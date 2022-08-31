Changelog
#########


1.x-Series
""""""""""


latest
------

- Model instances have a new attribute ``exctract_1d``, which returns a layered
  (1D) model, extracted from the 3D model according the provided parameters;
  see :attr:`emg3d.models.Model.extract_1d`.

- CLI takes new the boolean ``add_noise`` in the section ``[noise_opts]``
  (default is True).

- Maps: New function ``ellipse_indices`` returning a boolean indicating which
  points fall within a general ellipse for the provided input parameters.

- Bug fixes, small improvements and maintenance

  - Simulation.misfit returns an ndarray again instead of an DataArray (was
    erroneously changed in v1.2.1).
  - Write json can now handle NumPy int/bool/float.
  - A clean on a Simulation now removes correctly the weights.
  - Capture error in jtvec if weight is complex NaN (should be real).
  - Model: ``mapping`` can now be an already instantiated map (sort of
    undocumented).
  - Cleaned-up the namespace by setting ``dir()`` explicitly.
  - Replace ``pytest-flake8`` by plain ``flake8``.
  - Moved all multiprocessing-related functions to ``_multiprocessing.py``.


v1.7.1 : Bugfix trimmed z-vector
--------------------------------

**2022-08-02**

- Meshing: Small fix to the automatic gridding from v1.5.0 (non-backwards
  compatible). A provided z-vector is new trimmed to the domain before the
  domain might be expanded due to the provided seasurface (which is as it was
  always intended, but not as it was implemented).
- Few small maintenance things in the meta files.


v1.7.0 : CLI-clean
------------------

**2022-05-21**

- CLI:

  - New command-line argument ``--clean``: If an existing simulation is loaded,
    setting clean will remove any existing computed data (fields, misfit,
    gradient, residuals, synthetic data) and replace the model with the
    currently provided one.
  - New command-line argument ``--cache`` (or as parameter ``cache`` in the
    configuration file under ``[files]``): Acts as a shortcut for ``--load
    --save`` using the same file name.
  - Parameters for noise generation should new be provided under their own
    section ``[noise_opts]``; providing them under ``[simulation]`` is
    deprecated and will be removed in v1.9.0.

- Simulation:

  - ``'all'`` is now the same as ``'computed'`` in ``to_file`` and ``to_dict``,
    meaning the grids are stored as well.
  - Deprecation: The ``'expand'``-functionality in the gridding options is
    deprecated and will be removed in v1.9.0. A property-complete model has to
    be provided.

- Meshes: Bumped the change of the default value for ``center_on_edge`` from
  ``True`` to ``False`` to v1.9.0, coinciding with the above deprecations.


v1.6.1 : Max offset
-------------------

**2022-05-11**

- Survey: ``add_noise`` takes new a ``max_offset`` argument; receivers
  responses at offsets greater than maximum offset are set to NaN (also
  available through the CLI).


v1.6.0 : Anisotropic gradient
-----------------------------

**2022-04-30**

- Simulation: ``gradient``, ``jvec``, and ``jtvec`` new support triaxial
  anisotropy (also through the CLI). As a consequence, ``gradient`` and
  ``jtvec`` return an ndarray of shape ``(nx, ny, nz)`` (isotropic) or
  ``({2;3}, nx, ny, nz)`` (VTI/HTI; triaxial), and ``jvec`` expects an ndarray
  of shape ``(nx, ny, nz)`` (isotropic) or ``({1;2;3}, nx, ny, nz)``
  (isotropic; VTI/HTI; triaxial).


v1.5.0 : Meshing: center on edge
--------------------------------

**2022-03-30**

- Meshes:

  - ``construct_mesh`` and ``origin_and_widths`` take a new variable
    ``center_on_edge``: If ``True``, the center is put on an edge, if
    ``False``, it is put at the cell center. Status quo is ``True``, but the
    **default will change** to ``False`` in v1.7.0. If not set, it will
    currently raise a FutureWarning making the user aware of the change.
    Setting ``center_on_edge`` explicitly will suppress the warning.
  - Constructed grids through ``construct_mesh`` and ``origin_and_widths`` with
    a defined ``seasurface`` might slightly change due to some improvements and
    refactoring in the course of the above changes to the center. The changes
    should not be severe.

- Simulation:

  - ``gradient``: Changed slightly to use the proper adjoint (changed *only if*
    the computational grids differ from the inversion grid; requires
    ``discretize``).
  - ``jvec``: Adjusted to work for any mapping, not only conductivity, and also
    with adaptive gridding. It expects new a Fortran-ordered vector with the
    shape of the model (or a vector of that size).
    Gently reminder that the functions ``gradient``, ``jvec``, and ``jtvec``
    are still considered *experimental*, and might change.
  - New optional keyword ``tqdm_opts``. With ``False`` you can switch off the
    progress bars. Alternatively one can provide a dict, which is forwarded
    to ``tqdm``.

- CLI:

  - Expose ``mean_noise`` and ``ntype``, in addition to ``min_offset``, to the
    CLI (for adding noise); also ``plain`` (for solver), and ``center_on_edge``
    (for gridding options).


v1.4.0 : Meshing: improve vector
--------------------------------

**2022-02-09**

- Meshes: Non-backwards compatible changes in ``construct_mesh``
  (``origin_and_widths``; ``estimate_gridding_options``) when providing
  ``vector``'s (implemented non-backwards compatible as the old rules were not
  intuitive nor logic; previous meshes can still be obtained, mostly, by
  setting the parameters carefully).

  - Priority-order changed to ``domain > distance > vector`` (before it was
    ``domain > vector > distance``).
  - A provided ``vector`` is new trimmed to the corresponding domain if it is
    larger than a also provided domain (from ``domain`` or ``distance``);
    trimmed at the first point where
    ``vector <= domain[0]``, ``vector >= domain[1]``.
  - A ``vector`` can new also be smaller than the defined domain, and the
    domain is then filled according to the normal rules; the last cell of
    ``vector`` in each direction is taken as starting width for the expansion.

- Bugfixes and maintenance:

  - Removed functions and modules that were deprecated in v1.2.1.
  - Fixed kwargs-error when adding ``add_noise`` explicitly to
    ``Simulation.compute()``.
  - Python 3.10 added to tests; Python 3.7 tests reduced to minimum.


v1.3.2 : Bugfix CLI-select
--------------------------

**2021-12-01**

CLI: Add ``remove_empty`` to parameter file; set to ``False`` by default
(pre-v1.3.1 behaviour, and therefore backwards compatible).


v1.3.1 : Select: remove empty pairs
-----------------------------------

**2021-11-20**

- ``Survey.select`` removes now empty source-receiver-frequency pairs. If you
  want the old behaviour set ``remove_empty=False``.

- Maintenance: Added a cron to GHA; 20th of every month at 14:14.


v1.3.0 : File-based computations
--------------------------------

**2021-10-27**

- ``electrodes``:

  - New source ``TxMagneticPoint`` (requires ``discretize``; mainly used as
    adjoint source for magnetic receivers; does not work in the presence of
    magnetic permeabilities in the vicinity of the source).
  - Both receivers (``Rx{Electric;Magnetic}Point``) can now produce their
    proper adjoint (thanks to @sgkang!).

- Changes in Simulation and parallel execution.

  - Parallel computation is not sharing the simulation any longer.
  - Parallel computation can new be done both file-based or all in memory.
    The new possibility for file-based computation should make it possible
    to compute responses for any amount of source-frequency pairs. See
    parameter ``file_dir`` in the Simulation class (or corresponding parameter
    in the CLI parameter file).
  - ``get_model`` and ``get_hfield`` are now done on the fly, they are not
    stored in a dict; ``simulation._dict_model`` and
    ``simulation._dict_hfield`` do not exist any longer.
  - New methods ``jvec`` (sensitivity times a vector) and ``jtvec``
    (sensitivity transpose times a vector). These methods are currently
    experimental; documentation and examples are lacking behind.

- Various small things:

  - Models and Fields return itself (not a copy) when the grid provided to
    ``interpolate_to_grid`` is the same as the current one.



v1.2.1 : Remove optimize & bug fix
----------------------------------

**2021-08-22**

- ``io``: Adjustment so that hdf5 tracks the order of dicts.

- ``simulations``:

  - Adjust printing: correct simulation results for adjusted solver printing
    levels; **default solver verbosity is new 1**; ``log`` can now be
    overwritten in ``solver_opts`` (mainly for debugging).

  - Functions moved out of ``simulations``: ``expand_grid_model`` moved to
    ``models`` and ``estimate_gridding_options`` to ``meshes``. The
    availability of these functions through ``simulations`` will be removed in
    v1.4.0.

- ``optimize``: the module is deprecated and will be removed in v1.4.0. The two
  functions ``optimize.{misfit;gradient}`` are embedded directly in
  ``Simulation.{misfit;gradient}``.


v1.2.0 : White noise
--------------------

**2021-07-27**

- CLI:

  - New parameters ``save`` and ``load`` to save and load an entire simulation.
    In the parameter file, they are under ``[files]``; on the command line,
    they are available as ``--save`` and ``--load``; they are followed by the
    filename including its path and suffix. (In turn, the parameter
    ``store_simulation`` was removed.)

- ``simulations.Simulation``:

  - Warns if the gradient is called, but ``receiver_interpolation`` is not
    ``'linear'``.
  - Slightly changed the added noise in ``compute(observed=True)``: It uses new
    the ``survey.add_noise`` attribute. There is new a flag to set if noise
    should be added or not (``add_noise``), and if the amplitudes should be
    chopped or not (``min_amplitude``). Also note that the added noise is new
    white noise with constant amplitude and random phase.

- ``surveys``:

  - New function ``random_noise``, which can be used to create random noise in
    different ways. The default noise is white noise, hence constant amplitude
    with random phase. (This is different to before, where random Gaussian
    noise was added separately to the real and imaginary part.) For the random
    noise it requires new at least NumPy 1.17.0.

  - New attribute ``Survey.add_noise``, which uses under the hood above
    function.

  - A ``Survey`` can new be instantiated without receivers by setting
    ``receivers`` to ``None``. This is useful if one is only interested in
    forward modelling the entire fields. In this case, the related data object
    and the noise floor and relative error have no meaning. Also, in
    conjunction with a Simulation, the misfit and the gradient will be zero.

- Various:

  - All emg3d-warnings (not solver warnings) are now set to ``'always'``, and
    corresponding print statements were removed.
  - Simplified (unified) ``_edge_curl_factor`` (private fct).


v1.1.0 : Adjoint-fix for electric receivers
-------------------------------------------

**2021-06-30**

This release contains, besides the usual small bugfixes, typos, and small
improvements, an important fix for ``optimize.gradient``. Keep in mind that
while the forward modelling is regarded as stable, the ``optimize`` module is
still work in progress.

The fixes with regard to ``optimize.gradient`` ensure that the gradient is
indeed using the proper adjoint to back-propagate the field. This is currently
*only* given for electric receivers, not yet for magnetic receivers. These
improvement happened mainly thanks to the help of Seogi (@sgkang).

The changes in more detail:

- ``fields``:

  - ``get_receiver`` has a new keyword ``method``, which can be ``'cubic'`` or
    ``'linear'``; default is the former, which is the same behaviour as before.
    However, if you want to compute the gradient, you should set it to
    ``'linear'`` in your Simulation parameters. Otherwise the adjoint-state
    gradient will not exactly be the adjoint state.
  - ``get_source_field`` returns new the real-valued, frequency-independent
    source vector if ``frequency=None``.
  - ``get_source_field`` uses the adjoint of trilinear interpolation for point
    sources (new). For dipoles and wires it the source is distributed onto the
    cells as fraction of the source length (as before).

- ``electrodes``: Re-introduced the point source as ``TxElectricPoint``.

- ``simulations.Simulation``:

  - New keyword ``receiver_interpolation``, which corresponds to the ``method``
    in ``get_receiver`` (see above). Cubic is more precise. However, if you are
    interested in the gradient, you need to choose linear interpolation at the
    moment, as the point source is the adjoint of linear interpolation. To be
    the proper adjoint for the gradient the receiver has to be interpolated
    linearly too.
  - If ``gridding`` is ``'same'`` or ``'input'``, it checks now if the provided
    grid is a sensible grid for emg3d; if not, it throws a warning.

- ``meshes``: New function ``check_grid`` to verify if a given grid is good for
  emg3d.

- ``optimize.gradient``: Changed order when going from computational grid to
  inversion grid. Changing the grids at the field stage (cubic interpolation)
  seems to be better than changing at the cell-averaged stage::

      New: field_comp -> field_inv -> cells_inv
      Old: field_comp -> cells_comp -> cells_inv

- ``cli``: Uses now by default linear receiver interpolation if the
  ``gradient`` is wanted (new), otherwise it uses cubic interpolation (as
  before). The new keyword ``receiver_interpolation`` of the simulation can be
  set in the parameter file, which overwrites the described default behaviour.


v1.0.0 : Stable API
-------------------

**2021-05-28**

Here it is, three months of hard labour lead to v1.0.0!

There are _many_ changes, and they are listed below for each module.

*Your existing code will break, and I apologize for it. Please do not hesitate
to get in touch if you have troubles updating your code.*

**API**: With version 1.0 the API becomes stable and you can expect that your
code will work fine for the duration of ``emg3d v1.x``.

- Removed all deprecated features.
- Reduced top namespace to principal functions; ``get_receiver`` is not in the
  top namespace any longer. It is advised to use directly the field method:
  ``field.get_receiver``.
- Moved emsig.github.io to emsig.xyz and emsig.readthedocs.io to
  emg3d.emsig.xyz.
- Changed principal repo branch from ``master`` to ``main``.


Detailed changes by module
''''''''''''''''''''''''''


**CLI**

- Because frequencies are now dicts as well in a Survey they have to be named
  by their key instead of their value when selecting data in the parameter
  file.
- Entire configuration is now added to the log file.


**Core**

- ``restrict_weights``: New signature.


**Electrodes**

- New module containing all sources and receivers. Currently implemented are
  ``TxElectricDipole``, ``TxMagneticDipole``, ``TxElectricWire``,
  ``RxElectricPoint``, and ``RxMagneticPoint``.
- New class ``TxElectricWire`` for an arbitrary electric wire.
- Receivers can be defined in absolute coordinates, or in coordinates relative
  to source position if they move with the source. Latter makes only sense
  within a Survey/Simulation.
- ``dip`` is new called ``elevation`` to make it clear that it is the angle
  positive upwards (anticlockwise from the horizontal plane).
- Bugfix of the loop area for a magnetic dipole (the area was previously wrong 
  except for dipoles of length of 1).
- Zero source strength does no longer mean "normalized", it means zero
  strength (hence no source).
- Besides the sources and receivers it contains utilities how to move
  electrodes in the coordinate system (e.g., ``rotation``).


**Fields**

- ``fields.Field``:

  - Is *not* a subclassed ndarray any longer; with all its advantages and
    disadvantages. E.g., operations on ``Field`` are not possible any longer
    and have to be carried out on ``Field.field``. However, it should be easier
    to maintain and expand in the future.
  - New signature.
  - Knows new its ``grid``. As a consequence, all functions that required
    previously the ``grid`` and the ``field`` require new only the ``field``;
    e.g., ``emg3d.fields.get_receiver``.
  - Has no property ``ensure_pec`` any longer, it is ensured directly in
    ``solver.prolongation``.
  - Has new the methods ``interpolate_to_grid`` and ``get_receiver``.

- Renamed parameters in all functions:

  - ``src`` to ``source``;
  - ``freq`` to ``frequency``;
  - ``rec`` to ``receiver``.

- Removed functions and classes:

  - ``SourceField``; it is just a regular ``Field`` now;
  - ``get_receiver`` (the name still exists, but it is now what was before
    ``fields.get_receiver_response``).

- Renamed functions and classes (both do not take a ``grid`` any longer):

  - ``get_h_field`` to ``get_magnetic_field``;
  - ``fields.get_receiver_response`` to ``fields.get_receiver``.


**I/O**

- ``Model``, ``Field``, ``Survey``, and ``Simulation`` instances saved with an
  older version of emg3d will not be able to de-serialize with version 1.0. You
  have to update those files, see this gist:
  https://gist.github.com/prisae/8345c3798e35f1c73efef617ac495538


**Maps**

- Changed function and class names:

  - ``_Map`` to ``BaseMap``;
  - ``grid2grid`` to ``interpolate`` (new signature);
  - ``edges2cellaverages`` to ``interp_edges_to_vol_averages`` (new signature);
  - ``volume_average`` to ``interp_volume_average`` (new signature);
  - ``interp3d`` to ``interp_spline_3d`` (new signature).

- ``maps.interpolate``:

  - Can now be used to interpolate values living on a grid to another grid or
    to points defined either by a tuple or by an ndarray.
  - The implemented interpolation methods are 'nearest' (new), 'linear',
    'cubic', and 'volume'. Volume averaging ('volume') only works for
    grid-to-grid interpolations, not for grid-to-points interpolations.
  - Does not accept entire fields any longer. Entire fields can be mapped with
    their own ``field.interpolate_to_grid`` method.

- Maps cannot be (de-)serialized any longer (``{to;from_dict}``); simply store
  its name, which can be provided to ``models.Model``.

- Function ``rotation`` should be used for anything involving angles to use
  the defined coordinate system consistently.


**Meshes**

- Changed function and class names:

  - ``_TensorMesh`` to ``BaseMesh``;
  - ``min_cell_width`` to ``cell_width``.
  - ``get_origin_widths`` to ``origin_and_widths`` (has new finer loops to fine
    grid sizes than before).

- ``meshes.BaseMesh``:

  - Reduced to the attributes ``origin``, ``h``, ``shape_{cells;nodes}``,
    ``n_{cells;edges;faces}``, ``n_{edges;faces}_{x;y;z}``,
    ``{nodes;cell_centers}_{x;y;z}``, ``shape_{edges;faces}_{x;y;z}``, and
    ``cell_volumes``. These are the only required attributes for ``emg3d``.

- ``meshes.construct_mesh``: ``domain``, ``vector``, ``distance``,
  ``stretching``, ``min_width_limits``, and ``min_width_pps`` can now also
  be provided as a dict containing the three keys ``'{x;y;z}'``.

- ``meshes.skin_depth`` takes new ``mu_r`` instead of ``mu``.

- ``good_mg_cell_nr``: ``max_prime`` is new ``max_lowest``, as it could also
  be, e.g., 9, which is not a prime.


**Models**

- ``models.Model``:

  - Knows new its ``grid``. As a consequence, all the functions that used to
    require the ``grid`` and the ``model`` require new only the ``model``;
    e.g., ``emg3d.solver.solve`` or ``emg3d.fields.get_magnetic_field``.

  - If ``property_y`` or ``property_z`` are not set they return now ``None``,
    not ``property_x``.

  - If a float is provided for a property it is new expanded to the shape of
    the model, and not kept as a float.

  - Has to be initiated with all desired properties; it cannot be changed
    afterwards. E.g., if it was initiated without electric permittivity, it
    cannot be added afterwards. However, it can be initiated with dummy values
    and adjusted later.

  - Renamed ``interpolate2grid`` to ``interpolate_to_grid``.

- ``models.VolumeModel``: Does not take a ``grid`` any longer.


**Simulations**

- ``Simulation``:

  - Works new for electric and magnetic dipole sources as well as electric wire
    sources; electric and magnetic point receivers.
  - Works now as well for surveys that contain receivers which are positioned
    relatively to the source.
  - New signature: no ``grid`` any longer, ``name`` is new an optional keyword
    parameter, new optional keyword parameter ``info``.
  - Method ``get_sfield`` is removed.

- ``expand_grid_model`` and ``estimate_gridding_opts`` have new signatures and
  do not take a ``grid`` any longer.


**Solver**

- ``solver.solve``:

  - New signature: no ``grid`` any longer; ``efield`` and ``cycle`` are moved
    to keyword arguments.

  - The defaults for ``sslsolver``, ``semicoarsening``, and ``linerelaxation``
    is new ``True`` (before it was ``False``). This is not necessarily the
    fastest setting, but generally the most robust setting.

  - New keyword parameter ``plain``, which is by default ``False``. If it is
    set to ``True`` it uses plain multigrid, hence ``sslsolver=False``,
    ``semicoarsening=False``, and ``linerelaxation=False``, unless these
    parameters were set to anything different than ``True``.

  - Some verbosity levels changed (for consistency reasons throughout emg3d).
    The new levels are [old levels in brackets]:

    - -1: Nothing [0]
    - 0: Warnings [1]
    - 1: One-liner at the end [2]
    - 2: One-liner (dynamically updated) [-1]
    - 3: Runtime and information about the method [same]
    - 4: Additional information for each MG-cycle [same]
    - 5: Everything (slower due to additional error computations) [same]

    Level three updates now dynamically, just as level 2.

- ``solve_source()``: New function, a shortcut for ``solve()``. It takes a
  ``source`` and a ``frequency`` instead of a ``sfield``, gets the ``sfield``
  internally, and forwards everything to ``solver.solve``.

- ``multigrid``, ``krylov``, ``smoothing``, ``restriction``, ``prolongation``,
  ``residual``, ``RegularGridProlongator``: New signature, mainly not taking a
  ``grid`` any longer.


**Surveys**

- ``Survey``:

  - ``frequencies`` is new a dict just like ``sources`` and ``receivers``.
  - ``sources`` and ``receivers`` must be tuples or dicts; lists are no longer
    permitted. For this, the module ``surveys``  has new convenience functions
    ``txrx_coordinates_to_dict`` and ``txrx_lists_to_dict``.
  - Has no attribute ``observed`` any longer; access it just like any other
    data through ``Survey.data.observed``.
  - ``rec_coords`` and ``src_coords`` attributes changed to the methods
    ``receiver_coordinates`` and ``source_coordinates``.
    ``receiver_coordinates`` takes an optional source key.
    For relatively located receivers, it returns by default all positions of
    this receiver for all source position. If a source-key is provided it only
    returns the receiver position for this source. This does not affect
    absolutely positioned receivers.
  - Has no attribute ``rec_types`` any longer.
  - ``name`` is new optional.
  - New optional keywords ``date`` and ``info``.
  - ``noise_floor`` and ``relative_error`` are new stored as data array if they
    are not floats.
  - The keyword ``fixed`` has been dropped. To simulate fixed surveys define
    the receivers with a relative offset to the source, instead of absolute
    coordinates.
  - ``data`` can be a dict containing many data set.
  - Automatic key names start now with 1 and have a hyphen between the prefix
    and the number; they also contain the abbreviated electrode name. E.g.,
    ``Tx0`` becomes ``TxED-1`` or ``TxMD-1`` or ``TxEW-1``. Similar, ``Rx9``
    becomes ``RxEP-10`` or ``RxMp-10``, and ``f0`` becomes ``f-1``.
  - ``Survey.size`` is now the total number, ``Survey.count`` is the count of
    the data that actually has non-NaN values.
  - Now completely functional for receivers which are positioned relatively to
    the source.

- New functions ``txrx_coordinates_to_dict`` and ``txrx_lists_to_dict`` to
  collocate many sources or receivers into dicts (also
  ``frequencies_to_dict``).

- ``Dipole``: Replaced by the new source and receiver classes in the new module
  ``electrodes``.

**Time**

- Moved ``Fourier`` from ``emg3d.utils`` to its own module ``emg3d.time``.

- Renamed parameters:

  - ``freq_req`` to ``freq_required``;
  - ``freq_calc`` to ``freq_compute``;
  - ``freq_calc_i`` to ``ifreq_compute``;
  - ``freq_inp`` to ``input_freq``;
  - ``freq_extrapolate_i`` to ``ifreq_extrapolate``;
  - ``freq_interpolate_i`` to ``ifreq_interpolate``;


**Utils**

- Renamed ``Time`` to ``Timer``.
- Moved ``Fourier`` to its own module ``emg3d.time.Fourier``.
- ``_process_map`` new avoids ``concurrent.futures`` if ``max_workers<2``.



0.x-Series
""""""""""


v0.17.0 : Magnetics in Simulation
---------------------------------

**2021-03-03**

- ``Simulation``:

  - Sources and receivers can now be magnetic, also for the adjoint-state
    gradient (unit loops, not yet arbitrarily loops).

- ``fields.get_source_field``:

  - The recommended way to use ``get_source_field`` is new to provide a
    ``Tx*``-source instance.
  - The ``msrc`` argument introduced in v0.16.0 is renamed to ``electric``, and
    has the opposite meaning. If True, the source is electric, if False, the
    source is magnetic. This was made to streamline the meaning with the
    meaning given in ``surveys.Dipole``. The old parameter ``msrc`` is
    deprecated and will be removed. Warning, if ``msrc`` was provided as
    positional argument instead of as keyword argument it will now be taken as
    ``electric``, with the opposite meaning (**backwards incompatible**).
  - The magnetic source was corrected and has the opposite sign now (factor -1;
    **backwards incompatible**).

- Bug fixes:

  - Simulation: Stop overwriting synthetic data if provided in the survey to a
    simulation.
  - CLI: Removed configuration info from output data; caused problems when
    storing to h5. This has to be resolved with properly addressing the io
    stuff. Currently only stores the data selection to output data.


v0.16.1 : Verbosity & Logging
-----------------------------

**2021-02-09**

- ``Solve`` has a new keyword ``log``, which enables to log the solver messages
  in the returned info dictionary instead of printing them to screen. This is
  utilized in the CLI and in the ``Simulation`` class to log the solver info.

- ``Survey`` has a new attribute ``select``, which returns a reduced survey
  containing the selected sources, receivers, and frequencies.

- CLI:

  - Configuration info is added to output data.
  - Checks now first if all required files and directories exist, and exits
    gracefully otherwise informing the user. (The default thrown Python errors
    would be good enough; but user of the CLI interface might not be familiar
    with Python, so it is better to throw a very simple, clear message.)
  - Log is more verbose with regards to solver (rel. error, time, nr of it.).

- ``Dipole`` throws new an error instead of a warning if it received an unknown
  keyword.

- Various small things with regard to how things are logged or shown on screen.

- Changed all ``DeprecationWarnings`` to ``FutureWarnings``, meaning they will
  be removed in the next release.

- Bug fix with regards to data selection in the CLI; moved to ``Survey`` (see
  above).


v0.16.0 : Arbitrarily shaped sources
------------------------------------

**2021-01-13**

- ``fields.get_source_field``:

  - Arbitrarily shaped sources (and therefore also loops) can now be created by
    providing a ``src`` that consists of x-, y-, and z-coordinates of all
    endpoints of the individual segments.

  - Simple "magnetic dipole" sources can now be created by providing a point
    dipole (``[x, y, z, azm, dip]``) and set ``msrc=True``. This will create a
    square loop of ``length``x``length`` m perpendicular to the defined point
    dipole, hence simulating a magnetic source. Default length is 1 meter.

  - Point dipoles and finite length dipoles were before treated differently.
    Point dipoles are new converted into finite length dipoles of provided
    length (default is 1 meter), and treated as finite length dipoles. This is
    backwards incompatible and means that the source field for point dipoles
    might not be exactly the same as before. However, in any properly set-up
    simulation this should have no influence on the result.

  - Bugfix: Fix floating point issue when the smaller coordinate of a finite
    length dipole source was very close to a node, but not exactly. This is
    done by rounding the grid locations and source position, and the precision
    can be controlled via ``decimals``; default is micrometer.

- ``fields``: Values outside the grid in ``get_receiver`` and
  ``get_receiver_response`` are new set to NaN's instead of zeroes.
  Additionally, the first and last values in each direction of the fields are
  ignored, to avoid effects form the boundary condition (receivers should not
  be placed that close to the boundary anyway).

- ``simulations``:

  - Within the automatic gridding the ``properties`` are estimated much more
    conservative now, if not provided: before the log10-average of the last
    slice in a given direction was used; now it uses the maximum resistivity.
    This is usually the air value for x/y and positive z. This is very
    conservative, but avoids that users use too small computational domains in
    the case of land and shallow marine surveys. The downside is that it
    heavily over-estimates the required domain in the deep marine case.
    However, slower but safe is better in this case.
  - New method ``print_grids``, which prints the info of all created grids.
    This is also used for logging in the CLI interface.

- ``maps``: ``interp3d`` takes a new keyword ``cval``, which is passed to
  ``map_coordinates``.


v0.15.3 : Move to EMSiG
-----------------------

**2020-12-09**

Various small things, mostly related to the automatic meshing.

- New parameter ``distance`` for ``get_origin_widths``, as an alternative for
  ``domain`` and ``vector``: distance defines the survey domain as distance
  from the center. This is then also available in ``construct_mesh`` and
  ``Simulation``, including the CLI.
- Removed ``precision`` from ``skin_depth``, ``wavelength``,
  ``min_cell_width``; all in ``meshes``. It caused problems for high
  frequencies.
- All data is stored in the ``Survey``, not partly in ``Survey`` and partly
  in ``Simulation``.
- Deprecated ``collect_classes`` in ``io``.
- Expanded the ``what``-parameter in the ``Simulation``-class to include
  properties related to the gradient.
- Moved from github.com/empymod to github.com/emsig.


*v0.15.2* : Bugfix deploy II
----------------------------

**2020-12-04**

- Fixing deploy script with GHA.


*v0.15.1* : Bugfix deploy
-------------------------

**2020-12-04**


Small bugfix release, as ``v0.15.0`` never got deployed.

- Fix CI deploy script.
- Makefile for the most common dev-tasks.


*v0.15.0* : discretize restructure
----------------------------------

**2020-12-04**


The package discretize went through a major restructuring with many name
changes and consequent deprecations (see below for a list of affected
mesh-properties for ``emg3d``). This version updates ``emg3d`` to be compatible
with ``discretize>=0.6.0`` in the long run. It also means that emg3d will, from
``emg3d>=0.15.0`` onwards, only work with ``discretize>=0.6.0``.

Other notable changes:

- Bug fix re storing/loading synthetics
- Moved from Travis CI to GitHub Actions.

The relevant aliases and deprecations for ``emg3d`` are (consult the release
notes of ``discretize`` for all changes):

**Aliases:** Aliases (left) remain valid pointers to the new names (right).

- ``x0`` => ``origin``
- ``nC`` => ``n_cells``
- ``vnC`` => ``shape_cells``
- ``nN`` => ``n_nodes``
- ``vnN`` => ``shape_nodes``
- ``nE`` => ``n_edges``
- ``nEx`` => ``n_edges_x``
- ``nEy`` => ``n_edges_y``
- ``nEz`` => ``n_edges_z``
- ``vnE`` => ``n_edges_per_direction``
- ``vnEx`` => ``shape_edges_x``
- ``vnEy`` => ``shape_edges_y``
- ``vnEz`` => ``shape_edges_z``

**Deprecations:** Deprecated properties (left) raise a deprecation warning and
will be removed in the future. Currently, they still work and point to the new
names (right).

- ``hx`` => ``h[0]``
- ``hy`` => ``h[1]``
- ``hz`` => ``h[2]``
- ``nCx`` => ``shape_cells[0]``
- ``nCy`` => ``shape_cells[1]``
- ``nCz`` => ``shape_cells[2]``
- ``nNx`` => ``shape_nodes[0]``
- ``nNy`` => ``shape_nodes[1]``
- ``nNz`` => ``shape_nodes[2]``
- ``vectorNx`` => ``nodes_x``
- ``vectorNy`` => ``nodes_y``
- ``vectorNz`` => ``nodes_z``
- ``vectorCCx`` => ``cell_centers_x``
- ``vectorCCy`` => ``cell_centers_y``
- ``vectorCCz`` => ``cell_centers_z``
- ``vol`` => ``cell_volumes``


*v0.14.3* : Bug fix
-------------------

**2020-11-19**

- Bug fix for ``discretize>=0.6.0``.


*v0.14.2* : Bug fix
-------------------

**2020-11-18**

- Bug fix for Windows affecting ``good_mg_cell_nr`` (int32 issue).


*v0.14.1* : Bug fix
-------------------

**2020-11-14**

- Fix for ``h5py>=3.0``.
- Improved docs re automatic gridding.


*v0.14.0* : Automatic gridding
------------------------------

**2020-11-07**

The simulation class comes new with an automatic gridding functionality, which
should make it much easier to compute CSEM data. With that the entire
optimization routine was improved too. See the API docs for more info of the
relevant implementation.

- ``simulation``:

  - ``Simulation``: New gridding options ``'single'``, ``'frequency'``
    ``'source'``, and ``'both'``; new default is ``'single'``.
  - ``compute()`` takes a new argument, ``min_offset``. If ``observed=True``,
    it will add Gaussian random noise according to the standard deviation of
    the data; it will set receivers responses below the minimum offset to NaN.
  - There is no longer a ``reference`` model.
  - ``misfit`` and ``gradient`` can now handle observations with NaN's.

- ``survey``: A ``Survey`` has new attributes ``standard_error``,
  ``noise_floor``, and ``relative_error``.

- ``optimize``: Completely changed misfit and data-weighting to more sensible
  functions.

- ``cli``:

  - As a consequence of the changes the ``data_weight_opts`` got removed.
  - New sections ``[data]`` to select the wanted data and ``[gridding_opts]``
    for options of the automatic gridding.
  - Section ``[simulation]`` has a new parameter ``min_offset`` (for creating
    observed data).
  - Output has a new parameter ``n_observations`` if ``misfit`` or ``gradient``
    were called, which is the number of observations that were used to compute
    the misfit.

- ``meshes``:

  - New functions ``construct_mesh``, ``get_origin_widths``,
    ``good_mg_cell_nr`` and other, smaller helper routines.
  - Deprecated the old meshing routines ``get_hx_h0``, ``get_cell_numbers``,
    ``get_stretched_h``, ``get_domain``, ``get_hx``; they will be removed in
    the future.
  - Default of ``good_mg_cell_nr`` changed, and the documentation (and
    verbosity) with regards to «good» number of cells was improved.

- Bug fixes:

  - ``maps``: Fixed the mapping of the gradients (``Conductivity`` is the only
    mapping that was not affected by this bug).

- Removed deprecated features:

  - ``models.Model``: Removed parameters ``res_{x;y;z}``.
  - ``io.save``: Removed deprecated parameter ``backend``.
  - ``io.save``: Removed default, file extension has to be provided.


*v0.13.0* : CLI
---------------

**2020-09-22**

- New Module ``cli`` for command-line interaction:

  The command-line interface can currently be used to forward model an entire
  ``Simulation``, and also to compute the misfit of it with respect to some
  data and the gradient of the misfit function. See the section "CLI interface"
  in the documentation for more info.


*Note that, while* ``cli`` *(v0.13.0) and* ``optimize`` *(v0.12.0) are
implemented, they are still in development and are likely going to change
throughout the next two minor releases or so.*

- Other changes:

  - ``solver``: Changes in ``verbosity`` for ``emg3d.solve``:

    - New default verbosity is 1 (only warnings; before it was 2).
    - Verbosities {-1;0;1} remain unchanged.
    - Verbosities {2;3;4} => {3;4;5}.
    - New verbosity 2: Only shows a one-liner at the end (plus warnings).

  - ``survey`` and ``simulation``: ``to_file`` and ``from_file`` have new a
    parameter ``name``, to store and load with a particular name instead of the
    default ``survey``/``simulation`` (useful when storing, e.g., many surveys
    in one file).

  - ``survey``: stores new also the reference-data; different data (observed,
    reference) is contained in a data-dict when storing.

  - ``simulation``: takes new a ``verb`` parameter.

  - ``optimize``:

    - Gradient now possible for arbitrarily rotated sources and receivers.
    - Falls back to ``synthetic`` instead of ``observed`` now if ``reference``
      not found.

  - ``io``: ``np.bool_`` are converted back to ``bool`` when loading.

  - Re-arrange, improve, and update documentation.


*v0.12.0* : Survey & Simulation
-------------------------------

**2020-07-25**

This is a big release with many new features, and unfortunately not completely
backwards compatible. The main new features are the new **Survey** and
**Simulation** classes, as well as some initial work for **optimization**
(misfit, gradient). Also, a **Model** can now be a resistivity model, a
conductivity model, or the logarithm (natural or base 10) therefore. Receivers
can now be arbitrarily rotated, just as the sources. In addition to the
existing **soft-dependencies** ``empymod``, ``discretize``, and ``h5py`` there
are the new soft-dependencies ``xarray`` and ``tqm``; ``discretize`` is now
much tighter integrated. For the new survey and simulation classes ``xarray``
is a required dependency. However, the only hard dependency remain ``scipy``
and ``numba``, if you use ``emg3d`` purely as a solver. Data reading and
writing has new a JSON-backend, in addition to the existing HDF5 and
NumPy-backends.

In more detail:

- Modules:

  - ``surveys`` (**new**; requires ``xarray``):

    - Class ``surveys.Survey``, which combines sources, receivers, and data.
    - Class ``surveys.Dipole``, which defines electric or magnetic point
      dipoles and finite length dipoles.

  - ``simulations`` (**new**; requires ``xarray``; soft-dependency ``tqdm``):

    - Class ``simulations.Simulation``, which combines a survey with a model. A
      simulation computes the e-field (and h-field) asynchronously using
      ``concurrent.futures``. This class will include automatic, source- and
      frequency-dependent gridding in the future. If ``tqdm`` is installed it
      displays a progress bar for the asynchronous computation. Note that the
      simulation class has still some limitations, consult the class
      documentation.

  - ``models``:

    - Model instances take new the parameters ``property_{x;y;z}`` instead of
      ``res_{x;y;z}``. The properties can be either resistivity, conductivity,
      or log_{e;10} thereof. What is actually provided has to be defined with
      the parameter ``mapping``. By default, it remains resistivity, as it was
      until now. The keywords ``res_{x;y;z}`` are **deprecated**, but still
      accepted at the moment. The attributes ``model.res_{x;y;z}`` are still
      available too, but equally **deprecated**. However, it is **no longer
      possible to assign values to these attributes**, which is a **backwards
      incompatible** change.
    - A model knows now how to interpolate itself from its grid to another grid
      (``interpolate2grid``).

  - ``maps``:

    - **New** mappings for ``models.Model`` instances: The mappings take care
      of how to transform the investigation variable to conductivity and back,
      and how it affects its derivative.
    - **New** interpolation routine ``edges2cellaverages``.

  - ``fields``:

    - Function ``get_receiver_response`` (**new**), which returns the response
      for arbitrarily rotated receivers.
    - Improvements to ``Field`` and ``SourceField``:

      - ``_sval`` and ``_smu0`` not stored any longer, derived from ``_freq``.
      - ``SourceField`` is now using the ``copy()`` and ``from_dict()`` from
        its parents class ``Field``.

  - ``io``:

    - File-format ``json`` (**new**), writes to a hierarchical, plain json
      file.
    - **Deprecated** the use of ``backend``, it uses the file extension of
      ``fname`` instead.
    - This means ``.npz`` (instead of ``numpy``), ``.h5`` (instead of
      ``h5py``), and new ``.json``.
    - New parameter ``collect_classes``, which can be used to switch-on
      collection of the main classes in root-level dictionaries. By default,
      they are no longer collected (**changed**).

  - ``meshes``:

    - ``meshes.TensorMesh`` **new** inherits from ``discretize`` if installed.
    - Added ``__eq__`` to ``models.TensorMesh`` to compare meshes.

  - ``optimize`` (**new**)

    - Functionalities related to inversion (data misfit, gradient, data
      weighting, and depth weighting). This module is in an early stage, and
      the API will likely change in the future. Current functions are
      ``misfit``, ``gradient`` (using the adjoint-state method), and
      ``data_weighting``. These functionalities are best accessed through the
      ``Simulation`` class.

- Dependencies:

  - ``empymod`` is now a soft dependency (no longer a hard dependency), only
    required for ``utils.Fourier`` (time-domain modelling).
  - Existing soft dependency ``discretize`` is now baked straight into
    ``meshes``.
  - New soft dependency ``xarray`` for the ``Survey`` class (and therefore also
    for the ``Simulation`` class and the ``optimize`` module).
  - New soft dependency ``tqdm`` for nice progress bars in asynchronous
    computation.

- **Deprecations** and removals:

  - Removed deprecated functions ``data_write`` and ``data_read``.
  - Removed all deprecated functions from ``utils``.

- Miscellaneous:

  - Re-organise API-docs.
  - Much bookkeeping (improve error raising and checking; chaining errors,
    numpy types, etc).


*v0.11.0* : Refactor
--------------------

**2020-05-05**

Grand refactor with new internal layout. Mainly splitting-up ``utils`` into
smaller bits. Most functionalities (old names) are currently retained in
``utils`` and it should be mostly backwards compatible for now, but they are
deprecated and will eventually be removed. Some previously deprecated functions
were removed, however.

- Removed deprecated functions:

  - ``emg3d.solver.solver`` (use ``emg3d.solver.solve`` instead).
  - Aliases of ``emg3d.io.data_write`` and ``emg3d.io.data_read`` in
    ``emg3d.utils``.

- Changes:

  - ``SourceField`` has now the same signature as ``Field`` (this might break
    your code if you called ``SourceField`` directly, with positional
    arguments, and not through ``get_source_field``).
  - More functions and classes in the top namespace.
  - Replaced ``core.l2norm`` with ``scipy.linalg.norm``, as SciPy 1.4 got the
    following PR: https://github.com/scipy/scipy/pull/10397 (reason to raise
    minimum SciPy to 1.4).
  - Increased minimum required versions of dependencies to

    - ``scipy>=1.4.0`` (raised from 1.1, see note above)
    - ``empymod>=2.0.0`` (no min requirement before)
    - ``numba>=0.45.0`` (raised from 0.40)

- New layout

  - ``njitted`` -> ``core``.
  - ``utils`` split in ``fields``, ``meshes``, ``models``, ``maps``, and
    ``utils``.

- Bugfixes:

  - Fixed ``to_dict``, ``from_dict``, and ``copy`` for the ``SourceField``.
  - Fixed ``io`` for ``SourceField``, that was not implemented properly.


*v0.10.1* : Zero Source
-----------------------

**2020-04-29**

- Bug fixes:

  - Checks now if provided source-field is zero, and exists gracefully if so,
    returning a zero electric field. Until now it failed with a
    division-by-zero error.

- Improvements:

  - Warnings: If ``verb=1`` it prints a warning in case it did not converge (it
    finished silently until now).
  - Improvements to docs (figures-scaling; intersphinx).
  - Adjust ``Fields.pha`` and ``Fields.amp`` in accordance with ``empymod v2``:
    ``.pha`` and ``.amp`` are now methods; uses directly
    ``empymod.utils.EMArray``.
  - Adjust tests for ``empymod v2`` (Fields, Fourier).


*v0.10.0* : Data persistence
----------------------------

**2020-03-25**

- New:

  - New functions ``emg3d.save`` and ``emg3d.load`` to save and load all sort
    of ``emg3d`` instances. The currently implemented backends are
    ``h5py`` for ``.h5``-files (default, but requires ``h5py`` to be installed)
    and ``numpy`` for ``.npz``-files.
  - Classes ``emg3d.utils.Field``, ``emg3d.utils.Model``, and
    ``emg3d.utils.TensorMesh`` have new methods ``.copy()``, ``.to_dict()``,
    and ``.from_dict()``.
  - ``emg3d.utils.Model``: Possible to create new models by adding or
    subtracting existing models, and comparing two models (``+``, ``-``, ``==``
    and ``!=``). New attributes ``shape`` and ``size``.
  - ``emg3d.utils.Model`` does not store the volume any longer (just ``vnC``).

- Deprecations:

  - Deprecated ``data_write`` and ``data_read``.

- Internal and bug fixes:

  - All I/O-related stuff moved to its own file ``io.py``.
  - Change from ``NUMBA_DISABLE_JIT`` to use ``py_func`` for testing and
    coverage.
  - Bugfix: ``emg3d.njitted.restrict`` did not store the {x;y;z}-field if
    ``sc_dir`` was {4;5;6}, respectively.


*v0.9.3* : Sphinx gallery
-------------------------

**2020-02-11**

- Rename ``solver.solver`` to ``solver.solve``; load ``solve`` also into the
  main namespace as ``emg3d.solve``.
- Adjustment to termination criterion for *STAGNATION*: The current error is
  now compared to the last error of the same cycle type. Together with this the
  workaround for sslsolver when called with an initial efield introduced in
  v0.8.0 was removed.
- Adjustment to ``utils.get_hx_h0`` (this might change your boundaries): The
  computation domain is now computed so that the distance for the signal
  travelling from the source to the boundary and back to the most remote
  receiver is at least two wavelengths away. If this is within the provided
  domain, then now extra buffer is added around the domain. Additionally, the
  function has a new parameter ``max_domain``, which is the maximum distance
  from the center to the boundary; defaults to 100 km.
- New parameter ``log`` for ``utils.grid2grid``; if ``True``, then the
  interpolation is carried out on a log10-scale.
- Change from the notebook-based ``emg3d-examples``-repo to the
  ``sphinx``-based ``emg3d-gallery``-repo.


*v0.9.2* : Complex sources
--------------------------

**2019-12-26**

- Strength input for ``get_source_field`` can now be complex; it also stores
  now the source location and its strength and moment.
- ``get_receiver`` can now take entire ``Field`` instances, and returns in that
  case (``fx``, ``fy``, ``fz``) at receiver locations.
- Krylov subspace solvers:

  - Solver now finishes in the middle of preconditioning cycles if tolerance is
    reached.
  - Solver now aborts if solution diverges or stagnates also for the SSL
    solvers; it fails and returns a zero field.
  - Removed ``gmres`` and ``lgmres`` from the supported SSL solvers; they do
    not work nice for this problem. Supported remain ``bicgstab`` (default),
    ``cgs``, and ``gcrotmk``.

- Various small things:

  - New attribute ``Field.is_electric``, so the field knows if it is electric
    or magnetic.
  - New ``verb``-possibility: ``verb=-1`` is a continuously updated one-liner,
    ideal to monitor large sets of computations or in inversions.
  - The returned ``info`` dictionary contains new keys:

    - ``runtime_at_cycle``: accumulated total runtime at each cycle;
    - ``error_at_cycle``: absolute error at each cycle.

  - Simple ``__repr__`` for ``TensorMesh``, ``Model``, ``Fourier``, ``Time``.

- Bugfixes:

  - Related to ``get_hx_h0``, ``data_write``, printing in ``Fourier``.


*v0.9.1* : VolumeModel
----------------------

**2019-11-13**

- New class ``VolumeModel``; changes in ``Model``:

  - ``Model`` now only contains resistivity, magnetic permeability, and
    electric permittivity.
  - ``VolumeModel`` contains the volume-averaged values eta and zeta; called
    from within ``emg3d.solver.solver``.
  - Full wave equation is enabled again, via ``epsilon_r``; by default it is
    set to None, hence diffusive approximation.
  - Model parameters are now internally stored as 1D arrays.
  - An {isotropic, VTI, HTI} initiated model can be changed by providing the
    missing resistivities.

- Bugfix: Up and till version 0.8.1 there was a bug. If resistivity was set
  with slices, e.g., ``model.res[:, :, :5]=1e10``, it DID NOT update the
  corresponding eta. This bug was unintentionally fixed in 0.9.0, but only
  realised now.

- Various:

  - The log now lists the version of emg3d.
  - PEP8: internal imports now use absolute paths instead of relative ones.
  - Move from conda-channel ``prisae`` to ``conda-forge``.
  - Automatic deploy for PyPi and conda-forge.


*v0.9.0* : Fourier
------------------

**2019-11-07**

- New routine:

  - ``emg3d.utils.Fourier``, a class to handle Fourier-transform related stuff
    for time-domain modelling. See the example notebooks for its usage.

- Utilities:

  - ``Fields`` and returned receiver-arrays (``EMArray``) both have amplitude
    (``.amp``) and phase (``.pha``) attributes.
  - ``Fields`` have attributes containing frequency-information (``freq``,
    ``smu0``).
  - New class ``SourceField``; a subclass of ``Field``, adding ``vector`` and
    ``v{x,y,z}`` attributes for the real valued source vectors.
  - The ``Model`` is not frequency-dependent any longer and does NOT take
    a ``freq``-parameter any more (currently it still takes it, but it is
    deprecated and will be removed in the future).
  - ``data_write`` automatically removes ``_vol`` from ``TensorMesh`` instances
    and ``_eta_{x,y,z}``, ``_zeta`` from ``Model`` instances. This makes the
    archives smaller, and they are not required, as they are simply
    reconstructed if needed.

- Internal changes:

  - The multigrid method, as implemented, only works for the diffusive
    approximation. Nevertheless, we always used ``\sigma-i\omega\epsilon``,
    hence a complex number. This is now changed and ``\epsilon`` set to 0,
    leaving only ``\sigma``.
  - Change time convention from ``exp(-iwt)`` to ``exp(iwt)``, as used in
    ``empymod`` and commonly in CSEM. Removed the parameter ``conjugate`` from
    the solver, to simplify.
  - Change own private class variables from ``__`` to ``_``.
  - ``res`` and ``mu_r`` are now checked to ensure they are >0; ``freq`` is
    checked to ensure !=0.

- New dependencies and maintenance:

  - ``empymod`` is a new dependency.
  - Travis now checks all the url's in the documentation, so there should be no
    broken links down the road. (Check is allowed to fail, it is visual QC.)

- Bugfixes:

  - Fixes to the ``setuptools_scm``-implementation (``MANIFEST.in``).


*v0.8.1* : setuptools_scm
-------------------------

**2019-10-22**

- Implement ``setuptools_scm`` for versioning (adds git hashes for
  dev-versions).


*v0.8.0* : Laplace
------------------

**2019-10-04**

- Laplace-domain computation: By providing a negative ``freq``-value to
  ``utils.get_source_field`` and ``utils.Model``, the computation is carried
  out in the real Laplace domain ``s = freq`` instead of the complex frequency
  domain ``s = 2i*pi*freq``.
- New meshing helper routines (particularly useful for transient modelling
  where frequency-dependent/adaptive meshes are inevitable):

  - ``utils.get_hx_h0`` to get cell widths and origin for given parameters
    including a few fixed interfaces (center plus two, e.g. top anomaly,
    sea-floor, and sea-surface).
  - ``utils.get_cell_numbers`` to get good values of number of cells for given
    primes.

- Speed-up ``njitted.volume_average`` significantly thanks to Joe (@jcapriot).
- Bugfixes and other minor things:

  - Abort if l2-norm is NaN (only works for MG).
  - Workaround for the case where a ``sslsolver`` is used together with a
    provided initial ``efield``.
  - Changed parameter ``rho`` to ``res`` for consistency reasons in
    ``utils.get_domain``.
  - Changed parameter ``h_min`` to ``min_width`` for consistency reasons in
    ``utils.get_stretched_h``.


*v0.7.1* : JOSS article
-----------------------

**2019-07-17**

- Version of the JOSS article, https://doi.org/10.21105/joss.01463 .
- New function ``utils.grid2grid`` to move from one grid to another. Both
  functions (``utils.get_receiver`` and ``utils.grid2grid``) can be used for
  fields and model parameters (with or without extrapolation). They are very
  similar, the former taking coordinates (x, y, z) as new points, the latter
  one another TensorMesh instance.
- New jitted function ``njitted.volume_average`` for interpolation using the
  volume-average technique.
- New parameter ``conjugate`` in ``solver.solver`` to permit both Fourier
  transform conventions.
- Added ``exit_status`` and ``exit_message`` to ``info_dict``.
- Add section ``Related ecosystem`` to documentation.


*v0.7.0* : H-field
------------------

**2019-07-05**

- New routines:

  - ``utils.get_h_field``: Small routine to compute the magnetic field from
    the electric field using Faraday's law.
  - ``utils.get_receiver``: Small wrapper to interpolate a field at receiver
    positions. Added 3D spline interpolation; is the new default.

- Re-implemented the possibility to define isotropic magnetic permeabilities in
  ``utils.Model``. Magnetic permeability is not tri-axially included in the
  solver currently; however, it would not be too difficult to include if there
  is a need.
- CPU-graph added on top of RAM-graph.
- Expand ``utils.Field`` to work with pickle/shelve.
- Jit ``np.linalg.norm`` (``njitted.l2norm``).
- Use ``scooby`` (soft dependency) for versioning, rename ``Version`` to
  ``Report`` (backwards incompatible).

- Bug fixes:

  - Small bugfix introduced in ebd2c9d5: ``sc_cycle`` and ``lr_cycle`` was not
    updated any longer at the end of a cycle (only affected ``sslsolver=True``.
  - Small bugfix in ``utils.get_hx``.


*v0.6.2* : CPU & RAM
--------------------

**2019-06-03**

Further speed and memory improvements:

- Add *CPU & RAM*-page to documentation.
- Change loop-order from x-z-y to z-x-y in Gauss-Seidel smoothing with line
  relaxation in y-direction. Hence reversed lexicographical order. This results
  in a significant speed-up, as x is the fastest changing axis.
- Move total residual computation from ``solver.residual`` into
  ``njitted.amat_x``.
- Simplifications in ``utils``:

  - Simplify ``utils.get_source_field``.
  - Simplify ``utils.Model``.
  - Removed unused timing-stuff from early development.


*v0.6.1* : Memory
-----------------

**2019-05-28**

Memory and speed improvements:

- Only compute residual and l2-norm when absolutely necessary.
- Inplace computations for ``np.conjugate`` in ``solver.solver`` and
  ``np.subtract`` in ``solver.residual``.


*v0.6.0* : RegularGridInterpolator
----------------------------------

**2019-05-26**

- Replace :class:`scipy.interpolate.RegularGridInterpolator` with a custom
  tailored version of it (class:`emg3d.solver.RegularGridProlongator`); results
  in twice as fast prolongation.
- Simplify the fine-grid computation in ``prolongation`` without using
  ``gridE*``; memory friendlier.
- Submission to JOSS.
- Add *Multi-what?*-page to documentation.
- Some major refactoring, particularly in ``solver``.
- Removed ``discretize`` as hard dependency.
- Rename ``rdir`` and ``ldir`` (and related ``p*dir``; ``*cycle``) to the more
  descriptive ``sc_dir`` and ``lr_dir``.


v0.5.0 : Accept any grid size
-----------------------------

**2019-05-01**

- First open-source version.
- Include RTD, Travis, Coveralls, Codacy, and Zenodo. No benchmarks yet.
- Accepts now *any* grid size (warns if a bad grid size for MG is provided).
- Coarsens now to the lowest level of each dimension, not only to the coarsest
  level of the smallest dimension.
- Combined ``restrict_rx``, ``restrict_ry``, and ``restrict_rz`` to
  ``restrict``.
- Improve speed by passing pre-allocated arrays to jitted functions.
- Store ``res_y``, ``res_z`` and corresponding ``eta_y``, ``eta_z`` only if
  ``res_y``, ``res_z`` were provided in initial call to ``utils.model``.
- Change ``zeta`` to ``v_mu_r``.
- Include rudimentary ``TensorMesh``-class in ``utils``; removes hard
  dependency on ``discretize``.
- Bugfix: Take a provided ``efield`` into account; don't return if provided.


v0.4.0 : Cholesky
-----------------

**2019-03-29**

- Use ``solve_chol`` for everything, remove ``solve_zlin``.
- Moved ``mesh.py`` and some functionalities from ``solver.py`` into
  ``utils.py``.
- New mesh-tools. Should move to ``discretize`` eventually.
- Improved source generation tool. Might also move to ``discretize``.
- ``printversion`` is now included in ``utils``.
- Many bug fixes.
- Lots of improvements to tests.
- Lots of improvements to documentation. Amongst other, moved docs from
  ``__init__.py`` into the docs rst.


v0.3.0 : Semicoarsening
-----------------------

**2019-01-18**

- Semicoarsening option.
- Number of cells must still be 2^n, but n can be different in the x-, y-, and
  z-directions.
- Many other iterative solvers from :mod:`scipy.sparse.linalg` can be used. It
  seems to work fine with the following methods:

  - :func:`scipy.sparse.linalg.bicgstab`:  BIConjugate Gradient STABilize;
  - :func:`scipy.sparse.linalg.cgs`: Conjugate Gradient Squared;
  - :func:`scipy.sparse.linalg.gmres`: Generalized Minimal RESidual;
  - :func:`scipy.sparse.linalg.lgmres`: Improvement of GMRES using alternating
    residual vectors;
  - :func:`scipy.sparse.linalg.gcrotmk`: GCROT: Generalized Conjugate Residual
    with inner Orthogonalization and Outer Truncation.

- The SciPy-solver or MG can be used all in combination or on its own, hence
  only MG, SciPy-solver with MG preconditioning, only SciPy-solver.


v0.2.0 : Line relaxation
------------------------

**2019-01-14**

- Line relaxation option.


v0.1.0 : Initial
----------------

**2018-12-28**

- Standard multigrid with or without BiCGSTAB.
- Tri-axial anisotropy.
- Number of cells must be 2^n, and n has to be the same in the x-, y-, and
  z-directions.
