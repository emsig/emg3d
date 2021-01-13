Changelog
#########


recent versions
"""""""""""""""

v0.16.0: Arbitrarily shaped sources
-----------------------------------

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


v0.11.0 - v0.15.3
"""""""""""""""""

v0.15.3: Move to EMSiG
----------------------

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


v0.8.0 - v0.10.1
""""""""""""""""

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

- Speed-up ``njitted.volume_average`` significantly thanks to @jcapriot.
- Bugfixes and other minor things:

  - Abort if l2-norm is NaN (only works for MG).
  - Workaround for the case where a ``sslsolver`` is used together with a
    provided initial ``efield``.
  - Changed parameter ``rho`` to ``res`` for consistency reasons in
    ``utils.get_domain``.
  - Changed parameter ``h_min`` to ``min_width`` for consistency reasons in
    ``utils.get_stretched_h``.


v0.1.0 - v0.7.1
"""""""""""""""

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
