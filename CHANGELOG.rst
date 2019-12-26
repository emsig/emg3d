Changelog
#########


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
    ideal to monitor large sets of calculations or in inversions.
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

- Laplace-domain calculation: By providing a negative ``freq``-value to
  ``utils.get_source_field`` and ``utils.Model``, the calculation is carried
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

  - ``utils.get_h_field``: Small routine to calculate the magnetic field from
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
- Move total residual calculation from ``solver.residual`` into
  ``njitted.amat_x``.
- Simplifications in ``utils``:

  - Simplify ``utils.get_source_field``.
  - Simplify ``utils.Model``.
  - Removed unused timing-stuff from early development.


*v0.6.1* : Memory
-----------------

**2019-05-28**

Memory and speed improvements:

- Only calculate residual and l2-norm when absolutely necessary.
- Inplace calculations for ``np.conjugate`` in ``solver.solver`` and
  ``np.subtract`` in ``solver.residual``.


*v0.6.0* : RegularGridInterpolator
----------------------------------

**2019-05-26**

- Replace :class:`scipy.interpolate.RegularGridInterpolator` with a custom
  tailored version of it (`solver.RegularGridProlongator`); results in twice
  as fast prolongation.
- Simplify the fine-grid calculation in ``prolongation`` without using
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
