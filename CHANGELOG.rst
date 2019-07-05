Changelog
#########


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
