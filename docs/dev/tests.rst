Tests and Benchmarks
====================

If you ended up here you probably think about fixing some bugs or contributing
some code. **Awesome!** Just open a PR, and we will guide you through the
process. The following section contains some more detailed information of the
continues integration (CI) procedure we follow. In the end, each commit has to
pass them before it can be merged into the main branch on GitHub.


The first step to develop code is to clone the GitHub repo locally:

.. code-block:: console

   git clone git@github.com:emsig/emg3d.git

All requirements for the dev-toolchain are collected in the
``requirements-dev.txt`` file, so you can install them all by running

.. code-block:: console

   pip install -r requirements_dev.txt

With this you have all the basic tools to run the tests, lint your code, build
the documentation, and so on.

Continuous Integration
----------------------

The CI elements are:

1. Linting: ``flake8``
2. Tests: ``pytest``
3. Code coverage: ``coveralls``
4. Link checks: ``sphinx``
5. Code quality: ``codacy``
6. Documentation: ``sphinx``
7. Benchmarks: ``asv``


(1) to (6) are run automatically through GitHub actions when committing changes
to GitHub. Any code change should pass these tests. Additionally, it is crucial
that new code comes with the appropriate tests and documentation, and if
applicable also with the appropriate benchmarks. However, you do not need any
of that to start a PR - everything can go step-by-step!

Many of the tests are set up in the Makefile (only tested on Linux):

- To install the current branch in editable mode:

  .. code-block:: console

     make install

- To check linting:

  .. code-block:: console

     make flake8

- To run pytest:

  .. code-block:: console

     make pytest

- To build the documentation:

  .. code-block:: console

     make html

- Or to list all the possibilities, simply run:

  .. code-block:: console

     make

There is also a benchmark suite using *airspeed velocity*, located in the
`emsig/emg3d-asv <https://github.com/emsig/emg3d-asv>`_-repository. The results
of my machine can be found in the `emsig/emg3d-bench
<https://github.com/emsig/emg3d-bench>`_, its rendered version at
`emsig.xyz/emg3d-asv <https://emsig.xyz/emg3d-asv>`_. They ensure that we do
not slow than the computation by introducing regressions, particularly when we
make changes to :mod:`emg3d.core` or :mod:`emg3d.solver`.


.. _improve-cpu-ram:

CPU & RAM
---------

Here some information if someone is interested in tackling the very core of
emg3d, trying to make it faster or reduce the memory consumption. The multigrid
method is attractive because it shows optimal scaling for both runtime and
memory consumption. Below some insights about what has been tried and what
still could be tried in order to improve the current code.


Runtime
~~~~~~~

The costliest functions (for big models) are:

   - >90 %: :func:`emg3d.solver.smoothing` (:func:`emg3d.core.gauss_seidel`)
   - <5 % each, in decreasing importance:

      - :func:`emg3d.solver.prolongation`
        (:class:`emg3d.solver.RegularGridProlongator`)
      - :func:`emg3d.solver.residual` (:func:`emg3d.core.amat_x`)
      - :func:`emg3d.solver.restriction`

Example with 262,144 / 2,097,152 cells (``nu_{i,1,c,2}=0,2,1,2``;
``sslsolver=False``; ``semicoarsening=True``; ``linerelaxation=True``):

   - 93.7 / 95.8 % ``smoothing``
   - 3.6 / 2.0 % ``prolongation``
   - 1.9 / 1.9 % ``residual``
   - 0.6 / 0.4 % ``restriction``

The rest can be ignored. For small models, the percentage of ``smoothing`` goes
down and of ``prolongation`` and ``restriction`` go up. But then the modeller
is fast anyway.

:func:`emg3d.core.gauss_seidel` and :func:`emg3d.core.amat_x` are written
in ``numba``; jitting :class:`emg3d.solver.RegularGridProlongator` turned out
to not improve things, and many functions used in the restriction are jitted
too. The costliest functions (RAM- and CPU-wise) are therefore already written
in ``numba``.

**Any serious attempt to improve the speed will have to tackle the smoothing
itself.**


**Things which could be tried**

- Not much has been tested with the ``numba``-options ``parallel``; ``prange``;
  and ``nogil``.
- There might be an additional gain by making :class:`emg3d.meshes.TensorMesh`,
  :class:`emg3d.models.Model`, and :class:`emg3d.fields.Field` instances jitted
  classes.

**Things which have been tried**

- One important aspect of the smoothing part is the memory layout.
  :func:`emg3d.core.gauss_seidel` and :func:`emg3d.core.gauss_seidel_x`
  are ideal for F-arrays (loop z-y-x, hence slowest to fastest axis).
  :func:`emg3d.core.gauss_seidel_y` and
  :func:`emg3d.core.gauss_seidel_z`, however, would be optimal for C-arrays.
  But copying the arrays to C-order and afterwards back is costlier in most
  cases for both CPU and RAM. The one possible and therefore implemented
  solution was to swap the loop-order in :func:`emg3d.core.gauss_seidel_y`.
- Restriction and prolongation information could be saved in a dictionary
  instead of recomputing it every time. Turns out to be not worth the
  trouble.
- Rewrite :class:`emg3d.solver.RegularGridProlongator` as jitted function, but
  the iterator approach seems to be better for large grids.


Memory
~~~~~~

Most of the memory requirement comes from storing the data itself, mainly the
fields (source field, electric field, and residual field) and the model
parameters (resistivity, eta, mu). For a big model, they some up; e.g., almost
3 GB for an isotropic model with 256 x 256 x 256 cells. Anyhow, memory
consumption is pretty low already, and there is probably not much to gain, at
least in the solver part (:mod:`emg3d.core` and :mod:`emg3d.solver`). That
looks different for some of the interpolation and plotting routines, which
could be improved .


Benchmark scripts for status quo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test CPU and RAM on your machine, you can use and adjust the following
script. The old notebooks which were used to generate the above figures in the
manual can be found at

- RAM: `4a_RAM-requirements.ipynb
  <https://github.com/emsig/emg3d-examples/blob/master/4a_RAM-requirements.ipynb>`_,
- CPU: `4b_Runtime.ipynb
  <https://github.com/emsig/emg3d-examples/blob/master/4b_Runtime.ipynb>`_.

.. ipython::
  :verbatim:

  In [1]: import emg3d
     ...: import numpy as np
     ...: import matplotlib.pyplot as plt
     ...: from memory_profiler import memory_usage

  In [2]: def compute(nx):
     ...:     """Simple computation routine.
     ...:
     ...:     This is the actual model it runs. Adjust this to your needs.
     ...:
     ...:     - Model size is nx * nx * nx, centered around the origin.
     ...:     - Source is at the origin, x-directed.
     ...:     - Frequency is 1 Hz.
     ...:     - Homogenous space of 1 Ohm.m.
     ...:
     ...:     """
     ...:
     ...:     # Grid
     ...:     hx = np.ones(nx)*50
     ...:     x0 = -nx//2*50
     ...:     grid = emg3d.TensorMesh([hx, hx, hx], x0=(x0, x0, x0))
     ...:
     ...:     # Model and source field
     ...:     model = emg3d.Model(grid, property_x=1.0)
     ...:     sfield = emg3d.get_source_field(
     ...:             grid, source=[0, 0, 0, 0, 0], frequency=1.0)
     ...:
     ...:     # Compute the field
     ...:     _, inf = emg3d.solve(
     ...:             model, sfield, verb=0, plain=True, return_info=True)
     ...:
     ...:     return inf['time']

  In [3]: # Loop over model sizes (adjust to your needs).
     ...: nsizes = np.array([32, 48, 64, 96, 128, 192, 256, 384])
     ...: memory = np.zeros(nsizes.shape)
     ...: runtime = np.zeros(nsizes.shape)
     ...:
     ...: # Loop over nx
     ...: for i, nx in enumerate(nsizes):
     ...:     print(f"  => {nx}^3 = {nx**3:12,d} cells")
     ...:     mem, time = memory_usage((compute, (nx, ), {}), retval=True)
     ...:     memory[i] = max(mem)
     ...:     runtime[i] = time
     ...:

  In [4]: # Plot CPU
     ...: plt.figure()
     ...: plt.title('Runtime')
     ...: plt.loglog(nsizes**3/1e6, runtime, '.-')
     ...: plt.xlabel('Number of cells (in millions)')
     ...: plt.ylabel('CPU (s)')
     ...: plt.axis('equal')
     ...: plt.show()

  In [5]: # Plot RAM
     ...: plt.figure()
     ...: plt.title('Memory')
     ...: plt.loglog(nsizes**3/1e6, memory/1e3, '-', zorder=10)
     ...: plt.xlabel('Number of cells (in millions)')
     ...: plt.ylabel('RAM (GB)')
     ...: plt.axis('equal')
     ...: plt.show()


Scripts for solver investigations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The non-standard Cholesky solver, :func:`emg3d.core.solve`, does almost all the
work, in the end. Improving the speed of that part only slightly would have a
huge effect overall. Here some notes from some dabbling.

**Benchmark Tests for Cholesky Solve**

- `numba`, `numpy`, `scipy`, `lapack`
- Benchmarks:
  - small and big
  - real and complex valued

"Givens":

- Diagonal != 0
- Diagonal values are large (no pivoting)
- Only diagonal values would be complex

.. ipython::
  :verbatim:

  In [1]: import numba as nb
     ...: import numpy as np
     ...: import scipy as sp
     ...: from numpy.testing import assert_allclose
     ...: from scipy.linalg.lapack import get_lapack_funcs
     ...:
     ...: _numba_setting = {'nogil': True, 'fastmath': True, 'cache': True}

Status quo

.. ipython::
  :verbatim:

  In [1]: @nb.njit(**_numba_setting)
     ...: def _emg3d_solve(amat, bvec):
     ...:     n = len(bvec)
     ...:     h = np.zeros(1, dtype=amat.dtype)[0]  # Pre-allocate
     ...:     d = 1./amat[0]
     ...:
     ...:     for i in range(1, min(n, 6)):
     ...:         amat[i] *= d
     ...:
     ...:     for j in range(1, n):
     ...:         h *= 0.  # Reset h
     ...:         for k in range(max(0, j-5), j):
     ...:             h += amat[j+5*k]*amat[j+5*k]*amat[6*k]
     ...:         amat[6*j] -= h
     ...:         d = 1./amat[6*j]
     ...:         for i in range(j+1, min(n, j+6)):
     ...:             h *= 0.  # Reset h
     ...:             for k in range(max(0, i-5), j):
     ...:                 h += amat[i+5*k]*amat[j+5*k]*amat[6*k]
     ...:             amat[i+5*j] -= h
     ...:             amat[i+5*j] *= d
     ...:
     ...:     amat[6*(n-1)] = d
     ...:     for j in range(n-2, -1, -1):
     ...:         amat[6*j] = 1./amat[6*j]
     ...:
     ...:     for j in range(1, n):
     ...:         h *= 0.  # Reset h
     ...:         for k in range(max(0, j-5), j):
     ...:             h += amat[j+5*k]*bvec[k]
     ...:         bvec[j] -= h
     ...:
     ...:     for j in range(n):
     ...:         bvec[j] *= amat[6*j]
     ...:
     ...:     for j in range(n-2, -1, -1):
     ...:         h *= 0.  # Reset h
     ...:         for k in range(j+1, min(n, j+6)):
     ...:             h += amat[k+5*j]*bvec[k]
     ...:         bvec[j] -= h
     ...:
     ...: def emg3d_solve(amat, bvec):
     ...:     out = bvec.copy()
     ...:     _emg3d_solve(amat.copy(), out)
     ...:     return out


An alternative

.. ipython::
  :verbatim:

  In [1]: @nb.njit(**_numba_setting)
     ...: def _emg3d_solve2(amat, bvec):
     ...:     n = len(bvec)
     ...:     h = np.zeros(1, dtype=amat.dtype)[0]  # Pre-allocate
     ...:     d = 1./amat[0]
     ...:
     ...:     for i in range(1, min(n, 6)):
     ...:         amat[i] *= d
     ...:
     ...:     for j in range(1, n):
     ...:         h *= 0.  # Reset h
     ...:         for k in range(max(0, j-5), j):
     ...:             h += amat[j+5*k]*amat[j+5*k]*amat[6*k]
     ...:         amat[6*j] -= h
     ...:         d = 1./amat[6*j]
     ...:         for i in range(j+1, min(n, j+6)):
     ...:             h *= 0.  # Reset h
     ...:             for k in range(max(0, i-5), j):
     ...:                 h += amat[i+5*k]*amat[j+5*k]*amat[6*k]
     ...:
     ...:             amat[i+5*j] = d*(amat[i+5*j] - h)
     ...:
     ...:     amat[6*(n-1)] = d
     ...:     for j in range(n-2, -1, -1):
     ...:         amat[6*j] = 1./amat[6*j]
     ...:
     ...:     for j in range(1, n):
     ...:         h *= 0.  # Reset h
     ...:         for k in range(max(0, j-5), j):
     ...:             h += amat[j+5*k]*bvec[k]
     ...:         bvec[j] -= h
     ...:
     ...:     for j in range(n):
     ...:         bvec[j] *= amat[6*j]
     ...:
     ...:     for j in range(n-2, -1, -1):
     ...:         h *= 0.  # Reset h
     ...:         for k in range(j+1, min(n, j+6)):
     ...:             h += amat[k+5*j]*bvec[k]
     ...:         bvec[j] -= h
     ...:
     ...:
     ...: def emg3d_solve2(amat, bvec):
     ...:     out = bvec.copy()
     ...:     _emg3d_solve2(amat.copy(), out)
     ...:     return out

SciPy and NumPy solvers

.. ipython::
  :verbatim:

  In [1]: def np_linalg_solve(A, b):
     ...:     return np.linalg.solve(A, b)
     ...:
     ...: def sp_linalg_solve(A, b):
     ...:     out = b.copy()
     ...:     sp.linalg.solve(A.copy(), out, overwrite_a=True,
     ...:                     overwrite_b=True, check_finite=False)
     ...:     return out
     ...:
     ...: def sp_linalg_lu_solve(A, b):
     ...:     out = b.copy()
     ...:     lu_and_piv = sp.linalg.lu_factor(A.copy(), overwrite_a=True,
     ...:                                      check_finite=False)
     ...:     xlu = sp.linalg.lu_solve(lu_and_piv, out, overwrite_b=True,
     ...:                              check_finite=False)
     ...:     return out
     ...:
     ...: def sp_linalg_cho_solve(A, b):
     ...:     amat = A.copy()
     ...:     clow = sp.linalg.cho_factor(amat, lower=True, overwrite_a=True,
     ...:                                 check_finite=False)
     ...:     out = b.copy()
     ...:     sp.linalg.cho_solve(clow, out, overwrite_b=True,
     ...:                         check_finite=False)
     ...:     return out
     ...:
     ...: def sp_linalg_cho_banded(A, b):
     ...:     amat = A.copy()
     ...:     c = sp.linalg.cholesky_banded(amat, overwrite_ab=True,
     ...:                                   lower=True, check_finite=False)
     ...:     out = b.copy()
     ...:     sp.linalg.cho_solve_banded((c, True), out, overwrite_b=True,
     ...:                                check_finite=False)
     ...:     return out


Measuring them. You can get the data at
https://github.com/emsig/data/raw/main/emg3d/benchmarks/CholeskySolveBenchmark.npz

.. ipython::
  :verbatim:

  In [1]: data = np.load('CholeskySolveBenchmark.npz')
     ...:
     ...: for cr in ['real', 'cplx']:
     ...:     for bs in ['small', 'big']:
     ...:
     ...:         print(f"dtype={cr}; size={bs}")
     ...:
     ...:         # Get test data.
     ...:         amat = data[cr+'_'+bs+'_'+'amat']
     ...:         bvec = data[cr+'_'+bs+'_'+'bvec']
     ...:         out = data[cr+'_'+bs+'_'+'out']
     ...:
     ...:         # Re-arrange to full (symmetric) or banded matrix
     ...:         # for some solvers.
     ...:         n = bvec.size
     ...:         A = np.zeros((n, n), dtype=amat.dtype)
     ...:         Ab = np.zeros((6, n), dtype=amat.dtype)
     ...:         for i in range(n):
     ...:             A[i, i] = amat[i*6]
     ...:             Ab[0, i] = amat[i*6]
     ...:             for j in range(1, 6):
     ...:                 if i+j < n:
     ...:                     A[i, i+j] = amat[i*6+j]
     ...:                     A[i+j, i] = amat[i*6+j]
     ...:                 Ab[j, i] = amat[i*6+j]
     ...:
     ...:         # Assert result is correct
     ...:         assert_allclose(emg3d_solve(amat, bvec), out, rtol=1e-6)
     ...:         assert_allclose(emg3d_solve2(amat, bvec), out, rtol=1e-6)
     ...:         assert_allclose(np_linalg_solve(A, bvec), out, rtol=1e-6)
     ...:         assert_allclose(sp_linalg_solve(A, bvec), out, rtol=1e-6)
     ...:         assert_allclose(sp_linalg_lu_solve(A, bvec), out, rtol=1e-6)
     ...:         if cr == 'real':
     ...:             assert_allclose(sp_linalg_cho_solve(A, bvec),
     ...:                             out, rtol=1e-6)
     ...:             assert_allclose(sp_linalg_cho_banded(Ab, bvec),
     ...:                             out, rtol=1e-6)
     ...:
     ...:         # Test speed
     ...:         print('  np.linalg.solve      : ', end='')
     ...:         %timeit np_linalg_solve(A, bvec)
     ...:
     ...:         print('  sp.linalg.solve      : ', end='')
     ...:         %timeit sp_linalg_solve(A, bvec)
     ...:
     ...:         print('  sp.linalg.lu_solve   : ', end='')
     ...:         %timeit sp_linalg_lu_solve(A, bvec)
     ...:
     ...:         if cr == 'real':
     ...:
     ...:             print('  sp.linalg.cho_solve  : ', end='')
     ...:             %timeit sp_linalg_cho_solve(A, bvec)
     ...:
     ...:             print('  sp.linalg.cho_banded : ', end='')
     ...:             %timeit sp_linalg_cho_banded(Ab, bvec)
     ...:
     ...:         print('  emg3d.solve          : ', end='')
     ...:         %timeit emg3d_solve(amat, bvec)
     ...:
     ...:         print('  emg3d.solve2         : ', end='')
     ...:         %timeit emg3d_solve2(amat, bvec)
     ...:
     ...:         print(80*'-')
