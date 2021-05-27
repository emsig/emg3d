Tests and Benchmarks
====================

.. todo::

    Rework this for version 1.

.. todo::

    Add solve example/benchmark

The modeller comes with a test suite using ``pytest``. If you want to run the
tests, just install ``pytest`` and run it within the ``emg3d``-top-directory.

.. code-block:: console

    > make pytest

It should run all tests successfully. Please let us know if not!

Note that installations of ``em3gd`` via conda or pip do not have the
test-suite included. To run the test-suite you must download or clone ``emg3d``
from GitHub.

There is also a benchmark suite using *airspeed velocity*, located in the
`emsig/emg3d-asv <https://github.com/emsig/emg3d-asv>`_-repository. The results
of my machine can be found in the `emsig/emg3d-bench
<https://github.com/emsig/emg3d-bench>`_, its rendered version at
`emsig.xyz/emg3d-asv <https://emsig.xyz/emg3d-asv>`_.


.. _improve-cpu-ram:

CPU & RAM
---------

.. todo::

    Currently just copied over from :ref:`info-tips-tricks`.

The multigrid method is attractive because it shows optimal scaling for both
runtime and memory consumption. In the following are a few notes regarding
memory and runtime requirements. It also contains information about what has
been tried and what still could be tried in order to improve the current code.


Runtime
```````

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
``````

Most of the memory requirement comes from storing the data itself, mainly the
fields (source field, electric field, and residual field) and the model
parameters (resistivity, eta, mu). For a big model, they some up; e.g., almost
3 GB for an isotropic model with 256x256x256 cells.

The theory of multigrid says that in an ideal scenario, multigrid requires
8/7 (a bit over 1.14) the memory requirement of carrying out one Gauss-Seidel
step on the finest grid. As can be seen in the figure, for models up to 2
million cells that holds pretty much, afterwards it becomes a bit worse.

However, for this estimation one has to run the model first. Another way to
estimate the requirement is by starting from the RAM used to store the fields
and parameters. As can be seen in the figure, for big models one is on the
save side estimating the required RAM as 1.35 times the storage required for
the fields and model parameters.

The figure also shows nicely the linear behaviour of multigrid; for twice the
number of cells twice the memory is required (from a certain size onwards).

**Attempts at improving memory usage should focus on the difference between the
red line (actual usage) and the dashed black line (1.14 x base usage).**

Scripts
```````

To test CPU and RAM on your machine, you can use and adjust the following
script. The old notebooks which were used to generate the above figures can be
found at

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
