.. _GettingStarted:

Getting started
###############

The code ``emg3d`` ([WeMS19]_) is a three-dimensional modeller for
electromagnetic (EM) diffusion as used, for instance, in controlled-source EM
(CSEM) surveys frequently applied in the search for, amongst other,
groundwater, hydrocarbons, and minerals.

The core of the code is primarily based on [Muld06]_, [Muld07]_, and [Muld08]_.
You can read more about the background of the code in the chapter
:doc:`credits`. An introduction to the underlying theory of multigrid methods
is given in the chapter :doc:`theory`, and further literature is provided in
the :doc:`references`.


Installation
------------

You can install emg3d either via ``conda``:

.. code-block:: console

   conda install -c conda-forge emg3d

or via ``pip``:

.. code-block:: console

   pip install emg3d

Minimum requirements are Python version 3.7 or higher and the modules ``scipy``
and ``numba``. Various other packages are recommended or required for some
advanced functionalities, namely:

- ``xarray``: For the ``Survey`` class (many sources and receivers at once).
- ``discretize``: For advanced meshing tools (fancy mesh-representations and
  plotting utilities).
- ``matplotlib``: To use the plotting utilities within ``discretize``.
- ``h5py``: Save and load data in the HDF5 format.
- ``empymod``: Time-domain modelling (``utils.Fourier``).
- ``scooby``: For the version and system report (``emg3d.Report()``).

If you are new to Python we recommend using a Python distribution, which will
ensure that all dependencies are met, specifically properly compiled versions
of ``NumPy`` and ``SciPy``; we recommend using `Anaconda
<https://www.anaconda.com/distribution>`_. If you install Anaconda you can
simply start the *Anaconda Navigator*, add the channel ``conda-forge`` and
``emg3d`` will appear in the package list and can be installed with a click.

Using NumPy and SciPy with the Intel Math Kernel Library (*mkl*) can
significantly improve computation time. You can check if ``mkl`` is used via
``conda list``: The entries for the BLAS and LAPACK libraries should contain
something with ``mkl``, not with ``openblas``. To enforce it you might have to
create a file ``pinned``, containing the line ``libblas[build=*mkl]`` in the
folder ``path-to-your-conda-env/conda-meta/``.


Basic Example
-------------

TODO : This section is outdated (text doesn't match code); code is now a basic
example executed directly when compiling the docs. However, numba is disabled
on RTD, so only tiny examples can run.

Here we show a *very* basic example. To see some more realistic models have a
look at the `gallery <https://emsig.github.io/emg3d-gallery>`_. This
particular example is also there, with some further explanations and examples
to show how to plot the model and the data; see `«Minimum working example»
<https://emsig.github.io/emg3d-gallery/gallery/tutorials/minimum_example.html>`_.
It also contains an example without using ``discretize``.

First, we load ``emg3d`` and ``discretize`` (to create a mesh), along with
``numpy``:

.. ipython::

    In [1]: import emg3d
       ...: import numpy as np
       ...: from matplotlib.colors import LogNorm

First, we define the mesh (see :class:`discretize.TensorMesh` for more info).
In reality, this task requires some careful considerations. E.g., to avoid edge
effects, the mesh should be large enough in order for the fields to dissipate,
yet fine enough around source and receiver to accurately model them. This grid
is too small, but serves as a minimal example.

.. ipython::

    In [2]: # Create a simple grid, 8 cells of length 2 in each direction,
       ...: # centered around the origin.
       ...: cw = 2*np.ones(8)
       ...: grid = emg3d.TensorMesh(h=[cw, cw, cw], origin=(-8, -8, -8))
       ...: grid
    Out[2]:  TensorMesh: 512 cells
       ...:
       ...:                      MESH EXTENT             CELL WIDTH      FACTOR
       ...:  dir    nC        min           max         min       max      max
       ...:  ---   ---  ---------------------------  ------------------  ------
       ...:   x      8         -8.00          8.00      2.00      2.00    1.00
       ...:   y      8         -8.00          8.00      2.00      2.00    1.00
       ...:   z      8         -8.00          8.00      2.00      2.00    1.00
       ...:

Next we define a very simple fullspace model with
:math:`\rho_x=1.5\,\Omega\,\text{m}`, :math:`\rho_y=1.8\,\Omega\,\text{m}`, and
:math:`\rho_z=3.3\,\Omega\,\text{m}`. The source is an x-directed dipole at the
origin, with a 10 Hz signal of 1 A.

.. ipython::

    In [3]: # The model is a fullspace with tri-axial anisotropy.
       ...: model = emg3d.Model(grid, property_x=1.5, property_y=1.8,
       ...:                     property_z=3.3, mapping='Resistivity')
       ...: model
    Out[3]:    Model [resistivity]; tri-axial; 8 x 8 x 8 (512)

    In [4]: # The source is a x-directed, horizontal dipole at (0, 0, 0)
       ...: # with a frequency of 1 Hz.
       ...: sfield = emg3d.fields.get_source_field(
       ...:         grid, src=[0, 0, 0, 0, 0], freq=1)

Now we can compute the electric field with ``emg3d``:

.. ipython::

    In [5]: # Compute the electric signal.
       ...: efield = emg3d.solve(model, sfield, verb=4)
    Out[5]: :: emg3d START :: 11:14:25 :: v1.0.0
       ...:
       ...: MG-cycle       : 'F'                 sslsolver : False
       ...: semicoarsening : False [0]           tol       : 1e-06
       ...: linerelaxation : False [0]           maxit     : 50
       ...: nu_{i,1,c,2}   : 0, 2, 1, 2          verb      : 4
       ...: Original grid  :   8 x   8 x   8     => 512 cells
       ...: Coarsest grid  :   2 x   2 x   2     => 8 cells
       ...: Coarsest level :   2 ;   2 ;   2     :: Grid not optimal for MG solver ::
       ...:
       ...: [hh:mm:ss]  rel. error                  [abs. error, last/prev]   l s
       ...:
       ...:     h_
       ...:     2h_ \    /
       ...:     4h_  \/\/
       ...:
       ...: [11:14:40]   2.284e-02  after   1 F-cycles   [1.275e-06, 0.023]   0 0
       ...: [11:14:40]   1.565e-03  after   2 F-cycles   [8.739e-08, 0.069]   0 0
       ...: [11:14:40]   1.295e-04  after   3 F-cycles   [7.232e-09, 0.083]   0 0
       ...: [11:14:40]   1.197e-05  after   4 F-cycles   [6.685e-10, 0.092]   0 0
       ...: [11:14:40]   1.233e-06  after   5 F-cycles   [6.886e-11, 0.103]   0 0
       ...: [11:14:40]   1.415e-07  after   6 F-cycles   [7.899e-12, 0.115]   0 0
       ...:
       ...: > CONVERGED
       ...: > MG cycles        : 6
       ...: > Final rel. error : 1.415e-07
       ...:
       ...: :: emg3d END   :: 11:14:40 :: runtime = 0:00:15

So the computation required seven multigrid F-cycles and took just a bit more
than 2 seconds. It was able to coarsen in each dimension four times, where the
input grid had 49,152 cells, and the coarsest grid had 12 cells.

.. ipython::

    @savefig basic_example.png width=4in
    In [1]: # Get cell-averaged values of the real component.
       ...: ccr_efield = grid.aveE2CCV * efield.real
       ...:
       ...: grid.plot_slice(
       ...:     ccr_efield, normal='Y', v_type='CCv', view='vec',
       ...:     pcolor_opts={'norm': LogNorm()},
       ...: );


Related ecosystem
-----------------

To create advanced meshes it is recommended to use `discretize
<https://discretize.simpeg.xyz>`_ from the SimPEG framework. It also comes with
some neat plotting functionalities to plot model parameters and resulting
fields. Furthermore, it can serve as a link to use `PyVista
<https://docs.pyvista.org>`_ to create nice 3D plots even within a notebook.

Projects which can be used to compare or validate the results are, e.g.,
`empymod <https://emsig.github.io>`_ for layered models or `SimPEG
<https://simpeg.xyz>`_ for 3D models. It is also possible to create a
geological model with `GemPy <https://www.gempy.org>`_ and, again via
discretize, move it to emg3d to compute CSEM responses for it.

Have a look at the `gallery <https://emsig.github.io/emg3d-gallery>`_ for
many examples of how to use emg3d together with the mentioned projects and
more!


Tips and Tricks
---------------

The function :func:`emg3d.solver.solve` is the main entry point, and it takes
care whether multigrid is used as a solver or as a preconditioner (or not at
all), while the actual multigrid solver is :func:`emg3d.solver.multigrid`. Most
input parameters for :func:`emg3d.solver.solve` are sufficiently described in
its docstring. Here a few additional information.

- You can input any three-dimensional tensor mesh into `emg3d`. However, the
  implemented multigrid technique works with the existing nodes, meaning there
  are no new nodes created as coarsening is done by combining adjacent
  cells. The more times the grid dimension can be divided by two the better it
  is suited for MG. Ideally, the number should be dividable by two a few times
  and the dimension of the coarsest grid should be a low prime number
  :math:`p`, for which good sizes can then be computed with :math:`p 2^n`. Good
  grid sizes (in each direction) up to 1024 are

  - :math:`2·2^{3, 4, ..., 9}`: 16,  32,  64, 128, 256, 512, 1024,
  - :math:`3·2^{3, 4, ..., 8}`: 24,  48,  96, 192, 384, 768,
  - :math:`5·2^{3, 4, ..., 7}`: 40,  80, 160, 320, 640,
  - :math:`7·2^{3, 4, ..., 7}`: 56, 112, 224, 448, 896,

  and preference decreases from top to bottom row (stick to the first two or
  three rows if possible). Good grid sizes in sequential order, excluding p=7:
  16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768,
  1024. You can get this list via :func:`emg3d.meshes.good_mg_cell_nr()`.

- The multigrid method can be used as a solver or as a preconditioner, for
  instance for BiCGSTAB. Using multigrid as a preconditioner for BiCGSTAB
  together with semicoarsening and line relaxation is the most stable version,
  but expensive, and therefore only recommended on highly stretched grids.
  Which combination of solver is best (fastest) depends to a large extent on
  the grid stretching, but also on anisotropy and general model complexity.
  See `«Parameter tests»
  <https://emsig.github.io/emg3d-gallery/gallery/tutorials/parameter_tests.html>`_
  in the gallery for an example how to run some tests on your particular
  problem.


Contributing and Roadmap
------------------------

New contributions, bug reports, or any kind of feedback is always welcomed!
Have a look at the `Roadmap-project
<https://github.com/emsig/emg3d/projects/1>`_ to get an idea of things that
could be implemented. The GitHub `issues
<https://github.com/emsig/emg3d/issues>`_ and
`PR's <https://github.com/emsig/emg3d/pulls>`_ are also a good starting
point. The best way for interaction is at https://github.com/emsig or by
joining the `Slack channel <http://slack.simpeg.xyz>`_ «em-x-d» of SimPEG. If
you prefer to get in touch outside of GitHub/Slack use the contact form on
https://werthmuller.org.

To install emg3d from source, you can download the latest version from GitHub
and install it in your python distribution via:

.. code-block:: console

   python setup.py install

Please make sure your code follows the pep8-guidelines by using, for instance,
the python module ``flake8``, and also that your code is covered with
appropriate tests. Just get in touch if you have any doubts.


Tests and benchmarks
--------------------

The modeller comes with a test suite using ``pytest``. If you want to run the
tests, just install ``pytest`` and run it within the ``emg3d``-top-directory.

.. code-block:: console

    > pytest --cov=emg3d --flake8

It should run all tests successfully. Please let us know if not!

Note that installations of ``em3gd`` via conda or pip do not have the
test-suite included. To run the test-suite you must download ``emg3d`` from
GitHub.

There is also a benchmark suite using *airspeed velocity*, located in the
`emsig/emg3d-asv <https://github.com/emsig/emg3d-asv>`_-repository. The results
of my machine can be found in the `emsig/emg3d-bench
<https://github.com/emsig/emg3d-bench>`_, its rendered version at
`emsig.github.io/emg3d-asv <https://emsig.github.io/emg3d-asv>`_.


License
-------

Copyright 2018-2021 The emg3d Developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
