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

Required are Python version 3.7 or higher and the modules ``NumPy`` and
``SciPy``, ``Numba``, and ``empymod``; ``discretize`` (from `SimPEG
<https://simpeg.xyz>`_) is highly recommended.

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

Here we show a *very* basic example. To see some more realistic models have a
look at the `gallery <https://empymod.github.io/emg3d-gallery>`_. This
particular example is also there, with some further explanations and examples
to show how to plot the model and the data; see `Minimum working example
<https://empymod.github.io/emg3d-gallery/gallery/tutorials/minimum_example.html>`_.
It also contains an example without using ``discretize``.

First, we load ``emg3d`` and ``discretize`` (to create a mesh), along with
``numpy``:

.. code-block:: python

    >>> import emg3d
    >>> import discretize
    >>> import numpy as np


First, we define the mesh (see :class:`discretize.TensorMesh` for more info).
In reality, this task requires some careful considerations. E.g., to avoid edge
effects, the mesh should be large enough in order for the fields to dissipate,
yet fine enough around source and receiver to accurately model them. This grid
is too small, but serves as a minimal example.

.. code-block:: python

    >>> grid = discretize.TensorMesh(
    >>>         [[(25, 10, -1.04), (25, 28), (25, 10, 1.04)],
    >>>          [(50, 8, -1.03), (50, 16), (50, 8, 1.03)],
    >>>          [(30, 8, -1.05), (30, 16), (30, 8, 1.05)]],
    >>>         x0='CCC')
    >>> print(grid)

      TensorMesh: 49,152 cells

                          MESH EXTENT             CELL WIDTH      FACTOR
      dir    nC        min           max         min       max      max
      ---   ---  ---------------------------  ------------------  ------
       x     48       -662.16        662.16     25.00     37.01    1.04
       y     32       -857.96        857.96     50.00     63.34    1.03
       z     32       -540.80        540.80     30.00     44.32    1.05


Next we define a very simple fullspace model with
:math:`\rho_x=1.5\,\Omega\,\text{m}`, :math:`\rho_y=1.8\,\Omega\,\text{m}`, and
:math:`\rho_z=3.3\,\Omega\,\text{m}`. The source is an x-directed dipole at the
origin, with a 10 Hz signal of 1 A.

.. code-block:: python

    >>> model = emg3d.models.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)
    >>> sfield = emg3d.fields.get_source_field(
    >>>     grid, src=[0, 0, 0, 0, 0], freq=10.0)

Now we can calculate the electric field with ``emg3d``:

.. code-block:: python

    >>> efield = emg3d.solve(grid, model, sfield, verb=3)

    :: emg3d START :: 15:24:40 :: v0.9.1

       MG-cycle       : 'F'                 sslsolver : False
       semicoarsening : False [0]           tol       : 1e-06
       linerelaxation : False [0]           maxit     : 50
       nu_{i,1,c,2}   : 0, 2, 1, 2          verb      : 3
       Original grid  :  48 x  32 x  32     => 49,152 cells
       Coarsest grid  :   3 x   2 x   2     => 12 cells
       Coarsest level :   4 ;   4 ;   4

       [hh:mm:ss]  rel. error                  [abs. error, last/prev]   l s

           h_
          2h_ \                  /
          4h_  \          /\    /
          8h_   \    /\  /  \  /
         16h_    \/\/  \/    \/

       [11:18:17]   2.623e-02  after   1 F-cycles   [1.464e-06, 0.026]   0 0
       [11:18:17]   2.253e-03  after   2 F-cycles   [1.258e-07, 0.086]   0 0
       [11:18:17]   3.051e-04  after   3 F-cycles   [1.704e-08, 0.135]   0 0
       [11:18:17]   5.500e-05  after   4 F-cycles   [3.071e-09, 0.180]   0 0
       [11:18:18]   1.170e-05  after   5 F-cycles   [6.531e-10, 0.213]   0 0
       [11:18:18]   2.745e-06  after   6 F-cycles   [1.532e-10, 0.235]   0 0
       [11:18:18]   6.873e-07  after   7 F-cycles   [3.837e-11, 0.250]   0 0

       > CONVERGED
       > MG cycles        : 7
       > Final rel. error : 6.873e-07

    :: emg3d END   :: 15:24:42 :: runtime = 0:00:02

So the calculation required seven multigrid F-cycles and took just a bit more
than 2 seconds. It was able to coarsen in each dimension four times, where the
input grid had 49,152 cells, and the coarsest grid had 12 cells.


Related ecosystem
-----------------

The hard dependencies for emg3d are with ``numpy``, ``scipy``, ``numba``, and
``empymod`` comparably low. However, emg3d is, as such, "only" a solver. It
does not contain fancy grid- nor model-creation routines or plotting functions.
There exist other packages which do that much better.

To create advanced meshes it is recommended to use `discretize
<https://discretize.simpeg.xyz>`_ from the SimPEG framework. It also comes with
some neat plotting functionalities to plot model parameters and resulting
fields. Furthermore, it can serve as a link to use `PyVista
<https://docs.pyvista.org>`_ to create nice 3D plots even within a notebook.

Projects which can be used to compare or validate the results are, e.g.,
`empymod <https://empymod.github.io>`_ for layered models or `SimPEG
<https://simpeg.xyz>`_ for 3D models. It is also possible to create a
geological model with `GemPy <https://www.gempy.org>`_ and, again via
discretize, move it to emg3d to calculate CSEM responses for it.

Have a look at the `gallery <https://empymod.github.io/emg3d-gallery>`_ for
many examples of how to use emg3d together with the mentioned projects and
more!


Tipps and Tricks
----------------

The function :func:`emg3d.solve` is the main entry point, and it takes care
whether multigrid is used as a solver or as a preconditioner (or not at all),
while the actual multigrid solver is :func:`emg3d.solver.multigrid`. Most input
parameters for :func:`emg3d.solve` are sufficiently described in its docstring.
Here a few additional information.

- You can input any three-dimensional grid into `emg3d`. However, the
  implemented multigrid technique works with the existing nodes, meaning there
  are no new nodes created as coarsening is done by combining adjacent
  cells. The more times the grid dimension can be divided by two the better it
  is suited for MG. Ideally, the dimension of the coarsest grid should be a low
  prime number :math:`p`, for which good sizes can then be calculated with
  :math:`p 2^n`. Good grid sizes (in each direction) up to 1024 are

  - :math:`2·2^{0, 1, ..., 9}`: 2,  4,  8, 16,  32,  64, 128, 256, 512, 1024,
  - :math:`3·2^{0, 1, ..., 8}`: 3,  6, 12, 24,  48,  96, 192, 384, 768,
  - :math:`5·2^{0, 1, ..., 7}`: 5, 10, 20, 40,  80, 160, 320, 640,
  - :math:`7·2^{0, 1, ..., 7}`: 7, 14, 28, 56, 112, 224, 448, 896,

  and preference decreases from top to bottom row. Good grid sizes in
  sequential order: 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40,
  48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640,
  768, 896, 1024.

- The multigrid method can be used as a solver or as a preconditioner, for
  instance for BiCGSTAB. Using multigrid as a preconditioner for BiCGSTAB
  together with semicoarsening and line relaxation is the most stable version,
  but expensive, and therefore only recommended on highly stretched grids.
  Which combination of solver is best (fastest) depends to a large extent on
  the grid stretching. As a rule of thumb:

  - No stretching: Multigrid (MG);
  - Moderate stretching (< 1.04): BiCGSTAB with MG as pre-conditioner;
  - Strong stretching (> 1.04): BicGSTAB with MG as preconditioner and
    line relaxation/semicoarsening.


Contributing and Roadmap
------------------------

New contributions, bug reports, or any kind of feedback is always welcomed!
Have a look at the `Roadmap-project
<https://github.com/empymod/emg3d/projects/1>`_ to get an idea of things that
could be implemented. The GitHub `issues
<https://github.com/empymod/emg3d/issues>`_ and
`PR's <https://github.com/empymod/emg3d/pulls>`_ are also a good starting
point. The best way for interaction is at https://github.com/empymod or by
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

The structure of ``emg3d`` is:

- ``solver``: These are the main routines, the flow of the multigrid method;
- ``njited``: The expensive parts (computation, memory) are here in jitted
  functions; and
- ``utils``: Some helper routines.


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
`empymod/emg3d-asv <https://github.com/empymod/emg3d-asv>`_-repository. The
results of my machine can be found in the `empymod/emg3d-bench
<https://github.com/empymod/emg3d-bench>`_, its rendered version at
`empymod.github.io/emg3d-asv <https://empymod.github.io/emg3d-asv>`_.


License
-------

Copyright 2018-2020 The emg3d Developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
