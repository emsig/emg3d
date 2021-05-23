.. _installation:

Installation
============

You can install emg3d either via ``conda``:

.. code-block:: console

   conda install -c conda-forge emg3d

or via ``pip``:

.. code-block:: console

   pip install emg3d

Minimum requirements are Python version 3.7 or higher and the modules ``scipy``
and ``numba``. Various other packages are recommended or required for some
advanced functionalities, namely:

- ``xarray``: For the :class:`emg3d.surveys.Survey` and
  :class:`emg3d.simulations.Simulation` classes (many sources and receivers at
  once).
- ``discretize``: For advanced meshing tools (fancy mesh-representations and
  plotting utilities).
- ``matplotlib``: To use the plotting utilities within ``discretize``.
- ``h5py``: Save and load data in the HDF5 format.
- ``empymod``: Time-domain modelling (:class:`emg3d.utils.Fourier`).
- ``scooby``: For the version and system report (:class:`emg3d.utils.Report`).

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
