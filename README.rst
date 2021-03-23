.. image:: https://raw.githubusercontent.com/emsig/emg3d-logo/master/logo-emg3d-transp-web250px.png
   :target: https://emsig.github.io
   :alt: emg3d logo
   
----

.. image:: https://img.shields.io/pypi/v/emg3d.svg
   :target: https://pypi.python.org/pypi/emg3d
   :alt: PyPI
.. image:: https://img.shields.io/conda/v/conda-forge/emg3d.svg
   :target: https://anaconda.org/conda-forge/emg3d
   :alt: conda-forge
.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :alt: Supported Python Versions
.. image:: https://img.shields.io/badge/platform-linux,win,osx-blue.svg
   :alt: Linux, Windows, OSX

|

A multigrid solver for 3D electromagnetic diffusion in Python.

- **Website:** https://emsig.github.io
- **Documentation:** https://emg3d.readthedocs.io
- **Source Code:** https://github.com/emsig/emg3d
- **Bug reports:** https://github.com/emsig/emg3d/issues
- **Contributing:** https://emg3d.readthedocs.io/en/latest/development
- **Contact:** http://slack.simpeg.xyz
- **Zenodo:** https://doi.org/10.5281/zenodo.3229006


Features
========

- **Iterative, matrix-free multigrid solver**, scaling linearly (CPU & RAM)
  with the number of unknowns, O(N).
- Uses **regular, stretched grids**.
- Handles **triaxial electrical anisotropy**, isotropic electric permittivity,
  and isotropic magnetic permeability.
- Written **purely in Python** using the NumPy/SciPy-stack, where the most time-
  and memory-consuming parts are sped up through jitted **Numba**-functions;
  works **cross-platform** on Linux, Mac, and Windows.
- Can solve in the complex-valued **frequency domain** or the real-valued
  **Laplace domain**. Includes routines to compute the 3D EM field in the
  **time domain**.
- **Command-line interface (CLI)**, through which emg3d can be used as forward
  modelling kernel in inversion routines.
- Computes the **gradient of the misfit function** using the adjoint-state
  method.
- Can handle **entire surveys** with **many sources, receivers, and
  frequencies**, computing the solution in **parallel**.


Installation
------------

Installable with ``pip`` from PyPI and with ``conda`` through the
``conda-forge`` channel. Minimum requirements are Python version 3.7 or higher
and the modules ``scipy`` and ``numba``. Various other packages are recommended
or required for some advanced functionalities (``xarray``, ``discretize``,
``matplotlib``, ``h5py``, ``empymod``, ``scooby``). Consult the installation
notes in the `manual
<https://emg3d.readthedocs.io/en/stable/user_guide/installation.html>`_ for
more information regarding installation, requirements, and soft dependencies.
