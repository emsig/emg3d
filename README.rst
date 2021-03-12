.. image:: https://raw.githubusercontent.com/emsig/emg3d-logo/master/logo-emg3d-transp-web250px.png
   :target: https://emsig.github.io
   :alt: emg3d logo
   
----

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3229006.svg
   :target: https://doi.org/10.5281/zenodo.3229006
   :alt: Zenodo DOI
.. image:: https://img.shields.io/badge/Slack-simpeg-4A154B.svg?logo=slack
    :target: http://slack.simpeg.xyz
.. image:: https://readthedocs.org/projects/emg3d/badge/?version=latest
   :target: https://emg3d.readthedocs.io/en/latest
   :alt: Documentation Status
.. image:: https://github.com/emsig/emg3d/workflows/pytest/badge.svg?branch=master
   :target: https://github.com/emsig/emg3d/actions
   :alt: GitHub Actions

====

.. sphinx-inclusion-marker


A multigrid solver for 3D electromagnetic diffusion in Python.

- **Website:** https://emsig.github.io
- **Documentation:** https://emg3d.readthedocs.io
- **Source Code:** https://github.com/emsig/emg3d
- **Bug reports:** https://github.com/emsig/emg3d/issues
- **Contributing:** https://emg3d.readthedocs.io/en/latest/development


Features
--------

- **Iterative, matrix-free multigrid solver**, scaling linearly (CPU & RAM)
  with the number of unknowns, O(N).
- Uses **regular, stretched grids**.
- Handles **tri-axial electrical anisotropy**, isotropic electric permittivity,
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
============

Installable with ``pip`` from PyPI and with ``conda`` through the
``conda-forge`` channel. Minimum requirements are Python version 3.7 or higher
and the modules ``scipy`` and ``numba``. Various other packages are recommended
or required for some advanced functionalities (``xarray``, ``discretize``,
``matplotlib``, ``h5py``, ``empymod``, ``scooby``). Consult the installation
notes in the `manual
<https://emg3d.readthedocs.io/en/stable/user_guide/installation.html>`_ for
more information regarding installation, requirements, and soft dependencies.
