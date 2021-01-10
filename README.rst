.. image:: https://raw.githubusercontent.com/emsig/emg3d-logo/master/logo-emg3d-transp-web250px.png
   :target: https://emsig.github.io
   :alt: emg3d logo
   
----

.. image:: https://readthedocs.org/projects/emg3d/badge/?version=latest
   :target: https://emg3d.readthedocs.io/en/latest
   :alt: Documentation Status
.. image:: https://github.com/emsig/emg3d/workflows/pytest/badge.svg?branch=master
   :target: https://github.com/emsig/emg3d/actions
   :alt: GitHub Actions
.. image:: https://coveralls.io/repos/github/emsig/emg3d/badge.svg?branch=master
   :target: https://coveralls.io/github/emsig/emg3d?branch=master
   :alt: Coveralls
.. image:: https://app.codacy.com/project/badge/Grade/0412e617e8cd42fea05303fe490b09b5
   :target: https://www.codacy.com/gh/emsig/emg3d/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=emsig/emg3d&amp;utm_campaign=Badge_Grade
   :alt: Codacy
.. image:: https://img.shields.io/badge/benchmark-asv-blue.svg?style=flat
   :target: https://emsig.github.io/emg3d-asv
   :alt: Airspeed Velocity
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3229006.svg
   :target: https://doi.org/10.5281/zenodo.3229006
   :alt: Zenodo DOI

.. sphinx-inclusion-marker

A multigrid solver for 3D electromagnetic diffusion with tri-axial electrical
anisotropy. The matrix-free solver can be used as main solver or as
preconditioner for one of the Krylov subspace methods implemented in
`scipy.sparse.linalg`, and the governing equations are discretized on a
staggered Yee grid. The code is written completely in Python using the
NumPy/SciPy-stack, where the most time- and memory-consuming parts are sped up
through jitted numba-functions.


More information
================
For more information regarding installation, usage, contributing, roadmap, bug
reports, and much more, see

- **Website**: https://emsig.github.io,
- **Documentation**: https://emg3d.readthedocs.io,
- **Source Code**: https://github.com/emsig/emg3d,
- **Examples**: https://emsig.github.io/emg3d-gallery.


Features
========

- Multigrid solver for 3D electromagnetic (EM) diffusion with regular grids
  (where source and receiver can be electric or magnetic).
- Compute the 3D EM field in the complex frequency domain or in the real
  Laplace domain.
- Includes also routines to compute the 3D EM field in the time domain.
- Can be used together with the `SimPEG <https://simpeg.xyz>`_-framework.
- Can be used as a standalone solver or as a pre-conditioner for various Krylov
  subspace methods implemented in SciPy, e.g., BiCGSTAB
  (`scipy.sparse.linalg.bicgstab`) or CGS (`scipy.sparse.linalg.cgs`).
- Tri-axial electrical anisotropy.
- Isotropic magnetic permeability.
- Semicoarsening and line relaxation.
- Grid-size can be anything.
- As a multigrid method it scales with the number of unknowns *N* and has
  therefore optimal complexity *O(N)*.


Installation
============

You can install emg3d either via ``conda`` (preferred):

.. code-block:: console

   conda install -c conda-forge emg3d

or via ``pip``:

.. code-block:: console

   pip install emg3d

Minimum requirements are Python version 3.7 or higher and the modules ``scipy``
and ``numba``. Various other packages are recommended or required for some
advanced functionalities (``xarray``, ``discretize``, ``matplotlib``, ``h5py``,
``empymod``, ``scooby``). Consult the installation notes in the `manual
<https://emg3d.readthedocs.io/en/stable/usage.html#installation>`_ for more
information regarding installation, requirements, and soft dependencies.


Citation
========

If you publish results for which you used `emg3d`, please give credit by citing
`Werthmüller et al. (2019) <https://doi.org/10.21105/joss.01463>`_:

    Werthmüller, D., W. A. Mulder, and E. C. Slob, 2019,
    emg3d: A multigrid solver for 3D electromagnetic diffusion:
    Journal of Open Source Software, 4(39), 1463;
    DOI: `10.21105/joss.01463 <https://doi.org/10.21105/joss.01463>`_.


All releases have a Zenodo-DOI, which can be found on `10.5281/zenodo.3229006
<https://doi.org/10.5281/zenodo.3229006>`_.

See `CREDITS` for the history of the code.


License information
===================

Copyright 2018-2021 The emg3d Developers.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
