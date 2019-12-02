.. image:: https://raw.githubusercontent.com/empymod/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://empymod.github.io
   :alt: emg3d logo
   
----

.. image:: https://readthedocs.org/projects/emg3d/badge/?version=latest
   :target: http://emg3d.readthedocs.io/en/latest
   :alt: Documentation Status
.. image:: https://travis-ci.org/empymod/emg3d.svg?branch=master
   :target: https://travis-ci.org/empymod/emg3d
   :alt: Travis-CI
.. image:: https://coveralls.io/repos/github/empymod/emg3d/badge.svg?branch=master
   :target: https://coveralls.io/github/empymod/emg3d?branch=master
   :alt: Coveralls
.. image:: https://img.shields.io/codacy/grade/a15b80f75cd64be3bca73da30f191a83/master.svg
   :target: https://www.codacy.com/app/prisae/emg3d
   :alt: Codacy
.. image:: https://img.shields.io/badge/benchmark-asv-blue.svg?style=flat
   :target: https://empymod.github.io/emg3d-asv
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

- **Website**: https://empymod.github.io,
- **Documentation**: https://emg3d.readthedocs.io,
- **Source Code**: https://github.com/empymod/emg3d,
- **Examples**: https://github.com/empymod/emg3d-examples.


Features
========

- Multigrid solver for 3D electromagnetic (EM) diffusion with regular grids
  (where source and receiver can be electric or magnetic).
- Calculate the 3D EM field in the complex frequency domain or in the real
  Laplace domain.
- Includes also routines to calculate the 3D EM field in the time domain.
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

Required are Python version 3.7 or higher and the modules ``NumPy``, ``SciPy``,
``numba``, and ``empymod``; ``discretize`` (from `SimPEG
<https://simpeg.xyz>`_) is highly recommended. Consult the installation notes
in the `manual
<https://emg3d.readthedocs.io/en/stable/usage.html#installation>`_ for more
information regarding installation and requirements.


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

Copyright 2018-2019 The emg3d Developers.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
