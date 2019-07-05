.. image:: https://raw.githubusercontent.com/empymod/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://empymod.github.io
   :alt: emg3d logo
   
----

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

- Multigrid solver for 3D electromagnetic diffusion with regular grids.
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

   conda install -c prisae emg3d

or via ``pip``:

.. code-block:: console

   pip install emg3d

Required are Python version 3.7 or higher and the modules ``NumPy``, ``SciPy``,
and ``numba``; ``discretize`` (from `SimPEG <https://simpeg.xyz>`_) is highly
recommended.

If you are new to Python we recommend using a Python distribution, which will
ensure that all dependencies are met, specifically properly compiled versions
of ``NumPy`` and ``SciPy``; we recommend using `Anaconda
<https://www.anaconda.com/download>`_. If you install Anaconda you can simply
start the *Anaconda Navigator*, add the channel ``prisae`` and ``emg3d`` will
appear in the package list and can be installed with a click.


Citation
========

If you publish results for which you used `emg3d`, please give credit by citing
`Werthmüller et al. (2019)
<http://joss.theoj.org/papers/d559f2dbd8538007937797122887df0c>`_:

    Werthmüller, D., W. A. Mulder, and E. C. Slob, 2019, emg3d: A multigrid
    solver for 3D electromagnetic diffusion: submitted to the Journal of Open
    Source Software, 4(37), 1463; DOI:  `10.21105/joss.01463
    <http://joss.theoj.org/papers/d559f2dbd8538007937797122887df0c>`_.


All releases have a Zenodo-DOI, provided on the `release-page
<https://github.com/empymod/emg3d/releases>`_.

See `CREDITS` for the history of the code.


License information
===================

Copyright 2018-2019 The emg3d Developers.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
