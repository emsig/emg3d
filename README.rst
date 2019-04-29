.. image:: https://raw.githubusercontent.com/empymod/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://empymod.github.io
   :alt: emg3d logo
   
----

.. sphinx-inclusion-marker

A multigrid solver for 3D electromagnetic diffusion with tri-axial electrical
anisotropy. The matrix-free solver can be used as main solver or as
preconditioner for one of the Krylov subspace methods implemented in
:mod:`scipy.sparse.linalg`, and the governing equations are discretized on a
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

- Multigrid solver for 3D electromagnetic diffusion with regular grids (using
  :class:`discretize.TensorMesh`).
- Tri-axial electrical anisotropy.
- Can be used as a standalone solver or as a pre-conditioner for various Krylov
  subspace methods implemented in SciPy, e.g., BiCGSTAB
  (:func:`scipy.sparse.linalg.bicgstab`) or CGS
  (:func:`scipy.sparse.linalg.cgs`).
- Semicoarsening and line relaxation.
- Grid-size can be anything.


Installation
============

You can install emg3d either via ``conda``:

.. code-block:: console

   conda install -c prisae emg3d

or via ``pip``:

.. code-block:: console

   pip install emg3d

Required are Python version 3.7 or higher and the modules `NumPy`, `SciPy`,
`numba`, and `discretize` (from SimPEG). Consult the installation notes in the
`manual <https://emg3d.readthedocs.io/en/stable/usage.html#installation>`_ for
more information regarding installation and requirements.


Citation
========

If you publish results for which you used `emg3d`, please give credit by citing
us. We will soon submit an article to `JOSS <https://joss.theoj.org>`_, and
will post here the details as soon as we have them.

All releases have a Zenodo-DOI, provided on the `release-page
<https://github.com/empymod/emg3d/releases>`_.

See :doc:`credits` for the history of the code.


License information
===================

Copyright 2018-2019 The emg3d Developers.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
