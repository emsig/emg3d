.. image:: https://raw.githubusercontent.com/empymod/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://github.com/empymod/emg3d
   :alt: emg3d logo
   
----

.. sphinx-inclusion-marker

A multigrid solver for 3D electromagnetic diffusion with tri-axial electrical
anisotropy. The matrix-free solver can be used as main solver or as
preconditioner for one of the Krylov subspace methods implemented in
:mod:`scipy.sparse.linalg`, and the governing equations are discretized on a
staggered Yee grid. The code is written completely in Python using the
``numpy``/``scipy``-stack, where the most time- and memory-consuming parts are
sped up through jitted ``numba``-functions.


More information
================
For more information regarding installation, usage, contributing, roadmap, bug
reports, and much more, see

- **Website**: TODO,
- **Documentation**: TODO,
- **Source Code**: https://github.com/empymod/emg3d,
- **Examples**: TODO


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

Copyright 2018-2019 Dieter Werthmüller; TU Delft.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
