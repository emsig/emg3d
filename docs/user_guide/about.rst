About
=====

The code ``emg3d`` ([WeMS19]_) is a three-dimensional modeller for
electromagnetic (EM) diffusion as used, for instance, in controlled-source EM
(CSEM) surveys frequently applied in the search for, amongst other,
groundwater, hydrocarbons, and minerals.

The core of the code is primarily based on [Muld06]_, [Muld07]_, and [Muld08]_.
You can read more about the background of the code in the chapter
:doc:`credits`. An introduction to the underlying theory of multigrid methods
is given in the chapter :doc:`theory`, and further literature is provided in
the :doc:`references`.


Shares all the advantages and disadvantages of iterative solvers as well as
regular grids.... blabla TODO.

TODO TODO TODO

from the readme:

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
