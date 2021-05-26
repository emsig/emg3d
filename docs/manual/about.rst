.. _about:

About
=====

The code ``emg3d`` [WeMS19]_ is a three-dimensional modeller for
electromagnetic (EM) diffusion as used for instance in geophysical
controlled-source EM (CSEM) surveys. This includes use cases in the search for
resources such as groundwater, geothermal energy, hydrocarbons, and minerals,
or civil engineering and environmental applications.

The core of the code is primarily based on [Muld06]_, [Muld07]_, and [Muld08]_.
You can read more about the background of the code in the :doc:`credits`. An
introduction to the underlying theory of multigrid methods is given in the
:doc:`theory`, and further literature is provided in the :doc:`references`. The
code is currently restricted to regular, stretched grids. As a matrix-free
multigrid solver it scales linearly with the number of cells for both CPU and
RAM. This makes it possible to use emg3d for models with several millions of
cells on a regular laptop.



What is emg3d? (Features)
-------------------------

- A community driven, open-source 3D CSEM modelling tool.
- Can handle **entire surveys** with **many sources, receivers, and
  frequencies**, computing the solution in **parallel**.
- Can model electric and magnetic dipoles and arbitrarily shaped electric
  wires.
- Computes the **gradient of the misfit function** using the adjoint-state
  method.
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
  modelling kernel in inversion routines written in any language.


What is it _not_?
-----------------

- The code is meant to be used in Python or in a terminal. There is **no** GUI.
- Some knowledge of EM fields in particular and numerical modelling in general
  is definitely helpful, as GIGO applies («garbage in, garbage out»). For
  example, placing your receivers very close to the computational boundary
  *will* result in bad or wrong responses.
- It is not a model builder; there are other tools that can be used to generate
  complex geological models, for instance `GemPy <https://www.gempy.org>`_.


Related ecosystem
-----------------

To create advanced meshes it is recommended to use `discretize
<https://discretize.simpeg.xyz>`_ from the `SimPEG <https://simpeg.xyz>`_
framework. It also comes with some neat plotting functionalities to plot model
parameters and resulting fields. Furthermore, it can serve as a link to use
`PyVista <https://docs.pyvista.org>`_ to create nice 3D plots even within a
notebook.

There are some first successful attempts of using emsig as a forward modeller
in both `SimPEG <https://simpeg.xyz>`_ and `pyGIMLi <https://pygimli.org>`_
inversions. Get in touch if you are interested in these developments.

See also the note about the `EM & Potential Geo-Exploration Python Ecosystem
<https://emsig.xyz/#related-ecosystem>`_ on `emsig.xyz <https://emsig.xyz>`_.
