.. _about:

About
=====

The code ``emg3d`` [WeMS19]_ is a three-dimensional modeller for
electromagnetic (EM) diffusion as used, for instance, in controlled-source EM
(CSEM) surveys frequently applied in the search for resources such as
groundwater, geothermal energy, hydrocarbons, and minerals.

The core of the code is primarily based on [Muld06]_, [Muld07]_, and [Muld08]_.
You can read more about the background of the code in the chapter
:doc:`credits`. An introduction to the underlying theory of multigrid methods
is given in the chapter :doc:`theory`, and further literature is provided in
the :doc:`references`. The code is currently restricted to regular, stretched
grids. As a matrix-free multigrid solver it scales linearly with the number of
cells for both CPU and RAM. This makes it possible to use emg3d for models with
several millions of cells on a regular laptop.



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
- Some knowledge of EM fields is definitely helpful, as GIGO applies («garbage
  in, garbage out»). For example, placing your receivers very close to the
  computational boundary *will* result in bad or wrong responses.
- It is not a model builder; there are other tools that can be used to generate
  complex geological models, e.g., `GemPy <https://www.gempy.org>`_.


Related ecosystem
-----------------

To create advanced meshes it is recommended to use `discretize
<https://discretize.simpeg.xyz>`_ from the `SimPEG <https://simpeg.xyz>`_
framework. It also comes with some neat plotting functionalities to plot model
parameters and resulting fields. Furthermore, it can serve as a link to use
`PyVista <https://docs.pyvista.org>`_ to create nice 3D plots even within a
notebook.

`EMSiG <https://emsig.xyz>`_ with its codes `empymod
<https://empymod.emsig.xyz>`_ and `emg3d <https://emsig.emsig.xyz>`_ is part
of a bigger, fast growing, open-source **EM & Potential Geo-Exploration Python
Ecosystem**:

.. raw:: html

   <p>

   <a href=https://pygimli.org><img src="https://www.pygimli.org/_static/gimli_logo.svg" style="max-height: 2cm;"></a>

   <a href=https://simpeg.xyz><img src="https://raw.github.com/simpeg/simpeg/master/docs/images/simpeg-logo.png" style="max-height: 2.5cm;"></a>

   <a href=http://petgem.bsc.es><img src="http://petgem.bsc.es/_static/figures/petgem_logo.png" style="max-height: 3cm;"></a>

   <a href=https://gitlab.com/Rochlitz.R/custEM><img src="https://custem.readthedocs.io/en/latest/_static/custEMlogo.png" style="max-height: 1.5cm;"></a>

   <a href=https://docs.pyvista.org><img src="https://raw.githubusercontent.com/pyvista/pyvista/master/docs/_static/pyvista_logo_sm.png" style="max-height: 2.5cm;"></a>

   <a href=https://www.gempy.org><img src="https://raw.githubusercontent.com/cgre-aachen/gempy/master/docs/source/_static/logos/gempy.png" style="max-height: 2.5cm;"></a>

   <a href=https://www.fatiando.org><img src="https://raw.githubusercontent.com/fatiando/logo/master/fatiando-logo-background.png" style="max-height: 3cm;"></a>

   </p>

