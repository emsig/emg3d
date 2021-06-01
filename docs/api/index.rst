.. _api:

#############
API reference
#############

:Release: |version|
:Date: |today|

----

.. module:: emg3d

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   electrodes
   fields
   io
   maps
   meshes
   models
   optimize
   simulations
   solver
   surveys
   time
   utils


.. panels::
    :container: container-lg pb-1
    :column: col-lg-12 p-2

    Grid: :class:`emg3d.meshes.TensorMesh`

    ---
    :column: col-lg-12 p-2

    Model: :class:`emg3d.models.Model`

    ---
    :column: col-lg-12 p-2

    Survey: :class:`emg3d.surveys.Survey`

    ---
    :column: col-lg-12 p-2

    Simulation: :class:`emg3d.simulations.Simulation`

    ---
    :column: col-lg-12 p-2

    Solver: :func:`emg3d.solver.solve`

    ---
    :column: col-lg-12 p-2

    All sources and receivers are in :mod:`emg3d.electrodes`
