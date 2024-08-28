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
   simulations
   solver
   surveys
   time
   utils
   inversion/index


.. grid:: 1
    :gutter: 2

    .. grid-item-card::

        Grid: :class:`emg3d.meshes.TensorMesh`

    .. grid-item-card::

        Model: :class:`emg3d.models.Model`

    .. grid-item-card::

        Survey: :class:`emg3d.surveys.Survey`

    .. grid-item-card::

        Simulation: :class:`emg3d.simulations.Simulation`

    .. grid-item-card::

        Solver: :func:`emg3d.solver.solve`

    .. grid-item-card::

        All sources and receivers are in :mod:`emg3d.electrodes`

    .. grid-item-card::

        Inversion: :mod:`emg3d.inversion`
