"""
Minimum working example
=======================

This is a simple minimum working example to get started, along the lines of the
one given in https://emg3d.readthedocs.io/en/stable/usage.html#basic-example.

To see some more realistic models have a look at the other examples in this
gallery.

This notebooks uses ``discretize`` to create meshes easily and plot the model
as well as the resulting electric field, which also requires ``matplotlib``. If
you are interested in a basic example that only requires ``emg3d`` here it is:

.. code-block:: python

    import emg3d
    import numpy as np

    # Create a simple grid, 8 cells of length 1 in each direction,
    # starting at the origin.
    grid = emg3d.utils.TensorMesh(
            [np.ones(8), np.ones(8), np.ones(8)], x0=np.array([0, 0, 0]))

    # The model is a fullspace with tri-axial anisotropy.
    model = emg3d.utils.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)

    # The source is a x-directed, horizontal dipole at (4, 4, 4),
    # frequency is 10 Hz.
    sfield = emg3d.utils.get_source_field(grid, src=[4, 4, 4, 0, 0], freq=10.0)

    # Calculate the electric signal.
    efield = emg3d.solver.solver(grid, model, sfield, verb=3)

    # Get the corresponding magnetic signal.
    hfield = emg3d.utils.get_h_field(grid, model, efield)


First, we load ``emg3d`` and ``discretize`` (to create a mesh), along with
``numpy`` and ``matplotlib``:

"""
import emg3d
# import discretize
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
plt.style.use('ggplot')

###############################################################################
# 1. Mesh
# -------
#
# Here we define the mesh (see ``discretize.TensorMesh`` for more info). In
# reality, this task requires some careful considerations. E.g., to avoid edge
# effects, the mesh should be large enough in order for the fields to
# dissipate, yet fine enough around source and receiver to accurately model
# them. This grid is too small, but serves as a minimal example.

# With discretize
# grid = discretize.TensorMesh(
#         [np.ones(64)*25, np.ones(32)*50, np.ones(32)*30], x0=[0, 0, 0])

# Without discretize
grid = emg3d.utils.TensorMesh(
        [np.ones(64)*25, np.ones(32)*50, np.ones(32)*30],
        x0=[-800, -800, -500])
grid

###############################################################################
# 2. Model
# --------
#
# Next we define a very simple fullspace model with
# :math:`\rho_x=1.5\,\Omega\,\rm{m}`, :math:`\rho_y=1.8\,\Omega\,\rm{m}`, and
# :math:`\rho_z=3.3\,\Omega\,\rm{m}`.

model = emg3d.utils.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)

###############################################################################
# We can plot the model using ``discretize``; in this case it is obviously
# rather a boring plot, as it shows a homogeneous fullspace.

# grid.plot_3d_slicer(np.ones(grid.vnC)*model.res_x)  # x-resistivity

###############################################################################
# 3. Source field
# ---------------
#
# The source is an x-directed dipole at the origin, with a 10 Hz signal of 1 A
# (``src`` is defined either as ``[x, y, z, dip, azimuth]`` or ``[x0, x1, y0,
# y1, z0, z1]``; the strength can be set via the ``strength`` parameter).

sfield = emg3d.utils.get_source_field(grid, src=[0, 0, 0, 0, 0], freq=10)

###############################################################################
# 4. Calculate the electric field
# -------------------------------
#
# Now we can calculate the electric field with ``emg3d``:

efield = emg3d.solve(grid, model, sfield, verb=3)

###############################################################################
# The calculation requires in this case seven multigrid F-cycles and taken just
# a few seconds. It was able to coarsen in $x$ five times and in $y/z$ four
# times, where the input grid had 65,536 cells, and the coarsest grid had 8
# cells.
#
# 5. Plot the result
# ------------------
#
# We can again utilize the in-built functions of a ``discretize``-grid to plot,
# e.g., the x-directed electric field.

# With discretize
# grid.plot_3d_slicer(
#         efield.fx.ravel('F'), view='abs', vType='Ex',
#         pcolorOpts={'norm': LogNorm()}
# )

# Without discretize
plt.figure()
plt.pcolormesh(grid.vectorNx, grid.vectorCCy,
               np.log10(efield.fx[:, 1:-1, 17].amp).T)
plt.show()

###############################################################################

emg3d.Report()
