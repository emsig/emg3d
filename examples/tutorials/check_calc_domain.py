r"""
8. Ensure computation domain is big enough
==========================================

Ensure the boundary in :math:`\pm x`, :math:`\pm y`, and :math:`+ z` is big
enough for :math:`\rho_\text{air}`.

The air is very resistive, and EM waves propagate at the speed of light as a
wave, not diffusive any longer. The whole concept of skin depth does therefore
not apply to the air layer. The only attenuation is caused by geometrical
spreading. In order to not have any effects from the boundary one has to choose
the air layer appropriately.

The important bit is that the resistivity of air has to be taken into account
also for the horizontal directions, not only for positive :math:`z` (upwards
into the sky). This is an example to test boundaries on a simple marine model
(air, water, subsurface) and compare them to the 1D result.
"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Model, Survey, and Analytical Solution
# --------------------------------------

water_depth = 500                        # 500 m water depth
offsets = np.linspace(2000, 7000, 501)   # Offsets
src_coo = [0, 0, -water_depth+50, 0, 0]  # Src at origin, 50 m above seafloor
rec_coo = [offsets, offsets*0, -water_depth, 0, 0]  # Receivers on the seafloor
depth = [-water_depth, 0]                # Simple model
resistivity = [1, 0.3, 1e8]              # Simple model
frequency = 0.1                          # Frequency

source = emg3d.TxElectricDipole(src_coo)

# Compute analytical solution
epm = empymod.bipole(src_coo, rec_coo, depth, resistivity, frequency)

###############################################################################
# 3D Modelling
# ------------

# Parameter we keep the same for both grids
grid_inp = {
    'frequency': frequency,
    'center': [src_coo[0], src_coo[1], -water_depth-100],
    'domain': ([src_coo[0]-500, offsets[-1]+500],
               [src_coo[1], src_coo[1]],
               [-600, 100]),
    'seasurface': 0,
    'min_width_limits': 100,
    'stretching': [1, 1.25],
    'lambda_from_center': True,
    'verb': 1,
}

###############################################################################
# 1st grid, only considering air resistivity for +z
# -------------------------------------------------
#
# Here we are in the water, so the signal is attenuated before it enters the
# air. So we don't use the resistivity of air to compute the required
# boundary, but 100 Ohm.m instead. (100 is the result of a quick parameter test
# with :math:`\rho=1e4, 1e3, 1e2, 1e1`, and the result was that after 100 there
# is not much improvement any longer.)
#
# Also note that the function :func:`emg3d.meshes.construct_mesh` internally
# uses six times the skin depth for the boundary. For :math:`\rho` = 100 Ohm.m
# and :math:`f` = 0.1 Hz, the skin depth :math:`\delta` is roughly 16 km, which
# therefore results in a boundary of roughly 96 km.
#
# See the documentation of :func:`emg3d.meshes.get_hx_h0` for more information
# on how the grid is created.

grid_1 = emg3d.construct_mesh(
    properties=[resistivity[1], resistivity[0], resistivity[0], 100],
    **grid_inp,
)
grid_1

###############################################################################

# Create corresponding model
res_1 = resistivity[0]*np.ones(grid_1.n_cells)
res_1[grid_1.cell_centers[:, 2] > -water_depth] = resistivity[1]
res_1[grid_1.cell_centers[:, 2] > 0] = resistivity[2]
model_1 = emg3d.Model(grid_1, property_x=res_1, mapping='Resistivity')

# QC
grid_1.plot_3d_slicer(
        np.log10(model_1.property_x), zlim=(-2000, 100), clim=[-1, 2])

# Solve the system
efield_1 = emg3d.solve_source(model_1, source, frequency, verb=3)


###############################################################################
# 2nd grid, considering air resistivity for +/- x, +/- y, and +z
# --------------------------------------------------------------
#
# See comments below the heading of the 1st grid regarding boundary.

grid_2 = emg3d.construct_mesh(
    properties=[resistivity[1], resistivity[0], 100],
    **grid_inp,
)
grid_2

###############################################################################

# Create corresponding model
res_2 = resistivity[0]*np.ones(grid_2.n_cells)
res_2[grid_2.cell_centers[:, 2] > -water_depth] = resistivity[1]
res_2[grid_2.cell_centers[:, 2] > 0] = resistivity[2]
model_2 = emg3d.Model(grid_2, property_x=res_2, mapping='Resistivity')

# QC
# grid_2.plot_3d_slicer(
#         np.log10(model_2.property_x), zlim=(-2000, 100), clim=[-1, 2])

# Define source and solve the system
efield_2 = emg3d.solve_source(model_2, source, frequency, verb=3)


###############################################################################
# Plot receiver responses
# -----------------------

# Interpolate fields at receiver positions
emg_1 = efield_1.get_receiver(tuple(rec_coo))
emg_2 = efield_2.get_receiver(tuple(rec_coo))


###############################################################################

plt.figure(figsize=(10, 7))

# Real, log-lin
ax1 = plt.subplot(321)
plt.title('(a) lin-lin Real')
plt.plot(offsets/1e3, epm.real, 'k', lw=2, label='analytical')
plt.plot(offsets/1e3, emg_1.real, 'C0--', label='grid 1')
plt.plot(offsets/1e3, emg_2.real, 'C1:', label='grid 2')
plt.ylabel('$E_x$ (V/m)')
plt.legend()

# Real, log-symlog
ax3 = plt.subplot(323, sharex=ax1)
plt.title('(c) lin-symlog Real')
plt.plot(offsets/1e3, epm.real, 'k')
plt.plot(offsets/1e3, emg_1.real, 'C0--')
plt.plot(offsets/1e3, emg_2.real, 'C1:')
plt.ylabel('$E_x$ (V/m)')
plt.yscale('symlog', linthresh=1e-15)

# Real, error
ax5 = plt.subplot(325, sharex=ax3)
plt.title('(e) clipped 0.01-10')

# Compute the error
err_real_1 = np.clip(100*abs((epm.real-emg_1.real)/epm.real), 0.01, 10)
err_real_2 = np.clip(100*abs((epm.real-emg_2.real)/epm.real), 0.01, 10)

plt.ylabel('Rel. error %')
plt.plot(offsets/1e3, err_real_1, 'C0--')
plt.plot(offsets/1e3, err_real_2, 'C1:')
plt.axhline(1, color='.4')

plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Offset (km)')

# Imaginary, log-lin
ax2 = plt.subplot(322)
plt.title('(b) lin-lin Imag')
plt.plot(offsets/1e3, epm.imag, 'k')
plt.plot(offsets/1e3, emg_1.imag, 'C0--')
plt.plot(offsets/1e3, emg_2.imag, 'C1:')

# Imaginary, log-symlog
ax4 = plt.subplot(324, sharex=ax2)
plt.title('(d) lin-symlog Imag')
plt.plot(offsets/1e3, epm.imag, 'k')
plt.plot(offsets/1e3, emg_1.imag, 'C0--')
plt.plot(offsets/1e3, emg_2.imag, 'C1:')

plt.yscale('symlog', linthresh=1e-15)

# Imaginary, error
ax6 = plt.subplot(326, sharex=ax2)
plt.title('(f) clipped 0.01-10')

# Compute error
err_imag_1 = np.clip(100*abs((epm.imag-emg_1.imag)/epm.imag), 0.01, 10)
err_imag_2 = np.clip(100*abs((epm.imag-emg_2.imag)/epm.imag), 0.01, 10)

plt.plot(offsets/1e3, err_imag_1, 'C0--')
plt.plot(offsets/1e3, err_imag_2, 'C1:')
plt.axhline(1, color='.4')

plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Offset (km)')

plt.tight_layout()
plt.show()


###############################################################################
# Plot entire fields to analyze and compare
# -----------------------------------------
#
# 1st grid
# ````````
#
# Upper plot shows the entire grid. One can see that the airwave attenuates to
# amplitudes in the order of 1e-17 at the boundary, absolutely good enough.
# However, the amplitudes in the horizontal directions are very high even at
# the boundaries :math:`\pm x` and :math:`\pm y`.

grid_1.plot_3d_slicer(
    efield_1.fx.ravel('F'), view='abs', v_type='Ex',
    xslice=src_coo[0], yslice=src_coo[1], zslice=rec_coo[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})
grid_1.plot_3d_slicer(
    efield_1.fx.ravel('F'), view='abs', v_type='Ex',
    zlim=[-5000, 1000],
    xslice=src_coo[0], yslice=src_coo[1], zslice=rec_coo[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})


###############################################################################
# 2nd grid
# ````````
#
# Again, upper plot shows the entire grid. One can see that the field
# attenuated sufficiently in all directions. Lower plot shows the same cut-out
# as the lower plot for the first grid, our zone of interest.

grid_2.plot_3d_slicer(
    efield_2.fx.ravel('F'), view='abs', v_type='Ex',
    xslice=src_coo[0], yslice=src_coo[1], zslice=rec_coo[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})
grid_2.plot_3d_slicer(
    efield_2.fx.ravel('F'), view='abs', v_type='Ex',
    xlim=[grid_1.nodes_x[0], grid_1.nodes_x[-1]],  # Same square as grid_1
    ylim=[grid_1.nodes_y[0], grid_1.nodes_y[-1]],  # Same square as grid_1
    zlim=[-5000, 1000],
    xslice=src_coo[0], yslice=src_coo[1], zslice=rec_coo[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})


###############################################################################

emg3d.Report()
