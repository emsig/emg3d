"""
1. empymod: 1D VTI resistivity
==============================

The code ``empymod`` is an open-source code which can model CSEM responses for
a layered medium including VTI electrical anisotropy, see `emsig.xyz
<https://emsig.xyz>`_.

Content:

1. Full-space VTI model for a finite length, finite strength, rotated bipole.
2. Layered model for a deep water model with a point dipole source.


"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_path = '_static/thumbs/empymod-iw.png'


###############################################################################
# 1. Full-space VTI model for a finite length, finite strength, rotated bipole
# ----------------------------------------------------------------------------
#
# In order to shorten the build-time of the gallery we use a coarse model.
# Set ``coarse_model = False`` to obtain a result of higher accuracy.
coarse_model = True


###############################################################################
# Survey and model parameters
# ```````````````````````````

# Receiver coordinates
if coarse_model:
    x = (np.arange(256))*20-2550
else:
    x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()
frx, fry = rx.ravel(), ry.ravel()
rz = -400.0
azimuth = 33
elevation = 18

# Source coordinates, frequency, and strength
source = emg3d.TxElectricDipole(
    coordinates=[-50, 50, -30, 30, -320., -280.],  # [x1, x2, y1, y2, z1, z2]
    strength=np.pi,  # A
)
frequency = 1.1  # Hz

# Model parameters
h_res = 1.              # Horizontal resistivity
aniso = np.sqrt(2.)     # Anisotropy
v_res = h_res*aniso**2  # Vertical resistivity


###############################################################################
# 1.a Regular VTI case
# ````````````````````
#
# empymod
# ```````
# Note: The coordinate system of empymod is positive z down, for emg3d it is
# positive z up. We have to switch therefore src_z, rec_z, and elevation.

# Collect common input for empymod.
inp = {
    'src': np.r_[source.coordinates[:4], -source.coordinates[4:]],
    'depth': [],
    'res': h_res,
    'aniso': aniso,
    'strength': source.strength,
    'srcpts': 5,
    'freqtime': frequency,
    'htarg': {'pts_per_dec': -1},
}

# Compute
epm = empymod.bipole(
    rec=[frx, fry, -rz, azimuth, -elevation], verb=3, **inp
).reshape(np.shape(rx))

###############################################################################
# emg3d
# `````

if coarse_model:
    min_width_limits = 40
    stretching = [1.045, 1.045]
else:
    min_width_limits = 20
    stretching = [1.03, 1.045]

# Create stretched grid
grid = emg3d.construct_mesh(
    frequency=frequency,
    properties=h_res,
    center=source.center,
    domain=([-2500, 2500], [-2500, 2500], [-2900, 2100]),
    min_width_limits=min_width_limits,
    stretching=stretching,
    lambda_from_center=True,
    lambda_factor=0.8,
)
grid

###############################################################################

# Define the model
model = emg3d.Model(
    grid, property_x=h_res, property_z=v_res, mapping='Resistivity')

# Compute the electric field
efield = emg3d.solve_source(model, source, frequency, verb=4, plain=True)


###############################################################################
# Plot function
# `````````````

def plot(epm, e3d, title, vmin, vmax):

    # Start figure.
    a_kwargs = {'cmap': "viridis", 'vmin': vmin, 'vmax': vmax,
                'shading': 'nearest'}

    e_kwargs = {'cmap': plt.cm.get_cmap("RdBu_r", 8),
                'vmin': -2, 'vmax': 2, 'shading': 'nearest'}

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True,
                            subplot_kw={'box_aspect': 1})

    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs
    x3 = x/1000  # km

    # Plot Re(data)
    ax1.set_title(r"(a) |Re(empymod)|")
    cf0 = ax1.pcolormesh(x3, x3, np.log10(epm.real.amp()), **a_kwargs)

    ax2.set_title(r"(b) |Re(emg3d)|")
    ax2.pcolormesh(x3, x3, np.log10(e3d.real.amp()), **a_kwargs)

    ax3.set_title(r"(c) Error real part")
    rel_error = 100*np.abs((epm.real - e3d.real) / epm.real)
    cf2 = ax3.pcolormesh(x3, x3, np.log10(rel_error), **e_kwargs)

    # Plot Im(data)
    ax4.set_title(r"(d) |Im(empymod)|")
    ax4.pcolormesh(x3, x3, np.log10(epm.imag.amp()), **a_kwargs)

    ax5.set_title(r"(e) |Im(emg3d)|")
    ax5.pcolormesh(x3, x3, np.log10(e3d.imag.amp()), **a_kwargs)

    ax6.set_title(r"(f) Error imaginary part")
    rel_error = 100*np.abs((epm.imag - e3d.imag) / epm.imag)
    ax6.pcolormesh(x3, x3, np.log10(rel_error), **e_kwargs)

    # Colorbars
    unit = "(V/m)" if "E" in title else "(A/m)"
    fig.colorbar(cf0, ax=axs[0, :], label=r"$\log_{10}$ Amplitude "+unit)
    cbar = fig.colorbar(cf2, ax=axs[1, :], label=r"Relative Error")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([r"$0.01\,\%$", r"$0.1\,\%$", r"$1\,\%$",
                             r"$10\,\%$", r"$100\,\%$"])

    ax1.set_xlim(min(x3), max(x3))
    ax1.set_ylim(min(x3), max(x3))

    # Axis label
    fig.text(0.4, 0.05, "Inline Offset (km)", fontsize=14)
    fig.text(0.05, 0.3, "Crossline Offset (km)", rotation=90, fontsize=14)
    fig.suptitle(title, y=1, fontsize=20)

    print(f"- Source: {source}")
    print(f"- Frequency: {frequency} Hz")
    rtype = "Electric" if "E" in title else "Magnetic"
    print(f"- {rtype} receivers: z={rz} m; θ={azimuth}°, φ={elevation}°")

    fig.show()


###############################################################################
# Plot
# ````

e3d = efield.get_receiver((rx, ry, rz, azimuth, elevation))
plot(epm, e3d, r'Diffusive Fullspace $E$', vmin=-12, vmax=-6)


#############################################################################
# 2. Layered model for a deep water model with a point dipole source
# ------------------------------------------------------------------
#
# Survey and model parameters
# ```````````````````````````

# Receiver coordinates
if coarse_model:
    x = (np.arange(256))*20-2550
else:
    x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()
frx, fry = rx.ravel(), ry.ravel()
rz = -950.0
azimuth = 30
elevation = 5

# Source coordinates and frequency
source = emg3d.TxElectricDipole(coordinates=[0, 0, -900, 0, 0])
frequency = 1.0  # Hz

# Model parameters
h_res = [1, 50, 1, 0.3, 1e12]     # Horizontal resistivity
aniso = np.sqrt([2, 2, 2, 1, 1])  # Anisotropy
v_res = h_res*aniso**2            # Vertical resistivity
depth = np.array([-2200, -2000, -1000, 0])  # Layer boundaries


###############################################################################
# empymod
# ```````

epm_d = empymod.bipole(
    src=source.coordinates,
    depth=depth,
    res=h_res,
    aniso=aniso,
    freqtime=frequency,
    htarg={'pts_per_dec': -1},
    rec=[frx, fry, rz, azimuth, elevation],
    verb=3,
).reshape(np.shape(rx))


###############################################################################
# emg3d
# `````

if coarse_model:
    min_width_limits = 40
else:
    min_width_limits = 20

# Create stretched grid
grid = emg3d.construct_mesh(
    frequency=frequency,
    properties=[h_res[3], h_res[0]],
    center=source.center,
    domain=([-2500, 2500], [-2500, 2500], None),
    vector=(None, None, -2200 + np.arange(111)*20),
    min_width_limits=min_width_limits,
    stretching=[1.1, 1.5],
)
grid

###############################################################################

# Create the model: horizontal resistivity
res_x_full = h_res[0]*np.ones(grid.n_cells)  # Background
res_x_full[grid.cell_centers[:, 2] >= depth[0]] = h_res[1]  # Target
res_x_full[grid.cell_centers[:, 2] >= depth[1]] = h_res[2]  # Overburden
res_x_full[grid.cell_centers[:, 2] >= depth[2]] = h_res[3]  # Water
res_x_full[grid.cell_centers[:, 2] >= depth[3]] = h_res[4]  # Air

# Create the model: vertical resistivity
res_z_full = v_res[0]*np.ones(grid.n_cells)  # Background
res_z_full[grid.cell_centers[:, 2] >= depth[0]] = v_res[1]
res_z_full[grid.cell_centers[:, 2] >= depth[1]] = v_res[2]
res_z_full[grid.cell_centers[:, 2] >= depth[2]] = v_res[3]
res_z_full[grid.cell_centers[:, 2] >= depth[3]] = v_res[4]

# Get the model
model = emg3d.Model(
        grid, property_x=res_x_full, property_z=res_z_full,
        mapping='Resistivity')

###############################################################################

# Compute the electric field
efield = emg3d.solve_source(model, source, frequency, verb=4)


###############################################################################
# Plot
# ````
e3d_d = efield.get_receiver((rx, ry, rz, azimuth, elevation))
plot(epm_d, e3d_d, r'Deep water point dipole $E$', vmin=-14, vmax=-8)


###############################################################################

emg3d.Report()
