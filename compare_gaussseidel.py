import emg3d
import numpy as np

from emg3d import _ccore


atol = 0
rtol = 1e-12

def compare(a, b, atol, rtol):
    same = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    error = np.abs(a - b)
    print(f"\n\n  Same: {['❌', '✅'][same]}")
    nza = np.nonzero(a)[0]
    nzb = np.nonzero(b)[0]
    print(f"     => Non-zero sizes: Numba: {nza.size}; C: {nzb.size}")
    print(f"     => Non-zero elements; Numba ***")
    print(a[nza])
    print(f"     => Non-zero elements; C     ***")
    print(b[nzb])
    print()
    print(
        f"rtol={rtol}, atol={atol}\n"
        f"max abs error: {np.max(error[nza]):.4g}; max rel "
        f"error: {np.max(error[nza]/np.abs(a[nza])):.4g}\n\n"
    )

# Rotate the source, so we have a strong enough signal in all directions
src = [0, 0, 0, 45, 45]
freq = 0.9
nu = 2  # One back-and-forth

nx = 4
ny = 4
nz = 4

# Get this grid.
def widths(ncore, npad, width, factor):
    """Get cell widths for TensorMesh."""
    pad = ((np.ones(npad)*np.abs(factor))**(np.arange(npad)+1))*width
    return np.r_[pad[::-1], np.ones(ncore)*width, pad]

hx = widths(0, nx, 80, 1.1)
hy = widths(0, ny, 100, 1.3)
hz = widths(0, nz, 200, 1.2)
grid = emg3d.TensorMesh(
    [hx, hy, hz], np.array([-hx.sum()/2, -hy.sum()/2, -hz.sum()/2]))

# Initialize model with some resistivities.
property_x = np.arange(grid.n_cells)+1
property_y = 0.5*np.arange(grid.n_cells)+1
property_z = 2*np.arange(grid.n_cells)+1

model = emg3d.Model(grid, property_x, property_y, property_z)

# Initialize source field.
sfield = emg3d.get_source_field(grid, src, freq)

# Get volume-averaged model parameters.
vmodel = emg3d.models.VolumeModel(model, sfield)

# Run two iterations to get some e-field.
efield = emg3d.solve(model, sfield, sslsolver=False,
                     semicoarsening=False, linerelaxation=False,
                     maxit=2, verb=1)

inp = (sfield.fx, sfield.fy, sfield.fz, vmodel.eta_x, vmodel.eta_y,
        vmodel.eta_z, vmodel.zeta, grid.h[0], grid.h[1], grid.h[2], nu)

# Get result from `gauss_seidel`.
nfield = emg3d.Field(grid, efield.field.copy(), frequency=efield._frequency)
emg3d.core.gauss_seidel(nfield.fx, nfield.fy, nfield.fz, *inp)

# Get result from `cgaussseidel`
cfield = emg3d.Field(grid, efield.field.copy(), frequency=efield._frequency)
_ccore.gaussseidel(cfield.fx, cfield.fy, cfield.fz, *inp)

# Check the resulting field.
compare(nfield.field, cfield.field, atol, rtol)
# assert nfield == cfield
