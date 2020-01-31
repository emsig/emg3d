r"""Some models to catch regression with status quo while developing."""
import numpy as np
from discretize import TensorMesh

from emg3d import utils, solver


# # # # # # # # # # 1. Homogeneous VTI fullspace # # # # # # # # # #

freq = 1.
hx_min, xdomain = utils.get_domain(x0=0, freq=freq)
hy_min, ydomain = utils.get_domain(x0=0, freq=freq)
hz_min, zdomain = utils.get_domain(x0=250, freq=freq)
nx = 2**3
hx = utils.get_stretched_h(hx_min, xdomain, nx, 0)
hy = utils.get_stretched_h(hy_min, ydomain, nx, 0)
hz = utils.get_stretched_h(hz_min, zdomain, nx, 250)
input_grid = {'h': [hx, hy, hz], 'x0': (xdomain[0], ydomain[0], zdomain[0])}
grid = utils.TensorMesh(**input_grid)

input_model = {
    'grid': grid,
    'res_x': 1.5,
    'res_y': 2.0,
    'res_z': 3.3,
    }
model = utils.Model(**input_model)

input_source = {
    'grid': grid,
    'src': [0, 0, 250., 30, 10],  # A rotated source to include all
    'freq': freq
    }

# Fields
sfield = utils.get_source_field(**input_source)

# F-cycle
fefield = solver.solve(grid, model, sfield)

# W-cycle
wefield = solver.solve(grid, model, sfield, cycle='W')

# V-cycle
vefield = solver.solve(grid, model, sfield, cycle='V')

# BiCGSTAB; F-cycle
bicefield = solver.solve(grid, model, sfield, sslsolver=True)

out = {
    'input_grid': input_grid,
    'input_model': input_model,
    'input_source': input_source,
    'grid': grid,
    'model': model,
    'sfield': sfield,
    'Fresult': fefield,
    'Wresult': wefield,
    'Vresult': vefield,
    'bicresult': bicefield,
    }

# # # # # # # # # # 2. Inhomogeneous case # # # # # # # # # #

# Parameters
src = [50., 110., 250., 25, 15]
freq = 0.375

hx_min, xdomain = utils.get_domain(x0=0, freq=.1)
hy_min, ydomain = utils.get_domain(x0=0, freq=.1)
hz_min, zdomain = utils.get_domain(x0=250, freq=.1)
hx = utils.get_stretched_h(hx_min, xdomain, 8, 0)
hy = utils.get_stretched_h(hy_min, ydomain, 4, 0)
hz = utils.get_stretched_h(hz_min, zdomain, 16, 250)
grid = utils.TensorMesh([hx, hy, hz], x0=(xdomain[0], ydomain[0], zdomain[0]))


# Initialize model
# Create a model with random resistivities between [0, 50)
res_x = np.random.random(grid.nC)*50
res_y = np.random.random(grid.nC)*50
res_z = np.random.random(grid.nC)*50
model = utils.Model(grid, res_x, res_y, res_z)

# Initialize source field
sfield = utils.get_source_field(grid, src, freq)

semicoarsening = True  # Loop 1, 2, 3
linerelaxation = 456   # Loop over 4, 5, 5
verb = 4    # high verbosity
tol = 1e-4  # low tolerance
maxit = 4   # Restrict
nu_init = 2
nu_pre = 2
nu_coarse = 1
nu_post = 2
clevel = 10  # Way to high

efield = solver.solve(
        grid, model, sfield,
        semicoarsening=semicoarsening, linerelaxation=linerelaxation,
        verb=verb, tol=tol, maxit=maxit, nu_init=nu_init, nu_pre=nu_pre,
        nu_coarse=nu_coarse, nu_post=nu_post, clevel=clevel)

hfield = utils.get_h_field(grid, model, efield)

# Store input and result
reg_2 = {
    'grid': grid,
    'model': model,
    'sfield': sfield,
    'inp': {
        'semicoarsening': semicoarsening,
        'linerelaxation': linerelaxation,
        'verb': verb,
        'tol': tol,
        'maxit': maxit,
        'nu_init': nu_init,
        'nu_pre': nu_pre,
        'nu_coarse': nu_coarse,
        'nu_post': nu_post,
        'clevel': clevel,
        },
    'result': efield.field,
    'hresult': hfield.field,
}

# # # # # # # # # # 3. TensorMesh check # # # # # # # # # #
# Create an advanced grid with discretize.

grid = TensorMesh(
        [[(10, 10, -1.1), (10, 20, 1), (10, 10, 1.1)],
         [(33, 20, 1), (33, 10, 1.5)],
         [20]],
        x0='CN0')

# List of all attributes in emg3d-grid.
all_attr = [
    'hx', 'hy', 'hz', 'vectorNx', 'vectorNy', 'vectorNz', 'vectorCCx', 'nE',
    'vectorCCy', 'vectorCCz', 'nEx', 'nEy', 'nEz', 'nCx', 'nCy', 'nCz', 'vnC',
    'nNx', 'nNy', 'nNz', 'vnN', 'vnEx', 'vnEy', 'vnEz', 'vnE', 'nC', 'nN', 'x0'
]

mesh = {'attr': all_attr}

for attr in all_attr:
    mesh[attr] = getattr(grid, attr)


# # # # # # # # # # 4. Homogeneous VTI fullspace LAPLACE # # # # # # # # # #

freq = -2*np.pi
hx_min, xdomain = utils.get_domain(x0=0, freq=freq)
hy_min, ydomain = utils.get_domain(x0=0, freq=freq)
hz_min, zdomain = utils.get_domain(x0=250, freq=freq)
nx = 2**3
hx = utils.get_stretched_h(hx_min, xdomain, nx, 0)
hy = utils.get_stretched_h(hy_min, ydomain, nx, 0)
hz = utils.get_stretched_h(hz_min, zdomain, nx, 250)
input_grid_l = {'h': [hx, hy, hz], 'x0': (xdomain[0], ydomain[0], zdomain[0])}
grid_l = utils.TensorMesh(**input_grid_l)

input_model_l = {
    'grid': grid_l,
    'res_x': 1.5,
    'res_y': 2.0,
    'res_z': 3.3,
    }
model_l = utils.Model(**input_model_l)

input_source_l = {
    'grid': grid_l,
    'src': [0, 0, 250., 30, 10],  # A rotated source to include all
    'freq': freq
    }

# Fields
sfield_l = utils.get_source_field(**input_source_l)

# F-cycle
fefield_l = solver.solve(grid_l, model_l, sfield_l)

# BiCGSTAB; F-cycle
bicefield_l = solver.solve(grid_l, model_l, sfield_l, sslsolver=True)

out_l = {
    'input_grid': input_grid_l,
    'input_model': input_model_l,
    'input_source': input_source_l,
    'grid': grid_l,
    'model': model_l,
    'sfield': sfield_l,
    'Fresult': fefield_l,
    'bicresult': bicefield_l,
    }


np.savez_compressed(
        '../data/regression.npz', res=out, reg_2=reg_2, grid=mesh, lap=out_l)
