r"""Some models to catch regression with status quo while developing."""
import numpy as np
from discretize import TensorMesh

from emg3d import solver, meshes, models, fields, io


# # # # # # # # # # 1. Homogeneous VTI fullspace # # # # # # # # # #

freq = 1.
grid = meshes.construct_mesh(
        frequency=freq,
        center=(0, 0, 0),
        properties=[2, 2, 3.3, 3.3],
        domain=[[-50, 50], [-50, 50], [200, 300]],
        max_buffer=1500,
        lambda_from_center=True,
        cell_numbers=meshes.good_mg_cell_nr(100, 2, 1),
)

input_grid = {'hx': grid.h[0], 'hy': grid.h[1], 'hz': grid.h[2],
              'origin': grid.origin}

input_model = {
    'grid': grid,
    'property_x': 1.5,
    'property_y': 2.0,
    'property_z': 3.3,
    'mapping': 'Resistivity'
    }
model = models.Model(**input_model)

input_source = {
    'grid': grid,
    'src': [0, 0, 250., 30, 10],  # A rotated source to include all
    'freq': freq
    }

# Fields
sfield = fields.get_source_field(**input_source)

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

grid = meshes.construct_mesh(
        frequency=freq,
        center=(0, 0, 0),
        properties=[50, 50, 3.3, 3.3],
        domain=[[src[0], src[1]], [src[0], src[1]], [src[2]-20, src[2]+20]],
        max_buffer=5000,
        lambda_from_center=True,
        cell_numbers=meshes.good_mg_cell_nr(100, 2, 1),
)

input_grid = {'hx': grid.h[0], 'hy': grid.h[1], 'hz': grid.h[2],
              'origin': grid.origin}

# Initialize model
# Create a model with random resistivities between [0, 50)
property_x = np.random.random(grid.n_cells)*50
property_y = np.random.random(grid.n_cells)*50
property_z = np.random.random(grid.n_cells)*50
model = models.Model(grid, property_x, property_y, property_z,
                     mapping='Resistivity')

# Initialize source field
sfield = fields.get_source_field(grid, src, freq)

semicoarsening = 123  # Loop 1, 2, 3
linerelaxation = 456  # Loop over 4, 5, 5
verb = 4    # high verbosity
tol = 1e-4  # low tolerance
maxit = 4   # Restrict
nu_init = 2
nu_pre = 2
nu_coarse = 1
nu_post = 2
clevel = 10  # Way to high

efield = solver.solve(
        grid, model, sfield, semicoarsening=semicoarsening,
        linerelaxation=linerelaxation, tol=tol, maxit=maxit, nu_init=nu_init,
        nu_pre=nu_pre, nu_coarse=nu_coarse, nu_post=nu_post, clevel=clevel)

hfield = fields.get_h_field(model, efield)

# Store input and result
reg_2 = {
    'grid': grid,
    'model': model,
    'sfield': sfield,
    'inp': {
        'semicoarsening': semicoarsening,
        'linerelaxation': linerelaxation,
        'verb': 4,
        'tol': tol,
        'maxit': maxit,
        'nu_init': nu_init,
        'nu_pre': nu_pre,
        'nu_coarse': nu_coarse,
        'nu_post': nu_post,
        'clevel': clevel,
        },
    'result': efield,
    'hresult': hfield,
}

# # # # # # # # # # 3. TensorMesh check # # # # # # # # # #
# Create an advanced grid with discretize.

grid = TensorMesh(
        [[(10, 10, -1.1), (10, 20, 1), (10, 10, 1.1)],
         [(33, 20, 1), (33, 10, 1.5)],
         [20]],
        origin='CN0')

# List of all attributes in emg3d-grid.
all_attr = [
    'origin',
    'shape_cells', 'shape_nodes',
    'n_cells', 'n_edges_x', 'n_edges_y', 'n_edges_z',
    'nodes_x', 'nodes_y', 'nodes_z',
    'cell_centers_x', 'cell_centers_y', 'cell_centers_z',
    'shape_edges_x', 'shape_edges_y', 'shape_edges_z',
]

mesh = {}

for attr in all_attr:
    mesh[attr] = getattr(grid, attr)
mesh['hx'] = grid.h[0]
mesh['hy'] = grid.h[1]
mesh['hz'] = grid.h[2]


# # # # # # # # # # 4. Homogeneous VTI fullspace LAPLACE # # # # # # # # # #

freq = -2*np.pi

grid_l = meshes.construct_mesh(
        frequency=freq,
        center=(0, 0, 0),
        properties=[3.3, 2, 3.3, 3.3],
        domain=[[-1, 1], [-1, 1], [240, 260]],
        max_buffer=1500,
        lambda_from_center=True,
        cell_numbers=meshes.good_mg_cell_nr(100, 2, 1),
)

input_grid_l = {'hx': grid_l.h[0], 'hy': grid_l.h[1], 'hz': grid_l.h[2],
                'origin': grid_l.origin}

input_model_l = {
    'grid': grid_l,
    'property_x': 1.5,
    'property_y': 2.0,
    'property_z': 3.3,
    'mapping': 'Resistivity'
    }
model_l = models.Model(**input_model_l)

input_source_l = {
    'grid': grid_l,
    'src': [0, 0, 250., 30, 10],  # A rotated source to include all
    'freq': freq
    }

# Fields
sfield_l = fields.get_source_field(**input_source_l)

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


io.save('../data/regression.npz', res=out, reg_2=reg_2, grid=mesh, lap=out_l)
