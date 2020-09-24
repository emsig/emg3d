import pytest
import numpy as np
# from numpy.testing import assert_allclose

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None

from emg3d import meshes, models, surveys, simulations, optimize


def random_fd_gradient(n, survey, model, mesh, grad, data_misfit, sim_inp,
                       epsilon=1e-4):
    """Compute FD gradient for random cell and compare to AS gradient.

    We use the forward finite difference approach,

    n :: int; number of random cells.
    epsilon :: float; difference to add.
    """

    print(f"   === Compare Gradients  ::  epsilon={epsilon} ===\n\n"
          f"{{xi;iy;iz}}     Adjoint-state       Forward FD    NRMSD (%)\n"
          f"----------------------------------------------------------")

    avg_nrmsd = 0.0

    for i in range(n):
        ix = np.random.randint(15, 50)  # btw 1500-5000 m
        iy = np.random.randint(25, 40)  # btw 2500-4000 m
        iz = np.random.randint(25, 40)  # btw 2500-4000 m

        # Add epsilon to random cell.
        model_diff = model.copy()
        model_diff.property_x[ix, iy, iz] += epsilon

        sim_data = simulations.Simulation(model=model_diff, **sim_inp)
        sim_data._tqdm_opts = {'disable': True}

        # Compute FD-gradient
        fdgrad = float((sim_data.misfit - data_misfit)/epsilon)

        # Compute NRMSD
        nrmsd = 200*abs(grad[ix, iy, iz]-fdgrad)/(
                abs(grad[ix, iy, iz])+abs(fdgrad))
        avg_nrmsd += nrmsd/n

        print(f"{{{ix:2d};{iy:2d};{iz:2d}}}     {grad[ix, iy, iz]:+.6e}    "
              f"{fdgrad:+.6e}    {nrmsd:9.5f}")

    return avg_nrmsd


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestOptimize():
    if xarray is not None:
        # Create a simple survey
        sources = (0, [1000, 3000, 5000], -950, 0, 0)
        receivers = (np.arange(12)*500, 0, -1000, 0, 0)
        frequencies = (1.0, 2.0)

        survey = surveys.Survey('Test', sources, receivers, frequencies)

        # Create a simple grid and model
        grid = meshes.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model1 = models.Model(grid, 1)
        model2 = models.Model(grid, np.arange(1, grid.nC+1).reshape(grid.vnC))

        # Create a simulation, compute all fields.
        simulation = simulations.Simulation(
                'Data', survey, grid, model1, max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding='same')

        simulation.compute(observed=True)

        simulation = simulations.Simulation(
                'Model', survey, grid, model2, max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding='same')

    def test_errors(self):

        # Anisotropic models.
        simulation = self.simulation.copy()
        simulation.model = models.Model(self.grid, 1, 2, 3)
        with pytest.raises(NotImplementedError, match='for isotropic models'):
            optimize.gradient(simulation)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_derivative(capsys):
    # Create a simple mesh.
    hx = np.ones(64)*100
    mesh = meshes.TensorMesh([hx, hx, hx], x0=[0, 0, 0])

    # Define a simple survey.
    survey = surveys.Survey(
        name='Gradient Test',
        sources=(1650, 3200, 3200, 0, 0),
        receivers=(4750, 3200, 3200, 0, 0),
        frequencies=1.0,
        noise_floor=1e-15,
        relative_error=0.05,
    )

    # Background Model
    con_init = np.ones(mesh.vnC)

    # Target Model 1: One Block
    con_true = np.ones(mesh.vnC)
    con_true[22:32, 32:42, 20:30] = 0.001

    model_init = models.Model(mesh, con_init, mapping='Conductivity')
    model_true = models.Model(mesh, con_true, mapping='Conductivity')

    solver_opts = {
        'sslsolver': False,
        'semicoarsening': False,
        'linerelaxation': False,
        'tol': 5e-5,  # Reduce tolerance to speed-up
    }
    sim_inp = {
        'name': 'Testing',
        'survey': survey,
        'grid': mesh,
        'solver_opts': solver_opts,
        'max_workers': 1,
        'gridding': 'same',
        'verb': 3,
    }

    # Compute data (pre-computed and passed to Survey above)
    sim_data = simulations.Simulation(model=model_true, **sim_inp)
    sim_data.compute(observed=True)

    # Compute adjoint state misfit and gradient
    sim_data = simulations.Simulation(model=model_init, **sim_inp)
    data_misfit = sim_data.misfit
    grad = sim_data.gradient

    # Note: We test a random cell (within bounds); Generally, the NRMSD will be
    #       below 0.01 %. However, if there are random failures (from
    #       sign-switches) then we have to fix these random indices.
    nrmsd = random_fd_gradient(
            1, survey, model_init, mesh, grad, data_misfit, sim_inp)
    assert nrmsd < 1.0
