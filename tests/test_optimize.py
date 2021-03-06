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

    # If `n=False`: One pseudo-random computation.
    # Pseudo-random to avoid error from region where the gradient crosses zero.
    pseudo = n is False
    if pseudo:
        n = 1
    ixiz = [[25, 27], [25, 28], [25, 31], [25, 32], [25, 35], [25, 36],
            [25, 37], [25, 38], [26, 27], [26, 31], [26, 32], [26, 35],
            [26, 36], [26, 37], [26, 38], [27, 27], [27, 31], [27, 32],
            [27, 35], [27, 36], [27, 37], [27, 38], [28, 27], [28, 30],
            [28, 31], [28, 32], [28, 36], [28, 37], [28, 38], [29, 27],
            [29, 31], [29, 32], [29, 33], [29, 35], [29, 36], [29, 37],
            [29, 38], [30, 27], [30, 30], [30, 31], [30, 32], [30, 33],
            [30, 36], [30, 37], [30, 38], [31, 27], [31, 29], [31, 30],
            [31, 31], [31, 32], [31, 33], [31, 36], [31, 37], [31, 38],
            [32, 27], [32, 29], [32, 31], [32, 32], [32, 36], [32, 37],
            [32, 38], [33, 27], [33, 30], [33, 31], [33, 32], [33, 33],
            [33, 36], [33, 37], [33, 38], [34, 27], [34, 31], [34, 32],
            [34, 36], [34, 37], [34, 38], [35, 27], [35, 30], [35, 31],
            [35, 32], [35, 33], [35, 36], [35, 37], [35, 38], [36, 27],
            [36, 31], [36, 32], [36, 36], [36, 37], [36, 38], [37, 27],
            [37, 28], [37, 31], [37, 32], [37, 36], [37, 37], [37, 38],
            [38, 27], [38, 28], [38, 31], [38, 32], [38, 33], [38, 35],
            [38, 36], [38, 37], [38, 38], [39, 27], [39, 28], [39, 31],
            [39, 32], [39, 35], [39, 36], [39, 37], [39, 38], [40, 27],
            [40, 28], [40, 29], [40, 31], [40, 32], [40, 35], [40, 36],
            [40, 37], [40, 38]]

    avg_nrmsd = 0.0

    for i in range(n):
        if pseudo:
            ix, iz = ixiz[np.random.randint(len(ixiz))]
            iy = 33
        else:
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

        # Model with electric permittivity.
        simulation.model = models.Model(self.grid, 1, epsilon_r=3)
        with pytest.raises(NotImplementedError, match='for el. permittivity'):
            optimize.gradient(simulation)

        # Model with magnetic permeability.
        simulation.model = models.Model(
                self.grid, 1, mu_r=np.ones(self.grid.vnC)*np.pi)
        with pytest.raises(NotImplementedError, match='for magn. permeabili'):
            optimize.gradient(simulation)

        # Missing noise_floor / std.
        simulation.model = models.Model(
                self.grid, np.arange(1, self.grid.nC+1).reshape(self.grid.vnC),
                epsilon_r=None, mu_r=np.ones(self.grid.vnC))
        with pytest.raises(ValueError, match="Either `noise_floor` or"):
            optimize.misfit(simulation)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_derivative(capsys):
    # Create a simple mesh.
    hx = np.ones(64)*100
    mesh = meshes.TensorMesh([hx, hx, hx], origin=[0, 0, 0])

    # Loop over electric and magnetic receivers.
    for dip, electric in zip([0, 90], [True, False]):

        # Define a simple survey.
        survey = surveys.Survey(
            name='Gradient Test',
            sources=(1650, 3200, 3200, 0, 0),
            receivers=(4750, 3200, 3200, 0, dip, electric),
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

        # Note: We test a pseudo-random cell.
        # Generally, the NRMSD will be below 0.01 %. However, in boundary
        # regions or regions where the gradient changes from positive to
        # negative this would fail, so we run a pseudo-random element.
        nrmsd = random_fd_gradient(
                False, survey, model_init, mesh, grad, data_misfit, sim_inp)

        if electric:
            assert nrmsd < 1.0
        else:
            assert nrmsd < 5.0
