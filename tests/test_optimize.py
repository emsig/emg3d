import pytest
import numpy as np
# from numpy.testing import assert_allclose

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None

from emg3d import meshes, models, surveys, simulations, optimize


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

        simulation.compute(reference=True)

    def test_errors(self):

        # Rotated receivers.
        simulation = self.simulation.copy()
        simulation.survey.receivers['Rx01'].dip = 10
        with pytest.raises(NotImplementedError, match='x-directed electric r'):
            optimize.gradient(simulation)

        # Anisotropic models.
        simulation = self.simulation.copy()
        simulation.model = models.Model(self.grid, 1, 2, 3)
        with pytest.raises(NotImplementedError, match='for isotropic models'):
            optimize.gradient(simulation)
