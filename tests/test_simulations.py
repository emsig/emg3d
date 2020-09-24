import pytest
import numpy as np
from numpy.testing import assert_allclose

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None

from emg3d import meshes, models, surveys, simulations, fields, solver


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestSimulation():
    if xarray is not None:
        # Create a simple survey
        sources = (0, [1000, 3000, 5000], -950, 0, 0)
        receivers = (np.arange(12)*500, 0, -1000, 0, 0)
        frequencies = (1.0, 2.0)

        survey = surveys.Survey('Test', sources, receivers, frequencies,
                                noise_floor=1e-15, relative_error=0.05)

        # Create a simple grid and model
        grid = meshes.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model = models.Model(grid, 1)

        # Create a simulation, compute all fields.
        simulation = simulations.Simulation(
                'Test1', survey, grid, model, max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding='same')

        # Do first one single and then all together.
        simulation.get_efield('Tx0', 2.0)

    def test_derived(self):

        # Check model
        assert self.simulation.get_model('Tx1', 1.0) == self.model

        # Check grid
        assert self.simulation.get_grid('Tx1', 1.0) == self.grid

    def test_fields(self, capsys):
        # Check sfield
        sfield = fields.get_source_field(
                self.grid, self.survey.sources['Tx1'].coordinates,
                freq=1.0, strength=0)
        assert_allclose(self.simulation.get_sfield('Tx1', 1.0), sfield)

        # Check efield
        efield, info = solver.solve(
                self.grid, self.model, sfield, return_info=True,
                **self.simulation.solver_opts)
        assert_allclose(self.simulation.get_efield('Tx1', 1.0), efield)

        # Unknown keyword
        with pytest.raises(TypeError, match='Unexpected '):
            self.simulation.get_efield('Tx1', 1.0, unknownkeyward=True)

        # See a single one
        self.simulation._dict_efield['Tx1'][1.0] = None
        _, _ = capsys.readouterr()
        self.simulation.get_efield('Tx1', 1.0)

        info = self.simulation.get_efield_info('Tx1', 1.0)
        assert 'MAX. ITERATION REACHED, NOT CONVERGED' in info['exit_message']

        # Check hfield
        hfield = fields.get_h_field(self.grid, self.model, efield)
        assert_allclose(self.simulation.get_hfield('Tx1', 1.0), hfield)
        s_hfield = self.simulation.get_hfield('Tx1', 1.0)
        assert_allclose(s_hfield, hfield)
        assert_allclose(
                self.simulation._dict_efield_info['Tx1'][1.0]['abs_error'],
                info['abs_error'])
        assert_allclose(
                self.simulation._dict_efield_info['Tx1'][1.0]['rel_error'],
                info['rel_error'])
        exit = self.simulation._dict_efield_info['Tx1'][1.0]['exit']
        assert exit == info['exit'] == 1

    def test_responses(self):
        rec_resp = fields.get_receiver_response(
                self.grid, self.simulation.get_efield('Tx1', 1.0),
                self.survey.rec_coords)
        assert_allclose(
                self.simulation.data.synthetic[1, :, 0].data,
                rec_resp, atol=1e-16)

    def test_errors(self):
        with pytest.raises(TypeError, match='Unexpected '):
            simulations.Simulation(
                    'Test2', self.survey, self.grid, self.model, unknown=True)

        with pytest.raises(TypeError, match="Unknown `gridding`-option"):
            simulations.Simulation(
                    'Test2', self.survey, self.grid, self.model,
                    gridding='frequency')

        tsurvey = self.survey.copy()
        tsurvey.fixed = True
        with pytest.raises(NotImplementedError, match="`survey.fixed=False`"):
            simulations.Simulation(
                    'Test2', tsurvey, self.grid, self.model)

        tsurvey.fixed = False
        tsurvey.sources['Tx1'].electric = False
        with pytest.raises(NotImplementedError, match="for magnetic sources"):
            simulations.Simulation(
                    'Test2', tsurvey, self.grid, self.model)

        tsurvey.receivers['Rx01'].electric = False
        tsurvey.sources['Tx1'].electric = True
        with pytest.raises(NotImplementedError, match="or magnetic receivers"):
            simulations.Simulation(
                    'Test2', tsurvey, self.grid, self.model)

        # Provide grids
        grids = self.simulation._dict_grid.copy()

        # Ensure it works normally
        sim = simulations.Simulation(
                'Test2', self.survey, self.grid, self.model, gridding=grids)
        sim.get_model('Tx1', 1.0)

        # Delete one, ensure it fails.
        grids['Tx1'][1.0] = None
        sim = simulations.Simulation(
                'Test2', self.survey, self.grid, self.model, gridding=grids)
        with pytest.raises(TypeError, match="Provided grid-dict misses "):
            sim.get_grid('Tx1', 1.0)
        with pytest.raises(TypeError, match="Provided grid-dict misses "):
            sim.get_model('Tx1', 1.0)

    def test_reprs(self):
        test = self.simulation.__repr__()

        assert "*Simulation* «Test1»" in test
        assert "of Survey «Test»" in test
        assert "Survey: 3 sources; 12 receivers; 2 frequencies" in test
        assert "Model [resistivity]; isotropic; 32 x 16 x 16 (8,192)" in test
        assert "Gridding: Same grid as for model" in test

        test = self.simulation._repr_html_()
        assert "Simulation «Test1»" in test
        assert "of Survey «Test»" in test
        assert "Survey: 3 sources; 12 receivers; 2 frequencies" in test
        assert "Model [resistivity]; isotropic; 32 x 16 x 16 (8,192)" in test
        assert "Gridding: Same grid as for model" in test

    def test_copy(self, tmpdir):
        with pytest.raises(TypeError, match="Unrecognized `what`: nothing"):
            self.simulation.copy('nothing')

        sim2 = self.simulation.copy()
        assert self.simulation.name == sim2.name
        assert self.simulation.survey.sources == sim2.survey.sources
        assert_allclose(self.simulation.get_efield('Tx1', 1.0),
                        sim2.get_efield('Tx1', 1.0))

        # Also check to_file()/from_file().
        sim_dict = self.simulation.to_dict('all')
        sim2 = simulations.Simulation.from_dict(sim_dict)
        assert self.simulation.name == sim2.name
        assert self.simulation.survey.name == sim2.survey.name
        assert self.simulation.max_workers == sim2.max_workers
        assert self.simulation.gridding == sim2.gridding
        assert self.simulation.grid == sim2.grid
        assert self.simulation.model == sim2.model

        del sim_dict['survey']
        with pytest.raises(KeyError, match="Variable 'survey' missing"):
            simulations.Simulation.from_dict(sim_dict)

        # Also check to_file()/from_file().
        self.simulation.to_file(tmpdir+'/test.npz', what='all')
        sim2 = simulations.Simulation.from_file(tmpdir+'/test.npz')
        assert self.simulation.name == sim2.name
        assert self.simulation.survey.name == sim2.survey.name
        assert self.simulation.max_workers == sim2.max_workers
        assert self.simulation.gridding == sim2.gridding
        assert self.simulation.grid == sim2.grid
        assert self.simulation.model == sim2.model

        # Clean and ensure it is empty
        self.simulation.clean('all')
        assert self.simulation._dict_efield['Tx1'][1.0] is None
        assert self.simulation._dict_hfield['Tx1'][1.0] is None
        assert self.simulation._dict_efield_info['Tx1'][1.0] is None
        with pytest.raises(TypeError, match="Unrecognized `what`: nothing"):
            self.simulation.clean('nothing')

    def test_gradient(self):
        # Create another mesh, so there will be a difference.
        newgrid = meshes.TensorMesh(
                [np.ones(16)*500, np.ones(8)*1000, np.ones(8)*1000],
                np.array([-1250, -1250, -2250]))

        simulation = simulations.Simulation(
                'TestX', self.survey, self.grid, self.model, max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding=newgrid)

        grad = simulation.gradient

        assert grad.shape == self.model.shape


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_source_strength():

    # Create a simple survey; source with and source without strength.
    strength = 5.678
    srccoords = (0, 3000, -950, 0, 0)
    sources = [surveys.Dipole('NoStrength', srccoords),
               surveys.Dipole('Strength', srccoords, strength=strength)]
    receivers = (np.arange(12)*500, 0, -1000, 0, 0)
    frequencies = 1.0
    survey = surveys.Survey('Test', sources, receivers, frequencies)

    # Create a simple grid and model.
    grid = meshes.TensorMesh(
            [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
            np.array([-1250, -1250, -2250]))
    model = models.Model(grid, 1)

    # Create a simulation, compute all fields.
    simulation = simulations.Simulation(
            'Test2', survey, grid, model, max_workers=1,
            solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                         'linerelaxation': False, 'semicoarsening': False},
            gridding='same')

    simulation.compute()

    data = survey.data['synthetic'].values
    assert_allclose(data[0, :, :]*strength, data[1, :, :])
