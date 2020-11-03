import pytest
import numpy as np
from numpy.testing import assert_allclose

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None

from emg3d import meshes, models, surveys, simulations, fields, solver, io


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestSimulationSame():
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
        simulation.compute(observed=True)

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
        with pytest.raises(NotImplementedError, match="sources and receivers"):
            simulations.Simulation(
                    'Test2', tsurvey, self.grid, self.model)

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

    def test_synthetic(self):
        sim = self.simulation.copy()

        # Switch off noise_floor, relative_error, min_offset => No noise.
        sim.survey.noise_floor = None
        sim.survey.relative_error = None
        sim._dict_efield = sim._dict_initiate  # Reset
        sim.compute(observed=True)
        assert_allclose(sim.data.synthetic, sim.data.observed)
        assert sim.survey.size == sim.data.observed.size


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestSimulationOthers():
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
        simulation.compute(observed=True)

    def test_dicts_provided(self):
        grids = self.simulation._dict_grid.copy()

        # dict
        sim1 = simulations.Simulation(
                'Test2', self.survey, self.grid, self.model, gridding='dict',
                gridding_opts=grids)
        m1 = sim1.get_model('Tx1', 1.0)
        g1 = sim1.get_grid('Tx1', 1.0)

        # provide
        sim2 = simulations.Simulation(
                'Test2', self.survey, self.grid, self.model,
                gridding='input', gridding_opts=grids['Tx1'][1.0])
        m2 = sim2.get_model('Tx1', 1.0)
        g2 = sim2.get_grid('Tx1', 1.0)

        assert m1 == m2
        assert g1 == g2

    def test_errors(self):

        # gridding='same' with gridding_opts.
        with pytest.raises(TypeError, match="`gridding_opts` is not permitt"):
            simulations.Simulation(
                    'Test', self.survey, self.grid, self.model,
                    gridding='same', gridding_opts={'bummer': True})

    def test_gradient(self):
        # Create another mesh, so there will be a difference.
        newgrid = meshes.TensorMesh(
                [np.ones(16)*500, np.ones(8)*1000, np.ones(8)*1000],
                np.array([-1250, -1250, -2250]))

        simulation = simulations.Simulation(
                'TestX', self.survey, self.grid, self.model, max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding='input', gridding_opts=newgrid)

        grad = simulation.gradient

        assert grad.shape == self.model.shape

    def test_synthetic(self):
        sim = self.simulation.copy()

        # Switch off noise_floor, relative_error, min_offset => No noise.
        sim.survey.noise_floor = None
        sim.survey.relative_error = None
        sim._dict_efield = sim._dict_initiate  # Reset
        sim.compute(observed=True)
        assert_allclose(sim.data.synthetic, sim.data.observed)
        assert sim.survey.size == sim.data.observed.size

    def test_gridding(self):
        pass
        # (1) expand
        # (2) to/from_dict: 'input', 'dict', '!=same'
        # (3) get_grid
        # (4) get_model


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


def test_expand_grid_model():
    grid = meshes.TensorMesh([[4, 2, 2, 4], [2, 2, 2, 2], [1, 1]], (0, 0, 0))
    model = models.Model(grid, 1, np.ones(grid.vnC)*2, mu_r=3, epsilon_r=5)

    og, om = simulations.expand_grid_model(grid, model, [2, 3], 5)

    # Grid.
    assert_allclose(grid.vectorNz, og.vectorNz[:-2])
    assert og.vectorNz[-2] == 5
    assert og.vectorNz[-1] == 105

    # Property x (from float).
    assert_allclose(om.property_x[:, :, :-2], 1)
    assert_allclose(om.property_x[:, :, -2], 2)
    assert_allclose(om.property_x[:, :, -1], 3)

    # Property y (from vnC).
    assert_allclose(om.property_y[:, :, :-2], model.property_y)
    assert_allclose(om.property_y[:, :, -2], 2)
    assert_allclose(om.property_y[:, :, -1], 3)

    # Property y (from property_x).
    assert_allclose(om.property_z[:, :, :-2], 1)
    assert_allclose(om.property_z[:, :, -2], 2)
    assert_allclose(om.property_z[:, :, -1], 3)

    # Property mu_r (from float).
    assert_allclose(om.mu_r[:, :, :-2], 3)
    assert_allclose(om.mu_r[:, :, -2], 1)
    assert_allclose(om.mu_r[:, :, -1], 1)

    # Property epsilon_r (from float).
    assert_allclose(om.epsilon_r[:, :, :-2], 5)
    assert_allclose(om.epsilon_r[:, :, -2], 1)
    assert_allclose(om.epsilon_r[:, :, -1], 1)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestEstimateGriddingOpts():
    if xarray is not None:
        # Create a simple survey
        sources = (0, [1000, 3000, 5000], -950, 0, 0)
        receivers = (np.arange(11)*500, 2000, -1000, 0, 0)
        frequencies = (0.1, 10.0)

        survey = surveys.Survey('Test', sources, receivers, frequencies,
                                noise_floor=1e-15, relative_error=0.05)

        # Create a simple grid and model
        grid = meshes.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model = models.Model(grid, 0.1, np.ones(grid.vnC)*10)
        model.property_y[5, 8, 3] = 100000  # Cell at source center

    def test_empty_dict(self):
        gdict = simulations.estimate_gridding_opts(
                {}, self.grid, self.model, self.survey)

        assert gdict['frequency'] == 1.0
        assert gdict['mapping'].name == self.model.map.name
        assert_allclose(gdict['center'], (0, 3000, -950))
        assert_allclose(gdict['domain'][0], (-250, 5250))
        assert_allclose(gdict['domain'][1], (800, 5200))
        assert_allclose(gdict['domain'][2], (-3432.5, -677.5))
        assert_allclose(gdict['properties'], [100, 1, 1, 1, 1, 1, 1])

    def test_mapping_vector(self):
        gridding_opts = {
            'mapping': "LgConductivity",
            'vector': 'xZ',
            }
        gdict = simulations.estimate_gridding_opts(
                gridding_opts, self.grid, self.model, self.survey)

        assert_allclose(
                gdict['properties'],
                np.log10(1/np.array([100, 1, 1, 1, 1, 1, 1])), atol=1e-15)
        assert_allclose(gdict['vector'][0], self.grid.vectorNx)
        assert gdict['vector'][1] is None
        assert_allclose(gdict['vector'][2], self.grid.vectorNz)

    def test_pass_along(self):
        gridding_opts = {
            'vector': (None, 1, None),
            'stretching': [1.2, 1.3],
            'seasurface': -500,
            'cell_numbers': [10, 20, 30],
            'lambda_factor': 0.8,
            'max_buffer': 10000,
            'min_width_limits': [20, 40],
            'min_width_pps': 4,
            'verb': 3,
            }

        gdict = simulations.estimate_gridding_opts(
                gridding_opts.copy(), self.grid, self.model, self.survey)

        # Check that all parameters passed unchanged.
        gdict2 = {k: gdict[k] for k, _ in gridding_opts.items()}
        assert io._compare_dicts(gdict2, gridding_opts)

    def test_factor(self):

        sources = (0, 3000, -950, 0, 0)
        receivers = (0, 3000, -1000, 0, 0)

        # Adjusted x-domain.
        survey = surveys.Survey('Test', self.sources, receivers,
                                self.frequencies, noise_floor=1e-15,
                                relative_error=0.05)

        gdict = simulations.estimate_gridding_opts(
                    {}, self.grid, self.model, survey)

        assert_allclose(gdict['domain'][0], (-733, 733))

        # Adjusted x-domain.
        survey = surveys.Survey('Test', sources, self.receivers,
                                self.frequencies, noise_floor=1e-15,
                                relative_error=0.05)

        gdict = simulations.estimate_gridding_opts(
                    {}, self.grid, self.model, survey)

        assert_allclose(gdict['domain'][1], (1583, 3417))

    def test_error(self):
        with pytest.raises(TypeError, match='Unexpected gridding_opts'):
            _ = simulations.estimate_gridding_opts(
                    {'what': True}, self.grid, self.model, self.survey)
