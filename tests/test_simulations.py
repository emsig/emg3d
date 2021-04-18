import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import simulations

from . import helpers


# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestSimulation():
    if xarray is not None:
        # Create a simple survey

        # Sources: 1 Electric Dipole, 1 Magnetic Dipole, 1 Electric Wire.
        s1 = emg3d.TxElectricDipole((0, 1000, -950, 0, 0))
        s2 = emg3d.TxMagneticDipole((0, 3000, -950, 90, 0))
        s3 = emg3d.TxElectricWire(
                ([0, 4900, -950], [-20, 5000, -950], [20, 5100, -950]))
        sources = emg3d.surveys.txrx_lists_to_dict([s1, s2, s3])

        # Receivers: 1 Electric Point, 1 Magnetic Point
        e_rec = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint,
                (np.arange(6)*1000, 0, -1000, 0, 0))
        m_rec = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxMagneticPoint,
                (np.arange(6)*1000, 0, -1000, 90, 0))
        receivers = emg3d.surveys.txrx_lists_to_dict([e_rec, m_rec])

        # Frequencies
        frequencies = (1.0, 2.0)

        survey = emg3d.Survey(
                sources, receivers, frequencies, name='TestSurv',
                noise_floor=1e-15, relative_error=0.05)

        # Create a simple grid and model
        grid = emg3d.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model = emg3d.Model(grid, 1)

        # Create a simulation, compute all fields.
        simulation = simulations.Simulation(
                survey, model, name='TestSim', max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding='same')

        # Do first one single and then all together.
        simulation.get_efield('TxED-1', 'f-1')
        simulation.compute(observed=True, min_offset=1100)

    def test_same(self):
        assert self.simulation.get_model('TxMD-2', 'f-1') == self.model
        assert self.simulation.get_grid('TxEW-3', 1.0) == self.grid

    def test_fields(self, capsys):
        # Check sfield
        sfield = emg3d.get_source_field(
                self.grid, self.survey.sources['TxEW-3'], frequency=1.0)

        # Check efield
        efield, info = emg3d.solve(
                self.model, sfield, **self.simulation.solver_opts)
        assert self.simulation.get_efield('TxEW-3', 'f-1') == efield

        # Unknown keyword
        with pytest.raises(TypeError, match='Unexpected '):
            self.simulation.get_efield('TxEW-3', 1.0, unknownkeyward=True)

        # See a single one
        self.simulation._dict_efield['TxEW-3'][1.0] = None
        _, _ = capsys.readouterr()
        self.simulation.get_efield('TxEW-3', 1.0)

        info = self.simulation.get_efield_info('TxEW-3', 1.0)
        assert 'MAX. ITERATION REACHED, NOT CONVERGED' in info['exit_message']

        # Check hfield
        hfield = emg3d.get_magnetic_field(self.model, efield)
        assert self.simulation.get_hfield('TxEW-3', 1.0) == hfield
        s_hfield = self.simulation.get_hfield('TxEW-3', 1.0)
        assert s_hfield == hfield
        assert_allclose(
                self.simulation._dict_efield_info[
                    'TxEW-3']['f-1']['abs_error'],
                info['abs_error'])
        assert_allclose(
                self.simulation._dict_efield_info[
                    'TxEW-3']['f-1']['rel_error'],
                info['rel_error'])
        exit = self.simulation._dict_efield_info['TxEW-3']['f-1']['exit']
        assert exit == info['exit'] == 1

    def test_responses(self):
        # Check min_offset were switched-off
        assert_allclose(self.simulation.data.observed[0, 0, 0].data, np.nan)
        assert_allclose(self.simulation.data.observed[0, 6, 0].data, np.nan)

        # Get efield responses
        e_resp = self.simulation.get_efield('TxMD-2', 1.0).get_receiver(
                self.survey.receivers.values())
        assert_allclose(
                self.simulation.data.synthetic[1, :6, 0].data,
                e_resp[:6], atol=1e-16)

        # Get hfield responses
        m_resp = self.simulation.get_hfield('TxMD-2', 1.0).get_receiver(
                self.survey.receivers.values())
        assert_allclose(
                self.simulation.data.synthetic[1, 6:, 0].data,
                m_resp[6:], atol=1e-16)

    def test_errors(self):
        with pytest.raises(TypeError, match='Unexpected '):
            simulations.Simulation(self.survey, self.model,
                                   unknown=True, name='Test2')

        # gridding='same' with gridding_opts.
        with pytest.raises(TypeError, match="`gridding_opts` is not permitt"):
            simulations.Simulation(
                    self.survey, self.model, name='Test',
                    gridding='same', gridding_opts={'bummer': True})

        # expand without seasurface
        with pytest.raises(KeyError, match="is required if"):
            simulations.Simulation(
                    self.survey, self.model, name='Test',
                    gridding='single', gridding_opts={'expand': [1, 2]})

    def test_reprs(self):
        test = self.simulation.__repr__()

        assert "Simulation «TestSim»" in test
        assert "Survey «TestSurv»: 3 sources; 12 receivers; 2 frequenc" in test
        assert "Model: resistivity; isotropic; 32 x 16 x 16 (8,192)" in test
        assert "Gridding: Same grid as for model" in test

        test = self.simulation._repr_html_()
        assert "Simulation «TestSim»" in test
        assert "Survey «TestSurv»:    3 sources;    12 receivers;    2" in test
        assert "Model: resistivity; isotropic; 32 x 16 x 16 (8,192)" in test
        assert "Gridding: Same grid as for model" in test

    def test_copy(self, tmpdir):
        with pytest.raises(TypeError, match="Unrecognized `what`: nothing"):
            self.simulation.copy('nothing')

        sim2 = self.simulation.copy()
        assert self.simulation.name == sim2.name
        assert self.simulation.survey.sources == sim2.survey.sources
        assert_allclose(self.simulation.get_efield('TxED-1', 1.0).field,
                        sim2.get_efield('TxED-1', 1.0).field)

        # Also check to_file()/from_file().
        sim_dict = self.simulation.to_dict('all')
        sim2 = simulations.Simulation.from_dict(sim_dict.copy())
        assert self.simulation.name == sim2.name
        assert self.simulation.survey.name == sim2.survey.name
        assert self.simulation.max_workers == sim2.max_workers
        assert self.simulation.gridding == sim2.gridding
        assert self.simulation.model == sim2.model

        del sim_dict['survey']
        with pytest.raises(KeyError, match="'survey'"):
            simulations.Simulation.from_dict(sim_dict)

        # Also check to_file()/from_file().
        self.simulation.to_file(tmpdir+'/test.npz', what='all')
        sim2 = simulations.Simulation.from_file(tmpdir+'/test.npz')
        assert self.simulation.name == sim2.name
        assert self.simulation.survey.name == sim2.survey.name
        assert self.simulation.max_workers == sim2.max_workers
        assert self.simulation.gridding == sim2.gridding
        assert self.simulation.model == sim2.model

        sim9 = simulations.Simulation.from_file(tmpdir+'/test.npz', verb=-1)
        assert sim2.name == sim9[0].name
        assert 'Data loaded from' in sim9[1]

        # Clean and ensure it is empty
        sim3 = self.simulation.copy()
        sim3.clean('all')
        assert sim3._dict_efield['TxMD-2']['f-1'] is None
        assert sim3._dict_hfield['TxMD-2']['f-1'] is None
        assert sim3._dict_efield_info['TxMD-2']['f-1'] is None
        with pytest.raises(TypeError, match="Unrecognized `what`: nothing"):
            sim3.clean('nothing')

    def test_dicts_provided(self):
        grids = self.simulation._dict_grid.copy()

        # dict
        sim1 = simulations.Simulation(
                self.survey, self.model, gridding='dict',
                gridding_opts=grids, name='Test2')
        m1 = sim1.get_model('TxEW-3', 1.0)
        g1 = sim1.get_grid('TxEW-3', 1.0)

        # provide
        sim2 = simulations.Simulation(
                self.survey, self.model, name='Test2',
                gridding='input', gridding_opts=grids['TxEW-3']['f-1'])
        m2 = sim2.get_model('TxEW-3', 1.0)
        g2 = sim2.get_grid('TxEW-3', 1.0)

        assert m1 == m2
        assert g1 == g2

        # to/from_dict
        sim1copy = sim1.copy()
        sim2copy = sim2.copy()
        m1c = sim1copy.get_model('TxEW-3', 1.0)
        g1c = sim1copy.get_grid('TxEW-3', 1.0)
        m2c = sim2copy.get_model('TxEW-3', 1.0)
        g2c = sim2copy.get_grid('TxEW-3', 1.0)
        assert m1 == m1c
        assert g1 == g1c
        assert m2 == m2c
        assert g2 == g2c

    def test_synthetic(self):
        sim = self.simulation.copy()

        # Switch off noise_floor, relative_error, min_offset => No noise.
        sim.survey.noise_floor = None
        sim.survey.relative_error = None
        sim._dict_efield = sim._dict_initiate  # Reset
        sim.compute(observed=True)
        assert_allclose(sim.data.synthetic, sim.data.observed)
        assert sim.survey.size == sim.data.observed.size

    def test_input_gradient(self):
        # Create another mesh, so there will be a difference.
        newgrid = emg3d.TensorMesh(
                [np.ones(16)*500, np.ones(8)*1000, np.ones(8)*1000],
                np.array([-1250, -1250, -2250]))

        simulation = simulations.Simulation(
                self.survey, self.model, max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'sslsolver': False,
                             'linerelaxation': False, 'semicoarsening': False},
                gridding='input', gridding_opts=newgrid, name='TestX')

        grad = simulation.gradient

        # Ensure the gradient has the shape of the model, not of the input.
        assert grad.shape == self.model.shape

        sim2 = simulation.to_dict(what='all', copy=True)
        sim3 = simulation.to_dict(what='plain', copy=True)
        assert 'residual' in sim2['survey']['data'].keys()
        assert 'residual' not in sim3['survey']['data'].keys()

        simulation.clean('all')  # Should remove 'residual', 'bfield-dicts'
        sim5 = simulation.to_dict('all')
        assert 'residual' not in sim5['survey']['data'].keys()
        assert '_dict_bfield' not in sim5.keys()

    def test_simulation_automatic(self, capsys):
        # Create a simple survey
        sources = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.TxElectricDipole,
                (0, [1000, 3000, 5000], -950, 0, 0))
        receivers = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint,
                ([-3000, 0, 3000], [0, 3000, 6000], -1000, 0, 0))
        frequencies = (0.1, 1.0, 10.0)

        survey = emg3d.Survey(
                sources, receivers, frequencies, name='Test',
                noise_floor=1e-15, relative_error=0.05)

        # Create a simple grid and model
        grid = emg3d.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(4)*500],
                np.array([-1250, -1250, -2250]))
        model = emg3d.Model(grid, 1)

        # Create a simulation, compute all fields.
        inp = {'survey': survey, 'model': model,
               'gridding_opts': {
                   'expand': [1, 0.5], 'seasurface': 0, 'verb': 1}}
        b_sim = simulations.Simulation(name='both', gridding='both', **inp)
        f_sim = simulations.Simulation(
                name='freq', gridding='frequency', **inp)
        t_sim = simulations.Simulation(name='src', gridding='source', **inp)
        s_sim = simulations.Simulation(
                name='single', gridding='single', **inp)

        # Quick repr test.
        assert " 24 x 24 (13,824) - 160 x 160 x 96 (2,457," in b_sim.__repr__()
        assert " 24 x 24 (13,824) - 160 x 160 x 96 (2,457," in f_sim.__repr__()
        assert "Source-dependent grids; 64 x 64 x 40 (163," in t_sim.__repr__()
        assert "ources and frequencies; 64 x 64 x 40 (163," in s_sim.__repr__()

        # Quick print_grid test:
        _, _ = capsys.readouterr()  # empty
        b_sim.print_grid_info()
        out, _ = capsys.readouterr()
        assert "= Source: TxED-1; Frequency: 10.0 Hz =" in out
        assert "== GRIDDING IN X ==" in out
        assert b_sim.get_grid('TxED-1', 1.0).__repr__() in out

        _, _ = capsys.readouterr()  # empty
        out = f_sim.print_grid_info(return_info=True)
        out2, _ = capsys.readouterr()
        assert out2 == ""
        assert "= Source: all" in out
        assert "== GRIDDING IN X ==" in out
        assert f_sim.get_grid('TxED-3', 1.0).__repr__() in out

        t_sim.print_grid_info()
        out, _ = capsys.readouterr()
        assert "; Frequency: all" in out
        assert "== GRIDDING IN X ==" in out
        assert t_sim.get_grid('TxED-1', 10.0).__repr__() in out

        _, _ = capsys.readouterr()  # empty
        out = s_sim.print_grid_info(return_info=True)
        out2, _ = capsys.readouterr()
        assert out2 == ""
        assert "= Source: all; Frequency: all =" in out
        assert "== GRIDDING IN X ==" in out
        assert s_sim.get_grid('TxED-3', 0.1).__repr__() in out

        assert s_sim.print_grid_info(verb=-1) is None

        # Grids: Middle source / middle frequency should be the same in all.
        assert f_sim.get_grid('TxED-2', 1.0) == t_sim.get_grid('TxED-2', 1.0)
        assert f_sim.get_grid('TxED-2', 1.0) == s_sim.get_grid('TxED-2', 1.0)
        assert f_sim.get_grid('TxED-2', 1.0) == b_sim.get_grid('TxED-2', 1.0)
        assert f_sim.get_model('TxED-2', 1.0) == t_sim.get_model('TxED-2', 1.0)
        assert f_sim.get_model('TxED-2', 1.0) == s_sim.get_model('TxED-2', 1.0)
        assert f_sim.get_model('TxED-2', 1.0) == b_sim.get_model('TxED-2', 1.0)

        # Copy:
        f_sim_copy = f_sim.copy()
        assert (f_sim.get_grid('TxED-1', 1.0) ==
                f_sim_copy.get_grid('TxED-1', 1.0))
        assert 'expand' not in f_sim.gridding_opts.keys()
        assert 'expand' not in f_sim_copy.gridding_opts.keys()
        s_sim_copy = s_sim.copy()
        assert (s_sim.get_grid('TxED-1', 1.0) ==
                s_sim_copy.get_grid('TxED-1', 1.0))
        assert 'expand' not in s_sim.gridding_opts.keys()
        assert 'expand' not in s_sim_copy.gridding_opts.keys()
        t_sim_copy = t_sim.copy()
        assert (t_sim.get_grid('TxED-1', 1.0) ==
                t_sim_copy.get_grid('TxED-1', 1.0))
        assert 'expand' not in t_sim.gridding_opts.keys()
        assert 'expand' not in t_sim_copy.gridding_opts.keys()

    def test_print_solver(self, capsys):
        grid = emg3d.TensorMesh(
                h=[[(25, 10, -1.04), (25, 28), (25, 10, 1.04)],
                   [(50, 8, -1.03), (50, 16), (50, 8, 1.03)],
                   [(30, 8, -1.05), (30, 16), (30, 8, 1.05)]],
                origin='CCC')

        model = emg3d.Model(grid, property_x=1.5, property_y=1.8,
                            property_z=3.3, mapping='Resistivity')

        sources = emg3d.TxElectricDipole((0, 0, 0, 0, 0))
        receivers = [emg3d.RxElectricPoint((x, 0, 0, 0, 0))
                     for x in [-10000, 10000]]
        survey = emg3d.Survey(
            name='Test', sources=sources,
            receivers=receivers,
            frequencies=1.0, noise_floor=1e-15, relative_error=0.05,
        )

        inp = {'name': 'Test', 'survey': survey, 'model': model,
               'gridding': 'same'}
        sol = {'sslsolver': False, 'semicoarsening': False,
               'linerelaxation': False, 'maxit': 1}

        # Errors but verb=-1.
        _, _ = capsys.readouterr()  # empty
        simulation = simulations.Simulation(verb=-1, **inp, solver_opts=sol)
        simulation.compute()
        out, _ = capsys.readouterr()
        assert out == ""

        # Errors with verb=0.
        out = simulation.print_solver_info('efield', verb=0, return_info=True)
        assert "= Source TxED-1; Frequency 1.0 Hz = MAX. ITERATION REAC" in out

        # Errors with verb=1.
        _, _ = capsys.readouterr()  # empty
        simulation.print_solver_info('efield', verb=1)
        out, _ = capsys.readouterr()
        assert "= Source TxED-1; Frequency 1.0 Hz = 2.6e-02; 1; 0:00:" in out

        # No errors; solver-verb 3.
        simulation = simulations.Simulation(
                verb=1, **inp, solver_opts={'verb': 3})
        simulation.compute()
        out, _ = capsys.readouterr()
        assert 'MG-cycle' in out
        assert 'CONVERGED' in out

        # No errors; solver-verb 0.
        simulation = simulations.Simulation(
                verb=1, **inp, solver_opts={'verb': 0})
        simulation.compute()
        out, _ = capsys.readouterr()
        assert "= Source TxED-1; Frequency 1.0 Hz = CONVERGED" in out

    def test_rel_abs_rec(self):
        # Sources
        sources = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.TxElectricDipole, ([0, 100, 200], 0, 0, 0, 0))

        # Abs and rel Receivers
        a_e_rec = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint, (1000+np.arange(3)*100, 0, -100, 0, 0))
        r_e_rec = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint, (1000, 0, -100, 0, 0), relative=True)
        a_h_rec = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxMagneticPoint, (1000+np.arange(3)*100, 0, -100, 0, 0))
        r_h_rec = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxMagneticPoint, (1000, 0, -100, 0, 0), relative=True)
        receivers = emg3d.surveys.txrx_lists_to_dict(
                [a_e_rec, r_e_rec, a_h_rec, r_h_rec])

        # Frequencies
        frequencies = (1.0)

        survey = emg3d.Survey(
                sources, receivers, frequencies, name='TestSurv',
                noise_floor=1e-15, relative_error=0.05)

        # Create a simple grid and model
        grid = emg3d.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model = emg3d.Model(grid, 1)

        # Create a simulation, compute all fields.
        simulation = simulations.Simulation(
                survey, model, name='TestSim', max_workers=1,
                solver_opts={'maxit': 1, 'verb': 0, 'plain': True},
                gridding='same')

        simulation.compute()

        # Relative receivers must be same as corresponding absolute receivers
        assert_allclose(
            [simulation.data.synthetic[i, i, 0].data for i in range(3)],
            simulation.data.synthetic[:, 3, 0].data
        )

        assert_allclose(
            [simulation.data.synthetic[i, i+4, 0].data for i in range(3)],
            simulation.data.synthetic[:, 7, 0].data
        )


def test_expand_grid_model():
    grid = emg3d.TensorMesh([[4, 2, 2, 4], [2, 2, 2, 2], [1, 1]], (0, 0, 0))
    model = emg3d.Model(grid, 1, np.ones(grid.shape_cells)*2, mu_r=3,
                        epsilon_r=5)

    o_model = simulations.expand_grid_model(model, [2, 3], 5)

    # Grid.
    assert_allclose(grid.nodes_z, o_model.grid.nodes_z[:-2])
    assert o_model.grid.nodes_z[-2] == 5
    assert o_model.grid.nodes_z[-1] == 105

    # Property x (from float).
    assert_allclose(o_model.property_x[:, :, :-2], 1)
    assert_allclose(o_model.property_x[:, :, -2], 2)
    assert_allclose(o_model.property_x[:, :, -1], 3)

    # Property y (from shape_cells).
    assert_allclose(o_model.property_y[:, :, :-2], model.property_y)
    assert_allclose(o_model.property_y[:, :, -2], 2)
    assert_allclose(o_model.property_y[:, :, -1], 3)

    # Property z.
    assert o_model.property_z is None

    # Property mu_r (from float).
    assert_allclose(o_model.mu_r[:, :, :-2], 3)
    assert_allclose(o_model.mu_r[:, :, -2], 1)
    assert_allclose(o_model.mu_r[:, :, -1], 1)

    # Property epsilon_r (from float).
    assert_allclose(o_model.epsilon_r[:, :, :-2], 5)
    assert_allclose(o_model.epsilon_r[:, :, -2], 1)
    assert_allclose(o_model.epsilon_r[:, :, -1], 1)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestEstimateGriddingOpts():
    if xarray is not None:
        # Create a simple survey
        sources = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.TxElectricDipole,
                (0, [1000, 3000, 5000], -950, 0, 0))
        receivers = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint,
                (np.arange(11)*500, 2000, -1000, 0, 0))
        frequencies = (0.1, 10.0)

        survey = emg3d.Survey(
                sources, receivers, frequencies, noise_floor=1e-15,
                relative_error=0.05)

        # Create a simple grid and model
        grid = emg3d.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model = emg3d.Model(grid, 0.1, np.ones(grid.shape_cells)*10)
        model.property_y[5, 8, 3] = 100000  # Cell at source center

    def test_empty_dict(self):
        gdict = simulations.estimate_gridding_opts({}, self.model, self.survey)

        assert gdict['frequency'] == 1.0
        assert gdict['mapping'] == self.model.map.name
        assert_allclose(gdict['center'], (0, 3000, -950))
        assert_allclose(gdict['domain']['x'], (-500, 5500))
        assert_allclose(gdict['domain']['y'], (600, 5400))
        assert_allclose(gdict['domain']['z'], (-3651, -651))
        assert_allclose(gdict['properties'], [100000, 10, 10, 10, 10, 10, 10])

    def test_mapping_vector(self):
        gridding_opts = {
            'mapping': "LgConductivity",
            'vector': 'xZ',
            }
        gdict = simulations.estimate_gridding_opts(
                gridding_opts, self.model, self.survey)

        assert_allclose(
                gdict['properties'],
                np.log10(1/np.array([100000, 10, 10, 10, 10, 10, 10])),
                atol=1e-15)
        assert_allclose(gdict['vector']['x'], self.grid.nodes_x)
        assert gdict['vector']['y'] is None
        assert_allclose(gdict['vector']['z'], self.grid.nodes_z)

    def test_vector_domain_distance(self):
        gridding_opts = {
                'vector': 'Z',
                'domain': (None, [-1000, 1000], None),
                'distance': [[5, 10], None, None],
                }
        gdict = simulations.estimate_gridding_opts(
                gridding_opts, self.model, self.survey)

        assert gdict['vector']['x'] == gdict['vector']['y'] is None
        assert_allclose(gdict['vector']['z'], self.model.grid.nodes_z)

        assert gdict['domain']['x'] is None
        assert gdict['domain']['y'] == [-1000, 1000]
        assert gdict['domain']['z'] == [self.model.grid.nodes_z[0],
                                        self.model.grid.nodes_z[-1]]
        assert gdict['distance']['x'] == [5, 10]
        assert gdict['distance']['y'] == gdict['distance']['z'] is None

        # As dict
        gridding_opts = {
                'vector': 'Z',
                'domain': {'x': None, 'y': [-1000, 1000], 'z': None},
                'distance': {'x': [5, 10], 'y': None, 'z': None},
                }
        gdict = simulations.estimate_gridding_opts(
                gridding_opts, self.model, self.survey)

        assert gdict['vector']['x'] == gdict['vector']['y'] is None
        assert_allclose(gdict['vector']['z'], self.model.grid.nodes_z)

        assert gdict['domain']['x'] is None
        assert gdict['domain']['y'] == [-1000, 1000]
        assert gdict['domain']['z'] == [self.model.grid.nodes_z[0],
                                        self.model.grid.nodes_z[-1]]
        assert gdict['distance']['x'] == [5, 10]
        assert gdict['distance']['y'] == gdict['distance']['z'] is None

    def test_pass_along(self):
        gridding_opts = {
            'vector': {'x': None, 'y': 1, 'z': None},
            'stretching': [1.2, 1.3],
            'seasurface': -500,
            'cell_numbers': [10, 20, 30],
            'lambda_factor': 0.8,
            'max_buffer': 10000,
            'min_width_limits': ([20, 40], [20, 40], [20, 40]),
            'min_width_pps': 4,
            'verb': -1,
            }

        gdict = simulations.estimate_gridding_opts(
                gridding_opts.copy(), self.model, self.survey)

        # Check that all parameters passed unchanged.
        gdict2 = {k: gdict[k] for k, _ in gridding_opts.items()}
        # Except the tuple, which should be a dict now
        gridding_opts['min_width_limits'] = {
                'x': gridding_opts['min_width_limits'][0],
                'y': gridding_opts['min_width_limits'][1],
                'z': gridding_opts['min_width_limits'][2]
        }
        assert helpers.compare_dicts(gdict2, gridding_opts)

    def test_factor(self):
        sources = emg3d.TxElectricDipole((0, 3000, -950, 0, 0))
        receivers = emg3d.RxElectricPoint((0, 3000, -1000, 0, 0))

        # Adjusted x-domain.
        survey = emg3d.Survey(
                self.sources, receivers, self.frequencies, noise_floor=1e-15,
                relative_error=0.05)

        gdict = simulations.estimate_gridding_opts({}, self.model, survey)

        assert_allclose(gdict['domain']['x'], (-800, 800))

        # Adjusted x-domain.
        survey = emg3d.Survey(
                sources, self.receivers, self.frequencies, noise_floor=1e-15,
                relative_error=0.05)

        gdict = simulations.estimate_gridding_opts({}, self.model, survey)

        assert_allclose(gdict['domain']['y'], (1500, 3500))

    def test_error(self):
        with pytest.raises(TypeError, match='Unexpected gridding_opts'):
            _ = simulations.estimate_gridding_opts(
                    {'what': True}, self.model, self.survey)
