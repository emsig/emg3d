import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import simulations

from . import alternatives, helpers


# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None
try:
    import discretize
except ImportError:
    discretize = None
try:
    import h5py
except ImportError:
    h5py = False


# Random number generator.
rng = np.random.default_rng()


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

        # First hfield, ensure efield/hfield get computed.
        sim = self.simulation.copy(what='all')
        sim._dict_efield['TxEW-3']['f-1'] = None
        sim.get_hfield('TxEW-3', 'f-1')
        assert sim._dict_efield['TxEW-3']['f-1'] is not None

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
            with pytest.warns(FutureWarning, match='A property-complete'):
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
        _ = sim3.misfit
        sim3.clean('all')
        assert sim3._dict_efield['TxMD-2']['f-1'] is None
        assert sim3._dict_efield_info['TxMD-2']['f-1'] is None
        with pytest.raises(TypeError, match="Unrecognized `what`: nothing"):
            sim3.clean('nothing')
        with pytest.raises(AttributeError, match="no attribute 'weights'"):
            sim3.data.weights
        with pytest.raises(AttributeError, match="no attribute 'residuals'"):
            sim3.data.residuals

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

    def test_grid_provided(self):
        # Check bad grid
        hx = np.ones(17)*20
        grid = emg3d.TensorMesh([hx, hx, hx], (0, 0, 0))
        with pytest.warns(UserWarning, match='optimal for MG solver. Good n'):
            simulations.Simulation(self.survey, self.model, gridding='input',
                                   gridding_opts=grid)

    def test_synthetic(self):
        sim = self.simulation.copy()
        sim._dict_efield = sim._dict_initiate  # Reset
        sim.compute(observed=True, add_noise=False)
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

        with pytest.warns(UserWarning, match='Receiver responses were obtain'):
            grad = simulation.gradient

        with pytest.warns(UserWarning, match='Receiver responses were obtain'):
            vec = simulation.data.residual.data.copy()
            vec *= simulation.data.weights.data
            jtvec = simulation.jtvec(vec)
        assert_allclose(grad, jtvec)

        jvec = simulation.jvec(np.ones(self.grid.shape_cells))
        assert jvec.shape == simulation.data.observed.data.shape

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
        inp = {
            'survey': survey, 'model': model,
            'gridding_opts': {
                'expand': [1, 0.5], 'seasurface': 0, 'verb': 1,
                'center_on_edge': (False, True, False),
            },
        }

        with pytest.warns(FutureWarning, match='A property-complete'):
            b_sim = simulations.Simulation(name='both', gridding='both', **inp)
        with pytest.warns(FutureWarning, match='A property-complete'):
            f_sim = simulations.Simulation(
                name='freq', gridding='frequency', **inp)
        with pytest.warns(FutureWarning, match='A property-complete'):
            t_sim = simulations.Simulation(
                name='src', gridding='source', **inp)
        with pytest.warns(FutureWarning, match='A property-complete'):
            s_sim = simulations.Simulation(
                name='single', gridding='single', **inp)

        # Quick repr test.
        assert " 24 x 16 (12,288) - 160 x 160 x 96 (2,457," in b_sim.__repr__()
        assert " 24 x 16 (12,288) - 160 x 160 x 96 (2,457," in f_sim.__repr__()
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

        # Two sources, only compute 1, assure printing works.
        sources = [emg3d.TxElectricDipole((x, 0, 0, 0, 0)) for x in [0, 10]]
        survey = emg3d.Survey(
            name='Test', sources=sources,
            receivers=receivers,
            frequencies=1.0, noise_floor=1e-15, relative_error=0.05,
        )

        inp = {'name': 'Test', 'survey': survey, 'model': model,
               'gridding': 'same'}
        simulation = simulations.Simulation(**inp, solver_opts={'verb': 0})
        _ = simulation.get_efield('TxED-2', 'f-1')
        simulation.print_solver_info(verb=1)
        out, _ = capsys.readouterr()
        assert "= Source TxED-2; Frequency 1.0 Hz = CONVERGED" in out

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

    def test_tqdm(self):
        inp = {'survey': self.survey, 'model': self.model}

        sim = simulations.Simulation(tqdm_opts=False, **inp)
        assert sim._tqdm_opts['disable']

        sim = simulations.Simulation(tqdm_opts=True, **inp)
        assert not sim._tqdm_opts['disable']
        assert sim._tqdm_opts['bar_format']

        tqdm_opts = {'bar_format': '{bar}'}
        sim = simulations.Simulation(tqdm_opts={'bar_format': '{bar}'}, **inp)
        assert sim._tqdm_opts == tqdm_opts


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestLayeredSimulation():
    if xarray is not None:
        # Create a simple survey

        # Sources: 1 Electric Dipole, 1 Magnetic Dipole.
        s1 = emg3d.TxElectricDipole((0, 1000, -950, 0, 0))
        s2 = emg3d.TxMagneticDipole((0, 3000, -950, 90, 0))
        sources = emg3d.surveys.txrx_lists_to_dict([s1, s2])

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
            survey, model, name='TestSim', max_workers=1, gridding='single',
            layered=True, layered_opts={'ellipse': {'minor': 0.99}}
        )

        # Compute observed.
        simulation.compute(observed=True, min_offset=1100)

    def test_reprs(self):
        test = self.simulation.__repr__()

        assert "Simulation «TestSim»" in test
        assert "Survey «TestSurv»: 2 sources; 12 receivers; 2 frequenc" in test
        assert "Model: resistivity; isotropic; 32 x 16 x 16 (8,192)" in test
        assert "Gridding: layered computation using method 'cylinder'" in test

        test = self.simulation._repr_html_()
        assert "Simulation «TestSim»" in test
        assert "Survey «TestSurv»:    2 sources;    12 receivers;    2" in test
        assert "Model: resistivity; isotropic; 32 x 16 x 16 (8,192)" in test
        assert "Gridding: layered computation using method 'cylinder'" in test

    def test_copy(self, tmpdir):

        sim2 = self.simulation.copy()
        assert self.simulation.layered == sim2.layered
        assert helpers.compare_dicts(self.simulation.layered_opts,
                                     sim2.layered_opts)
        assert sim2.layered_opts['ellipse']['minor'] == 0.99

    def test_layered_opts(self):
        with pytest.raises(AttributeError, match="can't set attribute"):
            self.simulation.layered_opts = {}

        inp = {'ellipse': {'minor': 0.99}}
        simulation = simulations.Simulation(
            self.survey, self.model, gridding='single', layered_opts=inp
        )

        # layered=False: layered_opts just stored as is
        assert helpers.compare_dicts(simulation.layered_opts, inp)

        # layered=True: layered_opts complete
        simulation.layered = True
        assert simulation.layered_opts['ellipse']['factor'] == 1.2


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_misfit():
    data = 1
    syn = 5
    rel_err = 0.05
    sources = emg3d.TxElectricDipole((0, 0, 0, 0, 0))
    receivers = emg3d.RxElectricPoint((5, 0, 0, 0, 0))

    survey = emg3d.Survey(
        sources=sources, receivers=receivers, frequencies=100,
        data=np.zeros((1, 1, 1))+data, relative_error=0.05,
    )

    grid = emg3d.TensorMesh([np.ones(10)*2, [2, 2], [2, 2]], (-10, -2, -2))
    model = emg3d.Model(grid, 1)

    simulation = simulations.Simulation(survey=survey, model=model)

    field = emg3d.Field(grid, dtype=np.float64)
    field.field += syn
    simulation._dict_efield['TxED-1']['f-1'] = field
    simulation.data['synthetic'] = simulation.data['observed']*0 + syn

    misfit = 0.5*((syn-data)/(rel_err*data))**2

    def dummy():
        pass

    simulation.compute = dummy  # => switch of compute()

    assert_allclose(simulation.misfit, misfit)

    # Missing noise_floor / std.
    survey = emg3d.Survey(sources, receivers, 100)
    simulation = simulations.Simulation(
        survey=survey, model=model, gridding_opts={'center_on_edge': True})
    with pytest.raises(ValueError, match="Either `noise_floor` or"):
        simulation.misfit


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestGradient:

    if xarray is not None:

        # Create a simple mesh.
        hx = np.ones(32)*200
        mesh = emg3d.TensorMesh([hx, hx, hx], origin=[0, 0, 0])

        # Define a simple survey, including 1 el. & 1 magn. receiver
        survey = emg3d.Survey(
            sources=emg3d.TxElectricDipole((1680, 3220, 3210, 20, 5)),
            receivers=[
                emg3d.RxElectricPoint((4751, 3280, 3220, 33, 14)),
                emg3d.RxMagneticPoint((4758, 3230, 3250, 15, 70)),
            ],
            frequencies=1.0,
            relative_error=0.01,
        )

        # Background Model
        con_init = np.ones(mesh.shape_cells)*np.pi

        # Target Model 1: One Block
        con_true = np.ones(mesh.shape_cells)*np.pi
        con_true[13:18, 13:18, 7:13] = 0.001

        # Random mapping
        mapping = rng.choice([
            'Conductivity', 'LgConductivity', 'LnConductivity',
            'Resistivity', 'LgResistivity', 'LnResistivity',
        ])
        k = getattr(emg3d.maps, 'Map'+mapping)()

        model_init = emg3d.Model(mesh, k.forward(con_init), mapping=mapping)
        model_true = emg3d.Model(mesh, k.forward(con_true), mapping=mapping)

        # mesh.plot_3d_slicer(con_true)  # For debug / QC, needs discretize

        sim_inp = {
            'survey': survey,
            'solver_opts': {'plain': True},
            'max_workers': 1,
            'gridding': 'same',
            'verb': 0,
            'receiver_interpolation': 'linear',
        }

        # Compute data (pre-computed and passed to Survey above)
        sim_data = simulations.Simulation(model=model_true, **sim_inp)
        sim_data.compute(observed=True)
        del sim_inp['survey'].data['synthetic']

    def test_errors(self):
        mesh = emg3d.TensorMesh([[2, 2], [2, 2], [2, 2]], origin=(-1, -1, -1))
        survey = emg3d.Survey(
            sources=emg3d.TxElectricDipole((-1.5, 0, 0, 0, 0)),
            receivers=emg3d.RxElectricPoint((1.5, 0, 0, 0, 0)),
            frequencies=1.0,
            relative_error=0.01,
        )
        sim_inp = {'survey': survey, 'gridding': 'same',
                   'receiver_interpolation': 'linear'}

        # Model with electric permittivity.
        simulation = simulations.Simulation(
                model=emg3d.Model(mesh, epsilon_r=3), **sim_inp)
        with pytest.raises(NotImplementedError, match='for el. permittivity'):
            simulation.gradient

        # Model with magnetic permeability.
        simulation = simulations.Simulation(
                model=emg3d.Model(mesh, mu_r=np.ones(mesh.shape_cells)*np.pi),
                **sim_inp)
        with pytest.raises(NotImplementedError, match='for magn. permeabili'):
            simulation.gradient

    def test_as_vs_fd_gradient(self, capsys):

        # Compute adjoint state misfit and gradient
        sim = simulations.Simulation(model=self.model_init, **self.sim_inp)
        data_misfit = sim.misfit
        grad = sim.gradient

        # For Debug / QC, needs discretize
        # from matplotlib.colors import LogNorm, SymLogNorm
        # mesh.plot_3d_slicer(
        #         grad.ravel('F'),
        #         pcolor_opts={
        #             'cmap': 'RdBu_r',
        #             'norm': SymLogNorm(linthresh=1e-2, base=10,
        #                                vmin=-1e1, vmax=1e1)}
        #         )

        # We test a random cell. The NRMSD should be way below 1 %. However,
        # - close to the boundary,
        # - in regions where the gradient is almost zero,
        # - in regions where the gradient changes sign, and
        # - close to the source,
        # the NRMSD can become large. This is mainly due to numerics and our
        # coarse mesh (MAIN REASON; used for speed). We limit therefore the
        # "randomness" to cells where the gradient (resp. its
        # "conductivity-equivalent") is bigger than 3.0 (gradient varies
        # roughly from -70 to +70), exclude cells close to the source, and only
        # assure that the NRMSD is less than 1.5 %.

        # The mapping is random, but for the limitation check, we use the
        # conductivity equivalent by undoing map.derivative_chain().
        cgrad = grad.copy()
        if self.mapping == 'Resistivity':
            cgrad /= -self.con_init**2
        elif self.mapping == 'LgResistivity':
            cgrad /= -self.con_init*np.log(10)
        elif self.mapping == 'LnResistivity':
            cgrad /= -self.con_init
        elif self.mapping == 'LgConductivity':
            cgrad /= self.con_init*np.log(10)
        elif self.mapping == 'LnConductivity':
            cgrad /= self.con_init

        # Exclude cells around the source.
        cgrad[5:12, 12:20, 12:20] = 0.0

        # Get possible indices and select a random set.
        indices = np.argwhere(abs(cgrad) > 3.0)
        ix, iy, iz = indices[np.random.randint(indices.shape[0])]

        # Compute NRMSD.
        nrmsd = alternatives.fd_vs_as_gradient(
                (ix, iy, iz), self.model_init, grad, data_misfit, self.sim_inp)

        assert nrmsd < 1.5

    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    @pytest.mark.skipif(h5py is None, reason="h5py not installed.")
    def test_adjoint(self, tmpdir):

        sim = simulations.Simulation(
                model=self.model_init, file_dir=str(tmpdir), **self.sim_inp)

        discretize.tests.assert_isadjoint(
            lambda u: sim.jvec(u).real,  # Because jtvec returns .real
            sim.jtvec,
            self.mesh.shape_cells,
            self.survey.shape,
        )

        sim = simulations.Simulation(
                model=self.model_init, file_dir=str(tmpdir),
                **{**self.sim_inp, 'gridding': 'both',
                   'gridding_opts': {'vector': 'xyz'}}
        )

        discretize.tests.assert_isadjoint(
            lambda u: sim.jvec(u).real,  # Because jtvec returns .real
            sim.jtvec,
            self.mesh.shape_cells,
            self.survey.shape,
        )

    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    @pytest.mark.skipif(h5py is None, reason="h5py not installed.")
    def test_adjoint_hti(self):

        model_init = emg3d.Model(
            self.mesh,
            self.k.forward(self.con_init),
            property_y=self.k.forward(1.1*self.con_init),
            mapping=self.mapping
        )

        sim = simulations.Simulation(model=model_init, **self.sim_inp)

        discretize.tests.assert_isadjoint(
            lambda u: sim.jvec(u).real,
            sim.jtvec,
            (2, *self.mesh.shape_cells),
            self.survey.shape,
        )

    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    @pytest.mark.skipif(h5py is None, reason="h5py not installed.")
    def test_adjoint_vti(self):

        model_init = emg3d.Model(
            self.mesh,
            self.k.forward(self.con_init),
            property_z=self.k.forward(1.4*self.con_init),
            mapping=self.mapping
        )

        sim = simulations.Simulation(model=model_init, **self.sim_inp)

        discretize.tests.assert_isadjoint(
            lambda u: sim.jvec(u).real,
            sim.jtvec,
            (2, *self.mesh.shape_cells),
            self.survey.shape,
        )

    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    @pytest.mark.skipif(h5py is None, reason="h5py not installed.")
    def test_adjoint_triaxial(self):

        model_init = emg3d.Model(
            self.mesh,
            self.k.forward(self.con_init),
            property_y=self.k.forward(1.1*self.con_init),
            property_z=self.k.forward(1.4*self.con_init),
            mapping=self.mapping
        )

        sim = simulations.Simulation(model=model_init, **self.sim_inp)

        discretize.tests.assert_isadjoint(
            lambda u: sim.jvec(u).real,
            sim.jtvec,
            (3, *self.mesh.shape_cells),
            self.survey.shape,
        )

    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    @pytest.mark.skipif(h5py is None, reason="h5py not installed.")
    def test_misfit(self, tmpdir):

        sim = simulations.Simulation(
                model=self.model_init, file_dir=str(tmpdir), **self.sim_inp)
        m0 = 2*sim.model.property_x

        def func2(x):
            sim.model.property_x[...] = m0
            return sim.jvec(x.reshape(sim.model.shape, order='F'))

        def func1(x):
            sim.model.property_x[...] = x.reshape(sim.model.shape)
            sim.clean('computed')

            # Quick test that clean() removes the files
            assert len(os.listdir(tmpdir)) == 0

            sim.compute()
            return sim.data.synthetic.data, func2

        assert discretize.tests.check_derivative(
            func1, m0, plotIt=False, num=3,
        )

    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    @pytest.mark.skipif(h5py is None, reason="h5py not installed.")
    def test_jtvec_gradient(self):
        # Gradient is the same as jtvec(residual*weights).
        sim = simulations.Simulation(model=self.model_init, **self.sim_inp)
        g = sim.gradient
        j = sim.jtvec(vector=sim.survey.data.residual*sim.survey.data.weights)
        assert_allclose(g, j)


def test_all_dir():
    assert set(simulations.__all__) == set(dir(simulations))
