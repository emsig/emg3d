import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import simulations, optimize

from . import alternatives


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

        # See a single one
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

        # Test deprecation v1.4.0
        with pytest.warns(FutureWarning, match="removed in v1.4.0"):
            grad2 = optimize.gradient(simulation)
        assert_allclose(grad, grad2)

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

    # Test deprecation v1.4.0
    with pytest.warns(FutureWarning, match="removed in v1.4.0"):
        misfit2 = optimize.misfit(simulation)
    assert_allclose(misfit, misfit2)

    # Missing noise_floor / std.
    survey = emg3d.Survey(sources, receivers, 100)
    simulation = simulations.Simulation(survey=survey, model=model)
    with pytest.raises(ValueError, match="Either `noise_floor` or"):
        simulation.misfit


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestGradient:

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

        # Anisotropic models.
        simulation = simulations.Simulation(
                model=emg3d.Model(mesh, 1, 2, 3), **sim_inp)
        with pytest.raises(NotImplementedError, match='for isotropic models'):
            simulation.gradient

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
        # Create a simple mesh.
        hx = np.ones(64)*100
        mesh = emg3d.TensorMesh([hx, hx, hx], origin=[0, 0, 0])

        # Define a simple survey, including 1 el. & 1 magn. receiver
        survey = emg3d.Survey(
            sources=emg3d.TxElectricDipole((1650, 3200, 3200, 0, 0)),
            receivers=[
                emg3d.RxElectricPoint((4750, 3200, 3200, 0, 0)),
                emg3d.RxMagneticPoint((4750, 3200, 3200, 90, 0)),
            ],
            frequencies=1.0,
            relative_error=0.01,
        )

        # Background Model
        con_init = np.ones(mesh.shape_cells)

        # Target Model 1: One Block
        con_true = np.ones(mesh.shape_cells)
        con_true[27:37, 27:37, 15:25] = 0.001

        model_init = emg3d.Model(mesh, con_init, mapping='Conductivity')
        model_true = emg3d.Model(mesh, con_true, mapping='Conductivity')
        # mesh.plot_3d_slicer(con_true)  # For debug / QC, needs discretize

        sim_inp = {
            'survey': survey,
            'solver_opts': {'plain': True, 'tol': 5e-5},  # Red. tol 4 speed
            'max_workers': 1,
            'gridding': 'same',
            'verb': 0,
            'receiver_interpolation': 'linear',
        }

        # Compute data (pre-computed and passed to Survey above)
        sim_data = simulations.Simulation(model=model_true, **sim_inp)
        sim_data.compute(observed=True)

        # Compute adjoint state misfit and gradient
        sim_data = simulations.Simulation(model=model_init, **sim_inp)
        data_misfit = sim_data.misfit
        grad = sim_data.gradient

        # For Debug / QC, needs discretize
        # from matplotlib.colors import LogNorm, SymLogNorm
        # mesh.plot_3d_slicer(
        #         grad.ravel('F'),
        #         pcolor_opts={
        #             'cmap': 'RdBu_r',
        #             'norm': SymLogNorm(linthresh=1e-2, base=10,
        #                                vmin=-1e1, vmax=1e1)}
        #         )

        # We test a pseudo-random cell from the inline xz slice.
        #
        # The NRMSD is (should) be below 1 %. However, (a) close to the
        # boundary, (b) in regions where the gradient is almost zero, and (c)
        # in regions where the gradient changes sign the NRMSD can become
        # large. This is mainly due to numerics, our coarse mesh, and the
        # reduced tolerance (which we reduced for speed). As such we only
        # sample pseudo-random from 200 cells.
        ixyz = ([20, 32, 17], [20, 32, 23], [20, 32, 24], [20, 32, 25],
                [20, 32, 26], [20, 32, 27], [20, 32, 28], [20, 32, 29],
                [20, 32, 30], [20, 32, 31], [20, 32, 32], [20, 32, 33],
                [20, 32, 34], [20, 32, 35], [20, 32, 36], [20, 32, 37],
                [20, 32, 38], [20, 32, 39], [21, 32, 23], [21, 32, 24],
                [21, 32, 25], [21, 32, 26], [21, 32, 27], [21, 32, 28],
                [21, 32, 29], [21, 32, 30], [21, 32, 32], [21, 32, 33],
                [21, 32, 34], [21, 32, 35], [21, 32, 36], [21, 32, 37],
                [21, 32, 38], [21, 32, 39], [22, 32, 16], [22, 32, 23],
                [22, 32, 24], [22, 32, 25], [22, 32, 26], [22, 32, 27],
                [22, 32, 28], [22, 32, 29], [22, 32, 30], [22, 32, 31],
                [22, 32, 32], [22, 32, 33], [22, 32, 34], [22, 32, 35],
                [22, 32, 36], [22, 32, 37], [22, 32, 38], [22, 32, 39],
                [23, 32, 16], [23, 32, 23], [23, 32, 24], [23, 32, 25],
                [23, 32, 26], [23, 32, 27], [23, 32, 28], [23, 32, 31],
                [23, 32, 32], [23, 32, 34], [23, 32, 35], [23, 32, 36],
                [23, 32, 37], [23, 32, 38], [23, 32, 39], [24, 32, 15],
                [24, 32, 22], [24, 32, 23], [24, 32, 24], [24, 32, 25],
                [24, 32, 26], [24, 32, 27], [24, 32, 28], [24, 32, 29],
                [24, 32, 31], [24, 32, 32], [24, 32, 34], [24, 32, 35],
                [24, 32, 37], [24, 32, 38], [24, 32, 39], [25, 32, 15],
                [25, 32, 22], [25, 32, 23], [25, 32, 24], [25, 32, 25],
                [25, 32, 26], [25, 32, 28], [25, 32, 31], [25, 32, 32],
                [25, 32, 34], [25, 32, 35], [25, 32, 37], [25, 32, 38],
                [25, 32, 39], [26, 32, 15], [26, 32, 22], [26, 32, 23],
                [26, 32, 24], [26, 32, 25], [26, 32, 26], [26, 32, 31],
                [26, 32, 32], [26, 32, 35], [26, 32, 37], [26, 32, 38],
                [26, 32, 39], [27, 32, 15], [27, 32, 22], [27, 32, 23],
                [27, 32, 24], [27, 32, 25], [27, 32, 26], [27, 32, 28],
                [27, 32, 29], [27, 32, 31], [27, 32, 32], [27, 32, 35],
                [27, 32, 37], [27, 32, 38], [27, 32, 39], [28, 32, 22],
                [28, 32, 23], [28, 32, 24], [28, 32, 25], [28, 32, 26],
                [28, 32, 31], [28, 32, 32], [28, 32, 37], [28, 32, 38],
                [28, 32, 39], [29, 32, 22], [29, 32, 23], [29, 32, 24],
                [29, 32, 25], [29, 32, 26], [29, 32, 31], [29, 32, 32],
                [29, 32, 38], [29, 32, 39], [30, 32, 23], [30, 32, 24],
                [30, 32, 25], [30, 32, 31], [30, 32, 32], [30, 32, 38],
                [30, 32, 39], [31, 32, 23], [31, 32, 24], [31, 32, 25],
                [31, 32, 31], [31, 32, 32], [31, 32, 39], [32, 32, 23],
                [32, 32, 24], [32, 32, 25], [32, 32, 32], [32, 32, 39],
                [33, 32, 23], [33, 32, 24], [33, 32, 25], [33, 32, 32],
                [33, 32, 39], [34, 32, 23], [34, 32, 24], [34, 32, 25],
                [34, 32, 32], [34, 32, 38], [34, 32, 39], [35, 32, 15],
                [35, 32, 24], [35, 32, 25], [35, 32, 38], [35, 32, 39],
                [36, 32, 15], [36, 32, 24], [36, 32, 25], [36, 32, 38],
                [36, 32, 39], [37, 32, 15], [37, 32, 24], [37, 32, 25],
                [37, 32, 38], [38, 32, 25], [38, 32, 38], [39, 32, 16],
                [39, 32, 25], [39, 32, 26], [39, 32, 37], [40, 32, 16],
                [40, 32, 26], [40, 32, 37], [42, 32, 17], [42, 32, 27],
                [42, 32, 36], [43, 32, 18], [43, 32, 28], [43, 32, 35])

        nrmsd = alternatives.fd_vs_as_gradient(
                ixyz[np.random.randint(len(ixyz))],
                model_init, grad, data_misfit, sim_inp)

        assert nrmsd < 0.3
