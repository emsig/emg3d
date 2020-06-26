import pytest
import numpy as np
from numpy.testing import assert_allclose

from emg3d import meshes, models, surveys, simulations, fields, solver


class TestSimulation():
    # Create a simple survey
    sources = (0, [1000, 3000, 5000], -950, 0, 0)
    receivers = (np.arange(12)*500, 0, -1000, 0, 0)
    frequencies = (1.0, 2.0)
    survey = surveys.Survey('Test', sources, receivers, frequencies)

    # Create a simple grid and model
    grid = meshes.TensorMesh(
            [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
            np.array([-1250, -1250, -2250]))
    model = models.Model(grid, 1)

    def test_simple_simulation(self):
        simulation = simulations.Simulation(
                self.survey, self.grid, self.model, comp_grids='same')

        # Check model
        assert simulation.comp_models('Tx1', 1.0) == self.model

        # Check grid
        assert simulation.comp_grids('Tx1', 1.0) == self.grid

        # Check sfield
        sfield = fields.get_source_field(
                self.grid, self.survey.sources['Tx1'].coordinates,
                freq=1.0, strength=0)
        assert_allclose(simulation.sfields('Tx1', 1.0), sfield)

        # Check efield
        efield, info = solver.solve(
                self.grid, self.model, sfield, sslsolver=True,
                semicoarsening=True, linerelaxation=True, verb=-1,
                return_info=True
                )
        assert_allclose(simulation.efields('Tx1', 1.0), efield)

        # Check efield with return_info and recomp
        s_efield, s_info = simulation.efields(
                'Tx1', 1.0, recomp=True, return_info=True)
        assert_allclose(s_efield, efield)
        assert_allclose(s_info['abs_error'], info['abs_error'])
        assert_allclose(s_info['rel_error'], info['rel_error'])
        assert s_info['exit'] == info['exit'] == 0

        # Check hfield
        hfield = fields.get_h_field(self.grid, self.model, efield)
        assert_allclose(simulation.hfields('Tx1', 1.0), hfield)
        s_hfield, s_info = simulation.hfields(
                'Tx1', 1.0, recomp=True, return_info=True)
        assert_allclose(s_hfield, hfield)
        assert_allclose(s_info['abs_error'], info['abs_error'])
        assert_allclose(s_info['rel_error'], info['rel_error'])
        assert s_info['exit'] == info['exit'] == 0

        # psolve and check
        simulation.psolve(verb=-1)
        rec_resp = fields.get_receiver_response(
                self.grid, efield, self.survey.rec_coords)
        assert_allclose(
                simulation.synthetic[1, :, 0].data, rec_resp, atol=1e-16)

        # Clean and ensure it is empty
        simulation.clean()
        assert simulation._efields['Tx1'][1.0] is None
        assert simulation._hfields['Tx1'][1.0] is None
        assert simulation._solver_info['Tx1'][1.0] is None

    def test_errors(self):
        with pytest.raises(TypeError):
            simulations.Simulation(
                    self.survey, self.grid, self.model, unknown=True)
