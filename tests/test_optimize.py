import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import optimize

from . import alternatives


# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None


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

    simulation = emg3d.Simulation(survey=survey, grid=grid, model=model)

    field = emg3d.Field(grid, dtype=np.float64)
    field.field += syn
    simulation._dict_efield['TxED-1']['f-1'] = field
    simulation.data['synthetic'] = simulation.data['observed']*0 + syn

    misfit = 0.5*((syn-data)/(rel_err*data))**2
    assert_allclose(optimize.misfit(simulation), misfit)

    # Missing noise_floor / std.
    survey = emg3d.Survey(sources, receivers, 100)
    simulation = emg3d.Simulation(survey=survey, grid=grid, model=model)
    with pytest.raises(ValueError, match="Either `noise_floor` or"):
        optimize.misfit(simulation)


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
        sim_inp = {'survey': survey, 'grid': mesh, 'gridding': 'same'}

        # Anisotropic models.
        simulation = emg3d.Simulation(
                model=emg3d.Model(mesh, 1, 2, 3), **sim_inp)
        with pytest.raises(NotImplementedError, match='for isotropic models'):
            optimize.gradient(simulation)

        # Model with electric permittivity.
        simulation = emg3d.Simulation(
                model=emg3d.Model(mesh, epsilon_r=3), **sim_inp)
        with pytest.raises(NotImplementedError, match='for el. permittivity'):
            optimize.gradient(simulation)

        # Model with magnetic permeability.
        simulation = emg3d.Simulation(
                model=emg3d.Model(mesh, mu_r=np.ones(mesh.shape_cells)*np.pi),
                **sim_inp)
        with pytest.raises(NotImplementedError, match='for magn. permeabili'):
            optimize.gradient(simulation)

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
            'grid': mesh,
            'solver_opts': {'plain': True, 'tol': 5e-5},  # Red. tol 4 speed
            'max_workers': 1,
            'gridding': 'same',
            'verb': 0,
        }

        # Compute data (pre-computed and passed to Survey above)
        sim_data = emg3d.Simulation(model=model_true, **sim_inp)
        sim_data.compute(observed=True)

        # Compute adjoint state misfit and gradient
        sim_data = emg3d.Simulation(model=model_init, **sim_inp)
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
