import pytest
import shelve
import numpy as np
from scipy import constants
from os.path import join, dirname
from numpy.testing import assert_allclose

import emg3d
from emg3d import fields

from . import alternatives, helpers


# Import soft dependencies.
try:
    import discretize
    # Backwards compatibility; remove latest for version 1.0.0.
    dv = discretize.__version__.split('.')
    if int(dv[0]) == 0 and int(dv[1]) < 6:
        discretize = None
except ImportError:
    discretize = None

# Data generated with tests/create_data/regression.py
REGRES = emg3d.load(join(dirname(__file__), 'data', 'regression.npz'))


class TestGetReceiver:
    def test_runs_warnings(self):
        # The interpolation happens in maps.interp_spline_3d.
        # Here we just check the 'other' things: warning and errors, and that
        # it is composed correctly.

        # Check cubic spline runs fine (NOT CHECKING ACTUAL VALUES!.
        grid = emg3d.TensorMesh(
                [np.ones(4), np.array([1, 2, 3, 1]), np.array([2, 1, 1, 1])],
                [0, 0, 0])
        field = fields.Field(grid)
        field.field = np.ones(field.field.size) + 1j*np.ones(field.field.size)

        grid = emg3d.TensorMesh(
                [np.ones(6), np.array([1, 1, 2, 3, 1]),
                 np.array([1, 2, 1, 1, 1])], [-1, -1, -1])
        efield = fields.Field(grid, frequency=1)
        n = efield.field.size
        efield.field = np.ones(n) + 1j*np.ones(n)

        # Provide wrong rec_loc input:
        with pytest.raises(ValueError, match='`receiver` needs to be in the'):
            fields.get_receiver(efield, (1, 1, 1))

    def test_basics(self):

        # Coarse check with emg3d.solve and empymod.
        x = np.array([400, 450, 500, 550])
        rec = (x, x*0, 0, 20, 70)
        res = 0.3
        src = (0, 0, 0, 0, 0)
        freq = 10

        grid = emg3d.construct_mesh(
                frequency=freq,
                center=(0, 0, 0),
                properties=res,
                domain=[[0, 1000], [-25, 25], [-25, 25]],
                min_width_limits=20,
        )

        model = emg3d.Model(grid, res)
        sfield = fields.get_source_field(grid, src, freq)
        efield = emg3d.solve(model, sfield, semicoarsening=True,
                             sslsolver=True, linerelaxation=True, verb=1)

        # epm = empymod.bipole(src, rec, [], res, freq, verb=1)
        epm = np.array([-1.27832028e-11+1.21383502e-11j,
                        -1.90064149e-12+7.51937145e-12j,
                        1.09602131e-12+3.33066197e-12j,
                        1.25359248e-12+1.02630145e-12j])
        e3d = fields.get_receiver(efield, rec)

        # 10 % is still OK, grid is very coarse for fast comp (2s)
        assert_allclose(epm, e3d, rtol=0.1)

        # Test with a list of receiver instance.
        rec_inst = [emg3d.RxElectricPoint((rec[0][i], rec[1][i], *rec[2:]))
                    for i in range(rec[0].size)]
        e3d_inst = fields.get_receiver(efield, rec_inst)
        assert_allclose(e3d_inst, e3d)

        # Only one receiver
        e3d_inst = fields.get_receiver(efield, rec_inst[0])
        assert_allclose(e3d_inst, e3d[0])

        # Ensure responses outside and in the last cell are set to NaN.
        h = np.ones(4)*100
        grid = emg3d.TensorMesh([h, h, h], (-200, -200, -200))
        model = emg3d.Model(grid)
        sfield = fields.get_source_field(
                grid=grid, source=[0, 0, 0, 0, 0], frequency=10)
        out = emg3d.solve(model=model, sfield=sfield, plain=True, verb=0)
        off = np.arange(11)*50-250
        resp = out.get_receiver((off, off, off, 0, 0))
        assert_allclose(np.isfinite(resp),
                        np.r_[3*[False, ], 5*[True, ], 3*[False, ]])


def test_get_magnetic_field():
    # Check it does still the same (pure regression).
    dat = REGRES['reg_2']
    model = dat['model']
    efield = dat['result']
    hfield = dat['hresult']

    hout = fields.get_magnetic_field(model, efield)
    assert_allclose(hfield.field, hout.field)

    # Add some mu_r - Just 1, to trigger, and compare.
    dat = REGRES['res']
    efield = dat['Fresult']
    model1 = emg3d.Model(**dat['input_model'])
    model2 = emg3d.Model(**dat['input_model'], mu_r=1.)

    hout1 = fields.get_magnetic_field(model1, efield)
    hout2 = fields.get_magnetic_field(model2, efield)
    assert_allclose(hout1.field, hout2.field)

    # Test division by mu_r.
    model3 = emg3d.Model(**dat['input_model'], mu_r=2.)
    hout3 = fields.get_magnetic_field(model3, efield)
    assert_allclose(hout1.field, hout3.field*2)

    # Comparison to alternative.
    # Using very unrealistic value, unrealistic stretching, to test.
    grid = emg3d.TensorMesh(
        h=[[1, 100, 25, 33], [1, 1, 33.3, 0.3, 1, 1], [2, 4, 8, 16]],
        origin=(88, 20, 9))
    model = emg3d.Model(grid, mu_r=np.arange(1, grid.n_cells+1)/10)
    new = 10**np.arange(grid.n_edges)-grid.n_edges/2
    efield = fields.Field(grid, data=new, frequency=np.pi)
    hfield_nb = fields.get_magnetic_field(model, efield)
    hfield_np = alternatives.alt_get_magnetic_field(model, efield)
    assert_allclose(hfield_nb.field, hfield_np.field)

    # Test using discretize
    if discretize:
        h = np.ones(4)
        grid = emg3d.TensorMesh([h*200, h*300, h*400], (0, 0, 0))
        model = emg3d.Model(grid, property_x=3.24)
        sfield = fields.get_source_field(
                grid, (350, 550, 750, 30, 30), frequency=10)
        efield = emg3d.solve(model, sfield, plain=True, verb=0)
        mfield = fields.get_magnetic_field(model, efield)
        dfield = grid.edge_curl*efield.field/sfield.smu0
        assert_allclose(
                dfield[:80].reshape((5, 4, 4), order='F')[1:-1, :, :],
                mfield.fx)
        assert_allclose(
                dfield[80:160].reshape((4, 5, 4), order='F')[:, 1:-1, :],
                mfield.fy)
        assert_allclose(
                dfield[160:].reshape((4, 4, 5), order='F')[:, :, 1:-1],
                mfield.fz)


class TestDipoleVector:
    def test_basics_xdir_on_x(self):
        h = [2, 1, 1, 2]
        grid = emg3d.TensorMesh([h, h, h], (-3, -3, -3))

        # x-directed source in the middle
        source = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
        vfield = fields._dipole_vector(grid, source)

        # x: exact in the middle on the two edges
        assert_allclose(vfield.fx[1:-1, 2:-2, 2:-2].ravel(), [0.5, 0.5])
        # y: as exact "on" x-grid, falls to the right
        assert_allclose(vfield.fy[1:-1, 2:-1, 2:-2].ravel(), 0)
        # z: as exact "on" x-grid, falls to top
        assert_allclose(vfield.fz[1:-1, 2:-2, 2:-1].ravel(), 0)

    def test_basics_diag(self):
        h = [2, 1, 1, 2]
        grid = emg3d.TensorMesh([h, h, h], (-3, -3, -3))

        # Diagonal source in the middle
        source = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        vfield = fields._dipole_vector(grid, source)

        # x: exact in the middle on the two edges
        assert_allclose(vfield.fx[1:-2, 1:-2, 1:-2].ravel(),
                        [0.03125, 0.09375, 0.09375, 0.28125])
        # compare lower-left-front with upper-right-back
        assert_allclose(vfield.fx[1:-2, 1:-2, 1:-2].ravel(),
                        vfield.fx[2:-1, 2:-1, 2:-1].ravel()[::-1])

        # Source is 3D symmetric, compare all fields are the same
        assert_allclose(vfield.fx[1:-2, 1:-2, 1:-2].ravel(),
                        vfield.fy[1:-2, 1:-2, 1:-2].ravel())

        assert_allclose(vfield.fx[1:-2, 1:-2, 1:-2].ravel(),
                        vfield.fz[1:-2, 1:-2, 1:-2].ravel())

        assert_allclose(vfield.fx[2:-1, 2:-1, 2:-1].ravel(),
                        vfield.fy[2:-1, 2:-1, 2:-1].ravel())

        assert_allclose(vfield.fx[2:-1, 2:-1, 2:-1].ravel(),
                        vfield.fz[2:-1, 2:-1, 2:-1].ravel())

    def test_basics_diag_large(self):
        h = [2, 1, 1, 2]
        grid = emg3d.TensorMesh([h, h, h], (-3, -3, -3))

        # Large diagonal source in the middle
        source = np.array([[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]])
        vfield = fields._dipole_vector(grid, source)

        # Source is 3D symmetric, compare all fields are the same
        assert_allclose(vfield.fx[0, :2, :2].ravel(),
                        vfield.fy[:2, 0, :2].ravel())
        assert_allclose(vfield.fx[0, :2, :2].ravel(),
                        vfield.fz[:2, :2, 0].ravel())

        assert_allclose(vfield.fx[1, 1:3, 1:3].ravel(),
                        vfield.fy[1:3, 1, 1:3].ravel())
        assert_allclose(vfield.fx[1, 1:3, 1:3].ravel(),
                        vfield.fz[1:3, 1:3, 1].ravel())

        assert_allclose(vfield.fx[0, :2, :2].ravel(),
                        vfield.fx[3, 3:, 3:].ravel()[::-1])

        assert_allclose(vfield.fx[1, 1:3, 1:3].ravel(),
                        vfield.fx[2, 2:4, 2:4].ravel()[::-1])

    def test_decimals(self):
        h1 = [2, 1, 1, 2]
        h2 = [2, 1.04, 1.04, 2]
        grid1 = emg3d.TensorMesh([h1, h1, h1], (-3, -3, -3))
        grid2 = emg3d.TensorMesh([h2, h2, h2], (-3, -3, -3))
        source = np.array([[-0.5, 0, 0], [0.5, 0, 0]])

        sfield1 = fields._dipole_vector(grid1, source)
        sfield2a = fields._dipole_vector(grid2, source, decimals=1)
        sfield2b = fields._dipole_vector(grid2, source)

        assert_allclose(sfield1.fx, sfield2a.fx)
        with pytest.raises(AssertionError, match='Not equal to tolerance'):
            assert_allclose(sfield1.fx, sfield2b.fx)

    def test_warnings(self):
        h = np.ones(4)
        grid = emg3d.TensorMesh([h, h, h], (0, 0, 0))
        source = np.array([[5, 2, 2], [2, 2, 2]])
        with pytest.raises(ValueError, match='Provided source outside grid'):
            fields._dipole_vector(grid, source)

        source = np.array([[2, 2, 2], [2, 2, 2]])
        with pytest.raises(ValueError, match='Provided finite dipole'):
            fields._dipole_vector(grid, source)

        # This is a warning that should never be raised...
        hx, x0 = np.ones(4), -2
        grid = emg3d.TensorMesh([hx, hx, hx], (x0, x0, x0))
        source = np.array([[-2, 2, 0], [0, -2, 0]])
        with pytest.warns(UserWarning, match="Normalizing Source: 1.25000000"):
            fields._dipole_vector(grid, source, 30)


@pytest.mark.parametrize("njit", [True, False])
def test_edge_curl_factor(njit):
    if njit:
        edge_curl_factor = fields._edge_curl_factor
    else:
        edge_curl_factor = fields._edge_curl_factor.py_func

    ex = 1*np.arange(1, 19).reshape((2, 3, 3), order='F')
    ey = 2*np.arange(1, 19).reshape((3, 2, 3), order='F')
    ez = 3*np.arange(1, 19).reshape((3, 3, 2), order='F')
    mx = np.zeros((1, 2, 2))
    my = np.zeros((2, 1, 2))
    mz = np.zeros((2, 2, 1))
    hx = np.array([10., 10])
    hy = np.array([20., 20])
    hz = np.array([30., 30])
    factor = 0.1*np.ones((2, 2, 2))

    edge_curl_factor(mx, my, mz, ex, ey, ez, hx, hy, hz, factor)
    assert_allclose(mx, 0.5)  # (9/20 - 12/30) * 2 / 0.2
    assert_allclose(my, -1)  # (6/30 - 3/10) / 0.2 * 2
    assert_allclose(mz, 1)  # (2/10 - 2/20) / 0.2 * 2
