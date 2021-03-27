import pytest
import shelve
import numpy as np
from scipy import constants
from os.path import join, dirname
from numpy.testing import assert_allclose, assert_array_equal

from emg3d import electrodes, io, meshes, models, fields, solver

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
REGRES = io.load(join(dirname(__file__), 'data/regression.npz'))


class TestField:
    grid = meshes.TensorMesh([[.5, 8], [1, 4], [2, 8]], (0, 0, 0))

    ex = helpers.dummy_field(*grid.shape_edges_x, imag=True)
    ey = helpers.dummy_field(*grid.shape_edges_y, imag=True)
    ez = helpers.dummy_field(*grid.shape_edges_z, imag=True)

    # Test the views
    field = np.r_[ex.ravel('F'), ey.ravel('F'), ez.ravel('F')]

    def test_basic(self):
        ee = fields.Field(self.grid, self.field)
        assert_allclose(ee.field, self.field)
        assert_allclose(ee.fx, self.ex)
        assert_allclose(ee.fy, self.ey)
        assert_allclose(ee.fz, self.ez)
        assert ee.smu0 is None
        assert ee.sval is None
        assert ee.frequency is None
        assert ee.field.dtype == self.field.dtype

        # Check representation of Field.
        assert f"Field: {ee.grid.shape_cells[0]} x" in ee.__repr__()

        # Test amplitude and phase.

        assert_allclose(ee.fx.amp(), np.abs(ee.fx))
        assert_allclose(ee.fy.pha(unwrap=False), np.angle(ee.fy))

        # Test the other possibilities to initiate a Field-instance.
        frequency = 1.0
        ee3 = fields.Field(self.grid, frequency=frequency,
                           dtype=self.field.dtype)
        assert ee.field.size == ee3.field.size
        assert ee.field.dtype == np.complex128
        assert ee3.frequency == frequency

        # Try setting values
        ee3.field = ee.field
        assert ee3.smu0/ee3.sval == constants.mu_0
        assert ee != ee3  # First has no frequency
        ee3.fx = ee.fx
        ee3.fy = ee.fy
        ee3.fz = ee.fz

        # Negative
        ee4 = fields.Field(self.grid, frequency=-frequency)
        assert ee.field.size == ee4.field.size
        assert ee4.field.dtype == np.float64
        assert ee4.frequency == frequency
        assert ee4._frequency == -frequency
        assert ee4.smu0/ee4.sval == constants.mu_0

    def test_dtype(self):
        with pytest.raises(ValueError, match="must be f>0"):
            _ = fields.Field(self.grid, frequency=0.0)

        with pytest.warns(np.ComplexWarning, match="Casting complex values"):
            lp = fields.Field(self.grid, self.field, frequency=-1)
        assert lp.field.dtype == np.float64

        ignore = fields.Field(self.grid, frequency=-1, dtype=np.int64)
        assert ignore.field.dtype == np.float64

        ignore = fields.Field(self.grid, self.field, dtype=np.int64)
        assert ignore.field.dtype == np.complex128

        respected = fields.Field(self.grid, dtype=np.int64)
        assert respected.field.dtype == np.int64

        default = fields.Field(self.grid)
        assert default.field.dtype == np.complex128

    def test_copy_dict(self, tmpdir):
        ee = fields.Field(self.grid, self.field)
        # Test copy
        e2 = ee.copy()
        assert ee == e2
        assert_allclose(ee.fx, e2.fx)
        assert_allclose(ee.fy, e2.fy)
        assert_allclose(ee.fz, e2.fz)
        assert not np.may_share_memory(ee.field, e2.field)

        edict = ee.to_dict()
        del edict['grid']
        with pytest.raises(KeyError, match="'grid'"):
            fields.Field.from_dict(edict)

        # Ensure it can be pickled.
        with shelve.open(tmpdir+'/test') as db:
            db['field'] = ee
        with shelve.open(tmpdir+'/test') as db:
            test = db['field']
        assert test == ee

    def test_interpolate_to_grid(self):
        # We only check here that it gives the same as calling the function
        # itself; the rest should be tested in interpolate().
        grid1 = meshes.TensorMesh(
                [np.ones(8), np.ones(8), np.ones(8)], (0, 0, 0))
        grid2 = meshes.TensorMesh([[2, 2, 2, 2], [3, 3], [4, 4]], (0, 0, 0))
        ee = fields.Field(grid1)
        ee.field = np.ones(ee.field.size) + 2j*np.ones(ee.field.size)
        e2 = ee.interpolate_to_grid(grid2)
        assert_allclose(e2.field, 1+2j)

    def test_get_receiver(self):
        # We only check here that it gives the same as calling the function
        # itself; the rest should be tested in get_receiver().
        grid1 = meshes.TensorMesh(
                [np.ones(8), np.ones(8), np.ones(8)], (0, 0, 0))
        ee = fields.Field(grid1)
        ee.field = np.arange(ee.field.size) + 2j*np.arange(ee.field.size)
        resp = ee.get_receiver((4, 4, 4, 0, 0))
        print(80*'=')
        print(resp)
        print(80*'=')
        assert_allclose(resp, 323.5 + 647.0j)


class TestGetSourceField:
    def test_get_source_field(self, capsys):
        src = [100, 200, 300, 27, 31]
        h = np.ones(4)
        grid = meshes.TensorMesh([h*200, h*400, h*800], (-450, -850, -1650))
        freq = 1.2458

        sfield = fields.get_source_field(grid, src, freq, strength=1+1j)
        assert_array_equal(sfield.strength, complex(1+1j))

        sfield = fields.get_source_field(grid, src, freq, strength=0)
        assert_array_equal(sfield.strength, float(0))
        iomegamu = 2j*np.pi*freq*constants.mu_0

        # Check number of edges
        assert 4 == sfield.fx[sfield.fx != 0].size
        assert 4 == sfield.fy[sfield.fy != 0].size
        assert 4 == sfield.fz[sfield.fz != 0].size

        # Check source cells
        h = np.cos(np.deg2rad(src[4]))
        y = np.sin(np.deg2rad(src[3]))*h
        x = np.cos(np.deg2rad(src[3]))*h
        z = np.sin(np.deg2rad(src[4]))
# TODO  assert_allclose(np.sum(sfield.fx/x/iomegamu).real, -1)
# TODO  assert_allclose(np.sum(sfield.fy/y/iomegamu).real, -1)
# TODO  assert_allclose(np.sum(sfield.fz/z/iomegamu).real, -1)
        assert sfield._frequency == freq
        assert sfield.frequency == freq
        assert_allclose(sfield.smu0, -iomegamu)

        # Put source on final node, should still work.
        src = [grid.nodes_x[0], grid.nodes_x[0]+1,
               grid.nodes_y[-1]-1, grid.nodes_y[-1],
               grid.nodes_z[0], grid.nodes_z[0]+1]
        sfield = fields.get_source_field(grid, src, freq)
        tot_field = np.linalg.norm(
                [np.sum(sfield.fx), np.sum(sfield.fy), np.sum(sfield.fz)])
# TODO  assert_allclose(tot_field/np.abs(np.sum(iomegamu)), 1.0)

        out, _ = capsys.readouterr()  # Empty capsys

        # Provide wrong source definition. Ensure it fails.
        with pytest.raises(ValueError, match='`coordinates` must be of shap'):
            sfield = fields.get_source_field(grid, [0, 0, 0, 0], 1)

        # Put finite dipole of zero length. Ensure it fails.
        with pytest.raises(ValueError, match='The two poles are identical'):
            src = [0, 0, 100, 100, -200, -200]
            sfield = fields.get_source_field(grid, src, 1)

        # Same for Laplace domain
        src = [100, 200, 300, 27, 31]
        h = np.ones(4)
        grid = meshes.TensorMesh([h*200, h*400, h*800], (-450, -850, -1650))
        freq = 1.2458
        sfield = fields.get_source_field(grid, src, -freq)
        smu = freq*constants.mu_0

        # Check number of edges
        assert 4 == sfield.fx[sfield.fx != 0].size
        assert 4 == sfield.fy[sfield.fy != 0].size
        assert 4 == sfield.fz[sfield.fz != 0].size

        # Check source cells
        h = np.cos(np.deg2rad(src[4]))
        y = np.sin(np.deg2rad(src[3]))*h
        x = np.cos(np.deg2rad(src[3]))*h
        z = np.sin(np.deg2rad(src[4]))
# TODO  assert_allclose(np.sum(sfield.fx/x/smu), -1)
# TODO  assert_allclose(np.sum(sfield.fy/y/smu), -1)
# TODO  assert_allclose(np.sum(sfield.fz/z/smu), -1)
        assert sfield._frequency == -freq
        assert sfield.frequency == freq
        assert_allclose(sfield.smu0, -freq*constants.mu_0)

    def test_arbitrarily_shaped_source(self):
        h = np.ones(4)
        grid = meshes.TensorMesh([h*200, h*400, h*800], [-400, -800, -1600])
        freq = 1.11
        strength = np.pi
        src = (0, 0, 0, 0, 90)

        # Manually
        sman = fields.Field(grid, frequency=freq)
        src4xxyyzz = [
            np.r_[src[0]-0.5, src[0]+0.5, src[1]-0.5,
                  src[1]-0.5, src[2], src[2]],
            np.r_[src[0]+0.5, src[0]+0.5, src[1]-0.5,
                  src[1]+0.5, src[2], src[2]],
            np.r_[src[0]+0.5, src[0]-0.5, src[1]+0.5,
                  src[1]+0.5, src[2], src[2]],
            np.r_[src[0]-0.5, src[0]-0.5, src[1]+0.5,
                  src[1]-0.5, src[2], src[2]],
        ]
        for srcl in src4xxyyzz:
            sman.field += fields.get_source_field(
                    grid, srcl, freq, strength=strength).field

        # Computed
        src5xyz = (
            [src[0]-0.5, src[0]+0.5, src[0]+0.5, src[0]-0.5, src[0]-0.5],
            [src[1]-0.5, src[1]-0.5, src[1]+0.5, src[1]+0.5, src[1]-0.5],
            [src[2], src[2], src[2], src[2], src[2]]
        )
        scomp = fields.get_source_field(grid, src5xyz, freq, strength=strength)
        assert sman == scomp

        # Normalized
        sman = fields.Field(grid, frequency=freq)
        for srcl in src4xxyyzz:
            sman.field += fields.get_source_field(grid, srcl, freq, 0.25).field
        scomp = fields.get_source_field(grid, src5xyz, freq)
        assert sman == scomp

    def test_source_field(self):
        # Create some dummy data
        grid = meshes.TensorMesh(
                [np.array([.5, 8]), np.array([1, 4]), np.array([2, 8])],
                np.zeros(3))

        freq = np.pi
        ss = fields.Field(grid, frequency=freq)
        assert_allclose(ss.smu0, -2j*np.pi*freq*constants.mu_0)

        # Check 0 Hz frequency.
        with pytest.raises(ValueError, match='`frequency` must be f>0'):
            ss = fields.Field(grid, frequency=0)

        sdict = ss.to_dict()
        del sdict['grid']
        with pytest.raises(KeyError, match="'grid'"):
            fields.Field.from_dict(sdict)


def test_get_receiver():
    # The interpolation happens in maps.interp_spline_3d.
    # Here we just check the 'other' things: warning and errors, and that
    # it is composed correctly.

    # Check cubic spline runs fine (NOT CHECKING ACTUAL VALUES!.
    grid = meshes.TensorMesh(
            [np.ones(4), np.array([1, 2, 3, 1]), np.array([2, 1, 1, 1])],
            [0, 0, 0])
    field = fields.Field(grid)
    field.field = np.ones(field.field.size) + 1j*np.ones(field.field.size)

    grid = meshes.TensorMesh(
            [np.ones(6), np.array([1, 1, 2, 3, 1]), np.array([1, 2, 1, 1, 1])],
            [-1, -1, -1])
    efield = fields.Field(grid, frequency=1)
    efield.field = np.ones(efield.field.size) + 1j*np.ones(efield.field.size)

    # Provide wrong rec_loc input:
    with pytest.raises(ValueError, match='`receiver` needs to be in the form'):
        fields.get_receiver(efield, (1, 1, 1))

    # Provide particular field instead of field instance:
    with pytest.raises(ValueError, match='`field` must be a `Field`-inst'):
        fields.get_receiver(efield.fx, (1, 1, 1, 0, 0))

    # Coarse check with emg3d.solve and empymod.
    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = meshes.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )

    model = models.Model(grid, res)
    sfield = fields.get_source_field(grid, src, freq)
    efield = solver.solve(model, sfield, semicoarsening=True,
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
    rec_inst = [electrodes.RxElectricPoint((rec[0][i], rec[1][i], *rec[2:]))
                for i in range(rec[0].size)]
    e3d_inst = fields.get_receiver(efield, rec_inst)
    assert_allclose(e3d_inst, e3d)

    # Only one receiver
    e3d_inst = fields.get_receiver(efield, rec_inst[0])
    assert_allclose(e3d_inst, e3d[0])

    # Ensure responses outside and in the last cell are set to NaN.
    h = np.ones(4)*100
    grid = meshes.TensorMesh([h, h, h], (-200, -200, -200))
    model = models.Model(grid)
    sfield = fields.get_source_field(
            grid=grid, source=[0, 0, 0, 0, 0], frequency=10)
    out = solver.solve(model=model, sfield=sfield, plain=True, verb=0)
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
    model1 = models.Model(**dat['input_model'])
    model2 = models.Model(**dat['input_model'], mu_r=1.)

    hout1 = fields.get_magnetic_field(model1, efield)
    hout2 = fields.get_magnetic_field(model2, efield)
    assert_allclose(hout1.field, hout2.field)

    # Test division by mu_r.
    model3 = models.Model(**dat['input_model'], mu_r=2.)
    hout3 = fields.get_magnetic_field(model3, efield)
    assert_allclose(hout1.field, hout3.field*2)

    # Comparison to alternative.
    # Using very unrealistic value, unrealistic stretching, to test.
    grid = meshes.TensorMesh(
        h=[[1, 100, 25, 33], [1, 1, 33.3, 0.3, 1, 1], [2, 4, 8, 16]],
        origin=(88, 20, 9))
    model = models.Model(grid, mu_r=np.arange(1, grid.n_cells+1)/10)
    new = 10**np.arange(grid.n_edges)-grid.n_edges/2
    efield = fields.Field(grid, data=new, frequency=np.pi)
    hfield_nb = fields.get_magnetic_field(model, efield)
    hfield_np = alternatives.alt_get_magnetic_field(model, efield)
    assert_allclose(hfield_nb.field, hfield_np.field)

    # Test using discretize
    if discretize:
        h = np.ones(4)
        grid = meshes.TensorMesh([h*200, h*300, h*400], (0, 0, 0))
        model = models.Model(grid, property_x=3.24)
        sfield = fields.get_source_field(
                grid, (350, 550, 750, 30, 30), frequency=10)
        efield = solver.solve(model, sfield, plain=True, verb=0)
        mfield = fields.get_magnetic_field(model, efield)
        dfield = -grid.edge_curl*efield.field/sfield.smu0
        assert_allclose(
                dfield[:80].reshape((5, 4, 4), order='F')[1:-1, :, :],
                mfield.fx)
        assert_allclose(
                dfield[80:160].reshape((4, 5, 4), order='F')[:, 1:-1, :],
                mfield.fy)
        assert_allclose(
                dfield[160:].reshape((4, 4, 5), order='F')[:, :, 1:-1],
                mfield.fz)


class TestFiniteSourceXYZ:

    def test_get_source_field_point_vs_finite(self, capsys):
        # === Point dipole to finite dipole comparisons ===
        def get_xyz(d_src):
            """Return dimensions corresponding to azimuth and dip."""
            h = np.cos(np.deg2rad(d_src[4]))
            dys = np.sin(np.deg2rad(d_src[3]))*h
            dxs = np.cos(np.deg2rad(d_src[3]))*h
            dzs = np.sin(np.deg2rad(d_src[4]))
            return [dxs, dys, dzs]

        def get_f_src(d_src, slen=1.0):
            """Return d_src and f_src for d_src input."""
            xyz = get_xyz(d_src)
            f_src = [d_src[0]-xyz[0]*slen/2, d_src[0]+xyz[0]*slen/2,
                     d_src[1]-xyz[1]*slen/2, d_src[1]+xyz[1]*slen/2,
                     d_src[2]-xyz[2]*slen/2, d_src[2]+xyz[2]*slen/2]
            return d_src, f_src

        # 1a. Source within one cell, normalized.
        h = np.ones(3)*500
        grid1 = meshes.TensorMesh([h, h, h], np.array([-750, -750, -750]))
        d_src, f_src = get_f_src([0, 0., 0., 23, 15])
        dsf = fields.get_source_field(grid1, d_src, 1)
        fsf = fields.get_source_field(grid1, f_src, 1)
        assert fsf == dsf

        # 1b. Source within one cell, source strength = pi.
        d_src, f_src = get_f_src([0, 0., 0., 32, 53])
        dsf = fields.get_source_field(grid1, d_src, 3.3, strength=np.pi)
        fsf = fields.get_source_field(grid1, f_src, 3.3, strength=np.pi)
        assert fsf == dsf

        # 1c. Source over various cells, normalized.
        h = np.ones(8)*200
        grid2 = meshes.TensorMesh([h, h, h], np.array([-800, -800, -800]))
        d_src, f_src = get_f_src([0, 0., 0., 40, 20], 300.0)
        dsf = fields.get_source_field(grid2, d_src, 10.0, strength=0)
        fsf = fields.get_source_field(grid2, f_src, 10.0, strength=0)
# TODO  assert_allclose(fsf.fx.sum(), dsf.fx.sum())
# TODO  assert_allclose(fsf.fy.sum(), dsf.fy.sum())
# TODO  assert_allclose(fsf.fz.sum(), dsf.fz.sum())

        # 1d. Source over various cells, source strength = pi.
        slen = 300
        strength = np.pi
        d_src, f_src = get_f_src([0, 0., 0., 20, 30], slen)
        dsf = fields.get_source_field(
                grid2, d_src, 1.3, strength=slen*strength)
        fsf = fields.get_source_field(grid2, f_src, 1.3, strength=strength)
# TODO  assert_allclose(fsf.fx.sum(), dsf.fx.sum())
# TODO  assert_allclose(fsf.fy.sum(), dsf.fy.sum())
# TODO  assert_allclose(fsf.fz.sum(), dsf.fz.sum())

        # 1e. Source over various stretched cells, source strength = pi.
        h1 = helpers.get_h(4, 2, 200, 1.1)
        h2 = helpers.get_h(4, 2, 200, 1.2)
        h3 = helpers.get_h(4, 2, 200, 1.2)
        origin = np.array([-h1.sum()/2, -h2.sum()/2, -h3.sum()/2])
        grid3 = meshes.TensorMesh([h1, h2, h3], origin)
        slen = 333
        strength = np.pi
        d_src, f_src = get_f_src([0, 0., 0., 50, 33], slen)
        dsf = fields.get_source_field(
                grid3, d_src, 0.7, strength=slen*strength)
        fsf = fields.get_source_field(grid3, f_src, 0.7, strength=strength)
# TODO  assert_allclose(fsf.fx.sum(), dsf.fx.sum())
# TODO  assert_allclose(fsf.fy.sum(), dsf.fy.sum())
# TODO  assert_allclose(fsf.fz.sum(), dsf.fz.sum())

    def test_source_norm_warning(self):
        # This is a warning that should never be raised...
        hx, x0 = np.ones(4), -2
        mesh = meshes.TensorMesh([hx, hx, hx], (x0, x0, x0))
        sfield = fields.Field(mesh, frequency=1)
        sfield.fx += 1  # Add something to the field.
        with pytest.warns(UserWarning, match="Normalizing Source: 101.000000"):
            _ = fields._finite_source_xyz(
                    mesh, (-0.5, 0.5, 0, 0, 0, 0), sfield.fx, 30)

    def test_warnings(self):
        h = np.ones(4)
        grid = meshes.TensorMesh([h*200, h*400, h*800], (-450, -850, -1650))
        # Put source way out. Ensure it fails.
        with pytest.raises(ValueError, match='Provided source outside grid'):
            _ = fields.get_source_field(grid, [1e10, 1e10, 1e10, 0, 0], 1)


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
