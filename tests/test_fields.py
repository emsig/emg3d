import pytest
import shelve
import numpy as np
from scipy import constants
from os.path import join, dirname
from numpy.testing import assert_allclose, assert_array_equal

# Import soft dependencies.
try:
    import discretize
    # Backwards compatibility; remove latest for version 1.0.0.
    dv = discretize.__version__.split('.')
    if int(dv[0]) == 0 and int(dv[1]) < 6:
        discretize = None
except ImportError:
    discretize = None

from emg3d import io, meshes, models, fields, solver, utils

# Data generated with tests/create_data/regression.py
REGRES = io.load(join(dirname(__file__), 'data/regression.npz'))


def get_h(ncore, npad, width, factor):
    """Get cell widths for TensorMesh."""
    pad = ((np.ones(npad)*np.abs(factor))**(np.arange(npad)+1))*width
    return np.r_[pad[::-1], np.ones(ncore)*width, pad]


def create_dummy(nx, ny, nz, imag=True):
    """Return complex dummy arrays of shape nx*ny*nz.

    Numbers are from 1..nx*ny*nz for the real part, and 1/100 of it for the
    imaginary part.

    """
    if imag:
        out = np.arange(1, nx*ny*nz+1) + 1j*np.arange(1, nx*ny*nz+1)/100.
    else:
        out = np.arange(1, nx*ny*nz+1)
    return out.reshape(nx, ny, nz)


def test_get_source_field(capsys):
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
    assert_allclose(np.sum(sfield.fx/x/iomegamu).real, -1)
    assert_allclose(np.sum(sfield.fy/y/iomegamu).real, -1)
    assert_allclose(np.sum(sfield.fz/z/iomegamu).real, -1)
    assert_allclose(np.sum(sfield.vx/x), 1)
    assert_allclose(np.sum(sfield.vy/y), 1)
    assert_allclose(np.sum(sfield.vz/z), 1)
    assert sfield._freq == freq
    assert sfield.freq == freq
    assert_allclose(sfield.smu0, -iomegamu)

    # Put source on final node, should still work.
    src = [grid.nodes_x[0], grid.nodes_x[0]+1,
           grid.nodes_y[-1]-1, grid.nodes_y[-1],
           grid.nodes_z[0], grid.nodes_z[0]+1]
    sfield = fields.get_source_field(grid, src, freq)
    tot_field = np.linalg.norm(
            [np.sum(sfield.fx), np.sum(sfield.fy), np.sum(sfield.fz)])
    assert_allclose(tot_field/np.abs(np.sum(iomegamu)), 1.0)

    out, _ = capsys.readouterr()  # Empty capsys

    # Provide wrong source definition. Ensure it fails.
    with pytest.raises(ValueError, match='Source is wrong defined'):
        sfield = fields.get_source_field(grid, [0, 0, 0, 0], 1)

    # Put source way out. Ensure it fails.
    with pytest.raises(ValueError, match='Provided source outside grid'):
        src = [1e10, 1e10, 1e10, 0, 0]
        sfield = fields.get_source_field(grid, src, 1)

    # Put finite dipole of zero length. Ensure it fails.
    with pytest.raises(ValueError, match='Provided finite dipole has no leng'):
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
    assert_allclose(np.sum(sfield.fx/x/smu), -1)
    assert_allclose(np.sum(sfield.fy/y/smu), -1)
    assert_allclose(np.sum(sfield.fz/z/smu), -1)
    assert_allclose(np.sum(sfield.vx/x), 1)
    assert_allclose(np.sum(sfield.vy/y), 1)
    assert_allclose(np.sum(sfield.vz/z), 1)
    assert sfield._freq == -freq
    assert sfield.freq == freq
    assert_allclose(sfield.smu0, -freq*constants.mu_0)


def test_arbitrarily_shaped_source():
    h = np.ones(4)
    grid = meshes.TensorMesh([h*200, h*400, h*800], [-400, -800, -1600])
    freq = 1.11
    strength = np.pi
    src = (0, 0, 0, 0, 90)

    with pytest.raises(ValueError, match='All source coordinates must have'):
        fields.get_source_field(grid, ([1, 2], 1, 1), freq, strength)

    # Manually
    sman = fields.SourceField(grid, freq=freq)
    src4xxyyzz = [
        np.r_[src[0]-0.5, src[0]+0.5, src[1]-0.5, src[1]-0.5, src[2], src[2]],
        np.r_[src[0]+0.5, src[0]+0.5, src[1]-0.5, src[1]+0.5, src[2], src[2]],
        np.r_[src[0]+0.5, src[0]-0.5, src[1]+0.5, src[1]+0.5, src[2], src[2]],
        np.r_[src[0]-0.5, src[0]-0.5, src[1]+0.5, src[1]-0.5, src[2], src[2]],
    ]
    for srcl in src4xxyyzz:
        sman += fields.get_source_field(grid, srcl, freq, strength)

    # Computed
    src5xyz = ([src[0]-0.5, src[0]+0.5, src[0]+0.5, src[0]-0.5, src[0]-0.5],
               [src[1]-0.5, src[1]-0.5, src[1]+0.5, src[1]+0.5, src[1]-0.5],
               [src[2], src[2], src[2], src[2], src[2]])
    scomp = fields.get_source_field(grid, src5xyz, freq, strength)

    # Computed 2
    with pytest.warns(FutureWarning, match="``msrc`` is deprecated "):
        scomp2a = fields.get_source_field(grid, src, freq, strength, msrc=True)
    with pytest.raises(TypeError, match='Unexpected'):
        fields.get_source_field(grid, src, freq, strength, whatever=True)
    scomp2 = fields.get_source_field(grid, src, freq, strength, electric=False)

    assert_allclose(scomp2a, scomp2)
    assert_allclose(sman.field, scomp.field)
    assert_allclose(-scomp2.vector, scomp.vector, rtol=1e-6, atol=1e-15)

    # Normalized
    sman = fields.SourceField(grid, freq=freq)
    for srcl in src4xxyyzz:
        sman += fields.get_source_field(grid, srcl, freq, 0.25)
    scomp = fields.get_source_field(grid, src5xyz, freq)
    assert_allclose(sman.field, scomp.field)


def test_get_source_field_point_vs_finite(capsys):
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
    assert_allclose(fsf, dsf)

    # 1b. Source within one cell, source strength = pi.
    d_src, f_src = get_f_src([0, 0., 0., 32, 53])
    dsf = fields.get_source_field(grid1, d_src, 3.3, np.pi)
    fsf = fields.get_source_field(grid1, f_src, 3.3, np.pi)
    assert_allclose(fsf, dsf)

    # 1c. Source over various cells, normalized.
    h = np.ones(8)*200
    grid2 = meshes.TensorMesh([h, h, h], np.array([-800, -800, -800]))
    d_src, f_src = get_f_src([0, 0., 0., 40, 20], 300.0)
    dsf = fields.get_source_field(grid2, d_src, 10.0, 0)
    fsf = fields.get_source_field(grid2, f_src, 10.0, 0)
    assert_allclose(fsf.fx.sum(), dsf.fx.sum())
    assert_allclose(fsf.fy.sum(), dsf.fy.sum())
    assert_allclose(fsf.fz.sum(), dsf.fz.sum())

    # 1d. Source over various cells, source strength = pi.
    slen = 300
    strength = np.pi
    d_src, f_src = get_f_src([0, 0., 0., 20, 30], slen)
    dsf = fields.get_source_field(grid2, d_src, 1.3, slen*strength)
    fsf = fields.get_source_field(grid2, f_src, 1.3, strength)
    assert_allclose(fsf.fx.sum(), dsf.fx.sum())
    assert_allclose(fsf.fy.sum(), dsf.fy.sum())
    assert_allclose(fsf.fz.sum(), dsf.fz.sum())

    # 1e. Source over various stretched cells, source strength = pi.
    h1 = get_h(4, 2, 200, 1.1)
    h2 = get_h(4, 2, 200, 1.2)
    h3 = get_h(4, 2, 200, 1.2)
    origin = np.array([-h1.sum()/2, -h2.sum()/2, -h3.sum()/2])
    grid3 = meshes.TensorMesh([h1, h2, h3], origin)
    slen = 333
    strength = np.pi
    d_src, f_src = get_f_src([0, 0., 0., 50, 33], slen)
    dsf = fields.get_source_field(grid3, d_src, 0.7, slen*strength)
    fsf = fields.get_source_field(grid3, f_src, 0.7, strength)
    assert_allclose(fsf.fx.sum(), dsf.fx.sum())
    assert_allclose(fsf.fy.sum(), dsf.fy.sum())
    assert_allclose(fsf.fz.sum(), dsf.fz.sum())


def test_field(tmpdir):
    # Create some dummy data
    grid = meshes.TensorMesh(
            [np.array([.5, 8]), np.array([1, 4]), np.array([2, 8])],
            np.zeros(3))

    ex = create_dummy(*grid.vnEx)
    ey = create_dummy(*grid.vnEy)
    ez = create_dummy(*grid.vnEz)

    # Test the views
    ee = fields.Field(ex, ey, ez)
    assert_allclose(ee, np.r_[ex.ravel('F'), ey.ravel('F'), ez.ravel('F')])
    assert_allclose(ee.fx, ex)
    assert_allclose(ee.fy, ey)
    assert_allclose(ee.fz, ez)
    assert ee.smu0 is None
    assert ee.sval is None

    # Check it knows it is electric.
    assert ee.is_electric is True

    # Test amplitude and phase.
    assert_allclose(ee.fx.amp(), np.abs(ee.fx))
    assert_allclose(ee.fy.pha(unwrap=False), np.angle(ee.fy))

    # Test the other possibilities to initiate a Field-instance.
    ee2 = fields.Field(grid, ee.field)
    assert_allclose(ee.field, ee2.field)
    assert_allclose(ee.fx, ee2.fx)

    ee3 = fields.Field(grid)
    assert ee.shape == ee3.shape

    # Try setting values
    ee3.field = ee.field
    assert_allclose(ee.field, ee3.field)
    ee3.fx = ee.fx
    ee3.fy = ee.fy
    ee3.fz = ee.fz
    assert_allclose(ee.field, ee3.field)

    # Check PEC
    ee.ensure_pec
    assert abs(np.sum(ee.fx[:, 0, :] + ee.fx[:, -1, :])) == 0
    assert abs(np.sum(ee.fx[:, :, 0] + ee.fx[:, :, -1])) == 0
    assert abs(np.sum(ee.fy[0, :, :] + ee.fy[-1, :, :])) == 0
    assert abs(np.sum(ee.fy[:, :, 0] + ee.fy[:, :, -1])) == 0
    assert abs(np.sum(ee.fz[0, :, :] + ee.fz[-1, :, :])) == 0
    assert abs(np.sum(ee.fz[:, 0, :] + ee.fz[:, -1, :])) == 0

    # Test copy
    e2 = ee.copy()
    assert_allclose(ee.field, e2.field)
    assert_allclose(ee.fx, e2.fx)
    assert_allclose(ee.fy, e2.fy)
    assert_allclose(ee.fz, e2.fz)
    assert ee.field.base is not e2.field.base

    edict = ee.to_dict()
    del edict['field']
    with pytest.raises(KeyError, match="Variable 'field' missing"):
        fields.Field.from_dict(edict)

    # Set a dimension from the mesh to None, ensure field fails.
    if discretize is None:
        grid.nEx = None
    else:
        grid = discretize.TensorMesh([1, 1], [1, 1])
    with pytest.raises(ValueError, match='Provided grid must be a 3D grid'):
        fields.Field(grid)

    # Ensure it can be pickled.
    with shelve.open(tmpdir+'/test') as db:
        db['field'] = ee2
    with shelve.open(tmpdir+'/test') as db:
        test = db['field']
    assert_allclose(test, ee2)


def test_source_field():
    # Create some dummy data
    grid = meshes.TensorMesh(
            [np.array([.5, 8]), np.array([1, 4]), np.array([2, 8])],
            np.zeros(3))

    freq = np.pi
    ss = fields.SourceField(grid, freq=freq)
    assert_allclose(ss.smu0, -2j*np.pi*freq*constants.mu_0)
    assert hasattr(ss, 'vector')
    assert hasattr(ss, 'vx')

    # Check 0 Hz frequency.
    with pytest.raises(ValueError, match='`freq` must be >0'):
        ss = fields.SourceField(grid, freq=0)

    # Check no frequency.
    with pytest.raises(ValueError, match='SourceField requires the frequency'):
        ss = fields.SourceField(grid)

    sdict = ss.to_dict()
    del sdict['field']
    with pytest.raises(KeyError, match="Variable 'field' missing in `inp`"):
        fields.SourceField.from_dict(sdict)


def test_get_h_field():
    # Mainly regression tests, not ideal.

    # Check it does still the same (pure regression).
    dat = REGRES['reg_2']
    grid = dat['grid']
    model = dat['model']
    efield = dat['result']
    hfield = dat['hresult']

    hout = fields.get_h_field(grid, model, efield)
    assert_allclose(hfield, hout)
    # Check it knows it is magnetic.
    assert hout.is_electric is False

    # Add some mu_r - Just 1, to trigger, and compare.
    dat = REGRES['res']
    grid = dat['grid']
    efield = dat['Fresult']
    model1 = models.Model(**dat['input_model'])
    model2 = models.Model(**dat['input_model'], mu_r=1.)

    hout1 = fields.get_h_field(grid, model1, efield)
    hout2 = fields.get_h_field(grid, model2, efield)
    assert_allclose(hout1, hout2)

    # Ensure they are not the same if mu_r!=1/None provided
    model3 = models.Model(**dat['input_model'], mu_r=2.)
    hout3 = fields.get_h_field(grid, model3, efield)
    with pytest.raises(AssertionError):
        assert_allclose(hout1, hout3)


def test_get_receiver():
    grid = meshes.TensorMesh(
            [np.array([1, 1, 2, 1]), np.ones(4), np.ones(4)], [-1, -2, -2])
    field = fields.Field(grid)

    # Provide wrong rec_loc input:
    with pytest.raises(ValueError, match='Coordinates needs to be in the'):
        fields.get_receiver(grid, field.fx, (1, 1))

    # Simple linear interpolation test.
    field.fx = np.arange(1, field.fx.size+1)
    field = field.real  # For simplicity
    out1 = fields.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, [50., 50+1/3, 51])
    out1a, out1b, out1c = fields.get_receiver(  # Check recursion
            grid, field, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, out1a)
    assert_allclose(out1b, out1c)
    assert_allclose(out1b, [0, 0, 0])
    assert out1b.__class__ == utils.EMArray

    out2 = fields.get_receiver(
            grid, field.fx, ([0.5, 1, 2], 1/3, 0.25), 'linear')
    assert_allclose(out2, [56+1/3., 56+2/3, 57+1/3])

    # Check 'cubic' is re-set to 'linear for tiny grids.
    out3 = fields.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'cubic')
    assert_allclose(out1, out3)

    # Check cubic spline runs fine (NOT CHECKING ACTUAL VALUES!.
    grid = meshes.TensorMesh(
            [np.ones(4), np.array([1, 2, 3, 1]), np.array([2, 1, 1, 1])],
            [0, 0, 0])
    field = fields.Field(grid)
    field.field = np.ones(field.size) + 1j*np.ones(field.size)

    out4 = fields.get_receiver(
            grid, field.fx, ([0.5, 1, 2], [0.5, 2, 3], 2), 'linear')
    out5 = fields.get_receiver(grid, field.fx, ([0.5, 1, 2], [0.5, 2, 3], 2))
    out5real = fields.get_receiver(
            grid, field.fx.real, ([0.5, 1, 2], [0.5, 2, 3], 2))
    assert_allclose(out5, out4)
    assert_allclose(out5real, out4.real)

    # Check amplitude and phase
    assert_allclose(out5.amp(), np.abs(out5))
    assert_allclose(out5.pha(unwrap=False), np.angle(out5))

    # Check it returns 0 if outside.
    out6 = fields.get_receiver(grid, field.fx, (-10, -10, -10), 'linear')
    out7 = fields.get_receiver(grid, field.fx, (-10, -10, -10), 'cubic')

    assert_allclose(out6, np.nan+0j*np.nan)
    assert_allclose(out7, np.nan+0j*np.nan)

    # Check it does not return 0 if inside.
    out8 = fields.get_receiver(grid, field.fx, (-10, -10, -10), 'linear', True)
    out9 = fields.get_receiver(grid, field.fx, (-10, -10, -10), 'cubic', True)

    assert_allclose(out8, 1.+1j)
    assert_allclose(out9, 1.+1j)

    # Check it works with model parameters.
    model = models.Model(grid, np.ones(grid.vnC))
    out10 = fields.get_receiver(
            grid, model.property_x, (-10, -10, -10), 'linear', True)
    assert_allclose(out10, 1.)
    assert out10.__class__ != utils.EMArray


def test_get_receiver_response():
    grid = meshes.TensorMesh(
            [np.ones(6), np.array([1, 1, 2, 3, 1]), np.array([1, 2, 1, 1, 1])],
            [-1, -1, -1])
    efield = fields.Field(grid, freq=1)
    efield.field = np.ones(efield.size) + 1j*np.ones(efield.size)

    # Provide wrong rec_loc input:
    with pytest.raises(ValueError, match='`rec` needs to be in the form'):
        fields.get_receiver_response(grid, efield, (1, 1, 1))

    # Provide particular field instead of field instance:
    with pytest.raises(ValueError, match='`field` must be a `Field`-inst'):
        fields.get_receiver_response(grid, efield.fx, (1, 1, 1, 0, 0))

    # Comparison to `get_receiver`.
    rec = ([0.5, 1, 2], [0.5, 2, 3], 2)
    out1a = fields.get_receiver(grid, efield.fx, rec)
    out1b = fields.get_receiver(grid, efield.fy, rec)
    out1c = fields.get_receiver(grid, efield.fz, rec)
    out2a = fields.get_receiver_response(
                grid, efield, (rec[0], rec[1], rec[2], 0, 0))
    out2b = fields.get_receiver_response(
                grid, efield, (rec[0], rec[1], rec[2], 90, 0))
    out2c = fields.get_receiver_response(
                grid, efield, (rec[0], rec[1], rec[2], 0, 90))
    assert_allclose(out1a, out2a)
    assert_allclose(out1b, out2b)
    assert_allclose(out1c, out2c)

    # Check it returns 0 if outside.
    out3 = fields.get_receiver_response(grid, efield, (-10, -10, -10, 0, 0))
    assert_allclose(out3, np.nan+1j*np.nan)

    # Same for magnetic field.
    model = models.Model(grid)
    hfield = fields.get_h_field(grid, model, efield)

    # Comparison to `get_receiver`.
    rec = ([0.5, 1, 2], [0.5, 2, 3], 2)
    out4a = fields.get_receiver(grid, hfield.fx, rec)
    out4b = fields.get_receiver(grid, hfield.fy, rec)
    out4c = fields.get_receiver(grid, hfield.fz, rec)
    out5a = fields.get_receiver_response(
                grid, hfield, (rec[0], rec[1], rec[2], 0, 0))
    out5b = fields.get_receiver_response(
                grid, hfield, (rec[0], rec[1], rec[2], 90, 0))
    out5c = fields.get_receiver_response(
                grid, hfield, (rec[0], rec[1], rec[2], 0, 90))
    assert_allclose(out4a, out5a)
    assert_allclose(out4b, out5b)
    assert_allclose(out4c, out5c)

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
    efield = solver.solve(grid, model, sfield, semicoarsening=True,
                          sslsolver=True, linerelaxation=True, verb=1)

    # epm = empymod.bipole(src, rec, [], res, freq, verb=1)
    epm = np.array([-1.27832028e-11+1.21383502e-11j,
                    -1.90064149e-12+7.51937145e-12j,
                    1.09602131e-12+3.33066197e-12j,
                    1.25359248e-12+1.02630145e-12j])
    e3d = fields.get_receiver_response(grid, efield, rec)

    # 10 % is still OK, grid is very coarse for fast comp (2s)
    assert_allclose(epm, e3d, rtol=0.1)


def test_source_norm_warning():
    # This is a warning that should never be raised...
    hx, x0 = np.ones(4), -2
    mesh = meshes.TensorMesh([hx, hx, hx], (x0, x0, x0))
    sfield = fields.SourceField(mesh, freq=1)
    sfield.fx += 1  # Add something to the field.
    with pytest.warns(UserWarning, match="Normalizing Source: 101.0000000000"):
        _ = fields._finite_source_xyz(
                mesh, (-0.5, 0.5, 0, 0, 0, 0), sfield.fx, 0, 30)
