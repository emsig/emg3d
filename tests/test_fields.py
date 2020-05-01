import pytest
import empymod
import numpy as np
from scipy import constants
from os.path import join, dirname
from numpy.testing import assert_allclose, assert_array_equal

from emg3d import io, meshes, models, fields

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
    grid = meshes.TensorMesh([h*200, h*400, h*800], -np.array(src[:3]))
    freq = 1.2458

    sfield = fields.get_source_field(grid, src, freq, strength=1+1j)
    assert_array_equal(sfield.strength, complex(1+1j))

    sfield = fields.get_source_field(grid, src, freq, strength=0)
    assert_array_equal(sfield.strength, float(0))
    iomegamu = 2j*np.pi*freq*constants.mu_0

    # Check zeros
    assert 0 == np.sum(np.r_[sfield.fx[:, 2:, :].ravel(),
                             sfield.fy[:, 2:, :].ravel(),
                             sfield.fz[:, 2:, :].ravel()])

    # Check source cells
    h = np.cos(np.deg2rad(src[4]))
    y = np.sin(np.deg2rad(src[3]))*h
    x = np.cos(np.deg2rad(src[3]))*h
    z = np.sin(np.deg2rad(src[4]))
    assert_allclose(np.sum(sfield.fx[:2, 1, :2]/x/iomegamu).real, -1)
    assert_allclose(np.sum(sfield.fy[1, :2, :2]/y/iomegamu).real, -1)
    assert_allclose(np.sum(sfield.fz[1, 1:2, :2]/z/iomegamu).real, -1)
    assert_allclose(np.sum(sfield.vx[:2, 1, :2]/x), 1)
    assert_allclose(np.sum(sfield.vy[1, :2, :2]/y), 1)
    assert_allclose(np.sum(sfield.vz[1, 1:2, :2]/z), 1)
    assert sfield._freq == freq
    assert sfield.freq == freq
    assert_allclose(sfield.smu0, -iomegamu)

    # Put source on final node, should still work.
    src = [grid.vectorNx[-1], grid.vectorNy[-1], grid.vectorNz[-1],
           src[3], src[4]]
    sfield = fields.get_source_field(grid, src, freq)
    tot_field = np.linalg.norm(
            [np.sum(sfield.fx), np.sum(sfield.fy), np.sum(sfield.fz)])
    assert_allclose(tot_field/np.abs(np.sum(iomegamu)), 1.0)

    out, _ = capsys.readouterr()  # Empty capsys

    # Provide wrong source definition. Ensure it fails.
    with pytest.raises(ValueError):
        sfield = fields.get_source_field(grid, [0, 0, 0], 1)
    out, _ = capsys.readouterr()
    assert "ERROR   :: Source is wrong defined. Must be" in out

    # Put source way out. Ensure it fails.
    with pytest.raises(ValueError):
        src = [1e10, 1e10, 1e10, 0, 0]
        sfield = fields.get_source_field(grid, src, 1)
    out, _ = capsys.readouterr()
    assert "ERROR   :: Provided source outside grid" in out

    # Put finite dipole of zero length. Ensure it fails.
    with pytest.raises(ValueError):
        src = [0, 0, 100, 100, -200, -200]
        sfield = fields.get_source_field(grid, src, 1)
    out, _ = capsys.readouterr()
    assert "* ERROR   :: Provided source is a point dipole" in out

    # Same for Laplace domain
    src = [100, 200, 300, 27, 31]
    h = np.ones(4)
    grid = meshes.TensorMesh([h*200, h*400, h*800], -np.array(src[:3]))
    freq = 1.2458
    sfield = fields.get_source_field(grid, src, -freq)
    smu = freq*constants.mu_0

    # Check zeros
    assert 0 == np.sum(np.r_[sfield.fx[:, 2:, :].ravel(),
                             sfield.fy[:, 2:, :].ravel(),
                             sfield.fz[:, 2:, :].ravel()])

    # Check source cells
    h = np.cos(np.deg2rad(src[4]))
    y = np.sin(np.deg2rad(src[3]))*h
    x = np.cos(np.deg2rad(src[3]))*h
    z = np.sin(np.deg2rad(src[4]))
    assert_allclose(np.sum(sfield.fx[:2, 1, :2]/x/smu), -1)
    assert_allclose(np.sum(sfield.fy[1, :2, :2]/y/smu), -1)
    assert_allclose(np.sum(sfield.fz[1, 1:2, :2]/z/smu), -1)
    assert_allclose(np.sum(sfield.vx[:2, 1, :2]/x), 1)
    assert_allclose(np.sum(sfield.vy[1, :2, :2]/y), 1)
    assert_allclose(np.sum(sfield.vz[1, 1:2, :2]/z), 1)
    assert sfield._freq == -freq
    assert sfield.freq == freq
    assert_allclose(sfield.smu0, -freq*constants.mu_0)


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
    x0 = np.array([-h1.sum()/2, -h2.sum()/2, -h3.sum()/2])
    grid3 = meshes.TensorMesh([h1, h2, h3], x0)
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
    with pytest.raises(KeyError):
        fields.Field.from_dict(edict)


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
    with pytest.raises(ValueError):
        ss = fields.SourceField(grid, freq=0)

    # Check no frequency.
    with pytest.raises(ValueError):
        ss = fields.SourceField(grid)

    sdict = ss.to_dict()
    del sdict['field']
    with pytest.raises(KeyError):
        fields.SourceField.from_dict(sdict)


def test_get_h_field():
    # Mainly regression tests, not ideal.

    # Check it does still the same (pure regression).
    dat = REGRES['Data']['reg_2']
    grid = dat['grid']
    model = dat['model']
    efield = dat['result']
    hfield = dat['hresult']

    hout = fields.get_h_field(grid, model, efield)
    assert_allclose(hfield, hout)
    # Check it knows it is magnetic.
    assert hout.is_electric is False

    # Add some mu_r - Just 1, to trigger, and compare.
    dat = REGRES['Data']['res']
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
            [np.array([1, 2]), np.array([1]), np.array([1])],
            [0, 0, 0])
    field = fields.Field(grid)

    # Provide wrong rec_loc input:
    with pytest.raises(ValueError):
        fields.get_receiver(grid, field.fx, (1, 1))

    # Simple linear interpolation test.
    field.fx = np.arange(1, field.fx.size+1)
    field = field.real  # For simplicity
    out1 = fields.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, [1., 1+1/3, 2])
    out1a, out1b, out1c = fields.get_receiver(  # Check recursion
            grid, field, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, out1a)
    assert_allclose(out1b, out1c)
    assert_allclose(out1b, [0, 0, 0])
    assert out1b.__class__ == empymod.utils.EMArray

    out2 = fields.get_receiver(
            grid, field.fx, ([0.5, 1, 2], 1/3, 0.25), 'linear')
    assert_allclose(out2, [2+2/3., 3, 3+2/3])

    # Check 'cubic' is re-set to 'linear for tiny grids.
    out3 = fields.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'cubic')
    assert_allclose(out1, out3)

    # Check cubic spline runs fine (NOT CHECKING ACTUAL VALUES!.
    grid = meshes.TensorMesh(
            [np.ones(4), np.array([1, 2, 3]), np.array([2, 1, 1])],
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

    assert_allclose(out6, 0.+0j)
    assert_allclose(out7, 0.+0j)

    # Check it does not return 0 if outside.
    out8 = fields.get_receiver(grid, field.fx, (-10, -10, -10), 'linear', True)
    out9 = fields.get_receiver(grid, field.fx, (-10, -10, -10), 'cubic', True)

    assert_allclose(out8, 1.+1j)
    assert_allclose(out9, 1.+1j)

    # Check it works with model parameters.
    model = models.Model(grid, np.ones(grid.vnC))
    out10 = fields.get_receiver(
            grid, model.res_x, (-10, -10, -10), 'linear', True)
    assert_allclose(out10, 1.)
    assert out10.__class__ != empymod.utils.EMArray
