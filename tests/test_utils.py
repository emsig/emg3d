import re
import time
import pytest
import numpy as np
from scipy import constants
from os.path import join, dirname
from numpy.testing import assert_allclose

# Optional imports
try:
    import IPython
except ImportError:
    IPython = False

from emg3d import utils

# Data generated with create_data/regression.py
REGRES = np.load(join(dirname(__file__), 'data/regression.npz'),
                 allow_pickle=True)


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


# HELPER FUNCTIONS TO CREATE MESH
def test_get_domain():
    # Test default values (and therefore skindepth etc)
    h1, d1 = utils.get_domain()
    assert_allclose(h1, 55.133753)
    assert_allclose(d1, [-1378.343816, 1378.343816])

    # Ensure fact_min/fact_neg/fact_pos
    h2, d2 = utils.get_domain(fact_min=1, fact_neg=10, fact_pos=20)
    assert h2 == 5*h1
    assert 2*d1[0] == d2[0]
    assert -2*d2[0] == d2[1]

    # Check limits and min_width
    h3, d3 = utils.get_domain(limits=[-10000, 10000], min_width=[1, 10])
    assert h3 == 10
    assert np.sum(d3) == 0

    h4, d4 = utils.get_domain(limits=[-10000, 10000], min_width=5.5)
    assert h4 == 5.5
    assert np.sum(d4) == 0


def test_get_stretched_h(capsys):
    # Test min_space bigger (11) then required (10)
    h1 = utils.get_stretched_h(11, [0, 100], nx=10)
    assert_allclose(np.ones(10)*10, h1)

    # Test with range, wont end at 100
    h2 = utils.get_stretched_h(10, [-100, 100], nx=10, x0=20, x1=60)
    assert_allclose(np.ones(4)*10, h2[5:9])
    assert -100+np.sum(h2) != 100

    # Now ensure 100
    h3 = utils.get_stretched_h(10, [-100, 100], nx=10, x0=20, x1=60,
                               resp_domain=True)
    assert -100+np.sum(h3) == 100

    out, _ = capsys.readouterr()  # Empty capsys
    _ = utils.get_stretched_h(10, [-100, 100], nx=5, x0=20, x1=60)
    out, _ = capsys.readouterr()
    assert "Warning :: Not enough points for non-stretched part" in out


def test_get_hx():
    # Test alpha <= 0
    hx1 = utils.get_hx(-.5, [0, 10], 5, 3.33)
    assert_allclose(np.ones(5)*2, hx1)

    # Test x0 on domain
    hx2a = utils.get_hx(0.1, [0, 10], 5, 0)
    assert_allclose(np.ones(4)*1.1, hx2a[1:]/hx2a[:-1])
    hx2b = utils.get_hx(0.1, [0, 10], 5, 10)
    assert_allclose(np.ones(4)/1.1, hx2b[1:]/hx2b[:-1])
    assert np.sum(hx2b) == 10.0

    # Test resp_domain
    hx3 = utils.get_hx(0.1, [0, 10], 3, 8, False)
    assert np.sum(hx3) != 10.0


def test_get_source_field(capsys):
    src = [100, 200, 300, 27, 31]
    h = np.ones(4)
    grid = utils.TensorMesh([h*200, h*400, h*800], -np.array(src[:3]))
    freq = 1.2458
    sfield = utils.get_source_field(grid, src, freq)
    iomegamu = 2j*np.pi*freq*4e-7*np.pi

    # Check zeros
    assert 0 == np.sum(np.r_[sfield.fx[:, 2:, :].ravel(),
                             sfield.fy[:, 2:, :].ravel(),
                             sfield.fz[:, 2:, :].ravel()])

    # Check source cells
    h = np.cos(np.deg2rad(src[4]))
    y = np.sin(np.deg2rad(src[3]))*h
    x = np.cos(np.deg2rad(src[3]))*h
    z = np.sin(np.deg2rad(src[4]))
    assert_allclose(np.sum(sfield.fx[:2, 1, :2]/x/iomegamu).real, 1)
    assert_allclose(np.sum(sfield.fy[1, :2, :2]/y/iomegamu).real, 1)
    assert_allclose(np.sum(sfield.fz[1, 1:2, :2]/z/iomegamu).real, 1)

    # Put source on final node, should still work.
    src = [grid.vectorNx[-1], grid.vectorNy[-1], grid.vectorNz[-1],
           src[3], src[4]]
    sfield = utils.get_source_field(grid, src, freq)
    tot_field = np.linalg.norm(
            [np.sum(sfield.fx), np.sum(sfield.fy), np.sum(sfield.fz)])
    assert_allclose(tot_field/np.abs(np.sum(iomegamu)), 1.0)

    out, _ = capsys.readouterr()  # Empty capsys

    # Provide wrong source definition. Ensure it fails.
    with pytest.raises(ValueError):
        sfield = utils.get_source_field(grid, [0, 0, 0], 1)
    out, _ = capsys.readouterr()
    assert "ERROR   :: Source is wrong defined. Must be" in out

    # Put source way out. Ensure it fails.
    with pytest.raises(ValueError):
        src = [1e10, 1e10, 1e10, 0, 0]
        sfield = utils.get_source_field(grid, src, 1)
    out, _ = capsys.readouterr()
    assert "ERROR   :: Provided source outside grid" in out

    # Put finite dipole of zero length. Ensure it fails.
    with pytest.raises(ValueError):
        src = [0, 0, 100, 100, -200, -200]
        sfield = utils.get_source_field(grid, src, 1)
    out, _ = capsys.readouterr()
    assert "* ERROR   :: Provided source is a point dipole" in out


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
    grid1 = utils.TensorMesh([h, h, h], np.array([-750, -750, -750]))
    d_src, f_src = get_f_src([0, 0., 0., 23, 15])
    dsf = utils.get_source_field(grid1, d_src, 1)
    fsf = utils.get_source_field(grid1, f_src, 1)
    assert_allclose(fsf, dsf)

    # 1b. Source within one cell, source strength = pi.
    d_src, f_src = get_f_src([0, 0., 0., 32, 53])
    dsf = utils.get_source_field(grid1, d_src, 3.3, np.pi)
    fsf = utils.get_source_field(grid1, f_src, 3.3, np.pi)
    assert_allclose(fsf, dsf)

    # 1c. Source over various cells, normalized.
    h = np.ones(8)*200
    grid2 = utils.TensorMesh([h, h, h], np.array([-800, -800, -800]))
    d_src, f_src = get_f_src([0, 0., 0., 40, 20], 300.0)
    dsf = utils.get_source_field(grid2, d_src, 10.0, 0)
    fsf = utils.get_source_field(grid2, f_src, 10.0, 0)
    assert_allclose(fsf.fx.sum(), dsf.fx.sum())
    assert_allclose(fsf.fy.sum(), dsf.fy.sum())
    assert_allclose(fsf.fz.sum(), dsf.fz.sum())

    # 1d. Source over various cells, source strength = pi.
    slen = 300
    strength = np.pi
    d_src, f_src = get_f_src([0, 0., 0., 20, 30], slen)
    dsf = utils.get_source_field(grid2, d_src, 1.3, slen*strength)
    fsf = utils.get_source_field(grid2, f_src, 1.3, strength)
    assert_allclose(fsf.fx.sum(), dsf.fx.sum())
    assert_allclose(fsf.fy.sum(), dsf.fy.sum())
    assert_allclose(fsf.fz.sum(), dsf.fz.sum())

    # 1e. Source over various stretched cells, source strength = pi.
    h1 = get_h(4, 2, 200, 1.1)
    h2 = get_h(4, 2, 200, 1.2)
    h3 = get_h(4, 2, 200, 1.2)
    x0 = np.array([-h1.sum()/2, -h2.sum()/2, -h3.sum()/2])
    grid3 = utils.TensorMesh([h1, h2, h3], x0)
    slen = 333
    strength = np.pi
    d_src, f_src = get_f_src([0, 0., 0., 50, 33], slen)
    dsf = utils.get_source_field(grid3, d_src, 0.7, slen*strength)
    fsf = utils.get_source_field(grid3, f_src, 0.7, strength)
    assert_allclose(fsf.fx.sum(), dsf.fx.sum())
    assert_allclose(fsf.fy.sum(), dsf.fy.sum())
    assert_allclose(fsf.fz.sum(), dsf.fz.sum())


def test_TensorMesh():
    # Load mesh created with discretize.TensorMesh.
    grid = REGRES['grid'][()]

    # Use this grid instance to create emg3d equivalent.
    emg3dgrid = utils.TensorMesh(
            [grid['hx'], grid['hy'], grid['hz']], grid['x0'])

    # Ensure they are the same.
    for attr in grid['attr']:
        assert_allclose(grid[attr], getattr(emg3dgrid, attr))


# MODEL AND FIELD CLASSES
def test_model():
    # Mainly regression tests

    # Create some dummy data
    grid = utils.TensorMesh(
            [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
            np.zeros(3))

    res_x = create_dummy(*grid.vnC, False)
    res_y = res_x/2.0
    res_z = res_x*1.4

    # Using defaults
    model1 = utils.Model(grid)
    assert model1.mu_0 == constants.mu_0            # Check constants
    assert model1.epsilon_0 == constants.epsilon_0  # Check constants
    assert_allclose(model1.res_x, model1.res_y)
    assert_allclose(model1.nC, grid.nC)
    assert_allclose(model1.vnC, grid.vnC)
    assert_allclose(model1.vol, grid.vol)

    # Assert you can not set res_y nor res_z if not provided from the start.
    with pytest.raises(ValueError):
        model1.res_y = 2*model1.res_x
    with pytest.raises(ValueError):
        model1.res_z = 2*model1.res_x

    # Using ints
    model2 = utils.Model(grid, 2., 3., 4.)
    assert_allclose(model2.res_x*1.5, model2.res_y)
    assert_allclose(model2.res_x*2, model2.res_z)

    # VTI: Setting res_x and res_z, not res_y
    model2b = utils.Model(grid, 2., res_z=4.)
    assert_allclose(model2b.res_x, model2b.res_y)
    assert_allclose(model2b.eta_x, model2b.eta_y)
    model2b.res_z = model2b.res_x
    model2c = utils.Model(grid, 2., res_z=model2b.res_z.flatten('F'))
    assert_allclose(model2c.res_x, model2c.res_z)
    assert_allclose(model2c.eta_x, model2c.eta_z)

    # HTI: Setting res_x and res_y, not res_z
    model2d = utils.Model(grid, 2., 4.)
    assert_allclose(model2d.res_x, model2d.res_z)
    assert_allclose(model2d.eta_x, model2d.eta_z)

    # Check wrong shape
    with pytest.raises(ValueError):
        utils.Model(grid, res_x)
    with pytest.raises(ValueError):
        utils.Model(grid, res_y=res_y)
    with pytest.raises(ValueError):
        utils.Model(grid, res_z=res_z)

    # Check with all inputs
    model3 = utils.Model(grid, res_x.ravel('F'), res_y.ravel('F'),
                         res_z.ravel('F'), freq=1.234)
    assert_allclose(model3.res_x, model3.res_y*2)
    assert_allclose(model3.res_x.shape, grid.vnC)
    assert_allclose(model3.res_x, model3.res_z/1.4)
    assert_allclose(model3.res, np.r_[res_x.ravel('F'), res_y.ravel('F'),
                                      res_z.ravel('F')])
    assert model3.iomega == 2j*np.pi*model3.freq

    # Check setters
    model3.res_x = np.ones(grid.vnC)*2.0
    model3.res_y = np.ones(grid.vnC)*3.0
    model3.res_z = np.ones(grid.vnC)*4.0
    assert_allclose(model2.res, model3.res)
    model3.res = np.ones(grid.nC*3)
    assert_allclose(model1.res, model3.res[:grid.nC])

    # Check eta
    iommu = model3.iomega*model3.mu_0
    iomep = model3.iomega*model3.epsilon_0
    eta_x = iommu*(1./model3.res_x.ravel('F') - iomep)*model3.vol
    eta_y = iommu*(1./model3.res_y.ravel('F') - iomep)*model3.vol
    eta_z = iommu*(1./model3.res_z.ravel('F') - iomep)*model3.vol
    assert_allclose(model3.eta_x.ravel('F'), eta_x)
    assert_allclose(model3.eta_y.ravel('F'), eta_y)
    assert_allclose(model3.eta_z.ravel('F'), eta_z)


def test_field():
    # Create some dummy data
    grid = utils.TensorMesh(
            [np.array([.5, 8]), np.array([1, 4]), np.array([2, 8])],
            np.zeros(3))

    ex = create_dummy(*grid.vnEx)
    ey = create_dummy(*grid.vnEy)
    ez = create_dummy(*grid.vnEz)

    # Test the views
    ee = utils.Field(ex, ey, ez)
    assert_allclose(ee, np.r_[ex.ravel('F'), ey.ravel('F'), ez.ravel('F')])
    assert_allclose(ee.fx, ex)
    assert_allclose(ee.fy, ey)
    assert_allclose(ee.fz, ez)

    # Test the other possibilities to initiate a Field-instance.
    ee2 = utils.Field(grid, ee.field)
    assert_allclose(ee.field, ee2.field)
    assert_allclose(ee.fx, ee2.fx)

    ee3 = utils.Field(grid)
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


# FUNCTIONS RELATED TO TIMING
def test_Time():
    t0 = utils.timeit()  # Create almost at the same time a
    time = utils.Time()  # t0-stamp and a Time-instance.

    assert_allclose(t0, time.t0)
    assert time.now == utils.now()
    assert time.runtime[:-3] == utils.timeit(time.t0)[:-3]


def test_now():
    out = utils.now()
    assert re.match(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]', out)


def test_timeit():
    t0 = utils.timeit()
    out = utils.timeit(t0-1)
    assert "0:00:01.00" in out
    out = utils.timeit(t0-10)
    assert "0:00:10" == out


def test_ctimeit(capsys):
    with utils.ctimeit("The command sleep(1) took ", " long!"):
        time.sleep(1)
    out, _ = capsys.readouterr()

    assert "The command sleep(1) took " in out
    assert " long!" in out
    assert "0:00:01" in out


# FUNCTIONS RELATED TO DATA MANAGEMENT
def test_data_write_read(tmpdir):
    # Create test data
    grid = utils.TensorMesh(
            [np.array([100, 4]), np.array([100, 8]), np.array([100, 16])],
            np.zeros(3))

    e1 = create_dummy(*grid.vnEx)
    e2 = create_dummy(*grid.vnEy)
    e3 = create_dummy(*grid.vnEz)
    ee = utils.Field(e1, e2, e3)

    # Write and read data, single arguments
    utils.data_write('testthis', 'ee', ee, tmpdir)
    ee_out = utils.data_read('testthis', 'ee', tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)

    # Write and read data, multi arguments
    utils.data_write('testthis', ('grid', 'ee'), (grid, ee), tmpdir)
    grid_out, ee_out = utils.data_read('testthis', ('grid', 'ee'), tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(grid_out, attr)

    # Write and read data, None
    utils.data_write('testthis', ('grid', 'ee'), (grid, ee), tmpdir)
    out = utils.data_read('testthis', path=tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(out['grid'], attr)


# OTHER
def test_versions(capsys):

    # Check the default
    print(utils.Versions())
    out1, _ = capsys.readouterr()

    # Check one of the standard packages
    assert 'numpy' in out1

    # Check with an additional package
    print(utils.Versions(add_pckg=re))
    out2, _ = capsys.readouterr()

    # Check the provided package, with number
    assert re.__version__ + ' : re' in out2

    # Check the 'text'-version, providing a package as tuple
    v1 = utils.Versions(add_pckg=(re, ))
    print(v1.__repr__())
    out3, _ = capsys.readouterr()

    # They have to be the same, except time (run at slightly different times)
    assert out2[75:] == out3[75:]

    # Check 'HTML'/'html'-version, providing a package as a list
    v2 = utils.Versions(add_pckg=[re])
    print(v2._repr_html_())
    out4, _ = capsys.readouterr()

    assert 'numpy' in out4
    assert 'td style=' in out4

    # Check row of provided package, with number
    teststr = "<td style='text-align: right; background-color: #ccc; "
    teststr += "border: 2px solid #fff;'>"
    teststr += re.__version__
    teststr += "</td>\n    <td style='"
    teststr += "text-align: left; border: 2px solid #fff;'>re</td>"
    assert teststr in out4
