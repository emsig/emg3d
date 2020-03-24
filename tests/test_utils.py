import re
import os
import pytest
import empymod
import numpy as np
from scipy import constants
from timeit import default_timer
from os.path import join, dirname
from numpy.testing import assert_allclose, assert_array_equal

# Optional import
try:
    import scooby
except ImportError:
    scooby = False

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
def test_get_hx_h0(capsys):

    # == A == Just the defaults, no big thing (regression).
    out1 = utils.get_hx_h0(
            freq=.5, res=10, fixed=900, domain=[-2000, 2000],
            possible_nx=[20, 32], return_info=True)
    outstr1, _ = capsys.readouterr()

    # Partially regression, partially from output-info.
    info = (
        "   Skin depth          [m] : 2251\n"
        "   Survey domain       [m] : -2000 - 2000\n"
        "   Calculation domain  [m] : -14692 - 15592\n"
        "   Final extent        [m] : -15698 - 15998\n"
        f"   Min/max cell width  [m] : {out1[2]['dmin']:.0f} / 750 / 3382\n"
        "   Alpha survey/calc       : "
        f"{out1[2]['amin']:.3f} / {out1[2]['amax']:.3f}\n"
        "   Number of cells (s/c/r) : 20 (6/14/0)\n"
    )

    # Just check x0 and the output.
    assert out1[1] == -15698.299823718215
    assert info in outstr1

    # == B == Laplace and verb=0, parameter positions and defaults.
    out2 = utils.get_hx_h0(
            -.5/np.pi/2, 10, [-2000, 2000], 900, [20, 32], None, 3,
            [1.05, 1.5, 0.01], 100000., False, 0, True)
    outstr2, _ = capsys.readouterr()

    # Assert they are the same.
    assert_allclose(out1[0], out2[0])

    # Assert nothing is printed with verb=0.
    assert outstr2 == ""

    # == C == User limits.
    out3 = utils.get_hx_h0(
            freq=.5, res=10, fixed=900, domain=[-11000, 14000],
            possible_nx=[20, 32, 64, 128], min_width=[20, 600],
            return_info=True)
    outstr3, _ = capsys.readouterr()

    # Check dmin.
    assert out3[2]['dmin'] == 600
    # Calculation domain has to be at least domain.
    assert out3[1]+np.sum(out3[0]) > 14000
    assert out3[1] <= -11000

    # == D == Check failure.
    # (a) With raise.
    with pytest.raises(ArithmeticError):
        utils.get_hx_h0(
                freq=.5, res=[10., 12.], fixed=900, domain=[-10000, 10000],
                possible_nx=[20])

    # (b) With raise=False.
    out4 = utils.get_hx_h0(
            freq=.5, res=10, fixed=900, domain=[-500, 500],
            possible_nx=[32, 40], min_width=40.,
            alpha=[1.045, 1.66, 0.005], raise_error=False, return_info=True)
    outstr4, _ = capsys.readouterr()
    assert out4[0] is None
    assert out4[1] is None
    assert_allclose(out4[2]['amin'], 1.045)  # If fails, must have big. def.
    assert_allclose(out4[2]['amax'], 1.66)   # anis-values for both domains.

    # == E == Fixed boundaries
    # Too many values.
    with pytest.raises(ValueError):
        utils.get_hx_h0(
            freq=1, res=1, fixed=[-900, -1000, 0, 5], domain=[-2000, 0],
            possible_nx=[64, 128])
    # Two additional values, but both on same side.
    with pytest.raises(ValueError):
        utils.get_hx_h0(
            freq=1, res=1, fixed=[900, -1000, -1200], domain=[-2000, 0],
            possible_nx=[64, 128])

    # One additional fixed.
    out5 = utils.get_hx_h0(
            freq=1, res=1, fixed=[-900, 0], domain=[-2000, 0],
            possible_nx=[64, 128], min_width=[50, 100],
            alpha=[1., 1, 1, 1], return_info=True)
    outstr5, _ = capsys.readouterr()

    nodes5 = out5[1]+np.cumsum(out5[0])
    assert_allclose(0.0, min(abs(nodes5)), atol=1e-8)  # Check sea-surface.
    assert out5[2]['amax'] < 1.02

    # Two additional fixed.
    out6 = utils.get_hx_h0(
            freq=1, res=1, fixed=[-890, 0, -1000], domain=[-2000, 0],
            possible_nx=[64, 128], min_width=[60, 70])
    outstr6, _ = capsys.readouterr()
    nodes6 = out6[1]+np.cumsum(out6[0])
    assert_allclose(0.0, min(abs(nodes6)), atol=1e-8)  # Check sea-surface.
    assert_allclose(0.0, min(abs(nodes6+1000)), atol=1e-8)  # Check seafloor.

    # == F == Several resistivities
    out7 = utils.get_hx_h0(1, [0.3, 10], [-1000, 1000], alpha=[1, 1, 1])
    assert out7[1] < -10000
    assert out7[1]+np.sum(out7[0]) > 10000

    out8 = utils.get_hx_h0(1, [0.3, 1, 90], [-1000, 1000], alpha=[1, 1, 1])
    assert out8[1] > -5000                  # Left buffer much smaller than
    assert out8[1]+np.sum(out8[0]) > 30000  # right buffer.


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

    # Ensure Laplace and frequency
    h5a, d5a = utils.get_domain(freq=1.)
    h5b, d5b = utils.get_domain(freq=-1./2/np.pi)
    assert h5a == h5b
    assert d5a == d5b


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


def test_get_cell_numbers(capsys):
    numbers = utils.get_cell_numbers(max_nr=128, max_prime=5, min_div=3)
    assert_allclose([16, 24, 32, 40, 48, 64, 80, 96, 128], numbers)

    with pytest.raises(ValueError):
        numbers = utils.get_cell_numbers(max_nr=128, max_prime=25, min_div=3)
    out, _ = capsys.readouterr()
    assert "* ERROR   :: Highest prime is 25" in out

    numbers = utils.get_cell_numbers(max_nr=50, max_prime=3, min_div=5)
    assert len(numbers) == 0


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

    sfield = utils.get_source_field(grid, src, freq, strength=1+1j)
    assert_array_equal(sfield.strength, complex(1+1j))

    sfield = utils.get_source_field(grid, src, freq, strength=0)
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

    # Same for Laplace domain
    src = [100, 200, 300, 27, 31]
    h = np.ones(4)
    grid = utils.TensorMesh([h*200, h*400, h*800], -np.array(src[:3]))
    freq = 1.2458
    sfield = utils.get_source_field(grid, src, -freq)
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

    # Copy
    cgrid = emg3dgrid.copy()
    assert_allclose(cgrid.vol, emg3dgrid.vol)
    dgrid = emg3dgrid.to_dict()
    cdgrid = utils.TensorMesh.from_dict(dgrid)
    assert_allclose(cdgrid.vol, emg3dgrid.vol)
    del dgrid['hx']
    with pytest.raises(KeyError):
        utils.TensorMesh.from_dict(dgrid)


# MODEL AND FIELD CLASSES
def test_Model(capsys):
    # Mainly regression tests

    # Create some dummy data
    grid = utils.TensorMesh(
            [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
            np.zeros(3))

    res_x = create_dummy(*grid.vnC, False)
    res_y = res_x/2.0
    res_z = res_x*1.4
    mu_r = res_x*1.11

    # Check representation of TensorMesh.
    assert 'TensorMesh: 2 x 2 x 2 (8)' in grid.__repr__()

    _, _ = capsys.readouterr()  # Clean-up
    # Using defaults; check backwards compatibility for freq.
    sfield = utils.SourceField(grid, freq=1)
    model1 = utils.Model(grid)

    # Check representation of Model.
    assert 'Model; isotropic resistivities' in model1.__repr__()

    model1e = utils.Model(grid, freq=1)
    model1.mu_r = None
    model1.epsilon_r = None
    vmodel1 = utils.VolumeModel(grid, model1, sfield)
    out, _ = capsys.readouterr()
    assert '* WARNING :: ``Model`` is independent of frequency' in out
    assert_allclose(model1.res_x, model1.res_y)
    assert_allclose(model1.nC, grid.nC)
    assert_allclose(model1.vnC, grid.vnC)
    assert_allclose(vmodel1.eta_z, vmodel1.eta_y)
    assert model1.mu_r is None
    assert model1.epsilon_r is None

    # Assert that the cases changes if we assign y- and z- resistivities (I).
    model1.res_x = model1.res_x
    assert model1.case == 0
    model1.res_y = 2*model1.res_x
    assert model1.case == 1
    model1.res_z = 2*model1.res_x
    assert model1.case == 3

    assert model1e.case == 0
    model1e.res_z = 2*model1.res_x
    assert model1e.case == 2
    model1e.res_y = 2*model1.res_x
    assert model1e.case == 3

    model1.epsilon_r = 1  # Just set to 1 CURRENTLY.
    vmodel1b = utils.VolumeModel(grid, model1, sfield)
    assert_allclose(vmodel1b.eta_x, vmodel1.eta_x)
    assert np.iscomplex(vmodel1b.eta_x[0, 0, 0])

    # Using ints
    model2 = utils.Model(grid, 2., 3., 4.)
    vmodel2 = utils.VolumeModel(grid, model2, sfield)
    assert_allclose(model2.res_x*1.5, model2.res_y)
    assert_allclose(model2.res_x*2, model2.res_z)

    # VTI: Setting res_x and res_z, not res_y
    model2b = utils.Model(grid, 2., res_z=4.)
    vmodel2b = utils.VolumeModel(grid, model2b, sfield)
    assert_allclose(model2b.res_x, model2b.res_y)
    assert_allclose(vmodel2b.eta_y, vmodel2b.eta_x)
    model2b.res_z = model2b.res_x
    model2c = utils.Model(grid, 2., res_z=model2b.res_z.copy())
    vmodel2c = utils.VolumeModel(grid, model2c, sfield)
    assert_allclose(model2c.res_x, model2c.res_z)
    assert_allclose(vmodel2c.eta_z, vmodel2c.eta_x)

    # HTI: Setting res_x and res_y, not res_z
    model2d = utils.Model(grid, 2., 4.)
    vmodel2d = utils.VolumeModel(grid, model2d, sfield)
    assert_allclose(model2d.res_x, model2d.res_z)
    assert_allclose(vmodel2d.eta_z, vmodel2d.eta_x)

    # Pure air, epsilon_r init with 0 => should be 1!
    model6 = utils.Model(grid, 2e14)
    assert model6.epsilon_r is None

    # Check wrong shape
    with pytest.raises(ValueError):
        utils.Model(grid, np.arange(1, 11))
    with pytest.raises(ValueError):
        utils.Model(grid, res_y=np.ones((2, 5, 6)))
    with pytest.raises(ValueError):
        utils.Model(grid, res_z=np.array([1, 3]))
    with pytest.raises(ValueError):
        utils.Model(grid, mu_r=np.array([[1, ], [3, ]]))

    # Check with all inputs
    gridvol = grid.vol.reshape(grid.vnC, order='F')
    model3 = utils.Model(grid, res_x, res_y, res_z, mu_r=mu_r)
    vmodel3 = utils.VolumeModel(grid, model3, sfield)
    assert_allclose(model3.res_x, model3.res_y*2)
    assert_allclose(model3.res_x.shape, grid.vnC)
    assert_allclose(model3.res_x, model3.res_z/1.4)
    assert_allclose(gridvol/mu_r, vmodel3.zeta)
    # Check with all inputs
    model3b = utils.Model(grid, res_x.ravel('F'), res_y.ravel('F'),
                          res_z.ravel('F'), mu_r=mu_r.ravel('F'))
    vmodel3b = utils.VolumeModel(grid, model3b, sfield)
    assert_allclose(model3b.res_x, model3b.res_y*2)
    assert_allclose(model3b.res_x.shape, grid.vnC)
    assert_allclose(model3b.res_x, model3b.res_z/1.4)
    assert_allclose(gridvol/mu_r, vmodel3b.zeta)

    # Check setters vnC
    tres = np.ones(grid.vnC)
    model3.res_x[:, :, :] = tres*2.0
    model3.res_y[:, :1, :] = tres[:, :1, :]*3.0
    model3.res_y[:, 1:, :] = tres[:, 1:, :]*3.0
    model3.res_z = tres*4.0
    model3.mu_r = tres*5.0
    model3.epsilon_r = tres*6.0
    assert_allclose(tres*2., model3.res_x)
    assert_allclose(tres*3., model3.res_y)
    assert_allclose(tres*4., model3.res_z)
    assert_allclose(tres*6., model3.epsilon_r)

    # Check eta
    iomep = sfield.sval*utils.epsilon_0
    eta_x = sfield.smu0*(1./model3.res_x + iomep)*gridvol
    eta_y = sfield.smu0*(1./model3.res_y + iomep)*gridvol
    eta_z = sfield.smu0*(1./model3.res_z + iomep)*gridvol
    vmodel3 = utils.VolumeModel(grid, model3, sfield)
    assert_allclose(vmodel3.eta_x, eta_x)
    assert_allclose(vmodel3.eta_y, eta_y)
    assert_allclose(vmodel3.eta_z, eta_z)

    # Check volume
    assert_allclose(grid.vol.reshape(grid.vnC, order='F'), vmodel2.zeta)
    model4 = utils.Model(grid, 1)
    vmodel4 = utils.VolumeModel(grid, model4, sfield)
    assert_allclose(vmodel4.zeta, grid.vol.reshape(grid.vnC, order='F'))

    # Check a couple of out-of-range failures
    with pytest.raises(ValueError):
        _ = utils.Model(grid, res_x=res_x*0)
    Model = utils.Model(grid, res_x=res_x)
    with pytest.raises(ValueError):
        Model._check_parameter(res_x*0, 'res_x')
    with pytest.raises(ValueError):
        Model._check_parameter(-1.0, 'res_x')
    with pytest.raises(ValueError):
        _ = utils.Model(grid, res_y=np.inf)
    with pytest.raises(ValueError):
        _ = utils.Model(grid, res_z=res_z*np.inf)
    with pytest.raises(ValueError):
        _ = utils.Model(grid, mu_r=-1)


class TestModelOperators:

    # Define two different sized meshes.
    mesh_base = utils.TensorMesh(
            [np.ones(3), np.ones(4), np.ones(5)], x0=np.array([0, 0, 0]))
    mesh_diff = utils.TensorMesh(
            [np.ones(3), np.ones(4), np.ones(6)], x0=np.array([0, 0, 0]))

    # Define a couple of models.
    model_int = utils.Model(mesh_base, 1.)
    model_1_a = utils.Model(mesh_base, 1., 2.)
    model_1_b = utils.Model(mesh_base, 2., 4.)
    model_2_a = utils.Model(mesh_base, 1., res_z=3.)
    model_2_b = utils.Model(mesh_base, 2., res_z=6.)
    model_3_a = utils.Model(mesh_base, 1., 2., 3.)
    model_3_b = utils.Model(mesh_base, 2., 4., 6.)
    model_mu_a = utils.Model(mesh_base, 1., mu_r=1.)
    model_mu_b = utils.Model(mesh_base, 2., mu_r=2.)
    model_epsilon_a = utils.Model(mesh_base, 1., epsilon_r=1.)
    model_epsilon_b = utils.Model(mesh_base, 2., epsilon_r=2.)

    model_int_diff = utils.Model(mesh_diff, 1.)
    model_nC = utils.Model(mesh_base, np.ones(mesh_base.nC)*2.)
    model_vnC = utils.Model(mesh_base, np.ones(mesh_base.vnC))

    def test_operator_test(self):
        with pytest.raises(ValueError):
            self.model_int + self.model_1_a
        with pytest.raises(ValueError):
            self.model_int - self.model_2_a
        with pytest.raises(ValueError):
            self.model_int + self.model_3_a
        with pytest.raises(ValueError):
            self.model_int - self.model_mu_a
        with pytest.raises(ValueError):
            self.model_int + self.model_epsilon_a
        with pytest.raises(ValueError):
            self.model_int - self.model_int_diff

    def test_add(self):
        a = self.model_int + self.model_nC
        b = self.model_nC + self.model_vnC
        c = self.model_int + self.model_vnC
        assert a == b
        assert a != c
        assert a.res_x[0, 0, 0] == 3.0
        assert b.res_x[0, 0, 0] == 3.0
        assert c.res_x[0, 0, 0] == 2.0

        # Addition with something else than a model
        with pytest.raises(TypeError):
            self.model_int + self.mesh_base

        # All different cases
        a = self.model_1_a + self.model_1_a
        assert a.res_x == self.model_1_b.res_x
        assert a.res_y == self.model_1_b.res_y
        assert a.res_z.base is a.res_x.base

        a = self.model_2_a + self.model_2_a
        assert a.res_x == self.model_2_b.res_x
        assert a.res_y.base is a.res_x.base
        assert a.res_z == self.model_2_b.res_z

        a = self.model_3_a + self.model_3_a
        assert a.res_x == self.model_3_b.res_x
        assert a.res_y == self.model_3_b.res_y
        assert a.res_z == self.model_3_b.res_z

        # mu_r and epsilon_r
        a = self.model_mu_a + self.model_mu_a
        b = self.model_epsilon_a + self.model_epsilon_a
        assert a.mu_r == self.model_mu_b.mu_r
        assert b.epsilon_r == self.model_epsilon_b.epsilon_r

    def test_sub(self):
        # Addition with something else than a model
        with pytest.raises(TypeError):
            self.model_int - self.mesh_base

        # All different cases
        a = self.model_1_b - self.model_1_a
        assert a.res_x == self.model_1_a.res_x
        assert a.res_y == self.model_1_a.res_y
        assert a.res_z.base is a.res_x.base

        a = self.model_2_b - self.model_2_a
        assert a.res_x == self.model_2_a.res_x
        assert a.res_y.base is a.res_x.base
        assert a.res_z == self.model_2_a.res_z

        a = self.model_3_b - self.model_3_a
        assert a.res_x == self.model_3_a.res_x
        assert a.res_y == self.model_3_a.res_y
        assert a.res_z == self.model_3_a.res_z

        # mu_r and epsilon_r
        a = self.model_mu_b - self.model_mu_a
        b = self.model_epsilon_b - self.model_epsilon_a
        assert a.mu_r == self.model_mu_a.mu_r
        assert b.epsilon_r == self.model_epsilon_a.epsilon_r

    def test_eq(self):

        assert (self.model_int == self.mesh_base) is False

        out = self.model_int == self.model_int_diff
        assert not out

        out = self.model_int == self.model_int
        assert out

    def test_general(self):

        # Check shape and size
        assert_allclose(self.model_int.shape, self.mesh_base.vnC)
        assert self.model_int.size == self.mesh_base.nC

    def test_copy(self):
        model_new1 = self.model_vnC.copy()
        model_new2 = self.model_3_a.copy()
        model_new3 = self.model_mu_a.copy()
        model_new4 = self.model_epsilon_a.copy()

        assert model_new1 == self.model_vnC
        assert model_new2 == self.model_3_a
        assert model_new3 == self.model_mu_a
        assert model_new4 == self.model_epsilon_a

        assert model_new1.res_x.base is not self.model_vnC.res_x.base
        assert model_new2.res_y.base is not self.model_3_a.res_y.base
        assert model_new2.res_z.base is not self.model_3_a.res_z.base
        assert model_new3.mu_r.base is not self.model_mu_a.mu_r.base
        assert (model_new4.epsilon_r.base is not
                self.model_epsilon_a.epsilon_r.base)

    def test_dict(self):
        # dict is already tested via copy. Just the other cases here.
        mdict = self.model_3_b.to_dict()
        keys = ['res_x', 'res_y', 'res_z', 'mu_r', 'epsilon_r', 'vnC']
        for key in keys:
            assert key in mdict.keys()
        for key in keys[:3]:
            val = getattr(self.model_3_b, key)
            assert_allclose(mdict[key], val)

        del mdict['res_x']
        with pytest.raises(KeyError):
            utils.Model.from_dict(mdict)


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

    # Check it knows it is electric.
    assert ee.is_electric is True

    # Test amplitude and phase.
    assert_allclose(ee.fx.amp, np.abs(ee.fx))
    assert_allclose(ee.fy.pha, np.rad2deg(np.unwrap(np.angle(ee.fy))))

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
        utils.Field.from_dict(edict)


def test_source_field():
    # Create some dummy data
    grid = utils.TensorMesh(
            [np.array([.5, 8]), np.array([1, 4]), np.array([2, 8])],
            np.zeros(3))

    freq = np.pi
    ss = utils.SourceField(grid, freq=freq)
    assert_allclose(ss.smu0, -2j*np.pi*freq*constants.mu_0)
    assert hasattr(ss, 'vector')
    assert hasattr(ss, 'vx')

    # Check 0 Hz frequency.
    with pytest.raises(ValueError):
        ss = utils.SourceField(grid, freq=0)


def test_get_h_field():
    # Mainly regression tests, not ideal.

    # Check it does still the same (pure regression).
    dat = REGRES['reg_2'][()]
    grid = dat['grid']
    model = dat['model']
    efield = dat['result']
    hfield = dat['hresult']

    hout = utils.get_h_field(grid, model, efield)
    assert_allclose(hfield, hout)
    # Check it knows it is magnetic.
    assert hout.is_electric is False

    # Add some mu_r - Just 1, to trigger, and compare.
    dat = REGRES['res'][()]
    grid = dat['grid']
    efield = dat['Fresult']
    model1 = utils.Model(**dat['input_model'])
    model2 = utils.Model(**dat['input_model'], mu_r=1.)

    hout1 = utils.get_h_field(grid, model1, efield)
    hout2 = utils.get_h_field(grid, model2, efield)
    assert_allclose(hout1, hout2)

    # Ensure they are not the same if mu_r!=1/None provided
    model3 = utils.Model(**dat['input_model'], mu_r=2.)
    hout3 = utils.get_h_field(grid, model3, efield)
    with pytest.raises(AssertionError):
        assert_allclose(hout1, hout3)


def test_get_receiver():
    grid = utils.TensorMesh(
            [np.array([1, 2]), np.array([1]), np.array([1])],
            [0, 0, 0])
    field = utils.Field(grid)

    # Provide wrong rec_loc input:
    with pytest.raises(ValueError):
        utils.get_receiver(grid, field.fx, (1, 1))

    # Simple linear interpolation test.
    field.fx = np.arange(1, field.fx.size+1)
    field = field.real  # For simplicity
    out1 = utils.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, [1., 1+1/3, 2])
    out1a, out1b, out1c = utils.get_receiver(  # Check recursion
            grid, field, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, out1a)
    assert_allclose(out1b, out1c)
    assert_allclose(out1b, [0, 0, 0])
    assert out1b.__class__ == empymod.utils.EMArray

    out2 = utils.get_receiver(
            grid, field.fx, ([0.5, 1, 2], 1/3, 0.25), 'linear')
    assert_allclose(out2, [2+2/3., 3, 3+2/3])

    # Check 'cubic' is re-set to 'linear for tiny grids.
    out3 = utils.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'cubic')
    assert_allclose(out1, out3)

    # Check cubic spline runs fine (NOT CHECKING ACTUAL VALUES!.
    grid = utils.TensorMesh(
            [np.ones(4), np.array([1, 2, 3]), np.array([2, 1, 1])],
            [0, 0, 0])
    field = utils.Field(grid)
    field.field = np.ones(field.size) + 1j*np.ones(field.size)

    out4 = utils.get_receiver(
            grid, field.fx, ([0.5, 1, 2], [0.5, 2, 3], 2), 'linear')
    out5 = utils.get_receiver(grid, field.fx, ([0.5, 1, 2], [0.5, 2, 3], 2))
    out5real = utils.get_receiver(
            grid, field.fx.real, ([0.5, 1, 2], [0.5, 2, 3], 2))
    assert_allclose(out5, out4)
    assert_allclose(out5real, out4.real)

    # Check amplitude and phase
    assert_allclose(out5.amp, np.abs(out5))
    assert_allclose(out5.pha, np.rad2deg(np.unwrap(np.angle(out5))))

    # Check it returns 0 if outside.
    out6 = utils.get_receiver(grid, field.fx, (-10, -10, -10), 'linear')
    out7 = utils.get_receiver(grid, field.fx, (-10, -10, -10), 'cubic')

    assert_allclose(out6, 0.+0j)
    assert_allclose(out7, 0.+0j)

    # Check it does not return 0 if outside.
    out8 = utils.get_receiver(grid, field.fx, (-10, -10, -10), 'linear', True)
    out9 = utils.get_receiver(grid, field.fx, (-10, -10, -10), 'cubic', True)

    assert_allclose(out8, 1.+1j)
    assert_allclose(out9, 1.+1j)

    # Check it works with model parameters.
    model = utils.Model(grid, np.ones(grid.vnC))
    out10 = utils.get_receiver(
            grid, model.res_x, (-10, -10, -10), 'linear', True)
    assert_allclose(out10, 1.)
    assert out10.__class__ != empymod.utils.EMArray


def test_grid2grid_volume():
    # == X == Simple 1D model
    grid_in = utils.TensorMesh(
            [np.ones(5)*10, np.array([1, ]), np.array([1, ])],
            x0=np.array([0, 0, 0]))
    grid_out = utils.TensorMesh(
            [np.array([10, 25, 10, 5, 2]), np.array([1, ]), np.array([1, ])],
            x0=np.array([-5, 0, 0]))
    values_in = np.array([1., 5., 3, 7, 2])[:, None, None]
    values_out = utils.grid2grid(grid_in, values_in, grid_out, 'volume')

    # Result 2nd cell: (5*1+10*5+10*3)/25=3.4
    assert_allclose(values_out[:, 0, 0], np.array([1, 3.4, 7, 2, 2]))

    # Check log:
    vlogparam = utils.grid2grid(
            grid_in, values_in, grid_out, 'volume', log=True)
    vlinloginp = utils.grid2grid(
            grid_in, np.log10(values_in), grid_out, 'volume')
    assert_allclose(vlogparam, 10**vlinloginp)

    # == Y ==  Reverse it
    grid_out = utils.TensorMesh(
            [np.array([1, ]), np.ones(5)*10, np.array([1, ])],
            x0=np.array([0, 0, 0]))
    grid_in = utils.TensorMesh(
            [np.array([1, ]), np.array([10, 25, 10, 5, 2]), np.array([1, ])],
            x0=np.array([0, -5, 0]))
    values_in = np.array([1, 3.4, 7, 2, 2])[None, :, None]
    values_out = utils.grid2grid(grid_in, values_in, grid_out, 'volume')

    # Result 1st cell: (5*1+5*3.4)/10=2.2
    assert_allclose(values_out[0, :, 0], np.array([2.2, 3.4, 3.4, 7, 2]))

    # == Z == Another 1D test
    grid_in = utils.TensorMesh(
            [np.array([1, ]), np.array([1, ]), np.ones(9)*10],
            x0=np.array([0, 0, 0]))
    grid_out = utils.TensorMesh(
            [np.array([1, ]), np.array([1, ]), np.array([20, 41, 9, 30])],
            x0=np.array([0, 0, 0]))
    values_in = np.arange(1., 10)[None, None, :]
    values_out = utils.grid2grid(grid_in, values_in, grid_out, 'volume')

    assert_allclose(values_out[0, 0, :], np.array([1.5, 187/41, 7, 260/30]))

    # == 3D ==
    grid_in = utils.TensorMesh(
            [np.array([1, 1, 1]), np.array([10, 10, 10, 10, 10]),
             np.array([10, 2, 10])], x0=np.array([0, 0, 0]))
    grid_out = utils.TensorMesh(
            [np.array([1, 2, ]), np.array([10, 25, 10, 5, 2]),
             np.array([4, 4, 4])], x0=np.array([0, -5, 6]))
    create = np.array([[1, 2., 1]])
    create = np.array([create, 2*create, create])
    values_in = create*np.array([1., 5., 3, 7, 2])[None, :, None]

    values_out = utils.grid2grid(grid_in, values_in, grid_out, 'volume')

    check = np.array([[1, 1.5, 1], [1.5, 2.25, 1.5]])[:, None, :]
    check = check*np.array([1, 3.4, 7, 2, 2])[None, :, None]

    assert_allclose(values_out, check)

    # == If the extent is the same, volume*values must remain constant. ==
    grid_in = utils.TensorMesh(
            [np.array([1, 1, 1]), np.array([10, 10, 10, 10, 10]),
             np.array([10, 2, 10])], x0=np.array([0, 0, 0]))
    grid_out = utils.TensorMesh(
            [np.array([1, 2, ]), np.array([5, 25, 10, 5, 5]),
             np.array([9, 4, 9])], x0=np.array([0, 0, 0]))
    create = np.array([[1, 2., 1]])
    create = np.array([create, 2*create, create])
    values_in = create*np.array([1., 5., 3, 7, 2])[None, :, None]
    vol_in = np.outer(np.outer(grid_in.hx, grid_in.hy).ravel('F'), grid_in.hz)
    vol_in = vol_in.ravel('F').reshape(grid_in.vnC, order='F')

    values_out = utils.grid2grid(grid_in, values_in, grid_out, 'volume')
    vol_out = np.outer(np.outer(grid_out.hx, grid_out.hy).ravel('F'),
                       grid_out.hz)
    vol_out = vol_out.ravel('F').reshape(grid_out.vnC, order='F')

    assert_allclose(np.sum(values_out*vol_out), np.sum(values_in*vol_in))


def test_grid2grid():
    igrid = utils.TensorMesh(
            [np.array([1, 1]), np.array([1]), np.array([1])],
            [0, 0, 0])
    ogrid = utils.TensorMesh(
            [np.array([1]), np.array([1]), np.array([1])],
            [0, 0, 0])
    values = np.array([1.0, 2.0]).reshape(igrid.vnC)

    # Provide wrong dimension:
    with pytest.raises(ValueError):
        utils.grid2grid(igrid, values[1:, :, :], ogrid)

    # Simple, linear example.
    out = utils.grid2grid(igrid, values, ogrid, 'linear')
    np.allclose(out, np.array([1.5]))

    # Provide ogrid.gridCC.
    ogrid.gridCC = np.array([[0.5, 0.5, 0.5]])
    out2 = utils.grid2grid(igrid, values, ogrid, 'linear')
    np.allclose(out2, np.array([1.5]))

    # Check 'linear' and 'cubic' yield almost the same result for a well
    # determined, very smoothly changing example.

    # Fine grid.
    fgrid = utils.TensorMesh(
        [np.ones(2**6)*10, np.ones(2**5)*100, np.ones(2**4)*1000],
        x0=np.array([-320., -1600, -8000]))

    # Smoothly changing model for fine grid.
    cmodel = np.arange(1, fgrid.nC+1).reshape(fgrid.vnC, order='F')

    # Coarser grid.
    cgrid = utils.TensorMesh(
        [np.ones(2**5)*15, np.ones(2**4)*150, np.ones(2**3)*1500],
        x0=np.array([-240., -1200, -6000]))

    # Interpolate linearly and cubic spline.
    lin_model = utils.grid2grid(fgrid, cmodel, cgrid, 'linear')
    cub_model = utils.grid2grid(fgrid, cmodel, cgrid, 'cubic')

    # Compare
    assert np.max(np.abs((lin_model-cub_model)/lin_model*100)) < 1.0

    # Assert it is 'nearest' or extrapolate if points are outside.
    tgrid = utils.TensorMesh(
            [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]),
             np.array([1, 1, 1, 1])], x0=np.array([0., 0, 0]))
    tmodel = np.ones(tgrid.nC).reshape(tgrid.vnC, order='F')
    tmodel[:, 0, :] = 2
    t2grid = utils.TensorMesh(
            [np.array([1]), np.array([1]), np.array([1])],
            x0=np.array([2, -1, 2]))

    # Nearest with cubic.
    out = utils.grid2grid(tgrid, tmodel, t2grid, 'cubic')
    assert_allclose(out, 2.)

    # Same, but with log.
    vlog = utils.grid2grid(tgrid, tmodel, t2grid, 'cubic', log=True)
    vlin = utils.grid2grid(tgrid, np.log10(tmodel), t2grid, 'cubic')
    assert_allclose(vlog, 10**vlin)

    # Extrapolate with linear.
    out = utils.grid2grid(tgrid, tmodel, t2grid, 'linear')
    assert_allclose(out, 3.)

    # Same, but with log.
    vlog = utils.grid2grid(tgrid, tmodel, t2grid, 'linear', log=True)
    vlin = utils.grid2grid(tgrid, np.log10(tmodel), t2grid, 'linear')
    assert_allclose(vlog, 10**vlin)

    # Assert it is 0 if points are outside.
    out = utils.grid2grid(tgrid, tmodel, t2grid, 'cubic', False)
    assert_allclose(out, 0.)
    out = utils.grid2grid(tgrid, tmodel, t2grid, 'linear', False)
    assert_allclose(out, 0.)

    # Provide a Field instance
    grid = utils.TensorMesh(
            [np.array([1, 2]), np.array([1, 2]), np.array([1, 2])],
            [0, 0, 0])
    cgrid = utils.TensorMesh(
            [np.array([1.5, 1]), np.array([1.5]), np.array([1.5])],
            [0, 0, 0])
    field = utils.Field(grid)

    # Simple linear interpolation test.
    field.fx = np.arange(1, field.fx.size+1)
    field.fy = np.arange(1, field.fy.size+1)
    field.fz = np.arange(1, field.fz.size+1)

    new_field = utils.grid2grid(grid, field, cgrid, method='linear')
    fx = utils.grid2grid(grid, field.fx, cgrid, method='linear')
    fy = utils.grid2grid(grid, field.fy, cgrid, method='linear')
    fz = utils.grid2grid(grid, field.fz, cgrid, method='linear')
    assert_allclose(fx, new_field.fx)
    assert_allclose(fy, new_field.fy)
    assert_allclose(fz, new_field.fz)

    new_field = utils.grid2grid(grid, field, cgrid, method='cubic')
    fx = utils.grid2grid(grid, field.fx, cgrid, method='cubic')
    fy = utils.grid2grid(grid, field.fy, cgrid, method='cubic')
    fz = utils.grid2grid(grid, field.fz, cgrid, method='cubic')
    assert_allclose(fx, new_field.fx)
    assert_allclose(fy, new_field.fy)
    assert_allclose(fz, new_field.fz)

    # Ensure Field fails with 'volume'.
    with pytest.raises(ValueError):
        utils.grid2grid(grid, field, cgrid, method='volume')


# TIME DOMAIN
class TestFourier:
    def test_defaults(self, capsys):
        time = np.logspace(-2, 2)
        fmin = 0.01
        fmax = 100

        Fourier = utils.Fourier(time, fmin, fmax)
        out, _ = capsys.readouterr()

        # Check representation of Fourier.
        assert 'ffht' in Fourier.__repr__()
        assert '0.01-100.0 s' in Fourier.__repr__()
        assert '0.01-100 Hz' in Fourier.__repr__()

        assert Fourier.every_x_freq is None
        assert Fourier.fmin == fmin
        assert Fourier.fmax == fmax
        assert Fourier.ft == 'ffht'
        assert Fourier.ftarg[1] == -1.0   # Convolution DLF
        assert Fourier.ftarg[2] == 'sin'  # Sine-DLF is default
        assert Fourier.signal == 0        # Impulse respons
        assert_allclose(time, Fourier.time, 0, 0)
        assert Fourier.verb == 3          # Verbose by default
        assert 'Key 201 CosSin (2012)' in out
        assert 'Req. freq' in out
        assert 'Calc. freq' in out
        assert Fourier.freq_calc.min() >= fmin
        assert Fourier.freq_calc.max() <= fmax

        # Check frequencies to extrapolate.
        assert_allclose(Fourier.freq_extrapolate,
                        Fourier.freq_req[Fourier.freq_req < fmin])

        # If not freq_inp nor every_x_freq, interpolate and calc have to be the
        # same.
        assert_allclose(Fourier.freq_interpolate, Fourier.freq_calc)

        # Change time, ensure it changes required frequencies.
        freq_req = Fourier.freq_req
        time2 = np.logspace(-1, 2)
        Fourier.time = time2
        assert freq_req.size != Fourier.freq_req.size

    def test_kwargs(self, capsys):
        time = np.logspace(-1, 1)
        fmin = 0.1
        fmax = 10
        freq_inp = np.logspace(-1, 1, 11)
        xfreq = 10

        # freq_inp; verb=0
        _, _ = capsys.readouterr()
        Fourier1 = utils.Fourier(time, fmin, fmax, freq_inp=freq_inp, verb=0)
        out, _ = capsys.readouterr()
        assert '' == out
        assert_allclose(freq_inp, Fourier1.freq_calc, 0, 0)

        # freq_inp AND every_x_freq => re-sets every_x_freq.
        Fourier2 = utils.Fourier(time, fmin, fmax, every_x_freq=xfreq,
                                 freq_inp=freq_inp, verb=1)
        out, _ = capsys.readouterr()
        assert 'Re-setting `every_x_freq=None`' in out
        assert_allclose(freq_inp, Fourier2.freq_calc, 0, 0)
        assert_allclose(Fourier1.freq_calc, Fourier2.freq_calc)

        # Now set every_x_freq again => re-sets freq_inp.
        Fourier2.every_x_freq = xfreq
        out, _ = capsys.readouterr()
        assert 'Re-setting `freq_inp=None`' in out
        assert_allclose(Fourier2.freq_coarse, Fourier2.freq_req[::xfreq])
        assert Fourier2.freq_inp is None
        test = Fourier2.freq_req[::xfreq][
                (Fourier2.freq_req[::xfreq] >= fmin) &
                (Fourier2.freq_req[::xfreq] <= fmax)]
        assert_allclose(Fourier2.freq_calc, test)

        # And back
        Fourier2.freq_inp = freq_inp
        out, _ = capsys.readouterr()
        assert 'Re-setting `every_x_freq=None`' in out
        assert_allclose(Fourier2.freq_calc, freq_inp)
        assert Fourier2.every_x_freq is None

        # Unknown argument, must fail with TypeError.
        with pytest.raises(TypeError):
            utils.Fourier(time, fmin, fmax, does_not_exist=0)

    def test_setters(self, capsys):
        time = np.logspace(-1.4, 1.4)
        fmin = 0.1
        fmax = 10

        # freq_inp; verb=0
        _, _ = capsys.readouterr()
        Fourier1 = utils.Fourier(time, fmin=np.pi/10, fmax=np.pi*10)
        Fourier1.fmin = fmin
        Fourier1.fmax = fmax
        Fourier1.signal = -1
        Fourier1.fourier_arguments('fftlog', {'pts_per_dec': 5})
        assert Fourier1.ft == 'fftlog'
        assert Fourier1.ftarg[0] == 5
        assert Fourier1.ftarg[3] == -0.5  # cosine, as signal == -1

    def test_interpolation(self, capsys):
        time = np.logspace(-2, 1, 201)
        model = {'src': [0, 0, 0], 'rec': [900, 0, 0], 'res': 1,
                 'depth': [], 'verb': 1}
        Fourier = utils.Fourier(time, 0.005, 10)

        # Calculate data.
        data_true = empymod.dipole(freqtime=Fourier.freq_req, **model)
        data = empymod.dipole(freqtime=Fourier.freq_calc, **model)

        # Interpolate.
        data_int = Fourier.interpolate(data)

        # Compare, extrapolate < 0.05; interpolate equal.
        assert_allclose(data_int[Fourier.freq_extrapolate_i].imag,
                        data_true[Fourier.freq_extrapolate_i].imag, rtol=0.05)
        assert_allclose(data_int[Fourier.freq_calc_i].imag,
                        data_true[Fourier.freq_calc_i].imag)

        # Now set every_x_freq and again.
        Fourier.every_x_freq = 2

        data = empymod.dipole(freqtime=Fourier.freq_calc, **model)

        # Interpolate.
        data_int = Fourier.interpolate(data)

        # Compare, extrapolate < 0.05; interpolate < 0.01.
        assert_allclose(data_int[Fourier.freq_extrapolate_i].imag,
                        data_true[Fourier.freq_extrapolate_i].imag, rtol=0.05)
        assert_allclose(data_int[Fourier.freq_interpolate_i].imag,
                        data_true[Fourier.freq_interpolate_i].imag, rtol=0.01)

    def test_freq2transform(self, capsys):
        time = np.linspace(0.1, 10, 101)
        x = 900
        model = {'src': [0, 0, 0], 'rec': [x, 0, 0], 'res': 1,
                 'depth': [], 'verb': 1}

        # Initiate Fourier instance.
        Fourier = utils.Fourier(time, 0.001, 100)

        # Calculate required frequencies.
        data = empymod.dipole(freqtime=Fourier.freq_calc, **model)

        # Transform the data.
        tdata = Fourier.freq2time(data, x)

        # Calculate data in empymod.
        data_true = empymod.dipole(freqtime=time, signal=0, **model)

        # Compare.
        assert_allclose(data_true, tdata, rtol=1e-4)


# FUNCTIONS RELATED TO TIMING
def test_Time():
    t0 = default_timer()  # Create almost at the same time a
    time = utils.Time()   # t0-stamp and a Time-instance.

    # Ensure they are the same.
    assert_allclose(t0, time.t0, atol=1e-3)

    # Ensure `now` is a string of numbers and :.
    out = time.now
    assert re.match(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]', out)

    # This should have taken less then 1s.
    out = time.runtime
    assert "0:00:00" == str(out)

    # Check representation of Time.
    assert 'Runtime : 0:00:0' in time.__repr__()


# FUNCTIONS RELATED TO DATA MANAGEMENT
def test_data_write_read(tmpdir, capsys):
    # Create test data
    grid = utils.TensorMesh(
            [np.array([100, 4]), np.array([100, 8]), np.array([100, 16])],
            np.zeros(3))

    freq = np.pi

    model = utils.Model(grid, res_x=1., res_y=2., res_z=3., mu_r=4.)

    e1 = create_dummy(*grid.vnEx)
    e2 = create_dummy(*grid.vnEy)
    e3 = create_dummy(*grid.vnEz)
    ee = utils.Field(e1, e2, e3, freq=freq)

    # Write and read data, single arguments
    utils.data_write('testthis', 'ee', ee, tmpdir, -1)
    ee_out = utils.data_read('testthis', 'ee', tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    assert_allclose(ee.smu0, ee_out.smu0)
    assert_allclose(ee.sval, ee_out.sval)
    assert_allclose(ee.freq, ee_out.freq)

    # Write and read data, multi arguments
    args = ('grid', 'ee', 'model')
    utils.data_write('testthis', args, (grid, ee, model), tmpdir, -1)
    grid_out, ee_out, model_out = utils.data_read('testthis', args, tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(grid_out, attr)

    # Ensure volume averages got deleted or do not exist anyway.
    assert hasattr(grid_out, '_vol') is False
    assert hasattr(model_out, '_eta_x') is False
    assert hasattr(model_out, '_zeta') is False

    # Ensure they can be reconstructed
    assert_allclose(grid.vol, grid_out.vol)

    # Write and read data, None
    utils.data_write('testthis', ('grid', 'ee'), (grid, ee), tmpdir, -1)
    out = utils.data_read('testthis', path=tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(out['grid'], attr)

    # Test exists-argument 0
    _, _ = capsys.readouterr()  # Clean-up
    utils.data_write('testthis', 'ee', ee*2, tmpdir, 0)
    out, _ = capsys.readouterr()
    datout = utils.data_read('testthis', path=tmpdir)
    assert 'NOT SAVING THE DATA' in out
    assert_allclose(datout['ee'], ee)

    # Test exists-argument 1
    utils.data_write('testthis', 'ee2', ee, tmpdir, 1)
    out, _ = capsys.readouterr()
    assert 'appending to it' in out
    utils.data_write('testthis', ['ee', 'ee2'], [ee*2, ee], tmpdir, 1)
    out, _ = capsys.readouterr()
    assert "overwriting existing key(s) 'ee', 'ee2'" in out
    datout = utils.data_read('testthis', path=tmpdir)
    assert_allclose(datout['ee'], ee*2)
    assert_allclose(datout['ee2'], ee)

    # Check if file is missing.
    os.remove(tmpdir+'/testthis.dat')
    out = utils.data_read('testthis', path=tmpdir)
    assert out is None
    out1, out2 = utils.data_read('testthis', ['ee', 'ee2'], path=tmpdir)
    assert out1 is None
    assert out2 is None
    utils.data_write('testthis', ['ee', 'ee2'], [ee*2, ee], tmpdir, -1)


# OTHER
def test_report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    # Reporting is now done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    if scooby:
        out1 = utils.Report()
        out2 = scooby.Report(
                core=['numpy', 'scipy', 'numba', 'emg3d'],
                optional=['IPython', 'matplotlib'],
                ncol=4)

        # Ensure they're the same; exclude time to avoid errors.
        assert out1.__repr__()[115:] == out2.__repr__()[115:]

    else:  # soft dependency
        _ = utils.Report()
        out, _ = capsys.readouterr()  # Empty capsys
        assert 'WARNING :: `emg3d.Report` requires `scooby`' in out
