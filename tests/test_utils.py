import re
import os
import pytest
import numpy as np
from scipy import constants
from timeit import default_timer
from os.path import join, dirname
from numpy.testing import assert_allclose

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
    assert utils.mu_0 == constants.mu_0            # Check constants
    assert utils.epsilon_0 == constants.epsilon_0  # Check constants
    assert_allclose(model1.res_x, model1.res_y)
    assert_allclose(model1.nC, grid.nC)
    assert_allclose(model1.vnC, grid.vnC)

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
    model2c = utils.Model(grid, 2., res_z=model2b.res_z.copy())
    assert_allclose(model2c.res_x, model2c.res_z)
    assert_allclose(model2c.eta_x, model2c.eta_z)

    # HTI: Setting res_x and res_y, not res_z
    model2d = utils.Model(grid, 2., 4.)
    assert_allclose(model2d.res_x, model2d.res_z)
    assert_allclose(model2d.eta_x, model2d.eta_z)

    # Check wrong shape
    with pytest.raises(ValueError):
        model1.res_x = model1.res_x.ravel('F')
    with pytest.raises(ValueError):
        utils.Model(grid, np.arange(1, 11))
    with pytest.raises(ValueError):
        utils.Model(grid, res_y=np.ones((2, 5, 6)))
    with pytest.raises(ValueError):
        utils.Model(grid, res_z=np.array([1, 3]))

    # Check with all inputs
    model3 = utils.Model(grid, res_x, res_y, res_z, freq=1.234, mu_r=res_x)
    assert_allclose(model3.res_x, model3.res_y*2)
    assert_allclose(model3.res_x.shape, grid.vnC)
    assert_allclose(model3.res_x, model3.res_z/1.4)
    assert_allclose(model3._Model__vol/res_x, model3.v_mu_r)
    # Check with all inputs
    model3b = utils.Model(grid, res_x.ravel('F'), res_y.ravel('F'),
                          res_z.ravel('F'), freq=1.234, mu_r=res_y.ravel('F'))
    assert_allclose(model3b.res_x, model3b.res_y*2)
    assert_allclose(model3b.res_x.shape, grid.vnC)
    assert_allclose(model3b.res_x, model3b.res_z/1.4)
    assert_allclose(model3b._Model__vol/res_y, model3b.v_mu_r)

    # Check setters vnC
    tres = np.ones(grid.vnC)
    model3.res_x = tres*2.0
    model3.res_y = tres*3.0
    model3.res_z = tres*4.0
    assert_allclose(tres*2., model3.res_x)
    assert_allclose(tres*3., model3.res_y)
    assert_allclose(tres*4., model3.res_z)

    # Check eta
    iommu = 2j*np.pi*model3.freq*utils.mu_0
    iomep = 2j*np.pi*model3.freq*utils.epsilon_0
    eta_x = iommu*(1./model3.res_x - iomep)*model3._Model__vol
    eta_y = iommu*(1./model3.res_y - iomep)*model3._Model__vol
    eta_z = iommu*(1./model3.res_z - iomep)*model3._Model__vol
    assert_allclose(model3.eta_x, eta_x)
    assert_allclose(model3.eta_y, eta_y)
    assert_allclose(model3.eta_z, eta_z)

    # Check volume
    vol = np.outer(np.outer(grid.hx, grid.hy).ravel('F'), grid.hz)
    vol = vol.ravel('F').reshape(grid.vnC, order='F')
    assert_allclose(vol, model2.v_mu_r)
    grid.vol = vol
    model4 = utils.Model(grid, 1, freq=1)
    assert_allclose(model4.v_mu_r, vol)


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

    # Provide Field instance instead of Field.f{x/y/z}:
    with pytest.raises(ValueError):
        utils.get_receiver(grid, field, (1, 1, 1))

    # Provide wrong rec_loc input:
    with pytest.raises(ValueError):
        utils.get_receiver(grid, field.fx, (1, 1))

    # Simple linear interpolation test.
    field.fx = np.arange(1, field.fx.size+1)
    field = field.real  # For simplicity
    out1 = utils.get_receiver(grid, field.fx, ([0.5, 1, 2], 0, 0), 'linear')
    assert_allclose(out1, [1., 1+1/3, 2])
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

    # Extrapolate with linear.
    out = utils.grid2grid(tgrid, tmodel, t2grid, 'linear')
    assert_allclose(out, 3.)

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


# FUNCTIONS RELATED TO DATA MANAGEMENT
def test_data_write_read(tmpdir, capsys):
    # Create test data
    grid = utils.TensorMesh(
            [np.array([100, 4]), np.array([100, 8]), np.array([100, 16])],
            np.zeros(3))

    e1 = create_dummy(*grid.vnEx)
    e2 = create_dummy(*grid.vnEy)
    e3 = create_dummy(*grid.vnEz)
    ee = utils.Field(e1, e2, e3)

    # Write and read data, single arguments
    utils.data_write('testthis', 'ee', ee, tmpdir, -1)
    ee_out = utils.data_read('testthis', 'ee', tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)

    # Write and read data, multi arguments
    utils.data_write('testthis', ('grid', 'ee'), (grid, ee), tmpdir, -1)
    grid_out, ee_out = utils.data_read('testthis', ('grid', 'ee'), tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(grid_out, attr)

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


def test_interp3d():
    pass
