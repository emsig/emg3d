import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from emg3d import meshes, io

# Data generated with create_data/regression.py
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


def test_get_hx_h0(capsys):

    # == A == Just the defaults, no big thing (regression).
    out1 = meshes.get_hx_h0(
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
    out2 = meshes.get_hx_h0(
            -.5/np.pi/2, 10, [-2000, 2000], 900, [20, 32], None, 3,
            [1.05, 1.5, 0.01], 100000., False, 0, True)
    outstr2, _ = capsys.readouterr()

    # Assert they are the same.
    assert_allclose(out1[0], out2[0])

    # Assert nothing is printed with verb=0.
    assert outstr2 == ""

    # == C == User limits.
    out3 = meshes.get_hx_h0(
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
        meshes.get_hx_h0(
                freq=.5, res=[10., 12.], fixed=900, domain=[-10000, 10000],
                possible_nx=[20])

    # (b) With raise=False.
    out4 = meshes.get_hx_h0(
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
        meshes.get_hx_h0(
            freq=1, res=1, fixed=[-900, -1000, 0, 5], domain=[-2000, 0],
            possible_nx=[64, 128])
    # Two additional values, but both on same side.
    with pytest.raises(ValueError):
        meshes.get_hx_h0(
            freq=1, res=1, fixed=[900, -1000, -1200], domain=[-2000, 0],
            possible_nx=[64, 128])

    # One additional fixed.
    out5 = meshes.get_hx_h0(
            freq=1, res=1, fixed=[-900, 0], domain=[-2000, 0],
            possible_nx=[64, 128], min_width=[50, 100],
            alpha=[1., 1, 1, 1], return_info=True)
    outstr5, _ = capsys.readouterr()

    nodes5 = out5[1]+np.cumsum(out5[0])
    assert_allclose(0.0, min(abs(nodes5)), atol=1e-8)  # Check sea-surface.
    assert out5[2]['amax'] < 1.02

    # Two additional fixed.
    out6 = meshes.get_hx_h0(
            freq=1, res=1, fixed=[-890, 0, -1000], domain=[-2000, 0],
            possible_nx=[64, 128], min_width=[60, 70])
    outstr6, _ = capsys.readouterr()
    nodes6 = out6[1]+np.cumsum(out6[0])
    assert_allclose(0.0, min(abs(nodes6)), atol=1e-8)  # Check sea-surface.
    assert_allclose(0.0, min(abs(nodes6+1000)), atol=1e-8)  # Check seafloor.

    # == F == Several resistivities
    out7 = meshes.get_hx_h0(1, [0.3, 10], [-1000, 1000], alpha=[1, 1, 1])
    assert out7[1] < -10000
    assert out7[1]+np.sum(out7[0]) > 10000

    out8 = meshes.get_hx_h0(1, [0.3, 1, 90], [-1000, 1000], alpha=[1, 1, 1])
    assert out8[1] > -5000                  # Left buffer much smaller than
    assert out8[1]+np.sum(out8[0]) > 30000  # right buffer.


def test_get_domain():
    # Test default values (and therefore skindepth etc)
    h1, d1 = meshes.get_domain()
    assert_allclose(h1, 55.133753)
    assert_allclose(d1, [-1378.343816, 1378.343816])

    # Ensure fact_min/fact_neg/fact_pos
    h2, d2 = meshes.get_domain(fact_min=1, fact_neg=10, fact_pos=20)
    assert h2 == 5*h1
    assert 2*d1[0] == d2[0]
    assert -2*d2[0] == d2[1]

    # Check limits and min_width
    h3, d3 = meshes.get_domain(limits=[-10000, 10000], min_width=[1, 10])
    assert h3 == 10
    assert np.sum(d3) == 0

    h4, d4 = meshes.get_domain(limits=[-10000, 10000], min_width=5.5)
    assert h4 == 5.5
    assert np.sum(d4) == 0

    # Ensure Laplace and frequency
    h5a, d5a = meshes.get_domain(freq=1.)
    h5b, d5b = meshes.get_domain(freq=-1./2/np.pi)
    assert h5a == h5b
    assert d5a == d5b


def test_get_stretched_h(capsys):
    # Test min_space bigger (11) then required (10)
    h1 = meshes.get_stretched_h(11, [0, 100], nx=10)
    assert_allclose(np.ones(10)*10, h1)

    # Test with range, wont end at 100
    h2 = meshes.get_stretched_h(10, [-100, 100], nx=10, x0=20, x1=60)
    assert_allclose(np.ones(4)*10, h2[5:9])
    assert -100+np.sum(h2) != 100

    # Now ensure 100
    h3 = meshes.get_stretched_h(10, [-100, 100], nx=10, x0=20, x1=60,
                                resp_domain=True)
    assert -100+np.sum(h3) == 100

    out, _ = capsys.readouterr()  # Empty capsys
    _ = meshes.get_stretched_h(10, [-100, 100], nx=5, x0=20, x1=60)
    out, _ = capsys.readouterr()
    assert "Warning :: Not enough points for non-stretched part" in out


def test_get_cell_numbers(capsys):
    numbers = meshes.get_cell_numbers(max_nr=128, max_prime=5, min_div=3)
    assert_allclose([16, 24, 32, 40, 48, 64, 80, 96, 128], numbers)

    with pytest.raises(ValueError):
        numbers = meshes.get_cell_numbers(max_nr=128, max_prime=25, min_div=3)
    out, _ = capsys.readouterr()
    assert "* ERROR   :: Highest prime is 25" in out

    numbers = meshes.get_cell_numbers(max_nr=50, max_prime=3, min_div=5)
    assert len(numbers) == 0


def test_get_hx():
    # Test alpha <= 0
    hx1 = meshes.get_hx(-.5, [0, 10], 5, 3.33)
    assert_allclose(np.ones(5)*2, hx1)

    # Test x0 on domain
    hx2a = meshes.get_hx(0.1, [0, 10], 5, 0)
    assert_allclose(np.ones(4)*1.1, hx2a[1:]/hx2a[:-1])
    hx2b = meshes.get_hx(0.1, [0, 10], 5, 10)
    assert_allclose(np.ones(4)/1.1, hx2b[1:]/hx2b[:-1])
    assert np.sum(hx2b) == 10.0

    # Test resp_domain
    hx3 = meshes.get_hx(0.1, [0, 10], 3, 8, False)
    assert np.sum(hx3) != 10.0


def test_TensorMesh():
    # Load mesh created with discretize.TensorMesh.
    grid = REGRES['Data']['grid']

    # Use this grid instance to create emg3d equivalent.
    emg3dgrid = meshes.TensorMesh(
            [grid['hx'], grid['hy'], grid['hz']], grid['x0'])

    # Ensure they are the same.
    for key, value in grid.items():
        assert_allclose(value, getattr(emg3dgrid, key))

    # Copy
    cgrid = emg3dgrid.copy()
    assert_allclose(cgrid.vol, emg3dgrid.vol)
    dgrid = emg3dgrid.to_dict()
    cdgrid = meshes.TensorMesh.from_dict(dgrid)
    assert_allclose(cdgrid.vol, emg3dgrid.vol)
    del dgrid['hx']
    with pytest.raises(KeyError):
        meshes.TensorMesh.from_dict(dgrid)

    # Check __eq__.
    assert emg3dgrid == cgrid
    # Dummies to check __eq__.
    cgrid.vnC = cgrid.vnC[:2]
    cgrid = emg3dgrid.copy()
    cgrid.hx = cgrid.hx*2
    assert emg3dgrid != cgrid
    cgrid = emg3dgrid.copy()
    cgrid.vnC = np.array([99, 1, 1])
    assert emg3dgrid != cgrid
