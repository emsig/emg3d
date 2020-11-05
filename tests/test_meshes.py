from os.path import join, dirname

import pytest
import numpy as np
from scipy.constants import mu_0
from numpy.testing import assert_allclose

from emg3d import meshes, io

# Import soft dependencies.
try:
    import discretize
except ImportError:
    discretize = None

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
        "   Computation domain  [m] : -14692 - 15592\n"
        "   Final extent        [m] : -15700 - 15999\n"
        f"   Min/max cell width  [m] : {out1[2]['dmin']:.0f} / 750 / 3382\n"
        "   Alpha survey/comp       : "
        f"{out1[2]['amin']:.3f} / {out1[2]['amax']:.3f}\n"
        "   Number of cells (s/c/r) : 20 (6/14/0)\n"
    )

    # Just check x0 and the output.
    assert out1[1] == -15699.582761656933
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
    # Computation domain has to be at least domain.
    assert out3[1]+np.sum(out3[0]) > 14000
    assert out3[1] <= -11000

    # == D == Check failure.
    # (a) With raise.
    with pytest.raises(RuntimeError, match='No suitable grid found; '):
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
    with pytest.raises(ValueError, match='Maximum three fixed boundaries'):
        meshes.get_hx_h0(
            freq=1, res=1, fixed=[-900, -1000, 0, 5], domain=[-2000, 0],
            possible_nx=[64, 128])
    # Two additional values, but both on same side.
    with pytest.raises(ValueError, match='2nd and 3rd fixed boundaries'):
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
    assert_allclose(h1, 55.1328)
    assert_allclose(d1, [-1378.32, 1378.32])

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

    with pytest.raises(ValueError, match='Highest prime is 25, '):
        numbers = meshes.get_cell_numbers(max_nr=128, max_prime=25, min_div=3)

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
    grid = REGRES['grid']

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
    with pytest.raises(KeyError, match="Variable 'hx' missing in `inp`"):
        meshes.TensorMesh.from_dict(dgrid)

    # Check __eq__.
    assert emg3dgrid == cgrid
    newgrid = meshes.TensorMesh(
            [np.ones(3), np.ones(3), np.ones(3)], np.zeros(3))
    assert emg3dgrid != newgrid


def test_TensorMesh_repr():
    # Create some dummy data
    grid = meshes.TensorMesh([np.ones(2), np.ones(2), np.ones(2)], np.zeros(3))

    # Check representation of TensorMesh.
    if discretize is None:
        assert 'TensorMesh: 2 x 2 x 2 (8)' in grid.__repr__()
        assert not hasattr(grid, '_repr_html_')
    else:
        assert 'TensorMesh: 8 cells' in grid.__repr__()
        assert hasattr(grid, '_repr_html_')


def test_skin_depth():
    t1 = meshes.skin_depth(1/np.pi, 2.0, 2.0, 20)
    assert t1 == 0.5

    t2 = meshes.skin_depth(1/np.pi, 2.0, 1.0, 0)
    assert t2 == 1

    t3 = meshes.skin_depth(1/np.pi, 1/mu_0)
    assert t3 == 1

    t4 = meshes.skin_depth(-1/(2*np.pi**2), 2.0, 2.0, 20)
    assert t4 == 0.5


def test_wavelength():
    t1 = meshes.wavelength(1/np.pi, 2.0, 2.0, 20)
    assert t1 == np.round(np.pi, 20)

    t2 = meshes.wavelength(1/np.pi, 2.0, 3.0, 0)
    assert t2 == 3

    t3 = meshes.wavelength(1/np.pi, 1/mu_0)
    assert t3 == 6.283

    t4 = meshes.wavelength(-1/(2*np.pi**2), 2.0, 2.0, 20)
    assert t4 == np.round(np.pi, 20)


def test_min_width_cell():
    t1 = meshes.min_cell_width(1/np.pi, 1/mu_0, pps=1)
    assert t1 == 1

    t2 = meshes.min_cell_width(1/np.pi, 1/mu_0)
    assert t2 == 1/3

    t3 = meshes.min_cell_width(1, 1, limits=10)
    assert t3 == 10

    t3 = meshes.min_cell_width(1, 1, limits=[100, 120])
    assert t3 == 120


class TestGetOriginWidths:

    def test_errors(self, capsys):
        with pytest.raises(TypeError, match='Unexpected '):
            meshes.get_origin_widths(1, 1, 0, [-1, 1], unknown=True)

        with pytest.raises(ValueError, match="At least one of `domain` and"):
            meshes.get_origin_widths(1, 1, 0)

        with pytest.raises(ValueError, match="Provided vector MUST at least"):
            meshes.get_origin_widths(1, 1, 0, [-1, 1], np.array([0, 1, 2]))

        with pytest.raises(ValueError, match="The `seasurface` but be bigger"):
            meshes.get_origin_widths(1, 1, 0, [-1, 1], seasurface=-2)

        # No suitable grid warning.
        with pytest.raises(RuntimeError, match="No suitable grid found; "):
            meshes.get_origin_widths(1, 1, 0, [-100, 100], cell_numbers=[1, ])

        out = meshes.get_origin_widths(
                1, 1, 0, [-100, 100], cell_numbers=[1, ], raise_error=False,
                verb=1)
        outstr, _ = capsys.readouterr()

        assert out[0] is None
        assert out[1] is None
        assert "* ERROR   :: No suitable grid found; relax your" in outstr

        # Stretching warning.
        meshes.get_origin_widths(
                1/np.pi, 9*mu_0, -0.2, [-1, 2], stretching=[1, 1],
                seasurface=1.2, verb=3)
        out, _ = capsys.readouterr()
        assert "Note: Stretching in DS >> 1.0.\nThe reason " in out

    def test_basics(self, capsys):
        x0, hx = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1], verb=2)
        out, _ = capsys.readouterr()

        assert x0 == -20
        assert_allclose(np.ones(40), hx)

        assert "Skin depth          [m] : 3  [correspond" in out
        assert "Survey domain DS    [m] : -1 - 1" in out
        assert "Comp. domain DC     [m] : -20 - 20" in out
        assert "Final extent        [m] : -20 - 20" in out
        assert "Cell widths         [m] : 1 / 1 / 1  [min(DS) / max(DS" in out
        assert "Number of cells         : 40 (2 / 38 / 0)  [Total (DS/" in out
        assert "Max stretching          : 1.000 (1.000) / 1.000  [DS (" in out

        _ = meshes.get_origin_widths(
                1/np.pi, [8.9*mu_0, 9*mu_0], 0.0, [-1, 1],
                stretching=[1, 1], verb=2)
        out, _ = capsys.readouterr()

        assert "3 / 3  [corresponding to `properties`]" in out

        _ = meshes.get_origin_widths(
                1/np.pi, [8.9*mu_0, 9*mu_0, 9.1*mu_0], 0.0, [-1, 1],
                stretching=[1, 1], verb=2)
        out, _ = capsys.readouterr()

        assert "3 / 3 / 3  [corresponding to `properties`]" in out

    def test_domain_vector(self):
        x01, hx1 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1])
        x02, hx2 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, vector=np.array([-1, 0, 1]),
                stretching=[1, 1])
        assert_allclose(x01, x02)
        assert_allclose(hx1, hx2)

    def test_seasurface(self):
        x01, hx1 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1])
        x02, hx2 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, -0.5, [-1, 0], seasurface=0.0,
                stretching=[1, 1])
        assert_allclose(x01, x02)
        assert_allclose(hx1, hx2)

    def test_status_quo_with_all(self, capsys):
        meshes.get_origin_widths(
            frequency=0.2,
            properties=[3.3, 1, 500],
            center=-950,
            domain=[-2000, -1000],
            vector=-np.arange(20)[::-1]*100-600,
            seasurface=-500,
            stretching=[1.2, 1.5],
            min_width_limits=[20, 500],
            min_width_pps=5,
            lambda_factor=20,
            max_buffer=10000,
            mapping='Conductivity',
            cell_numbers=[20, 40, 80, 160],
            verb=1,
            raise_error=False,
            )

        out, _ = capsys.readouterr()

        assert "Skin depth          [m] : 620 / 1125 / 50" in out
        assert "Survey domain DS    [m] : -2000 - -1000" in out
        assert "Comp. domain DC     [m] : -12000 - 5320" in out
        assert "Final extent        [m] : -13945 - 5425" in out
        assert "Cell widths         [m] : 100 / 100 / 3191" in out
        assert "Number of cells         : 40 (20 / 20 / 0)" in out
        assert "Max stretching          : 1.000 (1.000) / 1.370" in out


class TestConstructMesh:

    def test_verb(self, capsys):
        _ = meshes.construct_mesh(1.0, 1.0, (0, 0, 0), [-1, 1], verb=1)
        out, _ = capsys.readouterr()
        assert "         == GRIDDING IN X ==" in out

    def test_compare_to_gow1(self):
        f = 1/np.pi
        p = 9*mu_0
        c = (1, 2, 3)
        d = [-1, 1]
        x0, hx = meshes.get_origin_widths(f, p, c[0], d, stretching=[1, 1.3])
        y0, hz = meshes.get_origin_widths(f, p, c[1], d, stretching=[1.5, 1])
        z0, hz = meshes.get_origin_widths(f, p, c[2], d, stretching=[1, 1])
        m = meshes.construct_mesh(
                f, [p, p, p, p], c, d, stretching=([1, 1.3], [1.5, 1], [1, 1]))

        assert_allclose(m.x0, (x0, y0, z0))

    def test_compare_to_gow2(self):
        vz = np.arange(100)[::-1]*-20
        x0, hx = meshes.get_origin_widths(
                0.77, [0.3, 1, 2], 0, [-1000, 1000], min_width_limits=[20, 40])
        y0, hz = meshes.get_origin_widths(
                0.77, [0.3, 2, 1], 0, [-2000, 2000], min_width_limits=[20, 40])
        z0, hz = meshes.get_origin_widths(
                0.77, [0.3, 2, 1e8], 0, vector=vz, min_width_limits=[20, 40])
        m = meshes.construct_mesh(
                frequency=0.77,
                properties=[0.3, 1, 2, 2, 1, 2, 1e8],
                center=(0, 0, 0),
                domain=([-1000, 1000], [-2000, 2000], None),
                vector=(None, None, vz),
                min_width_limits=[20, 40],
                )

        assert_allclose(m.x0, (x0, y0, z0))
