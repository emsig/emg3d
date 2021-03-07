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


def test_TensorMesh():
    # Load mesh created with discretize.TensorMesh.
    grid = REGRES['grid']
    grid['h'] = [grid.pop('hx'), grid.pop('hy'), grid.pop('hz')]

    mesh = meshes._TensorMesh(grid['h'], origin=grid['origin'])

    # Use this grid instance to create emg3d equivalent.
    emg3dgrid = meshes.TensorMesh(grid['h'], origin=grid['origin'])

    # Ensure they are the same.
    for key, value in grid.items():
        if key == 'h':
            for i in range(3):
                assert_allclose(value[i], getattr(emg3dgrid, key)[i])
        else:
            assert_allclose(value, getattr(emg3dgrid, key))

    # Copy
    cgrid = emg3dgrid.copy()
    assert_allclose(cgrid.cell_volumes, emg3dgrid.cell_volumes)
    dgrid = emg3dgrid.to_dict()
    cdgrid = meshes.TensorMesh.from_dict(dgrid)
    assert_allclose(cdgrid.cell_volumes, emg3dgrid.cell_volumes)
    del dgrid['hx']
    with pytest.raises(KeyError, match="Variable 'hx' missing in `inp`"):
        meshes.TensorMesh.from_dict(dgrid)

    # Check __eq__.
    assert emg3dgrid == cgrid
    newgrid = meshes.TensorMesh(
            [np.ones(3), np.ones(3), np.ones(3)], np.zeros(3))
    assert emg3dgrid != newgrid

    assert 'TensorMesh: 40 x 30 x 1 (1,200)' in mesh.__repr__()
    assert not hasattr(mesh, '_repr_html_')

    assert mesh.cell_volumes.sum() > 69046392


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
    t1 = meshes.skin_depth(1/np.pi, 2.0, 2.0)
    assert t1 == 0.5

    t2 = meshes.skin_depth(1/np.pi, 1/mu_0)
    assert t2 == 1

    t3 = meshes.skin_depth(-1/(2*np.pi**2), 2.0, 2.0)
    assert t3 == 0.5


def test_wavelength():
    t1 = meshes.wavelength(0.5)
    assert_allclose(t1, np.pi)


def test_min_cell_width():
    t1 = meshes.min_cell_width(1, pps=1)
    assert t1 == 1

    t2 = meshes.min_cell_width(1)
    assert np.round(t2, 3) == 0.333

    t3 = meshes.min_cell_width(503.0, limits=10)
    assert t3 == 10

    t3 = meshes.min_cell_width(503.0, limits=[100, 120])
    assert t3 == 120


class TestGetOriginWidths:

    def test_errors(self, capsys):
        with pytest.raises(TypeError, match='Unexpected '):
            meshes.get_origin_widths(1, 1, 0, [-1, 1], unknown=True)

        with pytest.raises(ValueError, match="At least one of `domain`, `d"):
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
        assert "No suitable grid found; relax your criteria." in outstr

        # Stretching warning.
        meshes.get_origin_widths(
                1/np.pi, 9*mu_0, -0.2, [-1, 2], stretching=[1, 1],
                seasurface=1.2, verb=3)
        out, _ = capsys.readouterr()
        assert "Note: Stretching in DS >> 1.0.\nThe reason " in out

    def test_basics(self, capsys):
        x0, hx = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1], verb=1)
        out, _ = capsys.readouterr()

        assert_allclose(x0, -20)
        assert_allclose(np.ones(40), hx)

        assert "Skin depth     [m] : 3.0  [corr." in out
        assert "Survey dom. DS [m] : -1.0 - 1.0" in out
        assert "Comp. dom. DC  [m] : -19.8 - 19.8" in out
        assert "Final extent   [m] : -20.0 - 20.0" in out
        assert "Cell widths    [m] : 1.0 / 1.0 / 1.0  [min(DS) / m" in out
        assert "Number of cells    : 40 (4 / 36 / 0)  [Total (DS/" in out
        assert "Max stretching     : 1.000 (1.000) / 1.000  [DS (" in out

        _ = meshes.get_origin_widths(
                1/np.pi, [8.9*mu_0, 9*mu_0], 0.0, [-1, 1],
                stretching=[1, 1], verb=1)
        out, _ = capsys.readouterr()

        assert "2.98 / 3.00  [corr. to `properties`]" in out

        _ = meshes.get_origin_widths(
                1/np.pi, [8.9*mu_0, 9*mu_0, 9.1*mu_0], 0.0, [-1, 1],
                stretching=[1, 1], verb=1)
        out, _ = capsys.readouterr()

        assert "2.98 / 3.00 / 3.02  [corr. to `properties`]" in out

    def test_domain_vector(self):
        x01, hx1 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1])
        x02, hx2 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, vector=np.array([-1, 0, 1]),
                stretching=[1, 1])
        assert_allclose(x01, x02)
        assert_allclose(hx1, hx2)

        x03, hx3 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, distance=[1, 1], stretching=[1, 1])
        assert_allclose(x01, x03)
        assert_allclose(hx1, hx3)

    def test_seasurface(self):
        x01, hx1 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1])
        x02, hx2 = meshes.get_origin_widths(
                1/np.pi, 9*mu_0, -0.5, [-1, 0], seasurface=0.0,
                stretching=[1, 1])
        assert_allclose(x01, x02)
        assert_allclose(hx1, hx2)

    def test_status_quo_with_all(self, capsys):
        # Defaults.
        meshes.get_origin_widths(
            frequency=0.2,
            properties=[0.3, 1, 50],
            center=-950,
            domain=[-2000, -1000],
            verb=1,
            )

        out, _ = capsys.readouterr()

        assert "Skin depth     [m] : 616 / 1125 / 7958" in out
        assert "Survey dom. DS [m] : -2000 - -1000" in out
        assert "Comp. dom. DC  [m] : -9071 - 49000" in out
        assert "Final extent   [m] : -10310 - 52091" in out
        assert "Cell widths    [m] : 205 / 205 / 12083" in out
        assert "Number of cells    : 32 (7 / 25 / 0)" in out
        assert "Max stretching     : 1.000 (1.000) / 1.290" in out

        # All set.
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
            lambda_from_center=True,
            max_buffer=10000,
            mapping='Conductivity',
            cell_numbers=[20, 40, 80, 160],
            verb=1,
            raise_error=False,
            )

        out, _ = capsys.readouterr()

        assert "Skin depth     [m] : 620 / 1125 / 50" in out
        assert "Survey dom. DS [m] : -2000 - -1000" in out
        assert "Comp. dom. DC  [m] : -10950 - 5300" in out
        assert "Final extent   [m] : -13945 - 5425" in out
        assert "Cell widths    [m] : 100 / 100 / 3191" in out
        assert "Number of cells    : 40 (20 / 20 / 0)" in out
        assert "Max stretching     : 1.000 (1.000) / 1.370" in out

        # High frequencies.
        meshes.get_origin_widths(
            frequency=1e8,
            properties=[5, 1, 50],
            center=0,
            domain=[-1, 1],
            verb=1,
            )

        out, _ = capsys.readouterr()

        assert "Skin depth     [m] : 0.113 / 0.050 / 0.356" in out
        assert "Survey dom. DS [m] : -1.000 - 1.000" in out
        assert "Comp. dom. DC  [m] : -1.316 - 3.236" in out
        assert "Final extent   [m] : -1.331 - 3.376" in out
        assert "Cell widths    [m] : 0.038 / 0.038 / 0.252" in out
        assert "Number of cells    : 80 (54 / 26 / 0)" in out
        assert "Max stretching     : 1.000 (1.000) / 1.100" in out


class TestConstructMesh:

    def test_verb(self, capsys):
        mesh = meshes.construct_mesh(1.0, 1.0, (0, 0, 0), [-1, 1], verb=1)
        out, _ = capsys.readouterr()
        assert "         == GRIDDING IN X ==" in out
        assert "         == GRIDDING IN X ==" in mesh.construct_mesh_info

        mesh = meshes.construct_mesh(1.0, 1.0, (0, 0, 0), [-1, 1], verb=0)
        out, _ = capsys.readouterr()
        assert "" == out
        assert "         == GRIDDING IN X ==" in mesh.construct_mesh_info

        mesh = meshes.construct_mesh(1.0, 1.0, (0, 0, 0), [-1, 1], verb=-1)
        out, _ = capsys.readouterr()
        assert "" == out
        assert "         == GRIDDING IN X ==" in mesh.construct_mesh_info

        # No suitable grid warning.
        with pytest.raises(RuntimeError, match="No suitable grid found; "):
            meshes.construct_mesh(1, 1, (0, 0, 0), [-1, 1], cell_numbers=[1, ])

    def test_compare_to_gow1(self):
        f = 1/np.pi
        p = 9*mu_0
        c = (1, 2, 3)
        d = [-1, 1]
        x0, hx = meshes.get_origin_widths(f, p, c[0], d, stretching=[1, 1.3])
        y0, hy = meshes.get_origin_widths(f, p, c[1], d, stretching=[1.5, 1])
        z0, hz = meshes.get_origin_widths(f, p, c[2], d, stretching=[1, 1])
        m = meshes.construct_mesh(
                f, [p, p, p, p], c, d, stretching=([1, 1.3], [1.5, 1], [1, 1]))

        assert_allclose(m.origin, (x0, y0, z0))
        assert_allclose(m.h[0], hx)
        assert_allclose(m.h[1], hy)
        assert_allclose(m.h[2], hz)

    def test_compare_to_gow2(self):
        vz = np.arange(100)[::-1]*-20
        x0, hx = meshes.get_origin_widths(
                0.77, [0.3, 1, 2], 0, [-1000, 1000], min_width_limits=[20, 40])
        y0, hy = meshes.get_origin_widths(
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

        assert_allclose(m.origin, (x0, y0, z0))
        assert_allclose(m.h[0], hx)
        assert_allclose(m.h[1], hy)
        assert_allclose(m.h[2], hz)

    def test_compare_to_gow3(self):
        x0, hx = meshes.get_origin_widths(
                0.2, [1, 1], -423, [-3333, 222], min_width_limits=20)
        y0, hy = meshes.get_origin_widths(
                0.2, [1.0, 2.0], 16, [-1234, 8956], min_width_limits=20)
        z0, hz = meshes.get_origin_widths(
                0.2, [1.0, 3.0], -33.3333, [-100, 100], min_width_limits=20)
        m = meshes.construct_mesh(
                frequency=0.2,
                properties=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                center=(-423, 16, -33.3333),
                domain=([-3333, 222], [-1234, 8956], [-100, 100]),
                min_width_limits=20,
                )

        assert_allclose(m.origin, (x0, y0, z0), atol=1e-3)
        assert_allclose(m.h[0], hx)
        assert_allclose(m.h[1], hy)
        assert_allclose(m.h[2], hz)

    def test_compare_to_gow4(self):
        inp = {'frequency': 1.234, 'center': (0, 0, 0),
               'domain': ([-100, 100], [-100, 100], [-100, 100])}
        m3 = meshes.construct_mesh(properties=[0.3, 1.0, 1e8], **inp)
        m4 = meshes.construct_mesh(properties=[0.3, 1e8, 1.0, 1e8], **inp)

        assert_allclose(m3.origin, m4.origin)
        assert_allclose(m3.h[0], m4.h[0])
        assert_allclose(m3.h[1], m4.h[1])
        assert_allclose(m3.h[2], m4.h[2])


def test_good_mg_cell_numbers(capsys):
    numbers = meshes.good_mg_cell_nr(max_nr=128, max_prime=5, min_div=3)
    assert_allclose([16, 24, 32, 40, 48, 64, 80, 96, 128], numbers)

    with pytest.raises(ValueError, match='Highest prime is 25, '):
        numbers = meshes.good_mg_cell_nr(max_nr=128, max_prime=25, min_div=3)

    numbers = meshes.good_mg_cell_nr(max_nr=50, max_prime=3, min_div=5)
    assert len(numbers) == 0
