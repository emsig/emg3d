from os.path import join, dirname

import pytest
import numpy as np
from scipy.constants import mu_0
from numpy.testing import assert_allclose

import emg3d
from emg3d import meshes

from . import helpers


# Import soft dependencies.
try:
    import discretize
except ImportError:
    discretize = None

try:
    import xarray
except ImportError:
    xarray = None

# Data generated with create_data/regression.py
REGRES = emg3d.load(join(dirname(__file__), 'data', 'regression.npz'))


def test_BaseMesh():
    hx = [1, 1]
    hy = [2, 2]
    hz = [3, 3]
    origin = (10, 100, 1000)
    grid = meshes.BaseMesh(h=[hx, hy, hz], origin=origin)

    assert_allclose(grid.origin, origin)
    assert_allclose(grid.h[0], hx)
    assert_allclose(grid.h[1], hy)
    assert_allclose(grid.h[2], hz)

    assert_allclose(grid.shape_nodes, [3, 3, 3])
    assert_allclose(grid.nodes_x, [10, 11, 12])
    assert_allclose(grid.nodes_y, [100, 102, 104])
    assert_allclose(grid.nodes_z, [1000, 1003, 1006])

    assert_allclose(grid.shape_cells, [2, 2, 2])
    assert_allclose(grid.n_cells, 8)
    assert_allclose(grid.cell_centers_x, [10.5, 11.5])
    assert_allclose(grid.cell_centers_y, [101, 103])
    assert_allclose(grid.cell_centers_z, [1001.5, 1004.5])

    assert_allclose(grid.shape_edges_x, [2, 3, 3])
    assert_allclose(grid.shape_edges_y, [3, 2, 3])
    assert_allclose(grid.shape_edges_z, [3, 3, 2])
    assert_allclose(grid.n_edges_x, 18)
    assert_allclose(grid.n_edges_y, 18)
    assert_allclose(grid.n_edges_z, 18)

    assert "TensorMesh: 2 x 2 x 2 (8)" in grid.__repr__()

    assert_allclose(grid.cell_volumes, np.ones(8)*6)


class TestTensorMesh:
    def test_TensorMesh(self):
        # Load mesh created with discretize.TensorMesh.
        grid = REGRES['grid']
        grid['h'] = [grid.pop('hx'), grid.pop('hy'), grid.pop('hz')]

        mesh = meshes.BaseMesh(grid['h'], origin=grid['origin'])

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
        cdgrid = meshes.TensorMesh.from_dict(dgrid.copy())
        assert_allclose(cdgrid.cell_volumes, emg3dgrid.cell_volumes)
        del dgrid['hx']
        with pytest.raises(KeyError, match="'hx'"):
            meshes.TensorMesh.from_dict(dgrid)

        # Check __eq__.
        assert emg3dgrid == cgrid
        newgrid = meshes.TensorMesh(
                [np.ones(3), np.ones(3), np.ones(3)], np.zeros(3))
        assert emg3dgrid != newgrid

        assert 'TensorMesh: 40 x 30 x 1 (1,200)' in mesh.__repr__()
        assert not hasattr(mesh, '_repr_html_')

        assert mesh.cell_volumes.sum() > 69046392

    def test_TensorMesh_repr(self):
        # Create some dummy data
        grid = meshes.TensorMesh(
                [np.ones(2), np.ones(2), np.ones(2)], np.zeros(3))

        # Check representation of TensorMesh.
        if discretize is None:
            assert 'TensorMesh: 2 x 2 x 2 (8)' in grid.__repr__()
            assert not hasattr(grid, '_repr_html_')
        else:
            assert 'TensorMesh: 8 cells' in grid.__repr__()
            assert hasattr(grid, '_repr_html_')


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
        x0, hx = meshes.origin_and_widths(f, p, c[0], d, stretching=[1, 1.3])
        y0, hy = meshes.origin_and_widths(f, p, c[1], d, stretching=[1.5, 1])
        z0, hz = meshes.origin_and_widths(f, p, c[2], d, stretching=[1, 1])
        m = meshes.construct_mesh(
                f, [p, p, p, p], c, d, stretching=([1, 1.3], [1.5, 1], [1, 1]))

        assert_allclose(m.origin, (x0, y0, z0))
        assert_allclose(m.h[0], hx)
        assert_allclose(m.h[1], hy)
        assert_allclose(m.h[2], hz)

        # As dict
        m = meshes.construct_mesh(
                f, [p, p, p, p], c, d,
                stretching={'x': [1, 1.3], 'y': [1.5, 1], 'z': [1, 1]})

        assert_allclose(m.origin, (x0, y0, z0))
        assert_allclose(m.h[0], hx)
        assert_allclose(m.h[1], hy)
        assert_allclose(m.h[2], hz)

    def test_compare_to_gow2(self):
        vz = np.arange(100)[::-1]*-20
        x0, hx = meshes.origin_and_widths(
                0.77, [0.3, 1, 2], 0, [-1000, 1000], min_width_limits=[20, 40])
        y0, hy = meshes.origin_and_widths(
                0.77, [0.3, 2, 1], 0, [-2000, 2000], min_width_limits=[20, 40])
        z0, hz = meshes.origin_and_widths(
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

        # As dict
        m = meshes.construct_mesh(
                frequency=0.77,
                properties=[0.3, 1, 2, 2, 1, 2, 1e8],
                center=(0, 0, 0),
                domain={'x': [-1000, 1000], 'y': [-2000, 2000], 'z': None},
                vector={'x': None, 'y': None, 'z': vz},
                min_width_limits=[20, 40],
                )

        assert_allclose(m.origin, (x0, y0, z0))
        assert_allclose(m.h[0], hx)
        assert_allclose(m.h[1], hy)
        assert_allclose(m.h[2], hz)

    def test_compare_to_gow3(self):
        x0, hx = meshes.origin_and_widths(
                0.2, [1, 1], -423, [-3333, 222], min_width_limits=20)
        y0, hy = meshes.origin_and_widths(
                0.2, [1.0, 2.0], 16, [-1234, 8956], min_width_limits=20)
        z0, hz = meshes.origin_and_widths(
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

        # As dict.
        m = meshes.construct_mesh(
                frequency=0.2,
                properties=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                center=(-423, 16, -33.3333),
                domain={'x': [-3333, 222], 'y': [-1234, 8956],
                        'z': [-100, 100]},
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


class TestOriginAndWidths:
    def test_errors(self, capsys):
        with pytest.raises(TypeError, match='Unexpected '):
            meshes.origin_and_widths(1, 1, 0, [-1, 1], unknown=True)

        with pytest.raises(ValueError, match="At least one of `domain`, `d"):
            meshes.origin_and_widths(1, 1, 0)

        with pytest.raises(ValueError, match="Provided vector MUST at least"):
            meshes.origin_and_widths(1, 1, 0, [-1, 1], np.array([0, 1, 2]))

        with pytest.raises(ValueError, match="The `seasurface` must be bigge"):
            meshes.origin_and_widths(1, 1, 0, [-1, 1], seasurface=-2)

        # No suitable grid warning.
        with pytest.raises(RuntimeError, match="No suitable grid found; "):
            meshes.origin_and_widths(1, 1, 0, [-100, 100], cell_numbers=[1, ])

        out = meshes.origin_and_widths(
                1, 1, 0, [-100, 100], cell_numbers=[1, ], raise_error=False,
                verb=1)
        outstr, _ = capsys.readouterr()

        assert out[0] is None
        assert out[1] is None
        assert "No suitable grid found; relax your criteria." in outstr

        # Stretching warning.
        meshes.origin_and_widths(
                1/np.pi, 9*mu_0, -0.2, [-1, 2], stretching=[1, 1],
                seasurface=1.2, verb=3)
        out, _ = capsys.readouterr()
        assert "Note: Stretching in DS >> 1.0.\nThe reason " in out

    def test_basics(self, capsys):
        x0, hx = meshes.origin_and_widths(
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

        _ = meshes.origin_and_widths(
                1/np.pi, [8.9*mu_0, 9*mu_0], 0.0, [-1, 1],
                stretching=[1, 1], verb=1)
        out, _ = capsys.readouterr()

        assert "2.98 / 3.00  [corr. to `properties`]" in out

        _ = meshes.origin_and_widths(
                1/np.pi, [8.9*mu_0, 9*mu_0, 9.1*mu_0], 0.0, [-1, 1],
                stretching=[1, 1], verb=1)
        out, _ = capsys.readouterr()

        assert "2.98 / 3.00 / 3.02  [corr. to `properties`]" in out

    def test_domain_vector(self):
        inp = {'frequency': 1/np.pi, 'properties': 9*mu_0, 'center': 0.0,
               'stretching': [1, 1]}
        x01, hx1 = meshes.origin_and_widths(domain=[-1, 1], **inp)
        x02, hx2 = meshes.origin_and_widths(vector=np.array([-1, 0, 1]), **inp)
        x03, hx3 = meshes.origin_and_widths(  # vector will be cut
                domain=[-1, 1], vector=np.array([-2, -1, 0, 1, 2]), **inp)
        x04, hx4 = meshes.origin_and_widths(  # vector will be cut
                distance=[1, 1], vector=np.array([-2, -1, 0, 1, 2]), **inp)
        assert_allclose(x01, x02)
        assert_allclose(x01, x03)
        assert_allclose(x01, x04)
        assert_allclose(hx1, hx2)
        assert_allclose(hx1, hx3)
        assert_allclose(hx1, hx4)

        x03, hx3 = meshes.origin_and_widths(
                1/np.pi, 9*mu_0, 0.0, distance=[1, 1], stretching=[1, 1])
        assert_allclose(x01, x03)
        assert_allclose(hx1, hx3)

    def test_seasurface(self):
        x01, hx1 = meshes.origin_and_widths(
                1/np.pi, 9*mu_0, 0.0, [-1, 1], stretching=[1, 1])
        x02, hx2 = meshes.origin_and_widths(
                1/np.pi, 9*mu_0, -0.5, [-1, 0], seasurface=0.0,
                stretching=[1, 1])
        assert_allclose(x01, x02)
        assert_allclose(hx1, hx2)

    def test_status_quo_with_all(self, capsys):
        # Defaults.
        meshes.origin_and_widths(
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
        assert "Final extent   [m] : -10223 - 50988" in out
        assert "Cell widths    [m] : 205 / 205 / 11769" in out
        assert "Number of cells    : 32 (7 / 25 / 0)" in out
        assert "Max stretching     : 1.000 (1.000) / 1.288" in out

        # All set.
        meshes.origin_and_widths(
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
        assert "Final extent   [m] : -11118 - 6651" in out
        assert "Cell widths    [m] : 100 / 100 / 1968" in out
        assert "Number of cells    : 40 (15 / 25 / 0)" in out
        assert "Max stretching     : 1.000 (1.000) / 1.258" in out

        # High frequencies.
        _, _, out = meshes.origin_and_widths(
            frequency=1e8,
            properties=[5, 1, 50],
            center=0,
            domain=[-1, 1],
            verb=-1,
            )

        assert "Skin depth     [m] : 0.113 / 0.050 / 0.356" in out
        assert "Survey dom. DS [m] : -1.000 - 1.000" in out
        assert "Comp. dom. DC  [m] : -1.316 - 3.236" in out
        assert "Final extent   [m] : -1.327 - 3.262" in out
        assert "Cell widths    [m] : 0.038 / 0.038 / 0.234" in out
        assert "Number of cells    : 80 (54 / 26 / 0)" in out
        assert "Max stretching     : 1.000 (1.000) / 1.096" in out


def test_good_mg_cell_nr(capsys):
    numbers = meshes.good_mg_cell_nr(max_nr=128, max_lowest=5, min_div=3)
    assert_allclose([16, 24, 32, 40, 48, 64, 80, 96, 128], numbers)

    with pytest.raises(ValueError, match='Maximum lowest is 25, '):
        numbers = meshes.good_mg_cell_nr(max_nr=128, max_lowest=25, min_div=3)

    numbers = meshes.good_mg_cell_nr(max_nr=50, max_lowest=3, min_div=5)
    assert len(numbers) == 0


def test_skin_depth():
    t1 = meshes.skin_depth(1/np.pi, 2.0, 2.0/mu_0)
    assert t1 == 0.5

    t2 = meshes.skin_depth(1/np.pi, 1/mu_0)
    assert t2 == 1

    t3 = meshes.skin_depth(-1/(2*np.pi**2), 2.0)
    assert_allclose(t3, np.sqrt(0.5/mu_0))


def test_wavelength():
    t1 = meshes.wavelength(0.5)
    assert_allclose(t1, np.pi)


def test_cell_width():
    t1 = meshes.cell_width(1, pps=1)
    assert t1 == 1

    t2 = meshes.cell_width(1)
    assert np.round(t2, 3) == 0.333

    t3 = meshes.cell_width(503.0, limits=10)
    assert t3 == 10

    t3 = meshes.cell_width(503.0, limits=[100, 120])
    assert t3 == 120


def test_check_mesh():

    # Bad class name.
    grid = meshes.BaseMesh(h=[2, 2, 2], origin=(0, 0, 0))
    grid.__class__.__name__ = 'Test'
    with pytest.raises(TypeError, match="Mesh must be a TensorMesh."):
        meshes.check_mesh(grid)

    # Wrong dimension.
    if discretize is None:
        grid = meshes.TensorMesh(h=[2, 2, 2], origin=(0, 0, 0))
        grid.origin = (0, 0)
    else:
        grid = meshes.TensorMesh(h=[2, 2], origin=(0, 0))
    with pytest.raises(TypeError, match="Mesh must be a 3D mesh."):
        meshes.check_mesh(grid)

    # Bad cell number.
    grid = meshes.TensorMesh(h=[[2, ], [2, ], [2, 2]], origin=(0, 0, 0))
    with pytest.warns(UserWarning, match='ptimal for MG solver. Good numbers'):
        meshes.check_mesh(grid)

    # A good one, nothing should happen.
    hx = np.ones(16)*20
    grid = meshes.TensorMesh(h=[hx, hx, hx], origin=(0, 0, 0))
    meshes.check_mesh(grid)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestEstimateGriddingOpts():
    if xarray is not None:
        # Create a simple survey
        sources = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.TxElectricDipole,
                (0, [1000, 3000, 5000], -950, 0, 0))
        receivers = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint,
                (np.arange(11)*500, 2000, -1000, 0, 0))
        frequencies = (0.1, 10.0)

        survey = emg3d.Survey(
                sources, receivers, frequencies, noise_floor=1e-15,
                relative_error=0.05)

        # Create a simple grid and model
        grid = meshes.TensorMesh(
                [np.ones(32)*250, np.ones(16)*500, np.ones(16)*500],
                np.array([-1250, -1250, -2250]))
        model = emg3d.Model(grid, 0.1, np.ones(grid.shape_cells)*10)
        model.property_y[5, 8, 3] = 100000  # Cell at source center

    def test_empty_dict(self):
        gdict = meshes.estimate_gridding_opts({}, self.model, self.survey)

        assert gdict['frequency'] == 1.0
        assert gdict['mapping'] == self.model.map.name
        assert_allclose(gdict['center'], (0, 3000, -950))
        assert_allclose(gdict['domain']['x'], (-500, 5500))
        assert_allclose(gdict['domain']['y'], (600, 5400))
        assert_allclose(gdict['domain']['z'], (-3651, -651))
        assert_allclose(gdict['properties'], [100000, 10, 10, 10, 10, 10, 10])

    def test_mapping_vector(self):
        gridding_opts = {
            'mapping': "LgConductivity",
            'vector': 'xZ',
            }
        gdict = meshes.estimate_gridding_opts(
                gridding_opts, self.model, self.survey)

        assert_allclose(
                gdict['properties'],
                np.log10(1/np.array([100000, 10, 10, 10, 10, 10, 10])),
                atol=1e-15)
        assert_allclose(gdict['vector']['x'], self.grid.nodes_x)
        assert gdict['vector']['y'] is None
        assert_allclose(gdict['vector']['z'], self.grid.nodes_z)

    def test_vector_domain_distance(self):
        gridding_opts = {
                'vector': 'Z',
                'domain': (None, [-1000, 1000], None),
                'distance': [[5, 10], None, None],
                }
        gdict = meshes.estimate_gridding_opts(
                gridding_opts, self.model, self.survey)

        assert gdict['vector']['x'] == gdict['vector']['y'] is None
        assert_allclose(gdict['vector']['z'], self.model.grid.nodes_z)

        assert gdict['domain']['x'] is None
        assert gdict['domain']['y'] == [-1000, 1000]
        assert gdict['domain']['z'] == [self.model.grid.nodes_z[0],
                                        self.model.grid.nodes_z[-1]]
        assert gdict['distance']['x'] == [5, 10]
        assert gdict['distance']['y'] == gdict['distance']['z'] is None

        # As dict
        gridding_opts = {
                'vector': 'Z',
                'domain': {'x': None, 'y': [-1000, 1000], 'z': None},
                'distance': {'x': [5, 10], 'y': None, 'z': None},
                }
        gdict = meshes.estimate_gridding_opts(
                gridding_opts, self.model, self.survey)

        assert gdict['vector']['x'] == gdict['vector']['y'] is None
        assert_allclose(gdict['vector']['z'], self.model.grid.nodes_z)

        assert gdict['domain']['x'] is None
        assert gdict['domain']['y'] == [-1000, 1000]
        assert gdict['domain']['z'] == [self.model.grid.nodes_z[0],
                                        self.model.grid.nodes_z[-1]]
        assert gdict['distance']['x'] == [5, 10]
        assert gdict['distance']['y'] == gdict['distance']['z'] is None

    def test_pass_along(self):
        gridding_opts = {
            'vector': {'x': None, 'y': 1, 'z': None},
            'stretching': [1.2, 1.3],
            'seasurface': -500,
            'cell_numbers': [10, 20, 30],
            'lambda_factor': 0.8,
            'max_buffer': 10000,
            'min_width_limits': ([20, 40], [20, 40], [20, 40]),
            'min_width_pps': 4,
            'verb': -1,
            }

        gdict = meshes.estimate_gridding_opts(
                gridding_opts.copy(), self.model, self.survey)

        # Check that all parameters passed unchanged.
        gdict2 = {k: gdict[k] for k, _ in gridding_opts.items()}
        # Except the tuple, which should be a dict now
        gridding_opts['min_width_limits'] = {
                'x': gridding_opts['min_width_limits'][0],
                'y': gridding_opts['min_width_limits'][1],
                'z': gridding_opts['min_width_limits'][2]
        }
        assert helpers.compare_dicts(gdict2, gridding_opts)

    def test_factor(self):
        sources = emg3d.TxElectricDipole((0, 3000, -950, 0, 0))
        receivers = emg3d.RxElectricPoint((0, 3000, -1000, 0, 0))

        # Adjusted x-domain.
        survey = emg3d.Survey(
                self.sources, receivers, self.frequencies, noise_floor=1e-15,
                relative_error=0.05)

        gdict = meshes.estimate_gridding_opts({}, self.model, survey)

        assert_allclose(gdict['domain']['x'], (-800, 800))

        # Adjusted x-domain.
        survey = emg3d.Survey(
                sources, self.receivers, self.frequencies, noise_floor=1e-15,
                relative_error=0.05)

        gdict = meshes.estimate_gridding_opts({}, self.model, survey)

        assert_allclose(gdict['domain']['y'], (1500, 3500))

    def test_error(self):
        with pytest.raises(TypeError, match='Unexpected gridding_opts'):
            _ = meshes.estimate_gridding_opts(
                    {'what': True}, self.model, self.survey)
