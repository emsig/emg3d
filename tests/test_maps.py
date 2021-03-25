import pytest
import numpy as np
from numpy.testing import assert_allclose

from . import alternatives
from emg3d import fields, maps, meshes, models, electrodes

# Import soft dependencies.
try:
    import discretize
    # Backwards compatibility; remove latest for version 1.0.0.
    dv = discretize.__version__.split('.')
    if int(dv[0]) == 0 and int(dv[1]) < 6:
        discretize = None
except ImportError:
    discretize = None


# MAPS
class TestMaps:
    mesh = meshes.TensorMesh([[1, 1], [1, 1], [1]], (0, 0, 0))

    values = np.array([0.01, 10, 3, 4])

    def test_new(self):

        class MapNew(maps.BaseMap):
            def __init__(self):
                super().__init__(description='my new map')

        testmap = MapNew()

        assert "MapNew: my new map" in testmap.__repr__()

        with pytest.raises(NotImplementedError, match='Forward map not imple'):
            testmap.forward(1)

        with pytest.raises(NotImplementedError, match='Backward map not impl'):
            testmap.backward(1)

        with pytest.raises(NotImplementedError, match='Derivative chain not '):
            testmap.derivative_chain(1, 1)

    def test_conductivity(self):
        model = models.Model(self.mesh, self.values, mapping='Conductivity')

        # Forward
        forward = model.map.forward(self.values)
        assert_allclose(forward, self.values)

        # Backward
        backward = model.map.backward(forward)
        assert_allclose(backward, self.values)

        # Derivative
        gradient = 2*np.ones(model.property_x.shape)
        derivative = gradient.copy()
        model.map.derivative_chain(gradient, model.property_x)
        assert_allclose(derivative, derivative)

    def test_lgconductivity(self):
        model = models.Model(self.mesh, np.log10(self.values),
                             mapping='LgConductivity')

        # Forward
        forward = model.map.forward(self.values)
        assert_allclose(forward, np.log10(self.values))

        # Backward
        backward = model.map.backward(forward)
        assert_allclose(backward, self.values)

        # Derivative
        gradient = 2*np.ones(model.property_x.shape)
        derivative = gradient.copy()
        model.map.derivative_chain(gradient, model.property_x)
        assert_allclose(gradient, derivative*10**model.property_x*np.log(10))

    def test_lnconductivity(self):
        model = models.Model(self.mesh, np.log(self.values),
                             mapping='LnConductivity')

        # Forward
        forward = model.map.forward(self.values)
        assert_allclose(forward, np.log(self.values))

        # Backward
        backward = model.map.backward(forward)
        assert_allclose(backward, self.values)

        # Derivative
        gradient = 2*np.ones(model.property_x.shape)
        derivative = gradient.copy()
        model.map.derivative_chain(gradient, model.property_x)
        assert_allclose(gradient, derivative*np.exp(model.property_x))

    def test_resistivity(self):
        model = models.Model(self.mesh, 1/self.values, mapping='Resistivity')

        # Forward
        forward = model.map.forward(self.values)
        assert_allclose(forward, 1/self.values)

        # Backward
        backward = model.map.backward(forward)
        assert_allclose(backward, self.values)

        # Derivative
        gradient = 2*np.ones(model.property_x.shape)
        derivative = gradient.copy()
        model.map.derivative_chain(gradient, model.property_x)
        assert_allclose(gradient, -derivative*(1/model.property_x)**2)

    def test_lgresistivity(self):
        model = models.Model(self.mesh, np.log10(1/self.values),
                             mapping='LgResistivity')

        # Forward
        forward = model.map.forward(self.values)
        assert_allclose(forward, np.log10(1/self.values))

        # Backward
        backward = model.map.backward(forward)
        assert_allclose(backward, self.values)

        # Derivative
        gradient = 2*np.ones(model.property_x.shape)
        derivative = gradient.copy()
        model.map.derivative_chain(gradient, model.property_x)
        assert_allclose(gradient, -derivative*10**-model.property_x*np.log(10))

    def test_lnresistivity(self):
        model = models.Model(self.mesh, np.log(self.values),
                             mapping='LnResistivity')

        # Forward
        forward = model.map.forward(self.values)
        assert_allclose(forward, np.log(1/self.values))

        # Backward
        backward = model.map.backward(forward)
        assert_allclose(backward, self.values)

        # Derivative
        gradient = 2*np.ones(model.property_x.shape)
        derivative = gradient.copy()
        model.map.derivative_chain(gradient, model.property_x)
        assert_allclose(gradient, -derivative*np.exp(-model.property_x))


# INTERPOLATIONS
class TestInterpolate:
    # emg3d.interpolate is only a dispatcher function, calling other
    # interpolation routines; there are lots of small dummy tests here, but in
    # the end it is not up to emg3d.interpolate to check the accuracy of the
    # actual interpolation; the only relevance is to check if it calls the
    # right function.

    def test_linear(self):
        igrid = meshes.TensorMesh(
                [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])],
                [0, -1, -1])
        ogrid = meshes.TensorMesh(
                [np.array([1]), np.array([1]), np.array([1])],
                [0.5, 0, 0])
        values = np.r_[9*[1.0, ], 9*[2.0, ]].reshape(igrid.shape_cells)

        # Simple, linear example.
        out = maps.interpolate(
                grid=igrid, values=values, xi=ogrid, method='linear')
        assert_allclose(out[0, 0, 0], 1.5)

        # Provide ogrid.gridCC.
        ogrid._gridCC = np.array([[0.5, 0.5, 0.5]])
        out2 = maps.interpolate(igrid, values, ogrid, 'linear')
        assert_allclose(out2[0, 0, 0], 1.5)

    def test_linear_cubic(self):
        # Check 'linear' and 'cubic' yield almost the same result for a well
        # determined, very smoothly changing example.

        # Fine grid.
        fgrid = meshes.TensorMesh(
            [np.ones(2**6)*10, np.ones(2**5)*100, np.ones(2**4)*1000],
            origin=np.array([-320., -1600, -8000]))

        # Smoothly changing model for fine grid.
        cmodel = np.arange(1, fgrid.n_cells+1).reshape(
                fgrid.shape_cells, order='F')

        # Coarser grid.
        cgrid = meshes.TensorMesh(
            [np.ones(2**5)*15, np.ones(2**4)*150, np.ones(2**3)*1500],
            origin=np.array([-240., -1200, -6000]))

        # Interpolate linearly and cubic spline.
        lin_model = maps.interpolate(fgrid, cmodel, cgrid, 'linear')
        cub_model = maps.interpolate(fgrid, cmodel, cgrid, 'cubic')

        # Compare
        assert np.max(np.abs((lin_model-cub_model)/lin_model*100)) < 1.0

    def test_nearest(self):
        # Assert it is 'nearest' or extrapolate if points are outside.
        tgrid = meshes.TensorMesh(
                [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]),
                 np.array([1, 1, 1, 1])], origin=np.array([0., 0, 0]))
        tmodel = np.ones(tgrid.n_cells).reshape(tgrid.shape_cells, order='F')
        tmodel[:, 0, :] = 2
        t2grid = meshes.TensorMesh(
                [np.array([1]), np.array([1]), np.array([1])],
                origin=np.array([2, -1, 2]))

        # Nearest with cubic.
        out = maps.interpolate(tgrid, tmodel, t2grid, 'cubic')
        assert_allclose(out, 2.)

        # Same, but with log.
        vlog = maps.interpolate(tgrid, tmodel, t2grid, 'cubic', log=True)
        vlin = maps.interpolate(tgrid, np.log10(tmodel), t2grid, 'cubic')
        assert_allclose(vlog, 10**vlin)

        # Extrapolate with linear.
        out = maps.interpolate(tgrid, tmodel, t2grid, 'linear')
        assert_allclose(out, 3.)

        # Same, but with log.
        vlog = maps.interpolate(tgrid, tmodel, t2grid, 'linear', log=True)
        vlin = maps.interpolate(tgrid, np.log10(tmodel), t2grid, 'linear')
        assert_allclose(vlog, 10**vlin)

        # Assert it is 0 if points are outside.
        out = maps.interpolate(tgrid, tmodel, t2grid, 'cubic', False)
        assert_allclose(out, 0.)
        out = maps.interpolate(tgrid, tmodel, t2grid, 'linear', False)
        assert_allclose(out, 0.)

    def test_volume(self):
        # == X == Simple 1D model
        grid_in = meshes.TensorMesh(
                [np.ones(5)*10, np.array([1, ]), np.array([1, ])],
                origin=np.array([0, 0, 0]))
        grid_out = meshes.TensorMesh(
                [[10, 25, 10, 5, 2], [1, ], [1, ]], origin=(-5, 0, 0))
        values_in = np.array([1., 5., 3, 7, 2])[:, None, None]
        values_out = maps.interpolate(grid_in, values_in, grid_out, 'volume')

        # Result 2nd cell: (5*1+10*5+10*3)/25=3.4
        assert_allclose(values_out[:, 0, 0], np.array([1, 3.4, 7, 2, 2]))

        # Check log:
        vlogparam = maps.interpolate(
                grid_in, values_in, grid_out, 'volume', log=True)
        vlinloginp = maps.interpolate(
                grid_in, np.log10(values_in), grid_out, 'volume')
        assert_allclose(vlogparam, 10**vlinloginp)

        # == Y ==  Reverse it
        grid_out = meshes.TensorMesh(
                [np.array([1, ]), np.ones(5)*10, np.array([1, ])],
                origin=np.array([0, 0, 0]))
        grid_in = meshes.TensorMesh(
                [[1, ], [10, 25, 10, 5, 2], [1, ]], origin=[0, -5, 0])
        values_in = np.array([1, 3.4, 7, 2, 2])[None, :, None]
        values_out = maps.interpolate(grid_in, values_in, grid_out, 'volume')

        # Result 1st cell: (5*1+5*3.4)/10=2.2
        assert_allclose(values_out[0, :, 0], np.array([2.2, 3.4, 3.4, 7, 2]))

        # == Z == Another 1D test
        grid_in = meshes.TensorMesh(
                [np.array([1, ]), np.array([1, ]), np.ones(9)*10],
                origin=np.array([0, 0, 0]))
        grid_out = meshes.TensorMesh(
                [np.array([1, ]), np.array([1, ]), np.array([20, 41, 9, 30])],
                origin=np.array([0, 0, 0]))
        values_in = np.arange(1., 10)[None, None, :]
        values_out = maps.interpolate(grid_in, values_in, grid_out, 'volume')

        assert_allclose(values_out[0, 0, :],
                        np.array([1.5, 187/41, 7, 260/30]))

        # == 3D ==
        grid_in = meshes.TensorMesh(
                [[1, 1, 1], [10, 10, 10, 10, 10], [10, 2, 10]],
                origin=(0, 0, 0))
        grid_out = meshes.TensorMesh(
                [[1, 2, ], [10, 25, 10, 5, 2], [4, 4, 4]], origin=[0, -5, 6])
        create = np.array([[1, 2., 1]])
        create = np.array([create, 2*create, create])
        values_in = create*np.array([1., 5., 3, 7, 2])[None, :, None]

        values_out = maps.interpolate(grid_in, values_in, grid_out, 'volume')

        check = np.array([[1, 1.5, 1], [1.5, 2.25, 1.5]])[:, None, :]
        check = check*np.array([1, 3.4, 7, 2, 2])[None, :, None]

        assert_allclose(values_out, check)

        # == If the extent is the same, volume*values must remain constant. ==
        grid_in = meshes.TensorMesh(
                [np.array([1, 1, 1]), np.array([10, 10, 10, 10, 10]),
                 np.array([10, 2, 10])], origin=np.array([0, 0, 0]))
        grid_out = meshes.TensorMesh(
                [np.array([1, 2, ]), np.array([5, 25, 10, 5, 5]),
                 np.array([9, 4, 9])], origin=np.array([0, 0, 0]))
        create = np.array([[1, 2., 1]])
        create = np.array([create, 2*create, create])
        values_in = create*np.array([1., 5., 3, 7, 2])[None, :, None]
        vol_in = np.outer(np.outer(
            grid_in.h[0], grid_in.h[1]).ravel('F'), grid_in.h[2])
        vol_in = vol_in.ravel('F').reshape(grid_in.shape_cells, order='F')

        values_out = maps.interpolate(grid_in, values_in, grid_out, 'volume')
        vol_out = np.outer(np.outer(grid_out.h[0], grid_out.h[1]).ravel('F'),
                           grid_out.h[2])
        vol_out = vol_out.ravel('F').reshape(grid_out.shape_cells, order='F')

        assert_allclose(np.sum(values_out*vol_out), np.sum(values_in*vol_in))

    def test_all_run(self):
        hx = [1, 1, 1, 2, 4, 8]
        grid = meshes.TensorMesh([hx, hx, hx], (0, 0, 0))
        grid2 = meshes.TensorMesh([[2, 4, 5], [1, 1], [4, 5]], (0, 1, 0))
        field = fields.Field(grid)
        field.fx = np.arange(1, field.fx.size+1).reshape(
                field.fx.shape, order='F')
        model = models.Model(grid, 1, 2, 3)

        model.property_x[1, :, :] = 2
        model.property_x[2, :, :] = 3
        model.property_x[3, :, :] = 4
        model.property_x[4, :, :] = np.arange(1, 37).reshape((6, 6), order='F')
        model.property_x[5, :, :] = 200

        xi = (1, [8, 7, 6, 8, 9], [1])

        # == NEAREST ==
        # property - grid
        _ = maps.interpolate(grid, model.property_x, grid2, method='nearest')
        # field - grid
        _ = maps.interpolate(grid, field.fx, grid2, method='nearest')
        # property - points
        _ = maps.interpolate(grid, model.property_x, xi, method='nearest')
        # field - points
        _ = maps.interpolate(grid, field.fx, xi, method='nearest')

        # == LINEAR ==
        # property - grid
        _ = maps.interpolate(grid, model.property_x, grid2, method='linear')
        # field - grid
        _ = maps.interpolate(grid, field.fx, grid2, method='linear')
        # property - points
        _ = maps.interpolate(grid, model.property_x, xi, method='linear')
        # field - points
        _ = maps.interpolate(grid, field.fx, xi, method='linear')

        # == CUBIC ==
        # property - grid
        _ = maps.interpolate(grid, model.property_x, grid2, method='cubic')
        # field - grid
        _ = maps.interpolate(grid, field.fx, grid2, method='cubic')
        # property - points
        _ = maps.interpolate(grid, model.property_x, xi, method='cubic')
        # field - points
        _ = maps.interpolate(grid, field.fx, xi, method='cubic')

        # == VOLUME ==
        # property - grid
        _ = maps.interpolate(grid, model.property_x, grid2, method='volume')
        # field - grid
        with pytest.raises(ValueError, match="for cell-centered properties"):
            maps.interpolate(grid, field.fx, grid2, method='volume')
        # property - points
        with pytest.raises(ValueError, match="only implemented for TensorM"):
            maps.interpolate(grid, model.property_x, xi, method='volume')
        # field - points
        with pytest.raises(ValueError, match="only implemented for TensorM"):
            maps.interpolate(grid, field.fx, xi, method='volume')


def test_points_from_grids():
    hx = [1, 1, 1, 2, 4, 8]
    grid = meshes.TensorMesh([hx, hx, hx], (0, 0, 0))
    grid2 = meshes.TensorMesh([[2, 4, 5], [1, 1], [4, 5]], (0, 1, 0))

    xi_tuple = (1, [8, 7, 6, 8, 9], [1])
    xi_array = np.arange(18).reshape(-1, 3)

    v_prop = np.ones(grid.shape_cells)
    v_field = np.ones(grid.shape_edges_x)

    # linear  - values = prop  - xi = grid
    out = maps._points_from_grids(grid, v_prop, grid2, 'linear')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][0], [0.5, 1.5, 2.5, 4., 7., 13.])
    assert_allclose(out[1][0, :], [1., 1.5, 2])
    assert out[2] == (3, 2, 2)

    # nearest - values = field - xi = grid
    out = maps._points_from_grids(grid, v_field, grid2, 'nearest')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][1], [0., 1., 2., 3., 5., 9., 17.])
    assert_allclose(out[1][1, :], [4., 1., 0.])
    assert out[2] == (3, 3, 3)

    # cubic   - values = prop  - xi = tuple
    out = maps._points_from_grids(grid, v_prop, xi_tuple, 'cubic')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][2], [0.5, 1.5, 2.5, 4., 7., 13.])
    assert_allclose(out[1][2, :], [1., 6., 1.])
    assert out[2] == (5, )

    # linear  - values = field - xi = tuple
    out = maps._points_from_grids(grid, v_field, xi_tuple, 'linear')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][0], [0.5, 1.5, 2.5, 4., 7., 13.])
    assert_allclose(out[1][-1, :], [1., 9., 1.])
    assert out[2] == (5, )

    # nearest - values = prop  - xi = ndarray
    out = maps._points_from_grids(grid, v_prop, xi_array, 'nearest')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][0], [0.5, 1.5, 2.5, 4., 7., 13.])
    assert_allclose(out[1], xi_array)
    assert out[2] == (6, )

    # cubic   - values = field - xi = ndarray
    out = maps._points_from_grids(grid, v_field, xi_array, 'cubic')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][0], [0.5, 1.5, 2.5, 4., 7., 13.])
    assert_allclose(out[1], xi_array)
    assert out[2] == (6, )

    # cubic   - values = 1Darr - xi = grid  - FAILS
    with pytest.raises(ValueError, match='must be a 3D ndarray'):
        maps._points_from_grids(grid, v_field.ravel(), grid2, 'cubic')

    # volume  - values = prop  - xi = grid
    out = maps._points_from_grids(grid, v_prop, grid2, 'volume')
    assert isinstance(out[1], tuple)
    assert_allclose(out[0][0], [0., 1, 2, 3, 5, 9, 17])
    assert_allclose(out[1][0], [0., 2, 6, 11])
    assert out[2] == (3, 2, 2)

    # volume  - values = field - xi = grid  - FAILS
    with pytest.raises(ValueError, match='only implemented for cell-centered'):
        maps._points_from_grids(grid, v_field, grid2, 'volume')

    # volume  - values = prop  - xi = tuple - FAILS
    with pytest.raises(ValueError, match='only implemented for TensorMesh'):
        maps._points_from_grids(grid, v_prop, xi_tuple, 'volume')

    # tuple can contain any dimension; it will work (undocumented).
    shape = (3, 2, 4, 5)
    coords = np.arange(np.prod(shape)).reshape(shape, order='F')
    xi_tuple2 = (1, coords, 10)
    out = maps._points_from_grids(grid, v_field, xi_tuple2, 'nearest')
    assert isinstance(out[1], np.ndarray)
    assert_allclose(out[0][0], [0.5, 1.5, 2.5, 4., 7., 13.])
    assert_allclose(out[1][-1, :], [1., 119, 10])
    assert out[2] == shape


def test_interp_spline_3d():
    x = np.array([1., 2, 4, 5])
    pts = (x, x, x)

    p1 = np.array([[3, 1, 3], [3, 3, 3], [3, 4, 3]])
    v1 = np.ones((4, 4, 4), order='F')
    v1[1, :, :] = 4.0
    v1[2, :, :] = 16.0
    v1[3, :, :] = 25.0

    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1, order=0)
    assert_allclose(out, 16)
    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1, order=1)
    assert_allclose(out, 10)
    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1, order=2)
    assert_allclose(out, 9.4)
    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1)
    assert_allclose(out, 9.25)
    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1, order=4)
    assert_allclose(out, 9.117647)
    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1, order=5)
    assert_allclose(out, 9.0625)

    p2 = np.array([[1, 3, 3], [3, 3, 3], [4, 3, 3]])
    v2 = np.rollaxis(v1, 1)
    out = maps.interp_spline_3d(points=pts, values=v2, xi=p2)
    assert_allclose(out, 9.25)

    p3 = np.array([[3, 3, 1], [3, 3, 3], [3, 3, 4]])
    v3 = np.rollaxis(v2, 1)
    v3 = v3 + 1j*v3
    out = maps.interp_spline_3d(points=pts, values=v3, xi=p3)
    assert_allclose(out, 9.25 + 9.25j)

    p1 = np.array([[3, 100, 3]])
    v1 = np.ones((4, 4, 4), order='F')
    v1[1, :, :] = 4.0
    v1[2, :, :] = 16.0
    v1[3, :, :] = 25.0
    out = maps.interp_spline_3d(points=pts, values=v1, xi=p1, cval=999)
    assert_allclose(out, 999)


@pytest.mark.parametrize("njit", [True, False])
def test_interp_volume_average(njit):
    if njit:
        interp_volume_average = maps.interp_volume_average
    else:
        interp_volume_average = maps.interp_volume_average.py_func

    # Comparison to alt_version.
    grid_in = meshes.TensorMesh(
            [np.ones(30), np.ones(20)*5, np.ones(10)*10],
            origin=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.arange(7)+1, np.arange(13)+1, np.arange(13)+1],
            origin=np.array([0.5, 3.33, 5]))

    values = np.arange(grid_in.n_cells, dtype=np.float64).reshape(
            grid_in.shape_cells, order='F')

    points = (grid_in.nodes_x, grid_in.nodes_y, grid_in.nodes_z)
    new_points = (grid_out.nodes_x, grid_out.nodes_y, grid_out.nodes_z)

    # Compute volume.
    vol = np.outer(np.outer(
        grid_out.h[0], grid_out.h[1]).ravel('F'), grid_out.h[2])
    vol = vol.ravel('F').reshape(grid_out.shape_cells, order='F')

    # New solution.
    new_values = np.zeros(grid_out.shape_cells, dtype=values.dtype)
    interp_volume_average(*points, values, *new_points, new_values, vol)

    # Old solution.
    new_values_alt = np.zeros(grid_out.shape_cells, dtype=values.dtype)
    alternatives.alt_volume_average(
            *points, values, *new_points, new_values_alt)

    assert_allclose(new_values, new_values_alt)


@pytest.mark.parametrize("njit", [True, False])
def test_volume_average_weights(njit):
    if njit:
        volume_avg_weights = maps._volume_average_weights
    else:
        volume_avg_weights = maps._volume_average_weights.py_func

    grid_in = meshes.TensorMesh(
            [np.ones(11), np.ones(10)*2, np.ones(3)*10],
            origin=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.arange(4)+1, np.arange(5)+1, np.arange(6)+1],
            origin=np.array([0.5, 3.33, 5]))

    wx, ix_in, ix_out = volume_avg_weights(grid_in.nodes_x, grid_out.nodes_x)
    assert_allclose(wx,
                    [0.5, 0.5, 0.5, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5])
    assert_allclose(ix_in, [0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10])
    assert_allclose(ix_out, [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    wy, iy_in, iy_out = volume_avg_weights(grid_in.nodes_y, grid_out.nodes_y)
    assert_allclose(wy, [0.67, 0.33, 1.67, 0.33, 1.67, 1.33, 0.67, 2.,
                         1.33, 0.67, 2, 2, 0.33])
    assert_allclose(iy_in, [1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9])
    assert_allclose(iy_out, [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

    wz, iz_in, iz_out = volume_avg_weights(grid_in.nodes_z, grid_out.nodes_z)
    assert_allclose(wz, [1, 2, 2, 1, 4, 5, 6])
    assert_allclose(iz_in, [0, 0, 0, 1, 1, 1, 2])
    assert_allclose(iz_out, [0, 1, 2, 2, 3, 4, 5])

    w, inp, out = volume_avg_weights(x_i=np.array([0., 5, 7, 10]),
                                     x_o=np.array([-1., 1, 4, 6, 7, 11]))
    assert_allclose(w, [1, 1, 3, 1, 1, 1, 3, 1])
    assert_allclose(inp, [0, 0, 0, 0, 1, 1, 2, 2])
    assert_allclose(out, [0, 0, 1, 2, 2, 3, 4, 4])


@pytest.mark.parametrize("njit", [True, False])
def test_interp_edges_to_vol_averages(njit):
    if njit:
        edges_to_vol_averages = maps.interp_edges_to_vol_averages
    else:
        edges_to_vol_averages = maps.interp_edges_to_vol_averages.py_func

    # To test it, we create a mesh 2x2x2 cells,
    # where all hx/hy/hz have distinct lengths.
    x0, x1 = 2, 3
    y0, y1 = 4, 5
    z0, z1 = 6, 7

    grid = meshes.TensorMesh([[x0, x1], [y0, y1], [z0, z1]], [0, 0, 0])
    field = fields.Field(grid)

    # Only three edges have a value, one in each direction.
    fx = 1.23+9.87j
    fy = 2.68-5.48j
    fz = 1.57+7.63j
    field.fx[0, 1, 1] = fx
    field.fy[1, 1, 1] = fy
    field.fz[1, 1, 0] = fz

    # Initiate gradient.
    grad_x = np.zeros(grid.shape_cells, order='F', dtype=complex)
    grad_y = np.zeros(grid.shape_cells, order='F', dtype=complex)
    grad_z = np.zeros(grid.shape_cells, order='F', dtype=complex)

    cell_volumes = grid.cell_volumes.reshape(grid.shape_cells, order='F')

    # Call function.
    edges_to_vol_averages(ex=field.fx, ey=field.fy, ez=field.fz,
                          volumes=cell_volumes,
                          ox=grad_x, oy=grad_y, oz=grad_z)
    grad = grad_x + grad_y + grad_z

    # Check all eight cells explicitly by
    # - computing the volume of the cell;
    # - multiplying with the present fields in that cell.
    assert_allclose(x0*y0*z0*(fx+fz)/4, grad[0, 0, 0])
    assert_allclose(x1*y0*z0*fz/4, grad[1, 0, 0])
    assert_allclose(x0*y1*z0*(fx+fy+fz)/4, grad[0, 1, 0])
    assert_allclose(x1*y1*z0*(fy+fz)/4, grad[1, 1, 0])
    assert_allclose(x0*y0*z1*fx/4, grad[0, 0, 1])
    assert_allclose(0j, grad[1, 0, 1])
    assert_allclose(x0*y1*z1*(fx+fy)/4, grad[0, 1, 1])
    assert_allclose(x1*y1*z1*fy/4, grad[1, 1, 1])

    # Separately.
    assert_allclose(x0*y0*z0*fx/4, grad_x[0, 0, 0])
    assert_allclose(x0*y0*z0*fz/4, grad_z[0, 0, 0])

    assert_allclose(x0*y1*z0*fx/4, grad_x[0, 1, 0])
    assert_allclose(x0*y1*z0*fy/4, grad_y[0, 1, 0])
    assert_allclose(x0*y1*z0*fz/4, grad_z[0, 1, 0])

    assert_allclose(x1*y1*z0*fy/4, grad_y[1, 1, 0])
    assert_allclose(x1*y1*z0*fz/4, grad_z[1, 1, 0])

    assert_allclose(x0*y1*z1*fx/4, grad_x[0, 1, 1])
    assert_allclose(x0*y1*z1*fy/4, grad_y[0, 1, 1])

    if discretize is not None:
        def volume_disc(grid, field):
            out = grid.average_edge_to_cell*field.field*grid.cell_volumes
            return out.reshape(grid.shape_cells, order='F')

        assert_allclose(grad, 3*volume_disc(grid, field))

    if discretize is not None:

        out_x = grid.average_edge_x_to_cell*field.fx.ravel('F')
        out_x *= grid.cell_volumes
        out_y = grid.average_edge_y_to_cell*field.fy.ravel('F')
        out_y *= grid.cell_volumes
        out_z = grid.average_edge_z_to_cell*field.fz.ravel('F')
        out_z *= grid.cell_volumes

        assert_allclose(grad_x, out_x.reshape(grid.shape_cells, order='F'))
        assert_allclose(grad_y, out_y.reshape(grid.shape_cells, order='F'))
        assert_allclose(grad_z, out_z.reshape(grid.shape_cells, order='F'))


class TestRotation:
    def test_rotation(self):
        assert_allclose(maps.rotation(0, 0), [1, 0, 0])
        assert_allclose(maps.rotation(90, 0), [0, 1, 0])
        assert_allclose(maps.rotation(-90, 0), [0, -1, 0])
        assert_allclose(maps.rotation(0, 90), [0, 0, 1])
        assert_allclose(maps.rotation(0, -90), [0, 0, -1])
        dazm, ddip = 30, 60
        razm, rdip = np.deg2rad(dazm), np.deg2rad(ddip)
        assert_allclose(
                maps.rotation(dazm, ddip),
                [np.cos(razm)*np.cos(rdip), np.sin(razm)*np.cos(rdip),
                 np.sin(rdip)])
        dazm, ddip = -45, 180
        razm, rdip = np.deg2rad(dazm), np.deg2rad(ddip)
        assert_allclose(
                maps.rotation(dazm, ddip),
                [np.cos(razm)*np.cos(rdip), np.sin(razm)*np.cos(rdip),
                 np.sin(rdip)],
                atol=1e-14)

        azm, dip = np.pi/3, np.pi/4
        rot1 = maps.rotation(azm, dip, rad=True)
        rot2 = maps.rotation(np.rad2deg(azm), np.rad2deg(dip), rad=False)
        assert_allclose(rot1, rot2)

    def test_get_angles_get_points(self):
        pi_1 = np.pi
        pi_2 = np.pi/2
        pi_4 = np.pi/4
        pi_34 = 3*np.pi/4

        # We test here an extensive list of simple cases:
        # - dipoles along the principal axes
        # - dipoles in the middle between two axes (faces)
        # - dipoles into the middle of three axes (quadrants)

        # Format: (x, y, z): azm_deg, dip_deg, azm_rad, dip_rad
        data = {
            # 6 axes
            (+1, +0, +0): [0, 0, 0, 0],
            (+0, +1, +0): [90, 0, pi_2, 0],
            (-1, +0, +0): [180, 0, pi_1, 0],
            (+0, -1, +0): [-90, 0, -pi_2, 0],
            (+0, +0, +1): [0, 90, 0, pi_2],
            (+0, +0, -1): [0, -90, 0, -pi_2],
            # 12 faces
            (+1, +1, +0): [45, 0, pi_4, 0],
            (-1, +1, +0): [135, 0, pi_34, 0],
            (-1, -1, +0): [-135, 0, -pi_34, 0],
            (+1, -1, +0): [-45, 0, -pi_4, 0],
            # --
            (+1, +0, +1): [0, 45, 0, pi_4],
            (-1, +0, +1): [180, 45, pi_1, pi_4],
            (-1, +0, -1): [180, -45, pi_1, -pi_4],
            (+1, +0, -1): [0, -45, 0, -pi_4],
            # --
            (+0, +1, +1): [90, 45, pi_2, pi_4],
            (+0, -1, +1): [-90, 45, -pi_2, pi_4],
            (+0, -1, -1): [-90, -45, -pi_2, -pi_4],
            (+0, +1, -1): [90, -45, pi_2, -pi_4],
            # 8 quadrants
            (+1, +1, +np.sqrt(2)): [45, 45, pi_4, pi_4],
            (-1, +1, +np.sqrt(2)): [135, 45, pi_34, pi_4],
            (-1, -1, +np.sqrt(2)): [-135, 45, -pi_34, pi_4],
            (+1, -1, +np.sqrt(2)): [-45, 45, -pi_4, pi_4],
            # --
            (+1, +1, -np.sqrt(2)): [45, -45, pi_4, -pi_4],
            (-1, +1, -np.sqrt(2)): [135, -45, pi_34, -pi_4],
            (-1, -1, -np.sqrt(2)): [-135, -45, -pi_34, -pi_4],
            (+1, -1, -np.sqrt(2)): [-45, -45, -pi_4, -pi_4],
        }

        for points, values in data.items():

            # 1. Check get_angles

            # 1.a Check angles degree
            coo = np.array([[0, 0, 0], points])
            azm, dip = maps.get_angles(coo)
            assert_allclose(azm, values[0])
            assert_allclose(dip, values[1])

            # 1.b Check angles radians
            azm, dip = maps.get_angles(coo, rad=True)
            assert_allclose(azm, values[2])
            assert_allclose(dip, values[3])

            # 2. Check get_points
            # Center (0, 0, 0); 2*length, so we can compare 2nd point.
            length = 2*np.linalg.norm(points)

            # Check points degree
            coo = maps.get_points(0, 0, 0, values[0], values[1], length)
            assert_allclose(coo[1, :], points)

            # Check points radians
            coo = maps.get_points(
                    0, 0, 0, values[2], values[3], length, rad=True)
            assert_allclose(coo[1, :], points, atol=1e-14)

    def test_get_angles(self):

        points = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),  # 6 axes
            (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),  # 12 faces
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),  # 8 quadrants
            (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)
        ]

        # This is just a test that electrodes.Dipole does the right thing.
        for pts in points:
            coo = np.array([[0, 0, 0], pts])
            s = electrodes.TxElectricDipole(coo)
            azm, dip = maps.get_angles(coo)
            assert_allclose(s.azimuth, azm, rtol=1e-4)
            assert_allclose(s.dip, dip, rtol=1e-4)

    def test_get_points(self):

        # Radians
        i_azm1, i_dip1 = 2*np.pi/3, np.pi/3
        coords1 = maps.get_points(0, 0, 0, i_azm1, i_dip1, 0.5, rad=True)
        o_azm1, o_dip1 = maps.get_angles(coords1, rad=True)
        assert_allclose(i_azm1, o_azm1)
        assert_allclose(i_dip1, o_dip1)

        # Degrees
        i_azm2, i_dip2 = -25, 88
        coords2 = maps.get_points(1e6, 1e-6, 10, i_azm2, i_dip2, 1e6)
        o_azm2, o_dip2 = maps.get_angles(coords2)
        assert_allclose(i_azm2, o_azm2)
        assert_allclose(i_dip2, o_dip2)

        # Degrees: For dip = +/- 90, azm is not defined and returns 0.0
        i_azm4, i_dip4 = -33.3, 90  # <= azm != 0.0
        coords4 = maps.get_points(1e6, -50, 3.33, i_azm4, i_dip4, 2)
        o_azm4, o_dip4 = maps.get_angles(coords4)
        assert_allclose(0.0, o_azm4)  # <= azm == 0.0
        assert_allclose(i_dip4, o_dip4)
