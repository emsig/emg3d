import pytest
import numpy as np
from numpy.testing import assert_allclose

from . import alternatives
from emg3d import fields, maps, meshes, models

# Import soft dependencies.
try:
    import discretize
    # Backwards compatibility; remove latest for version 1.0.0.
    dv = discretize.__version__.split('.')
    if int(dv[0]) == 0 and int(dv[1]) < 6:
        discretize = None
except ImportError:
    discretize = None


def test_grid2grid_volume():
    # == X == Simple 1D model
    grid_in = meshes.TensorMesh(
            [np.ones(5)*10, np.array([1, ]), np.array([1, ])],
            origin=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.array([10, 25, 10, 5, 2]), np.array([1, ]), np.array([1, ])],
            origin=np.array([-5, 0, 0]))
    values_in = np.array([1., 5., 3, 7, 2])[:, None, None]
    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')

    # Result 2nd cell: (5*1+10*5+10*3)/25=3.4
    assert_allclose(values_out[:, 0, 0], np.array([1, 3.4, 7, 2, 2]))

    # Check log:
    vlogparam = maps.grid2grid(
            grid_in, values_in, grid_out, 'volume', log=True)
    vlinloginp = maps.grid2grid(
            grid_in, np.log10(values_in), grid_out, 'volume')
    assert_allclose(vlogparam, 10**vlinloginp)

    # == Y ==  Reverse it
    grid_out = meshes.TensorMesh(
            [np.array([1, ]), np.ones(5)*10, np.array([1, ])],
            origin=np.array([0, 0, 0]))
    grid_in = meshes.TensorMesh(
            [np.array([1, ]), np.array([10, 25, 10, 5, 2]), np.array([1, ])],
            origin=np.array([0, -5, 0]))
    values_in = np.array([1, 3.4, 7, 2, 2])[None, :, None]
    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')

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
    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')

    assert_allclose(values_out[0, 0, :], np.array([1.5, 187/41, 7, 260/30]))

    # == 3D ==
    grid_in = meshes.TensorMesh(
            [np.array([1, 1, 1]), np.array([10, 10, 10, 10, 10]),
             np.array([10, 2, 10])], origin=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.array([1, 2, ]), np.array([10, 25, 10, 5, 2]),
             np.array([4, 4, 4])], origin=np.array([0, -5, 6]))
    create = np.array([[1, 2., 1]])
    create = np.array([create, 2*create, create])
    values_in = create*np.array([1., 5., 3, 7, 2])[None, :, None]

    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')

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
    vol_in = vol_in.ravel('F').reshape(grid_in.vnC, order='F')

    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')
    vol_out = np.outer(np.outer(grid_out.h[0], grid_out.h[1]).ravel('F'),
                       grid_out.h[2])
    vol_out = vol_out.ravel('F').reshape(grid_out.vnC, order='F')

    assert_allclose(np.sum(values_out*vol_out), np.sum(values_in*vol_in))


class TestGrid2grid():

    def test_linear(self):
        igrid = meshes.TensorMesh(
                [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])],
                [0, -1, -1])
        ogrid = meshes.TensorMesh(
                [np.array([1]), np.array([1]), np.array([1])],
                [0.5, 0, 0])
        values = np.r_[9*[1.0, ], 9*[2.0, ]].reshape(igrid.vnC)

        # Provide wrong dimension:
        with pytest.raises(ValueError, match='There are 2 points and 1 valu'):
            maps.grid2grid(igrid, values[1:, :, :], ogrid)

        # Simple, linear example.
        out = maps.grid2grid(igrid, values, ogrid, 'linear')
        assert_allclose(out[0, 0, 0], 1.5)

        # Provide ogrid.gridCC.
        ogrid._gridCC = np.array([[0.5, 0.5, 0.5]])
        out2 = maps.grid2grid(igrid, values, ogrid, 'linear')
        assert_allclose(out2[0, 0, 0], 1.5)

    def test_linear_cubic(self):
        # Check 'linear' and 'cubic' yield almost the same result for a well
        # determined, very smoothly changing example.

        # Fine grid.
        fgrid = meshes.TensorMesh(
            [np.ones(2**6)*10, np.ones(2**5)*100, np.ones(2**4)*1000],
            origin=np.array([-320., -1600, -8000]))

        # Smoothly changing model for fine grid.
        cmodel = np.arange(1, fgrid.nC+1).reshape(fgrid.vnC, order='F')

        # Coarser grid.
        cgrid = meshes.TensorMesh(
            [np.ones(2**5)*15, np.ones(2**4)*150, np.ones(2**3)*1500],
            origin=np.array([-240., -1200, -6000]))

        # Interpolate linearly and cubic spline.
        lin_model = maps.grid2grid(fgrid, cmodel, cgrid, 'linear')
        cub_model = maps.grid2grid(fgrid, cmodel, cgrid, 'cubic')

        # Compare
        assert np.max(np.abs((lin_model-cub_model)/lin_model*100)) < 1.0

    def test_nearest(self):
        # Assert it is 'nearest' or extrapolate if points are outside.
        tgrid = meshes.TensorMesh(
                [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]),
                 np.array([1, 1, 1, 1])], origin=np.array([0., 0, 0]))
        tmodel = np.ones(tgrid.nC).reshape(tgrid.vnC, order='F')
        tmodel[:, 0, :] = 2
        t2grid = meshes.TensorMesh(
                [np.array([1]), np.array([1]), np.array([1])],
                origin=np.array([2, -1, 2]))

        # Nearest with cubic.
        out = maps.grid2grid(tgrid, tmodel, t2grid, 'cubic')
        assert_allclose(out, 2.)

        # Same, but with log.
        vlog = maps.grid2grid(tgrid, tmodel, t2grid, 'cubic', log=True)
        vlin = maps.grid2grid(tgrid, np.log10(tmodel), t2grid, 'cubic')
        assert_allclose(vlog, 10**vlin)

        # Extrapolate with linear.
        out = maps.grid2grid(tgrid, tmodel, t2grid, 'linear')
        assert_allclose(out, 3.)

        # Same, but with log.
        vlog = maps.grid2grid(tgrid, tmodel, t2grid, 'linear', log=True)
        vlin = maps.grid2grid(tgrid, np.log10(tmodel), t2grid, 'linear')
        assert_allclose(vlog, 10**vlin)

        # Assert it is 0 if points are outside.
        out = maps.grid2grid(tgrid, tmodel, t2grid, 'cubic', False)
        assert_allclose(out, 0.)
        out = maps.grid2grid(tgrid, tmodel, t2grid, 'linear', False)
        assert_allclose(out, 0.)

    def test_field(self):
        # Provide a Field instance
        grid = meshes.TensorMesh(
                [np.array([1, 2]), np.array([1, 2]), np.array([1, 2])],
                [0, 0, 0])
        cgrid = meshes.TensorMesh(
                [np.array([1.5, 1]), np.array([1.5]), np.array([1.5])],
                [0, 0, 0])
        field = fields.Field(grid)

        # Simple linear interpolation test.
        field.fx = np.arange(1, field.fx.size+1)
        field.fy = np.arange(1, field.fy.size+1)
        field.fz = np.arange(1, field.fz.size+1)

        new_field = maps.grid2grid(grid, field, cgrid, method='linear')
        fx = maps.grid2grid(grid, field.fx, cgrid, method='linear')
        fy = maps.grid2grid(grid, field.fy, cgrid, method='linear')
        fz = maps.grid2grid(grid, field.fz, cgrid, method='linear')
        assert_allclose(fx, new_field.fx)
        assert_allclose(fy, new_field.fy)
        assert_allclose(fz, new_field.fz)

        new_field = maps.grid2grid(grid, field, cgrid, method='cubic')
        fx = maps.grid2grid(grid, field.fx, cgrid, method='cubic')
        fy = maps.grid2grid(grid, field.fy, cgrid, method='cubic')
        fz = maps.grid2grid(grid, field.fz, cgrid, method='cubic')
        assert_allclose(fx, new_field.fx)
        assert_allclose(fy, new_field.fy)
        assert_allclose(fz, new_field.fz)

        # Ensure Field fails with 'volume'.
        with pytest.raises(ValueError, match="``method='volume'`` not impl"):
            maps.grid2grid(grid, field, cgrid, method='volume')


@pytest.mark.parametrize("njit", [True, False])
def test_volume_average(njit):
    if njit:
        volume_average = maps.volume_average
    else:
        volume_average = maps.volume_average.py_func

    # Comparison to alt_version.
    grid_in = meshes.TensorMesh(
            [np.ones(30), np.ones(20)*5, np.ones(10)*10],
            origin=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.arange(7)+1, np.arange(13)+1, np.arange(13)+1],
            origin=np.array([0.5, 3.33, 5]))

    values = np.arange(grid_in.nC, dtype=np.float_).reshape(
            grid_in.vnC, order='F')

    points = (grid_in.nodes_x, grid_in.nodes_y, grid_in.nodes_z)
    new_points = (grid_out.nodes_x, grid_out.nodes_y, grid_out.nodes_z)

    # Compute volume.
    vol = np.outer(np.outer(
        grid_out.h[0], grid_out.h[1]).ravel('F'), grid_out.h[2])
    vol = vol.ravel('F').reshape(grid_out.vnC, order='F')

    # New solution.
    new_values = np.zeros(grid_out.vnC, dtype=values.dtype)
    volume_average(*points, values, *new_points, new_values, vol)

    # Old solution.
    new_values_alt = np.zeros(grid_out.vnC, dtype=values.dtype)
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

    w, inp, out = volume_avg_weights(np.array([0., 5, 7, 10]),
                                     np.array([-1., 1, 4, 6, 7, 11]))
    assert_allclose(w, [1, 1, 3, 1, 1, 1, 3, 1])
    assert_allclose(inp, [0, 0, 0, 0, 1, 1, 2, 2])
    assert_allclose(out, [0, 0, 1, 2, 2, 3, 4, 4])


class TestMaps:
    mesh = meshes.TensorMesh(
            [np.array([1, 1]), np.array([1, 1]), np.array([1])],
            np.array([0, 0, 0]))

    values = np.array([0.01, 10, 3, 4])

    def test_basic(self):
        class MyMap(maps._Map):
            def __init__(self):
                super().__init__('my awesome map')

        testmap = MyMap()

        assert "MyMap: my awesome map" in testmap.__repr__()

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

    def test_to_from_dict(self):
        m = maps.MapLgResistivity()
        d = m.to_dict()
        n = maps.MapLgResistivity.from_dict(d)
        assert m.name == n.name


@pytest.mark.parametrize("njit", [True, False])
def test_edges2cellaverages(njit):
    if njit:
        edges2cellaverages = maps.edges2cellaverages
    else:
        edges2cellaverages = maps.edges2cellaverages.py_func

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
    grad_x = np.zeros(grid.vnC, order='F', dtype=complex)
    grad_y = np.zeros(grid.vnC, order='F', dtype=complex)
    grad_z = np.zeros(grid.vnC, order='F', dtype=complex)

    # Call function.
    vol = grid.cell_volumes.reshape(grid.vnC, order='F')
    edges2cellaverages(field.fx, field.fy, field.fz,
                       vol, grad_x, grad_y, grad_z)
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
            out = grid.aveE2CC*field*grid.cell_volumes
            return out.reshape(grid.vnC, order='F')

        assert_allclose(grad, 3*volume_disc(grid, field))

    if discretize is not None:

        out_x = grid.aveEx2CC*field.fx.ravel('F')*grid.cell_volumes
        out_y = grid.aveEy2CC*field.fy.ravel('F')*grid.cell_volumes
        out_z = grid.aveEz2CC*field.fz.ravel('F')*grid.cell_volumes

        assert_allclose(grad_x, out_x.reshape(grid.vnC, order='F'))
        assert_allclose(grad_y, out_y.reshape(grid.vnC, order='F'))
        assert_allclose(grad_z, out_z.reshape(grid.vnC, order='F'))
