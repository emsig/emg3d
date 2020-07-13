import pytest
import numpy as np
from numpy.testing import assert_allclose

from . import alternatives
from emg3d import fields, maps, meshes


def test_grid2grid_volume():
    # == X == Simple 1D model
    grid_in = meshes.TensorMesh(
            [np.ones(5)*10, np.array([1, ]), np.array([1, ])],
            x0=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.array([10, 25, 10, 5, 2]), np.array([1, ]), np.array([1, ])],
            x0=np.array([-5, 0, 0]))
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
            x0=np.array([0, 0, 0]))
    grid_in = meshes.TensorMesh(
            [np.array([1, ]), np.array([10, 25, 10, 5, 2]), np.array([1, ])],
            x0=np.array([0, -5, 0]))
    values_in = np.array([1, 3.4, 7, 2, 2])[None, :, None]
    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')

    # Result 1st cell: (5*1+5*3.4)/10=2.2
    assert_allclose(values_out[0, :, 0], np.array([2.2, 3.4, 3.4, 7, 2]))

    # == Z == Another 1D test
    grid_in = meshes.TensorMesh(
            [np.array([1, ]), np.array([1, ]), np.ones(9)*10],
            x0=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.array([1, ]), np.array([1, ]), np.array([20, 41, 9, 30])],
            x0=np.array([0, 0, 0]))
    values_in = np.arange(1., 10)[None, None, :]
    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')

    assert_allclose(values_out[0, 0, :], np.array([1.5, 187/41, 7, 260/30]))

    # == 3D ==
    grid_in = meshes.TensorMesh(
            [np.array([1, 1, 1]), np.array([10, 10, 10, 10, 10]),
             np.array([10, 2, 10])], x0=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.array([1, 2, ]), np.array([10, 25, 10, 5, 2]),
             np.array([4, 4, 4])], x0=np.array([0, -5, 6]))
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
             np.array([10, 2, 10])], x0=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.array([1, 2, ]), np.array([5, 25, 10, 5, 5]),
             np.array([9, 4, 9])], x0=np.array([0, 0, 0]))
    create = np.array([[1, 2., 1]])
    create = np.array([create, 2*create, create])
    values_in = create*np.array([1., 5., 3, 7, 2])[None, :, None]
    vol_in = np.outer(np.outer(grid_in.hx, grid_in.hy).ravel('F'), grid_in.hz)
    vol_in = vol_in.ravel('F').reshape(grid_in.vnC, order='F')

    values_out = maps.grid2grid(grid_in, values_in, grid_out, 'volume')
    vol_out = np.outer(np.outer(grid_out.hx, grid_out.hy).ravel('F'),
                       grid_out.hz)
    vol_out = vol_out.ravel('F').reshape(grid_out.vnC, order='F')

    assert_allclose(np.sum(values_out*vol_out), np.sum(values_in*vol_in))


def test_grid2grid():
    igrid = meshes.TensorMesh(
            [np.array([1, 1]), np.array([1]), np.array([1])],
            [0, 0, 0])
    ogrid = meshes.TensorMesh(
            [np.array([1]), np.array([1]), np.array([1])],
            [0, 0, 0])
    values = np.array([1.0, 2.0]).reshape(igrid.vnC)

    # Provide wrong dimension:
    with pytest.raises(ValueError, match='There are 2 points and 1 values'):
        maps.grid2grid(igrid, values[1:, :, :], ogrid)

    # Simple, linear example.
    out = maps.grid2grid(igrid, values, ogrid, 'linear')
    np.allclose(out, np.array([1.5]))

    # Provide ogrid.gridCC.
    ogrid.gridCC = np.array([[0.5, 0.5, 0.5]])
    out2 = maps.grid2grid(igrid, values, ogrid, 'linear')
    np.allclose(out2, np.array([1.5]))

    # Check 'linear' and 'cubic' yield almost the same result for a well
    # determined, very smoothly changing example.

    # Fine grid.
    fgrid = meshes.TensorMesh(
        [np.ones(2**6)*10, np.ones(2**5)*100, np.ones(2**4)*1000],
        x0=np.array([-320., -1600, -8000]))

    # Smoothly changing model for fine grid.
    cmodel = np.arange(1, fgrid.nC+1).reshape(fgrid.vnC, order='F')

    # Coarser grid.
    cgrid = meshes.TensorMesh(
        [np.ones(2**5)*15, np.ones(2**4)*150, np.ones(2**3)*1500],
        x0=np.array([-240., -1200, -6000]))

    # Interpolate linearly and cubic spline.
    lin_model = maps.grid2grid(fgrid, cmodel, cgrid, 'linear')
    cub_model = maps.grid2grid(fgrid, cmodel, cgrid, 'cubic')

    # Compare
    assert np.max(np.abs((lin_model-cub_model)/lin_model*100)) < 1.0

    # Assert it is 'nearest' or extrapolate if points are outside.
    tgrid = meshes.TensorMesh(
            [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]),
             np.array([1, 1, 1, 1])], x0=np.array([0., 0, 0]))
    tmodel = np.ones(tgrid.nC).reshape(tgrid.vnC, order='F')
    tmodel[:, 0, :] = 2
    t2grid = meshes.TensorMesh(
            [np.array([1]), np.array([1]), np.array([1])],
            x0=np.array([2, -1, 2]))

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
            x0=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.arange(7)+1, np.arange(13)+1, np.arange(13)+1],
            x0=np.array([0.5, 3.33, 5]))

    values = np.arange(grid_in.nC, dtype=np.float_).reshape(
            grid_in.vnC, order='F')

    points = (grid_in.vectorNx, grid_in.vectorNy, grid_in.vectorNz)
    new_points = (grid_out.vectorNx, grid_out.vectorNy, grid_out.vectorNz)

    # Compute volume.
    vol = np.outer(np.outer(grid_out.hx, grid_out.hy).ravel('F'), grid_out.hz)
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
def test_volume_avg_weights(njit):
    if njit:
        volume_avg_weights = maps._volume_avg_weights
    else:
        volume_avg_weights = maps._volume_avg_weights.py_func

    grid_in = meshes.TensorMesh(
            [np.ones(11), np.ones(10)*2, np.ones(3)*10],
            x0=np.array([0, 0, 0]))
    grid_out = meshes.TensorMesh(
            [np.arange(4)+1, np.arange(5)+1, np.arange(6)+1],
            x0=np.array([0.5, 3.33, 5]))

    wx, ix_in, ix_out = volume_avg_weights(grid_in.vectorNx, grid_out.vectorNx)
    assert_allclose(wx, [0, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1,
                         1, 0.5, 0])
    assert_allclose(ix_in, [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10])
    assert_allclose(ix_out, [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    wy, iy_in, iy_out = volume_avg_weights(grid_in.vectorNy, grid_out.vectorNy)
    assert_allclose(wy, [0, 0, 0.67, 0.33, 1.67, 0.33, 1.67, 1.33, 0.67, 2.,
                         1.33, 0.67, 2, 2, 0.33, 0.])
    assert_allclose(iy_in, [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9, 9])
    assert_allclose(iy_out, [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4])

    wz, iz_in, iz_out = volume_avg_weights(grid_in.vectorNz, grid_out.vectorNz)
    assert_allclose(wz, [0, 1, 2, 2, 1, 4, 5, 6, 0])
    assert_allclose(iz_in, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    assert_allclose(iz_out, [0, 0, 1, 2, 2, 3, 4, 5, 5])

    w, inp, out = volume_avg_weights(np.array([0., 5, 7, 10]),
                                     np.array([-1., 1, 4, 6, 7, 11]))
    assert_allclose(w, [1, 1, 3, 1, 1, 1, 3, 1])
    assert_allclose(inp, [0, 0, 0, 0, 1, 1, 2, 2])
    assert_allclose(out, [0, 0, 1, 2, 2, 3, 4, 4])
