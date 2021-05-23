import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import core

from . import alternatives, helpers


@pytest.mark.parametrize("njit", [True, False])
def test_amat_x(njit):
    if njit:
        amat_x = core.amat_x
    else:
        amat_x = core.amat_x.py_func

    # 1. Compare to alternative amat_x

    # Create a grid
    src = [200, 300, -50., 5, 60]
    hx = helpers.widths(8, 4, 100, 1.2)
    hy = np.ones(8)*800
    hz = np.ones(4)*500
    grid = emg3d.TensorMesh(
            h=[hx, hy, hz],
            origin=np.array([-hx.sum()/2, -hy.sum()/2, -hz.sum()/2]))

    # Create some resistivity model
    x = np.arange(1, grid.shape_cells[0]+1)*2
    y = 1/np.arange(1, grid.shape_cells[1]+1)
    z = np.arange(1, grid.shape_cells[2]+1)[::-1]/10
    property_x = np.outer(np.outer(x, y), z).ravel()
    freq = 0.319
    model = emg3d.Model(
            grid=grid, property_x=property_x, property_y=0.8*property_x,
            property_z=2*property_x)

    # Create a source field
    sfield = emg3d.get_source_field(grid=grid, source=src, frequency=freq)

    # Get volume-averaged model parameters.
    vmodel = emg3d.models.VolumeModel(model, sfield)

    # Run two iterations to get a e-field
    efield = emg3d.solve(
            model=model, sfield=sfield, sslsolver=False, semicoarsening=False,
            linerelaxation=False, maxit=2, verb=1)

    # amat_x
    rr1 = emg3d.Field(grid)
    amat_x(rr1.fx, rr1.fy, rr1.fz, efield.fx, efield.fy, efield.fz,
           vmodel.eta_x, vmodel.eta_y, vmodel.eta_z, vmodel.zeta, grid.h[0],
           grid.h[1], grid.h[2])

    # amat_x - alternative
    rr2 = emg3d.Field(grid)
    alternatives.alt_amat_x(
            rr2.fx, rr2.fy, rr2.fz, efield.fx, efield.fy, efield.fz,
            vmodel.eta_x, vmodel.eta_y, vmodel.eta_z, vmodel.zeta, grid.h[0],
            grid.h[1], grid.h[2])

    # Check all fields (ex, ey, and ez)
    assert_allclose(-rr1.field, rr2.field, atol=1e-23)


@pytest.mark.parametrize("njit", [True, False])
def test_gauss_seidel(njit):
    if njit:
        gauss_seidel = core.gauss_seidel
        gauss_seidel_x = core.gauss_seidel_x
        gauss_seidel_y = core.gauss_seidel_y
        gauss_seidel_z = core.gauss_seidel_z
    else:
        gauss_seidel = core.gauss_seidel.py_func
        gauss_seidel_x = core.gauss_seidel_x.py_func
        gauss_seidel_y = core.gauss_seidel_y.py_func
        gauss_seidel_z = core.gauss_seidel_z.py_func

    # At the moment we only compare `gauss_seidel_x/y/z` to `gauss_seidel`.
    # Better tests would always be welcomed...

    # Rotate the source, so we have a strong enough signal in all directions
    src = [0, 0, 0, 45, 45]
    freq = 0.9
    nu = 2  # One back-and-forth

    for lr_dir in range(1, 4):

        # `gauss_seidel`/`_x/y/z` loop over z, then y, then x. Together with
        # `lr_dir`, we have to keep the dimension at 2 in order that they
        # agree.
        nx = [1, 4, 4][lr_dir-1]
        ny = [4, 1, 4][lr_dir-1]
        nz = [4, 4, 1][lr_dir-1]

        # Get this grid.
        hx = helpers.widths(0, nx, 80, 1.1)
        hy = helpers.widths(0, ny, 100, 1.3)
        hz = helpers.widths(0, nz, 200, 1.2)
        grid = emg3d.TensorMesh(
            [hx, hy, hz], np.array([-hx.sum()/2, -hy.sum()/2, -hz.sum()/2]))

        # Initialize model with some resistivities.
        property_x = np.arange(grid.n_cells)+1
        property_y = 0.5*np.arange(grid.n_cells)+1
        property_z = 2*np.arange(grid.n_cells)+1

        model = emg3d.Model(grid, property_x, property_y, property_z)

        # Initialize source field.
        sfield = emg3d.get_source_field(grid, src, freq)

        # Get volume-averaged model parameters.
        vmodel = emg3d.models.VolumeModel(model, sfield)

        # Run two iterations to get some e-field.
        efield = emg3d.solve(model, sfield, sslsolver=False,
                             semicoarsening=False, linerelaxation=False,
                             maxit=2, verb=1)

        inp = (sfield.fx, sfield.fy, sfield.fz, vmodel.eta_x, vmodel.eta_y,
               vmodel.eta_z, vmodel.zeta, grid.h[0], grid.h[1], grid.h[2], nu)

        # Get result from `gauss_seidel`.
        cfield = emg3d.Field(grid, efield.field.copy(),
                             frequency=efield._frequency)
        gauss_seidel(cfield.fx, cfield.fy, cfield.fz, *inp)

        # Get result from `gauss_seidel_x/y/z`.
        if lr_dir == 1:
            gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)
        elif lr_dir == 2:
            gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)
        elif lr_dir == 3:
            gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)

        # Check the resulting field.
        assert efield == cfield


@pytest.mark.parametrize("njit", [True, False])
def test_blocks_to_amat(njit):
    if njit:
        blocks_to_amat = core.blocks_to_amat
    else:
        blocks_to_amat = core.blocks_to_amat.py_func

    # Create some dummy data

    # Main matrix amat and RHS bvec
    # Size: 3 blocks of 5x5; 6 diagonals, hence 3*5*6
    amat = np.zeros(90)
    bvec = np.zeros(15)

    # Middle blocks
    middle1 = np.array([1, 2, 3, 4, 5, -1, 7, 8, 9, 10, -1, -1, 13,
                        14, 15, -1, -1, -1, 19, 20, -1, -1, -1, -1, 25])
    middle2 = np.array([31, 32, 33, 34, 35, -1, 37, 38, 39, 40, -1, -1,
                        43, 44, 45, -1, -1, -1, 49, 50, -1, -1, -1, -1, 55])
    middle3 = np.array([61, 62, 63, 64, 65, -1, 67, 68, 69, 70, -1, -1,
                        73, 74, 75, -1, -1, -1, 79, 80, -1, -1, -1, -1, 85])

    # Left blocks
    left1 = -np.ones(25)
    left2 = np.array([6, -1, -1, -1, -1, 11, 12, -1, -1, -1, 16, 17, 18,
                      -1, -1, 21, 22, 23, 24, -1, 26, 27, 28, 29, 30])
    left3 = np.array([36, -1, -1, -1, -1, 41, 42, -1, -1, -1, 46, 47, 48,
                      -1, -1, 51, 52, 53, 54, -1, 56, 57, 58, 59, 60])

    # RHS
    rhs1 = np.arange(1, 6)
    rhs2 = np.arange(6, 11)
    rhs3 = np.arange(11, 16)

    # Call blocks_to_amat three times to fill in all blocks
    blocks_to_amat(amat, bvec, middle1, left1, rhs1, 0, 3)
    blocks_to_amat(amat, bvec, middle2, left2, rhs2, 1, 3)
    blocks_to_amat(amat, bvec, middle3, left3, rhs3, 2, 3)

    # The result should look like this
    amat_res = np.arange(1., 91)
    amat_res[5] = amat_res[35] = amat_res[41] = 0
    amat_res[46:48] = 0
    amat_res[51:54] = 0
    amat_res[56:60] = 0
    amat_res[61:] = 0
    bvec_res = np.arange(1., 16)
    bvec_res[11:] = 0

    # Check it
    assert_allclose(amat_res, amat)
    assert_allclose(bvec_res, bvec)


@pytest.mark.parametrize("njit", [True, False])
def test_solve(njit):
    if njit:
        solve = core.solve
    else:
        solve = core.solve.py_func

    # Create real and complex symmetric matrix A in the form as required by
    # core.solve, hence only main diagonal and five lower diagonals in a
    # vector.

    # Create real symmetric matrix A.
    avec_real = np.zeros(36, dtype=np.float64)
    avec_real[::6] = np.array([100, 1, 1, 1, 2, 40])
    avec_real[1:-6:6] = np.array([2, 2, 2, 3, 3])
    avec_real[2:-12:6] = np.array([3, 10, 4, 4])
    avec_real[3:-18:6] = np.array([4, 5, 2])
    avec_real[4:-24:6] = np.array([5, 6])

    # Create complex symmetric matrix A.
    avec_complex = np.zeros(36, dtype=np.complex128)
    avec_complex[::6] = np.array([100+100j, 1, 1, 1, 2, 40+3j])
    avec_complex[1:-6:6] = np.array([2, 2, 2+10j, 3, 3+6j])
    avec_complex[2:-12:6] = np.array([3j, 10+10j, 4, 4])
    avec_complex[3:-18:6] = np.array([4, 5, 2])
    avec_complex[4:-24:6] = np.array([5, 6])

    # Create solution vector x
    x_real = np.array([10., 3., 2., 40., 4., 3.])
    x_complex = np.array([1.+1j, 1, 1j, 2+1j, 1+2j, 3+3.j])

    for avec, x in zip([avec_complex, avec_real], [x_complex, x_real]):
        # Re-arrange to full (symmetric) other numpy solvers.
        amat = np.zeros((6, 6), dtype=avec.dtype)
        for i in range(6):
            for j in range(i+1):
                amat[i, j] = avec[i+5*j]
                amat[j, i] = avec[i+5*j]

        # Compute b = A x
        b = amat@x

        # 1. Check with numpy
        # Ensure that our dummy-linear-equation system works fine.
        xnp = np.linalg.solve(amat, b)                 # Solve A x = b
        assert_allclose(x, xnp)                        # Check

        # 2. Check the implementation
        # The implemented non-standard Cholesky factorisation uses only the
        # main diagonal and first five lower off-diagonals, arranged in a
        # vector one after the other. Convert matrix A into required vector A.
        res1 = b.copy()
        solve(avec.copy(), res1)               # Solve A x = b
        assert_allclose(x, res1)                       # Check

        # 3. Compare to alternative solver

        # Re-arrange to full, F-ordered vector for alternative solver.
        amat = np.zeros(36, dtype=avec.dtype)
        for i in range(6):
            for j in range(i+1):
                amat[i+6*j] = avec[i+5*j]
                amat[j+6*i] = avec[i+5*j]

        res2 = b.copy()
        alternatives.alt_solve(amat, res2)             # Solve A x = b
        assert_allclose(x, res2)                       # Check


@pytest.mark.parametrize("njit", [True, False])
def test_restrict(njit):
    if njit:
        restrict = core.restrict
        restrict_weights = core.restrict_weights
    else:
        restrict = core.restrict.py_func
        restrict_weights = core.restrict_weights.py_func

    # Simple comparison using the most basic mesh.
    h = np.array([1, 1, 1, 1, 1, 1])

    # Fine grid.
    fgrid = emg3d.TensorMesh([h, h, h], origin=np.array([-3, -3, -3]))

    # Create fine field.
    ffield = emg3d.Field(fgrid)

    # Put in values.
    ffield.fx[:, 1:-1, 1:-1] = 1
    ffield.fy[1:-1, :, 1:-1] = 2
    ffield.fz[1:-1, 1:-1, :] = 4

    # Get weigths
    wlr = np.zeros(fgrid.shape_nodes[0], dtype=np.float64)
    w0 = np.ones(fgrid.shape_nodes[0], dtype=np.float64)
    fw = (wlr, w0, wlr)

    # # CASE 0 -- regular # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh([np.diff(fgrid.nodes_x[::2]),
                              np.diff(fgrid.nodes_y[::2]),
                              np.diff(fgrid.nodes_z[::2])],
                             fgrid.origin)

    # Regular grid, so all weights (wx, wy, wz) are the same...
    w = restrict_weights(fgrid.nodes_x, fgrid.cell_centers_x, fgrid.h[0],
                         cgrid.nodes_x, cgrid.cell_centers_x, cgrid.h[0])

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict it.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             w, w, w, 0)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fx[0, :, :]*2, cfield.fy[:, 0, :])
    assert_allclose(cfield.fy[:, 0, :]*2, cfield.fz[:, :, 0])

    # # CASE 1 -- y & z # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh([fgrid.h[0], np.diff(fgrid.nodes_y[::2]),
                              np.diff(fgrid.nodes_z[::2])], fgrid.origin)

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict field.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             fw, w, w, 1)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fy[:, 0, :]*2, cfield.fz[:, :, 0])

    # # CASE 2 -- x & z # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh([np.diff(fgrid.nodes_x[::2]), fgrid.h[1],
                              np.diff(fgrid.nodes_z[::2])], fgrid.origin)

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict field.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             w, fw, w, 2)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fx[0, :, :].T*4, cfield.fz[:, :, 0])

    # # CASE 3 -- x & y # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh([np.diff(fgrid.nodes_x[::2]),
                              np.diff(fgrid.nodes_y[::2]), fgrid.h[2]],
                             fgrid.origin)

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict field.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             w, w, fw, 3)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fx[0, :, :]*2, cfield.fy[:, 0, :])

    # # CASE 4 -- x # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh(
            [np.diff(fgrid.nodes_x[::2]), fgrid.h[1], fgrid.h[2]],
            fgrid.origin)

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict field.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             w, fw, fw, 4)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fy[:, 0, :]*2, cfield.fz[:, :, 0])

    # # CASE 5 -- y # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh(
            [fgrid.h[0], np.diff(fgrid.nodes_y[::2]), fgrid.h[2]],
            fgrid.origin)

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict field.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             fw, w, fw, 5)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fx[0, :, :].T*4, cfield.fz[:, :, 0])

    # # CASE 6 -- z # #

    # Coarse grid.
    cgrid = emg3d.TensorMesh(
            [fgrid.h[0], fgrid.h[1], np.diff(fgrid.nodes_z[::2])],
            fgrid.origin)

    # Create coarse field.
    cfield = emg3d.Field(cgrid)

    # Restrict field.
    restrict(cfield.fx, cfield.fy, cfield.fz, ffield.fx, ffield.fy, ffield.fz,
             fw, fw, w, 6)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    assert_allclose(cfield.fx[0, :, :]*2, cfield.fy[:, 0, :])


@pytest.mark.parametrize("njit", [True, False])
def test_restrict_weights(njit):
    if njit:
        restrict_weights = core.restrict_weights
    else:
        restrict_weights = core.restrict_weights.py_func

    # 1. Simple example following equation 9, [Muld06]_.
    edges = np.array([0., 500, 1200, 2000, 3000])
    width = (edges[1:]-edges[:-1])
    centr = edges[:-1]+width/2
    c_edges = edges[::2]
    c_width = (c_edges[1:] - c_edges[:-1])
    c_centr = c_edges[:-1] + c_width/2

    # Result
    wtl = np.array([350/250, 250/600, 400/900])
    wt0 = np.array([1., 1., 1.])
    wtr = np.array([350/600, 500/900, 400/500])

    # Result from implemented function
    wl, w0, wr = restrict_weights(
            edges, centr, width, c_edges, c_centr, c_width)

    assert_allclose(wtl, wl)
    assert_allclose(wt0, w0)
    assert_allclose(wtr, wr)

    # 2. Test with stretched grid and compare with alternative formulation

    # Create a highly stretched, non-centered grid
    hx = helpers.widths(2, 2, 200, 1.8)
    hy = helpers.widths(0, 8, 800, 1.2)
    hz = helpers.widths(0, 4, 400, 1.4)
    grid = emg3d.TensorMesh(
            [hx, hy, hz], np.array([-100000, 3000, 100]))

    # Create coarse grid thereof
    ch = [np.diff(grid.nodes_x[::2]), np.diff(grid.nodes_y[::2]),
          np.diff(grid.nodes_z[::2])]
    cgrid = emg3d.TensorMesh(ch, origin=grid.origin)

    # Compute the weights in a numpy-way, instead of numba-way
    wl, w0, wr = alternatives.alt_restrict_weights(
            grid.nodes_x, grid.cell_centers_x, grid.h[0], cgrid.nodes_x,
            cgrid.cell_centers_x, cgrid.h[0])

    # Get the implemented numba-result
    wxl, wx0, wxr = restrict_weights(
            grid.nodes_x, grid.cell_centers_x, grid.h[0], cgrid.nodes_x,
            cgrid.cell_centers_x, cgrid.h[0])

    # Compare
    assert_allclose(wxl, wl)
    assert_allclose(wx0, w0)
    assert_allclose(wxr, wr)
