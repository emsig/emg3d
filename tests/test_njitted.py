import numpy as np
from discretize import TensorMesh
from numpy.testing import assert_allclose

from . import alternatives
from emg3d import solver, njitted, utils


def test_amat_x():

    # 1. Compare to alternative amat_x

    # Create a grid
    src = [200, 300, -50., 5, 60]
    grid = TensorMesh([[(100, 16, 1.2)], [(800, 8)], [(500, 4)]], 'CCC')

    # Create some resistivity model
    x = np.arange(1, grid.nCx+1)*2
    y = 1/np.arange(1, grid.nCy+1)
    z = np.arange(1, grid.nCz+1)[::-1]/10
    res_x = np.outer(np.outer(x, y), z).ravel()
    freq = 0.319
    model = utils.Model(grid, res_x, 0.8*res_x, 2*res_x, freq=freq)

    # Create a source field
    sfield = utils.get_source_field(grid=grid, src=src, freq=freq)

    # Run two iterations to get a e-field
    efield = solver.solver(grid, model, sfield, maxit=2, verb=1)

    # amat_x
    rr1 = utils.Field(grid)
    njitted.amat_x(
            rr1.fx, rr1.fy, rr1.fz, efield.fx, efield.fy, efield.fz,
            model.eta_x, model.eta_y, model.eta_z, model.v_mu_r, grid.hx,
            grid.hy, grid.hz)

    # amat_x - alternative
    rr2 = utils.Field(grid)
    alternatives.alt_amat_x(
            rr2.fx, rr2.fy, rr2.fz, efield.fx, efield.fy, efield.fz,
            model.eta_x, model.eta_y, model.eta_z, model.v_mu_r, grid.hx,
            grid.hy, grid.hz)

    # Check all fields (ex, ey, and ez)
    assert_allclose(rr1, rr2)

    # 2. Compare to other solution

    # This should be possible using mesh-internals. Some ideas here:
    #
    # => fx example; fy and fz accordingly
    # Interpolation matrix
    # >>> Px = grid.getInterpolationMat(grid.gridCC, locType='Ex')
    # \eta_x E
    # >>> etaxE = model.eta_x.ravel('F')*Px*pfield.field
    # \nabla \times \E
    # >>> curlE = grid.edgeCurl*pfield.field
    # \nabla \times \mu_r^{-1}
    # >>> curlmu_r = grid.edgeCurl*(model.mu_r.ravel('F')*Px)
    # Collect it all
    # >>> fx = etaxE[:grid.nEx] - (curlmu_r*curlE)[:grid.nEx]
    #
    # This might be good to implement once the mesh.Model is reworked.
    # Places to look for:
    # - EM/Base:
    #     - MfMui
    #     - Problem3D_e => return C.T*MfMui*C + 1j*omega(freq)*MeSigma


def test_gauss_seidel():
    # At the moment we only compare `gauss_seidel_x/y/z` to `gauss_seidel`.
    # Better tests should be implemented.

    # Rotate the source, so we have a strong enough signal in all directions
    src = [0, 0, 0, 45, 45]
    freq = 0.9
    nu = 2  # One back-and-forth

    for ldir in range(1, 4):

        # `gauss_seidel`/`_x/y/z` loop over z, then y, then x. Together with
        # `ldir`, we have to keep the dimension at 2 in order that they agree.
        nx = [4, 1, 1][ldir-1]
        ny = [4, 1, 1][ldir-1]
        nz = [4, 4, 1][ldir-1]

        # Get this grid.
        grid = TensorMesh(
            [[(80, nx, -1.1), (80, nx, 1.1)],
             [(100, ny, -1.3), (100, ny, 1.3)],
             [(200, nz, -1.2), (200, nz, 1.2)]],
            x0='CCC'
        )

        # Initialize model with some resistivities.
        res_x = np.arange(grid.nC)+1
        res_y = 0.5*np.arange(grid.nC)+1
        res_z = 2*np.arange(grid.nC)+1

        model = utils.Model(grid, res_x, res_y, res_z, freq)

        # Initialize source field.
        sfield = utils.get_source_field(grid, src, freq)

        # Run two iterations to get some e-field.
        efield = solver.solver(grid, model, sfield, maxit=2, verb=1)

        inp = (sfield.fx, sfield.fy, sfield.fz, model.eta_x, model.eta_y,
               model.eta_z, model.v_mu_r, grid.hx, grid.hy, grid.hz, nu)

        # Get result from `gauss_seidel`.
        cfield = utils.Field(grid, efield.copy())
        njitted.gauss_seidel_x(cfield.fx, cfield.fy, cfield.fz, *inp)

        # Get result from `gauss_seidel_x/y/z`.
        if ldir == 1:
            njitted.gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)
        elif ldir == 2:
            njitted.gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)
        elif ldir == 3:
            njitted.gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)

        # Check the resulting field.
        assert_allclose(efield.field, cfield.field)


def test_solve():
    # Create complex symmetric matrix A
    amat = np.array([
        [100+100j, 2, 3j, 4, 5, 6],
        [2, 1, 2, 10+10j, 5, 0],
        [3j, 2, 1, 2+10j, 4, 2],
        [4, 10+10j, 2+10j, 1, 3, 4],
        [5, 5, 4, 3, 2, 3+6.j],
        [6, 0, 2, 4, 3+6.j, 40+3j]
    ])

    # Create solution vector x
    x = np.array([1.+1j, 1, 1j, 2+1j, 1+2j, 3+3.j])

    # Calculate b = A x
    b = amat@x

    # 1. Check with numpy
    # Ensure that our dummy-linear-equation system works fine.
    xnp = np.linalg.solve(amat, b)                 # Solve A x = b
    assert_allclose(x, xnp)                        # Check

    # 2. Check the implementation
    # The implemented non-standard Cholesky factorisation uses only the main
    # diagonal and first five lower off-diagonals, arranged in a vector one
    # after the other. Convert matrix A into required vector A.
    avec = np.zeros(36, dtype=complex)
    for i in range(6):
        for j in range(i+1):
            avec[i+5*j] = amat[i, j]
    res1 = b.copy()

    njitted.solve(avec, res1)                      # Solve A x = b
    assert_allclose(x, res1)                       # Check

    # 3. Compare to alternative solver
    res2 = b.copy()
    alternatives.alt_solve(amat.ravel('F'), res2)  # Solve A x = b
    assert_allclose(x, res2)                       # Check


def test_restrict():
    # Simple comparison using the most basic mesh.
    h = np.array([1, 1, 1, 1])

    # Fine and coarse grid.
    fgrid = TensorMesh([h, h, h], x0='CCC')
    cgrid = TensorMesh([np.diff(fgrid.vectorNx[::2]),
                        np.diff(fgrid.vectorNy[::2]),
                        np.diff(fgrid.vectorNz[::2])],
                       x0='CCC')

    # Regular grid, so all weights (wx, wy, wz) are the same...
    w = njitted.restrict_weights(
        fgrid.vectorNx, fgrid.vectorCCx, fgrid.hx, cgrid.vectorNx,
        cgrid.vectorCCx, cgrid.hx)

    # Create fields.
    cfield = utils.Field(cgrid)
    ffield = utils.Field(fgrid)

    # Put in values.
    ffield.fx[:, :, :] = 1
    ffield.fy[:, :, :] = 2
    ffield.fz[:, :, :] = 4

    # Ensure PEC and restrict them.
    ffield.ensure_pec
    njitted.restrict(cfield.fx, cfield.fy, cfield.fz,
                     ffield.fx, ffield.fy, ffield.fz,
                     w, w, w, 0)

    # Check sum of fine and coarse fields.
    assert cfield.fx.sum() == ffield.fx.sum()
    assert cfield.fy.sum() == ffield.fy.sum()
    assert cfield.fz.sum() == ffield.fz.sum()

    # Assert fields are multiples from each other.
    np.allclose(cfield.fx[0, :, :]*2, cfield.fy[:, 0, :])
    np.allclose(cfield.fy[:, 0, :]*2, cfield.fz[:, :, 0])


def test_restrict_weights():
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
    wl, w0, wr = njitted.restrict_weights(
            edges, centr, width, c_edges, c_centr, c_width)

    assert_allclose(wtl, wl)
    assert_allclose(wt0, w0)
    assert_allclose(wtr, wr)

    # 2. Test with stretched grid and compare with alternative formulation

    # Create a highly stretched, non-centered grid
    grid = TensorMesh(
        [[(200, 2, -1.8), (200, 2), (200, 2, 1.8)],
         [(800, 8, -1.2), (800, 8, 1.2)],
         [(400, 4, -1.4), (400, 4, 1.4)]])
    grid.x0 -= [100000, -3000, 100]

    # Create coarse grid thereof
    ch = [np.diff(grid.vectorNx[::2]), np.diff(grid.vectorNy[::2]),
          np.diff(grid.vectorNz[::2])]
    cgrid = TensorMesh(ch, x0=grid.x0)

    # Calculate the weights in a numpy-way, instead of numba-way
    wl, w0, wr = alternatives.alt_restrict_weights(
            grid.vectorNx, grid.vectorCCx, grid.hx, cgrid.vectorNx,
            cgrid.vectorCCx, cgrid.hx)

    # Get the implemented numba-result
    wxl, wx0, wxr = njitted.restrict_weights(
            grid.vectorNx, grid.vectorCCx, grid.hx, cgrid.vectorNx,
            cgrid.vectorCCx, cgrid.hx)

    # Compare
    assert_allclose(wxl, wl)
    assert_allclose(wx0, w0)
    assert_allclose(wxr, wr)


def test_blocks_to_amat():
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
    njitted.blocks_to_amat(amat, bvec, middle1, left1, rhs1, 0, 3)
    njitted.blocks_to_amat(amat, bvec, middle2, left2, rhs2, 1, 3)
    njitted.blocks_to_amat(amat, bvec, middle3, left3, rhs3, 2, 3)

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
