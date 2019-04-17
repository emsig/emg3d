import pytest
import numpy as np
from discretize import TensorMesh
from os.path import join, dirname
from numpy.testing import assert_allclose

from emg3d import solver, utils, njitted

# Data generated with create_data/regression.py
REGRES = np.load(join(dirname(__file__), 'data/regression.npz'),
                 allow_pickle=True)


def create_dummy(nx, ny, nz):
    """Return complex dummy arrays of shape (nx, ny, nz).

    Numbers are from 1..nx*ny*nz for the real part, and 1/100 of it for the
    imaginary part.

    """
    out = np.arange(1, nx*ny*nz+1) + 1j*np.arange(1, nx*ny*nz+1)/100.
    return out.reshape(nx, ny, nz)


def test_solver(capsys):

    # 1. Regression test for homogeneous halfspace.
    # Not very sophisticated; replace/extend by more detailed tests.
    dat = REGRES['res'][()]

    grid = TensorMesh(**dat['input_grid'])
    model = utils.Model(**dat['input_model'])
    sfield = utils.get_source_field(**dat['input_source'])

    # F-cycle
    efield = solver.solver(grid, model, sfield, verb=4)
    out, _ = capsys.readouterr()

    assert ' emg3d START ::' in out
    assert ' [hh:mm:ss] ' in out
    assert ' MG cycles ' in out
    assert ' Final l2-norm ' in out
    assert ' emg3d END :: ' in out

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Fresult'], efield)

    # W-cycle
    wfield = solver.solver(grid, model, sfield, cycle='W', verb=1)

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Wresult'], wfield)

    # V-cycle
    vfield = solver.solver(grid, model, sfield, cycle='V', verb=1)
    _, _ = capsys.readouterr()  # clear output

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Vresult'], vfield)

    # BiCGSTAB with some print checking.
    efield = solver.solver(grid, model, sfield, verb=3, sslsolver=True)
    out, _ = capsys.readouterr()
    assert ' emg3d START ::' in out
    assert ' [hh:mm:ss] ' in out
    assert ' CONVERGED' in out
    assert ' Solver steps ' in out
    assert ' MG prec. steps ' in out
    assert ' Final l2-norm ' in out
    assert ' emg3d END :: ' in out

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['bicresult'], efield)

    # BiCGSTAB with lower verbosity, print checking.
    _ = solver.solver(grid, model, sfield, verb=2, maxit=1, sslsolver=True)
    out, _ = capsys.readouterr()
    assert ' MAX. ITERATION REACHED' in out

    # Just check if it runs without failing for other solvers.
    _ = solver.solver(grid, model, sfield, verb=3, maxit=1, sslsolver='gmres')
    _ = solver.solver(grid, model, sfield, verb=3, maxit=1,
                      sslsolver='gcrotmk')

    # Provide initial field, ensure there is no output
    _, _ = capsys.readouterr()  # empty
    outarray = solver.solver(grid, model, sfield, efield)
    out, _ = capsys.readouterr()

    assert outarray is None
    assert "Provided efield already good enough!" in out

    # 2. Regression test for heterogeneous case.
    dat = REGRES['reg_2'][()]
    grid = dat['grid']
    model = dat['model']
    sfield = dat['sfield']
    sfield = utils.Field(grid, sfield)
    inp = dat['inp']

    efield = solver.solver(grid, model, sfield, **inp)

    assert_allclose(dat['result'], efield.field)


def test_smoothing():
    # 1. The only thing to test here is that smoothing returns the same as
    #    the corresponding jitted functions. Basically a copy of the function
    #    itself.

    nu = 2

    # Create a grid
    src = [0, -10, -10, 43, 13]
    grid = TensorMesh(
        [[(100, 2)],
         [(10, 27, -1.1), (10, 10), (10, 27, 1.1)],
         [(50, 4, 1.2)]],
        x0=src[:3])

    # Create some resistivity model
    x = np.arange(1, grid.nCx+1)*2
    y = 1/np.arange(1, grid.nCy+1)
    z = np.arange(1, grid.nCz+1)[::-1]/10
    res_x = np.outer(np.outer(x, y), z).ravel()
    freq = 0.319
    model = utils.Model(grid, res_x, 0.8*res_x, 2*res_x, freq=freq)

    # Create a source field
    sfield = utils.get_source_field(grid=grid, src=src, freq=freq)

    # Run two iterations to get an e-field
    field = solver.solver(grid, model, sfield, maxit=2, verb=1)

    # Collect Gauss-Seidel input (same for all routines)
    inp = (sfield.fx, sfield.fy, sfield.fz, model.eta_x, model.eta_y,
           model.eta_z, model.v_mu_r, grid.hx, grid.hy, grid.hz, nu)

    func = ['', '_x', '_y', '_z']
    for ldir in range(4):
        # Get it directly from njitted
        efield = utils.Field(grid, field)
        getattr(njitted, 'gauss_seidel'+func[ldir])(
                efield.fx, efield.fy, efield.fz, *inp)

        # Use solver.smoothing
        ofield = utils.Field(grid, field)
        solver.smoothing(grid, model, sfield, ofield, nu, ldir)

        # Compare
        assert_allclose(efield, ofield)


def test_restriction():

    # Simple test with restriction followed by prolongation.
    src = [0, 0, 0, 0, 45]
    grid = TensorMesh([[(100, 4)], [(100, 4)], [(100, 4)]])

    # Create dummy model and fields, parameters don't matter.
    model = utils.Model(grid, 1, 1, 1, 1)
    sfield = utils.get_source_field(grid, src, 1)

    rx = np.arange(sfield.fx.size, dtype=complex).reshape(sfield.fx.shape)
    ry = np.arange(sfield.fy.size, dtype=complex).reshape(sfield.fy.shape)
    rz = np.arange(sfield.fz.size, dtype=complex).reshape(sfield.fz.shape)
    rr = utils.Field(rx, ry, rz)

    # Restrict it
    cgrid, cmodel, csfield, cefield = solver.restriction(
            grid, model, sfield, rr, rdir=0)

    assert_allclose(csfield.fx[:, 1:-1, 1], np.array([[196.+0.j], [596.+0.j]]))
    assert_allclose(csfield.fy[1:-1, :, 1], np.array([[356.+0.j, 436.+0.j]]))
    assert_allclose(csfield.fz[1:-1, 1:-1, :],
                    np.array([[[388.+0.j, 404.+0.j]]]))
    assert cgrid.nNx == cgrid.nNy == cgrid.nNz == 3
    assert cmodel.eta_x[0, 0, 0]/8. == model.eta_x[0, 0, 0]
    assert np.sum(grid.hx) == np.sum(cgrid.hx)
    assert np.sum(grid.hy) == np.sum(cgrid.hy)
    assert np.sum(grid.hz) == np.sum(cgrid.hz)

    # Add pi to the coarse e-field
    efield = utils.Field(grid)
    cefield += np.pi

    # Prolong it
    solver.prolongation(grid, efield, cgrid, cefield, rdir=0)

    assert np.all(efield.fx[:, 1:-1, 1:-1] == np.pi)
    assert np.all(efield.fy[1:-1, :, 1:-1] == np.pi)
    assert np.all(efield.fz[1:-1, 1:-1, :] == np.pi)


def prolongation():
    pass
    # No test at the moment. Add one!


def test_residual():
    # The only thing to test here is that residual returns the same as
    # sfield-amat_x. Basically a copy of the function itself.

    # Create a grid
    src = [-10, 30, 0., 45, 45]
    grid = TensorMesh(
        [[(33.3, 2, -1.2), (10, 4), (33.3, 2, 1.2)],
         [(200, 16)],
         [(25, 2)]],
        x0=src[:3])

    # Create some resistivity model
    x = np.arange(1, grid.nCx+1)*2
    y = 1/np.arange(1, grid.nCy+1)
    z = np.arange(1, grid.nCz+1)[::-1]/10
    res_x = np.outer(np.outer(x, y), z).ravel()
    freq = 0.319
    model = utils.Model(grid, res_x, 0.8*res_x, 2*res_x, freq=freq)

    # Create a source field
    sfield = utils.get_source_field(grid=grid, src=src, freq=freq)

    # Run two iterations to get an e-field
    efield = solver.solver(grid, model, sfield, maxit=2, verb=1)

    # Use directly amat_x
    rfield = utils.Field(grid)
    njitted.amat_x(
            rfield.fx, rfield.fy, rfield.fz, efield.fx, efield.fy, efield.fz,
            model.eta_x, model.eta_y, model.eta_z, model.v_mu_r,
            grid.hx, grid.hy, grid.hz)

    # Calculate residual
    out = solver.residual(grid, model, sfield, efield)

    # Compare
    assert_allclose(out, sfield-rfield)


def test_mgparameters():
    vnC = (2**3, 2**5, 2**4)

    # 1. semicoarsening
    var = solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=True,
                              linerelaxation=False, vnC=vnC, verb=1)
    assert 'semicoarsening : True [1 2 3]' in var.__repr__()
    var = solver.MGParameters(cycle='V', sslsolver=False, semicoarsening=1213,
                              linerelaxation=False, vnC=vnC, verb=1)
    assert 'semicoarsening : True [1 2 1 3]' in var.__repr__()
    var = solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=2,
                              linerelaxation=False, vnC=vnC, verb=1)
    assert 'semicoarsening : True [2]' in var.__repr__()
    with pytest.raises(ValueError):
        solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=5,
                            linerelaxation=False, vnC=vnC, verb=1)

    # 2. linerelaxation
    var = solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=False,
                              linerelaxation=True, vnC=vnC, verb=1)
    assert 'linerelaxation : True [4 5 6]' in var.__repr__()
    var = solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=False,
                              linerelaxation=1247, vnC=vnC, verb=1)
    assert 'linerelaxation : True [1 2 4 7]' in var.__repr__()
    var = solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=False,
                              linerelaxation=1, vnC=vnC, verb=1, clevel=1)
    assert 'linerelaxation : True [1]' in var.__repr__()
    assert_allclose(var.clevel, 1)
    with pytest.raises(ValueError):
        solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=False,
                            linerelaxation=-9, vnC=vnC, verb=1)

    # 3. sslsolver and cycle
    with pytest.raises(ValueError):
        solver.MGParameters(cycle=None, sslsolver=False, semicoarsening=False,
                            linerelaxation=False, vnC=vnC, verb=1)
    var = solver.MGParameters(cycle='F', sslsolver=True, semicoarsening=True,
                              linerelaxation=False, vnC=vnC, verb=1, maxit=33)
    assert "sslsolver : 'bicgstab'" in var.__repr__()
    assert var.ssl_maxit == 33
    assert var.maxit == 3
    var = solver.MGParameters(cycle='F', sslsolver='gmres', semicoarsening=0,
                              linerelaxation=False, vnC=vnC, verb=1, maxit=5)
    assert "sslsolver : 'gmres'" in var.__repr__()
    assert var.ssl_maxit == 5
    assert var.maxit == 1
    assert_allclose(var.clevel, np.array([4, 4, 3, 4]))
    with pytest.raises(ValueError):
        solver.MGParameters(cycle='F', sslsolver='abcd', semicoarsening=0,
                            linerelaxation=False, vnC=vnC, verb=1)
    with pytest.raises(ValueError):
        solver.MGParameters(cycle='F', sslsolver=4, semicoarsening=0,
                            linerelaxation=False, vnC=vnC, verb=1)
    with pytest.raises(ValueError):
        solver.MGParameters(cycle='G', sslsolver=False, semicoarsening=False,
                            linerelaxation=False, vnC=vnC, verb=1)

    # 4. Wrong grid size
    with pytest.raises(ValueError):
        solver.MGParameters(cycle='F', sslsolver=False, semicoarsening=False,
                            linerelaxation=False, vnC=(1, 2, 3), verb=1)
