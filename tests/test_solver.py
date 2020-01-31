import pytest
import numpy as np
import scipy.interpolate as si
from os.path import join, dirname
from numpy.testing import assert_allclose

from emg3d import solver, utils, njitted

from .test_utils import get_h

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


def test_solver_homogeneous(capsys):
    # Regression test for homogeneous halfspace.
    # Not very sophisticated; replace/extend by more detailed tests.
    dat = REGRES['res'][()]

    grid = utils.TensorMesh(**dat['input_grid'])
    model = utils.Model(**dat['input_model'])
    sfield = utils.get_source_field(**dat['input_source'])

    # F-cycle
    efield = solver.solve(grid, model, sfield, verb=4)
    out, _ = capsys.readouterr()

    assert ' emg3d START ::' in out
    assert ' [hh:mm:ss] ' in out
    assert ' MG cycles ' in out
    assert ' Final rel. error ' in out
    assert ' emg3d END   :: ' in out

    # Experimental:
    # Check if norms are also the same, at least for first two cycles.
    assert "1.509e-01  after   1 F-cycles   [9.161e-07, 0.151]   0 0" in out
    assert "1.002e-01  after   2 F-cycles   [6.082e-07, 0.664]   0 0" in out

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Fresult'], efield)

    # W-cycle
    wfield = solver.solve(grid, model, sfield, cycle='W', verb=1)

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Wresult'], wfield)

    # V-cycle
    vfield = solver.solve(grid, model, sfield, cycle='V', verb=1)
    _, _ = capsys.readouterr()  # clear output

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Vresult'], vfield)

    # BiCGSTAB with some print checking.
    efield = solver.solve(grid, model, sfield, verb=3, sslsolver=True)
    out, _ = capsys.readouterr()
    assert ' emg3d START ::' in out
    assert ' [hh:mm:ss] ' in out
    assert ' CONVERGED' in out
    assert ' Solver steps ' in out
    assert ' MG prec. steps ' in out
    assert ' Final rel. error ' in out
    assert ' emg3d END   :: ' in out

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['bicresult'], efield)

    # Same as previous, without BiCGSTAB, but some print checking.
    efield = solver.solve(grid, model, sfield, verb=3)
    out, _ = capsys.readouterr()
    assert ' emg3d START ::' in out
    assert ' [hh:mm:ss] ' in out
    assert ' CONVERGED' in out
    assert ' MG cycles ' in out
    assert ' Final rel. error ' in out
    assert ' emg3d END   :: ' in out

    # Max it
    maxit = 2
    _, info = solver.solve(
            grid, model, sfield, verb=2, maxit=maxit, return_info=True)
    out, _ = capsys.readouterr()
    assert ' MAX. ITERATION REACHED' in out
    assert maxit == info['it_mg']
    assert info['exit'] == 1
    assert 'MAX. ITERATION REACHED' in info['exit_message']

    # BiCGSTAB with lower verbosity, print checking.
    _ = solver.solve(grid, model, sfield, verb=2, maxit=1, sslsolver=True)
    out, _ = capsys.readouterr()
    assert ' MAX. ITERATION REACHED' in out

    # Just check if it runs without failing for other solvers.
    _ = solver.solve(grid, model, sfield, verb=3, maxit=1,
                     sslsolver='gcrotmk')

    # Provide initial field.
    _, _ = capsys.readouterr()  # empty
    efield_copy = efield.copy()
    outarray = solver.solve(grid, model, sfield, efield_copy)
    out, _ = capsys.readouterr()

    # Ensure there is no output.
    assert outarray is None
    assert "NOTHING DONE (provided efield already good enough)" in out
    # Ensure the field did not change.
    assert_allclose(efield, efield_copy)

    # Provide initial field and return info.
    info = solver.solve(
            grid, model, sfield, efield_copy, return_info=True)
    assert info['it_mg'] == 0
    assert info['it_ssl'] == 0
    assert info['exit'] == 0
    assert info['exit_message'] == 'CONVERGED'

    # Provide initial field, ensure one initial multigrid is carried out
    # without linerelaxation nor semicoarsening.
    _, _ = capsys.readouterr()  # empty
    efield = utils.Field(grid)
    outarray = solver.solve(
            grid, model, sfield, efield, sslsolver=True, semicoarsening=True,
            linerelaxation=True, maxit=2, verb=3)
    out, _ = capsys.readouterr()
    assert "after                       1 F-cycles    4 1" in out
    assert "after                       2 F-cycles    5 2" in out

    # Provide an initial source-field without frequency information.
    wrong_sfield = utils.Field(grid)
    wrong_sfield.field = sfield.field
    with pytest.raises(ValueError):
        solver.solve(grid, model, wrong_sfield, efield=efield, verb=2)
    out, _ = capsys.readouterr()
    assert "ERROR   :: Source field is missing frequency information" in out

    # Check stagnation by providing an almost zero source field.
    _ = solver.solve(grid, model, sfield*0+1e-20, maxit=100)
    out, _ = capsys.readouterr()
    assert "STAGNATED" in out


def test_solver_heterogeneous(capsys):
    # Regression test for heterogeneous case.
    dat = REGRES['reg_2'][()]
    grid = dat['grid']
    model = dat['model']
    sfield = dat['sfield']
    inp = dat['inp']
    inp['verb'] = 4

    efield = solver.solve(grid, model, sfield, **inp)

    assert_allclose(dat['result'], efield.field)

    # Check with provided e-field; 2x2 iter should yield the same as 4 iter.
    efield2 = solver.solve(grid, model, sfield, maxit=4, verb=1)
    efield3 = solver.solve(grid, model, sfield, maxit=2, verb=1)
    solver.solve(grid, model, sfield, efield3, maxit=2, verb=1)

    assert_allclose(efield2, efield3)

    out, _ = capsys.readouterr()  # Clean up

    # One test without post-smoothing to check if it runs.
    efield4 = solver.solve(
            grid, model, sfield, sslsolver=True, semicoarsening=True,
            linerelaxation=True, maxit=20, nu_pre=0, nu_post=4, verb=3)
    efield5 = solver.solve(
            grid, model, sfield, sslsolver=True, semicoarsening=True,
            linerelaxation=True, maxit=20, nu_pre=4, nu_post=0, verb=3)
    # They don't converge, and hence don't agree. Just a lazy test.
    assert_allclose(efield4, efield5, atol=1e-15, rtol=1e-5)

    # Check the QC plot if it is too long.
    # Coincidently, this one also diverges if nu_pre=0!
    # Mesh: 2-cells in y- and z-direction; 2**9 in x-direction
    mesh = utils.TensorMesh(
            [np.ones(2**9)/np.ones(2**9).sum(), np.ones(2), np.ones(2)],
            x0=np.array([-0.5, -1, -1]))
    sfield = utils.get_source_field(mesh, [0, 0, 0, 0, 0], 1)
    model = utils.Model(mesh)
    _ = solver.solve(mesh, model, sfield, verb=3, nu_pre=0)
    out, _ = capsys.readouterr()
    assert "(Cycle-QC restricted to first 70 steps of 72 steps.)" in out
    assert "DIVERGED" in out


def test_solver_backwards(capsys):
    grid = utils.TensorMesh(
            [np.ones(8), np.ones(8), np.ones(8)], x0=np.array([0, 0, 0]))
    model = utils.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)
    sfield = utils.get_source_field(grid, src=[4, 4, 4, 0, 0], freq=10.0)

    out, _ = capsys.readouterr()
    _ = solver.solver(grid, model, sfield, verb=0)
    out, _ = capsys.readouterr()
    assert "* WARNING :: ``emg3d.solver.solver()`` is renamed to " in out


def test_one_liner(capsys):
    grid = utils.TensorMesh(
            [np.ones(8), np.ones(8), np.ones(8)], x0=np.array([0, 0, 0]))
    model = utils.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)
    sfield = utils.get_source_field(grid, src=[4, 4, 4, 0, 0], freq=10.0)

    out, _ = capsys.readouterr()
    _ = solver.solve(grid, model, sfield, verb=-1)
    out, _ = capsys.readouterr()
    assert '6; 0:00:' in out
    assert '; CONVERGED' in out

    out, _ = capsys.readouterr()
    _ = solver.solve(grid, model, sfield, sslsolver=True, verb=-1)
    out, _ = capsys.readouterr()
    assert '3(5); 0:00:' in out
    assert '; CONVERGED' in out


def test_solver_homogeneous_laplace():
    # Regression test for homogeneous halfspace in Laplace domain.
    # Not very sophisticated; replace/extend by more detailed tests.
    dat = REGRES['lap'][()]

    grid = utils.TensorMesh(**dat['input_grid'])
    model = utils.Model(**dat['input_model'])
    sfield = utils.get_source_field(**dat['input_source'])

    # F-cycle
    efield = solver.solve(grid, model, sfield, verb=1)

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['Fresult'], efield)

    # BiCGSTAB with some print checking.
    efield = solver.solve(grid, model, sfield, verb=1, sslsolver=True)

    # Check all fields (ex, ey, and ez)
    assert_allclose(dat['bicresult'], efield)

    # If efield is complex, assert it fails.
    efield = utils.Field(grid, dtype=complex)

    with pytest.raises(ValueError):
        efield = solver.solve(grid, model, sfield, efield=efield, verb=1)


def multigrid():
    pass
    # No test at the moment. Add one!


def test_smoothing():
    # 1. The only thing to test here is that smoothing returns the same as
    #    the corresponding jitted functions. Basically a copy of the function
    #    itself.

    nu = 2

    widths = [np.ones(2)*100, get_h(10, 27, 10, 1.1), get_h(2, 1, 50, 1.2)]
    x0 = [-w.sum()/2 for w in widths]
    src = [0, -10, -10, 43, 13]

    # Loop and move the 2-cell dimension (100, 2) from x to y to z.
    for xyz in range(3):

        # Create a grid
        grid = utils.TensorMesh(
            [widths[xyz % 3],
             widths[(xyz+1) % 3],
             widths[(xyz+2) % 3]],
            x0=np.array([x0[xyz % 3], x0[(xyz+1) % 3], x0[(xyz+2) % 3]])
        )

        # Create some resistivity model
        x = np.arange(1, grid.nCx+1)*2
        y = 1/np.arange(1, grid.nCy+1)
        z = np.arange(1, grid.nCz+1)[::-1]/10
        res_x = np.outer(np.outer(x, y), z).ravel()
        freq = 0.319
        model = utils.Model(grid, res_x, 0.8*res_x, 2*res_x)

        # Create a source field
        sfield = utils.get_source_field(grid=grid, src=src, freq=freq)

        # Get volume-averaged model parameters.
        vmodel = utils.VolumeModel(grid, model, sfield)

        # Run two iterations to get an e-field
        field = solver.solve(grid, model, sfield, maxit=2, verb=1)

        # Collect Gauss-Seidel input (same for all routines)
        inp = (sfield.fx, sfield.fy, sfield.fz, vmodel.eta_x, vmodel.eta_y,
               vmodel.eta_z, vmodel.zeta, grid.hx, grid.hy, grid.hz, nu)

        func = ['', '_x', '_y', '_z']
        for lr_dir in range(8):
            # Get it directly from njitted
            efield = utils.Field(grid, field)
            if lr_dir < 4:
                getattr(njitted, 'gauss_seidel'+func[lr_dir])(
                        efield.fx, efield.fy, efield.fz, *inp)
            elif lr_dir == 4:
                njitted.gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)
                njitted.gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)
            elif lr_dir == 5:
                njitted.gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)
                njitted.gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)
            elif lr_dir == 6:
                njitted.gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)
                njitted.gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)
            elif lr_dir == 7:
                njitted.gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)
                njitted.gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)
                njitted.gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)

            # Use solver.smoothing
            ofield = utils.Field(grid, field)
            solver.smoothing(grid, vmodel, sfield, ofield, nu, lr_dir)

            # Compare
            assert_allclose(efield, ofield)


def test_restriction():

    # Simple test with restriction followed by prolongation.
    src = [0, 0, 0, 0, 45]
    grid = utils.TensorMesh(
            [np.ones(4)*100, np.ones(4)*100, np.ones(4)*100], x0=np.zeros(3))

    # Create dummy model and fields, parameters don't matter.
    model = utils.Model(grid, 1, 1, 1, 1)
    sfield = utils.get_source_field(grid, src, 1)

    # Get volume-averaged model parameters.
    vmodel = utils.VolumeModel(grid, model, sfield)

    rx = np.arange(sfield.fx.size, dtype=complex).reshape(sfield.fx.shape)
    ry = np.arange(sfield.fy.size, dtype=complex).reshape(sfield.fy.shape)
    rz = np.arange(sfield.fz.size, dtype=complex).reshape(sfield.fz.shape)
    rr = utils.Field(rx, ry, rz)

    # Restrict it
    cgrid, cmodel, csfield, cefield = solver.restriction(
            grid, vmodel, sfield, rr, sc_dir=0)

    assert_allclose(csfield.fx[:, 1:-1, 1], np.array([[196.+0.j], [596.+0.j]]))
    assert_allclose(csfield.fy[1:-1, :, 1], np.array([[356.+0.j, 436.+0.j]]))
    assert_allclose(csfield.fz[1:-1, 1:-1, :],
                    np.array([[[388.+0.j, 404.+0.j]]]))
    assert cgrid.nNx == cgrid.nNy == cgrid.nNz == 3
    assert cmodel.eta_x[0, 0, 0]/8. == vmodel.eta_x[0, 0, 0]
    assert np.sum(grid.hx) == np.sum(cgrid.hx)
    assert np.sum(grid.hy) == np.sum(cgrid.hy)
    assert np.sum(grid.hz) == np.sum(cgrid.hz)

    # Add pi to the coarse e-field
    efield = utils.Field(grid)
    cefield += np.pi

    # Prolong it
    solver.prolongation(grid, efield, cgrid, cefield, sc_dir=0)

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
    src = [90, 1600, 25., 45, 45]
    grid = utils.TensorMesh(
        [get_h(4, 2, 20, 1.2), np.ones(16)*200, np.ones(2)*25], x0=np.zeros(3))

    # Create some resistivity model
    x = np.arange(1, grid.nCx+1)*2
    y = 1/np.arange(1, grid.nCy+1)
    z = np.arange(1, grid.nCz+1)[::-1]/10
    res_x = np.outer(np.outer(x, y), z).ravel()
    freq = 0.319
    model = utils.Model(grid, res_x, 0.8*res_x, 2*res_x)

    # Create a source field
    sfield = utils.get_source_field(grid=grid, src=src, freq=freq)

    # Get volume-averaged model parameters.
    vmodel = utils.VolumeModel(grid, model, sfield)

    # Run two iterations to get an e-field
    efield = solver.solve(grid, model, sfield, maxit=2, verb=1)

    # Use directly amat_x
    rfield = sfield.copy()
    njitted.amat_x(
            rfield.fx, rfield.fy, rfield.fz, efield.fx, efield.fy, efield.fz,
            vmodel.eta_x, vmodel.eta_y, vmodel.eta_z, vmodel.zeta, grid.hx,
            grid.hy, grid.hz)

    # Calculate residual
    out = solver.residual(grid, vmodel, sfield, efield)
    outnorm = solver.residual(grid, vmodel, sfield, efield, True)

    # Compare
    assert_allclose(out, rfield)
    assert_allclose(outnorm, np.linalg.norm(out))


def test_krylov(capsys):

    # Everything should be tested just fine in `test_solver`.
    # Just check here for bicgstab-error.

    # Load any case.
    dat = REGRES['res'][()]
    grid = utils.TensorMesh(**dat['input_grid'])
    model = utils.Model(**dat['input_model'])
    sfield = utils.get_source_field(**dat['input_source'])
    vmodel = utils.VolumeModel(grid, model, sfield)
    efield = utils.Field(grid)  # Initiate e-field.

    # Get var-instance
    var = solver.MGParameters(
            cycle=None, sslsolver=True, semicoarsening=False,
            linerelaxation=False, vnC=grid.vnC, verb=3,
            maxit=-1,  # Set stupid input to make bicgstab fail.
    )
    var.l2_refe = njitted.l2norm(sfield)

    # Call krylov and ensure it fails properly.
    solver.krylov(grid, vmodel, sfield, efield, var)
    out, _ = capsys.readouterr()
    assert '* ERROR   :: Error in bicgstab' in out


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


def test_RegularGridProlongator():

    def prolon_scipy(grid, cgrid, efield, cefield, yz_points):
        """Calculate SciPy alternative."""
        for ixc in range(cgrid.nCx):
            # Bilinear interpolation in the y-z plane
            fn = si.RegularGridInterpolator(
                    (cgrid.vectorNy, cgrid.vectorNz), cefield.fx[ixc, :, :],
                    bounds_error=False, fill_value=None)
            hh = fn(yz_points).reshape(grid.vnEx[1:], order='F')

            # Piecewise constant interpolation in x-direction
            efield[2*ixc, :, :] += hh
            efield[2*ixc+1, :, :] += hh

        return efield

    def prolon_emg3d(grid, cgrid, efield, cefield, yz_points):
        """Calculate emg3d alternative."""
        fn = solver.RegularGridProlongator(
                cgrid.vectorNy, cgrid.vectorNz, yz_points)

        for ixc in range(cgrid.nCx):
            # Bilinear interpolation in the y-z plane
            hh = fn(cefield.fx[ixc, :, :]).reshape(grid.vnEx[1:], order='F')

            # Piecewise constant interpolation in x-direction
            efield[2*ixc, :, :] += hh
            efield[2*ixc+1, :, :] += hh

        return efield

    # Create fine grid.
    nx = 2**7
    hx = 50*np.ones(nx)
    hx = np.array([4, 1.1, 2, 3])
    hy = np.array([2, 0.1, 20, np.pi])
    hz = np.array([1, 2, 5, 1])
    grid = utils.TensorMesh([hx, hy, hz], x0=np.array([0, 0, 0]))

    # Create coarse grid.
    chx = np.diff(grid.vectorNx[::2])
    cgrid = utils.TensorMesh([chx, chx, chx], x0=np.array([0, 0, 0]))

    # Create empty fine grid fields.
    efield1 = utils.Field(grid)
    efield2 = utils.Field(grid)

    # Create coarse grid field with some values.
    cefield = utils.Field(cgrid)
    cefield.fx = np.arange(cefield.fx.size)
    cefield.fx = 1j*np.arange(cefield.fx.size)/10

    # Required interpolation points.
    yz_points = solver._get_prolongation_coordinates(grid, 'y', 'z')

    # Compare
    out1 = prolon_scipy(grid, cgrid, efield1.fx, cefield, yz_points)
    out2 = prolon_emg3d(grid, cgrid, efield2.fx, cefield, yz_points)

    assert_allclose(out1, out2)
