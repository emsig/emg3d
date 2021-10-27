import pytest
import numpy as np
import scipy.linalg as sl
import scipy.interpolate as si
from os.path import join, dirname
from numpy.testing import assert_allclose

import emg3d
from emg3d import solver

from . import alternatives, helpers

# Data generated with tests/create_data/regression.py
REGRES = emg3d.load(join(dirname(__file__), 'data', 'regression.npz'))


class TestSolve:
    def test_homogeneous(self, capsys):
        # Regression test for homogeneous halfspace.
        dat = REGRES['res']

        model = emg3d.Model(**dat['input_model'])
        grid = model.grid
        sfield = emg3d.get_source_field(**dat['input_source'])

        # F-cycle
        efield = solver.solve(model=model, sfield=sfield, plain=True, verb=4)
        out, _ = capsys.readouterr()

        assert ' emg3d START ::' in out
        assert ' [hh:mm:ss] ' in out
        assert ' MG cycles ' in out
        assert ' Final rel. error ' in out
        assert ' emg3d END   :: ' in out

        # Experimental:
        # Check if norms are also the same, at least for first two cycles.
        assert "3.399e-02  after   1 F-cycles   [1.830e-07, 0.034]   0 " in out
        assert "3.535e-03  after   2 F-cycles   [1.903e-08, 0.104]   0 " in out

        # Check all fields (ex, ey, and ez)
        assert_allclose(dat['Fresult'].field, efield.field)

        # W-cycle
        wfield = solver.solve(model, sfield, plain=True, cycle='W')

        # Check all fields (ex, ey, and ez)
        assert_allclose(dat['Wresult'].field, wfield.field)

        # V-cycle
        vfield = solver.solve(model, sfield, plain=True, cycle='V')
        _, _ = capsys.readouterr()  # clear output

        # Check all fields (ex, ey, and ez)
        assert_allclose(dat['Vresult'].field, vfield.field)

        # BiCGSTAB with some print checking.
        efield = solver.solve(model, sfield, verb=4, sslsolver='bicgstab',
                              plain=True)
        out, _ = capsys.readouterr()
        assert ' emg3d START ::' in out
        assert ' [hh:mm:ss] ' in out
        assert ' CONVERGED' in out
        assert ' Solver steps ' in out
        assert ' MG prec. steps ' in out
        assert ' Final rel. error ' in out
        assert ' emg3d END   :: ' in out

        # Check all fields (ex, ey, and ez)
        assert_allclose(dat['bicresult'].field, efield.field)

        # Same as previous, without BiCGSTAB, but some print checking.
        efield = solver.solve(model, sfield, plain=True, verb=4)
        out, _ = capsys.readouterr()
        assert ' emg3d START ::' in out
        assert ' [hh:mm:ss] ' in out
        assert ' CONVERGED' in out
        assert ' MG cycles ' in out
        assert ' Final rel. error ' in out
        assert ' emg3d END   :: ' in out

        # Max it
        maxit = 2
        _, info = solver.solve(model, sfield, plain=True, verb=3, maxit=maxit,
                               return_info=True)
        out, _ = capsys.readouterr()
        assert ' MAX. ITERATION REACHED' in out
        assert maxit == info['it_mg']
        assert info['exit'] == 1
        assert 'MAX. ITERATION REACHED' in info['exit_message']

        # BiCGSTAB with lower verbosity, print checking.
        _ = solver.solve(model, sfield, sslsolver='bicgstab', plain=True,
                         verb=3, maxit=1)
        out, _ = capsys.readouterr()
        assert ' MAX. ITERATION REACHED' in out

        # Just check if it runs without failing for other solvers.
        _ = solver.solve(model, sfield, sslsolver='gcrotmk', plain=True,
                         verb=5, maxit=1)

        # Provide initial field.
        _, _ = capsys.readouterr()  # empty
        efield_copy = efield.copy()
        outarray = solver.solve(
                model, sfield, plain=True, efield=efield_copy, verb=3)
        out, _ = capsys.readouterr()

        # Ensure there is no output.
        assert outarray is None
        assert "NOTHING DONE (provided efield already good enough)" in out
        # Ensure the field did not change.
        assert_allclose(efield.field, efield_copy.field)

        # Provide initial field and return info.
        info = solver.solve(model, sfield, plain=True, efield=efield_copy,
                            return_info=True)
        assert info['it_mg'] == 0
        assert info['it_ssl'] == 0
        assert info['exit'] == 0
        assert info['exit_message'] == 'CONVERGED'

        # Provide initial field, ensure one initial multigrid is carried out
        # without linerelaxation nor semicoarsening.
        _, _ = capsys.readouterr()  # empty
        efield = emg3d.Field(grid)
        outarray = solver.solve(model, sfield, efield=efield, maxit=2, verb=4)
        out, _ = capsys.readouterr()
        assert "after                       1 F-cycles    4 1" in out
        assert "after                       2 F-cycles    5 2" in out

        # Provide an initial source-field without frequency information.
        wrong_sfield = emg3d.Field(grid)
        wrong_sfield.field = sfield.field
        with pytest.raises(ValueError, match="Source field is missing frequ"):
            solver.solve(
                    model, wrong_sfield, plain=True, efield=efield, verb=1)

        # Check stagnation by providing an almost zero source field.
        sfield.field = 1e-10
        _ = solver.solve(model, sfield, plain=True, maxit=100)
        out, _ = capsys.readouterr()
        assert "STAGNATED" in out

        # Check a zero field is returned for a zero source field.
        sfield.field = 0
        efield = solver.solve(model, sfield, plain=True, maxit=100, verb=3)
        out, _ = capsys.readouterr()
        assert "RETURN ZERO E-FIELD (provided sfield is zero)" in out
        assert np.linalg.norm(efield.field) == 0.0

    def test_heterogeneous(self, capsys):
        # Regression test for heterogeneous case.
        dat = REGRES['reg_2']
        model = dat['model']
        sfield = dat['sfield']
        inp = dat['inp']
        for n in ['nu_init', 'nu_pre', 'nu_coarse', 'nu_post']:
            inp[n] = int(inp[n])
        inp['verb'] = 5

        efield = solver.solve(model, sfield, sslsolver=False, **inp)

        assert_allclose(dat['result'].field, efield.field)

        _, _ = capsys.readouterr()  # Clean up

        # Check with provided e-field; 2x2 iter should yield the same as 4 iter
        efield2 = solver.solve(model, sfield, plain=True, maxit=4, verb=0)
        out, _ = capsys.readouterr()  # Clean up
        assert "* WARNING :: MAX. ITERATION REACHED, NOT CONVERGED" in out
        efield3 = solver.solve(model, sfield, plain=True, maxit=2, verb=0)
        solver.solve(
                model, sfield, plain=True, efield=efield3, maxit=2, verb=0)

        assert efield2 == efield3

        _, _ = capsys.readouterr()  # Clean up

        # One test without post-smoothing to check if it runs.
        efield4 = solver.solve(
                model, sfield, maxit=20, nu_pre=0, nu_post=4, verb=4)
        efield5 = solver.solve(
                model, sfield, maxit=20, nu_pre=4, nu_post=0, verb=4)
        # They don't converge, and hence don't agree. Just a lazy test.
        assert_allclose(efield4.field, efield5.field, atol=1e-15, rtol=1e-5)

        # Check the QC plot if it is too long.
        # Coincidently, this one also diverges if nu_pre=0!
        # Mesh: 2-cells in y- and z-direction; 2**9 in x-direction
        mesh = emg3d.TensorMesh(
                [np.ones(2**9)/np.ones(2**9).sum(), np.ones(2), np.ones(2)],
                origin=np.array([-0.5, -1, -1]))
        sfield = alternatives.alt_get_source_field(mesh, [0, 0, 0, 0, 0], 1)
        model = emg3d.Model(mesh)
        _ = solver.solve(model, sfield, plain=True, verb=4, nu_pre=0)
        out, _ = capsys.readouterr()
        assert "(Cycle-QC restricted to first 70 steps of 72 steps.)" in out
        assert "DIVERGED" in out

    def test_log(self, capsys):
        dat = REGRES['res']

        model = emg3d.Model(**dat['input_model'])
        sfield = emg3d.get_source_field(**dat['input_source'])
        inp = {'model': model, 'sfield': sfield, 'plain': True, 'maxit': 1}

        efield, info = solver.solve(return_info=True, log=-1, verb=3, **inp)
        out, _ = capsys.readouterr()
        assert out == ""
        assert ' emg3d START ::' in info['log']

        efield = solver.solve(return_info=True, log=0, verb=3, **inp)
        out, _ = capsys.readouterr()
        assert ' emg3d START ::' in out

        efield, info = solver.solve(return_info=True, log=1, verb=3, **inp)
        out, _ = capsys.readouterr()
        assert ' emg3d START ::' in out
        assert ' emg3d START ::' in info['log']

        efield, info = solver.solve(return_info=True, log=1, verb=1, **inp)
        out, _ = capsys.readouterr()
        assert 'MAX. ITERATION REACHED, NOT CONVERGED' in out
        assert 'MAX. ITERATION REACHED, NOT CONVERGED' in info['log']

    def test_laplace(self, ):
        # Regression test for homogeneous halfspace in Laplace domain.
        # Not very sophisticated; replace/extend by more detailed tests.
        dat = REGRES['lap']

        model = emg3d.Model(**dat['input_model'])
        grid = model.grid
        sfield = emg3d.get_source_field(**dat['input_source'])

        # F-cycle
        efield = solver.solve(model, sfield, plain=True)

        # Check all fields (ex, ey, and ez)
        assert_allclose(dat['Fresult'].field, efield.field, atol=1e-14)

        # BiCGSTAB with some print checking.
        efield = solver.solve(
                model, sfield, semicoarsening=False, linerelaxation=False)

        # Check all fields (ex, ey, and ez)
        assert_allclose(dat['bicresult'].field, efield.field, atol=1e-14)

        # If efield is complex, assert it fails.
        efield = emg3d.Field(grid, dtype=np.complex128)

        with pytest.raises(ValueError, match='Source field and electric fiel'):
            efield = solver.solve(model, sfield, plain=True, efield=efield)


def test_solve_source():
    dat = REGRES['res']
    model = emg3d.Model(**dat['input_model'])
    efield = solver.solve_source(
            model=model, source=dat['input_source']['source'],
            frequency=dat['input_source']['frequency'], plain=True)
    assert_allclose(dat['Fresult'].field, efield.field)


def test__solve():
    # Has keys [model, sfield, efield, solver_opts]
    dat = REGRES['res']
    inp = {'model': emg3d.Model(**dat['input_model']),
           'sfield': emg3d.get_source_field(**dat['input_source']),
           'efield': None,
           'solver_opts': {'plain': True}}
    efield, info = solver._solve(inp)
    assert_allclose(dat['Fresult'].field, efield.field)

    # Has keys [model, grid, source, frequency, efield, solver_opts]
    dat = REGRES['res']
    model = model = emg3d.Model(**dat['input_model'])
    inp = {'model': model,
           'grid': model.grid,
           'source': dat['input_source']['source'],
           'frequency': dat['input_source']['frequency'],
           'efield': None,
           'solver_opts': {'plain': True}}
    efield, info = solver._solve(inp)
    assert_allclose(dat['Fresult'].field, efield.field)


class TestMultigrid:
    # Everything should be tested just fine in `test_solver`. Just check here
    # that all code is reached.

    def test_basic(self, capsys):
        # This should reach every line of solver.multigrid.
        dat = REGRES['res']
        model = emg3d.Model(**dat['input_model'])
        grid = model.grid
        sfield = emg3d.get_source_field(**dat['input_source'])
        vmodel = emg3d.models.VolumeModel(model, sfield)
        efield = emg3d.Field(grid)  # Initiate e-field.

        # Get var-instance
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=True,
                linerelaxation=True, shape_cells=grid.shape_cells, verb=5,
                nu_init=2, maxit=-1,
        )
        var.l2_refe = sl.norm(sfield.field, check_finite=False)

        # Call multigrid.
        solver.multigrid(vmodel, sfield, efield, var)
        out, _ = capsys.readouterr()
        assert '> CONVERGED' in out


class TestKrylov:
    # Everything should be tested just fine in `test_solver`. Just check here
    # for bicgstab-error, and that all code is reached.

    def test_bicgstab_error(self, capsys):
        # Load any case.
        dat = REGRES['res']
        model = emg3d.Model(**dat['input_model'])
        grid = model.grid
        model.property_x *= 100000  # Set stupid input to make bicgstab fail.
        model.property_y /= 100000  # Set stupid input to make bicgstab fail.
        sfield = emg3d.get_source_field(**dat['input_source'])
        vmodel = emg3d.models.VolumeModel(model, sfield)
        efield = emg3d.Field(grid)  # Initiate e-field.

        # Get var-instance
        var = solver.MGParameters(
                cycle=None, sslsolver=True, semicoarsening=False,
                linerelaxation=False, shape_cells=grid.shape_cells, verb=3,
                maxit=-1,
        )
        var.l2_refe = sl.norm(sfield.field, check_finite=False)

        # Call krylov and ensure it fails properly.
        solver.krylov(vmodel, sfield, efield, var)
        out, _ = capsys.readouterr()
        assert '* ERROR   :: Error in bicgstab' in out

    def test_cycle_gcrotmk(self, capsys):

        # Load any case.
        dat = REGRES['res']
        model = emg3d.Model(**dat['input_model'])
        grid = model.grid
        sfield = emg3d.get_source_field(**dat['input_source'])
        vmodel = emg3d.models.VolumeModel(model, sfield)
        efield = emg3d.Field(grid)  # Initiate e-field.

        # Get var-instance
        var = solver.MGParameters(
                cycle='F', sslsolver='gcrotmk', semicoarsening=False,
                linerelaxation=False, shape_cells=grid.shape_cells, verb=4,
                maxit=5,
        )
        var.l2_refe = sl.norm(sfield.field, check_finite=False)

        # Call krylov and ensure it fails properly.
        solver.krylov(vmodel, sfield, efield, var)
        out, _ = capsys.readouterr()
        assert 'DIVERGED' in out

    def test_cycle(self, capsys):

        # Load any case.
        dat = REGRES['res']
        model = emg3d.Model(**dat['input_model'])
        grid = model.grid
        sfield = emg3d.get_source_field(**dat['input_source'])
        vmodel = emg3d.models.VolumeModel(model, sfield)
        efield = emg3d.Field(grid)  # Initiate e-field.

        # Get var-instance
        var = solver.MGParameters(
                cycle='F', sslsolver=True, semicoarsening=True,
                tol=1.0, linerelaxation=True, shape_cells=grid.shape_cells,
                verb=4, maxit=2,
        )
        var.l2_refe = sl.norm(sfield.field, check_finite=False)

        # Call krylov and ensure it fails properly.
        solver.krylov(vmodel, sfield, efield, var)
        out, _ = capsys.readouterr()
        assert '> CONVERGED' in out

        # Call krylov and ensure it fails properly.
        efield = emg3d.Field(grid)  # Initiate e-field.
        # Get var-instance
        var = solver.MGParameters(
                cycle='F', sslsolver=True, semicoarsening=False,
                linerelaxation=False, shape_cells=grid.shape_cells, verb=4,
                maxit=1,
        )
        var.l2_refe = sl.norm(sfield.field, check_finite=False)
        solver.krylov(vmodel, sfield, efield, var)
        out, _ = capsys.readouterr()
        assert 'MAX. ITERATION REACHED' in out


def test_smoothing():
    # 1. The only thing to test here is that smoothing returns the same as
    #    the corresponding jitted functions. Basically a copy of the function
    #    itself.

    nu = 2

    widths = [np.ones(2)*100, helpers.widths(10, 27, 10, 1.1),
              helpers.widths(2, 1, 50, 1.2)]
    origin = [-w.sum()/2 for w in widths]
    src = [0, -10, -10, 43, 13]

    # Loop and move the 2-cell dimension (100, 2) from x to y to z.
    for xyz in range(3):

        # Create a grid
        grid = emg3d.TensorMesh(
            [widths[xyz % 3],
             widths[(xyz+1) % 3],
             widths[(xyz+2) % 3]],
            origin=np.array(
                [origin[xyz % 3], origin[(xyz+1) % 3], origin[(xyz+2) % 3]])
        )

        # Create some resistivity model
        x = np.arange(1, grid.shape_cells[0]+1)*2
        y = 1/np.arange(1, grid.shape_cells[1]+1)
        z = np.arange(1, grid.shape_cells[2]+1)[::-1]/10
        property_x = np.outer(np.outer(x, y), z).ravel()
        freq = 0.319
        model = emg3d.Model(grid, property_x, 0.8*property_x, 2*property_x)

        # Create a source field
        sfield = emg3d.get_source_field(grid=grid, source=src, frequency=freq)

        # Get volume-averaged model parameters.
        vmodel = emg3d.models.VolumeModel(model, sfield)

        # Run two iterations to get an e-field
        field = solver.solve(model, sfield, maxit=2)

        # Collect Gauss-Seidel input (same for all routines)
        inp = (sfield.fx, sfield.fy, sfield.fz, vmodel.eta_x, vmodel.eta_y,
               vmodel.eta_z, vmodel.zeta, grid.h[0], grid.h[1], grid.h[2], nu)

        func = ['', '_x', '_y', '_z']
        for lr_dir in range(8):
            # Get it directly from core
            efield = emg3d.Field(grid, field.field)
            finp = (efield.fx, efield.fy, efield.fz)
            if lr_dir < 4:
                getattr(emg3d.core, 'gauss_seidel'+func[lr_dir])(
                        efield.fx, efield.fy, efield.fz, *inp)
            elif lr_dir == 4:
                emg3d.core.gauss_seidel_y(*finp, *inp)
                emg3d.core.gauss_seidel_z(*finp, *inp)
            elif lr_dir == 5:
                emg3d.core.gauss_seidel_x(*finp, *inp)
                emg3d.core.gauss_seidel_z(*finp, *inp)
            elif lr_dir == 6:
                emg3d.core.gauss_seidel_x(*finp, *inp)
                emg3d.core.gauss_seidel_y(*finp, *inp)
            elif lr_dir == 7:
                emg3d.core.gauss_seidel_x(*finp, *inp)
                emg3d.core.gauss_seidel_y(*finp, *inp)
                emg3d.core.gauss_seidel_z(*finp, *inp)

            # Use solver.smoothing
            ofield = emg3d.Field(grid, field.field)
            solver.smoothing(vmodel, sfield, ofield, nu, lr_dir)

            # Compare
            assert efield == ofield


class TestRestrictionProlongation:

    def test_sc_0(self):
        sc = 0

        # Simple test with restriction followed by prolongation.
        src = [150, 150, 150, 0, 45]
        grid = emg3d.TensorMesh(
                [np.ones(4)*100, np.ones(4)*100, np.ones(4)*100],
                origin=np.zeros(3))

        # Create dummy model and fields, parameters don't matter.
        model = emg3d.Model(grid, 1, 1, 1, 1)
        sfield = emg3d.get_source_field(grid, src, 1)

        # Get volume-averaged model parameters.
        vmodel = emg3d.models.VolumeModel(model, sfield)

        rx = np.arange(sfield.fx.size, dtype=np.complex128).reshape(
                sfield.fx.shape)
        ry = np.arange(sfield.fy.size, dtype=np.complex128).reshape(
                sfield.fy.shape)
        rz = np.arange(sfield.fz.size, dtype=np.complex128).reshape(
                sfield.fz.shape)
        field = np.r_[rx.ravel('F'), ry.ravel('F'), rz.ravel('F')]
        rr = emg3d.Field(grid, field)

        # Restrict it
        cmodel, csfield, cefield = solver.restriction(
                vmodel, sfield, rr, sc_dir=sc)

        assert_allclose(csfield.fx[:, 1:-1, 1],
                        np.array([[196.+0.j], [596.+0.j]]))
        assert_allclose(csfield.fy[1:-1, :, 1],
                        np.array([[356.+0.j, 436.+0.j]]))
        assert_allclose(csfield.fz[1:-1, 1:-1, :],
                        np.array([[[388.+0.j, 404.+0.j]]]))
        assert cmodel.grid.shape_nodes[0] == cmodel.grid.shape_nodes[1] == 3
        assert cmodel.grid.shape_nodes[2] == 3
        assert cmodel.eta_x[0, 0, 0]/8. == vmodel.eta_x[0, 0, 0]
        assert np.sum(grid.h[0]) == np.sum(cmodel.grid.h[0])
        assert np.sum(grid.h[1]) == np.sum(cmodel.grid.h[1])
        assert np.sum(grid.h[2]) == np.sum(cmodel.grid.h[2])

        # Add pi to the coarse e-field
        efield = emg3d.Field(grid)
        cefield.field += np.pi

        # Prolong it
        solver.prolongation(efield, cefield, sc_dir=sc)

        assert np.all(efield.fx[:, 1:-1, 1:-1] == np.pi)
        assert np.all(efield.fy[1:-1, :, 1:-1] == np.pi)
        assert np.all(efield.fz[1:-1, 1:-1, :] == np.pi)

    def test_sc_1(self):
        sc = 1

        # Simple test with restriction followed by prolongation.
        src = [150, 150, 150, 0, 45]
        grid = emg3d.TensorMesh(
                [np.ones(4)*100, np.ones(4)*100, np.ones(4)*100],
                origin=np.zeros(3))

        # Create dummy model and fields, parameters don't matter.
        model = emg3d.Model(grid, 1)
        sfield = emg3d.get_source_field(grid, src, 1)

        # Get volume-averaged model parameters.
        vmodel = emg3d.models.VolumeModel(model, sfield)

        rx = np.arange(sfield.fx.size, dtype=np.complex128).reshape(
                sfield.fx.shape)
        ry = np.arange(sfield.fy.size, dtype=np.complex128).reshape(
                sfield.fy.shape)
        rz = np.arange(sfield.fz.size, dtype=np.complex128).reshape(
                sfield.fz.shape)
        field = np.r_[rx.ravel('F'), ry.ravel('F'), rz.ravel('F')]
        rr = emg3d.Field(grid, field)

        # Restrict it
        cmodel, csfield, cefield = solver.restriction(
                vmodel, sfield, rr, sc_dir=sc)

        assert_allclose(csfield.fx[:, 1:-1, 1],
                        np.array([[48.+0.j], [148.+0.j],
                                  [248.+0.j], [348.+0.j]]))
        assert_allclose(csfield.fy[1:-1, :, 1],
                        np.array([[98.+0.j, 138.+0.j], [178.+0.j, 218.+0.j],
                                  [258.+0.j, 298.+0.j]]))
        assert_allclose(csfield.fz[1:-1, 1:-1, :],
                        np.array([[[114.+0.j, 122.+0.j]],
                                  [[194.+0.j, 202.+0.j]],
                                  [[274.+0.j, 282.+0.j]]]))
        assert cmodel.grid.shape_nodes[0] == 5
        assert cmodel.grid.shape_nodes[1] == 3
        assert cmodel.grid.shape_nodes[2] == 3
        assert cmodel.eta_x[0, 0, 0]/4. == vmodel.eta_x[0, 0, 0]
        assert np.sum(grid.h[0]) == np.sum(cmodel.grid.h[0])
        assert np.sum(grid.h[1]) == np.sum(cmodel.grid.h[1])
        assert np.sum(grid.h[2]) == np.sum(cmodel.grid.h[2])

        # Add pi to the coarse e-field
        efield = emg3d.Field(grid)
        cefield.field += np.pi

        # Prolong it
        solver.prolongation(efield, cefield, sc_dir=sc)

        assert np.all(efield.fx[:, 1:-1, 1:-1] == np.pi)
        assert np.all(efield.fy[1:-1, :, 1:-1] == np.pi)
        assert np.all(efield.fz[1:-1, 1:-1, :] == np.pi)

    def test_sc_4(self):
        sc = 4

        # Simple test with restriction followed by prolongation.
        src = [150, 150, 150, 0, 45]
        grid = emg3d.TensorMesh(
                [np.ones(4)*100, np.ones(4)*100, np.ones(4)*100],
                origin=np.zeros(3))

        # Create dummy model and fields, parameters don't matter.
        model = emg3d.Model(grid, 1, 1)
        sfield = emg3d.get_source_field(grid, src, 1)

        # Get volume-averaged model parameters.
        vmodel = emg3d.models.VolumeModel(model, sfield)

        rx = np.arange(sfield.fx.size, dtype=np.complex128).reshape(
                sfield.fx.shape)
        ry = np.arange(sfield.fy.size, dtype=np.complex128).reshape(
                sfield.fy.shape)
        rz = np.arange(sfield.fz.size, dtype=np.complex128).reshape(
                sfield.fz.shape)
        field = np.r_[rx.ravel('F'), ry.ravel('F'), rz.ravel('F')]
        rr = emg3d.Field(grid, field)

        # Restrict it
        cmodel, csfield, cefield = solver.restriction(
                vmodel, sfield, rr, sc_dir=sc)

        assert_allclose(csfield.fx[:, 1:-1, 1],
                        np.array([[37.+0.j, 47.+0.j, 57.+0.j],
                                  [137.+0.j, 147.+0.j, 157]]))
        assert_allclose(csfield.fy[1:-1, :, 1],
                        np.array([[82.+0.j, 92.+0.j, 102.+0.j, 112.+0.j]]))
        assert_allclose(csfield.fz[1:-1, 1:-1, :],
                        np.array([[[88.+0.j, 90, 92, 94],
                                   [96.+0.j, 98, 100, 102],
                                   [104.+0.j, 106, 108, 110]]]))
        assert cmodel.grid.shape_nodes[0] == 3
        assert cmodel.grid.shape_nodes[1] == 5
        assert cmodel.grid.shape_nodes[2] == 5
        assert cmodel.eta_x[0, 0, 0]/2. == vmodel.eta_x[0, 0, 0]
        assert np.sum(grid.h[0]) == np.sum(cmodel.grid.h[0])
        assert np.sum(grid.h[1]) == np.sum(cmodel.grid.h[1])
        assert np.sum(grid.h[2]) == np.sum(cmodel.grid.h[2])

        # Add pi to the coarse e-field
        efield = emg3d.Field(grid)
        cefield.field += np.pi

        # Prolong it
        solver.prolongation(efield, cefield, sc_dir=sc)

        assert np.all(efield.fx[:, 1:-1, 1:-1] == np.pi)
        assert np.all(efield.fy[1:-1, :, 1:-1] == np.pi)
        assert np.all(efield.fz[1:-1, 1:-1, :] == np.pi)


def test_residual():
    # The only thing to test here is that the residual returns the same as
    # sfield-amat_x. Basically a copy of the function itself.

    # Create a grid
    src = [90, 1600, 25., 45, 45]
    grid = emg3d.TensorMesh(
        [helpers.widths(4, 2, 20, 1.2), np.ones(16)*200, np.ones(2)*25],
        origin=np.zeros(3))

    # Create some resistivity model
    x = np.arange(1, grid.shape_cells[0]+1)*2
    y = 1/np.arange(1, grid.shape_cells[1]+1)
    z = np.arange(1, grid.shape_cells[2]+1)[::-1]/10
    property_x = np.outer(np.outer(x, y), z).ravel()
    freq = 0.319
    model = emg3d.Model(grid, property_x, 0.8*property_x, 2*property_x)

    # Create a source field
    sfield = emg3d.get_source_field(grid=grid, source=src, frequency=freq)

    # Get volume-averaged model parameters.
    vmodel = emg3d.models.VolumeModel(model, sfield)

    # Run two iterations to get an e-field
    efield = solver.solve(model, sfield, maxit=2)

    # Use directly amat_x
    rfield = sfield.copy()
    emg3d.core.amat_x(
            rfield.fx, rfield.fy, rfield.fz, efield.fx, efield.fy, efield.fz,
            vmodel.eta_x, vmodel.eta_y, vmodel.eta_z, vmodel.zeta, grid.h[0],
            grid.h[1], grid.h[2])

    # Compute residual
    out = solver.residual(vmodel, sfield, efield)
    outnorm = solver.residual(vmodel, sfield, efield, True)

    # Compare
    assert_allclose(out.field, rfield.field)
    assert_allclose(outnorm, np.linalg.norm(out.field))


class TestMGParameters:
    shape_cells = (2**3, 2**5, 2**4)

    def test_semicoarsening(self):
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=True,
                linerelaxation=False, shape_cells=self.shape_cells, verb=0)
        assert 'semicoarsening : True [1 2 3]' in var.__repr__()
        var = solver.MGParameters(
                cycle='V', sslsolver=False, semicoarsening=1213,
                linerelaxation=False, shape_cells=self.shape_cells, verb=0)
        assert 'semicoarsening : True [1 2 1 3]' in var.__repr__()
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=2,
                linerelaxation=False, shape_cells=self.shape_cells, verb=0)
        assert 'semicoarsening : True [2]' in var.__repr__()
        with pytest.raises(ValueError, match='`semicoarsening` must be one o'):
            solver.MGParameters(
                    cycle='F', sslsolver=False, semicoarsening=5,
                    linerelaxation=False, shape_cells=self.shape_cells, verb=0)

    def test_linerelaxation(self):
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=False,
                linerelaxation=True, shape_cells=self.shape_cells, verb=0)
        assert 'linerelaxation : True [4 5 6]' in var.__repr__()
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=False,
                linerelaxation=1247, shape_cells=self.shape_cells, verb=0)
        assert 'linerelaxation : True [1 2 4 7]' in var.__repr__()
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=False,
                linerelaxation=1, shape_cells=self.shape_cells, verb=0,
                clevel=1)
        assert 'linerelaxation : True [1]' in var.__repr__()
        assert_allclose(var.clevel, 1)
        with pytest.raises(ValueError, match='`linerelaxation` must be one o'):
            solver.MGParameters(
                    cycle='F', sslsolver=False, semicoarsening=False,
                    linerelaxation=-9, shape_cells=self.shape_cells, verb=0)

    def test_sslsolver_and_cycle(self):
        with pytest.raises(ValueError, match='At least `cycle` or `sslsolve'):
            solver.MGParameters(
                    cycle=None, sslsolver=False, semicoarsening=False,
                    linerelaxation=False, shape_cells=self.shape_cells, verb=0)
        var = solver.MGParameters(
                cycle='F', sslsolver=True, semicoarsening=True,
                linerelaxation=False, shape_cells=self.shape_cells, verb=0,
                maxit=33)
        assert "sslsolver : 'bicgstab'" in var.__repr__()
        assert var.ssl_maxit == 33
        assert var.maxit == 3
        with pytest.raises(ValueError, match='`sslsolver` must be True'):
            solver.MGParameters(
                    cycle='F', sslsolver='abcd', semicoarsening=0,
                    linerelaxation=False, shape_cells=self.shape_cells, verb=0)
        with pytest.raises(ValueError, match='`sslsolver` must be True'):
            solver.MGParameters(
                    cycle='F', sslsolver=4, semicoarsening=0,
                    linerelaxation=False, shape_cells=self.shape_cells, verb=0)
        with pytest.raises(ValueError, match='`cycle` must be one of'):
            solver.MGParameters(
                    cycle='G', sslsolver=False, semicoarsening=False,
                    linerelaxation=False, shape_cells=self.shape_cells, verb=0)

    # 4. Wrong grid size
    def test_wrong_grid_size(self):
        with pytest.raises(ValueError, match='Nr. of cells must be at least'):
            solver.MGParameters(
                    cycle='F', sslsolver=False, semicoarsening=False,
                    linerelaxation=False, shape_cells=(1, 2, 3), verb=0)

    def test_bad_grid_size(self):
        inp = {'cycle': 'F', 'sslsolver': False, 'semicoarsening': False,
               'linerelaxation': False, 'verb': 0}
        txt = ":: Grid not optimal for MG solver ::"

        # One large lowest => warning.
        var = solver.MGParameters(shape_cells=(11*2**3, 2**5, 2**4), **inp)
        assert txt in var.__repr__()

        # Large lowest, but clevel smaller => no warning.
        var = solver.MGParameters(
                shape_cells=(11*2**5, 11*2**4, 11*2**5), clevel=4, **inp)
        assert txt not in var.__repr__()

        # Large lowest, clevel bigger => warning.
        var = solver.MGParameters(
                shape_cells=(11*2**5, 11*2**4, 11*2**5), clevel=5, **inp)
        assert txt in var.__repr__()

        # Only 2 times dividable => warning.
        var = solver.MGParameters(shape_cells=(2**3, 2**3, 2**3), **inp)
        assert txt in var.__repr__()

    def test_cprint(self, capsys):
        var = solver.MGParameters(
                cycle='F', sslsolver=False, semicoarsening=True, log=1,
                linerelaxation=False, shape_cells=self.shape_cells, verb=2)
        var.cprint('test', 3)
        out, _ = capsys.readouterr()
        assert out == ""
        assert var.log_message == ""

        var.cprint('test', 1)
        out, _ = capsys.readouterr()
        assert out == "test\n"
        assert var.log_message == "test\n"


def test_RegularGridProlongator():

    def prolon_scipy(grid, cgrid, efield, cefield):
        CZ, CY = np.broadcast_arrays(grid.nodes_z, grid.nodes_y[:, None])
        yz = np.r_[CY.ravel('F'), CZ.ravel('F')].reshape(-1, 2, order='F')
        """Compute SciPy alternative."""
        for ixc in range(cgrid.shape_cells[0]):
            # Bilinear interpolation in the y-z plane
            fn = si.RegularGridInterpolator(
                    (cgrid.nodes_y, cgrid.nodes_z), cefield.fx[ixc, :, :],
                    bounds_error=False, fill_value=None)
            hh = fn(yz).reshape(grid.shape_edges_x[1:], order='F')

            # Piecewise constant interpolation in x-direction
            efield[2*ixc, :, :] += hh
            efield[2*ixc+1, :, :] += hh

        return efield

    def prolon_emg3d(grid, cgrid, efield, cefield):
        """Compute emg3d alternative."""
        fn = solver.RegularGridProlongator(
                cgrid.nodes_y, cgrid.nodes_z, grid.nodes_y, grid.nodes_z)

        for ixc in range(cgrid.shape_cells[0]):
            # Bilinear interpolation in the y-z plane
            hh = fn(cefield.fx[ixc, :, :]).reshape(
                    grid.shape_edges_x[1:], order='F')

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
    grid = emg3d.TensorMesh([hx, hy, hz], origin=np.array([0, 0, 0]))

    # Create coarse grid.
    chx = np.diff(grid.nodes_x[::2])
    cgrid = emg3d.TensorMesh([chx, chx, chx], origin=np.array([0, 0, 0]))

    # Create empty fine grid fields.
    efield1 = emg3d.Field(grid)
    efield2 = emg3d.Field(grid)

    # Create coarse grid field with some values.
    cefield = emg3d.Field(cgrid)
    cefield.fx = np.arange(cefield.fx.size)
    cefield.fx = 1j*np.arange(cefield.fx.size)/10

    # Compare
    out1 = prolon_scipy(grid, cgrid, efield1.fx, cefield)
    out2 = prolon_emg3d(grid, cgrid, efield2.fx, cefield)

    assert_allclose(out1, out2)


def test_current_sc_dir():
    hx = np.ones(4)
    grid = emg3d.TensorMesh([hx, hx, hx], (0, 0, 0))  # Big enough

    # Big enough, no change.
    for sc_dir in range(4):
        assert sc_dir == solver._current_sc_dir(sc_dir, grid)

    # Small in all directions => always 0
    grid = emg3d.TensorMesh([[2, 2], [2, 2], [2, 2]], (0, 0, 0))
    for sc_dir in range(4):
        assert 6 == solver._current_sc_dir(sc_dir, grid)

    # Small in y, z
    grid = emg3d.TensorMesh([hx, [2, 2], [2, 2]], (0, 0, 0))
    assert 4 == solver._current_sc_dir(0, grid)
    assert 6 == solver._current_sc_dir(1, grid)
    assert 4 == solver._current_sc_dir(2, grid)
    assert 4 == solver._current_sc_dir(3, grid)

    # Small in x, z
    grid = emg3d.TensorMesh([[2, 2], hx, [2, 2]], (0, 0, 0))
    assert 5 == solver._current_sc_dir(0, grid)
    assert 5 == solver._current_sc_dir(1, grid)
    assert 6 == solver._current_sc_dir(2, grid)
    assert 5 == solver._current_sc_dir(3, grid)


def test_current_lr_dir():
    hx = np.ones(4)
    grid = emg3d.TensorMesh([hx, hx, hx], (0, 0, 0))  # Big enough

    # Big enough, no change.
    for lr_dir in range(8):
        assert lr_dir == solver._current_lr_dir(lr_dir, grid)

    # Small in all directions => always 0
    grid = emg3d.TensorMesh([[2, 2], [2, 2], [2, 2]], (0, 0, 0))
    for lr_dir in range(8):
        assert 0 == solver._current_lr_dir(lr_dir, grid)

    # Small in y, z
    grid = emg3d.TensorMesh([hx, [2, 2], [2, 2]], (0, 0, 0))
    for lr_dir in [0, 1]:
        assert lr_dir == solver._current_lr_dir(lr_dir, grid)
    for lr_dir in [2, 3, 4]:
        assert 0 == solver._current_lr_dir(lr_dir, grid)
    for lr_dir in [5, 6, 7]:
        assert 1 == solver._current_lr_dir(lr_dir, grid)

    # Small in z
    grid = emg3d.TensorMesh([hx, hx, [2, 2]], (0, 0, 0))
    for lr_dir in [0, 1, 2, 6]:
        assert lr_dir == solver._current_lr_dir(lr_dir, grid)
    assert 0 == solver._current_lr_dir(3, grid)
    assert 2 == solver._current_lr_dir(4, grid)
    assert 1 == solver._current_lr_dir(5, grid)
    assert 6 == solver._current_lr_dir(7, grid)


def test_terminate():
    class MGParameters:
        """Fake MGParameters class."""
        def __init__(self, verb=0, sslsolver=None):
            self.verb = verb
            self.exit_message = ""
            self.sslsolver = sslsolver
            self.maxit = 5
            self.l2_refe = 1e-3
            self.tol = 1e-2

        def cprint(self, info, verbosity, **kwargs):
            self.info = info

    # Converged
    var = MGParameters()
    out = solver._terminate(var, 1e-6, 5e-6, 1)
    assert out is True
    assert var.exit_message == "CONVERGED"
    assert "   > " + var.exit_message in var.info

    # Diverged if it is 10x larger than last or not a number.
    var = MGParameters(verb=3)
    out = solver._terminate(var, np.inf, 5e-6, 1)
    assert out is True
    assert var.exit_message == "DIVERGED"
    assert "   > " + var.exit_message in var.info

    # Stagnated if it is >= the stagnation value.
    var = MGParameters()
    out = solver._terminate(var, 1e-3, 1e-4, 3)
    assert out is True
    assert var.exit_message == "STAGNATED"
    assert "   > " + var.exit_message in var.info
    out = solver._terminate(var, 1e-3, 1e-4, 1)  # Not on first iteration
    assert out is False
    var.sslsolver = True
    with pytest.raises(solver._ConvergenceError):
        solver._terminate(var, 1e-3, 1e-4, 3)

    # Maximum iterations reached.
    var = MGParameters(5)
    out = solver._terminate(var, 1e-5, 1e-4, 5)
    assert out is True
    assert var.exit_message == "MAX. ITERATION REACHED, NOT CONVERGED"
    assert "   > " + var.exit_message in var.info


def test_restrict_model_parameters():
    data = np.arange(1, 9).reshape((2, 2, 2), order='F')
    # array([[[1, 5],
    #         [3, 7]],
    #
    #        [[2, 6],
    #         [4, 8]]])

    assert_allclose(solver._restrict_model_parameters(data, 0).ravel('F'),
                    [1+2+3+4+5+6+7+8])

    assert_allclose(solver._restrict_model_parameters(data, 1).ravel('F'),
                    [1+3+5+7, 2+4+6+8])

    assert_allclose(solver._restrict_model_parameters(data, 2).ravel('F'),
                    [1+5+2+6, 3+7+4+8])

    assert_allclose(solver._restrict_model_parameters(data, 3).ravel('F'),
                    [1+2+3+4, 5+6+7+8])

    assert_allclose(solver._restrict_model_parameters(data, 4).ravel('F'),
                    [1+2, 3+4, 5+6, 7+8])

    assert_allclose(solver._restrict_model_parameters(data, 5).ravel('F'),
                    [1+3, 2+4, 5+7, 6+8])

    assert_allclose(solver._restrict_model_parameters(data, 6).ravel('F'),
                    [1+5, 2+6, 3+7, 4+8])


def test_get_restriction_weights():
    x = [500, 700, 800, 1000]
    cx = [1200, 1800]
    y = [2, 2, 2, 2]
    cy = [4, 4]

    grid = emg3d.TensorMesh([x, y, x], (0, 0, 0))
    cgrid = emg3d.TensorMesh([cx, cy, cx], (0, 0, 0))

    # 1. Simple example following equation 9, [Muld06]_.
    wxl = np.array([350/250, 250/600, 400/900])
    wx0 = np.array([1., 1., 1.])
    wxr = np.array([350/600, 500/900, 400/500])
    wyl = np.array([1, 0.5, 0.5])
    wy0 = np.array([1., 1., 1.])
    wyr = np.array([0.5, 0.5, 1])
    wdl = np.array([0., 0., 0., 0., 0.])  # dummy
    wd0 = np.array([1., 1., 1., 1., 1.])  # dummy
    wdr = np.array([0., 0., 0., 0., 0.])  # dummy

    for i in [0, 5, 6]:
        wx, wy, wz = solver._get_restriction_weights(grid, cgrid, i)

        if i not in [5, 6]:
            assert_allclose(wxl, wx[0])
            assert_allclose(wx0, wx[1])
            assert_allclose(wxr, wx[2])
        else:
            assert_allclose(wdl, wx[0])
            assert_allclose(wd0, wx[1])
            assert_allclose(wdr, wx[2])

        if i != 6:
            assert_allclose(wyl, wy[0])
            assert_allclose(wy0, wy[1])
            assert_allclose(wyr, wy[2])
        else:
            assert_allclose(wdl, wy[0])
            assert_allclose(wd0, wy[1])
            assert_allclose(wdr, wy[2])

        if i != 5:
            assert_allclose(wxl, wz[0])
            assert_allclose(wx0, wz[1])
            assert_allclose(wxr, wz[2])
        else:
            assert_allclose(wdl, wz[0])
            assert_allclose(wd0, wz[1])
            assert_allclose(wdr, wz[2])


def test_ConvergenceError():
    with pytest.raises(solver._ConvergenceError):
        raise solver._ConvergenceError


def test_print_cycle_info(capsys):
    var = solver.MGParameters(
            verb=4, cycle='F', sslsolver=False, linerelaxation=False,
            semicoarsening=False, shape_cells=(16, 8, 2))
    var.level_all = [0, 1, 2, 3, 2, 3]
    solver._print_cycle_info(var, 1.0, 2.0)
    out, _ = capsys.readouterr()

    assert "h_\n      2h_ \\    \n      4h_  \\   \n      8h_   \\/\\\n" in out
    assert "   1.000e+00  after   0 F-cycles   [1.000e+00, 0.500]   0 0" in out

    # Cuts at 70.
    var = solver.MGParameters(
            verb=5, cycle='F', sslsolver=True, linerelaxation=False,
            semicoarsening=False, shape_cells=(16, 8, 2))
    var.level_all = list(np.r_[0, np.array(50*[[1, 2, 3, 2, 1], ]).ravel()])
    var.it = 123
    solver._print_cycle_info(var, 1.0, 2.0)
    out, _ = capsys.readouterr()
    assert "restricted to first 70 steps" in out
    assert 21*" " + "123" in out
    assert out[-1] == "\n"

    # One-liner.
    var = solver.MGParameters(
            verb=3, cycle='F', sslsolver=False, linerelaxation=False,
            semicoarsening=False, shape_cells=(16, 8, 2))
    var.level_all = [0, 1, 2, 3, 2, 3]
    solver._print_cycle_info(var, 1.0, 2.0)
    out, _ = capsys.readouterr()
    assert ":: emg3d :: 1.0e+00; 0; 0:00:0" in out


def test_print_gs_info(capsys):
    var = solver.MGParameters(
            verb=5, cycle='F', sslsolver=False, linerelaxation=False,
            semicoarsening=False, shape_cells=(16, 8, 2))

    grid = emg3d.TensorMesh([[1, 1], [1, 1], [1, 1]], (0, 0, 0))
    solver._print_gs_info(var, 1, 2, 3, grid, 0.01, 'test')
    out, _ = capsys.readouterr()
    assert out == "      1 2 3 [  2,   2,   2]: 1.000e-02 test\n"


def test_print_one_liner(capsys):
    var = solver.MGParameters(
            verb=5, cycle='F', sslsolver=False, linerelaxation=False,
            semicoarsening=False, shape_cells=(16, 8, 2))

    solver._print_one_liner(var, 1e-2, False)
    out, _ = capsys.readouterr()
    assert ":: emg3d :: 1.0e-02; 0; 0:00:0" in out

    var = solver.MGParameters(
            verb=5, cycle='F', sslsolver=True, linerelaxation=False,
            semicoarsening=False, shape_cells=(16, 8, 2))
    var.exit_message = "TEST"

    solver._print_one_liner(var, 1e-2, True)
    out, _ = capsys.readouterr()
    assert ":: emg3d :: 1.0e-02; 0(0); 0:00:0" in out
    assert "TEST" in out

    grid = emg3d.TensorMesh(
            [np.ones(8), np.ones(8), np.ones(8)], origin=np.array([0, 0, 0]))
    model = emg3d.Model(grid, property_x=1.5, property_y=1.8, property_z=3.3)
    sfield = emg3d.get_source_field(grid, source=[4, 4, 4, 0, 0],
                                    frequency=10.0)

    # Dynamic one-liner.
    out, _ = capsys.readouterr()
    _ = solver.solve(model, sfield, sslsolver=False, semicoarsening=False,
                     linerelaxation=False, verb=1)
    out, _ = capsys.readouterr()
    assert '6; 0:00:' in out
    assert '; CONVERGED' in out

    out, _ = capsys.readouterr()
    _ = solver.solve(model, sfield, sslsolver=True, semicoarsening=False,
                     linerelaxation=False, verb=1)
    out, _ = capsys.readouterr()
    assert '3(5); 0:00:' in out
    assert '; CONVERGED' in out

    # One-liner.
    out, _ = capsys.readouterr()
    _ = solver.solve(model, sfield, sslsolver=True, semicoarsening=False,
                     linerelaxation=False, verb=1)
    out, _ = capsys.readouterr()
    assert '3(5); 0:00:' in out
    assert '; CONVERGED' in out
