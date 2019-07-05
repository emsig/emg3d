"""

:mod:`solver` -- Multigrid solver
=================================

The actual solver routines. The most computationally intensive parts, however,
are in the :mod:`emg3d.njitted` as numba-jitted functions.

"""
# Copyright 2018-2019 The emg3d Developers.
#
# This file is part of emg3d.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.


import itertools
import numpy as np
import scipy.sparse.linalg as ssl
from dataclasses import dataclass

from . import utils
from . import njitted

__all__ = ['solver', 'multigrid', 'smoothing', 'restriction', 'prolongation',
           'residual', 'krylov', 'MGParameters', 'RegularGridProlongator']


# MAIN USER-FACING FUNCTION
def solver(grid, model, sfield, efield=None, cycle='F', sslsolver=False,
           semicoarsening=False, linerelaxation=False, verb=2, **kwargs):
    r"""Solver for 3D CSEM data with tri-axial electrical anisotropy.

    The principal solver of `emg3d` is using the multigrid method as presented
    in [Muld06]_. Multigrid can be used as a standalone solver, or as a
    preconditioner for an iterative solver from the
    :mod:`scipy.sparse.linalg`-library, e.g.,
    :func:`scipy.sparse.linalg.bicgstab`. Alternatively, these Krylov subspace
    solvers can also be used without multigrid at all. See the ``cycle`` and
    ``sslsolver`` parameters.

    Implemented are the `F`-, `V`-, and `W`-cycle schemes for multigrid
    (``cycle`` parameter), and the amount of smoothing steps (initial
    smoothing, pre-smoothing, coarsest-grid smoothing, and post-smoothing) can
    be set individually (``nu_init``, ``nu_pre``, ``nu_coarse``, and
    ``nu_post``, respectively). The maximum level of coarsening can be
    restricted with the ``clevel`` parameter.

    Semicoarsening and line relaxation, as presented in [Muld07]_, are
    implemented, see the ``semicoarsening`` and ``linerelaxation`` parameters.
    Using the BiCGSTAB solver together with multigrid preconditioning with
    semicoarsening and line relaxation is slow but generally the most robust.
    Not using BiCGSTAB nor semicoarsening nor line relaxation is fast but may
    fail on stretched grids.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    model : Model
        Model; ``emg3d.utils.Model`` instance.

    sfield : Field instance
        Source field; ``emg3d.utils.Field`` instance.

    efield : Field instance, optional
        Initial electric field; ``emg3d.utils.Field`` instance. It is
        initiated with zeroes if not provided.

        If an initial efield is provided nothing is returned, but the final
        efield is directly put into the provided efield.

    cycle : str; optional.

        Type of multigrid cycle. Default is 'F'.

        - 'V': V-cycle, simplest version;
        - 'W': W-cycle, most expensive version;
        - 'F': F-cycle, sort of a compromise between 'V' and 'W';
        - None: Does not use multigrid, only ``sslsolver``.

        If None, ``sslsolver`` must be provided, and the ``sslsolver`` will be
        used without multigrid pre-conditioning.

        Comparison of V (left), F (middle), and W (right) cycles for the case
        of four grids (three relaxation and prolongation steps)::

            h_
           2h_   \    /   \          /   \            /
           4h_    \  /     \    /\  /     \    /\    /
           8h_     \/       \/\/  \/       \/\/  \/\/


    sslsolver : str, optional
        A :mod:`scipy.sparse.linalg`-solver, to use with MG as pre-conditioner
        or on its own (if ``cycle=None``). Default is False.

        Current possibilities:

            - True or 'bicgstab': BIConjugate Gradient STABilized
              :func:`scipy.sparse.linalg.bicgstab`;
            - 'cgs': Conjugate Gradient Squared
              :func:`scipy.sparse.linalg.cgs`;
            - 'gmres': Generalized Minimal RESidual
              :func:`scipy.sparse.linalg.gmres`;
            - 'lgmres': Improvement of GMRES using alternating residual
              vectors :func:`scipy.sparse.linalg.lgmres`;
            - 'gcrotmk': GCROT: Generalized Conjugate Residual with inner
              Orthogonalization and Outer Truncation
              :func:`scipy.sparse.linalg.gcrotmk`.

        It does currently not work with 'cg', 'bicg', 'qmr', and 'minres' for
        various reasons (e.g., some require ``rmatvec`` in addition to
        ``matvec``).

    semicoarsening : int; optional
        Semicoarsening. Default is False.

        - True: Cycling over 1, 2, 3.
        - 0 or False: No semicoarsening.
        - 1: Semicoarsening in x direction.
        - 2: Semicoarsening in y direction.
        - 3: Semicoarsening in z direction.
        - Multi-digit number containing digits from 0 to 3. Multigrid will
          cycle over these values, e.g., ``semicoarsening=1213`` will cycle
          over [1, 2, 1, 3].

    linerelaxation : int; optional
        Line relaxation. Default is False.

        This parameter is not respected on the coarsest grid, except if it is
        set to 0. If it is bigger than zero line relaxation on the coarsest
        grid is carried out along all dimensions which have more than 2 cells.

        - True: Cycling over [4, 5, 6].
        - 0 or False: No line relaxation.
        - 1: line relaxation in x direction.
        - 2: line relaxation in y direction.
        - 3: line relaxation in z direction.
        - 4: line relaxation in y and z directions.
        - 5: line relaxation in x and z directions.
        - 6: line relaxation in x and y directions.
        - 7: line relaxation in x, y, and z directions.
        - Multi-digit number containing digits from 0 to 7. Multigrid will
          cycle over these values, e.g., ``linerelaxation=1213`` will cycle
          over [1, 2, 1, 3].

        Note: Smoothing is generally done in lexicographical order, except for
        line relaxation in y direction; the reason is speed (memory access).

    verb : int; optional
        Level of verbosity (the higher the more verbose). Default is 2.

        - 0: Print nothing.
        - 1: Print warnings.
        - 2: Print runtime and information about the method.
        - 3: Print additional information for each MG-cycle.
        - 4: Print everything (slower due to additional error calculations).

    **kwargs : Optional solver options:

        - ``tol`` : float

          Convergence tolerance. Default is 1e-6.

          Iterations stop as soon as the norm of the residual has decreased by
          this factor, relative to the residual norm obtained for a zero
          electric field.

        - ``maxit`` : int

          Maximum number of multigrid iterations. Default is 50.

          If ``sslsolver`` is used, this applies to the ``sslsolver``.

          In the case that multigrid is used as a pre-conditioner for the
          ``sslsolver``, the maximum iteration for multigrid is defined by the
          maximum length of the ``linerelaxation`` and
          ``semicoarsening``-cycles.

        - ``nu_init`` : int

          Number of initial smoothing steps, before MG cycle. Default is 0.

        - ``nu_pre`` : int

          Number of pre-smoothing steps. Default is 2.

        - ``nu_coarse`` : int

          Number of smoothing steps on coarsest grid. Default is 1.

        - ``nu_post`` : int

          Number of post-smoothing steps. Default is 2.

        - ``clevel`` : int

          The maximum coarsening level can be different for each dimension and
          is, by default, automatically determined (``clevel=-1``). The
          parameter ``clevel`` can be used to restrict the maximum coarsening
          level in any direction by its value.
          Default is -1.

        - ``return_info`` : bool

          If True, a dictionary is returned with runtime info (final norm and
          number of iterations of MG and the sslsolver).


    Returns
    -------
    efield : Field instance
        Resulting electric field. Is not returned but replaced in-place if an
        initial efield was provided.

    info_dict : dict
        Dictionary with runtime info (errors and number of iterations); only if
        ``return_info=True``.


    Examples
    --------
    >>> import emg3d
    >>> import discretize
    >>> import numpy as np

    Define the grid (see :class:`discretize.TensorMesh` for more info)

    >>> grid = discretize.TensorMesh(
    >>>         [[(25, 10, -1.04), (25, 28), (25, 10, 1.04)],
    >>>          [(50, 8, -1.03), (50, 16), (50, 8, 1.03)],
    >>>          [(30, 8, -1.05), (30, 16), (30, 8, 1.05)]],
    >>>         x0='CCC')
    >>> print(grid)
    .
      TensorMesh: 49,152 cells
    .
                          MESH EXTENT             CELL WIDTH      FACTOR
      dir    nC        min           max         min       max      max
      ---   ---  ---------------------------  ------------------  ------
       x     48       -662.16        662.16     25.00     37.01    1.04
       y     32       -857.96        857.96     50.00     63.34    1.03
       z     32       -540.80        540.80     30.00     44.32    1.05

    Now we define a very simple fullspace model with ``res_x=1.5`` Ohm.m,
    ``res_y=1.8`` Ohm.m, and ``res_z=3.3`` Ohm.m. The source is an x-directed
    dipole at the origin, with a 10 Hz signal of 1 A.

    >>> freq = 10.0  # Hz
    >>> model = emg3d.utils.Model(
    >>>     grid, res_x=1.5, res_y=1.8, res_z=3.3, freq=10.)
    >>> sfield = emg3d.utils.get_source_field(
    >>>     grid, src=[0, 0, 0, 0, 0], freq=freq)

    Calculate the electric field

    >>> efield = emg3d.solver.solver(grid, model, sfield, verb=3)
    .
    :: emg3d START :: 15:24:40 ::
    .
       MG-cycle       : 'F'                 sslsolver : False
       semicoarsening : False [0]           tol       : 1e-06
       linerelaxation : False [0]           maxit     : 50
       nu_{i,1,c,2}   : 0, 2, 1, 2          verb      : 3
       Original grid  :  48 x  32 x  32     => 49,152 cells
       Coarsest grid  :   3 x   2 x   2     => 12 cells
       Coarsest level :   4 ;   4 ;   4
    .
       [hh:mm:ss]  rel. error                  [abs. error, last/prev]   l s
    .
           h_
          2h_ \                  /
          4h_  \          /\    /
          8h_   \    /\  /  \  /
         16h_    \/\/  \/    \/
    .
       [11:18:17]   2.623e-02  after   1 F-cycles   [1.464e-06, 0.026]   0 0
       [11:18:17]   2.253e-03  after   2 F-cycles   [1.258e-07, 0.086]   0 0
       [11:18:17]   3.051e-04  after   3 F-cycles   [1.704e-08, 0.135]   0 0
       [11:18:17]   5.500e-05  after   4 F-cycles   [3.071e-09, 0.180]   0 0
       [11:18:18]   1.170e-05  after   5 F-cycles   [6.531e-10, 0.213]   0 0
       [11:18:18]   2.745e-06  after   6 F-cycles   [1.532e-10, 0.235]   0 0
       [11:18:18]   6.873e-07  after   7 F-cycles   [3.837e-11, 0.250]   0 0
    .
       > CONVERGED
       > MG cycles        : 7
       > Final rel. error : 6.873e-07
    .
    :: emg3d END   :: 15:24:42 :: runtime = 0:00:02

    """

    # Get return_info from kwargs.
    return_info = kwargs.pop('return_info', False)

    # Solver settings; get from kwargs or set to default values.
    var = MGParameters(
            cycle=cycle, sslsolver=sslsolver, semicoarsening=semicoarsening,
            linerelaxation=linerelaxation, vnC=grid.vnC, verb=verb, **kwargs
    )

    # Start logging and print all parameters.
    var.cprint(f"\n:: emg3d START :: {var.time.now} ::\n", 1)
    var.cprint(var, 1)

    # Calculate reference error for tolerance.
    var.l2_refe = njitted.l2norm(sfield)

    # Get efield
    if efield is None:
        # If not provided, initiate an empty one.
        efield = utils.Field(grid)

        # Set flag to return the field.
        do_return = True
    else:
        # If provided, take the conjugate
        # (see explanation at the end of the solver.solver-routine).
        np.conjugate(efield, efield)

        # Set flag to NOT return the field.
        do_return = False

        # If efield is provided, check if it is already sufficiently good.
        var.l2 = residual(grid, model, sfield, efield, True)
        if var.l2 < var.tol*var.l2_refe:

            # Switch-off both sslsolver and multigrid.
            var.sslsolver = None
            var.cycle = None

            # Start final info.
            info = f"   > NOTHING DONE (provided efield already good enough)\n"

    # Print header for iteration log.
    header = f"   [hh:mm:ss]  {'rel. error':<22}"
    if var.sslsolver:
        header += f"{'solver':<20}"
        if var.cycle:
            header += f"{'MG':<11} l s"
        var.cprint(header+"\n", 2)
    elif var.cycle:
        var.cprint(header+f"{'[abs. error, last/prev]':>29}   l s\n", 2)

    # Solve the system with...
    if var.sslsolver:  # ... sslsolver.
        krylov(grid, model, sfield, efield, var)
    elif var.cycle:    # ... multigrid.
        multigrid(grid, model, sfield, efield, var)

    # Print runtime information.
    if var.sslsolver:  # sslsolver-specific info.
        info = f"   > Solver steps     : {var._ssl_it}\n"
        if var.cycle:
            info += f"   > MG prec. steps   : {var.it}\n"
    elif var.cycle:    # multigrid-specific info.
        info = f"   > MG cycles        : {var.it}\n"
    info += f"   > Final rel. error : {var.l2/var.l2_refe:.3e}\n\n"  # Error.
    info += f":: emg3d END   :: {var.time.now} :: "  # END and time.
    info += f"runtime = {var.time.runtime}\n"        # Total runtime.
    var.cprint(info, 1)

    # To use the same Fourier-transform convention as empymod and commonly
    # used in CSEM, we return the conjugate.
    np.conjugate(efield, efield)

    # Assemble the info_dict if return_info
    info_dict = {
        'abs_error': var.l2,          # Absolute error.
        'rel_error': var.l2/var.l2_refe,  # Relative error.
        'ref_error': var.l2_refe,     # Reference error [norm(sfield)].
        'tol': var.tol,               # Tolerance (abs_error<ref_error*tol).
        'it_mg': var.it,              # Multigrid iterations.
        'it_ssl': var._ssl_it         # SSL iterations.
    }

    # Return depending on input arguments; or nothing.
    if do_return and return_info:  # efield and info.
        return efield, info_dict
    elif do_return:                # efield.
        return efield
    elif return_info:              # info.
        return info_dict


# SOLVERS
def multigrid(grid, model, sfield, efield, var, **kwargs):
    """Multigrid solver for 3D controlled-source electromagnetic (CSEM) data.

    Multigrid solver as presented in [Muld06]_, including semicoarsening and
    line relaxation as presented in and [Muld07]_.

    - The electric field is stored in-place in ``efield``.
    - The number of multigrid cycles is stored in ``var.it``.
    - The current error (l2-norm) is stored in ``var.l2``.
    - The reference error (l2-norm of sfield) is stored in ``var.l2_refe``.

    This function is called by :func:`solver`.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    model : Model
        Model; ``emg3d.utils.Model`` instance.

    sfield, efield : Field
        Source and electric fields; ``emg3d.utils.Field`` instances.

    var : `MGParameters`-instance
        As returned by :func:`multigrid`.

    **kwargs : Recursion parameters.
        Do not use; only used internally by recursion; ``level`` (current
        coarsening level) and ``new_cycmax`` (new maximum of MG cycles, takes
        care of V/W/F-cycling).

    """
    # Get recursion parameters.
    level = kwargs.get('level', 0)
    new_cycmax = kwargs.get('new_cycmax', 0)

    # Initiate iteration count.
    it = 0

    # Get cycmax (depends on cycle and on level [as a fct of sc_dir]).
    # This defines the V, W, and F-cycle scheme.
    if level == var.clevel[var.sc_dir]:
        cycmax = 1
    elif new_cycmax == 0 or var.cycle != 'F':
        cycmax = var.cycmax
    else:
        cycmax = new_cycmax
    cyc = 0  # Initiate cycle count.

    # Calculate current error (l2-norms).
    l2_last = residual(grid, model, sfield, efield, True)

    # Keep track on the levels during the first cycle, for QC.
    if var._first_cycle:
        var._level_all.append(level)

    # Print initial call info.
    if level == 0:
        var.cprint("     it cycmax               error", 3)
        var.cprint("      level [  dimension  ]            info\n", 3)
        if var.verb > 3:
            _print_gs_info(it, level, cycmax, grid, l2_last, "initial error")

    # Initial smoothing (nu_init).
    if level == 0 and var.nu_init > 0:
        # Smooth and re-calculate error.
        smoothing(grid, model, sfield, efield, var.nu_init, var.lr_dir)

        # Print initial smoothing info.
        if var.verb > 3:
            norm = residual(grid, model, sfield, efield, True)
            _print_gs_info(it, level, cycmax, grid, norm, "initial smoothing")

    # Start the actual (recursive) multigrid cycle.
    while level == 0 or (level > 0 and it < cycmax):

        # Store previous error for comparisons.
        l2_prev = l2_last

        if level == var.clevel[var.sc_dir]:  # (A) Coarsest grid, solve system.
            # Note that coarsest grid depends on semicoarsening (sc_dir). If
            # semicoarsening is carried out along the biggest dimension it
            # reduces the number of coarsening levels.

            # Gauss-Seidel on the coarsest grid.
            smoothing(grid, model, sfield, efield, var.nu_coarse, var.lr_dir)

            # Print coarsest grid smoothing info.
            if var.verb > 3:
                norm = residual(grid, model, sfield, efield, True)
                _print_gs_info(it, level, cycmax, grid, norm, "coarsest level")

        else:                   # (B) Not yet on coarsest grid.

            # (B.1) Pre-smoothing (nu_pre).
            if var.nu_pre > 0:
                smoothing(grid, model, sfield, efield, var.nu_pre, var.lr_dir)

                # Print pre-smoothing info.
                if var.verb > 3:
                    norm = residual(grid, model, sfield, efield, True)
                    _print_gs_info(
                            it, level, cycmax, grid, norm, "pre-smoothing")

            # Get sc_dir for this grid.
            sc_dir = _current_sc_dir(var.sc_dir, grid)

            # (B.2) Restrict grid, model, and fields from fine to coarse grid.
            res = residual(grid, model, sfield, efield)  # Get residual.
            cgrid, cmodel, csfield, cefield = restriction(
                    grid, model, sfield, res, sc_dir)

            # (B.3) Recursive call for coarse-grid correction.
            multigrid(cgrid, cmodel, csfield, cefield, var, level=level+1,
                      new_cycmax=cycmax-cyc)

            # (B.4) Add coarse field residual to fine grid field.
            prolongation(grid, efield, cgrid, cefield, sc_dir)

            # Append current prolongation level for QC.
            if var._first_cycle:
                var._level_all.append(level)

            # (B.5) Post-smoothing (nu_post).
            if var.nu_post > 0:
                smoothing(grid, model, sfield, efield, var.nu_post, var.lr_dir)

                # Print post-smoothing info.
                if var.verb > 3:
                    norm = residual(grid, model, sfield, efield, True)
                    _print_gs_info(
                            it, level, cycmax, grid, norm, f"post-smoothing")

        # Update iterator counts.
        it += 1         # Local iterator.
        if level == 0:  # Global iterator (works also when preconditioner.)
            var.it += 1

        # End loop depending if we are on the original grid or not.
        if level > 0:  # Update cyc if on a coarse grid.
            cyc += 1

        else:          # Original grid reached, check termination criteria.

            # Get current error (l2-norm).
            l2_last = residual(grid, model, sfield, efield, True)

            # Print end-of-cycle info.
            _print_cycle_info(var, l2_last, l2_prev)

            # Adjust semicoarsening and line relaxation if they cycle.
            if var.sc_cycle:
                var.sc_dir = next(var.sc_cycle)
            if var.lr_cycle:
                var.lr_dir = next(var.lr_cycle)

            # Check if any termination criteria is fulfilled.
            if _terminate(var, l2_last, l2_prev, it):
                break

    # Store final error (l2-norm).
    var.l2 = l2_last


def krylov(grid, model, sfield, efield, var):
    """Krylov Subspace iterative solver for 3D CSEM data.

    Using a Krylov subspace iterative solver (defined in `var.sslsolver`)
    implemented in SciPy with or without multigrid as a pre-conditioner
    ([Muld06]_).

    - The electric field is stored in-place in ``efield``.
    - The current error (l2-norm) is stored in ``var.l2``.
    - The reference error (l2-norm of sfield) is stored in ``var.l2_refe``.

    This function is called by :func:`solver`.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    model : Model
        Model; ``emg3d.utils.Model`` instance.

    sfield, efield : Field
        Source and electric fields; ``emg3d.utils.Field`` instances.

    var : `MGParameters`-instance
        As returned by :func:`multigrid`.

    """

    # Define matrix operation A x as LinearOperator.
    def amatvec(efield):
        """Compute A x for solver; residual is b-Ax = src-amatvec."""

        # Cast current efield to Field instance.
        efield = utils.Field(grid, efield)

        # Calculate A x.
        rfield = utils.Field(grid)
        njitted.amat_x(
                rfield.fx, rfield.fy, rfield.fz,
                efield.fx, efield.fy, efield.fz, model.eta_x, model.eta_y,
                model.eta_z, model.v_mu_r, grid.hx, grid.hy, grid.hz)

        # Return Field instance.
        return -rfield

    # Initiate LinearOperator A x.
    A = ssl.LinearOperator(
            shape=(grid.nE, grid.nE), dtype=complex, matvec=amatvec)

    # Define MG pre-conditioner as LinearOperator, if `var.cycle`.
    def mg_matvec(sfield):
        """Use multigrid as pre-conditioner."""

        # Cast current fields to Field instances.
        sfield = utils.Field(grid, sfield)
        efield = utils.Field(grid)

        # Solve for these fields.
        multigrid(grid, model, sfield, efield, var)

        return efield

    # Initiate LinearOperator M.
    M = None
    if var.cycle:
        M = ssl.LinearOperator(
                shape=(grid.nE, grid.nE), dtype=complex, matvec=mg_matvec)

    # Define callback to keep track of sslsolver-iterations.
    def callback(x):
        """Solver iteration count and error (l2-norm)."""
        # Update iteration count.
        var._ssl_it += 1

        # Calculate and print error (only if verbose).
        if var.verb > 2:

            # 'gmres' returns the error, not the solution, in the callback.
            if var.sslsolver == 'gmres':
                var.l2 = x
            else:
                var.l2 = residual(
                        grid, model, sfield, utils.Field(grid, x), True)

            log = f"   [{var.time.now}]   {var.l2/var.l2_refe:.3e} "
            log += f" after {var._ssl_it:3} {var.sslsolver}-cycles"

            # For those solvers who run an iteration before the first
            # preconditioner run ['lgmres', 'gcrotmk'].
            if var._ssl_it == 1 and var.it == 0 and var.cycle is not None:
                log += "\n"

            var.cprint(log, 2)

    # Solve the system with sslsolver.
    efield.field, i = getattr(ssl, var.sslsolver)(
            A=A, b=sfield, x0=efield, tol=var.tol, maxiter=var.ssl_maxit,
            atol=1e-30, M=M, callback=callback)

    # Calculate final error, if not done in the callback.
    if var.verb < 3:
        var.l2 = residual(grid, model, sfield, utils.Field(grid, efield), True)

    # Convergence-checks for sslsolver.
    if i < 0:
        var.cprint(f"\n* ERROR   :: Error in {var.sslsolver}.", -1)
    elif i > 0:
        var.cprint("\n   > MAX. ITERATION REACHED, NOT CONVERGED", 1)
    else:
        var.cprint("\n   > CONVERGED", 1)


# MULTIGRID SUB-ROUTINES
def smoothing(grid, model, sfield, efield, nu, lr_dir):
    """Reducing high-frequency error by smoothing.

    Solves the linear equation system :math:`A x = b` iteratively using the
    Gauss-Seidel method. This acts as smoother or, on the coarsest grid, as a
    direct solver.


    This is a simple wrapper for the jitted calculation in
    :func:`emg3d.njitted.gauss_seidel`, :func:`emg3d.njitted.gauss_seidel_x`,
    :func:`emg3d.njitted.gauss_seidel_y`, and
    :func:`emg3d.njitted.gauss_seidel_z` (``@njit`` can not [yet] access class
    attributes). See these functions for more details and corresponding theory.

    The electric fields are updated in-place.

    This function is called by :func:`multigrid`.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    model : Model
        Model; ``emg3d.utils.Model`` instances.

    sfield, efield : Field
        Source and electric fields; ``emg3d.utils.Field`` instances.

    nu : int
        Number of Gauss-Seidel steps; odd numbers are forward, even numbers are
        reversed. E.g., ``nu=2`` is one symmetric Gauss-Seidel iteration, with
        a forward and a backward step.

    lr_dir : int
        Direction of line relaxation {0, 1, 2, 3, 4, 5, 6, 7}.

    """

    # Collect Gauss-Seidel input (same for all routines)
    inp = (sfield.fx, sfield.fy, sfield.fz, model.eta_x, model.eta_y,
           model.eta_z, model.v_mu_r, grid.hx, grid.hy, grid.hz, nu)

    # Avoid line relaxation in a direction where there are only two cells.
    lr_dir = _current_lr_dir(lr_dir, grid)

    # Calculate and store fields (in-place)
    if lr_dir == 0:             # Standard MG
        njitted.gauss_seidel(efield.fx, efield.fy, efield.fz, *inp)

    if lr_dir in [1, 5, 6, 7]:  # Line relaxation in x-direction
        njitted.gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)

    if lr_dir in [2, 4, 6, 7]:  # Line relaxation in y-direction
        njitted.gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)

    if lr_dir in [3, 4, 5, 7]:  # Line relaxation in z-direction
        njitted.gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)


def restriction(grid, model, sfield, residual, sc_dir):
    """Downsampling of grid, model, and fields to a coarser grid.

    The restriction of the residual is used as source term for the coarse grid.

    Corresponds to Equations 8 and 9 in [Muld06]_ and surrounding text. In the
    case of the restriction of the residual, this function is a wrapper for the
    jitted functions :func:`emg3d.njitted.restrict_weights` and
    :func:`emg3d.njitted.restrict` (``@njit`` can not [yet] access class
    attributes). See these functions for more details and corresponding theory.

    This function is called by :func:`multigrid`.


    Parameters
    ----------
    grid : TensorMesh
        Fine grid; ``emg3d.utils.TensorMesh`` instances.

    model : Model
        Fine model; ``emg3d.utils.Model`` instances.

    sfield : Field
        Fine source field; ``emg3d.utils.Field`` instances.

    sc_dir : int
        Direction of semicoarsening (0, 1, 2, or 3).


    Returns
    -------
    cgrid : TensorMesh
        Coarse grid; ``emg3d.utils.TensorMesh`` instances.

    cmodel : Model
        Coarse model; ``emg3d.utils.Model`` instances.

    csfield : Field
        Coarse source field; ``emg3d.utils.Field`` instances.
        Corresponds to the restriction of the fine-grid residual.

    cefield : Field
        Coarse electric field, complex zeroes; ``emg3d.utils.Field``
        instances.

    """

    # 1. RESTRICT GRID

    # We take every second element for the direction(s) of coarsening.
    rx, ry, rz = 2, 2, 2
    if sc_dir in [1, 5, 6]:  # No coarsening in x-direction.
        rx = 1
    if sc_dir in [2, 4, 6]:  # No coarsening in y-direction.
        ry = 1
    if sc_dir in [3, 4, 5]:  # No coarsening in z-direction.
        rz = 1

    # Calculate distances of coarse grid.
    ch = [np.diff(grid.vectorNx[::rx]),
          np.diff(grid.vectorNy[::ry]),
          np.diff(grid.vectorNz[::rz])]

    # Create new ``TensorMesh``-instance for coarse grid
    cgrid = utils.TensorMesh(ch, grid.x0)

    # 2. RESTRICT MODEL

    class Model:
        """Dummy class to create coarse-grid model."""
        def __init__(self, case):
            """Initialize with case."""
            self.case = case

    cmodel = Model(model.case)
    cmodel.eta_x = _restrict_model_parameters(model.eta_x, sc_dir)
    if model.case in [1, 3]:  # HTI or tri-axial.
        cmodel.eta_y = _restrict_model_parameters(model.eta_y, sc_dir)
    else:
        cmodel.eta_y = cmodel.eta_x
    if model.case in [2, 3]:  # VTI or tri-axial.
        cmodel.eta_z = _restrict_model_parameters(model.eta_z, sc_dir)
    else:
        cmodel.eta_z = cmodel.eta_x
    cmodel.v_mu_r = _restrict_model_parameters(model.v_mu_r, sc_dir)

    # 3. RESTRICT FIELDS

    # Get the weights (Equation 9 of [Muld06]_).
    wx, wy, wz = _get_restriction_weights(grid, cgrid, sc_dir)

    # Calculate the source terms (Equation 8 in [Muld06]_).
    csfield = utils.Field(cgrid)  # Create empty coarse source field instance.
    njitted.restrict(csfield.fx, csfield.fy, csfield.fz, residual.fx,
                     residual.fy, residual.fz, wx, wy, wz, sc_dir)

    # Ensure PEC and initiate empty e-field.
    csfield.ensure_pec
    cefield = utils.Field(cgrid)

    return cgrid, cmodel, csfield, cefield


def prolongation(grid, efield, cgrid, cefield, sc_dir):
    """Interpolating the electric field from coarse grid to fine grid.

    The prolongation from a coarser to a finer grid is the inverse process of
    the restriction (:func:`restriction`) from a finer to a coarser grid. The
    interpolated values of the coarse grid electric field are added to the fine
    grid electric field, in-place. Piecewise constant interpolation is used in
    the direction of the field, and bilinear interpolation in the other two
    directions.

    See Equation 10 in [Muld06]_ and surrounding text.

    This function is called by :func:`multigrid`.


    Parameters
    ----------
    grid, cgrid : TensorMesh
        Fine and coarse grids; ``emg3d.utils.TensorMesh`` instances.

    efield, cefield : Fields
        Fine and coarse grid electric fields; ``emg3d.utils.Field`` instances.

    sc_dir : int
        Direction of semicoarsening (0, 1, 2, or 3).

    """

    # Interpolate ex in y-z-slices.
    yz_points = _get_prolongation_coordinates(grid, 'y', 'z')
    fn = RegularGridProlongator(cgrid.vectorNy, cgrid.vectorNz, yz_points)
    for ixc in range(cgrid.nCx):
        # Bilinear interpolation in the y-z plane
        hh = fn(cefield.fx[ixc, :, :]).reshape(grid.vnEx[1:], order='F')

        # Piecewise constant interpolation in x-direction
        if sc_dir not in [1, 5, 6]:
            efield.fx[2*ixc, :, :] += hh
            efield.fx[2*ixc+1, :, :] += hh
        else:
            efield.fx[ixc, :, :] += hh

    # Interpolate ey in x-z-slices.
    xz_points = _get_prolongation_coordinates(grid, 'x', 'z')
    fn = RegularGridProlongator(cgrid.vectorNx, cgrid.vectorNz, xz_points)
    for iyc in range(cgrid.nCy):

        # Bilinear interpolation in the x-z plane
        hh = fn(cefield.fy[:, iyc, :]).reshape(grid.vnEy[::2], order='F')

        # Piecewise constant interpolation in y-direction
        if sc_dir not in [2, 4, 6]:
            efield.fy[:, 2*iyc, :] += hh
            efield.fy[:, 2*iyc+1, :] += hh
        else:
            efield.fy[:, iyc, :] += hh

    # Interpolate ez in x-y-slices.
    xy_points = _get_prolongation_coordinates(grid, 'x', 'y')
    fn = RegularGridProlongator(cgrid.vectorNx, cgrid.vectorNy, xy_points)
    for izc in range(cgrid.nCz):

        # Bilinear interpolation in the x-y plane
        hh = fn(cefield.fz[:, :, izc]).reshape(grid.vnEz[:-1], order='F')

        # Piecewise constant interpolation in z-direction
        if sc_dir not in [3, 4, 5]:
            efield.fz[:, :, 2*izc] += hh
            efield.fz[:, :, 2*izc+1] += hh
        else:
            efield.fz[:, :, izc] += hh

    # Ensure PEC boundaries
    efield.ensure_pec


def residual(grid, model, sfield, efield, norm=False):
    r"""Calculating the residual.

    Returns the complete residual as given in [Muld06]_, page 636, middle of
    the right column:

    .. math::

        \mathbf{r} = V \left( \mathrm{i}\omega\mu_0\mathbf{J_s}
                     + \mathrm{i}\omega\mu_0 \tilde{\sigma} \mathbf{E}
                     - \nabla \times \mu_\mathrm{r}^{-1} \nabla \times
                       \mathbf{E} \right) .

    This is a simple wrapper for the jitted calculation in
    :func:`emg3d.njitted.amat_x` (``@njit`` can not [yet] access class
    attributes). See :func:`emg3d.njitted.amat_x` for more details and
    corresponding theory.

    This function is called by :func:`multigrid`.


    Parameters
    ----------
    grid : TensorMesh
        Fine grid; ``emg3d.utils.TensorMesh``-instance.

    model : Model
        Fine model; ``emg3d.utils.Model`` instance.

    sfield, efield : Field
        Source and electric fields; ``emg3d.utils.Field`` instances.

    norm : bool
        If True, the error (l2-norm) of the residual is returned, not the
        residual.


    Returns
    -------
    residual : Field
        Returned if ``norm=False``. The residual field; ``emg3d.utils.Field``
        instance.

    norm : float
        Returned if ``norm=True``. The error (l2-norm) of the residual

    """
    # Get residual.
    rfield = sfield.copy()
    njitted.amat_x(rfield.fx, rfield.fy, rfield.fz, efield.fx, efield.fy,
                   efield.fz, model.eta_x, model.eta_y, model.eta_z,
                   model.v_mu_r, grid.hx, grid.hy, grid.hz)

    if norm:  # Return its error.
        return njitted.l2norm(rfield)
    else:     # Return residual.
        return rfield


# VARIABLE DATACLASS
@dataclass
class MGParameters:
    """Collect multigrid solver settings.

    This dataclass is used by the main :func:`solver`-routine. See
    :func:`solver` for a description of the mandatory and optional input
    parameters and more information .

    Returns
    -------
    var : `MGParameters`-instance
        As required by :func:`multigrid`.

    """

    # (A) Parameters without default values (mandatory).
    verb: int
    cycle: str
    sslsolver: str
    linerelaxation: int
    semicoarsening: int
    vnC: tuple  # Finest grid dimension

    # (B) Parameters with default values
    # Convergence tolerance.
    tol: float = 1e-6
    # Maximum iteration.
    maxit: int = 50
    # Initial fine-grid smoothing steps before first iteration.
    nu_init: int = 0
    # Pre-smoothing steps.
    nu_pre: int = 2
    # Smoothing steps on coarsest grid.
    nu_coarse: int = 1
    # Post-smoothing steps.
    nu_post: int = 2
    # Coarsest level; automatically determined if a negative number is given.
    clevel: int = -1

    def __post_init__(self):
        """Set and check some of the parameters."""

        # 0. Set some additional variables
        self._level_all = list()   # To keep track of the levels for QC-figure.
        self._first_cycle = True   # Flag if in first cycle for QC-figure.
        self.it = 0                # To store MG cycle count
        self._ssl_it = 0           # To store solver iteration count
        self.l2 = 1.0              # To store current error
        self.l2_refe = 1.0         # To store reference error

        self.time = utils.Time()   # Timer

        # 1. Set everything related to semicoarsening and line relaxation.
        self._semicoarsening()
        self._linerelaxation()

        # 2. Set everything to used solver and MG-cycle.
        self._solver_and_cycle()

        # 3. Check max coarsening level
        self.max_level

    def __repr__(self):
        """Print all relevant parameters."""

        outstring = (
            f"   MG-cycle       : {self.cycle!r:17}"
            f"   sslsolver : {self.sslsolver!r}\n"
            f"   semicoarsening : {self.__p_sc_dir:17}"
            f"   tol       : {self.tol}\n"
            f"   linerelaxation : {self.__p_lr_dir:17}"
            f"   maxit     : {self.__maxit}\n"
            f"   nu_{{i,1,c,2}}   : {self.nu_init}, {self.nu_pre}"
            f", {self.nu_coarse}, {self.nu_post}       "
            f"   verb      : {self.verb}\n"
            f"   Original grid  "
            f": {self.vnC[0]:3} x {self.vnC[1]:3} x {self.vnC[2]:3}  "
            f"   => {self.vnC[0]*self.vnC[1]*self.vnC[2]:,} cells\n"
            f"   Coarsest grid  : {self.pclevel['vnC'][0]:3} "
            f"x {self.pclevel['vnC'][1]:3} x {self.pclevel['vnC'][2]:3}  "
            f"   => {self.pclevel['nC']:,} cells\n"
            f"   Coarsest level : {self.pclevel['clevel'][0]:3} "
            f"; {self.pclevel['clevel'][1]:3} ;{self.pclevel['clevel'][2]:4} "
            f"  {self.pclevel['message']}"
            f"\n"
        )

        return outstring

    @property
    def max_level(self):
        r"""Sets dimension-dependent level variable ``clevel``.

        Requires at least two cells in each direction (for ``nCx``, ``nCy``,
        and ``nCz``).
        """

        # Store maximum division-by-two level for each dimension.
        # After that, clevel = [nx, ny, nz], where nx, ny, and nz are the
        # number of times you can divide by two in this dimension.
        clevel = np.zeros(3, dtype=int)
        for i in range(3):
            n = self.vnC[i]
            while n % 2 == 0 and n > 2:
                clevel[i] += 1
                n /= 2

        # Restrict to max coarsening level provided by user.
        olevel = self.clevel  # Store user input in olevel for checks below.
        for i in range(3):
            if self.clevel > -1 and self.clevel < clevel[i]:
                clevel[i] = self.clevel

        # Set overall clevel and store.
        self.clevel = np.array(
            [max(clevel[0], clevel[1], clevel[2]),  # Max-level if sc_dir=0
             max(clevel[1], clevel[2]),             # Max-level if sc_dir=1
             max(clevel[0], clevel[2]),             # Max-level if sc_dir=2
             max(clevel[0], clevel[1])]             # Max-level if sc_dir=3
        )

        # Store coarsest nr of cells on coarsest grid and dimension for the
        # log-printing.
        sx = int(self.vnC[0]/2**clevel[0])
        sy = int(self.vnC[1]/2**clevel[1])
        sz = int(self.vnC[2]/2**clevel[2])
        self.pclevel = {'nC': sx*sy*sz, 'vnC': (sx, sy, sz), 'clevel': clevel}

        # Check some grid characteristics (only if olevel > -1)
        # Good values up to 1024 are
        # - 2*2^{0, 1, ..., 9}: 2,  4,  8, 16,  32,  64, 128, 256, 512, 1024,
        # - 3*2^{0, 1, ..., 8}: 3,  6, 12, 24,  48,  96, 192, 384, 768,
        # - 5*2^{0, 1, ..., 7}: 5, 10, 20, 40,  80, 160, 320, 640,
        # - 7*2^{0, 1, ..., 7}: 7, 14, 28, 56, 112, 224, 448, 896,
        # and preference decreases from top to bottom row.
        self.pclevel['message'] = ""
        if olevel > -1:
            # Check if highest prime is not in {2, 3, 5, 7}.
            high_prime = np.any(np.array([sx, sy, sz]) > 7)

            # 105 corresponds to 3*5*7, hence coarsest grid of three smallest
            # primes.
            big_coarse = sx*sy*sz > 105

            if big_coarse or high_prime:
                self.pclevel['message'] = "\n\n"+11*" "+"=> Grid is "
                self.pclevel['message'] += "not optimal for MG solver <="

        # Check at least two cells in each direction
        if np.any(np.array(self.vnC) < 2):
            print("* ERROR   :: Nr. of cells must be at least two in each\n"
                  "             direction. Provided shape: "
                  f"({self.vnC[0]}, {self.vnC[1]}, {self.vnC[2]}).")
            raise ValueError('nCx/nCy/nCz')

    def cprint(self, info, verbosity, **kwargs):
        """Conditional printing.

        Prints `info` if `self.verb` > `verbosity`.

        Parameters
        ----------
        info : str
            String to be printed.

        verbosity : int
            Verbosity of info.

        kwargs : optional
            Arguments passed to `print`.

        """
        if self.verb > verbosity:
            print(info, **kwargs)

    def _semicoarsening(self):
        """Set everything related to semicoarsening."""

        # Check input.
        if self.semicoarsening is True:            # If True, cycle [1, 2, 3].
            sc_cycle = np.array([1, 2, 3])
            self.sc_cycle = itertools.cycle(sc_cycle)
        elif self.semicoarsening in np.arange(4):  # If 0-4, use this.
            sc_cycle = np.array([int(self.semicoarsening)])
            self.sc_cycle = False
        else:                                      # Else, use numbers.
            sc_cycle = np.array([int(x) for x in
                                 str(abs(self.semicoarsening))])
            self.sc_cycle = itertools.cycle(sc_cycle)

            # Ensure numbers are within 0 <= sc_dir <= 3
            if np.any(sc_cycle < 0) or np.any(sc_cycle > 3):
                print("* ERROR   :: `semicoarsening` must be one of  "
                      f"(False, True, 0, 1, 2, 3).\n"
                      f"{' ':>13} Or a combination of (0, 1, 2, 3) to cycle, "
                      f"e.g. 1213.\n{'Provided:':>23} "
                      f"semicoarsening={self.semicoarsening}.")
                raise ValueError('semicoarsening')

        # Get first (or only) direction.
        if self.sc_cycle:
            self.sc_dir = next(self.sc_cycle)
        else:
            self.sc_dir = sc_cycle[0]

        # Set semicoarsening to True/False; print statement
        self.semicoarsening = self.sc_dir != 0
        self.__p_sc_dir = f"{self.semicoarsening} {sc_cycle}"
        self.__raw_sc_cycle = sc_cycle

    def _linerelaxation(self):
        """Set everything related to line relaxation."""

        # Check input.
        if self.linerelaxation is True:            # If True, cycle [1, 2, 3].
            lr_cycle = np.array([4, 5, 6])
            self.lr_cycle = itertools.cycle(lr_cycle)
        elif self.linerelaxation in np.arange(8):  # If 0-7, use this.
            lr_cycle = np.array([int(self.linerelaxation)])
            self.lr_cycle = False
        else:                                      # Else, use numbers.
            lr_cycle = np.array([int(x) for x in
                                 str(abs(self.linerelaxation))])
            self.lr_cycle = itertools.cycle(lr_cycle)

            # Ensure numbers are within 0 <= lr_dir <= 7
            if np.any(lr_cycle < 0) or np.any(lr_cycle > 7):
                print("* ERROR   :: `linerelaxation` must be one of  "
                      f"(False, True, 0, 1, 2, 3, 4, 5, 6, 7).\n"
                      f"{' ':>13} Or a combination of (1, 2, 3, 4, 5, 6, 7) "
                      f"to cycle, e.g. 1213.\n{'Provided:':>23} "
                      f"linerelaxation={self.linerelaxation}.")
                raise ValueError('linerelaxation')

        # Get first (only) direction
        if self.lr_cycle:
            self.lr_dir = next(self.lr_cycle)
        else:
            self.lr_dir = lr_cycle[0]

        # Set linerelaxation to True/False; print statement
        self.linerelaxation = self.lr_dir != 0
        self.__p_lr_dir = f"{self.linerelaxation} {lr_cycle}"
        self.__raw_lr_cycle = lr_cycle

    def _solver_and_cycle(self):
        """Set everything related to solver and MG-cycle."""

        # sslsolver.
        solvers = ['bicgstab', 'cgs', 'gmres', 'lgmres', 'gcrotmk']
        if self.sslsolver is True:
            self.sslsolver = 'bicgstab'
        elif self.sslsolver is not False and self.sslsolver not in solvers:
            print(f"* ERROR   :: `sslsolver` must be True, False, or one of")
            print(f"             {solvers}.")
            print(f"             Provided: sslsolver={self.sslsolver!r}.")
            raise ValueError('sslsolver!r')

        if self.cycle not in ['F', 'V', 'W', None]:
            print("* ERROR   :: `cycle` must be one of {'F', 'V', 'W', None}."
                  f"\n             Provided: cycle={self.cycle}.")
            raise ValueError('cycle')

        # Add maximum MG cycles depending on cycle
        if self.cycle in ['F', 'W']:
            self.cycmax = 2
        else:
            self.cycmax = 1

        # Ensure at least cycle or sslsolver is set
        if not self.sslsolver and not self.cycle:
            print("* ERROR   :: At least `cycle` or `sslsolver` is "
                  "required.\n             Provided input: "
                  f"cycle={self.cycle}; sslsolver={self.sslsolver}.")
            raise ValueError('cycle/sslsolver')

        # Store maxit in ssl_maxit and adjust maxit if sslsolver.
        self.ssl_maxit = 0              # Maximum iteration
        self.__maxit = f"{self.maxit}"  # For printing
        if self.sslsolver:
            self.ssl_maxit = self.maxit
            if self.cycle is not None:  # Only if MG is used
                self.maxit = max(len(self.__raw_sc_cycle),
                                 len(self.__raw_lr_cycle))
                self.__maxit += f" ({self.maxit})"  # For printing


# INTERPOLATION DATACLASS
class RegularGridProlongator:
    """Prolongate field from coarse to fine grid.

    This is a heavily modified and adapted version of
    :class:`scipy.interpolate.RegularGridInterpolator`.

    The main difference (besides the pre-sets) is that this version allows to
    initiate an instance with the coarse and fine grids. This initialize will
    calculate the required weights, and it has therefore only to be done once.

    After this, interpolating values from the coarse to the fine grid can be
    carried out much faster.

    Simplifications in comparison to
    :class:`scipy.interpolate.RegularGridInterpolator`:

    - No sanity checks what-so-ever.
    - Only 2D data;
    - ``method='linear'``;
    - ``bounds_error=False``;
    - ``fill_value=None``.

    It results in a speed-up factor of about 2, independent of grid size, for
    this particular case. The prolongation is the second-most expensive part of
    multigrid after the smoothing. Trying to improve this further might
    therefore be useful.

    Parameters
    ----------
    x, y : ndarray
        The x/y-coordinates defining the coarse grid.

    cxy : ndarray of shape (..., 2)
        The ([[x], [y]]).T-coordinates defining the fine grid.

    """

    def __init__(self, x, y, cxy):
        self.size = cxy.shape[0]
        self._set_edges_and_weights((x, y), cxy)

    def __call__(self, values):
        """Return values of coarse grid on fine grid locations.

        Parameters
        ----------
        values : ndarray
            Values corresponding to x/y-coordinates.

        Returns
        -------
        result : ndarray
            Values corresponding to cxy-coordinates.

        """
        # Initiate result.
        result = 0.

        # Find relevant values.
        for n, edge_indices in enumerate(self._get_edges_copy()):
            result += np.asarray(values[edge_indices]) * self.weight[n, :]

        return result

    def _set_edges_and_weights(self, xy, cxy):
        """Calculate weights to go from xy- to cxy-coordinates."""

        # Find relevant edges between which cxy are situated.
        indices = []

        # Compute distance to lower edge in unity units.
        norm_distances = []

        # Iterate through dimensions.
        for x, grid in zip(cxy.T, xy):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))

        # Find relevant values; each i and i+1 represents a edge.
        self.edges = itertools.product(*[[i, i + 1] for i in indices])

        # Calculate weights.
        self.weight = np.ones((4, self.size))
        for n, edge_indices in enumerate(self._get_edges_copy()):
            partial_weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                partial_weight *= np.where(ei == i, 1 - yi, yi)
            self.weight[n, :] = partial_weight

    def _get_edges_copy(self):
        """Return a copy of the edges-iterator."""
        self.edges, edges = itertools.tee(self.edges)
        return edges


# MG HELPER ROUTINES
def _current_sc_dir(sc_dir, grid):
    """Return current direction(s) for semicoarsening.

    Semicoarsening is defined in ``self.sc_dir``. Here ``self.sc_dir`` is
    checked with which directions can actually still be halved, and
    depending on that, an adjusted ``sc_dir`` is returned for this
    particular grid.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    sc_dir : int
        Direction of semicoarsening.

    Returns
    -------
    c_sc_dir : int
        Current direction of semicoarsening.

    """
    # Find out in which direction we want to half the number of cells.
    # This depends on an (optional) direction of semicoarsening, and
    # if the number of cells in a direction can still be halved.
    xsc_dir = grid.nCx % 2 != 0 or grid.nCx < 3 or sc_dir == 1
    ysc_dir = grid.nCy % 2 != 0 or grid.nCy < 3 or sc_dir == 2
    zsc_dir = grid.nCz % 2 != 0 or grid.nCz < 3 or sc_dir == 3

    # Set current sc_dir depending on the above outcome.
    if xsc_dir:
        if ysc_dir:
            c_sc_dir = 6  # Only coarsen in z-direction.
        elif zsc_dir:
            c_sc_dir = 5  # Only coarsen in y-direction.
        else:
            c_sc_dir = 1  # Coarsen in y- and z-directions.
    elif ysc_dir:
        if zsc_dir:
            c_sc_dir = 4  # Only coarsen in x-direction.
        else:
            c_sc_dir = 2  # Coarsen in x- and z-directions.
    elif zsc_dir:
        c_sc_dir = 3  # Coarsen in x- and y-directions.
    else:
        c_sc_dir = 0  # Coarsen in all directions.

    return c_sc_dir


def _current_lr_dir(lr_dir, grid):
    """Return current direction(s) for line relaxation.

    Line relaxation is defined in ``self.lr_dir``. Here ``self.lr_dir`` is
    checked with the dimension of the grid, to avoid line relaxation in a
    direction where there are only two cells.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    lr_dir : int
        Direction of line relaxation {0, 1, 2, 3, 4, 5, 6, 7}.


    Returns
    -------
    lr_dir : int
        Current direction of line relaxation.

    """
    lr_dir = np.copy(lr_dir)

    if grid.nCx == 2:  # Check x-direction.
        if lr_dir == 1:
            lr_dir = 0
        elif lr_dir == 5:
            lr_dir = 3
        elif lr_dir == 6:
            lr_dir = 2
        elif lr_dir == 7:
            lr_dir = 4

    if grid.nCy == 2:  # Check y-direction.
        if lr_dir == 2:
            lr_dir = 0
        elif lr_dir == 4:
            lr_dir = 3
        elif lr_dir == 6:
            lr_dir = 1
        elif lr_dir == 7:
            lr_dir = 5

    if grid.nCz == 2:  # Check z-direction.
        if lr_dir == 3:
            lr_dir = 0
        elif lr_dir == 4:
            lr_dir = 2
        elif lr_dir == 5:
            lr_dir = 1
        elif lr_dir == 7:
            lr_dir = 6

    return lr_dir


def _print_cycle_info(var, l2_last, l2_prev):
    """Print cycle info to log.

    Parameters
    ----------
    var : `MGParameters`-instance
        As returned by :func:`multigrid`.

    l2_last, l2_prev : float
        Last and previous errors (l2-norms).

    """

    # Start info string, return if not enough verbose.
    if var.verb < 3:
        # Set first_cycle to False, to stop logging.
        var._first_cycle = False
        return
    elif var.verb > 3:
        info = "\n"
    else:
        info = ""

    # Add multigrid-cycle visual QC on first cycle.
    if var._first_cycle:

        # Cast levels into array, get maximum.
        _lvl_all = np.array(var._level_all, dtype=int)
        lvl_max = np.max(_lvl_all)

        # Get levels, multiply by difference to get +/-.
        lvl = (_lvl_all[1:] + _lvl_all[:-1])//2+1
        lvl *= _lvl_all[1:] - _lvl_all[:-1]

        # Create info string.
        out = [f"       h_\n"]
        slen = min(len(lvl), 70)
        for cl in range(lvl_max):
            out += f"   {2**(cl+1):4}h_ "
            out += [" " if abs(lvl[v]) != cl+1 else "\\" if
                    lvl[v] > 0 else "/" for v in range(slen)]
            if cl < lvl_max-1:
                out.append("\n")

        # Add the cycle to inf.
        info += "".join(out)
        info += "\n\n"
        if len(lvl) > 70:
            info += "  (Cycle-QC restricted to first 70 steps of "
            info += f"{len(lvl)} steps.)\n"

        # Set first_cycle to False, to reduce verbosity from now on.
        var._first_cycle = False

    # Add iteration log.
    info += f"   [{var.time.now}]   {l2_last/var.l2_refe:.3e}  "
    if var.sslsolver:  # For multigrid as preconditioner.
        info += f"after {19*' '} {var.it:3} {var.cycle}-cycles "

    else:              # For multigrid as solver.
        info += f"after {var.it:3} {var.cycle}-cycles   "
        info += f"[{l2_last:.3e}, {l2_last/l2_prev:.3f}]"
    info += f"   {var.lr_dir} {var.sc_dir}"

    if var.verb > 3:
        info += "\n"

    # Print the info.
    var.cprint(info, 2)


def _print_gs_info(it, level, cycmax, grid, norm, text):
    """Print info to log after each Gauss-Seidel smoothing step.

    Parameters
    ----------
    it : int
        Iteration number.

    level : int
        Current coarsening level.

    cycmax : int
        Maximum MG cycles.

    grid : TensorMesh
        Current grid; ``emg3d.utils.TensorMesh`` instance.

    norm : float
        Current error (l2-norm).

    text : str
        Info about Gauss-Seidel step.

    """

    info = f"     {it:2} {level} {cycmax} [{grid.nCx:3}, {grid.nCy:3}, "
    info += f"{grid.nCz:3}]: {norm:.3e} {text}"
    print(info)


def _terminate(var, l2_last, l2_prev, it):
    """Return multigrid termination flag.

    Checks all termination criteria and returns True if at least one is
    fulfilled.


    Parameters
    ----------
    var : `MGParameters`-instance
        As returned by :func:`multigrid`.

    l2_last, l2_prev : float
        Last and previous erros (l2-norms).

    it : int
        Iteration number.


    Returns
    -------
    finished : bool
        Boolean indicating if multigrid is finished.

    """

    finished = False

    if var.sslsolver:  # If multigrid as preconditioner, exit silently.
        if it == var.maxit:
            finished = True

    else:
        # Initialize info string.
        info = ""

        if var.verb < 4:
            add = "\n"
        else:
            add = ""

        if l2_last < var.tol*var.l2_refe:    # Converged.
            info += add+"   > CONVERGED\n"
            finished = True

        elif l2_last > 10*var.l2_refe:       # Diverged.
            info += add+"   > DIVERGED\n"
            finished = True

        elif it > 2 and l2_last >= l2_prev:  # Stagnated.
            info += add+"   > STAGNATED\n"
            finished = True

        elif it == var.maxit:                # Maximum iterations reached.
            info += add+"   > MAX. ITERATION REACHED, NOT CONVERGED\n"
            finished = True

        # Print info.
        var.cprint(info, 1, end="")

    return finished


def _restrict_model_parameters(param, sc_dir):
    """Restrict model parameters.

    Parameters
    ----------
    param : ndarray
        Model parameter to restrict.

    sc_dir : int
        Direction of semicoarsening.

    Returns
    -------
    out : ndarray
        Restricted model parameter.

    """
    if sc_dir == 1:    # Only sum the four cells in y-z-plane
        out = param[:, :-1:2, :-1:2] + param[:, 1::2, :-1:2]
        out += param[:, :-1:2, 1::2] + param[:, 1::2, 1::2]
    elif sc_dir == 2:  # Only sum the four cells in x-z-plane
        out = param[:-1:2, :, :-1:2] + param[1::2, :, :-1:2]
        out += param[:-1:2, :, 1::2] + param[1::2, :, 1::2]
    elif sc_dir == 3:  # Only sum the four cells in x-y-plane
        out = param[:-1:2, :-1:2, :] + param[1::2, :-1:2, :]
        out += param[:-1:2, 1::2, :] + param[1::2, 1::2, :]
    elif sc_dir == 4:  # Only sum the two cells in x-direction
        out = param[:-1:2, :, :] + param[1::2, :, :]
    elif sc_dir == 5:  # Only sum the two cells y-direction
        out = param[:, :-1:2, :] + param[:, 1::2, :]
    elif sc_dir == 6:  # Only sum the two cells z-direction
        out = param[:, :, :-1:2] + param[:, :, 1::2]
    else:            # Standard: Sum all 8 cells.
        out = param[:-1:2, :-1:2, :-1:2] + param[1::2, :-1:2, :-1:2]
        out += param[:-1:2, :-1:2, 1::2] + param[1::2, :-1:2, 1::2]
        out += param[:-1:2, 1::2, :-1:2] + param[1::2, 1::2, :-1:2]
        out += param[:-1:2, 1::2, 1::2] + param[1::2, 1::2, 1::2]
    return out


def _get_restriction_weights(grid, cgrid, sc_dir):
    """Return restriction weights.

    Return the weights (Equation 9 of [Muld06]_). The corresponding weights are
    not actually used in the case of semicoarsening. We still have to provide
    arrays of the correct format though, otherwise numba will complain in the
    jitted functions.


    Parameters
    ----------
    grid, cgrid : TensorMesh
        Fine and coarse grids; ``emg3d.utils.TensorMesh`` instances.

    sc_dir : int
        Direction of semicoarsening.


    Returns
    -------
    wx, wy, wz : ndarray
        Restriction weights.

    """
    if sc_dir not in [1, 5, 6]:
        wx = njitted.restrict_weights(
                grid.vectorNx, grid.vectorCCx, grid.hx, cgrid.vectorNx,
                cgrid.vectorCCx, cgrid.hx)
    else:
        wxlr = np.zeros(grid.nNx, dtype=float)
        wx0 = np.ones(grid.nNx, dtype=float)
        wx = (wxlr, wx0, wxlr)

    if sc_dir not in [2, 4, 6]:
        wy = njitted.restrict_weights(
                grid.vectorNy, grid.vectorCCy, grid.hy, cgrid.vectorNy,
                cgrid.vectorCCy, cgrid.hy)
    else:
        wylr = np.zeros(grid.nNy, dtype=float)
        wy0 = np.ones(grid.nNy, dtype=float)
        wy = (wylr, wy0, wylr)

    if sc_dir not in [3, 4, 5]:
        wz = njitted.restrict_weights(
                grid.vectorNz, grid.vectorCCz, grid.hz, cgrid.vectorNz,
                cgrid.vectorCCz, cgrid.hz)
    else:
        wzlr = np.zeros(grid.nNz, dtype=float)
        wz0 = np.ones(grid.nNz, dtype=float)
        wz = (wzlr, wz0, wzlr)

    return wx, wy, wz


def _get_prolongation_coordinates(grid, d1, d2):
    """Calculate required coordinates of finer grid for prolongation."""
    D2, D1 = np.broadcast_arrays(
            getattr(grid, 'vectorN'+d2), getattr(grid, 'vectorN'+d1)[:, None])
    return np.r_[D1.ravel('F'), D2.ravel('F')].reshape(-1, 2, order='F')
