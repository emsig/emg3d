"""
The actual multigrid solver routines. The computationally intensive parts,
however, are in :mod:`emg3d.core` as numba-jitted functions.

The only relevant function, from an end-user perspective, is
:func:`emg3d.solver.solve`. The other functions and classes are not meant to be
called directly, they are all used by the solver internally. It can, however,
still be insightful to look at the documentation and code of these functions if
you are interested in understanding how the multigrid solver works, the theory
and its implementation.

"""
# Copyright 2018-2021 The emsig community.
#
# This file is part of emg3d.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

import itertools
from typing import Union
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as ssl

from emg3d import core, meshes, models, fields, utils

__all__ = ['solve', 'solve_source', 'multigrid', 'krylov', 'smoothing',
           'restriction', 'prolongation', 'residual', 'MGParameters',
           'RegularGridProlongator']


# MAIN USER-FACING FUNCTIONS
def solve(model, sfield, sslsolver=True, semicoarsening=True,
          linerelaxation=True, verb=0, **kwargs):
    r"""Solver for three-dimensional electromagnetic diffusion.

    The principal solver of emg3d is using the multigrid method as presented in
    [Muld06]_. Multigrid can be used as a standalone solver, or as a
    preconditioner for an iterative solver from the
    :mod:`scipy.sparse.linalg`-library, e.g.,
    :func:`scipy.sparse.linalg.bicgstab`. Alternatively, these Krylov subspace
    solvers can also be used without multigrid at all (see the ``cycle`` and
    ``sslsolver`` parameters).

    Implemented are the F-, V-, and W-cycle schemes for multigrid (``cycle``
    parameter), and the amount of smoothing steps (initial smoothing,
    pre-smoothing, coarsest-grid smoothing, and post-smoothing) can be set
    individually (``nu_init``, ``nu_pre``, ``nu_coarse``, and ``nu_post``,
    respectively). The maximum level of coarsening can be restricted with the
    ``clevel`` parameter.

    Semicoarsening and line relaxation, as presented in [Muld07]_, are
    implemented, see the ``semicoarsening`` and ``linerelaxation`` parameters.
    Using the BiCGSTAB solver together with multigrid preconditioning,
    semicoarsening, and line relaxation is generally the most robust solution,
    albeit not necessarily the fastest. It is the default setting for its
    robustness. Just using traditional multigrid without BiCGSTAB nor
    semicoarsening nor line relaxation uses the least memory and is often
    faster but may fail on stretched grids or for models with strong
    anisotropy. However, only testing the parameters for a given model can give
    certainty to which parameters are best.


    Parameters
    ----------
    model : Model
        The model; a :class:`emg3d.models.Model` instance.

    sfield : Field
        The source field. See :func:`emg3d.fields.get_source_field`.

    sslsolver : {str, bool}, default: True
        A :mod:`scipy.sparse.linalg`-solver, to use with multigrid as
        pre-conditioner or on its own (if ``cycle=None``).

        Current possibilities:

            - ``True`` or ``'bicgstab'``: BiConjugate Gradient STABilized
              (:func:`scipy.sparse.linalg.bicgstab`).
            - ``'cgs'``: Conjugate Gradient Squared
              (:func:`scipy.sparse.linalg.cgs`).
            - ``'gcrotmk'``: GCROT: Generalized Conjugate Residual with inner
              Orthogonalization and Outer Truncation
              (:func:`scipy.sparse.linalg.gcrotmk`).

        It does currently not work with ``'cg'``, ``'bicg'``, ``'qmr'``, and
        ``'minres'`` for various reasons (e.g., some require ``rmatvec`` in
        addition to ``matvec``).

    semicoarsening : {int, bool}, default: True
        Semicoarsening.

        - ``True``: Cycling over 1, 2, 3.
        - ``0`` or ``False``: No semicoarsening.
        - ``1``: Semicoarsening in x direction.
        - ``2``: Semicoarsening in y direction.
        - ``3``: Semicoarsening in z direction.
        - Multi-digit number containing digits from 0 to 3. Multigrid will
          cycle over these values, e.g., ``semicoarsening=1213`` will cycle
          over [1, 2, 1, 3].

    linerelaxation : {int, bool}, default: True
        Line relaxation.

        - ``True``: Cycling over [4, 5, 6].
        - ``0`` or ``False``: No line relaxation.
        - ``1``: line relaxation in x direction.
        - ``2``: line relaxation in y direction.
        - ``3``: line relaxation in z direction.
        - ``4``: line relaxation in y and z directions.
        - ``5``: line relaxation in x and z directions.
        - ``6``: line relaxation in x and y directions.
        - ``7``: line relaxation in x, y, and z directions.
        - Multi-digit number containing digits from 0 to 7. Multigrid will
          cycle over these values, e.g., ``linerelaxation=1213`` will cycle
          over [1, 2, 1, 3].

        Note: Smoothing is generally done in lexicographical order, except for
        line relaxation in y direction; the reason is speed (memory access).

    verb : int, default: 0
        Level of verbosity (the higher the more verbose).

        - ``-1``: Nothing.
        - ``0``: Warnings.
        - ``1``: One-liner at the end.
        - ``2``: One-liner, dynamically updated.
        - ``3``: Information about the method plus dynamic one-liner.
        - ``4``: Additional information for each MG-cycle.
        - ``5``: Everything (slower due to additional error computations).

    cycle : {str, None}, default: 'F'
        Type of multigrid cycle.

        - ``'V'``: V-cycle, simplest version.
        - ``'W'``: W-cycle, most expensive version.
        - ``'F'``: F-cycle, sort of a compromise between V- and W-cycle.
        - ``None``: Does not use multigrid, only the chosen ``sslsolver``.

        A ``sslsolver`` has to be provided if ``cycle=None``, and the
        ``sslsolver`` will then be used without multigrid pre-conditioning.

        Comparison of V (left), F (middle), and W (right) cycles for the case
        of four grids (three relaxation and prolongation steps)::

            h_
           2h_   \    /   \          /   \            /
           4h_    \  /     \    /\  /     \    /\    /
           8h_     \/       \/\/  \/       \/\/  \/\/

    efield : Field, default: None
        Initial electric field. If is initiated with zeroes if not provided.

        If an initial efield is provided nothing is returned, but the final
        efield is directly put into the provided efield.

        If an initial field is provided and a sslsolver is used, then it first
        carries out one multigrid cycle without semicoarsening nor line
        relaxation. The sslsolver is at times unstable with an initial guess,
        carrying out one multigrid cycle helps to stabilize it.

        Note that the tangential field at the boundary of a provided efield is
        set to zero to ensure a PEC boundary (perfect electric conductor).

    tol : float, default: 1e-6
        Convergence tolerance.

        Iterations stop as soon as the norm of the residual has decreased by
        this factor, relative to the residual norm obtained for a zero
        electric field.

    maxit : int, default: 50
        Maximum number of multigrid iterations.

        If ``sslsolver`` is used, this applies to the ``sslsolver``.

        In the case that multigrid is used as a pre-conditioner for the
        ``sslsolver``, the maximum iteration for multigrid is defined by the
        maximum length of the ``linerelaxation`` and ``semicoarsening``-cycles.

    nu_init : int, default: 0
        Number of initial smoothing steps, before multigrid cycle.

    nu_pre : int, default: 2
        Number of pre-smoothing steps.

    nu_coarse : int, default: 1
        Number of smoothing steps on coarsest grid.

    nu_post : int, default: 2
        Number of post-smoothing steps.

    clevel : int, default: -1
        The maximum coarsening level can be different for each dimension and
        is, by default, automatically determined (``clevel=-1``). The
        parameter ``clevel`` restricts the maximum coarsening level by its
        value.

    return_info : bool, default: False
        If True, a dictionary is returned with runtime info (final norm,
        number of iterations of multigrid and the sslsolver, log, exit message,
        etc).

    log : int, default: 1
        Only relevant if ``return_info=True``.

        - ``-1``: LOG ONLY: Only store info in log, do not print on screen.
        - ``0``: SCREEN only: Only print info to screen, do not store in log.
        - ``1``: BOTH: Store info in log and print on screen.

    plain : bool, default: False
        Plain multigrid method. This is a shortcut for ``sslsolver=False,
        semicoarsening=False, linerelaxation=False``. The three parameters
        remain unchanged if they are set to anything else than ``True``.


    Returns
    -------
    efield : Field, returned if efield=None
        Resulting electric field. It is not returned but stored in-place if an
        initial efield was provided.

    info_dict : dict, returned if return_info=True
        Dictionary with solver info. Keys:

        - ``exit``: Exit status, 0=Success, 1=Failure;
        - ``exit_message``: Exit message, check this if ``exit=1``;
        - ``abs_error``: Absolute error;
        - ``rel_error``: Relative error;
        - ``ref_error``: Reference error [norm(sfield)];
        - ``tol``: Tolerance (abs_error<ref_error*tol);
        - ``it_mg``: Number of multigrid iterations;
        - ``it_ssl``: Number of SSL iterations;
        - ``time``: Runtime (s).
        - ``runtime_at_cycle``: Runtime after each cycle (s).
        - ``error_at_cycle``: Absolute error after each cycle.
        - ``log``: Stored log.


    Examples
    --------

    .. ipython::

       In [1]: import emg3d
          ...: import numpy as np

       In [2]: # Create a simple grid, 8 cells of length 1 in each direction,
          ...: # starting at the origin.
          ...: hx = np.ones(8)
          ...: grid = emg3d.TensorMesh([hx, hx, hx], origin=(0, 0, 0))

       In [3]: # Create a fullspace model with triaxial anisotropy.
          ...: model = emg3d.Model(grid, property_x=1.5, property_y=1.8,
          ...:                     property_z=3.3, mapping='Resistivity')

       In [4]: # The source is a x-directed, horizontal dipole at (4, 4, 4)
          ...: # with a frequency of 10 Hz.
          ...: coo = (4, 4, 4, 0, 0)  # (x, y, z, azimuth, elevation)
          ...: sfield = emg3d.fields.get_source_field(
          ...:             grid, source=coo, frequency=10)

       In [5]: # Solve for the electric field.
          ...: efield = emg3d.solve(model, sfield, plain=True, verb=4)

    """

    # Extract kwargs which do not go into MGParameters.
    if kwargs.pop('plain', False):
        sslsolver = False if sslsolver is True else sslsolver
        semicoarsening = False if semicoarsening is True else semicoarsening
        linerelaxation = False if linerelaxation is True else linerelaxation
    efield = kwargs.pop('efield', None)

    # Solver settings; get from kwargs or set to default values.
    var = MGParameters(
            sslsolver=sslsolver, semicoarsening=semicoarsening,
            linerelaxation=linerelaxation, shape_cells=model.shape, verb=verb,
            **kwargs
    )

    # Start logging and print all parameters.
    var.cprint(f"\n:: emg3d START :: {var.time.now} :: "
               f"v{utils.__version__}\n", 2)
    var.cprint(var, 2)

    # Compute reference error for tolerance.
    var.l2_refe = sl.norm(sfield.field, check_finite=False)
    var.error_at_cycle[0] = var.l2_refe

    # Check sfield.
    if sfield.frequency is None:
        raise ValueError(
            "Source field is missing frequency information; Create "
            "it with `emg3d.fields.get_source_field`, or initiate it "
            "with `emg3d.fields.Field`, providing frequency information."
        )

    # Initiate volume-averaged model values.
    vmodel = models.VolumeModel(model, sfield)

    # Get efield.
    if efield is None:
        # If not provided, initiate an empty one.
        efield = fields.Field(model.grid, dtype=sfield.field.dtype,
                              frequency=sfield._frequency)

        # Set flag to return the field.
        var.do_return = True
    else:

        # Ensure efield has same data type as sfield.
        if sfield.field.dtype != efield.field.dtype:
            raise ValueError(
                "Source field and electric field must have the same "
                "dtype; complex (f-domain) or real (s-domain). Provided:"
                f"sfield: {sfield.field.dtype}; efield: {efield.field.dtype}."
            )

        # If provided efield is missing frequency information, add it from the
        # source field.
        if efield.frequency is None:
            efield._frequency = sfield._frequency

        # Ensure PEC.
        efield.fx[:, 0, :] = efield.fx[:, -1, :] = 0.
        efield.fx[:, :, 0] = efield.fx[:, :, -1] = 0.
        efield.fy[0, :, :] = efield.fy[-1, :, :] = 0.
        efield.fy[:, :, 0] = efield.fy[:, :, -1] = 0.
        efield.fz[0, :, :] = efield.fz[-1, :, :] = 0.
        efield.fz[:, 0, :] = efield.fz[:, -1, :] = 0.

        # Set flag to NOT return the field.
        var.do_return = False

        # If efield is provided, check if it is already sufficiently good.
        var.l2 = residual(vmodel, sfield, efield, True)
        if var.l2 < var.tol*var.l2_refe:

            # Switch-off both sslsolver and multigrid.
            var.sslsolver = None
            var.cycle = None

            # Start final info.
            var.exit_message = "CONVERGED"
            info = "   > NOTHING DONE (provided efield already good enough)\n"

    # Check if sfield is zero.
    if var.l2_refe < 100*np.finfo(float).tiny:

        # To avoid division by zero for the log.
        var.l2_refe = np.nan

        # Switch-off both sslsolver and multigrid.
        var.sslsolver = None
        var.cycle = None

        # Start final info.
        var.exit_message = "CONVERGED"
        info = "   > RETURN ZERO E-FIELD (provided sfield is zero)\n"

        # Zero-source means zero e-field.
        efield = fields.Field(model.grid, dtype=sfield.field.dtype,
                              frequency=sfield._frequency)

    # Print header for iteration log.
    header = f"   [hh:mm:ss]  {'rel. error':<22}"
    if var.sslsolver:
        header += f"{'solver':<20}"
        if var.cycle:
            header += f"{'MG':<11} l s"
        var.cprint(header+"\n", 3)
    elif var.cycle:
        var.cprint(header+f"{'[abs. error, last/prev]':>29}   l s\n", 3)

    # Solve the system with...
    if var.sslsolver:  # ... sslsolver.
        krylov(vmodel, sfield, efield, var)
    elif var.cycle:    # ... multigrid.
        multigrid(vmodel, sfield, efield, var)

    # Get exit status.
    exit_status = int(var.exit_message != 'CONVERGED')

    # Print runtime information.
    if var.verb in [1, 2]:
        _print_one_liner(var, var.l2, True)
    elif var.verb > 2:
        if var.sslsolver:  # sslsolver-specific info.
            info = f"   > Solver steps     : {var.ssl_it}\n"
            if var.cycle:
                info += f"   > MG prec. steps   : {var.it}\n"
        elif var.cycle:    # multigrid-specific info.
            info = f"   > MG cycles        : {var.it}\n"
        info += f"   > Final rel. error : {var.l2/var.l2_refe:.3e}\n\n"
        info += f":: emg3d END   :: {var.time.now} :: "
        info += f"runtime = {var.time.runtime}\n"
        var.cprint(info, 2)
    elif var.verb == 0 and exit_status == 1:
        var.cprint(f"* WARNING :: {var.exit_message}", -1)

    # Assemble the info_dict if return_info
    if var.return_info:
        info_dict = {
            'exit': exit_status,               # Exit status.
            'exit_message': var.exit_message,  # Exit message.
            'abs_error': var.l2,               # Absolute error.
            'rel_error': var.l2/var.l2_refe,   # Relative error.
            'ref_error': var.l2_refe,    # Reference error [norm(sfield)].
            'tol': var.tol,              # Tolerance (abs_error<ref_error*tol).
            'it_mg': var.it,             # Multigrid iterations.
            'it_ssl': var.ssl_it,        # SSL iterations.
            'time': var.runtime_at_cycle[-1],          # Runtime (s).
            'runtime_at_cycle': var.runtime_at_cycle,  # Runtime at cycle (s).
            'error_at_cycle': var.error_at_cycle,      # Abs. error at cycle.
            'log': var.log_message,                    # Log.
        }

    # Return depending on input arguments; or nothing.
    if var.do_return and var.return_info:  # efield and info.
        return efield, info_dict
    elif var.do_return:                    # efield.
        return efield
    elif var.return_info:                  # info.
        return info_dict


def solve_source(model, source, frequency, **kwargs):
    """Return electric field for a given source and frequency.

    This function is a simple shortcut for the following::

       sfield = emg3d.get_source_field(grid, source, frequency, **kwargs)
       efield = emg3d.solve(model, sfield, **kwargs)

    See the documentation of :func:`emg3d.fields.get_source_field` for the
    description of ``model``, ``source``, and ``frequency``, and
    the documentation of :func:`emg3d.solver.solve` for all other
    input and output parameters.

    """
    sfield = fields.get_source_field(model.grid, source, frequency)
    return solve(model, sfield, **kwargs)


# SOLVERS
def multigrid(model, sfield, efield, var, **kwargs):
    """Multigrid solver for three-dimensional electromagnetic diffusion.

    Multigrid solver as presented in [Muld06]_, including semicoarsening and
    line relaxation as presented in [Muld07]_.

    - The electric field is stored in-place in ``efield``.
    - The number of multigrid cycles is stored in ``var.it``.
    - The current error (l2-norm) is stored in ``var.l2``.
    - The reference error (l2-norm of ``sfield``) is stored in ``var.l2_refe``.

    This function is the "heart" of the multigrid method, cycling through
    different grids, restricting and prolonging accordingly the grids, models,
    and fields.

    This function is called by :func:`emg3d.solver.solve`.


    Parameters
    ----------
    model : VolumeModel
        Input model; a :class:`emg3d.models.Model` instance.

    sfield : Field
        The source field; a :class:`emg3d.fields.Field` instance.

    efield : Field
        The electric field; a :class:`emg3d.fields.Field` instance.

    var : MGParameters
        A multigrid parameter instance used within
        :func:`emg3d.solver.multigrid`.

    level, new_cycmax : int, default: 0
        Parameters internally used for recursion (do not use):

        - ``level``: current coarsening level;
        - ``new_cycmax``: new maximum of multigrid cycles, takes care of
          V/W/F-cycling.

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

    # Compute current error (l2-norms).
    l2_last = residual(model, sfield, efield, True)

    # Initiate the error-array to check for stagnation.
    l2_stag = np.ones(var.maxcycle)*l2_last

    # Keep track on the levels during the first cycle, for QC.
    if var.first_cycle and var.verb > 3:
        var.level_all.append(level)

    # Print initial call info.
    if level == 0:
        var.cprint("     it cycmax               error", 4)
        var.cprint("      level [  dimension  ]            info\n", 4)
        if var.verb > 4:  # Cond. as it causes extra comp. time.
            _print_gs_info(var, it, level, cycmax, model.grid, l2_last,
                           "initial error")

    # Initial smoothing (nu_init).
    if level == 0 and var.nu_init > 0:
        # Smooth and re-compute error.
        smoothing(model, sfield, efield, var.nu_init, var.lr_dir)

        # Print initial smoothing info.
        if var.verb > 4:  # Cond. as it causes extra comp. time.
            norm = residual(model, sfield, efield, True)
            _print_gs_info(var, it, level, cycmax, model.grid, norm,
                           "initial smoothing")

    # Start the actual (recursive) multigrid cycle.
    while level == 0 or (level > 0 and it < cycmax):

        # Store errors for comparisons (previous and previous of same cycle).
        l2_prev = l2_last
        l2_stag[(it-1) % var.maxcycle] = l2_last

        # (A) Coarsest grid, solve system.
        if level == var.clevel[var.sc_dir]:
            # Note that the coarsest grid depends on semicoarsening (sc_dir).
            # If semicoarsening is carried out along the biggest dimension it
            # reduces the number of coarsening levels.

            # Gauss-Seidel on the coarsest grid.
            smoothing(model, sfield, efield, var.nu_coarse, var.lr_dir)

            # Print coarsest grid smoothing info.
            if var.verb > 4:  # Cond. as it causes extra comp. time.
                norm = residual(model, sfield, efield, True)
                _print_gs_info(var, it, level, cycmax, model.grid, norm,
                               "coarsest level")

        # (B) Not yet on coarsest grid.
        else:

            # (B.1) Pre-smoothing (nu_pre).
            if var.nu_pre > 0:
                smoothing(model, sfield, efield, var.nu_pre, var.lr_dir)

                # Print pre-smoothing info.
                if var.verb > 4:  # Cond. as it causes extra comp. time.
                    norm = residual(model, sfield, efield, True)
                    _print_gs_info(var, it, level, cycmax, model.grid, norm,
                                   "pre-smoothing")

            # Get sc_dir for this grid.
            sc_dir = _current_sc_dir(var.sc_dir, model.grid)

            # (B.2) Restrict grid, model, and fields from fine to coarse grid.
            res = residual(model, sfield, efield)  # Get residual.
            cmodel, csfield, cefield = restriction(model, sfield, res, sc_dir)

            # (B.3) Recursive call for coarse-grid correction.
            multigrid(cmodel, csfield, cefield, var, level=level+1,
                      new_cycmax=cycmax-cyc)

            # (B.4) Add coarse field residual to fine grid field.
            prolongation(efield, cefield, sc_dir)

            # Append current prolongation level for QC.
            if var.first_cycle and var.verb > 3:
                var.level_all.append(level)

            # (B.5) Post-smoothing (nu_post).
            if var.nu_post > 0:
                smoothing(model, sfield, efield, var.nu_post, var.lr_dir)

                # Print post-smoothing info.
                if var.verb > 4:  # Cond. as it causes extra comp. time.
                    norm = residual(model, sfield, efield, True)
                    _print_gs_info(var, it, level, cycmax, model.grid, norm,
                                   "post-smoothing")

        # Update iterator counts.
        it += 1         # Local iterator.
        if level == 0:  # Global iterator (works also when preconditioner.)
            var.it += 1

        # End loop depending if we are on the original grid or not.
        if level > 0:  # Update cyc if on a coarse grid.
            cyc += 1

        else:          # Original grid reached, check termination criteria.

            # Get current error (l2-norm).
            l2_last = residual(model, sfield, efield, True)

            # Print end-of-cycle info.
            _print_cycle_info(var, l2_last, l2_prev)

            # Adjust semicoarsening and line relaxation if they cycle.
            if var.sc_cycle:
                var.sc_dir = next(var.sc_cycle)
            if var.lr_cycle:
                var.lr_dir = next(var.lr_cycle)

            # Check if any termination criteria is fulfilled.
            if _terminate(var, l2_last, l2_stag[(it-1) % var.maxcycle], it):
                break

    # Store final error (l2-norm).
    var.l2 = l2_last


def krylov(model, sfield, efield, var):
    """Krylov subspace solver for three-dimensional electromagnetic diffusion.

    Using a Krylov subspace iterative solver (defined in ``var.sslsolver``)
    implemented in SciPy with or without multigrid as a pre-conditioner
    ([Muld06]_).

    - The electric field is stored in-place in ``efield``.
    - The current error (l2-norm) is stored in ``var.l2``.
    - The reference error (l2-norm of ``sfield``) is stored in ``var.l2_refe``.

    This function is called by :func:`emg3d.solver.solve`.


    Parameters
    ----------
    model : VolumeModel
        Input model; a :class:`emg3d.models.Model` instance.

    sfield : Field
        The source field; a :class:`emg3d.fields.Field` instance.

    efield : Field
        The electric field; a :class:`emg3d.fields.Field` instance.

    var : MGParameters
        A multigrid parameter instance used within
        :func:`emg3d.solver.multigrid`.

    """
    # Get frequency
    frequency = sfield._frequency

    # Define matrix operation A x as LinearOperator.
    def amatvec(efield):
        """Compute A x for solver; residual is b-Ax = source - amatvec."""

        # Cast current efield to Field instance.
        efield = fields.Field(model.grid, efield)

        # Compute A x.
        rfield = fields.Field(model.grid, dtype=efield.field.dtype,
                              frequency=frequency)
        core.amat_x(
                rfield.fx, rfield.fy, rfield.fz,
                efield.fx, efield.fy, efield.fz, model.eta_x, model.eta_y,
                model.eta_z, model.zeta,
                model.grid.h[0], model.grid.h[1], model.grid.h[2])

        # Return Field instance.
        return -rfield.field

    # Initiate LinearOperator A x.
    A = ssl.LinearOperator(
            shape=(sfield.field.size, sfield.field.size),
            dtype=sfield.field.dtype, matvec=amatvec)

    # Define multigrid pre-conditioner as LinearOperator, if `var.cycle`.
    def mg_matvec(sfield):
        """Use multigrid as pre-conditioner."""

        # Cast current fields to Field instances.
        sfield = fields.Field(model.grid, sfield, frequency=frequency)
        efield = fields.Field(model.grid, dtype=sfield.field.dtype,
                              frequency=frequency)

        # Solve for these fields.
        multigrid(model, sfield, efield, var)

        return efield.field

    # Initiate LinearOperator M.
    M = None
    if var.cycle:
        M = ssl.LinearOperator(
                shape=(sfield.field.size, sfield.field.size),
                dtype=sfield.field.dtype, matvec=mg_matvec)

    # Define callback to keep track of sslsolver-iterations.
    def callback(x):
        """Solver iteration count and error (l2-norm)."""

        # Update iteration count.
        var.ssl_it += 1

        # Add current runtime and error to var.
        var.runtime_at_cycle = np.r_[var.runtime_at_cycle, var.time.elapsed]
        var.l2 = residual(model, sfield, fields.Field(model.grid, x), True)
        var.error_at_cycle = np.r_[var.error_at_cycle, var.l2]

        # Print error (only if verbose).
        if var.verb > 3:

            log = f"   [{var.time.now}]   {var.l2/var.l2_refe:.3e} "
            log += f" after {var.ssl_it:3} {var.sslsolver}-cycles"

            # For those solvers who run an iteration before the first
            # preconditioner run ['gcrotmk'].
            if var.ssl_it == 1 and var.it == 0 and var.cycle is not None:
                log += "\n"

            var.cprint(log, 3)

        elif var.verb in [2, 3]:

            _print_one_liner(var, var.l2)

    # Solve the system with sslsolver.
    # The ssl solvers do not abort if the norm diverges or is not finite. We
    # therefore throw an exception in `_terminate`, and catch it here.
    try:
        efield.field, i = getattr(ssl, var.sslsolver)(
                A=A, b=sfield.field, x0=efield.field, tol=var.tol,
                maxiter=var.ssl_maxit, atol=1e-30, M=M, callback=callback)
    except _ConvergenceError:
        i = -1  # Mark it as error; returned field is all zero.
        var.exit_message += " (returned field is zero)"

    # Convergence-checks for sslsolver.
    if var.verb == 3:
        pre = 50*" " + "\r"
    else:
        pre = "\n"
    pre += "   > "
    if i < 0:
        if var.exit_message == '':
            var.exit_message = f"Error in {var.sslsolver} ({i})"
        pre = "\n* ERROR   :: "
    elif i > 0:
        var.exit_message = "MAX. ITERATION REACHED, NOT CONVERGED"
    else:
        var.exit_message = "CONVERGED"
    var.cprint(pre+var.exit_message, 2)


# MULTIGRID SUB-ROUTINES
def smoothing(model, sfield, efield, nu, lr_dir):
    """Reducing high-frequency error by smoothing.

    Solves the linear equation system :math:`A x = b` iteratively using the
    Gauss-Seidel method. This acts as smoother or, on the coarsest grid, as a
    direct solver.

    This is a simple wrapper for the jitted computation in
    :func:`emg3d.core.gauss_seidel`, :func:`emg3d.core.gauss_seidel_x`,
    :func:`emg3d.core.gauss_seidel_y`, and :func:`emg3d.core.gauss_seidel_z`
    (consult these functions for more details and corresponding theory).

    The electric fields are updated in-place.

    This function is called by :func:`emg3d.solver.multigrid`.


    Parameters
    ----------
    model : VolumeModel
        Input model; a :class:`emg3d.models.Model` instance.

    sfield : Field
        Input source field; a :class:`emg3d.fields.Field` instance.

    efield : Field
        Input electric field; a :class:`emg3d.fields.Field` instance.

    nu : int
        Number of Gauss-Seidel steps; odd numbers are forward, even numbers are
        reversed. E.g., ``nu=2`` is one symmetric Gauss-Seidel iteration, with
        a forward and a backward step.

    lr_dir : int
        Direction of line relaxation.

    """

    # Collect Gauss-Seidel input (same for all routines)
    inp = (sfield.fx, sfield.fy, sfield.fz,
           model.eta_x, model.eta_y, model.eta_z, model.zeta,
           model.grid.h[0], model.grid.h[1], model.grid.h[2],
           nu)

    # Avoid line relaxation in a direction where there are only two cells.
    c_lr_dir = _current_lr_dir(lr_dir, model.grid)

    # Compute and store fields (in-place)
    if c_lr_dir == 0:             # Standard MG
        core.gauss_seidel(efield.fx, efield.fy, efield.fz, *inp)

    if c_lr_dir in [1, 5, 6, 7]:  # Line relaxation in x-direction
        core.gauss_seidel_x(efield.fx, efield.fy, efield.fz, *inp)

    if c_lr_dir in [2, 4, 6, 7]:  # Line relaxation in y-direction
        core.gauss_seidel_y(efield.fx, efield.fy, efield.fz, *inp)

    if c_lr_dir in [3, 4, 5, 7]:  # Line relaxation in z-direction
        core.gauss_seidel_z(efield.fx, efield.fy, efield.fz, *inp)


def restriction(model, sfield, residual, sc_dir):
    """Downsampling of grid, model, and fields to a coarser grid.

    The restriction of the residual is used as source term for the coarse grid.

    Corresponds to Equations 8 and 9 and surrounding text in [Muld06]_. In the
    case of the restriction of the residual, this function is a wrapper for the
    jitted functions :func:`emg3d.core.restrict_weights` and
    :func:`emg3d.core.restrict` (consult these functions for more details and
    corresponding theory).

    This function is called by :func:`emg3d.solver.multigrid`.


    Parameters
    ----------
    model : VolumeModel
        Input model; a :class:`emg3d.models.Model` instance.

    sfield : Field
        Input source field; a :class:`emg3d.fields.Field` instance.

    sc_dir : int
        Direction of semicoarsening.


    Returns
    -------
    cmodel : VolumeModel
        Coarse model.

    csfield : Field
        Coarse source field. Corresponds to restriction of fine-grid residual.

    cefield : Field
        Coarse electric field, complex zeroes.

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

    # Compute distances of coarse grid.
    ch = [np.diff(model.grid.nodes_x[::rx]),
          np.diff(model.grid.nodes_y[::ry]),
          np.diff(model.grid.nodes_z[::rz])]

    # Create new `TensorMesh` instance for coarse grid
    cgrid = meshes.BaseMesh(ch, model.grid.origin)

    # 2. RESTRICT MODEL

    class VolumeModel:
        """Dummy class to create coarse-grid model."""
        def __init__(self, case, grid):
            """Initialize with case."""
            self.case = case
            self.grid = grid

    cmodel = VolumeModel(model.case, cgrid)
    cmodel.eta_x = _restrict_model_parameters(model.eta_x, sc_dir)
    if model.case in ['HTI', 'triaxial']:
        cmodel.eta_y = _restrict_model_parameters(model.eta_y, sc_dir)
    else:
        cmodel.eta_y = cmodel.eta_x
    if model.case in ['VTI', 'triaxial']:
        cmodel.eta_z = _restrict_model_parameters(model.eta_z, sc_dir)
    else:
        cmodel.eta_z = cmodel.eta_x
    cmodel.zeta = _restrict_model_parameters(model.zeta, sc_dir)

    # 3. RESTRICT FIELDS

    # Get the weights (Equation 9 of [Muld06]_).
    wx, wy, wz = _get_restriction_weights(model.grid, cmodel.grid, sc_dir)

    # Compute the source terms (Equation 8 in [Muld06]_).
    # Initiate zero field.
    csfield = fields.Field(cgrid, dtype=sfield.field.dtype,
                           frequency=sfield._frequency)
    core.restrict(csfield.fx, csfield.fy, csfield.fz, residual.fx,
                  residual.fy, residual.fz, wx, wy, wz, sc_dir)

    # Initiate empty e-field.
    cefield = fields.Field(cgrid, dtype=sfield.field.dtype,
                           frequency=sfield._frequency)

    return cmodel, csfield, cefield


def prolongation(efield, cefield, sc_dir):
    """Interpolating the electric field from coarse grid to fine grid.

    The prolongation from a coarser to a finer grid is the inverse process of
    the restriction (:func:`emg3d.solver.restriction`) from a finer to a
    coarser grid. The interpolated values of the coarse grid electric field are
    added to the fine grid electric field, in-place. Piecewise constant
    interpolation is used in the direction of the field, and bilinear
    interpolation in the other two directions. See Equation 10 in [Muld06]_ and
    surrounding text. Perfect Electric Conductor (PEC) boundary condition is
    enforced in this step.

    This function is called by :func:`emg3d.solver.multigrid`.


    Parameters
    ----------
    efield, cefield : Field
        Fine and coarse grid electric fields, :class:`emg3d.fields.Field`
        instances.

    sc_dir : int
        Direction of semicoarsening.

    """
    cgrid, grid = cefield.grid, efield.grid

    # Interpolate ex in y-z-slices.
    fn = RegularGridProlongator(
            cgrid.nodes_y, cgrid.nodes_z, grid.nodes_y, grid.nodes_z)
    for ixc in range(cgrid.shape_cells[0]):
        # Bilinear interpolation in the y-z plane
        hh = fn(cefield.fx[ixc, :, :]).reshape(
                (grid.shape_nodes[1], grid.shape_nodes[2]), order='F')

        # Piecewise constant interpolation in x-direction, ensuring PEC.
        if sc_dir not in [1, 5, 6]:
            efield.fx[2*ixc, 1:-1, 1:-1] += hh[1:-1, 1:-1]
            efield.fx[2*ixc+1, 1:-1, 1:-1] += hh[1:-1, 1:-1]
        else:
            efield.fx[ixc, 1:-1, 1:-1] += hh[1:-1, 1:-1]

    # Interpolate ey in x-z-slices.
    fn = RegularGridProlongator(
            cgrid.nodes_x, cgrid.nodes_z, grid.nodes_x, grid.nodes_z)
    for iyc in range(cgrid.shape_cells[1]):

        # Bilinear interpolation in the x-z plane
        hh = fn(cefield.fy[:, iyc, :]).reshape(
                (grid.shape_nodes[0], grid.shape_nodes[2]), order='F')

        # Piecewise constant interpolation in y-direction, ensuring PEC.
        if sc_dir not in [2, 4, 6]:
            efield.fy[1:-1, 2*iyc, 1:-1] += hh[1:-1, 1:-1]
            efield.fy[1:-1, 2*iyc+1, 1:-1] += hh[1:-1, 1:-1]
        else:
            efield.fy[1:-1, iyc, 1:-1] += hh[1:-1, 1:-1]

    # Interpolate ez in x-y-slices.
    fn = RegularGridProlongator(
            cgrid.nodes_x, cgrid.nodes_y, grid.nodes_x, grid.nodes_y)
    for izc in range(cgrid.shape_cells[2]):

        # Bilinear interpolation in the x-y plane
        hh = fn(cefield.fz[:, :, izc]).reshape(
                (grid.shape_nodes[0], grid.shape_nodes[1]), order='F')

        # Piecewise constant interpolation in z-direction, ensuring PEC.
        if sc_dir not in [3, 4, 5]:
            efield.fz[1:-1, 1:-1, 2*izc] += hh[1:-1, 1:-1]
            efield.fz[1:-1, 1:-1, 2*izc+1] += hh[1:-1, 1:-1]
        else:
            efield.fz[1:-1, 1:-1, izc] += hh[1:-1, 1:-1]


def residual(model, sfield, efield, norm=False):
    r"""Computing the residual.

    Returns the complete residual as given in [Muld06]_, page 636, middle of
    the right column. This is a simple wrapper for the jitted computation in
    :func:`emg3d.core.amat_x` (consult that function for more details and
    corresponding theory).

    This function is called by :func:`emg3d.solver.multigrid`.


    Parameters
    ----------
    model : VolumeModel
        Input model; a :class:`emg3d.models.Model` instance.

    sfield : Field
        Input source field; a :class:`emg3d.fields.Field` instance.

    efield : Field
        Input electric field; a :class:`emg3d.fields.Field` instance.

    norm : bool, default: False
        If True, the error (l2-norm) of the residual is returned, not the
        residual.


    Returns
    -------
    residual : Field, returned if norm=False
        The residual field; :class:`emg3d.fields.Field` instance.

    norm : float, returned if norm=True
        The error (l2-norm) of the residual.

    """
    # Get residual.
    rfield = sfield.copy()
    core.amat_x(rfield.fx, rfield.fy, rfield.fz, efield.fx, efield.fy,
                efield.fz, model.eta_x, model.eta_y, model.eta_z, model.zeta,
                model.grid.h[0], model.grid.h[1], model.grid.h[2])

    # Return error if norm.
    if norm:
        return sl.norm(rfield.field, check_finite=False)

    # Return residual if not norm.
    else:
        return rfield


# VARIABLE DATACLASS
@dataclass
class MGParameters:
    """Multigrid solver settings.

    This dataclass is used by the main solver routine to keep track of
    settings, errors, which level we currently are in, runtime, log, and so on.

    This class is instantiated by :func:`emg3d.solver.solve`. Consult that
    function for a description of the mandatory and optional input parameters
    and more information .

    """

    # (A) Mandatory parameters.
    # Verbosity.
    verb: int
    # SciPy Sparse Linalg solver flag.
    sslsolver: Union[str, bool]
    # Semicoarsening flag.
    semicoarsening: Union[int, bool]
    # Line relaxation flag.
    linerelaxation: Union[int, bool]
    # Shape of the input model.
    shape_cells: tuple

    # (B) Optional parameters with default values
    # Type of multigrid cycle.
    cycle: Union[str, None] = 'F'
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
    # Flag whether or not to return info.
    return_info: bool = False
    # Log if logging verbosity.
    log: int = 0

    def __post_init__(self):
        """Set and check some of the parameters."""

        # Levels and iterations.
        self.level_all = list()    # To keep track of the levels for QC-figure.
        self.first_cycle = True    # Flag if in first cycle for QC-figure.
        self.it = 0                # To store multigrid cycle count.
        self.ssl_it = 0            # To store solver iteration count.
        self.l2 = 1.0              # To store current error.
        self.l2_refe = 1.0         # To store reference error.
        self._max_level()          # Max coarsening levels.

        # Initiate logging strings, timer, errors, and return flag.
        self.exit_message = ''     # For convergence status.
        self.log_message = ''      # For returning info.
        self.time = utils.Timer()  # Timer.
        self.runtime_at_cycle = np.array([0.])  # Store runtime per cycle.
        self.error_at_cycle = np.array([0.])    # Store error per cycle.
        self.do_return = True     # Whether or not to return the efield.

        # Semicoarsening, line relaxation, and Solver/Cycle check.
        self._semicoarsening()
        self._linerelaxation()
        self._solver_and_cycle()

    def __repr__(self):
        """Print all relevant parameters."""

        outstring = (
            f"   MG-cycle       : {self.cycle!r:17}"
            f"   sslsolver : {self.sslsolver!r}\n"
            #
            f"   semicoarsening : {self._repr_sc_dir:17}"
            f"   tol       : {self.tol}\n"
            #
            f"   linerelaxation : {self._repr_lr_dir:17}"
            f"   maxit     : {self._repr_maxit}\n"
            #
            f"   nu_{{i,1,c,2}}   : {self.nu_init}, {self.nu_pre},"
            f" {self.nu_coarse}, {self.nu_post}       "
            f"   verb      : {self.verb}\n"
            #
            f"   Original grid  : {self.shape_cells[0]:3} x"
            f" {self.shape_cells[1]:3} x {self.shape_cells[2]:3}     =>"
            f" {self.shape_cells[0]*self.shape_cells[1]*self.shape_cells[2]:,}"
            f" cells\n"
            #
            f"   Coarsest grid  : {self._repr_clevel['shape_cells'][0]:3} x"
            f" {self._repr_clevel['shape_cells'][1]:3} x"
            f" {self._repr_clevel['shape_cells'][2]:3}  "
            f"   => {self._repr_clevel['n_cells']:,} cells\n"
            #
            f"   Coarsest level : {self._repr_clevel['clevel'][0]:3} ;"
            f" {self._repr_clevel['clevel'][1]:3}"
            f" ;{self._repr_clevel['clevel'][2]:4} "
            f"  {self._repr_clevel['message']}\n"
        )

        return outstring

    def cprint(self, info, verbosity, **kwargs):
        r"""Prints and logs ``info`` if ``self.verb`` > ``verbosity``.

        Parameters
        ----------
        info : str
            String to be printed.

        verbosity : int
            Verbosity of info.

        kwargs : optional
            Arguments passed to ``print()``, e.g., ``end='\r'``.

        """
        if self.verb > verbosity:
            if self.log != 0:
                self.log_message += str(info)+'\n'
            if self.log >= 0:
                print(info, **kwargs)

    def _max_level(self):
        r"""Sets dimension-dependent level variable ``clevel``.

        Requires at least two cells in each direction.

        """
        # Store input clevel for checks.
        inp_clevel = np.inf if self.clevel < 0 else self.clevel

        # Store maximum division-by-two level for each dimension.
        # After that, clevel = [nx, ny, nz], where nx, ny, and nz are the
        # number of times you can divide by two in this dimension.
        clevel = np.zeros(3, dtype=np.int64)
        for i in range(3):
            n = self.shape_cells[i]
            while n % 2 == 0 and n > 2:
                clevel[i] += 1
                n /= 2

        # Restrict to max coarsening level provided by user.
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

        # Store coarsest number of cells on coarsest grid and dimension for the
        # log-printing.
        sx = int(self.shape_cells[0]/2**clevel[0])
        sy = int(self.shape_cells[1]/2**clevel[1])
        sz = int(self.shape_cells[2]/2**clevel[2])
        self._repr_clevel = {'n_cells': sx*sy*sz, 'shape_cells': (sx, sy, sz),
                             'clevel': clevel}

        # Check some grid characteristics. Good values up to 1024 are:
        # - 2*2^{2, ..., 9}: 8, 16,  32,  64, 128, 256, 512, 1024,
        # - 3*2^{2, ..., 8}: 12, 24,  48,  96, 192, 384, 768,
        # - 5*2^{2, ..., 7}: 20, 40,  80, 160, 320, 640,
        # - 7*2^{2, ..., 7}: 28, 56, 112, 224, 448, 896,
        # and preference decreases from top to bottom row.

        # Check if number on coarsest grid is bigger than 7.
        # Ignore if clevel was provided and also reached (user wants it).
        check_inp = zip(clevel, [sx, sy, sz])
        max_low = any([cl < inp_clevel and sl > 7 for cl, sl in check_inp])

        # Check if it can be at least 3 (or inp_clevel) times coarsened.
        min_div = any(clevel < min(inp_clevel, 3))

        # Raise warning if necessary.
        if max_low or min_div:
            msg = "  :: Grid not optimal for MG solver ::"
            self._repr_clevel['message'] = msg
        else:
            self._repr_clevel['message'] = ""

        # Check at least two cells in each direction
        if np.any(np.array(self.shape_cells) < 2):
            raise ValueError(
                "Nr. of cells must be at least two in each direction "
                "Provided shape: ({self.shape_cells[0]}, "
                f"{self.shape_cells[1]}, {self.shape_cells[2]})."
            )

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
                raise ValueError(
                    "`semicoarsening` must be one of {False;True;0;1;2;3}. "
                    "Or a combination of {0;1;2;3} to cycle, e.g. 1213. "
                    f"Provided: {self.semicoarsening}."
                )

        # Get first (or only) direction.
        if self.sc_cycle:
            self.sc_dir = next(self.sc_cycle)
        else:
            self.sc_dir = sc_cycle[0]

        # Set semicoarsening to True/False; print statement
        self.semicoarsening = self.sc_dir != 0
        self._repr_sc_dir = f"{self.semicoarsening} {sc_cycle}"
        self.raw_sc_cycle = sc_cycle

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
                raise ValueError(
                    "`linerelaxation` must be one of "
                    "{False;True;0;1;2;3;4;5;6;7}. Or a combination of "
                    "{1;2;3;4;5;6;7} to cycle, e.g. 1213. "
                    f"Provided: {self.linerelaxation}."
                )

        # Get first (only) direction
        if self.lr_cycle:
            self.lr_dir = next(self.lr_cycle)
        else:
            self.lr_dir = lr_cycle[0]

        # Set linerelaxation to True/False; print statement
        self.linerelaxation = self.lr_dir != 0
        self._repr_lr_dir = f"{self.linerelaxation} {lr_cycle}"
        self.raw_lr_cycle = lr_cycle

    def _solver_and_cycle(self):
        """Set everything related to solver and MG-cycle."""

        # sslsolver.
        solvers = ['bicgstab', 'cgs', 'gcrotmk']
        if self.sslsolver is True:
            self.sslsolver = 'bicgstab'
        elif self.sslsolver is not False and self.sslsolver not in solvers:
            raise ValueError(
                f"`sslsolver` must be True, False, or one of {solvers}. "
                f"Provided: {self.sslsolver!r}."
            )

        if self.cycle not in ['F', 'V', 'W', None]:
            raise ValueError(
                "`cycle` must be one of {'F';'V';'W';None}. "
                f"Provided: {self.cycle}."
            )

        # Add maximum multigrid cycles depending on cycle
        if self.cycle in ['F', 'W']:
            self.cycmax = 2
        else:
            self.cycmax = 1

        # Ensure at least cycle or sslsolver is set
        if not self.sslsolver and not self.cycle:
            raise ValueError(
                "At least `cycle` or `sslsolver` is required. Provided"
                f"input: cycle={self.cycle}; sslsolver={self.sslsolver}."
            )

        # Store maxit in ssl_maxit and adjust maxit if sslsolver.
        self.ssl_maxit = 0             # Maximum iteration
        self._repr_maxit = f"{self.maxit}"  # For printing
        self.maxcycle = max(len(self.raw_sc_cycle), len(self.raw_lr_cycle))
        if self.sslsolver:
            self.ssl_maxit = self.maxit
            if self.cycle is not None:  # Only if multigrid is used
                self.maxit = self.maxcycle
                self._repr_maxit += f" ({self.maxit})"  # For printing


# INTERPOLATION DATACLASS
class RegularGridProlongator:
    """Prolongate field from coarse to fine grid.

    This is a heavily modified and adapted version of
    :class:`scipy.interpolate.RegularGridInterpolator`.

    The main difference (besides the different signature and different
    pre-sets) is that this version allows to initiate an instance with the
    coarse and fine grids. This initialize will compute the required weights,
    and it has therefore only to be done once.

    After this, interpolating values from the coarse to the fine grid can be
    carried out much faster.

    Simplifications in comparison to
    :class:`scipy.interpolate.RegularGridInterpolator`:

    - No sanity checks;
    - Only 2D data;
    - ``method='linear'``;
    - ``bounds_error=False``;
    - ``fill_value=None``.

    It results in a speed-up factor of about 2, independent of grid size, for
    this particular case.

    Parameters
    ----------
    cx, cy, x, y : ndarray
        The coordinates defining the coarse (cx, cy) and fine (x, y) grids.

    """

    def __init__(self, cx, cy, x, y):
        by, bx = np.broadcast_arrays(y, x[:, None])
        xy = np.r_[bx.ravel('F'), by.ravel('F')].reshape(-1, 2, order='F')
        self.size = xy.shape[0]
        self._set_edges_and_weights((cx, cy), xy)

    def __call__(self, values):
        """Return values of coarse grid on fine grid locations.

        Parameters
        ----------
        values : ndarray
            Values corresponding to fine-grid (x/y) coordinates.

        Returns
        -------
        result : ndarray
            Values corresponding to coarse-grid (cx/cy) coordinates.

        """
        # Initiate result.
        result = 0.

        # Find relevant values.
        for n, edge_indices in enumerate(self._get_edges_copy()):
            result += np.asarray(values[edge_indices]) * self.weight[n, :]

        return result

    def _set_edges_and_weights(self, cxy, xy):
        """Compute weights to go from coarse to fine grid coordinates."""

        # Find relevant edges between which xy are situated.
        indices = []

        # Compute distance to lower edge in unity units.
        norm_distances = []

        # Iterate through dimensions.
        for x, grid in zip(xy.T, cxy):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))

        # Find relevant values; each i and i+1 represents a edge.
        self.edges = itertools.product(*[[i, i + 1] for i in indices])

        # Compute weights.
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


# MULTIGRID HELPER ROUTINES (private, undocumented)
def _current_sc_dir(sc_dir, grid):
    """Return current direction(s) for semicoarsening.

    Checks which directions can actually still be halved and adjusts ``sc_dir``
    if necessary for this particular grid.


    Parameters
    ----------
    grid : TensorMesh
        Current grid; a :class:`emg3d.meshes.TensorMesh` instance.

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
    xsc_dir = (grid.shape_cells[0] % 2 != 0 or grid.shape_cells[0] < 3
               or sc_dir == 1)
    ysc_dir = (grid.shape_cells[1] % 2 != 0 or grid.shape_cells[1] < 3
               or sc_dir == 2)
    zsc_dir = (grid.shape_cells[2] % 2 != 0 or grid.shape_cells[2] < 3
               or sc_dir == 3)

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

    Checks which directions can actually still apply line relaxation and
    adjusts ``lr_dir`` if necessary for this particular grid.


    Parameters
    ----------
    grid : TensorMesh
        Current grid; a :class:`emg3d.meshes.TensorMesh` instance.

    lr_dir : int
        Direction of line relaxation.


    Returns
    -------
    c_lr_dir : int
        Current direction of line relaxation.

    """
    c_lr_dir = np.copy(lr_dir)

    if grid.shape_cells[0] == 2:  # Check x-direction.
        if c_lr_dir == 1:
            c_lr_dir = 0
        elif c_lr_dir == 5:
            c_lr_dir = 3
        elif c_lr_dir == 6:
            c_lr_dir = 2
        elif c_lr_dir == 7:
            c_lr_dir = 4

    if grid.shape_cells[1] == 2:  # Check y-direction.
        if c_lr_dir == 2:
            c_lr_dir = 0
        elif c_lr_dir == 4:
            c_lr_dir = 3
        elif c_lr_dir == 6:
            c_lr_dir = 1
        elif c_lr_dir == 7:
            c_lr_dir = 5

    if grid.shape_cells[2] == 2:  # Check z-direction.
        if c_lr_dir == 3:
            c_lr_dir = 0
        elif c_lr_dir == 4:
            c_lr_dir = 2
        elif c_lr_dir == 5:
            c_lr_dir = 1
        elif c_lr_dir == 7:
            c_lr_dir = 6

    return c_lr_dir


def _terminate(var, l2_last, l2_stag, it):
    """Return multigrid termination flag.

    Checks all termination criteria and returns True if at least one is
    fulfilled.


    Parameters
    ----------
    var : MGParameters
        A multigrid parameter instance used within
        :func:`emg3d.solver.multigrid`.

    l2_last, l2_stag : float
        Last error and error for stagnation comparison (l2-norms).

    it : int
        Iteration number.


    Returns
    -------
    finished : bool
        Boolean indicating if multigrid is finished.

    """

    finished = False
    sslabort = False

    # Converged if we reached the tolerance.
    if l2_last < var.tol*var.l2_refe:
        var.exit_message = "CONVERGED"
        finished = True

    # Diverged if it is 10x larger than last or not a number.
    elif l2_last > 10*var.l2_refe or not np.isfinite(l2_last):
        var.exit_message = "DIVERGED"
        finished = True
        sslabort = True  # Force abort if sslsolver.

    # Stagnated if it is >= the stagnation value.
    elif it > 2 and l2_last >= l2_stag:
        var.exit_message = "STAGNATED"
        finished = True
        sslabort = True  # Force abort if sslsolver.
        # Note: SSL will not fall into this, as it compares to the last value
        #       of the same cycle type. However, if used as preconditioner each
        #       cycle-type is only run once, before returning to the SSL.

    # Maximum iterations reached.
    elif it == var.maxit:
        if not var.sslsolver:
            var.exit_message = "MAX. ITERATION REACHED, NOT CONVERGED"
        finished = True

    # Force abort (ssl solver) or print info.
    if finished:

        # Force abortion of SSL solver.
        if var.sslsolver and sslabort:
            raise _ConvergenceError

        # Print info (if not preconditioner).
        elif not var.sslsolver:
            if var.verb == 3:
                add = 50*" " + "\r"
            elif var.verb < 5:
                add = "\n"
            else:
                add = ""
            var.cprint(add+"   > "+var.exit_message, 2)

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
    # Only sum the four cells in y-z-plane
    if sc_dir == 1:
        out = param[:, :-1:2, :-1:2] + param[:, 1::2, :-1:2]
        out += param[:, :-1:2, 1::2] + param[:, 1::2, 1::2]

    # Only sum the four cells in x-z-plane
    elif sc_dir == 2:
        out = param[:-1:2, :, :-1:2] + param[1::2, :, :-1:2]
        out += param[:-1:2, :, 1::2] + param[1::2, :, 1::2]

    # Only sum the four cells in x-y-plane
    elif sc_dir == 3:
        out = param[:-1:2, :-1:2, :] + param[1::2, :-1:2, :]
        out += param[:-1:2, 1::2, :] + param[1::2, 1::2, :]

    # Only sum the two cells in x-direction
    elif sc_dir == 4:
        out = param[:-1:2, :, :] + param[1::2, :, :]

    # Only sum the two cells y-direction
    elif sc_dir == 5:
        out = param[:, :-1:2, :] + param[:, 1::2, :]

    # Only sum the two cells z-direction
    elif sc_dir == 6:
        out = param[:, :, :-1:2] + param[:, :, 1::2]

    # Standard: Sum all 8 cells.
    else:
        out = param[:-1:2, :-1:2, :-1:2] + param[1::2, :-1:2, :-1:2]
        out += param[:-1:2, :-1:2, 1::2] + param[1::2, :-1:2, 1::2]
        out += param[:-1:2, 1::2, :-1:2] + param[1::2, 1::2, :-1:2]
        out += param[:-1:2, 1::2, 1::2] + param[1::2, 1::2, 1::2]

    return out


def _get_restriction_weights(grid, cgrid, sc_dir):
    """Return restriction weights.

    Return the weights as given in Equation 9 of [Muld06]_. It is a wrapper for
    the numba-jitted function :func:`emg3d.core.restrict_weights`. The
    corresponding weights are not actually used in the case of semicoarsening.
    We still have to provide arrays of the correct format though, otherwise
    numba will complain in the jitted functions.


    Parameters
    ----------
    grid, cgrid : TensorMesh
        Fine and coarse grids; :class:`emg3d.meshes.TensorMesh` instances.

    sc_dir : int
        Direction of semicoarsening.


    Returns
    -------
    wx, wy, wz : tuple
        Restriction weights in x, y, and z direction, consisting of
        (left, center, right) weights.

    """
    # x-directed weights.
    if sc_dir not in [1, 5, 6]:
        wx = core.restrict_weights(
                grid.nodes_x, grid.cell_centers_x, grid.h[0], cgrid.nodes_x,
                cgrid.cell_centers_x, cgrid.h[0])

    else:  # Dummy weights in case of semicoarsening.
        wxlr = np.zeros(grid.shape_nodes[0], dtype=np.float64)
        wx0 = np.ones(grid.shape_nodes[0], dtype=np.float64)
        wx = (wxlr, wx0, wxlr)

    # y-directed weights.
    if sc_dir not in [2, 4, 6]:
        wy = core.restrict_weights(
                grid.nodes_y, grid.cell_centers_y, grid.h[1], cgrid.nodes_y,
                cgrid.cell_centers_y, cgrid.h[1])

    else:  # Dummy weights in case of semicoarsening.
        wylr = np.zeros(grid.shape_nodes[1], dtype=np.float64)
        wy0 = np.ones(grid.shape_nodes[1], dtype=np.float64)
        wy = (wylr, wy0, wylr)

    # z-directed weights.
    if sc_dir not in [3, 4, 5]:
        wz = core.restrict_weights(
                grid.nodes_z, grid.cell_centers_z, grid.h[2], cgrid.nodes_z,
                cgrid.cell_centers_z, cgrid.h[2])

    else:  # Dummy weights in case of semicoarsening.
        wzlr = np.zeros(grid.shape_nodes[2], dtype=np.float64)
        wz0 = np.ones(grid.shape_nodes[2], dtype=np.float64)
        wz = (wzlr, wz0, wzlr)

    return wx, wy, wz


class _ConvergenceError(Exception):
    """Custom exception for convergence issues with SSL solvers."""


# VERBOSITY HELPER ROUTINES (private, undocumented)
def _print_cycle_info(var, l2_last, l2_prev):
    """Print cycle info to log.

    Parameters
    ----------
    var : MGParameters
        A multigrid parameter instance used within
        :func:`emg3d.solver.multigrid`.

    l2_last, l2_prev : float
        Last and previous errors (l2-norms).

    """
    # Add current runtime to var.
    var.runtime_at_cycle = np.r_[var.runtime_at_cycle, var.time.elapsed]
    var.error_at_cycle = np.r_[var.error_at_cycle, l2_last]

    # Start info string, return if not enough verbose.
    if var.verb in [2, 3]:  # One-liner
        _print_one_liner(var, l2_last)

    if var.verb < 4:
        # Set first_cycle to False, to stop logging.
        return
    elif var.verb > 4:
        info = "\n"
    else:
        info = ""

    # Add multigrid-cycle visual QC on first cycle.
    if var.first_cycle:

        # Cast levels into array, get maximum.
        _lvl_all = np.array(var.level_all, dtype=np.int64)
        lvl_max = np.max(_lvl_all)

        # Get levels, multiply by difference to get +/-.
        lvl = (_lvl_all[1:] + _lvl_all[:-1])//2+1
        lvl *= _lvl_all[1:] - _lvl_all[:-1]

        # Create info string.
        out = ["       h_\n"]
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
        var.first_cycle = False

    # Add iteration log.
    info += f"   [{var.time.now}]   {l2_last/var.l2_refe:.3e}  "
    if var.sslsolver:  # For multigrid as preconditioner.
        info += f"after {19*' '} {var.it:3} {var.cycle}-cycles "

    else:              # For multigrid as solver.
        info += f"after {var.it:3} {var.cycle}-cycles   "
        info += f"[{l2_last:.3e}, {l2_last/l2_prev:.3f}]"
    info += f"   {var.lr_dir} {var.sc_dir}"

    if var.verb > 4:
        info += "\n"

    # Print the info.
    var.cprint(info, 3)


def _print_gs_info(var, it, level, cycmax, grid, norm, add):
    """Return info-string to log after each Gauss-Seidel smoothing step.

    Parameters
    ----------
    it : int
        Iteration number.

    level : int
        Current coarsening level.

    cycmax : int
        Maximum multigrid cycles.

    grid : TensorMesh
        Current grid; a :class:`emg3d.meshes.TensorMesh` instance.

    norm : float
        Current error (l2-norm).

    add : str
        Information to add at the end.

    """
    info = f"     {it:2} {level} {cycmax} [{grid.shape_cells[0]:3}, "
    info += f"{grid.shape_cells[1]:3}, "
    info += f"{grid.shape_cells[2]:3}]: {norm:.3e} "
    var.cprint(info + add, 4)


def _print_one_liner(var, l2_last, last=False):
    """Print continuously updated one-liner.

    Parameters
    ----------
    l2_last : float
        Current error.

    last : bool
        If True, adds `exit_message` and finishes line.

    """
    # Collect info.
    info = f":: emg3d :: {l2_last/var.l2_refe:.1e}; "  # Absolute error.
    if var.sslsolver:  # For multigrid as preconditioner.
        info += f"{var.ssl_it}({var.it}); "
    else:               # Stand-alone multigrid.
        info += f"{var.it}; "
    info += f"{var.time.runtime}"  # Runtime

    # Print depending on `exit`.
    if last:
        var.cprint(info+f"; {var.exit_message}", -100)
    else:
        var.cprint(info, -100, end='\r')
