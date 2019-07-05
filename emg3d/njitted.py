"""

:mod:`njitted` -- Functions jitted with ``njit``
================================================

The core functionalities, the most computationally demanding parts, of the
:mod:`emg3d.solver` as just-in-time (jit) compiled functions using ``numba``.

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


import numba as nb
import numpy as np

# Numba-settings
_numba_setting = {'nogil': True, 'fastmath': True, 'cache': True}


# LinearOperator to calculate A x
@nb.njit(**_numba_setting)
def amat_x(rx, ry, rz, ex, ey, ez, eta_x, eta_y, eta_z, mu_r, hx, hy, hz):
    r"""Residual without or with source term.

    Calculate the residual as given in [Muld06]_ in middle of the right column
    on page 636, but without the source term:

    .. math::

        \mathbf{r} = V \left( \mathrm{i}\omega\mu_0 \tilde{\sigma} \mathbf{E}
                     - \nabla \times \mu_\mathrm{r}^{-1} \nabla \times
                       \mathbf{E} \right) .

    The calculation is carried out in a matrix-free manner; on said page 636
    (or in the :doc:`theory`) are the various steps laid out to discretise the
    different parts, for instance involved curls. This can also be understood
    as the left-hand-side of :math:`A x = b`, as given in Equation 2 in
    [Muld06]_ (here without the cell volumes V),

    .. math::

        \mathrm{i}\omega\mu_0 \tilde{\sigma} \mathrm{E}
        - \nabla \times \mu_r^{-1} \nabla \times \mathrm{E}
        = - \mathrm{i} \omega \mu_0 \mathrm{J_s} .

    It can therefore be used as ``matvec`` to create a ``LinearOperator``,
    which can be passed to a solver.

    It is assumed that ex, ey, and ez have PEC boundaries; otherwise the output
    will not have PEC boundaries.

    The residuals are subtracted in-place from ``rx``, ``ry``, and ``rz``.
    That means that if ``rx``, ``ry``, and ``rz`` contain the source field,
    they will contain the total residual afterwards; if they are empty fields,
    they will contain the negative partial residual afterwards.


    Parameters
    ----------
    rx, ry, rz : ndarray
        Source field or pre-allocated zero residual field in x-, y-, and
        z-directions.

    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    eta_x, eta_y, eta_z, mu_r : ndarray
        Model parameters (multiplied by volumes) as obtained from
        :func:`emg3d.utils.Model`.

    hx, hy, hz : ndarray
        Cell widths in x-, y-, and z-directions.

    """

    # Get dimensions
    nx = len(hx)
    ny = len(hy)
    nz = len(hz)

    # Loop over dimensions; x-fastest, then y, z.
    # NOTE about `i?m = max(0, i?-1)`:
    # In the cases when -1 is set to 0, these indices are only used in
    # parameters which are not actually used in these cases, see the note
    # towards the end. Resetting -1 to 0 is simply to avoid index errors.
    for iz in range(nz):
        izm = max(0, iz-1)
        izp = iz+1
        for iy in range(ny):
            iym = max(0, iy-1)
            iyp = iy+1
            for ix in range(nx):
                ixm = max(0, ix-1)
                ixp = ix+1

                # 1. Curl  [Muld06]_ Equation 7:
                # v = nabla x E.
                v1pp = ((ez[ix, iyp, iz] - ez[ix, iy, iz])/hy[iy] -
                        (ey[ix, iy, izp] - ey[ix, iy, iz])/hz[iz])
                v1mp = ((ez[ix, iy, iz] - ez[ix, iym, iz])/hy[iym] -
                        (ey[ix, iym, izp] - ey[ix, iym, iz])/hz[iz])
                v1pm = ((ez[ix, iyp, izm] - ez[ix, iy, izm])/hy[iy] -
                        (ey[ix, iy, iz] - ey[ix, iy, izm])/hz[izm])

                v2pp = ((ex[ix, iy, izp] - ex[ix, iy, iz])/hz[iz] -
                        (ez[ixp, iy, iz] - ez[ix, iy, iz])/hx[ix])
                v2mp = ((ex[ixm, iy, izp] - ex[ixm, iy, iz])/hz[iz] -
                        (ez[ix, iy, iz] - ez[ixm, iy, iz])/hx[ixm])
                v2pm = ((ex[ix, iy, iz] - ex[ix, iy, izm])/hz[izm] -
                        (ez[ixp, iy, izm] - ez[ix, iy, izm])/hx[ix])

                v3pp = ((ey[ixp, iy, iz] - ey[ix, iy, iz])/hx[ix] -
                        (ex[ix, iyp, iz] - ex[ix, iy, iz])/hy[iy])
                v3mp = ((ey[ix, iy, iz] - ey[ixm, iy, iz])/hx[ixm] -
                        (ex[ixm, iyp, iz] - ex[ixm, iy, iz])/hy[iy])
                v3pm = ((ey[ixp, iym, iz] - ey[ix, iym, iz])/hx[ix] -
                        (ex[ix, iy, iz] - ex[ix, iym, iz])/hy[iym])

                # 2. Multiply by average of mu_r [Muld06]_ p 636 bottom-left.
                # u = M v = V mu_r^-1 v = V mu_r^-1 nabla x E
                v1pp *= 0.5*(mu_r[ixm, iy, iz] + mu_r[ix, iy, iz])
                v1mp *= 0.5*(mu_r[ixm, iym, iz] + mu_r[ix, iym, iz])
                v1pm *= 0.5*(mu_r[ixm, iy, izm] + mu_r[ix, iy, izm])

                v2pp *= 0.5*(mu_r[ix, iym, iz] + mu_r[ix, iy, iz])
                v2mp *= 0.5*(mu_r[ixm, iym, iz] + mu_r[ixm, iy, iz])
                v2pm *= 0.5*(mu_r[ix, iym, izm] + mu_r[ix, iy, izm])

                v3pp *= 0.5*(mu_r[ix, iy, izm] + mu_r[ix, iy, iz])
                v3mp *= 0.5*(mu_r[ixm, iy, izm] + mu_r[ixm, iy, iz])
                v3pm *= 0.5*(mu_r[ix, iym, izm] + mu_r[ix, iym, iz])

                # 3. Another curl [Muld06]_ p. 636 bottom-right; completes:
                # nabla x M v = nabla x V mu_r^-1 nabla x E
                rrx = (v3pp/hy[iy]-v3pm/hy[iym]) - (v2pp/hz[iz]-v2pm/hz[izm])
                rry = (v1pp/hz[iz]-v1pm/hz[izm]) - (v3pp/hx[ix]-v3mp/hx[ixm])
                rrz = (v2pp/hx[ix]-v2mp/hx[ixm]) - (v1pp/hy[iy]-v1mp/hy[iym])

                # 4. Sigma-term, [Muld06]_ p. 636 top-left (average of # eta).
                # S = i omega mu_0 sigma~ V
                stx = 0.25*(eta_x[ix, iym, izm] + eta_x[ix, iym, iz] +
                            eta_x[ix, iy, izm] + eta_x[ix, iy, iz])
                sty = 0.25*(eta_y[ixm, iy, izm] + eta_y[ix, iy, izm] +
                            eta_y[ixm, iy, iz] + eta_y[ix, iy, iz])
                stz = 0.25*(eta_z[ixm, iym, iz] + eta_z[ix, iym, iz] +
                            eta_z[ixm, iy, iz] + eta_z[ix, iy, iz])

                # NOTE re zero boundary conditions for tangential E field:
                # At the moment these elements are calculated but now
                # discarded. This function could be adjusted to omit the
                # calculation of these. But one would have to test if it makes
                # it actually faster.
                if iy == 0 or iz == 0:  # assuming ex = 0
                    rrx = 0
                if ix == 0 or iz == 0:  # assuming ey = 0
                    rry = 0
                if ix == 0 or iy == 0:  # assuming ez = 0
                    rrz = 0

                # 5. [Muld06]_ p. 636 center-right; completes
                # -V (i omega mu_0 sigma~ E - nabla x mu_r^-1 nabla x E)
                # Subtracting this from the source terms will yield the
                # residual.
                rx[ix, iy, iz] -= rrx - stx*ex[ix, iy, iz]
                ry[ix, iy, iz] -= rry - sty*ey[ix, iy, iz]
                rz[ix, iy, iz] -= rrz - stz*ez[ix, iy, iz]


# Gauss-Seidel method
@nb.njit(**_numba_setting)
def gauss_seidel(ex, ey, ez, sx, sy, sz, eta_x, eta_y, eta_z, mu_r, hx, hy, hz,
                 nu):
    r"""Gauss-Seidel method.

    Solves the linear equation system :math:`A x = b` iteratively using the
    following method:

    .. math::

        \mathbf{x}^{(k+1)} =
        L_*^{-1} \left(\mathbf{b} - U \mathbf{x}^{(k)} \right) \ ,

    where :math:`L_*` is the lower triangular component, and :math:`U` the
    strictly upper triangular component, :math:`A = L_* + U`:

    .. math::

        L_* = \left[ \begin{array} {cccc}
              a_{11} &   0    & \cdots &    0   \\
              a_{21} & a_{22} & \cdots &    0   \\
              \vdots & \vdots & \ddots & \vdots \\
              a_{n1} & a_{n2} & \cdots & a_{nn}
              \end{array} \right] \ , \quad
        U = \left[ \begin{array} {cccc}
                 0   & a_{12} & \cdots & a_{1n} \\
                 0   &   0    & \cdots & a_{2n} \\
              \vdots & \vdots & \ddots & \vdots \\
                 0   &   0    & \cdots &   0
            \end{array} \right] \ .

    On the coarsest grid it acts as direct solver, whereas on the fine grid it
    acts as a smoother with only few iterations, defined by :math:`\nu`
    (``nu``). Odd numbers of ``nu`` use forward ordering, even numbers use
    backwards ordering. ``nu=2`` is therefore one symmetric Gauss-Seidel
    iteration, one forward ordered iteration followed by one backward ordered
    iteration.

    From [Muld06]_: The method proposed by [ArFW00]_ is chosen as a smoother.
    It selects one node of the grid and simultaneously solves for the six
    degrees of freedom on the six edges attached to the node. If node
    :math:`(x_k, y_l, z_m)` is selected, the six equations,
    :math:`r_{x;k\pm1/2,l,m} = 0`, :math:`r_{y;k,l\pm1/2,m} = 0` and
    :math:`r_{z;k,l,m\pm1/2} = 0`, are solved for :math:`e_{x;k\pm1/2,l,m}`,
    :math:`e_{y;k,l\pm1/2,m}` and :math:`e_{z;k,l,m\pm1/2}`. Here, this
    smoother is applied in a symmetric Gauss-Seidel fashion, following the
    lexicographical ordering of the nodes :math:`(x_k, y_l, z_m)`, with fastest
    index :math:`k=1, \dots, N_x-1`, intermediate index :math:`l=1, \dots,
    N_y-1`, and slowest index :math:`m=1, \ldots, N_z-1`.

    To actually solve the system of six equations a non-standard Cholesky
    factorisation is used, :func:`solve`.

    Tangential components at the boundaries are assumed to be zero (PEC
    boundaries).

    The result is stored in the provided electric fields ``ex``, ``ey``, and
    ``ez``.


    Parameters
    ----------
    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    sx, sy, sz :
        Source fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    eta_x, eta_y, eta_z, mu_r :
        Model parameters (multiplied by volumes) as obtained from
        :func:`emg3d.utils.Model`.

    hx, hy, hz : ndarray
        Cell widths in x-, y-, and z-directions.

    nu : int
        Number of Gauss-Seidel iterations.

    """

    # Get dimensions
    nCx = len(hx)
    nCy = len(hy)
    nCz = len(hz)

    # Get half of the inverse widths
    kx = 0.5/hx
    ky = 0.5/hy
    kz = 0.5/hz

    # Direction-switch for Gauss-Seidel
    iback = 0

    # Pre-allocating A for the six edges attached to one node; will be
    # overwritten at each iteration
    amat = np.zeros(36, dtype=np.complex128)

    # Smoothing steps
    for _ in range(nu):

        # Direction of Gauss-Seidel ordering; 0=forward, 1=backward
        iback = 1-iback

        # Loop over cells, keeping boundaries fixed; x-fastest, then y, z.
        for izh in range(1, nCz):

            # Back-forth-switch
            if iback:
                iz = nCz-izh
            else:
                iz = izh

            # Minus/plus indices
            izm = iz-1
            izp = iz+1

            for iyh in range(1, nCy):

                # Back-forth-switch
                if iback:
                    iy = nCy-iyh
                else:
                    iy = iyh

                # Minus/plus indices
                iym = iy-1
                iyp = iy+1

                for ixh in range(1, nCx):

                    # Back-forth-switch
                    if iback:
                        ix = nCx-ixh
                    else:
                        ix = ixh

                    # Minus/plus indices
                    ixm = ix-1
                    ixp = ix+1

                    # Averaging of 1/mu_r: mzyRxm etc.
                    mzyLxm = ky[iym]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzyRxm = ky[iy]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    myzLxm = kz[izm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myzRxm = kz[iz]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    mzyLxp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mzyRxp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    myzLxp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    myzRxp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mzxLym = kx[ixm]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzxRym = kx[ix]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mxzLym = kz[izm]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxzRym = kz[iz]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    mzxLyp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    mzxRyp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    mxzLyp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    mxzRyp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])
                    myxLzm = kx[ixm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myxRzm = kx[ix]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    mxyLzm = ky[iym]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxyRzm = ky[iy]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    myxLzp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    myxRzp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mxyLzp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    mxyRzp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])

                    # Diagonal elements
                    st0 = (eta_x[ixm, iy, iz] + eta_x[ixm, iy, izm] +
                           eta_x[ixm, iym, iz] + eta_x[ixm, iym, izm])
                    st1 = (eta_x[ix, iy, iz] + eta_x[ix, iy, izm] +
                           eta_x[ix, iym, iz] + eta_x[ix, iym, izm])
                    st2 = (eta_y[ix, iym, iz] + eta_y[ix, iym, izm] +
                           eta_y[ixm, iym, iz] + eta_y[ixm, iym, izm])
                    st3 = (eta_y[ix, iy, iz] + eta_y[ix, iy, izm] +
                           eta_y[ixm, iy, iz] + eta_y[ixm, iy, izm])
                    st4 = (eta_z[ix, iy, izm] + eta_z[ix, iym, izm] +
                           eta_z[ixm, iy, izm] + eta_z[ixm, iym, izm])
                    st5 = (eta_z[ix, iy, iz] + eta_z[ix, iym, iz] +
                           eta_z[ixm, iy, iz] + eta_z[ixm, iym, iz])

                    st = np.array([st0, st1, st2, st3, st4, st5])/4.

                    # Fill amat
                    amat[:] = 0+0j  # Reset

                    # Initial diagonal elements
                    for k in range(6):
                        amat[6*k] = -st[k]

                    # Complete diagonals
                    # A is symmetric and curl curl part is real-valued
                    amat[0] += mzyRxm/hy[iy] + mzyLxm/hy[iym]   # 0,0| 0
                    amat[0] += myzRxm/hz[iz] + myzLxm/hz[izm]
                    amat[6] += mzyRxp/hy[iy] + mzyLxp/hy[iym]   # 1,1| 6
                    amat[6] += myzRxp/hz[iz] + myzLxp/hz[izm]
                    amat[12] += mzxRym/hx[ix] + mzxLym/hx[ixm]  # 2,2|12
                    amat[12] += mxzRym/hz[iz] + mxzLym/hz[izm]
                    amat[18] += mzxRyp/hx[ix] + mzxLyp/hx[ixm]  # 3,3|18
                    amat[18] += mxzRyp/hz[iz] + mxzLyp/hz[izm]
                    amat[24] += myxRzm/hx[ix] + myxLzm/hx[ixm]  # 4,4|24
                    amat[24] += mxyRzm/hy[iy] + mxyLzm/hy[iym]
                    amat[30] += myxRzp/hx[ix] + myxLzp/hx[ixm]  # 5,5|30
                    amat[30] += mxyRzp/hy[iy] + mxyLzp/hy[iym]

                    # Off-diagonal elements
                    # Upper triangle not needed and not set.
                    # The elements
                    #   [1, 0] (1); [3, 2] (13); and [5, 4] (21)
                    # are all zero.
                    amat[2] = -mzyLxm/hx[ixm]   # 2,0| 2
                    amat[3] = mzyRxm/hx[ixm]    # 3,0| 3
                    amat[4] = -myzLxm/hx[ixm]   # 4,0| 4
                    amat[5] = myzRxm/hx[ixm]    # 5,0| 5
                    amat[7] = mzyLxp/hx[ix]     # 2,1| 7
                    amat[8] = -mzyRxp/hx[ix]    # 3,1| 8
                    amat[9] = myzLxp/hx[ix]     # 4,1| 9
                    amat[10] = -myzRxp/hx[ix]   # 5,1|10
                    amat[14] = -mxzLym/hy[iym]  # 4,2|14
                    amat[15] = mxzRym/hy[iym]   # 5,2|15
                    amat[19] = mxzLyp/hy[iy]    # 4,3|19
                    amat[20] = -mxzRyp/hy[iy]   # 5,3|20

                    # Fill residual (b - Ux^{(k)})
                    # Note: rhs is NOT the full residual at this point

                    # Get the 6 edges for ix, iy, and iz
                    rhs = np.array([sx[ixm, iy, iz], sx[ix, iy, iz],
                                    sy[ix, iym, iz], sy[ix, iy, iz],
                                    sz[ix, iy, izm], sz[ix, iy, iz]])

                    rhs[0] += mzyRxm*(ey[ixm, iy, iz]/hx[ixm] +
                                      ex[ixm, iyp, iz]/hy[iy])
                    rhs[0] += mzyLxm*(-ey[ixm, iym, iz]/hx[ixm] +
                                      ex[ixm, iym, iz]/hy[iym])
                    rhs[0] += myzRxm*(ez[ixm, iy, iz]/hx[ixm] +
                                      ex[ixm, iy, izp]/hz[iz])
                    rhs[0] += myzLxm*(-ez[ixm, iy, izm]/hx[ixm] +
                                      ex[ixm, iy, izm]/hz[izm])

                    rhs[1] += mzyRxp*(-ey[ixp, iy, iz]/hx[ix] +
                                      ex[ix, iyp, iz]/hy[iy])
                    rhs[1] += mzyLxp*(ey[ixp, iym, iz]/hx[ix] +
                                      ex[ix, iym, iz]/hy[iym])
                    rhs[1] += myzRxp*(-ez[ixp, iy, iz]/hx[ix] +
                                      ex[ix, iy, izp]/hz[iz])
                    rhs[1] += myzLxp*(ez[ixp, iy, izm]/hx[ix] +
                                      ex[ix, iy, izm]/hz[izm])

                    rhs[2] += mzxRym*(ey[ixp, iym, iz]/hx[ix] +
                                      ex[ix, iym, iz]/hy[iym])
                    rhs[2] += mzxLym*(ey[ixm, iym, iz]/hx[ixm] -
                                      ex[ixm, iym, iz]/hy[iym])
                    rhs[2] += mxzRym*(ez[ix, iym, iz]/hy[iym] +
                                      ey[ix, iym, izp]/hz[iz])
                    rhs[2] += mxzLym*(-ez[ix, iym, izm]/hy[iym] +
                                      ey[ix, iym, izm]/hz[izm])

                    rhs[3] += mzxRyp*(ey[ixp, iy, iz]/hx[ix] -
                                      ex[ix, iyp, iz]/hy[iy])
                    rhs[3] += mzxLyp*(ey[ixm, iy, iz]/hx[ixm] +
                                      ex[ixm, iyp, iz]/hy[iy])
                    rhs[3] += mxzRyp*(-ez[ix, iyp, iz]/hy[iy] +
                                      ey[ix, iy, izp]/hz[iz])
                    rhs[3] += mxzLyp*(ez[ix, iyp, izm]/hy[iy] +
                                      ey[ix, iy, izm]/hz[izm])

                    rhs[4] += myxRzm*(ez[ixp, iy, izm]/hx[ix] +
                                      ex[ix, iy, izm]/hz[izm])
                    rhs[4] += myxLzm*(ez[ixm, iy, izm]/hx[ixm] -
                                      ex[ixm, iy, izm]/hz[izm])
                    rhs[4] += mxyRzm*(ez[ix, iyp, izm]/hy[iy] +
                                      ey[ix, iy, izm]/hz[izm])
                    rhs[4] += mxyLzm*(ez[ix, iym, izm]/hy[iym] -
                                      ey[ix, iym, izm]/hz[izm])

                    rhs[5] += myxRzp*(ez[ixp, iy, iz]/hx[ix] -
                                      ex[ix, iy, izp]/hz[iz])
                    rhs[5] += myxLzp*(ez[ixm, iy, iz]/hx[ixm] +
                                      ex[ixm, iy, izp]/hz[iz])
                    rhs[5] += mxyRzp*(ez[ix, iyp, iz]/hy[iy] -
                                      ey[ix, iy, izp]/hz[iz])
                    rhs[5] += mxyLzp*(ez[ix, iym, iz]/hy[iym] +
                                      ey[ix, iym, izp]/hz[iz])

                    # Solve linear system A x = b.
                    solve(amat, rhs)

                    # Update efield (here we could apply damping weights).
                    ex[ixm, iy, iz] = rhs[0]
                    ex[ix, iy, iz] = rhs[1]
                    ey[ix, iym, iz] = rhs[2]
                    ey[ix, iy, iz] = rhs[3]
                    ez[ix, iy, izm] = rhs[4]
                    ez[ix, iy, iz] = rhs[5]


@nb.njit(**_numba_setting)
def gauss_seidel_x(ex, ey, ez, sx, sy, sz, eta_x, eta_y, eta_z, mu_r, hx, hy,
                   hz, nu):
    r"""Gauss-Seidel method with line relaxation in x-direction.

    This is the equivalent to :func:`gauss_seidel`, but with line relaxation in
    the x-direction. See :func:`gauss_seidel` for more details.

    The resulting system A x = b to solve consists of n unknowns (x-vector),
    and the corresponding matrix A is a banded matrix with the main diagonal
    and five upper and lower diagonals::

       .-0
       |X|\   0
       0-.-0       left:  middle:  right:
        \|X|\                      (not used)
         0-.-0      0-     .-      0
          \|X|\      \     |X      |\
           0-.-0
        0   \|X|
             0-.

       . 1*1, - 4*1, | 1*4, X 4*4, \ 4*4 upper or lower

    The matrix A is complex and symmetric (A = A^T), and therefore only the
    main diagonal and the lower five off-diagonals are required.

    - The right-hand-side b has length 5*nCx-4 (nCx even).
    - The matrix A has length of b and 1+2*5 diagonals; we use for it an array
      of length 6*len(b).

    The values are calculated in rows of 5 lines, with the indicated middle and
    left matrices as indicated in the above scheme. These blocks are filled
    into the main matrix A and vector b, and subsequently solved with a
    non-standard Cholesky factorisation, :func:`solve`.

    Tangential components at the boundaries are assumed to be 0 (PEC
    boundaries).

    The result is stored in the provided electric fields ``ex``, ``ey``, and
    ``ez``.


    Parameters
    ----------
    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    sx, sy, sz :
        Source fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    eta_x, eta_y, eta_z, mu_r :
        Model parameters (multiplied by volumes) as obtained from
        :func:`emg3d.utils.Model`.

    hx, hy, hz : ndarray
        Cell widths in x-, y-, and z-directions.

    nu : int
        Number of Gauss-Seidel iterations.

    """

    # Get dimensions
    nCx = len(hx)
    nCy = len(hy)
    nCz = len(hz)

    # Get half of the inverse widths
    kx = 0.5/hx
    ky = 0.5/hy
    kz = 0.5/hz

    # Direction-switch for Gauss-Seidel
    iback = 0

    # Pre-allocating middle and left for the 5x5-temporary middle and left
    # matrices; will be overwritten at each iteration
    middle = np.zeros(25, dtype=np.complex128)
    left = np.zeros(25)

    # Pre-allocating full RHS (bvec) and full matrix A (amat). Will be
    # overwritten after each complete x-loop.
    nr = 5*nCx-4  # Number of unknowns
    bvec = np.zeros(nr, dtype=np.complex128)
    amat = np.zeros(6*nr, dtype=np.complex128)

    # Smoothing steps
    for _ in range(nu):

        # Direction of Gauss-Seidel ordering; 0=forward, 1=backward
        iback = 1-iback

        # Loop over cells, keeping boundaries fixed; x-fastest, then y, z.
        for izh in range(1, nCz):

            # Back-forth-switch
            if iback:
                iz = nCz-izh
            else:
                iz = izh

            # Minus/plus indices
            izm = iz-1
            izp = iz+1

            for iyh in range(1, nCy):

                # Back-forth-switch
                if iback:
                    iy = nCy-iyh
                else:
                    iy = iyh

                # Minus/plus indices
                iym = iy-1
                iyp = iy+1

                for ixh in range(1, nCx+1):

                    # Index and minus index
                    ix = min(ixh, nCx-1)
                    ixm = ixh-1

                    # Averaging of 1/mu_r: mzyRxm etc.
                    mzyLxm = ky[iym]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzyRxm = ky[iy]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    myzLxm = kz[izm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myzRxm = kz[iz]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    # mzyLxp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    # mzyRxp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    # myzLxp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    # myzRxp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mzxLym = kx[ixm]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzxRym = kx[ix]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mxzLym = kz[izm]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxzRym = kz[iz]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    mzxLyp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    mzxRyp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    mxzLyp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    mxzRyp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])
                    myxLzm = kx[ixm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myxRzm = kx[ix]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    mxyLzm = ky[iym]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxyRzm = ky[iy]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    myxLzp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    myxRzp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mxyLzp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    mxyRzp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])

                    # Diagonal elements
                    st0 = (eta_x[ixm, iy, iz] + eta_x[ixm, iy, izm] +
                           eta_x[ixm, iym, iz] + eta_x[ixm, iym, izm])
                    # st1 = (eta_x[ix, iy, iz] + eta_x[ix, iy, izm] +
                    #        eta_x[ix, iym, iz] + eta_x[ix, iym, izm])
                    st2 = (eta_y[ix, iym, iz] + eta_y[ix, iym, izm] +
                           eta_y[ixm, iym, iz] + eta_y[ixm, iym, izm])
                    st3 = (eta_y[ix, iy, iz] + eta_y[ix, iy, izm] +
                           eta_y[ixm, iy, iz] + eta_y[ixm, iy, izm])
                    st4 = (eta_z[ix, iy, izm] + eta_z[ix, iym, izm] +
                           eta_z[ixm, iy, izm] + eta_z[ixm, iym, izm])
                    st5 = (eta_z[ix, iy, iz] + eta_z[ix, iym, iz] +
                           eta_z[ixm, iy, iz] + eta_z[ixm, iym, iz])

                    st = np.array([st0, st2, st3, st4, st5])/4.

                    # Fill middle matrix

                    # Initial diagonal elements
                    for k in range(5):
                        middle[6*k] = -st[k]

                    # Complete diagonals.
                    # middle is symmetric and curl curl part is real-valued.
                    middle[0] += mzyRxm/hy[iy] + mzyLxm/hy[iym]   # 0,0| 0
                    middle[0] += myzRxm/hz[iz] + myzLxm/hz[izm]
                    middle[6] += mzxRym/hx[ix] + mzxLym/hx[ixm]   # 1,1| 6
                    middle[6] += mxzRym/hz[iz] + mxzLym/hz[izm]
                    middle[12] += mzxRyp/hx[ix] + mzxLyp/hx[ixm]  # 2,2|12
                    middle[12] += mxzRyp/hz[iz] + mxzLyp/hz[izm]
                    middle[18] += myxRzm/hx[ix] + myxLzm/hx[ixm]  # 3,3|18
                    middle[18] += mxyRzm/hy[iy] + mxyLzm/hy[iym]
                    middle[24] += myxRzp/hx[ix] + myxLzp/hx[ixm]  # 4,4|24
                    middle[24] += mxyRzp/hy[iy] + mxyLzp/hy[iym]

                    # Off-diagonal elements of middle.
                    # Upper triangle not needed and not set.
                    # The elements
                    #   [2, 1] (7); [1, 2] (11); [4, 3] (19); and [3, 4] (23)
                    # are all zero.
                    middle[1] = -mzyLxm/hx[ixm]  # 1,0| 1 and 0,1| 5
                    middle[2] = mzyRxm/hx[ixm]   # 2,0| 2 and 0,2|10
                    middle[3] = -myzLxm/hx[ixm]  # 3,0| 3 and 0,3|15
                    middle[4] = myzRxm/hx[ixm]   # 4,0| 4 and 0,4|20
                    middle[8] = -mxzLym/hy[iym]  # 3,1| 8 and 1,3|16
                    middle[9] = mxzRym/hy[iym]   # 4,1| 9 and 1,4|21
                    middle[13] = mxzLyp/hy[iy]   # 3,2|13 and 2,3|17
                    middle[14] = -mxzRyp/hy[iy]  # 4,2|14 and 2,4|22

                    # Fill left matrix left
                    left[5] = mzyLxm/hx[ixm]    # 0,1| 5
                    left[10] = -mzyRxm/hx[ixm]  # 0,2|10
                    left[15] = myzLxm/hx[ixm]   # 0,3|15
                    left[20] = -myzRxm/hx[ixm]  # 0,4|20
                    left[6] = -mzxLym/hx[ixm]   # 1,1| 6
                    left[12] = -mzxLyp/hx[ixm]  # 2,2|12
                    left[18] = -myxLzm/hx[ixm]  # 3,3|18
                    left[24] = -myxLzp/hx[ixm]  # 4,4|24

                    # Fill residual (b - Ux^{(k)})
                    # Note: rhs is NOT the full residual at this point

                    # Residual / right-hand-side
                    r0 = sx[ixm, iy, iz]
                    # r1 = sx[ix, iy, iz]
                    r2 = sy[ix, iym, iz]
                    r3 = sy[ix, iy, iz]
                    r4 = sz[ix, iy, izm]
                    r5 = sz[ix, iy, iz]
                    rhs = np.array([r0, r2, r3, r4, r5])

                    rhs[0] += mzyRxm*ex[ixm, iyp, iz]/hy[iy]
                    rhs[0] += mzyLxm*ex[ixm, iym, iz]/hy[iym]
                    rhs[0] += myzRxm*ex[ixm, iy, izp]/hz[iz]
                    rhs[0] += myzLxm*ex[ixm, iy, izm]/hz[izm]

                    rhs[1] += (mzxRym*ex[ix, iym, iz] -
                               mzxLym*ex[ixm, iym, iz] +
                               mxzRym*ez[ix, iym, iz] -
                               mxzLym*ez[ix, iym, izm])/hy[iym]
                    rhs[1] += mxzRym*ey[ix, iym, izp]/hz[iz]
                    rhs[1] += mxzLym*ey[ix, iym, izm]/hz[izm]

                    rhs[2] += (mzxLyp*ex[ixm, iyp, iz] -
                               mzxRyp*ex[ix, iyp, iz] +
                               mxzLyp*ez[ix, iyp, izm] -
                               mxzRyp*ez[ix, iyp, iz])/hy[iy]
                    rhs[2] += mxzRyp*ey[ix, iy, izp]/hz[iz]
                    rhs[2] += mxzLyp*ey[ix, iy, izm]/hz[izm]

                    rhs[3] += (myxRzm*ex[ix, iy, izm] -
                               myxLzm*ex[ixm, iy, izm] +
                               mxyRzm*ey[ix, iy, izm] -
                               mxyLzm*ey[ix, iym, izm])/hz[izm]
                    rhs[3] += mxyRzm*ez[ix, iyp, izm]/hy[iy]
                    rhs[3] += mxyLzm*ez[ix, iym, izm]/hy[iym]

                    rhs[4] += (myxLzp*ex[ixm, iy, izp] -
                               myxRzp*ex[ix, iy, izp] +
                               mxyLzp*ey[ix, iym, izp] -
                               mxyRzp*ey[ix, iy, izp])/hz[iz]
                    rhs[4] += mxyRzp*ez[ix, iyp, iz]/hy[iy]
                    rhs[4] += mxyLzp*ez[ix, iym, iz]/hy[iym]

                    # Copy to big system
                    blocks_to_amat(amat, bvec, middle, left, rhs, ixm, nCx)

                # Solve linear system A x = b.
                solve(amat, bvec)

                # Update efield (here we could apply damping weights).
                for ix in range(1, nCx+1):
                    ixm = ix-1

                    ex[ixm, iy, iz] = bvec[5*ixm]
                    if ixm < nCx-1:
                        ey[ix, iym, iz] = bvec[1+5*ixm]
                        ey[ix, iy, iz] = bvec[2+5*ixm]
                        ez[ix, iy, izm] = bvec[3+5*ixm]
                        ez[ix, iy, iz] = bvec[4+5*ixm]


@nb.njit(**_numba_setting)
def gauss_seidel_y(ex, ey, ez, sx, sy, sz, eta_x, eta_y, eta_z, mu_r, hx, hy,
                   hz, nu):
    r"""Gauss-Seidel method with line relaxation in y-direction.

    This is the equivalent to :func:`gauss_seidel`, but with line relaxation in
    the y-direction. See :func:`gauss_seidel` for more details.

    The resulting system A x = b to solve consists of n unknowns (x-vector),
    and the corresponding matrix A is a banded matrix with the main diagonal
    and five upper and lower diagonals::

       .-0
       |X|\   0
       0-.-0       left:  middle:  right:
        \|X|\                      (not used)
         0-.-0      0-     .-      0
          \|X|\      \     |X      |\
           0-.-0
        0   \|X|
             0-.

       . 1*1, - 4*1, | 1*4, X 4*4, \ 4*4 upper or lower

    The matrix A is complex and symmetric (A = A^T), and therefore only the
    main diagonal and the lower five off-diagonals are required.

    - The right-hand-side b has length 5*nCy-4 (nCy even).
    - The matrix A has length of b and 1+2*5 diagonals; we use for it an array
      of length 6*len(b).

    The values are calculated in rows of 5 lines, with the indicated middle and
    left matrices as indicated in the above scheme. These blocks are filled
    into the main matrix A and vector b, and subsequently solved with a
    non-standard Cholesky factorisation, :func:`solve`.

    Note: The smoothing with linerelaxation in y-direction is carried out in
    reversed lexicographical order, in order to improve speed (memory access).
    All other smoothers (:func:`gauss_seidel`, :func:`gauss_seidel_x`, and
    :func:`gauss_seidel_z`) use lexicographical order.

    Tangential components at the boundaries are assumed to be 0 (PEC
    boundaries).

    The result is stored in the provided electric fields ``ex``, ``ey``, and
    ``ez``.


    Parameters
    ----------
    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    sx, sy, sz :
        Source fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    eta_x, eta_y, eta_z, mu_r :
        Model parameters (multiplied by volumes) as obtained from
        :func:`emg3d.utils.Model`.

    hx, hy, hz : ndarray
        Cell widths in x-, y-, and z-directions.

    nu : int
        Number of Gauss-Seidel iterations.

    """

    # Get dimensions
    nCx = len(hx)
    nCy = len(hy)
    nCz = len(hz)

    # Get half of the inverse widths
    kx = 0.5/hx
    ky = 0.5/hy
    kz = 0.5/hz

    # Direction-switch for Gauss-Seidel
    iback = 0

    # Pre-allocating middle and left for the 5x5-temporary middle and left
    # matrices; will be overwritten at each iteration
    middle = np.zeros(25, dtype=np.complex128)
    left = np.zeros(25)

    # Pre-allocating full RHS (bvec) and full matrix A (amat). Will be
    # overwritten after each complete y-loop.
    nr = 5*nCy-4  # Number of unknowns
    bvec = np.zeros(nr, dtype=np.complex128)
    amat = np.zeros(6*nr, dtype=np.complex128)

    # Smoothing steps
    for _ in range(nu):

        # Direction of Gauss-Seidel ordering; 0=forward, 1=backward
        iback = 1-iback

        # Loop over cells, keeping boundaries fixed; y-fastest, then z, x.
        for izh in range(1, nCz):

            # Back-forth-switch
            if iback:
                iz = nCz-izh
            else:
                iz = izh

            # Minus/plus indices
            izm = iz-1
            izp = iz+1

            for ixh in range(1, nCx):

                # Back-forth-switch
                if iback:
                    ix = nCx-ixh
                else:
                    ix = ixh

                # Minus/plus indices
                ixm = ix-1
                ixp = ix+1

                for iyh in range(1, nCy+1):

                    # Index and minus index
                    iy = min(iyh, nCy-1)
                    iym = iyh-1

                    # Averaging of 1/mu_r: mzyRxm etc.
                    mzyLxm = ky[iym]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzyRxm = ky[iy]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    myzLxm = kz[izm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myzRxm = kz[iz]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    mzyLxp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mzyRxp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    myzLxp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    myzRxp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mzxLym = kx[ixm]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzxRym = kx[ix]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mxzLym = kz[izm]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxzRym = kz[iz]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    # mzxLyp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    # mzxRyp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    # mxzLyp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    # mxzRyp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])
                    myxLzm = kx[ixm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myxRzm = kx[ix]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    mxyLzm = ky[iym]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxyRzm = ky[iy]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    myxLzp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    myxRzp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mxyLzp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    mxyRzp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])

                    # Diagonal elements
                    st0 = (eta_x[ixm, iy, iz] + eta_x[ixm, iy, izm] +
                           eta_x[ixm, iym, iz] + eta_x[ixm, iym, izm])
                    st1 = (eta_x[ix, iy, iz] + eta_x[ix, iy, izm] +
                           eta_x[ix, iym, iz] + eta_x[ix, iym, izm])
                    st2 = (eta_y[ix, iym, iz] + eta_y[ix, iym, izm] +
                           eta_y[ixm, iym, iz] + eta_y[ixm, iym, izm])
                    # st3 = (eta_y[ix, iy, iz] + eta_y[ix, iy, izm] +
                    #        eta_y[ixm, iy, iz] + eta_y[ixm, iy, izm])
                    st4 = (eta_z[ix, iy, izm] + eta_z[ix, iym, izm] +
                           eta_z[ixm, iy, izm] + eta_z[ixm, iym, izm])
                    st5 = (eta_z[ix, iy, iz] + eta_z[ix, iym, iz] +
                           eta_z[ixm, iy, iz] + eta_z[ixm, iym, iz])

                    st = np.array([st2, st0, st1, st4, st5])/4.

                    # Fill middle matrix

                    # Initial diagonal elements
                    for k in range(5):
                        middle[6*k] = -st[k]

                    # Complete diagonals.
                    # middle is symmetric and curl curl part is real-valued.
                    middle[0] += mzxRym/hx[ix] + mzxLym/hx[ixm]   # 0,0| 0
                    middle[0] += mxzRym/hz[iz] + mxzLym/hz[izm]
                    middle[6] += mzyRxm/hy[iy] + mzyLxm/hy[iym]   # 1,1| 6
                    middle[6] += myzRxm/hz[iz] + myzLxm/hz[izm]
                    middle[12] += mzyRxp/hy[iy] + mzyLxp/hy[iym]  # 2,2|12
                    middle[12] += myzRxp/hz[iz] + myzLxp/hz[izm]
                    middle[18] += myxRzm/hx[ix] + myxLzm/hx[ixm]  # 3,3|18
                    middle[18] += mxyRzm/hy[iy] + mxyLzm/hy[iym]
                    middle[24] += myxRzp/hx[ix] + myxLzp/hx[ixm]  # 4,4|24
                    middle[24] += mxyRzp/hy[iy] + mxyLzp/hy[iym]

                    # Off-diagonal elements of middle.
                    # Upper triangle not needed and not set.
                    # The elements
                    #   [2, 1] (7); [1, 2] (11); [4, 3] (19); and [3, 4] (23)
                    # are all zero.
                    middle[1] = -mzyLxm/hx[ixm]  # 1,0| 1 and 0,1| 5
                    middle[2] = mzyLxp/hx[ix]    # 2,0| 2 and 0,2|10
                    middle[3] = -mxzLym/hy[iym]  # 3,0| 3 and 0,3|15
                    middle[4] = mxzRym/hy[iym]   # 4,0| 4 and 0,4|20
                    middle[8] = -myzLxm/hx[ixm]  # 3,1| 8 and 1,3|16
                    middle[9] = myzRxm/hx[ixm]   # 4,1| 9 and 1,4|21
                    middle[13] = myzLxp/hx[ix]   # 3,2|13 and 2,3|17
                    middle[14] = -myzRxp/hx[ix]  # 4,2|14 and 2,4|22

                    # Fill left matrix left
                    left[5] = mzxLym/hy[iym]    # 0,1| 5
                    left[10] = -mzxRym/hy[iym]  # 0,2|10
                    left[15] = mxzLym/hy[iym]   # 0,3|15
                    left[20] = -mxzRym/hy[iym]  # 0,4|20
                    left[6] = -mzyLxm/hy[iym]   # 1,1| 6
                    left[12] = -mzyLxp/hy[iym]  # 2,2|12
                    left[18] = -mxyLzm/hy[iym]  # 3,3|18
                    left[24] = -mxyLzp/hy[iym]  # 4,4|24

                    # Fill residual (b - Ux^{(k)})
                    # Note: rhs is NOT the full residual at this point

                    # Residual / right-hand-side
                    r0 = sx[ixm, iy, iz]
                    r1 = sx[ix, iy, iz]
                    r2 = sy[ix, iym, iz]
                    # r3 = sy[ix, iy, iz]
                    r4 = sz[ix, iy, izm]
                    r5 = sz[ix, iy, iz]
                    rhs = np.array([r2, r0, r1, r4, r5])

                    rhs[0] += mzxRym*ey[ixp, iym, iz]/hx[ix]
                    rhs[0] += mzxLym*ey[ixm, iym, iz]/hx[ixm]
                    rhs[0] += mxzRym*ey[ix, iym, izp]/hz[iz]
                    rhs[0] += mxzLym*ey[ix, iym, izm]/hz[izm]

                    rhs[1] += (mzyRxm*ey[ixm, iy, iz] -
                               mzyLxm*ey[ixm, iym, iz] +
                               myzRxm*ez[ixm, iy, iz] -
                               myzLxm*ez[ixm, iy, izm])/hx[ixm]
                    rhs[1] += myzRxm*ex[ixm, iy, izp]/hz[iz]
                    rhs[1] += myzLxm*ex[ixm, iy, izm]/hz[izm]

                    rhs[2] += (mzyLxp*ey[ixp, iym, iz] -
                               mzyRxp*ey[ixp, iy, iz] +
                               myzLxp*ez[ixp, iy, izm] -
                               myzRxp*ez[ixp, iy, iz])/hx[ix]
                    rhs[2] += myzRxp*ex[ix, iy, izp]/hz[iz]
                    rhs[2] += myzLxp*ex[ix, iy, izm]/hz[izm]

                    rhs[3] += (myxRzm*ex[ix, iy, izm] -
                               myxLzm*ex[ixm, iy, izm] +
                               mxyRzm*ey[ix, iy, izm] -
                               mxyLzm*ey[ix, iym, izm])/hz[izm]
                    rhs[3] += myxRzm*ez[ixp, iy, izm]/hx[ix]
                    rhs[3] += myxLzm*ez[ixm, iy, izm]/hx[ixm]

                    rhs[4] += (myxLzp*ex[ixm, iy, izp] -
                               myxRzp*ex[ix, iy, izp] +
                               mxyLzp*ey[ix, iym, izp] -
                               mxyRzp*ey[ix, iy, izp])/hz[iz]
                    rhs[4] += myxRzp*ez[ixp, iy, iz]/hx[ix]
                    rhs[4] += myxLzp*ez[ixm, iy, iz]/hx[ixm]

                    # Copy to big system
                    blocks_to_amat(amat, bvec, middle, left, rhs, iym, nCy)

                # Solve linear system A x = b.
                solve(amat, bvec)

                # Update efield (here we could apply damping weights).
                for iy in range(1, nCy+1):
                    iym = iy-1

                    ey[ix, iym, iz] = bvec[5*iym]
                    if iym < nCy-1:
                        ex[ixm, iy, iz] = bvec[1+5*iym]
                        ex[ix, iy, iz] = bvec[2+5*iym]
                        ez[ix, iy, izm] = bvec[3+5*iym]
                        ez[ix, iy, iz] = bvec[4+5*iym]


@nb.njit(**_numba_setting)
def gauss_seidel_z(ex, ey, ez, sx, sy, sz, eta_x, eta_y, eta_z, mu_r, hx, hy,
                   hz, nu):
    r"""Gauss-Seidel method with line relaxation in z-direction.

    This is the equivalent to :func:`gauss_seidel`, but with line relaxation in
    the z-direction. See :func:`gauss_seidel` for more details.

    The resulting system A x = b to solve consists of n unknowns (x-vector),
    and the corresponding matrix A is a banded matrix with the main diagonal
    and five upper and lower diagonals::

       .-0
       |X|\   0
       0-.-0       left:  middle:  right:
        \|X|\                      (not used)
         0-.-0      0-     .-      0
          \|X|\      \     |X      |\
           0-.-0
        0   \|X|
             0-.

       . 1*1, - 4*1, | 1*4, X 4*4, \ 4*4 upper or lower

    The matrix A is complex and symmetric (A = A^T), and therefore only the
    main diagonal and the lower five off-diagonals are required.

    - The right-hand-side b has length 5*nCz-4 (nCz even).
    - The matrix A has length of b and 1+2*5 diagonals; we use for it an array
      of length 6*len(b).

    The values are calculated in rows of 5 lines, with the indicated middle and
    left matrices as indicated in the above scheme. These blocks are filled
    into the main matrix A and vector b, and subsequently solved with a
    non-standard Cholesky factorisation, :func:`solve`.

    Tangential components at the boundaries are assumed to be 0 (PEC
    boundaries).

    The result is stored in the provided electric fields ``ex``, ``ey``, and
    ``ez``.


    Parameters
    ----------
    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    sx, sy, sz :
        Source fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.utils.Field`.

    eta_x, eta_y, eta_z, mu_r :
        Model parameters (multiplied by volumes) as obtained from
        :func:`emg3d.utils.Model`.

    hx, hy, hz : ndarray
        Cell widths in x-, y-, and z-directions.

    nu : int
        Number of Gauss-Seidel iterations.

    """

    # Get dimensions
    nCx = len(hx)
    nCy = len(hy)
    nCz = len(hz)

    # Get half of the inverse widths
    kx = 0.5/hx
    ky = 0.5/hy
    kz = 0.5/hz

    # Direction-switch for Gauss-Seidel
    iback = 0

    # Pre-allocating middle and left for the 5x5-temporary middle and left
    # matrices; will be overwritten at each iteration
    middle = np.zeros(25, dtype=np.complex128)
    left = np.zeros(25)

    # Pre-allocating full RHS (bvec) and full matrix A (amat). Will be
    # overwritten after each complete z-loop.
    nr = 5*nCz-4  # Number of unknowns
    bvec = np.zeros(nr, dtype=np.complex128)
    amat = np.zeros(6*nr, dtype=np.complex128)

    # Smoothing steps
    for _ in range(nu):

        # Direction of Gauss-Seidel ordering; 0=forward, 1=backward
        iback = 1-iback

        # Loop over cells, keeping boundaries fixed; z-fastest, then x, y.
        for iyh in range(1, nCy):

            # Back-forth-switch
            if iback:
                iy = nCy-iyh
            else:
                iy = iyh

            # Minus/plus indices
            iym = iy-1
            iyp = iy+1

            for ixh in range(1, nCx):

                # Back-forth-switch
                if iback:
                    ix = nCx-ixh
                else:
                    ix = ixh

                # Minus/plus indices
                ixm = ix-1
                ixp = ix+1

                for izh in range(1, nCz+1):

                    # Index and minus index
                    iz = min(izh, nCz-1)
                    izm = izh-1

                    # Averaging of 1/mu_r: mzyRxm etc.
                    mzyLxm = ky[iym]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzyRxm = ky[iy]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    myzLxm = kz[izm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myzRxm = kz[iz]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    mzyLxp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mzyRxp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    myzLxp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    myzRxp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    mzxLym = kx[ixm]*(mu_r[ixm, iym, iz] + mu_r[ixm, iym, izm])
                    mzxRym = kx[ix]*(mu_r[ix, iym, iz] + mu_r[ix, iym, izm])
                    mxzLym = kz[izm]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxzRym = kz[iz]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    mzxLyp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iy, izm])
                    mzxRyp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iy, izm])
                    mxzLyp = kz[izm]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    mxzRyp = kz[iz]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])
                    myxLzm = kx[ixm]*(mu_r[ixm, iy, izm] + mu_r[ixm, iym, izm])
                    myxRzm = kx[ix]*(mu_r[ix, iy, izm] + mu_r[ix, iym, izm])
                    mxyLzm = ky[iym]*(mu_r[ix, iym, izm] + mu_r[ixm, iym, izm])
                    mxyRzm = ky[iy]*(mu_r[ix, iy, izm] + mu_r[ixm, iy, izm])
                    # myxLzp = kx[ixm]*(mu_r[ixm, iy, iz] + mu_r[ixm, iym, iz])
                    # myxRzp = kx[ix]*(mu_r[ix, iy, iz] + mu_r[ix, iym, iz])
                    # mxyLzp = ky[iym]*(mu_r[ix, iym, iz] + mu_r[ixm, iym, iz])
                    # mxyRzp = ky[iy]*(mu_r[ix, iy, iz] + mu_r[ixm, iy, iz])

                    # Diagonal elements
                    st0 = (eta_x[ixm, iy, iz] + eta_x[ixm, iy, izm] +
                           eta_x[ixm, iym, iz] + eta_x[ixm, iym, izm])
                    st1 = (eta_x[ix, iy, iz] + eta_x[ix, iy, izm] +
                           eta_x[ix, iym, iz] + eta_x[ix, iym, izm])
                    st2 = (eta_y[ix, iym, iz] + eta_y[ix, iym, izm] +
                           eta_y[ixm, iym, iz] + eta_y[ixm, iym, izm])
                    st3 = (eta_y[ix, iy, iz] + eta_y[ix, iy, izm] +
                           eta_y[ixm, iy, iz] + eta_y[ixm, iy, izm])
                    st4 = (eta_z[ix, iy, izm] + eta_z[ix, iym, izm] +
                           eta_z[ixm, iy, izm] + eta_z[ixm, iym, izm])
                    # st5 = (eta_z[ix, iy, iz] + eta_z[ix, iym, iz] +
                    #        eta_z[ixm, iy, iz] + eta_z[ixm, iym, iz])

                    st = np.array([st4, st0, st1, st2, st3])/4.

                    # Fill middle matrix

                    # Initial diagonal elements
                    for k in range(5):
                        middle[6*k] = -st[k]

                    # Complete diagonals.
                    # middle is symmetric and curl curl part is real-valued.
                    middle[0] += myxRzm/hx[ix] + myxLzm/hx[ixm]   # 0,0| 0
                    middle[0] += mxyRzm/hy[iy] + mxyLzm/hy[iym]
                    middle[6] += mzyRxm/hy[iy] + mzyLxm/hy[iym]   # 1,1| 6
                    middle[6] += myzRxm/hz[iz] + myzLxm/hz[izm]
                    middle[12] += mzyRxp/hy[iy] + mzyLxp/hy[iym]  # 2,2|12
                    middle[12] += myzRxp/hz[iz] + myzLxp/hz[izm]
                    middle[18] += mzxRym/hx[ix] + mzxLym/hx[ixm]  # 3,3|18
                    middle[18] += mxzRym/hz[iz] + mxzLym/hz[izm]
                    middle[24] += mzxRyp/hx[ix] + mzxLyp/hx[ixm]  # 4,4|24
                    middle[24] += mxzRyp/hz[iz] + mxzLyp/hz[izm]

                    # Off-diagonal elements of middle.
                    # Upper triangle not needed and not set.
                    # The elements
                    #   [2, 1] (7); [1, 2] (11); [4, 3] (19); and [3, 4] (23)
                    # are all zero.
                    middle[1] = -myzLxm/hx[ixm]  # 1,0| 1 and 0,1| 5
                    middle[2] = myzLxp/hx[ix]    # 2,0| 2 and 0,2|10
                    middle[3] = -mxzLym/hy[iym]  # 3,0| 3 and 0,3|15
                    middle[4] = mxzLyp/hy[iy]    # 4,0| 4 and 0,4|20
                    middle[8] = -mzyLxm/hx[ixm]  # 3,1| 8 and 1,3|16
                    middle[9] = mzyRxm/hx[ixm]   # 4,1| 9 and 1,4|21
                    middle[13] = mzyLxp/hx[ix]   # 3,2|13 and 2,3|17
                    middle[14] = -mzyRxp/hx[ix]  # 4,2|14 and 2,4|22

                    # Fill left matrix left
                    left[5] = myxLzm/hz[izm]    # 0,1| 5
                    left[10] = -myxRzm/hz[izm]  # 0,2|10
                    left[15] = mxyLzm/hz[izm]   # 0,3|15
                    left[20] = -mxyRzm/hz[izm]  # 0,4|20
                    left[6] = -myzLxm/hz[izm]   # 1,1| 6
                    left[12] = -myzLxp/hz[izm]  # 2,2|12
                    left[18] = -mxzLym/hz[izm]  # 3,3|18
                    left[24] = -mxzLyp/hz[izm]  # 4,4|24

                    # Fill residual (b - Ux^{(k)})
                    # Note: rhs is NOT the full residual at this point

                    # Residual / right-hand-side
                    r0 = sx[ixm, iy, iz]
                    r1 = sx[ix, iy, iz]
                    r2 = sy[ix, iym, iz]
                    r3 = sy[ix, iy, iz]
                    r4 = sz[ix, iy, izm]
                    # r5 = sz[ix, iy, iz]
                    rhs = np.array([r4, r0, r1, r2, r3])

                    rhs[0] += myxRzm*(ez[ixp, iy, izm]/hx[ix])
                    rhs[0] += myxLzm*(ez[ixm, iy, izm]/hx[ixm])
                    rhs[0] += mxyRzm*(ez[ix, iyp, izm]/hy[iy])
                    rhs[0] += mxyLzm*(ez[ix, iym, izm]/hy[iym])

                    rhs[1] += (mzyRxm*ey[ixm, iy, iz] -
                               mzyLxm*ey[ixm, iym, iz] +
                               myzRxm*ez[ixm, iy, iz] -
                               myzLxm*ez[ixm, iy, izm])/hx[ixm]
                    rhs[1] += mzyRxm*ex[ixm, iyp, iz]/hy[iy]
                    rhs[1] += mzyLxm*ex[ixm, iym, iz]/hy[iym]

                    rhs[2] += (mzyLxp*ey[ixp, iym, iz] -
                               mzyRxp*ey[ixp, iy, iz] +
                               myzLxp*ez[ixp, iy, izm] -
                               myzRxp*ez[ixp, iy, iz])/hx[ix]
                    rhs[2] += mzyRxp*ex[ix, iyp, iz]/hy[iy]
                    rhs[2] += mzyLxp*ex[ix, iym, iz]/hy[iym]

                    rhs[3] += (mzxRym*ex[ix, iym, iz] -
                               mzxLym*ex[ixm, iym, iz] +
                               mxzRym*ez[ix, iym, iz] -
                               mxzLym*ez[ix, iym, izm])/hy[iym]
                    rhs[3] += mzxRym*ey[ixp, iym, iz]/hx[ix]
                    rhs[3] += mzxLym*ey[ixm, iym, iz]/hx[ixm]

                    rhs[4] += (mzxLyp*ex[ixm, iyp, iz] -
                               mzxRyp*ex[ix, iyp, iz] +
                               mxzLyp*ez[ix, iyp, izm] -
                               mxzRyp*ez[ix, iyp, iz])/hy[iy]
                    rhs[4] += mzxRyp*ey[ixp, iy, iz]/hx[ix]
                    rhs[4] += mzxLyp*ey[ixm, iy, iz]/hx[ixm]

                    # Copy to big system
                    blocks_to_amat(amat, bvec, middle, left, rhs, izm, nCz)

                # Solve linear system A x = b.
                solve(amat, bvec)

                # Update efield (here we could apply damping weights).
                for iz in range(1, nCz+1):
                    izm = iz-1

                    ez[ix, iy, izm] = bvec[5*izm]
                    if izm < nCz-1:
                        ex[ixm, iy, iz] = bvec[1+5*izm]
                        ex[ix, iy, iz] = bvec[2+5*izm]
                        ey[ix, iym, iz] = bvec[3+5*izm]
                        ey[ix, iy, iz] = bvec[4+5*izm]


@nb.njit(**_numba_setting)
def blocks_to_amat(amat, bvec, middle, left, rhs, im, nC):
    r"""Insert middle, left, and rhs into main arrays amat and bvec.

    The banded matrix amat contains the main diagonal and the first five lower
    off-diagonals. They are stored one column after the other, in a 6*n
    ndarray.

    .. highlight:: none

    The complete main matrix ``amat`` and the ``middle`` and ``left`` blocks
    are given by::

       .-0
       |X|\   0
       0-.-0       left:  middle:  right:
        \|X|\                      (not used)
         0-.-0      0-     .-      0
          \|X|\      \     |X      |\
           0-.-0
        0   \|X|
             0-.

       . 1*1, - 4*1, | 1*4, X 4*4, \ 4*4 upper or lower


    Both, ``middle`` and ``left``, are 5x5 matrices. The corresponding
    right-hand-side ``rhs`` is filled into ``bvec``. The matrices ``left`` and
    ``middle`` provided in a single call are horizontally aligned (not
    vertically). The sorting of amat (banded matrix) and bvec are given by::

        amat (66,)             example: n = 11                   bvec (11,)
        --------------                                                 --
       |01            |                    FIRST CALL                  01
       |02 07         |                    Only `middle` and `rhs`     02
       |03 08 13      |                    are used, not `left`.       03
       |04 09 14 19   |                                                04
       |05 10 15 20 25|                                                05
        -------------- --------------                                  --
       | 0 11 16 21 26|31            |     SUBSEQUENT CALLS            06
       |   12 17 22 27|32 37         |     (normal case)               07
       |      18 23 28|33 38 43      |     Complete `left`,            08
       |         24 29|34 39 44 49   |     `middle` and `rhs`          09
       |            30|35 40 45 50 55|     are used.                   10
        -------------- -------------- ---                              --
                      | 0 41 46 51 56|61   LAST CALL                   11
                      |    0  0  0  0| 0   Only top row of `left`
                      |       0  0  0| 0   and the first elements
                      |          0  0| 0   of `middle` and `rhs`
                      |             0| 0   are used.
                       -------------- ---
                                     | 0

       Single zeros (0) show elements in amat which are 0, hence not used.
       Their location in amat can be deduced from their neighbours.

    .. highlight:: default

    Parameters
    ----------
    amat : ndarray
        Main banded matrix (stored as array) of length 6*n.

    bvec : ndarray
        Main right-hand-side of length n.

    middle : ndarray
        Middle block of size 5x5, as ndarray of length 25. Only
        the diagonal and the lower triangular part are used.

    left : ndarray
        Left block of size 5x5, as ndarray of length 25. Only the
        diagonal and the first row are used.

    rhs : ndarray
        Corresponding right-hand-side of length 5.

    im : int
        Current minus-index of direction of line relaxation, {ixm, iym, izm}.

    nC : int
        Total number of cells in direction of line relaxation, {nCx, nCy, nCz}.

    """
    # Define two often used indices
    fam = 5*im
    mam = fam-5

    if im == 0:                  # First block-row; only middle, no left

        # RHS
        for k in range(5):
            bvec[k] = rhs[k]

        # Middle block
        for k in range(5):
            for l in range(k+1):
                amat[k+5*l] = middle[k+5*l]

    elif im <= nC-2 and nC > 2:  # Normal case; full middle and left

        # RHS
        for k in range(5):
            bvec[k+fam] = rhs[k]

        # Left block
        for l in range(1, 5):
            for k in range(l+1):
                amat[k+fam+5*(l+mam)] = left[k+5*l]

        # Middle block
        for k in range(5):
            for l in range(k+1):
                amat[k+fam+5*(l+fam)] = middle[k+5*l]

    elif im == nC-1:             # The last point

        # RHS
        bvec[fam] = rhs[0]

        # First row from left block
        for l in range(1, 5):
            amat[fam+5*(l+mam)] = left[5*l]

        # First element from middle block
        amat[6*fam] = middle[0]


@nb.njit(**_numba_setting)
def solve(amat, bvec):
    r"""Solve A x = b using a non-standard Cholesky factorisation.

    Solve the system A x = b using a non-standard Cholesky factorisation
    without pivoting for a symmetric, complex matrix A tailored to the problem
    of the multigrid solver. The matrix A (amat) is an array of length 6*n,
    containing the main diagonal and the first five lower off-diagonals
    (ordered so that the first element of the main diagonal is followed by the
    first elements of the off diagonals, then the second elements and so on).
    The vector bvec has length b.

    The solution is placed in b (bvec), and A (amat) is replaced by its
    decomposition.

    1. Non-standard Cholesky factorisation.

        From [Muld07]_: We use a non-standard Cholesky factorisation. The
        standard factorisation factors a hermitian matrix A into L L^H, where L
        is a lower triangular matrix and L^H its complex conjugate transpose.
        In our case, the discretisation is based on the Finite Integration
        Technique ([Weil77]_) and provides a matrix A that is complex-valued
        and symmetric: A = A^T, where the superscript T denotes the transpose.
        The line relaxation scheme takes a matrix B that is a subset of A along
        the line. B is a complex symmetric band matrix with eleven diagonals.
        The non-standard Cholesky factorisation factors the matrix B into L
        L^T. Because of the symmetry, only the main diagonal and five lower
        diagonal elements of B need to be computed. The Cholesky factorisation
        replaces this matrix by L, containing six diagonals, after which the
        line relaxation can be carried out by simple back-substitution.

        :math:`A = L D L^T` factorisation without pivoting:

        .. math::

            D(j) &= A(j,j)-\sum_{k=1}^{j-1} L(j,k)^2 D(k),\ j=1,..,n ;\\
            L(i,j) &= \frac{1}{D(j)}
                     \left[A(i,j)-\sum_{k=1}^{j-1} L(i,k)L(j,k)D(k)\right],
                     \ i=j+1,..,n .

        A and L are in this case arrays, where :math:`A(i, j) \rightarrow
        A(i+5j)`.

    2. Solve A x = b.

        Solve A x = b, given L which is the result from the factorisation in
        the first step (and stored in A), hence, solve L x = b, where x is
        stored in b:

        .. math::

            b(j) = b(j) - \sum_{k=1}^{j-1} L(j,k) x(k), j = 2,..,n .

    The result is equivalent with simply using :func:`numpy.linalg.solve`, but
    faster for the particular use-case of this code.

    Note that in this custom solver there is no pivoting, and the diagonals of
    the matrix cannot be zero.


    Parameters
    ----------
    amat : ndarray
        Banded matrix A provided as a vector of length 6*n, containing main
        diagonal plus first five lower diagonals.

    bvec : ndarray
        Right-hand-side vector b of length n.

    """

    # Number of unknowns
    n = len(bvec)

    # 1. Get L from non-standard Cholesky L D L^T factorisation

    # First element (i = j = 0). Warning: Diagonals of amat cannot be 0!
    d = 1./amat[0]

    # Multiply to other elements of first column (j = 0)
    for i in range(1, min(n, 6)):
        amat[i] *= d

    # Other columns (1 to n)
    for j in range(1, n):

        h = 0+0j
        for k in range(max(0, j-5), j):
            h += amat[j+5*k]*amat[j+5*k]*amat[6*k]

        amat[6*j] -= h

        # Warning: Diagonals of amat cannot be 0!
        d = 1./amat[6*j]

        # Off-diagonals, rows i > j
        for i in range(j+1, min(n, j+6)):

            h = 0+0j
            for k in range(max(0, i-5), j):
                h += amat[i+5*k]*amat[j+5*k]*amat[6*k]

            amat[i+5*j] -= h
            amat[i+5*j] *= d

    # Replace diagonal by 1/D
    amat[6*(n-1)] = d  # Last one is still around
    for j in range(n-2, -1, -1):
        if amat[6*j].real != 0. + amat[6*j] != 0.:
            amat[6*j] = 1./amat[6*j]

    # 2. Solve A x = b

    # All elements except first column
    for j in range(1, n):

        h = 0.+0j
        for k in range(max(0, j-5), j):
            h += amat[j+5*k]*bvec[k]

        bvec[j] -= h

    # Divide by diagonal; A[j, j] (hence A[6j]) contains 1/D[j]
    for j in range(n):
        bvec[j] *= amat[6*j]

    # Solve L^T x = b, x stored in b, L is 1 on diagonal
    for j in range(n-2, -1, -1):

        h = 0.+0.j
        for k in range(j+1, min(n, j+6)):
            h += amat[k+5*j]*bvec[k]

        bvec[j] -= h


# Restriction
@nb.njit(**_numba_setting)
def restrict(crx, cry, crz, rx, ry, rz, wx, wy, wz, sc_dir):
    r"""Restriction of residual from fine to coarse grid.

    Corresponds to Equation 8 in [Muld06]_. The equation for the x-direction,
    using the notation :math:`\{x,y,z\}` instead of :math:`\{1,2,3\}`, is given
    by

    .. math::

        r_{x,K+1/2,L,M}^{2h} =
            &\sum_{j_y=-1}^1\sum_{j_z=-1}^1 w_{L,j_y}^y w_{M,j_z}^z \\
            &\times
            \left(r_{x,k+1/2,l+j_y,m+j_z}^h+r_{x,k+3/2,l+j_y,m+j_z}^h\right) .

    The superscripts :math:`h, 2h` indicate quantities defined on the coarse
    grid and on the fine grid, respectively. The indices :math:`\{K, L, M\}`
    on the coarse grid correspond to :math:`\{k, l, m\} = 2\{K, L, M\}` on the
    fine grid. The weights :math:`w` are obtained from
    :func:`restrict_weights`.

    The restrictions of ``rx``, ``ry``, and ``rz`` are stored directly in
    ``crx``, ``cry``, and ``crz``.

    Parameters
    ----------
    crx, cry, crz : ndarray
        Coarse grid {x,y,z}-directed residual (pre-allocated empty arrays).

    rx, ry, rz : ndarray
        Fine grid {x,y,z}-directed residual.

    wx, wy, wz: tuple
        Tuples containing the weights (wl, w0, wr) as returned from
        :func:`restrict_weights` for the x-, y-, and z-directions.

    sc_dir : int
        Direction of semicoarsening; 0 for no semicoarsening.

    """
    # Number of coarse grid edges.
    cnNx, cnNy, cnNz = cry.shape[0], crx.shape[1], crx.shape[2]

    # Number of fine grid edges.
    nNx, nNy, nNz = ry.shape[0], rx.shape[1], rx.shape[2]

    # Get weights
    wxl, wx0, wxr = wx
    wyl, wy0, wyr = wy
    wzl, wz0, wzr = wz

    if sc_dir == 0:  # Standard

        # Loop over coarse z-edges.
        for ciz in range(cnNz):
            iz = 2*ciz
            izm = max(0, iz-1)
            izp = min(nNz-1, iz+1)

            # Loop over coarse y-edges.
            for ciy in range(cnNy):
                iy = 2*ciy
                iym = max(0, iy-1)
                iyp = min(nNy-1, iy+1)

                # Sum the terms for x-field.
                crx[:, ciy, ciz] = wy0[ciy]*(
                        wz0[ciz]*(rx[::2, iy, iz] + rx[1::2, iy, iz]) +
                        wzl[ciz]*(rx[::2, iy, izm] + rx[1::2, iy, izm]) +
                        wzr[ciz]*(rx[::2, iy, izp] + rx[1::2, iy, izp])
                )

                crx[:, ciy, ciz] += wyl[ciy]*(
                        wz0[ciz]*(rx[::2, iym, iz] + rx[1::2, iym, iz]) +
                        wzl[ciz]*(rx[::2, iym, izm] + rx[1::2, iym, izm]) +
                        wzr[ciz]*(rx[::2, iym, izp] + rx[1::2, iym, izp])
                )

                crx[:, ciy, ciz] += wyr[ciy]*(
                        wz0[ciz]*(rx[::2, iyp, iz] + rx[1::2, iyp, iz]) +
                        wzl[ciz]*(rx[::2, iyp, izm] + rx[1::2, iyp, izm]) +
                        wzr[ciz]*(rx[::2, iyp, izp] + rx[1::2, iyp, izp])
                )

            # Loop over coarse x-edges.
            for cix in range(cnNx):
                ix = 2*cix
                ixm = max(0, ix-1)
                ixp = min(nNx-1, ix+1)

                # Sum the terms for y-field.
                cry[cix, :, ciz] = wx0[cix]*(
                        wz0[ciz]*(ry[ix, ::2, iz] + ry[ix, 1::2, iz]) +
                        wzl[ciz]*(ry[ix, ::2, izm] + ry[ix, 1::2, izm]) +
                        wzr[ciz]*(ry[ix, ::2, izp] + ry[ix, 1::2, izp])
                )

                cry[cix, :, ciz] += wxl[cix]*(
                        wz0[ciz]*(ry[ixm, ::2, iz] + ry[ixm, 1::2, iz]) +
                        wzl[ciz]*(ry[ixm, ::2, izm] + ry[ixm, 1::2, izm]) +
                        wzr[ciz]*(ry[ixm, ::2, izp] + ry[ixm, 1::2, izp])
                )

                cry[cix, :, ciz] += wxr[cix]*(
                        wz0[ciz]*(ry[ixp, ::2, iz] + ry[ixp, 1::2, iz]) +
                        wzl[ciz]*(ry[ixp, ::2, izm] + ry[ixp, 1::2, izm]) +
                        wzr[ciz]*(ry[ixp, ::2, izp] + ry[ixp, 1::2, izp])
                )

        # Loop over coarse y-edges.
        for ciy in range(cnNy):
            iy = 2*ciy
            iym = max(0, iy-1)
            iyp = min(nNy-1, iy+1)

            # Loop over coarse x-edges.
            for cix in range(cnNx):
                ix = 2*cix
                ixm = max(0, ix-1)
                ixp = min(nNx-1, ix+1)

                # Sum the terms for z-field.
                crz[cix, ciy, :] = wx0[cix]*(
                        wy0[ciy]*(rz[ix, iy, ::2] + rz[ix, iy, 1::2]) +
                        wyl[ciy]*(rz[ix, iym, ::2] + rz[ix, iym, 1::2]) +
                        wyr[ciy]*(rz[ix, iyp, ::2] + rz[ix, iyp, 1::2])
                )

                crz[cix, ciy, :] += wxl[cix]*(
                        wy0[ciy]*(rz[ixm, iy, ::2] + rz[ixm, iy, 1::2]) +
                        wyl[ciy]*(rz[ixm, iym, ::2] + rz[ixm, iym, 1::2]) +
                        wyr[ciy]*(rz[ixm, iyp, ::2] + rz[ixm, iyp, 1::2])
                )

                crz[cix, ciy, :] += wxr[cix]*(
                        wy0[ciy]*(rz[ixp, iy, ::2] + rz[ixp, iy, 1::2]) +
                        wyl[ciy]*(rz[ixp, iym, ::2] + rz[ixp, iym, 1::2]) +
                        wyr[ciy]*(rz[ixp, iyp, ::2] + rz[ixp, iyp, 1::2])
                )

    elif sc_dir == 1:  # Restrict in y- and z-directions

        # Loop over coarse z-edges.
        for ciz in range(cnNz):
            iz = 2*ciz
            izm = max(0, iz-1)
            izp = min(nNz-1, iz+1)

            # Sum the terms for y-field.
            cry[:, :, ciz] = wz0[ciz]*(ry[:, ::2, iz] + ry[:, 1::2, iz])
            cry[:, :, ciz] += wzl[ciz]*(ry[:, ::2, izm] + ry[:, 1::2, izm])
            cry[:, :, ciz] += wzr[ciz]*(ry[:, ::2, izp] + ry[:, 1::2, izp])

            # Loop over coarse y-edges.
            for ciy in range(cnNy):
                iy = 2*ciy
                iym = max(0, iy-1)
                iyp = min(nNy-1, iy+1)

                # Sum the terms for x-field.
                crx[:, ciy, ciz] = wy0[ciy]*(
                        wz0[ciz]*rx[:, iy, iz] +
                        wzl[ciz]*rx[:, iy, izm] +
                        wzr[ciz]*rx[:, iy, izp]
                )

                crx[:, ciy, ciz] += wyl[ciy]*(
                        wz0[ciz]*rx[:, iym, iz] +
                        wzl[ciz]*rx[:, iym, izm] +
                        wzr[ciz]*rx[:, iym, izp]
                )

                crx[:, ciy, ciz] += wyr[ciy]*(
                        wz0[ciz]*rx[:, iyp, iz] +
                        wzl[ciz]*rx[:, iyp, izm] +
                        wzr[ciz]*rx[:, iyp, izp]
                )

        # Loop over coarse y-edges.
        for ciy in range(cnNy):
            iy = 2*ciy
            iym = max(0, iy-1)
            iyp = min(nNy-1, iy+1)

            # Sum the terms
            crz[:, ciy, :] = wy0[ciy]*(rz[:, iy, ::2] + rz[:, iy, 1::2])
            crz[:, ciy, :] += wyl[ciy]*(rz[:, iym, ::2] + rz[:, iym, 1::2])
            crz[:, ciy, :] += wyr[ciy]*(rz[:, iyp, ::2] + rz[:, iyp, 1::2])

    elif sc_dir == 2:  # Restrict in x- and z-directions

        # Loop over coarse z-edges.
        for ciz in range(cnNz):
            iz = 2*ciz
            izm = max(0, iz-1)
            izp = min(nNz-1, iz+1)

            # Sum the terms for x-field.
            crx[:, :, ciz] = wz0[ciz]*(rx[::2, :, iz] + rx[1::2, :, iz])
            crx[:, :, ciz] += wzl[ciz]*(rx[::2, :, izm] + rx[1::2, :, izm])
            crx[:, :, ciz] += wzr[ciz]*(rx[::2, :, izp] + rx[1::2, :, izp])

            # Loop over coarse x-edges.
            for cix in range(cnNx):
                ix = 2*cix
                ixm = max(0, ix-1)
                ixp = min(nNx-1, ix+1)

                # Sum the terms for y-field.
                cry[cix, :, ciz] = wx0[cix]*(
                        wz0[ciz]*ry[ix, :, iz] +
                        wzl[ciz]*ry[ix, :, izm] +
                        wzr[ciz]*ry[ix, :, izp]
                )

                cry[cix, :, ciz] += wxl[cix]*(
                        wz0[ciz]*ry[ixm, :, iz] +
                        wzl[ciz]*ry[ixm, :, izm] +
                        wzr[ciz]*ry[ixm, :, izp]
                )

                cry[cix, :, ciz] += wxr[cix]*(
                        wz0[ciz]*ry[ixp, :, iz] +
                        wzl[ciz]*ry[ixp, :, izm] +
                        wzr[ciz]*ry[ixp, :, izp]
                )

        # Loop over coarse x-edges.
        for cix in range(cnNx):
            ix = 2*cix
            ixm = max(0, ix-1)
            ixp = min(nNx-1, ix+1)

            # Sum the terms for z-field.
            crz[cix, :, :] = wx0[cix]*(rz[ix, :, ::2] + rz[ix, :, 1::2])
            crz[cix, :, :] += wxl[cix]*(rz[ixm, :, ::2] + rz[ixm, :, 1::2])
            crz[cix, :, :] += wxr[cix]*(rz[ixp, :, ::2] + rz[ixp, :, 1::2])

    elif sc_dir == 3:  # Restrict in x- and y-directions

        # Loop over coarse y-edges.
        for ciy in range(cnNy):
            iy = 2*ciy
            iym = max(0, iy-1)
            iyp = min(nNy-1, iy+1)

            # Sum the term for x-field.
            crx[:, ciy, :] = wy0[ciy]*(rx[::2, iy, :] + rx[1::2, iy, :])
            crx[:, ciy, :] += wyl[ciy]*(rx[::2, iym, :] + rx[1::2, iym, :])
            crx[:, ciy, :] += wyr[ciy]*(rx[::2, iyp, :] + rx[1::2, iyp, :])

            # Loop over coarse x-edges.
            for cix in range(cnNx):
                ix = 2*cix
                ixm = max(0, ix-1)
                ixp = min(nNx-1, ix+1)

                # Sum the terms for z-field.
                crz[cix, ciy, :] = wx0[cix]*(
                        wy0[ciy]*rz[ix, iy, :] +
                        wyl[ciy]*rz[ix, iym, :] +
                        wyr[ciy]*rz[ix, iyp, :]
                )

                crz[cix, ciy, :] += wxl[cix]*(
                        wy0[ciy]*rz[ixm, iy, :] +
                        wyl[ciy]*rz[ixm, iym, :] +
                        wyr[ciy]*rz[ixm, iyp, :]
                )

                crz[cix, ciy, :] += wxr[cix]*(
                        wy0[ciy]*rz[ixp, iy, :] +
                        wyl[ciy]*rz[ixp, iym, :] +
                        wyr[ciy]*rz[ixp, iyp, :]
                )

        # Loop over coarse x-edges.
        for cix in range(cnNx):
            ix = 2*cix
            ixm = max(0, ix-1)
            ixp = min(nNx-1, ix+1)

            # Sum the term for y-field.
            cry[cix, :, :] = wx0[cix]*(ry[ix, ::2, :] + ry[ix, 1::2, :])
            cry[cix, :, :] += wxl[cix]*(ry[ixm, ::2, :] + ry[ixm, 1::2, :])
            cry[cix, :, :] += wxr[cix]*(ry[ixp, ::2, :] + ry[ixp, 1::2, :])

    elif sc_dir == 4:  # Restrict in x-direction

        # Sum the terms for x-field.
        crx = rx[::2, :, :] + rx[1::2, :, :]

        # Loop over coarse x-edges.
        for cix in range(cnNx):
            ix = 2*cix
            ixm = max(0, ix-1)
            ixp = min(nNx-1, ix+1)

            # Sum the terms for y-field.
            cry[cix, :, :] = wx0[cix]*ry[ix, :, :]
            cry[cix, :, :] += wxl[cix]*ry[ixm, :, :]
            cry[cix, :, :] += wxr[cix]*ry[ixp, :, :]

            # Sum the terms for z-field.
            crz[cix, :, :] = wx0[cix]*rz[ix, :, :]
            crz[cix, :, :] += wxl[cix]*rz[ixm, :, :]
            crz[cix, :, :] += wxr[cix]*rz[ixp, :, :]

    elif sc_dir == 5:  # Restrict in y-direction

        # Sum the terms for y-field.
        cry = ry[:, ::2, :] + ry[:, 1::2, :]

        # Loop over coarse y-edges.
        for ciy in range(cnNy):
            iy = 2*ciy
            iym = max(0, iy-1)
            iyp = min(nNy-1, iy+1)

            # Sum the terms for x-field.
            crx[:, ciy, :] = wy0[ciy]*rx[:, iy, :]
            crx[:, ciy, :] += wyl[ciy]*rx[:, iym, :]
            crx[:, ciy, :] += wyr[ciy]*rx[:, iyp, :]

            # Sum the terms for z-field.
            crz[:, ciy, :] = wy0[ciy]*rz[:, iy, :]
            crz[:, ciy, :] += wyl[ciy]*rz[:, iym, :]
            crz[:, ciy, :] += wyr[ciy]*rz[:, iyp, :]

    elif sc_dir == 6:  # Restrict in z-direction

        # Sum the terms for z-field.
        crz = rz[:, :, ::2] + rz[:, :, 1::2]

        # Loop over coarse z-edges.
        for ciz in range(cnNz):
            iz = 2*ciz
            izm = max(0, iz-1)
            izp = min(nNz-1, iz+1)

            # Sum the terms for x-field.
            crx[:, :, ciz] = wz0[ciz]*rx[:, :, iz]
            crx[:, :, ciz] += wzl[ciz]*rx[:, :, izm]
            crx[:, :, ciz] += wzr[ciz]*rx[:, :, izp]

            # Sum the terms for y-field.
            cry[:, :, ciz] = wz0[ciz]*ry[:, :, iz]
            cry[:, :, ciz] += wzl[ciz]*ry[:, :, izm]
            cry[:, :, ciz] += wzr[ciz]*ry[:, :, izp]


@nb.njit(**_numba_setting)
def restrict_weights(vectorN, vectorCC, h, cvectorN, cvectorCC, ch):
    r"""Restriction weights for the coarse-grid correction operator.

    Corresponds to Equation 9 in [Muld06]_. A generalized version of that
    equation is given by

    .. math::

        w_{Q,-1}^v &= \left(v_{q-1/2}^h-v_{Q-1/2}^{2h}\right)/d_{q-1}^v ,\\
        w_{Q,0}^v  &= 1 ,\\
        w_{Q,1}^v  &= \left(v_{Q+1/2}^{2h}-v_{q+1/2}^h \right)/d_{q+1}^v ,

    where :math:`d` are the dual grid cell widths, :math:`v` is one of
    :math:`\{x, y, z\}`, and :math:`Q, q` the corresponding entries of
    :math:`\{K, L, M\}, \{k, l, m\}`. The superscripts :math:`h, 2h` indicate
    quantities defined on the coarse grid and on the fine grid, respectively.
    The indices :math:`\{K, L, M\}` on the coarse grid correspond to
    :math:`\{k, l, m\} = 2\{K, L, M\}` on the fine grid.

    For the dual volume cell widths at the boundaries the scheme of [MoSu94]_
    is applied, where :math:`d_0^x = h_{1/2}^x/2` at :math:`k = 0`,
    :math:`d_{N_x}^x = h_{N_x-1/2}^x` at :math:`k = N_x`, and so on.

    The following parameters must all be in the same direction, hence, all must
    be either for the x, the y, or the z direction. The returned weights are
    for this direction.

    Parameters
    ----------
    vectorN, cvectorN : ndarray
        Cell edges of the fine (vectorN) and coarse (cvectorN) grids.

    vectorCC, cvectorCC : ndarray
        Cell centers of the fine (vectorCC) and coarse (cvectorCC) grids.

    h, ch : ndarray
        Cell widths of the fine (h) and coarse (ch) grids.

    Returns
    -------
    wl, w0, wr : ndarray
        Left, central, and right weights in the direction provided in the
        input.

    """
    # Get length of weights
    n = len(cvectorN)

    # Dual grid cell widths
    d = np.empty(n+1)
    d[0] = h[0]/2
    d[-1] = h[-1]/2
    for i in range(1, n):
        d[i] = (h[2*i-2]+h[2*i-1])/2.

    # Left weight
    wl = 1/d[:-1]
    wl[0] *= (vectorN[0]-h[0]/2) - (cvectorN[0]-ch[0]/2)
    for i in range(1, n):
        wl[i] *= vectorCC[2*i-1]-cvectorCC[i-1]

    # Central weight
    w0 = np.ones(n)

    # Right weight
    wr = 1/d[1:]
    wr[-1] *= (cvectorN[-1]+ch[-1]/2) - (vectorN[-1]+h[-1]/2)
    for i in range(n-1):
        wr[i] *= cvectorCC[i]-vectorCC[2*i]

    return wl, w0, wr


# Simple wrapped functions
@nb.njit(**_numba_setting)
def l2norm(x):
    """Jitted version of np.linalg.norm(x, ord=None); l2-norm.

    Similar speed could be achieved with

    ``sp.linalg.get_blas_funcs('nrm2', dtype=x.dtype)(x)``

    or with

    ``sp.linalg.norm(x, check_finite=False)``

    if this PR gets merged: https://github.com/scipy/scipy/pull/10397

    """
    return np.linalg.norm(x)
