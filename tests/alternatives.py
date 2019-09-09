"""

Alternatives
============

These are alternative versions to the ones implemented in the main code. For
instance a dense, neat numpy-version of a function that is implemented with
loops in numba in the main code.

They are kept here because they might become superior at some point (depending
how numba evolves) or simply for the record. They are useful in the tests as we
can use them for cross-checking with the main code. Also, they might (or might
not) be easier to understand.

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


def alt_amat_x(rx, ry, rz, ex, ey, ez, eta_x, eta_y, eta_z, mu_r, hx, hy, hz):
    r"""Residual without the source term.

    Corresponds more or less to page 636 of [Muld06]_.

    Alternative of ``njitted.amat_x``.
    """

    # Get dimensions
    nx = len(hx)
    ny = len(hy)
    nz = len(hz)

    ixm = np.r_[0, np.arange(nx)]
    ixp = np.r_[np.arange(nx), nx-1]
    iym = np.r_[0, np.arange(ny)]
    iyp = np.r_[np.arange(ny), ny-1]
    izm = np.r_[0, np.arange(nz)]
    izp = np.r_[np.arange(nz), nz-1]

    # Curl  [Muld06]_ equation 7:
    # v = nabla x E.
    v1 = ((ez[:, 1:, :] - ez[:, :-1, :])/hy[None, :, None] -
          (ey[:, :, 1:] - ey[:, :, :-1])/hz[None, None, :])
    v2 = ((ex[:, :, 1:] - ex[:, :, :-1])/hz[None, None, :] -
          (ez[1:, :, :] - ez[:-1, :, :])/hx[:, None, None])
    v3 = ((ey[1:, :, :] - ey[:-1, :, :])/hx[:, None, None] -
          (ex[:, 1:, :] - ex[:, :-1, :])/hy[None, :, None])

    # Multiply by average of mu_r [Muld06]_ p 636 bottom-left.
    # u = M v = V mu_r^-1 v = V mu_r^-1 nabla x E
    v1 *= 0.5*(mu_r[ixm, :, :] + mu_r[ixp, :, :])  # average of V/mu_r in x
    v2 *= 0.5*(mu_r[:, iym, :] + mu_r[:, iyp, :])  # average of V/mu_r in y
    v3 *= 0.5*(mu_r[:, :, izm] + mu_r[:, :, izp])  # average of V/mu_r in z

    # Another curl [Muld06]_ p. 636 bottom-right; completes:
    # nabla x V mu_r^-1 nabla x E
    hxr = hx.reshape((nx, 1, 1))
    hyr = hy.reshape((ny, 1))
    rx[:, :, 1:-1] -= (v2/hz)[:, :, 1:] - (v2/hz)[:, :, :-1]
    rx[:, 1:-1, :] += (v3/hyr)[:, 1:, :] - (v3/hyr)[:, :-1, :]
    ry[:, :, 1:-1] += (v1/hz)[:, :, 1:] - (v1/hz)[:, :, :-1]
    ry[1:-1, :, :] -= (v3/hxr)[1:, :, :] - (v3/hxr)[:-1, :, :]
    rz[1:-1, :, :] += (v2/hxr)[1:, :, :] - (v2/hxr)[:-1, :, :]
    rz[:, 1:-1, :] -= (v1/hyr)[:, 1:, :] - (v1/hyr)[:, :-1, :]

    # Sigma-term, [Muld06]_ p. 636 top-left (average of # eta).
    # S = i omega mu_0 sigma~ V
    stx = 0.25*(eta_x[:, iym, :][:, :, izm] + eta_x[:, iyp, :][:, :, izm] +
                eta_x[:, iym, :][:, :, izp] + eta_x[:, iyp, :][:, :, izp])

    # Average eta_y in x and z for ey
    sty = 0.25*(eta_y[ixm, :, :][:, :, izm] + eta_y[ixp, :, :][:, :, izm] +
                eta_y[ixm, :, :][:, :, izp] + eta_y[ixp, :, :][:, :, izp])

    # Average eta_z in x and y for e3
    stz = 0.25*(eta_z[ixm, :, :][:, iym, :] + eta_z[ixp, :, :][:, iym, :] +
                eta_z[ixm, :, :][:, iyp, :] + eta_z[ixp, :, :][:, iyp, :])

    # [Muld06]_ p. 636 center-right; completes
    # -V (i omega mu_0 sigma~ E - nabla x mu_r^-1 nabla x E)
    rx -= stx*ex
    ry -= sty*ey
    rz -= stz*ez
    # Subtracting this from the source terms will yield the
    # residual.


@nb.njit(**_numba_setting)
def alt_solve(amat, bvec):
    """Solve linear system A x = b for complex case.

    Tailored version for n=6 unknowns; A has length 36 ('F'-ordered), b has
    length 6. Solution x is put into b. A is replaced by LU decomposition.

    Does in a way the same as ``np.linalg.solve``, but a tad faster (for the
    particular use-case of this code).

    Alternative of ``njitted.solve``, which uses a non-standard Cholesky
    factorisation and is faster.

    Note that this requires the full matrix amat, not only the lower triangle.
    """
    n = 6
    ir = np.zeros(n, dtype=np.int8)
    jc = np.zeros(n, dtype=np.int8)
    pq = 0.
    pv = np.zeros(1, dtype=amat.dtype)[0]

    # 1. LU-decomposition
    for k in range(n):
        ir[k] = k
        jc[k] = k

        # Full pivoting: find biggest element in (sub)matrix.
        for j in range(k, n):
            for i in range(k, n):
                h = amat[i+6*j]
                hh = h.real*h.real + h.imag*h.imag
                if hh > pq:
                    pv = h
                    pq = hh
                    ir[k] = i
                    jc[k] = j

        # Interchange columns of A (change back at the end).
        j = jc[k]
        if j > k:
            for i in range(n):
                amat[i+6*k], amat[i+6*j] = amat[i+6*j], amat[i+6*k]

        # Interchange rows of A, b.
        i = ir[k]
        if i > k:
            for j in range(k, n):
                amat[k+6*j], amat[i+6*j] = amat[i+6*j], amat[k+6*j]

            bvec[k], bvec[i] = bvec[i], bvec[k]

        # Scale by pivot.
        if pq > 0.:
            pq = 1./pq
        pv *= pq

        for j in range(k+1, n):
            amat[k+6*j] *= pv.conjugate()
        bvec[k] *= pv.conjugate()

        # Subtract row k from rows i > k: A[i, j] -= A[i, k]*A[k, j].
        if k < n:
            for j in range(k+1, n):
                for i in range(k+1, n):
                    amat[i+6*j] -= amat[i+6*k]*amat[k+6*j]

            for i in range(k+1, n):  # b[i] -= A[i, k]*b[k]
                bvec[i] -= amat[i+6*k]*bvec[k]

    # 2. Back-solve for b.
    for i in range(n-2, -1, -1):
        for k in range(n-1, i, -1):  # b[i] -= A[i, k]*b[k].
            bvec[i] -= amat[i+6*k]*bvec[k]

    # 3. Interchange rows, back to original order.
    for k in range(n-1, -1, -1):
        i = jc[k]
        if i != k:
            bvec[k], bvec[i] = bvec[i], bvec[k]


def alt_restrict_weights(vectorN, vectorCC, h, cvectorN, cvectorCC, ch):
    r"""Restriction weights for the coarse-grid correction operator.

    Corresponds to Equation 9 in [Muld06]_.

    Alternative of ``njitted.restrict_weights``.

    """
    # Dual grid cell widths
    d = np.r_[h[0]/2, ((h[:-1]+h[1:])/2)[::2], h[-1]/2]

    # Left weight
    wl = np.r_[vectorN[0]-h[0]/2, vectorCC[1::2]]
    wl -= np.r_[cvectorN[0]-ch[0]/2, cvectorCC]
    wl /= d[:-1]

    # Central weight
    w0 = np.ones_like(wl)

    # Right weight
    wr = np.r_[cvectorCC, cvectorN[-1]+ch[-1]/2]
    wr -= np.r_[vectorCC[::2], vectorN[-1]+h[-1]/2]
    wr /= d[1:]

    return wl, w0, wr


@nb.njit(**_numba_setting)
def alt_volume_average(edges_x, edges_y, edges_z, values,
                       new_edges_x, new_edges_y, new_edges_z, new_values):
    """Interpolation using the volume averaging technique.

    This corresponds more or less to the version used by Mulder/Plessix, and
    is much slower then the improved version by Capriot.

    The result is added to new_values.


    Parameters
    ----------
    edges_[x, y, z] : ndarray
        The edges in x-, y-, and z-directions for the original grid.

    values : ndarray
        Values corresponding to ``grid``.

    new_edges_[x, y, z] : ndarray
        The edges in x-, y-, and z-directions for the new grid.

    new_values : ndarray
        Array where values corresponding to ``new_grid`` will be added.

    """

    # Get cell indices.
    # First and last edges ignored => first and last cells extend to +/- infty.
    ix_l = np.searchsorted(edges_x[1:-1], new_edges_x, 'left')
    ix_r = np.searchsorted(edges_x[1:-1], new_edges_x, 'right')
    iy_l = np.searchsorted(edges_y[1:-1], new_edges_y, 'left')
    iy_r = np.searchsorted(edges_y[1:-1], new_edges_y, 'right')
    iz_l = np.searchsorted(edges_z[1:-1], new_edges_z, 'left')
    iz_r = np.searchsorted(edges_z[1:-1], new_edges_z, 'right')

    # Get number of cells.
    ncx = len(new_edges_x)-1
    ncy = len(new_edges_y)-1
    ncz = len(new_edges_z)-1

    # Working arrays for edges.
    x_edges = np.empty(len(edges_x)+2)
    y_edges = np.empty(len(edges_y)+2)
    z_edges = np.empty(len(edges_z)+2)

    # Loop over new_grid cells.
    for iz in range(ncz):
        hz = np.diff(new_edges_z[iz:iz+2])[0]  # To calc. current cell volume.

        for iy in range(ncy):
            hyz = hz*np.diff(new_edges_y[iy:iy+2])[0]  # " "

            for ix in range(ncx):
                hxyz = hyz*np.diff(new_edges_x[ix:ix+2])[0]  # " "

                # Get start edge and number of cells of original grid involved.
                s_cx = ix_r[ix]
                n_cx = ix_l[ix+1] - s_cx

                s_cy = iy_r[iy]
                n_cy = iy_l[iy+1] - s_cy

                s_cz = iz_r[iz]
                n_cz = iz_l[iz+1] - s_cz

                # Get the involved original grid edges for this cell.
                x_edges[0] = new_edges_x[ix]
                for i in range(n_cx):
                    x_edges[i+1] = edges_x[s_cx+i+1]
                x_edges[n_cx+1] = new_edges_x[ix+1]

                y_edges[0] = new_edges_y[iy]
                for j in range(n_cy):
                    y_edges[j+1] = edges_y[s_cy+j+1]
                y_edges[n_cy+1] = new_edges_y[iy+1]

                z_edges[0] = new_edges_z[iz]
                for k in range(n_cz):
                    z_edges[k+1] = edges_z[s_cz+k+1]
                z_edges[n_cz+1] = new_edges_z[iz+1]

                # Loop over each (partial) cell of the original grid which
                # contributes to the current cell of the new grid and add its
                # (partial) value.
                for k in range(n_cz+1):
                    dz = np.diff(z_edges[k:k+2])[0]
                    k += s_cz

                    for j in range(n_cy+1):
                        dyz = dz*np.diff(y_edges[j:j+2])[0]
                        j += s_cy

                        for i in range(n_cx+1):
                            dxyz = dyz*np.diff(x_edges[i:i+2])[0]
                            i += s_cx

                            # Add this cell's contribution.
                            new_values[ix, iy, iz] += values[i, j, k]*dxyz

                # Normalize by new_grid-cell volume.
                new_values[ix, iy, iz] /= hxyz
