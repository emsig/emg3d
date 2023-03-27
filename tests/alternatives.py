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
# Copyright 2018-2023 The emsig community.
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

import emg3d
import numba as nb
import numpy as np

# Numba-settings
_numba_setting = {'nogil': True, 'fastmath': True, 'cache': True}


def alt_amat_x(rx, ry, rz, ex, ey, ez, eta_x, eta_y, eta_z, mu_r, hx, hy, hz):
    r"""Residual without the source term.

    Corresponds more or less to page 636 of [Muld06]_.

    Alternative `numpy`-version of the `numba`-version in ``core.amat_x``.
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

    Alternative of ``core.solve``, which uses a non-standard Cholesky
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

    Alternative of ``core.restrict_weights``.

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
    is much slower then the improved version by Joseph Capriotti.

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


def alt_get_source_field(grid, src, freq, strength=0):
    r"""Return the source field for point dipole sources."""
    # Cast some parameters.
    src = np.asarray(src, dtype=np.float64)
    strength = np.asarray(strength)

    # Ensure source is a point or a finite dipole.
    if len(src) != 5:
        raise ValueError("Source is wrong defined.")

    # Ensure source is within grid.
    outside = (src[0] < grid.nodes_x[0] or src[0] > grid.nodes_x[-1] or
               src[1] < grid.nodes_y[0] or src[1] > grid.nodes_y[-1] or
               src[2] < grid.nodes_z[0] or src[2] > grid.nodes_z[-1])
    if outside:
        raise ValueError(f"Provided source outside grid: {src}.")

    # Get source orientation (dxs, dys, dzs)
    h = np.cos(np.deg2rad(src[4]))
    dys = np.sin(np.deg2rad(src[3]))*h
    dxs = np.cos(np.deg2rad(src[3]))*h
    dzs = np.sin(np.deg2rad(src[4]))
    srcdir = np.array([dxs, dys, dzs])
    src = src[:3]

    # Set source strength.
    if strength == 0:  # 1 A m
        moment = srcdir
    else:              # Multiply source length with source strength
        moment = strength*srcdir

    def point_source(xx, yy, zz, src, s):
        """Set point dipole source."""
        nx, ny, nz = s.shape

        # Get indices of cells in which source resides.
        ix = max(0, np.where(src[0] < np.r_[xx, np.infty])[0][0]-1)
        iy = max(0, np.where(src[1] < np.r_[yy, np.infty])[0][0]-1)
        iz = max(0, np.where(src[2] < np.r_[zz, np.infty])[0][0]-1)

        def get_index_and_strength(ic, nc, csrc, cc):
            """Return index and field strength in c-direction."""
            if ic == nc-1:
                ic1 = ic
                rc = 1.0
                ec = 1.0
            else:
                ic1 = ic+1
                rc = (csrc-cc[ic])/(cc[ic1]-cc[ic])
                ec = 1.0-rc
            return rc, ec, ic1

        rx, ex, ix1 = get_index_and_strength(ix, nx, src[0], xx)
        ry, ey, iy1 = get_index_and_strength(iy, ny, src[1], yy)
        rz, ez, iz1 = get_index_and_strength(iz, nz, src[2], zz)

        s[ix, iy, iz] = ex*ey*ez
        s[ix1, iy, iz] = rx*ey*ez
        s[ix, iy1, iz] = ex*ry*ez
        s[ix1, iy1, iz] = rx*ry*ez
        s[ix, iy, iz1] = ex*ey*rz
        s[ix1, iy, iz1] = rx*ey*rz
        s[ix, iy1, iz1] = ex*ry*rz
        s[ix1, iy1, iz1] = rx*ry*rz

    # Initiate zero source field.
    sfield = emg3d.fields.Field(grid, frequency=freq)

    # Return source-field depending if point or finite dipole.
    vec1 = (grid.cell_centers_x, grid.nodes_y, grid.nodes_z)
    vec2 = (grid.nodes_x, grid.cell_centers_y, grid.nodes_z)
    vec3 = (grid.nodes_x, grid.nodes_y, grid.cell_centers_z)
    point_source(*vec1, src, sfield.fx)
    point_source(*vec2, src, sfield.fy)
    point_source(*vec3, src, sfield.fz)

    # Multiply by moment*s*mu in per direction.
    sfield.fx *= moment[0]*sfield.smu0
    sfield.fy *= moment[1]*sfield.smu0
    sfield.fz *= moment[2]*sfield.smu0

    # Add src and moment information.
    sfield.src = src
    sfield.strength = strength
    sfield.moment = moment

    return sfield


def alt_get_magnetic_field(model, efield):
    r"""Return magnetic field corresponding to provided electric field.

    Retrieve the magnetic field :math:`\mathbf{H}` from the electric field
    :math:`\mathbf{E}` using Farady's law, given by

    .. math::

        \nabla \times \mathbf{E} = \rm{i}\omega\mu\mathbf{H} .

    Note that the magnetic field is defined on the faces of the grid, or on the
    edges of the so-called dual grid. The grid of the returned magnetic field
    is the dual grid and has therefore one cell less in each direction.


    Parameters
    ----------
    model : Model
        The model; a :class:`emg3d.models.Model` instance.

    efield : Field
        The electric field; a :class:`emg3d.fields.Field` instance.


    Returns
    -------
    hfield : Field
        The magnetic field; a :class:`emg3d.fields.Field` instance.

    """
    grid = efield.grid

    # Carry out the curl (^ corresponds to differentiation axis):
    # H_x = (E_z^1 - E_y^2)
    e3d_hx = (np.diff(efield.fz, axis=1)/efield.grid.h[1][None, :, None] -
              np.diff(efield.fy, axis=2)/efield.grid.h[2][None, None, :])
    e3d_hx[0, :, :] = e3d_hx[-1, :, :] = 0

    # H_y = (E_x^2 - E_z^0)
    e3d_hy = (np.diff(efield.fx, axis=2)/efield.grid.h[2][None, None, :] -
              np.diff(efield.fz, axis=0)/efield.grid.h[0][:, None, None])
    e3d_hy[:, 0, :] = e3d_hy[:, -1, :] = 0

    # H_z = (E_y^0 - E_x^1)
    e3d_hz = (np.diff(efield.fy, axis=0)/efield.grid.h[0][:, None, None] -
              np.diff(efield.fx, axis=1)/efield.grid.h[1][None, :, None])
    e3d_hz[:, :, 0] = e3d_hz[:, :, -1] = 0

    # Divide by averaged relative magnetic permeability, if not not None.
    if model.mu_r is not None:

        # Get volume-averaged values.
        vmodel = emg3d.models.VolumeModel(model, efield)

        # Plus and minus indices.
        ixm = np.r_[0, np.arange(grid.shape_cells[0])]
        ixp = np.r_[np.arange(grid.shape_cells[0]), grid.shape_cells[0]-1]
        iym = np.r_[0, np.arange(grid.shape_cells[1])]
        iyp = np.r_[np.arange(grid.shape_cells[1]), grid.shape_cells[1]-1]
        izm = np.r_[0, np.arange(grid.shape_cells[2])]
        izp = np.r_[np.arange(grid.shape_cells[2]), grid.shape_cells[2]-1]

        # Average mu_r for dual-grid.
        zeta_x = (vmodel.zeta[ixm, :, :] + vmodel.zeta[ixp, :, :])/2.
        zeta_y = (vmodel.zeta[:, iym, :] + vmodel.zeta[:, iyp, :])/2.
        zeta_z = (vmodel.zeta[:, :, izm] + vmodel.zeta[:, :, izp])/2.

        hvx = grid.h[0][:, None, None]
        hvy = grid.h[1][None, :, None]
        hvz = grid.h[2][None, None, :]

        # Define the widths of the dual grid.
        dx = (np.r_[0., grid.h[0]] + np.r_[grid.h[0], 0.])/2.
        dy = (np.r_[0., grid.h[1]] + np.r_[grid.h[1], 0.])/2.
        dz = (np.r_[0., grid.h[2]] + np.r_[grid.h[2], 0.])/2.

        # Multiply fields by mu_r.
        e3d_hx *= zeta_x/(dx[:, None, None]*hvy*hvz)
        e3d_hy *= zeta_y/(hvx*dy[None, :, None]*hvz)
        e3d_hz *= zeta_z/(hvx*hvy*dz[None, None, :])

    new = np.r_[e3d_hx.ravel('F'), e3d_hy.ravel('F'), e3d_hz.ravel('F')]
    hfield = emg3d.Field(
            efield.grid, data=new, frequency=efield._frequency, electric=False)
    hfield.field /= efield.smu0

    # Return.
    return hfield


def fd_vs_as_gradient(ixyz, model, grad, data_misfit, sim_inp, epsilon=1e-4,
                      verb=2):
    """Compute FD gradient for cell ixyz and compare to AS gradient.

    We use the forward finite difference approach,

    Parameters
    ----------
    ixyz : list
        Indices of cell to test, [ix, iy, iz].

    model : Model
        The model; a :class:`emg3d.models.Model` instance.

    grad : ndarray
        Adjoint-state gradient for comparison.

    data_misfit : float
        Data misfit.

    sim_inp : dict
        Passed through to :class:`emg3d.simulations.Simulation`.

    epsilon : float, default:1e-4
        Difference to add to cell for finite-difference approach.

    verb : int, default: 2
        - 0: Nothing;
        - 1: Print result;
        - 2: Includes header line and result.

    Returns
    -------

    """
    ix, iy, iz = ixyz

    if verb > 1:
        print(f"   === Compare Gradients  ::  epsilon={epsilon} ===\n\n"
              f"{{xi;iy;iz}}     Adjoint-state       Forward FD    NRMSD (%)\n"
              f"----------------------------------------------------------")
    if verb > 0:
        print(
            f"{{{ix:2d};{iy:2d};{iz:2d}}}     {grad[ix, iy, iz]:+.6e}", end=''
        )

    # Add epsilon to given cell.
    model_diff = model.copy()
    model_diff.property_x[ix, iy, iz] += epsilon

    # Create simulation and compute FD-gradient
    sim_data = emg3d.Simulation(model=model_diff, **sim_inp, tqdm_opts=False)
    fdgrad = float((sim_data.misfit - data_misfit)/epsilon)

    # Compute NRMSD
    nrmsd = 200*abs(grad[ix, iy, iz]-fdgrad)
    nrmsd /= abs(grad[ix, iy, iz])+abs(fdgrad)
    if verb > 0:
        print(f"    {fdgrad:+.6e}    {nrmsd:9.5f}")

    return nrmsd
