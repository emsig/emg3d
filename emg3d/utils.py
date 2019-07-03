"""

:mod:`utils` -- Utilities
=========================

Utility functions for the multigrid solver.

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


import os
import shelve
import numpy as np
from timeit import default_timer
from datetime import datetime, timedelta
from scipy import optimize, interpolate, ndimage

# scooby is a soft dependency for emg3d
try:
    from scooby import Report as ScoobyReport
except ImportError:
    class ScoobyReport:
        def __init__(self, additional, core, optional, ncol, text_width, sort):
            print("\n* WARNING :: `emg3d.Report` requires `scooby`."
                  "\n             Install it via `pip install scooby`.\n")

__all__ = ['Model', 'Field', 'get_domain', 'get_stretched_h', 'get_hx',
           'get_source_field', 'get_receiver', 'get_h_field', 'TensorMesh',
           'Time', 'data_write', 'data_read', 'Report']


# CONSTANTS
c = 299792458              # Speed of light m/s
mu_0 = 4e-7*np.pi          # Magn. permeability of free space [H/m]
epsilon_0 = 1./(mu_0*c*c)  # Elec. permittivity of free space [F/m]


# HELPER FUNCTIONS TO CREATE MESH => These will probably move to discretize.
def get_domain(x0=0, freq=1, rho=0.3, limits=None, min_width=None,
               fact_min=0.2, fact_neg=5, fact_pos=None):
    r"""Get domain extent and minimum cell width as a function of skin depth.

    Returns the extent of the calculation domain and the minimum cell width as
    a multiple of the skin depth, with possible user restrictions on minimum
    calculation domain and range of possible minimum cell widths.

    .. math::

            \delta &= 503.3 \sqrt{\frac{\rho}{f}} , \\
            x_\text{start} &= x_0-k_\text{neg}\delta , \\
            x_\text{end} &= x_0+k_\text{pos}\delta , \\
            h_\text{min} &= k_\text{min} \delta .

    .. note::

        This utility will probably move into discretize in the future.


    Parameters
    ----------

    x0 : float
        Center of the calculation domain. Normally the source location.
        Default is 0.

    freq : float
        Frequency (Hz) to calculate the skin depth.
        Default is 1 Hz.

    rho : float, optional
        Resistivity (Ohm m) to calculate skin depth.
        Default is 0.3 Ohm m (sea water).

    limits : None or list
        [start, end] of model domain. This extent represents the minimum extent
        of the domain. The domain is therefore only adjusted if it has to reach
        outside of [start, end].
        Default is None.

    min_width : None, float, or list of two floats
        Minimum cell width is calculated as a function of skin depth:
        fact_min*sd. If ``min_width`` is a float, this is used. If a list of
        two values [min, max] are provided, they are used to restrain
        min_width. Default is None.

    fact_min, fact_neg, fact_pos : floats
        The skin depth is multiplied with these factors to estimate:

            - Minimum cell width (``fact_min``, default 0.2)
            - Domain-start (``fact_neg``, default 5), and
            - Domain-end (``fact_pos``, defaults to ``fact_neg``).


    Returns
    -------

    h_min : float
        Minimum cell width.

    domain : list
        Start- and end-points of calculation domain.

    """

    # Set fact_pos to fact_neg if not provided.
    if fact_pos is None:
        fact_pos = fact_neg

    # Calculate the skin depth.
    skind = 503.3*np.sqrt(rho/freq)

    # Estimate minimum cell width.
    h_min = fact_min*skind
    if min_width is not None:  # Respect user input.
        if np.array(min_width).size == 1:
            h_min = min_width
        else:
            h_min = np.clip(h_min, *min_width)

    # Estimate calculation domain.
    domain = [x0-fact_neg*skind, x0+fact_pos*skind]
    if limits is not None:  # Respect user input.
        domain = [min(limits[0], domain[0]), max(limits[1], domain[1])]

    return h_min, domain


def get_stretched_h(h_min, domain, nx, x0=0, x1=None, resp_domain=False):
    """Return cell widths for a stretched grid within the domain.

    Returns ``nx`` cell widths within ``domain``, where the minimum cell width
    is ``h_min``. The cells are not stretched within ``x0`` and ``x1``, and
    outside uses a power-law stretching. The actual stretching factor and the
    number of cells left and right of ``x0`` and ``x1`` are find in a
    minimization process.

    The domain is not completely respected. The starting point of the domain
    is, but the endpoint of the domain might slightly shift (this is more
    likely the case for small ``nx``, for big ``nx`` the shift should be
    small). The new endpoint can be obtained with ``domain[0]+np.sum(hx)``. If
    you want the domain to be respected absolutely, set ``resp_domain=True``.
    However, be aware that this will introduce one stretch-factor which is
    different from the other stretch factors, to accommodate the restriction.
    This one-off factor is between the left- and right-side of ``x0``, or, if
    ``x1`` is provided, just after ``x1``.

    .. note::

        This utility will probably move into discretize in the future.


    Parameters
    ----------

    h_min : float
        Minimum cell width.

    domain : list
        [start, end] of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. ``x0`` is restricted to ``domain``.
        Default is 0.

    x1 : float
        If provided, then no stretching is applied between ``x0`` and ``x1``.
        The non-stretched part starts at ``x0`` and stops at the first possible
        location at or after ``x1``. ``x1`` is restricted to ``domain``.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of ``x0``, or, if ``x1`` is provided, just after ``x1``.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """

    # Cast to arrays
    domain = np.array(domain, dtype=float)
    x0 = np.array(x0, dtype=float)
    x0 = np.clip(x0, *domain)  # Restrict to model domain
    if x1 is not None:
        x1 = np.array(x1, dtype=float)
        x1 = np.clip(x1, *domain)  # Restrict to model domain

    # If x1 is provided (a part is not stretched)
    if x1 is not None:

        # Store original values
        xlim_orig = domain.copy()
        nx_orig = int(nx)
        x0_orig = x0.copy()

        # Get number of non-stretched cells
        n_nos = int(np.ceil((x1-x0)/h_min))-1
        # Note that wee subtract one cell, because the standard scheme provides
        # one h_min-cell.

        # Reset x0, because the first h_min comes from normal scheme
        x0 += h_min

        # Reset xmax for normal scheme
        domain[1] -= n_nos*h_min

        # Reset nx for normal scheme
        nx -= n_nos

        # If there are not enough points reset to standard procedure
        # This five is arbitrary. However, nx should be much bigger than five
        # anyways, otherwise stretched grid doesn't make sense.
        if nx <= 5:
            print("Warning :: Not enough points for non-stretched part,"
                  "ignoring therefore `x1`.")
            domain = xlim_orig
            nx = nx_orig
            x0 = x0_orig
            x1 = None

    # Get stretching factor (a = 1+alpha).
    if h_min == 0 or h_min > np.diff(domain)/nx:
        # If h_min is bigger than the domain-extent divided by nx, no
        # stretching is required at all.
        alpha = 0
    else:

        # Wrap _get_dx into a minimization function to call with fsolve.
        def find_alpha(alpha, h_min, args):
            """Find alpha such that min(hx) = h_min."""
            return min(get_hx(alpha, *args))/h_min-1

        # Search for best alpha, must be at least 0
        args = (domain, nx, x0)
        alpha = max(0, optimize.fsolve(find_alpha, 0.02, (h_min, args)))

    # With alpha get actual cell spacing with `resp_domain` to respect the
    # users decision.
    hx = get_hx(alpha, domain, nx, x0, resp_domain)

    # Add the non-stretched center if x1 is provided
    if x1 is not None:
        hx = np.r_[hx[: np.argmin(hx)], np.ones(n_nos)*h_min,
                   hx[np.argmin(hx):]]

    # Print warning h_min could not be respected.
    if abs(hx.min() - h_min) > 0.1:
        print(f"Warning :: Minimum cell width ({np.round(hx.min(), 2)} m) is "
              "below `h_min`, because `nx` is too big for `domain`.")

    return hx


def get_hx(alpha, domain, nx, x0, resp_domain=True):
    r"""Return cell widths for given input.

    Find the number of cells left and right of ``x0``, ``nl`` and ``nr``
    respectively, for the provided alpha. For this, we solve

    .. math::   \frac{x_\text{max}-x_0}{x_0-x_\text{min}} =
                \frac{a^{nr}-1}{a^{nl}-1}

    where :math:`a = 1+\alpha`.

    .. note::

        This utility will probably move into discretize in the future.


    Parameters
    ----------

    alpha : float
        Stretching factor ``a`` is given by ``a=1+alpha``.

    domain : list
        [start, end] of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. ``x0`` is restricted to ``domain``.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of ``x0``, or, if ``x1`` is provided, just after ``x1``.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """
    if alpha <= 0.:  # If alpha <= 0: equal spacing (no stretching at all)
        hx = np.ones(nx)*np.diff(np.squeeze(domain))/nx

    else:            # Get stretched hx
        a = alpha+1

        # Get hx depending if x0 is on the domain boundary or not.
        if x0 == domain[0] or x0 == domain[1]:
            # Get al a's
            alr = np.diff(domain)*alpha/(a**nx-1)*a**np.arange(nx)
            if x0 == domain[1]:
                alr = alr[::-1]

            # Calculate differences
            hx = alr*np.diff(domain)/sum(alr)

        else:
            # Find number of elements left and right by solving:
            #     (xmax-x0)/(x0-xmin) = a**nr-1/(a**nl-1)
            nr = np.arange(2, nx+1)
            er = (domain[1]-x0)/(x0-domain[0]) - (a**nr[::-1]-1)/(a**nr-1)
            nl = np.argmin(abs(np.floor(er)))+1
            nr = nx-nl

            # Get all a's
            al = a**np.arange(nl-1, -1, -1)
            ar = a**np.arange(1, nr+1)

            # Calculate differences
            if resp_domain:
                # This version honours domain[0] and domain[1], but to achieve
                # this it introduces one stretch-factor which is different from
                # all the others between al to ar.
                hx = np.r_[al*(x0-domain[0])/sum(al),
                           ar*(domain[1]-x0)/sum(ar)]
            else:
                # This version moves domain[1], but each stretch-factor is
                # exactly the same.
                fact = (x0-domain[0])/sum(al)  # Take distance from al.
                hx = np.r_[al, ar]*fact

                # Note: this hx is equivalent as providing the following h
                # to TensorMesh:
                # h = [(h_min, nl-1, -a), (h_min, n_nos+1), (h_min, nr, a)]

    return hx


def get_source_field(grid, src, freq, strength=0):
    r"""Return the source field.

    The source field is given in Equation 2 in [Muld06]_,

    .. math::

        \mathrm{i} \omega \mu_0 \mathbf{J}_\mathrm{s} .

    Either finite length dipoles or infinitesimal small point dipoles can be
    defined, whereas the return source field corresponds to a normalized (1 Am)
    source distributed within the cell(s) it resides (can be changed with the
    ``strength``-parameter).


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a ``TensorMesh``-instance.

    src : list of floats
        Source coordinates (m). There are two formats:

          - Finite length dipole: ``[x0, x1, y0, y1, z0, z1]``.
          - Point dipole: ``[x, y, z, azimuth, dip]``.

    freq : float
        Source frequency (Hz).

    strength : float, optional
        Source strength (A):

          - If 0, output is normalized to a source of 1 m length, and source
            strength of 1 A.
          - If != 0, output is returned for given source length and strength.

        Default is 0.


    Returns
    -------
    sfield : :func:`Field`-instance
        Source field, normalized to 1 A m.

    """
    # Cast some parameters
    src = np.array(src, dtype=float)
    strength = float(strength)

    # Ensure source is a point or a finite dipole
    if len(src) not in [5, 6]:
        print("* ERROR   :: Source is wrong defined. Must be either a point,\n"
              "             [x, y, z, azimuth, dip], or a finite dipole,\n"
              "             [x1, x2, y1, y2, z1, z2]. Provided source:\n"
              f"             {src}.")
        raise ValueError("Source error")
    elif len(src) == 5:
        finite = False  # Infinitesimal small dipole.
    else:
        finite = True   # Finite length dipole.

        # Ensure finite length dipole is not a point dipole.
        if np.allclose(np.linalg.norm(src[1::2]-src[::2]), 0):
            print("* ERROR   :: Provided source is a point dipole, "
                  "use the format [x, y, z, azimuth, dip] instead.")
            raise ValueError("Source error")

    # Ensure source is within grid
    if finite:
        ii = [0, 1, 2, 3, 4, 5]
    else:
        ii = [0, 0, 1, 1, 2, 2]

    source_in = np.any(src[ii[0]] >= grid.vectorNx[0])
    source_in *= np.any(src[ii[1]] <= grid.vectorNx[-1])
    source_in *= np.any(src[ii[2]] >= grid.vectorNy[0])
    source_in *= np.any(src[ii[3]] <= grid.vectorNy[-1])
    source_in *= np.any(src[ii[4]] >= grid.vectorNz[0])
    source_in *= np.any(src[ii[5]] <= grid.vectorNz[-1])

    if not source_in:
        print(f"* ERROR   :: Provided source outside grid: {src}.")
        raise ValueError("Source error")

    # Get source orientation (dxs, dys, dzs)
    if not finite:  # Point dipole: convert azimuth/dip to weights.
        h = np.cos(np.deg2rad(src[4]))
        dys = np.sin(np.deg2rad(src[3]))*h
        dxs = np.cos(np.deg2rad(src[3]))*h
        dzs = np.sin(np.deg2rad(src[4]))
        srcdir = np.array([dxs, dys, dzs])
        src = src[:3]

    else:           # Finite dipole: get length and normalize.
        srcdir = np.diff(src.reshape(3, 2)).ravel()

        # Normalize to one if strength is 0.
        if strength == 0:
            srcdir /= np.linalg.norm(srcdir)

    # Set source strength.
    if strength == 0:  # 1 A m
        strength = srcdir
    else:              # Multiply source length with source strength
        strength *= srcdir

    def set_source(grid, strength, finite):
        """Set the source-field in idir."""

        # Initiate zero source field.
        sfield = Field(grid)

        # Return source-field depending if point or finite dipole.
        vec1 = (grid.vectorCCx, grid.vectorNy, grid.vectorNz)
        vec2 = (grid.vectorNx, grid.vectorCCy, grid.vectorNz)
        vec3 = (grid.vectorNx, grid.vectorNy, grid.vectorCCz)
        if finite:
            finite_source(*vec1, src, sfield.fx, 0, grid)
            finite_source(*vec2, src, sfield.fy, 1, grid)
            finite_source(*vec3, src, sfield.fz, 2, grid)
        else:
            point_source(*vec1, src, sfield.fx)
            point_source(*vec2, src, sfield.fy)
            point_source(*vec3, src, sfield.fz)

        # Multiply by strength*i*omega*mu in per direction.
        sfield.fx *= strength[0]*2j*np.pi*freq*mu_0
        sfield.fy *= strength[1]*2j*np.pi*freq*mu_0
        sfield.fz *= strength[2]*2j*np.pi*freq*mu_0

        return sfield

    def point_source(xx, yy, zz, src, s):
        """Set point dipole source."""
        nx, ny, nz = s.shape

        # Get indices of cells in which source resides.
        ix = max(0, np.where(src[0] < np.r_[xx, np.infty])[0][0]-1)
        iy = max(0, np.where(src[1] < np.r_[yy, np.infty])[0][0]-1)
        iz = max(0, np.where(src[2] < np.r_[zz, np.infty])[0][0]-1)

        # Indices and field strength in x-direction
        if ix == nx-1:
            rx = 1.0
            ex = 1.0
            ix1 = ix
        else:
            ix1 = ix+1
            rx = (src[0]-xx[ix])/(xx[ix1]-xx[ix])
            ex = 1.0-rx

        # Indices and field strength in y-direction
        if iy == ny-1:
            ry = 1.0
            ey = 1.0
            iy1 = iy
        else:
            iy1 = iy+1
            ry = (src[1]-yy[iy])/(yy[iy1]-yy[iy])
            ey = 1.0-ry

        # Indices and field strength in z-direction
        if iz == nz-1:
            rz = 1.0
            ez = 1.0
            iz1 = iz
        else:
            iz1 = iz+1
            rz = (src[2]-zz[iz])/(zz[iz1]-zz[iz])
            ez = 1.0-rz

        s[ix, iy, iz] = ex*ey*ez
        s[ix1, iy, iz] = rx*ey*ez
        s[ix, iy1, iz] = ex*ry*ez
        s[ix1, iy1, iz] = rx*ry*ez
        s[ix, iy, iz1] = ex*ey*rz
        s[ix1, iy, iz1] = rx*ey*rz
        s[ix, iy1, iz1] = ex*ry*rz
        s[ix1, iy1, iz1] = rx*ry*rz

    def finite_source(xx, yy, zz, src, s, idir, grid):
        """Set finite dipole source.

        Using adjoint interpolation method, probably not the most efficient
        implementation.
        """
        # Source lengths in x-, y-, and z-directions.
        d_xyz = src[1::2]-src[::2]

        # Inverse source lengths.
        id_xyz = d_xyz.copy()
        id_xyz[id_xyz != 0] = 1/id_xyz[id_xyz != 0]

        # Cell fractions.
        a1 = (grid.vectorNx-src[0])*id_xyz[0]
        a2 = (grid.vectorNy-src[2])*id_xyz[1]
        a3 = (grid.vectorNz-src[4])*id_xyz[2]

        # Get range of indices of cells in which source resides.
        def min_max_ind(vector, i):
            """Return [min, max]-index of cells in which source resides."""
            vmin = min(src[2*i:2*i+2])
            vmax = max(src[2*i:2*i+2])
            return [max(0, np.where(vmin < np.r_[vector, np.infty])[0][0]-1),
                    max(0, np.where(vmax < np.r_[vector, np.infty])[0][0]-1)]

        rix = min_max_ind(grid.vectorNx, 0)
        riy = min_max_ind(grid.vectorNy, 1)
        riz = min_max_ind(grid.vectorNz, 2)

        # Loop over these indices.
        for iz in range(riz[0], riz[1]+1):
            for iy in range(riy[0], riy[1]+1):
                for ix in range(rix[0], rix[1]+1):

                    # Determine centre of gravity of line segment in cell.
                    aa = np.vstack([[a1[ix], a1[ix+1]], [a2[iy], a2[iy+1]],
                                   [a3[iz], a3[iz+1]]])
                    aa = np.sort(aa[d_xyz != 0, :], 1)
                    al = max(0, aa[:, 0].max())  # Left and right
                    ar = min(1, aa[:, 1].min())  # elements.

                    # Characteristics of this cell.
                    xmin = src[::2]+al*d_xyz
                    xmax = src[::2]+ar*d_xyz
                    x_c = (xmin+xmax)/2.0
                    slen = np.linalg.norm(src[1::2]-src[::2])
                    x_len = np.linalg.norm(xmax-xmin)/slen

                    # Contribution to edge (coordinate idir)
                    rx = (x_c[0]-grid.vectorNx[ix])/grid.hx[ix]
                    ex = 1-rx
                    ry = (x_c[1]-grid.vectorNy[iy])/grid.hy[iy]
                    ey = 1-ry
                    rz = (x_c[2]-grid.vectorNz[iz])/grid.hz[iz]
                    ez = 1-rz

                    # Add to field (only if segment inside cell).
                    if min(rx, ry, rz) >= 0 and np.max(np.abs(ar-al)) > 0:

                        if idir == 0:
                            s[ix, iy, iz] += ey*ez*x_len
                            s[ix, iy+1, iz] += ry*ez*x_len
                            s[ix, iy, iz+1] += ey*rz*x_len
                            s[ix, iy+1, iz+1] += ry*rz*x_len
                        if idir == 1:
                            s[ix, iy, iz] += ex*ez*x_len
                            s[ix+1, iy, iz] += rx*ez*x_len
                            s[ix, iy, iz+1] += ex*rz*x_len
                            s[ix+1, iy, iz+1] += rx*rz*x_len
                        if idir == 2:
                            s[ix, iy, iz] += ex*ey*x_len
                            s[ix+1, iy, iz] += rx*ey*x_len
                            s[ix, iy+1, iz] += ex*ry*x_len
                            s[ix+1, iy+1, iz] += rx*ry*x_len

    # Return the source field.
    return set_source(grid, strength, finite)


class TensorMesh:
    """Rudimentary mesh for multigrid calculation.

    The tensor-mesh :class:`discretize.TensorMesh` is a powerful tool,
    including sophisticated mesh-generation possibilities in 1D, 2D, and 3D,
    plotting routines, and much more. However, in the multigrid solver we have
    to generate a mesh at each level, many times over and over again, and we
    only need a very limited set of attributes. This tensor-mesh class provides
    all required attributes. All attributes here are the same as their
    counterparts in :class:`discretize.TensorMesh` (both in name and value).

    .. warning::
        This is a slimmed-down version of :class:`discretize.TensorMesh`, meant
        principally for internal use by the multigrid modeller. It is highly
        recommended to use :class:`discretize.TensorMesh` to create the input
        meshes instead of this class. There are no input-checks carried out
        here, and there is only one accepted input format for ``h`` and ``x0``.


    Parameters
    ----------
    h : list of three ndarrays
        Cell widths in [x, y, z] directions.

    x0 : ndarray of dimension (3, )
        Origin (x, y, z).

    """

    def __init__(self, h, x0):
        """Initialize the mesh."""
        self.x0 = x0

        # Width of cells.
        self.hx = h[0]
        self.hy = h[1]
        self.hz = h[2]

        # Cell related properties.
        self.nCx = int(self.hx.size)
        self.nCy = int(self.hy.size)
        self.nCz = int(self.hz.size)
        self.vnC = np.array([self.hx.size, self.hy.size, self.hz.size])
        self.nC = int(self.vnC.prod())
        self.vectorCCx = np.r_[0, self.hx[:-1].cumsum()]+self.hx*0.5+self.x0[0]
        self.vectorCCy = np.r_[0, self.hy[:-1].cumsum()]+self.hy*0.5+self.x0[1]
        self.vectorCCz = np.r_[0, self.hz[:-1].cumsum()]+self.hz*0.5+self.x0[2]

        # Node related properties.
        self.nNx = self.nCx + 1
        self.nNy = self.nCy + 1
        self.nNz = self.nCz + 1
        self.vnN = np.array([self.nNx, self.nNy, self.nNz], dtype=int)
        self.nN = int(self.vnN.prod())
        self.vectorNx = np.r_[0., self.hx.cumsum()] + self.x0[0]
        self.vectorNy = np.r_[0., self.hy.cumsum()] + self.x0[1]
        self.vectorNz = np.r_[0., self.hz.cumsum()] + self.x0[2]

        # Edge related properties.
        self.vnEx = np.array([self.nCx, self.nNy, self.nNz], dtype=int)
        self.vnEy = np.array([self.nNx, self.nCy, self.nNz], dtype=int)
        self.vnEz = np.array([self.nNx, self.nNy, self.nCz], dtype=int)
        self.nEx = int(self.vnEx.prod())
        self.nEy = int(self.vnEy.prod())
        self.nEz = int(self.vnEz.prod())
        self.vnE = np.array([self.nEx, self.nEy, self.nEz], dtype=int)
        self.nE = int(self.vnE.sum())


# RELATED TO MODELS AND FIELDS
class Model:
    r"""Create a resistivity model.

    Class to provide model parameters (x-, y-, and z-directed resistivities) to
    the solver. Relative magnetic permeability :math:`\mu_r` is by default set
    to one, but can be provided (isotropically). Relative electric permittivity
    :math:`\varepsilon_r` is fixed at 1, as ``emg3d`` uses the diffusive
    approximation of Maxwell's equations.


    Parameters
    ----------
    grid : TensorMesh
        Grid on which to apply model.

    res_x, res_y, res_z : float or ndarray; default to 1.
        Resistivity in x-, y-, and z-directions. If ndarray, they must have the
        shape of grid.vnC (F-ordered) or grid.nC.

    freq : float
        Frequency.

    mu_r : float or ndarray
       Relative magnetic permeability (isotropic). If ndarray it must have the
       shape of grid.vnC (F-ordered) or grid.nC.

    """

    def __init__(self, grid, res_x=1., res_y=None, res_z=None, freq=1.,
                 mu_r=None):
        """Initiate a new resistivity model."""

        # Store frequency.
        self.freq = freq

        # Store required info from grid.
        self.nC = grid.nC
        self.vnC = grid.vnC

        # Construct cell volumes of the 3D model as 1D array.
        if hasattr(grid, 'vol'):  # If discretize-grid, take it from there.
            self.__vol = grid.vol.reshape(self.vnC, order='F')
        else:                     # Calculate it otherwise.
            vol = np.outer(np.outer(grid.hx, grid.hy).ravel('F'), grid.hz)
            self.__vol = vol.ravel('F').reshape(self.vnC, order='F')

        # Check case.
        if res_y is None and res_z is None:   # Isotropic (0).
            self.case = 0
        elif res_y is None or res_z is None:  # HTI (1) or VTI (2).
            if res_z is None:
                self.case = 1
            else:
                self.case = 2
        else:                                 # Tri-axial anisotropy (3).
            self.case = 3

        # Initiate x-directed resistivity.
        if isinstance(res_x, (float, int)):
            self.__res_x = res_x*np.ones(self.vnC)
        elif np.all(res_x.shape == self.vnC) and res_x.ndim == 3:
            self.__res_x = res_x
        elif res_x.size == self.nC and res_x.ndim == 1:
            self.__res_x = res_x.reshape(self.vnC, order='F')
        else:
            print(f"* ERROR   :: res_x must be {grid.vnC} or {grid.nC}.")
            print(f"             Provided: {res_x.shape}.")
            raise ValueError("Wrong Shape")
        self.__eta_x = self._calculate_eta(self.__res_x)

        # Initiate y-directed resistivity.
        if self.case in [1, 3]:
            if isinstance(res_y, (float, int)):
                self.__res_y = res_y*np.ones(self.vnC)
            elif np.all(res_y.shape == self.vnC) and res_y.ndim == 3:
                self.__res_y = res_y
            elif res_y.size == self.nC and res_y.ndim == 1:
                self.__res_y = res_y.reshape(self.vnC, order='F')
            else:
                print(f"* ERROR   :: res_y must be {grid.vnC} or {grid.nC}.")
                print(f"             Provided: {res_y.shape}.")
                raise ValueError("Wrong Shape")
            self.__eta_y = self._calculate_eta(self.__res_y)

        # Initiate z-directed resistivity.
        if self.case in [2, 3]:
            if isinstance(res_z, (float, int)):
                self.__res_z = res_z*np.ones(self.vnC)
            elif np.all(res_z.shape == self.vnC) and res_z.ndim == 3:
                self.__res_z = res_z
            elif res_z.size == self.nC and res_z.ndim == 1:
                self.__res_z = res_z.reshape(self.vnC, order='F')
            else:
                print(f"* ERROR   :: res_z must be {grid.vnC} or {grid.nC}.")
                print(f"             Provided: {res_z.shape}.")
                raise ValueError("Wrong Shape")
            self.__eta_z = self._calculate_eta(self.__res_z)

        # Store magnetic permeability.
        if mu_r is None or isinstance(mu_r, (float, int)):
            self.__mu_r = mu_r
        elif np.all(mu_r.shape == self.vnC) and mu_r.ndim == 3:
            self.__mu_r = mu_r
        elif mu_r.size == self.nC and mu_r.ndim == 1:
            self.__mu_r = mu_r.reshape(self.vnC, order='F')

    # RESISTIVITIES
    @property
    def res_x(self):
        r"""Resistivity in x-direction (:math:`\rho_x`)."""
        return self.__res_x

    @res_x.setter
    def res_x(self, res):
        r"""Update resistivity in x-direction (:math:`\rho_x`)."""
        self.__res_x = res
        self.__eta_x = self._calculate_eta(res)

    @property
    def res_y(self):
        r"""Resistivity in y-direction (:math:`\rho_y`)."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            return self.__res_y
        else:                    # Return res_x.
            return self.__res_x

    @res_y.setter
    def res_y(self, res):
        r"""Update resistivity in y-direction (:math:`\rho_y`)."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            self.__res_y = res
            self.__eta_y = self._calculate_eta(res)
        else:
            print("Cannot set res_y, as it was initialized as res_x.")
            raise ValueError

    @property
    def res_z(self):
        r"""Resistivity in z-direction (:math:`\rho_z`)."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            return self.__res_z
        else:                    # Return res_x.
            return self.__res_x

    @res_z.setter
    def res_z(self, res):
        r"""Update resistivity in z-direction (:math:`\rho_z`)."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            self.__res_z = res
            self.__eta_z = self._calculate_eta(res)
        else:
            print("Cannot set res_z, as it was initialized as res_x.")
            raise ValueError

    # ETA's
    @property
    def eta_x(self):
        r"""Volume*eta in x-direction (:math:`V\eta_x`)."""
        return self.__eta_x

    @property
    def eta_y(self):
        r"""Volume*eta in x-direction (:math:`V\eta_y`)."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            return self.__eta_y
        else:                    # Return eta_x.
            return self.__eta_x

    @property
    def eta_z(self):
        r"""Volume*eta in x-direction (:math:`V\eta_z`)."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            return self.__eta_z
        else:                    # Return eta_x.
            return self.__eta_x

    def _calculate_eta(self, res):
        r"""Calculate vol*eta (:math:`V\eta`)."""
        iomega = 2j*np.pi*self.freq
        return iomega*mu_0*(1./res - iomega*epsilon_0)*self.__vol

    # MU_R's
    @property
    def v_mu_r(self):
        r"""Volume divided by relative magnetic permeability."""
        if self.__mu_r is None:
            return self.__vol
        else:
            return self.__vol/self.__mu_r


class Field(np.ndarray):
    """Create a Field instance with x-, y-, and z-views of the field.

    A ``Field`` is an ``ndarray`` with additional views of the x-, y-, and
    z-directed fields as attributes, stored as ``fx``, ``fy``, and ``fz``. The
    default array contains the whole field, which can be the electric field,
    the source field, or the residual field, in a 1D array. A ``Field``
    instance has additionally the property ``ensure_pec`` which, if called,
    ensures Perfect Electric Conductor (PEC) boundary condition.

    A ``Field`` can be initiated in three ways:

    1. ``Field(grid)``:
    Calling it with a ``TensorMesh``-instance returns a ``Field``-instance of
    correct dimensions initiated with complex zeroes.

    2. ``Field(grid, field)``:
    Calling it with a ``TensorMesh``-instance and an ``ndarray`` returns a
    ``Field``-instance of the provided ``ndarray``.

    3. ``Field(fx, fy, fz)``:
    Calling it with three ``ndarray``'s which represent the field in x-, y-,
    and z-direction returns a ``Field``-instance with these views.

    Sort-order is 'F'.

    """

    def __new__(cls, grid, field=None, fz=None):
        """Initiate a new Field-instance."""

        # Collect field
        if field is None and fz is None:  # Empty Field with dimension grid.nE.
            new = np.zeros(grid.nE, dtype=complex)
        elif field is not None and fz is None:  # grid and field provided
            new = field
        else:                                   # fx, fy, fz provided
            new = np.r_[grid.ravel('F'), field.ravel('F'), fz.ravel('F')]

        # Store the field as object
        obj = np.asarray(new).view(cls)

        # Store relevant numbers for the views.
        if field is not None and fz is not None:  # Deduce from arrays
            obj.nEx = grid.size
            obj.nEy = field.size
            obj.nEz = fz.size
            obj.vnEx = grid.shape
            obj.vnEy = field.shape
            obj.vnEz = fz.shape
        else:                                     # If grid is provided
            attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz']
            for attr in attr_list:
                setattr(obj, attr, getattr(grid, attr))

        return obj

    def __array_finalize__(self, obj):
        """Ensure relevant numbers are stored no matter how created."""
        if obj is None:
            return

        self.nEx = getattr(obj, 'nEx', None)
        self.nEy = getattr(obj, 'nEy', None)
        self.nEz = getattr(obj, 'nEz', None)
        self.vnEx = getattr(obj, 'vnEx', None)
        self.vnEy = getattr(obj, 'vnEy', None)
        self.vnEz = getattr(obj, 'vnEz', None)

    def __reduce__(self):
        """Customize __reduce__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Get the parent's __reduce__ tuple.
        pickled_state = super(Field, self).__reduce__()

        # Create our own tuple to pass to __setstate__.
        new_state = pickled_state[2]
        attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz']
        for attr in attr_list:
            new_state += (getattr(self, attr),)

        # Return tuple that replaces parent's __setstate__ tuple with our own.
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Customize __setstate__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Set the necessary attributes (in reverse order).
        attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz']
        attr_list.reverse()
        for i, name in enumerate(attr_list):
            i += 1  # We need it 1..#attr instead of 0..#attr-1.
            setattr(self, name, state[-i])

        # Call the parent's __setstate__ with the other tuple elements.
        super(Field, self).__setstate__(state[0:-i])

    @property
    def field(self):
        """Entire field, 1D [fx, fy, fz]."""
        return self.view()

    @field.setter
    def field(self, field):
        """Update field, 1D [fx, fy, fz]."""
        self.view()[:] = field

    @property
    def fx(self):
        """View of the x-directed field in the x-direction (nCx, nNy, nNz)."""
        return self.view()[:self.nEx].reshape(self.vnEx, order='F')

    @fx.setter
    def fx(self, fx):
        """Update field in x-direction."""
        self.view()[:self.nEx] = fx.ravel('F')

    @property
    def fy(self):
        """View of the field in the y-direction (nNx, nCy, nNz)."""
        return self.view()[self.nEx:-self.nEz].reshape(self.vnEy, order='F')

    @fy.setter
    def fy(self, fy):
        """Update field in y-direction."""
        self.view()[self.nEx:-self.nEz] = fy.ravel('F')

    @property
    def fz(self):
        """View of the field in the z-direction (nNx, nNy, nCz)."""
        return self.view()[-self.nEz:].reshape(self.vnEz, order='F')

    @fz.setter
    def fz(self, fz):
        """Update electric field in z-direction."""
        self.view()[-self.nEz:] = fz.ravel('F')

    @property
    def ensure_pec(self):
        """Set Perfect Electric Conductor (PEC) boundary condition."""
        # Apply PEC to fx
        self.fx[:, 0, :] = 0.+0.j
        self.fx[:, -1, :] = 0.+0.j
        self.fx[:, :, 0] = 0.+0.j
        self.fx[:, :, -1] = 0.+0.j

        # Apply PEC to fy
        self.fy[0, :, :] = 0.+0.j
        self.fy[-1, :, :] = 0.+0.j
        self.fy[:, :, 0] = 0.+0.j
        self.fy[:, :, -1] = 0.+0.j

        # Apply PEC to fz
        self.fz[0, :, :] = 0.+0.j
        self.fz[-1, :, :] = 0.+0.j
        self.fz[:, 0, :] = 0.+0.j
        self.fz[:, -1, :] = 0.+0.j


def get_h_field(grid, model, field):
    r"""Return magnetic field corresponding to provided electric field.

    Retrieve the magnetic field :math:`\mathbf{H}` from the electric field
    :math:`\mathbf{H}`  using Farady's law, given by

    .. math::

        \nabla \times \mathbf{E} = \rm{i}\omega\mu\mathbf{H} .

    Note that the magnetic field in x-direction is defined in the center of the
    face defined by the electric field in y- and z-directions, and similar for
    the other field directions. This means that the provided electric field and
    the returned magnetic field have different dimensions::

       E-field:  x: [grid.vectorCCx,  grid.vectorNy,  grid.vectorNz]
                 y: [ grid.vectorNx, grid.vectorCCy,  grid.vectorNz]
                 z: [ grid.vectorNx,  grid.vectorNy, grid.vectorCCz]

       H-field:  x: [ grid.vectorNx, grid.vectorCCy, grid.vectorCCz]
                 y: [grid.vectorCCx,  grid.vectorNy, grid.vectorCCz]
                 z: [grid.vectorCCx, grid.vectorCCy,  grid.vectorNz]


    Parameters
    ----------
    grid : TensorMesh
        Model grid; ``emg3d.utils.TensorMesh`` instance.

    model : Model
        Model; ``emg3d.utils.Model`` instance.

    field : Field
        Electric field; ``emg3d.utils.Field`` instance.


    Returns
    -------
    hfield : Field
        Magnetic field; ``emg3d.utils.Field`` instance.

    """
    # Define the widths of the dual grid.
    dx = (np.r_[0., grid.hx] + np.r_[grid.hx, 0.])/2.
    dy = (np.r_[0., grid.hy] + np.r_[grid.hy, 0.])/2.
    dz = (np.r_[0., grid.hz] + np.r_[grid.hz, 0.])/2.

    # If relative magnetic permeability is not one, we have to take the volume
    # into account, as mu_r is volume-averaged.
    if model._Model__mu_r is not None:
        # Plus and minus indices.
        ixm = np.r_[0, np.arange(grid.nCx)]
        ixp = np.r_[np.arange(grid.nCx), grid.nCx-1]
        iym = np.r_[0, np.arange(grid.nCy)]
        iyp = np.r_[np.arange(grid.nCy), grid.nCy-1]
        izm = np.r_[0, np.arange(grid.nCz)]
        izp = np.r_[np.arange(grid.nCz), grid.nCz-1]

        # Average mu_r for dual-grid.
        mu_r_x = (model.v_mu_r[ixm, :, :] + model.v_mu_r[ixp, :, :])/2.
        mu_r_y = (model.v_mu_r[:, iym, :] + model.v_mu_r[:, iyp, :])/2.
        mu_r_z = (model.v_mu_r[:, :, izm] + model.v_mu_r[:, :, izp])/2.

    # Carry out the curl (^ corresponds to differentiation axis):
    # H_x = (E_z^1 - E_y^2)
    e3d_hx = (np.diff(field.fz, axis=1)/grid.hy[None, :, None] -
              np.diff(field.fy, axis=2)/grid.hz[None, None, :])

    # H_y = (E_x^2 - E_z^0)
    e3d_hy = (np.diff(field.fx, axis=2)/grid.hz[None, None, :] -
              np.diff(field.fz, axis=0)/grid.hx[:, None, None])

    # H_z = (E_y^0 - E_x^1)
    e3d_hz = (np.diff(field.fy, axis=0)/grid.hx[:, None, None] -
              np.diff(field.fx, axis=1)/grid.hy[None, :, None])

    # If relative magnetic permeability is not one, we have to take the volume
    # into account, as mu_r is volume-averaged.
    if model._Model__mu_r is not None:
        hvx = grid.hx[:, None, None]
        hvy = grid.hy[None, :, None]
        hvz = grid.hz[None, None, :]

        e3d_hx *= mu_r_x/(dx[:, None, None]*hvy*hvz)
        e3d_hy *= mu_r_y/(hvx*dy[None, :, None]*hvz)
        e3d_hz *= mu_r_z/(hvx*hvy*dz[None, None, :])

    # Create a Field-instance and divide by j omega mu_0 and return.
    hfield = Field(e3d_hx, e3d_hy, e3d_hz)/(2j*np.pi*model.freq*mu_0)

    return hfield


def get_receiver(grid, fieldf, rec_loc, method='cubic'):
    """Return field at receiver locations.

    Works for electric fields as well as magnetic fields obtained with
    :func:`get_h_field`.

    If ``rec_loc`` is outside of ``grid``, the returned field is zero.

    This is a modified version of :func:`scipy.interpolate.interpn`, using
    :class:`scipy.interpolate.RegularGridInterpolator` if ``method='linear'``
    and a custom-wrapped version of :func:`scipy.ndimage.map_coordinates` if
    ``method='cubic'``. If speed is important then choose 'linear', as it can
    be significantly faster.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a ``TensorMesh``-instance.

    fieldf : ndarray
        Field in a direction, e.g., for the x-direction ``Field.fx``, with
        shape ``Field.vnEx``.

    rec_loc : tuple (rec_x, rec_y, rec_z)
        ``rec_x``, ``rec_y``, and ``rec_z`` positions.

    method : str, optional
        The method of interpolation to perform, 'linear' or 'cubic'.
        Default is 'cubic' (forced to 'linear' if there are less than 3 points
        in any direction).


    Returns
    -------
    values : ndarray
        ``fieldf`` at positions ``rec_loc``.

    """
    # Ensure input field is a certain field, not a Field instance.
    if fieldf.ndim == 1:
        print("* ERROR   :: Field must be x-, y-, or z-directed with ndim=3.")
        print(f"             Shape of provided field: {fieldf.shape}.")
        raise ValueError("Field error")

    # Ensure rec_loc has three entries.
    if len(rec_loc) != 3:
        print("* ERROR   :: Receiver location needs to be (rx, ry, rz).")
        print(f"             Length of provided rec_loc: {len(rec_loc)}.")
        raise ValueError("Receiver location error")

    # Get the vectors corresponding to input data. Dimensions:
    #
    #        E-field       H-field
    #  x: [nCx, nNy, nNz]  [nNx, nCy, nCz]
    #  y: [nNx, nCy, nNz]  [nCx, nNy, nCz]
    #  z: [nNx, nNy, nCz]  [nCx, nCy, nNz]
    #
    points = tuple()
    for i, coord in enumerate(['x', 'y', 'z']):
        if fieldf.shape[i] == getattr(grid, 'nN'+coord):
            pts = (getattr(grid, 'vectorN'+coord), )
        else:
            pts = (getattr(grid, 'vectorCC'+coord), )

        # Add to points
        points += pts

        # We need at least 3 points in each direction for cubic spline.
        # This should never be an issue for a realistic 3D model.
        if pts[0].size < 4:
            method = 'linear'

    # Interpolation.
    if method == "linear":
        ifn = interpolate.RegularGridInterpolator(
                points=points, values=fieldf, method="linear",
                bounds_error=False, fill_value=0.0)

        return ifn(xi=rec_loc)

    else:

        # Replicate the same expansion of xi as used in
        # RegularGridInterpolator, so the input xi can be quite flexible.
        xi = interpolate.interpnd._ndim_coords_from_arrays(rec_loc, ndim=3)
        xi_shape = xi.shape
        xi = xi.reshape(-1, 3)

        # map_coordinates uses the indices of the input data (fieldf in this
        # case) as coordinates. We have therefore to transform our desired
        # output coordinates to this artificial coordinate system too.
        params = {'kind': 'cubic',
                  'bounds_error': False,
                  'fill_value': 'extrapolate'}
        x = interpolate.interp1d(
                points[0], np.arange(len(points[0])), **params)(xi[:, 0])
        y = interpolate.interp1d(
                points[1], np.arange(len(points[1])), **params)(xi[:, 1])
        z = interpolate.interp1d(
                points[2], np.arange(len(points[2])), **params)(xi[:, 2])
        coords = np.vstack([x, y, z])

        # map_coordinates only works for real data; split it up if complex.
        if 'complex' in fieldf.dtype.name:
            real = ndimage.map_coordinates(fieldf.real, coords, order=3)
            imag = ndimage.map_coordinates(fieldf.imag, coords, order=3)
            result = real + 1j*imag
        else:
            result = ndimage.map_coordinates(fieldf, coords, order=3)

        return result.reshape(xi_shape[:-1])


# TIMING FOR LOGS
class Time:
    """Class for timing (now; runtime)."""

    def __init__(self):
        """Initialize time zero (t0) with current time stamp."""
        self.__t0 = default_timer()

    @property
    def t0(self):
        """Return time zero of this class instance."""
        return self.__t0

    @property
    def now(self):
        """Return string of current time."""
        return datetime.now().strftime("%H:%M:%S")

    @property
    def runtime(self):
        """Return string of runtime since time zero."""
        t1 = default_timer() - self.__t0
        return str(timedelta(seconds=np.round(t1)))


# FUNCTIONS RELATED TO DATA MANAGEMENT
def data_write(fname, keys, values, path='data', exists=0):
    """Write all values with their corresponding key to file path/fname.


    Parameters
    ----------
    fname : str
        File name.

    keys : str or list of str
        Name(s) of the values to store in file.

    values : anything
        Values to store with keys in file.

    path : str, optional
        Absolute or relative path where to store. Default is 'data'.

    exists : int, optional
        Flag how to act if a shelve with the given name already exists:

        - < 0: Delete existing shelve.
        - 0 (default): Do nothing (print that it exists).
        - > 0: Append to existing shelve.

    """
    # Get absolute path, create if it doesn't exist.
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # File name full path.
    full_path = path+"/"+fname

    # Check if shelve exists.
    bak_exists = os.path.isfile(full_path+".bak")
    dat_exists = os.path.isfile(full_path+".dat")
    dir_exists = os.path.isfile(full_path+".dir")
    if any([bak_exists, dat_exists, dir_exists]):
        print("   > File exists, ", end="")
        if exists == 0:
            print("NOT SAVING THE DATA.")
            return
        elif exists > 0:
            print("appending to it", end='')
        else:
            print("overwriting it.")
            for ending in ["dat", "bak", "dir"]:
                try:
                    os.remove(full_path+"."+ending)
                except FileNotFoundError:
                    pass

    # Cast into list.
    if not isinstance(keys, (list, tuple)):
        keys = [keys, ]
        values = [values, ]

    # Shelve it.
    with shelve.open(full_path) as db:

        # If appending, print the keys which will be overwritten.
        if exists > 0:
            over = [j for j in keys if any(i == j for i in list(db.keys()))]
            if len(over) > 0:
                print(" (overwriting existing key(s) "+f"{over}"[1:-1]+").")
            else:
                print(".")

        # Writing it to the shelve.
        for i, key in enumerate(keys):
            db[key] = values[i]


def data_read(fname, keys=None, path="data"):
    """Read and return keys from file path/fname.


    Parameters
    ----------
    fname : str
        File name.

    keys : str, list of str, or None; optional
        Name(s) of the values to get from file. If None, returns everything as
        a dict. Default is None.

    path : str, optional
        Absolute or relative path where fname is stored. Default is 'data'.


    Returns
    -------
    out : values or dict
        Requested value(s) or dict containing everything if keys=None.

    """
    # Get absolute path.
    path = os.path.abspath(path)

    # File name full path.
    full_path = path+"/"+fname

    # Check if shelve exists.
    for ending in [".dat", ".bak", ".dir"]:
        if not os.path.isfile(full_path+ending):
            print(f"   > File <{full_path+ending}> does not exist.")
            if isinstance(keys, (list, tuple)):
                return len(keys)*(None, )
            else:
                return None

    # Get it from shelve.
    with shelve.open(path+"/"+fname) as db:
        if keys is None:                           # None
            out = dict()
            for key, item in db.items():
                out[key] = item
            return out

        elif not isinstance(keys, (list, tuple)):  # single parameter
            return db[keys]

        else:                                      # lists/tuples of parameters
            out = []
            for key in keys:
                out.append(db[key])
            return out


# OTHER
class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use ``scooby`` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``, ``emg3d``,
    ``numba``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython`` and
    ``matplotlib``. It also shows MKL information, if available.

    All modules provided in ``add_pckg`` are also shown.

    .. note::

        The package ``scooby`` has to be installed in order to use ``Report``:
        ``pip install scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from emg3d import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['numpy', 'scipy', 'numba', 'emg3d']

        # Optional packages.
        optional = ['IPython', 'matplotlib']

        super().__init__(additional=add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)
