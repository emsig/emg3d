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
import empymod
import numpy as np
from timeit import default_timer
from datetime import datetime, timedelta
from scipy.constants import mu_0, epsilon_0
from scipy import optimize, interpolate, ndimage
from scipy.interpolate import PchipInterpolator as Pchip
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from emg3d import njitted

# scooby is a soft dependency for emg3d
try:
    from scooby import Report as ScoobyReport
except ImportError:
    class ScoobyReport:
        def __init__(self, additional, core, optional, ncol, text_width, sort):
            print("\n* WARNING :: `emg3d.Report` requires `scooby`."
                  "\n             Install it via `pip install scooby`.\n")

# Version: We take care of it here instead of in __init__, so we can use it
# within the package itself (logs).
try:
    # - Released versions just tags:       0.8.0
    # - GitHub commits add .dev#+hash:     0.8.1.dev4+g2785721
    # - Uncommitted changes add timestamp: 0.8.1.dev4+g2785721.d20191022
    from emg3d.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. emg3d should be installed
    # properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')


__all__ = ['Field', 'SourceField', 'get_source_field', 'get_receiver',
           'get_h_field', 'Model', 'VolumeModel', 'grid2grid', 'TensorMesh',
           'get_hx_h0', 'get_cell_numbers', 'get_stretched_h', 'get_domain',
           'get_hx', 'Fourier', 'data_write', 'data_read', 'Time', 'Report']


# FIELDS
class Field(np.ndarray):
    r"""Create a Field instance with x-, y-, and z-views of the field.

    A ``Field`` is an ``ndarray`` with additional views of the x-, y-, and
    z-directed fields as attributes, stored as ``fx``, ``fy``, and ``fz``. The
    default array contains the whole field, which can be the electric field,
    the source field, or the residual field, in a 1D array. A ``Field``
    instance has additionally the property ``ensure_pec`` which, if called,
    ensures Perfect Electric Conductor (PEC) boundary condition. It also has
    the two attributes ``amp`` and ``pha`` for the amplitude and phase, as
    common in frequency-domain CSEM.

    A ``Field`` can be initiated in three ways:

    1. ``Field(grid, dtype=complex)``:
       Calling it with a :class:`TensorMesh`-instance returns a
       ``Field``-instance of correct dimensions initiated with zeroes of data
       type ``dtype``.
    2. ``Field(grid, field)``:
       Calling it with a :class:`TensorMesh`-instance and an ``ndarray``
       returns a ``Field``-instance of the provided ``ndarray``, of same data
       type.
    3. ``Field(fx, fy, fz)``:
       Calling it with three ``ndarray``'s which represent the field in x-, y-,
       and z-direction returns a ``Field``-instance with these views, of same
       data type.

    Sort-order is 'F'.


    Parameters
    ----------

    fx_or_grid : TensorMesh or ndarray
        Either a TensorMesh instance or an ndarray of shape grid.nEx or
        grid.vnEx. See explanations above. Only mandatory parameter; if the
        only one provided, it will initiate a zero-field of ``dtype``.

    fy_or_field : Field or ndarray
        Either a Field instance or an ndarray of shape grid.nEy or grid.vnEy.
        See explanations above.

    fz : ndarray
        An ndarray of shape grid.nEz or grid.vnEz. See explanations above.

    dtype : dtype,
        Only used if ``fy_or_field=None`` and ``fz=None``; the initiated
        zero-field for the provided TensorMesh has data type ``dtype``.
        Default: complex.

    freq : float, optional
        Source frequency (Hz), used to calculate the Laplace parameter ``s``.
        Either positive or negative:

        - ``freq`` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - ``freq`` < 0: Laplace domain, hence
          :math:`s = f` (real).

        Just added as info if provided.

    """

    def __new__(cls, fx_or_grid, fy_or_field=None, fz=None, dtype=complex,
                freq=None):
        """Initiate a new Field-instance."""

        # Collect field
        if fy_or_field is None and fz is None:          # Empty Field with
            new = np.zeros(fx_or_grid.nE, dtype=dtype)  # dimension grid.nE.
        elif fz is None:                  # grid and field provided
            new = fy_or_field
        else:                             # fx, fy, fz provided
            new = np.r_[fx_or_grid.ravel('F'), fy_or_field.ravel('F'),
                        fz.ravel('F')]

        # Store the field as object
        obj = np.asarray(new).view(cls)

        # Store relevant numbers for the views.
        if fy_or_field is not None and fz is not None:  # Deduce from arrays
            obj.nEx = fx_or_grid.size
            obj.nEy = fy_or_field.size
            obj.nEz = fz.size
            obj.vnEx = fx_or_grid.shape
            obj.vnEy = fy_or_field.shape
            obj.vnEz = fz.shape
        else:                                     # If grid is provided
            attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz']
            for attr in attr_list:
                setattr(obj, attr, getattr(fx_or_grid, attr))

        # Get Laplace parameter.
        if freq is None and hasattr(fy_or_field, 'freq'):
            freq = fy_or_field._freq
        obj._freq = freq
        if freq is not None:
            if freq > 0:  # Frequency domain; s = iw = 2i*pi*f.
                obj._sval = np.array(-2j*np.pi*freq)
                obj._smu0 = np.array(-2j*np.pi*freq*mu_0)
            elif freq < 0:  # Laplace domain; s.
                obj._sval = np.array(freq)
                obj._smu0 = np.array(freq*mu_0)
            else:
                print("* ERROR   :: ``freq`` must be >0 (frequency domain) "
                      "or <0 (Laplace domain)."
                      f"             Provided frequency: {freq} Hz.")
                raise ValueError("Source error")
        else:
            obj._sval = None
            obj._smu0 = None

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
        self._freq = getattr(obj, '_freq', None)
        self._sval = getattr(obj, '_sval', None)
        self._smu0 = getattr(obj, '_smu0', None)

    def __reduce__(self):
        """Customize __reduce__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Get the parent's __reduce__ tuple.
        pickled_state = super(Field, self).__reduce__()

        # Create our own tuple to pass to __setstate__.
        new_state = pickled_state[2]
        attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz', '_freq',
                     '_sval', '_smu0']
        for attr in attr_list:
            new_state += (getattr(self, attr),)

        # Return tuple that replaces parent's __setstate__ tuple with our own.
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Customize __setstate__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Set the necessary attributes (in reverse order).
        attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz', '_freq',
                     '_sval', '_smu0']
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
    def amp(self):
        """Amplitude of the electromagnetic field."""
        return np.abs(self.view())

    @property
    def pha(self):
        """Phase of the electromagnetic field, unwrapped and in degrees."""
        return 180*np.unwrap(np.angle(self.view()))/np.pi

    @property
    def freq(self):
        """Return frequency."""
        if self._freq is None:
            return None
        else:
            return abs(self._freq)

    @property
    def smu0(self):
        """Return s*mu_0; mu_0 = Magn. permeability of free space [H/m]."""
        return self._smu0

    @property
    def sval(self):
        """Return s; s=iw in frequency domain; s=freq in Laplace domain."""
        return self._sval

    @property
    def ensure_pec(self):
        """Set Perfect Electric Conductor (PEC) boundary condition."""
        # Apply PEC to fx
        self.fx[:, 0, :] = 0.
        self.fx[:, -1, :] = 0.
        self.fx[:, :, 0] = 0.
        self.fx[:, :, -1] = 0.

        # Apply PEC to fy
        self.fy[0, :, :] = 0.
        self.fy[-1, :, :] = 0.
        self.fy[:, :, 0] = 0.
        self.fy[:, :, -1] = 0.

        # Apply PEC to fz
        self.fz[0, :, :] = 0.
        self.fz[-1, :, :] = 0.
        self.fz[:, 0, :] = 0.
        self.fz[:, -1, :] = 0.

    @property
    def is_electric(self):
        """Returns True if Field is electric, False if it is magnetic."""
        return self.vnEx[0] < self.vnEy[0]


class SourceField(Field):
    r"""Create a Source-Field instance with x-, y-, and z-views of the field.

    A subclass of :class:`Field`. Additional properties are the real-valued
    source vector (``vector``, ``vx``, ``vy``, ``vz``), which sum is always
    one. For a ``SourceField`` frequency is a mandatory  parameter, unlike
    for a ``Field`` (recommended also for ``Field`` though),

    Parameters
    ----------

    grid : TensorMesh
        A TensorMesh instance.

    freq : float
        Source frequency (Hz), used to calculate the Laplace parameter ``s``.
        Either positive or negative:

        - ``freq`` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - ``freq`` < 0: Laplace domain, hence
          :math:`s = f` (real).

    """

    def __new__(cls, grid, freq):
        """Initiate a new Source Field."""
        if freq > 0:
            dtype = complex
        else:
            dtype = float
        return super().__new__(cls, grid, dtype=dtype, freq=freq)

    @property
    def vector(self):
        """Entire vector, 1D [vx, vy, vz]."""
        return np.real(self.field/self.smu0)

    @property
    def vx(self):
        """View of the x-directed vector in the x-direction (nCx, nNy, nNz)."""
        return np.real(self.field.fx/self.smu0)

    @property
    def vy(self):
        """View of the vector in the y-direction (nNx, nCy, nNz)."""
        return np.real(self.field.fy/self.smu0)

    @property
    def vz(self):
        """View of the vector in the z-direction (nNx, nNy, nCz)."""
        return np.real(self.field.fz/self.smu0)


def get_source_field(grid, src, freq, strength=0):
    r"""Return the source field.

    The source field is given in Equation 2 in [Muld06]_,

    .. math::

        s \mu_0 \mathbf{J}_\mathrm{s} ,

    where :math:`s = \mathrm{i} \omega`. Either finite length dipoles or
    infinitesimal small point dipoles can be defined, whereas the return source
    field corresponds to a normalized (1 Am) source distributed within the
    cell(s) it resides (can be changed with the ``strength``-parameter).

    The adjoint of the trilinear interpolation is used to distribute the
    point(s) to the grid edges, which corresponds to the discretization of a
    Dirac ([PlDM07]_).


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a ``TensorMesh``-instance.

    src : list of floats
        Source coordinates (m). There are two formats:

          - Finite length dipole: ``[x0, x1, y0, y1, z0, z1]``.
          - Point dipole: ``[x, y, z, azimuth, dip]``.

    freq : float
        Source frequency (Hz), used to calculate the Laplace parameter ``s``.
        Either positive or negative:

        - ``freq`` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - ``freq`` < 0: Laplace domain, hence
          :math:`s = f` (real).

    strength : float or complex, optional
        Source strength (A):

          - If 0, output is normalized to a source of 1 m length, and source
            strength of 1 A.
          - If != 0, output is returned for given source length and strength.

        Default is 0.


    Returns
    -------
    sfield : :func:`SourceField`-instance
        Source field, normalized to 1 A m.

    """
    # Cast some parameters.
    src = np.asarray(src, dtype=float)
    strength = np.asarray(strength)

    # Ensure source is a point or a finite dipole.
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

    # Ensure source is within grid.
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
        moment = srcdir
    else:              # Multiply source length with source strength
        moment = strength*srcdir

    def set_source(grid, moment, finite):
        """Set the source-field in idir."""

        # Initiate zero source field.
        sfield = SourceField(grid, freq)

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

        # Multiply by moment*s*mu in per direction.
        sfield.fx *= moment[0]*sfield.smu0
        sfield.fy *= moment[1]*sfield.smu0
        sfield.fz *= moment[2]*sfield.smu0

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

    # Get the source field.
    sfield = set_source(grid, moment, finite)

    # Add src and moment information.
    sfield.src = src
    sfield.strength = strength
    sfield.moment = moment

    return sfield


def get_receiver(grid, values, coordinates, method='cubic', extrapolate=False):
    """Return values corresponding to grid at coordinates.

    Works for electric fields as well as magnetic fields obtained with
    :func:`get_h_field`, and for model parameters.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a ``TensorMesh``-instance.

    values : ndarray
        Field instance, or a particular field (e.g. field.fx); Model
        parameters.

    coordinates : tuple (x, y, z)
        Coordinates (x, y, z) where to interpolate ``values``; e.g. receiver
        locations.

    method : str, optional
        The method of interpolation to perform, 'linear' or 'cubic'.
        Default is 'cubic' (forced to 'linear' if there are less than 3 points
        in any direction).

    extrapolate : bool
        If True, points on ``new_grid`` which are outside of ``grid`` are
        filled by the nearest value (if ``method='cubic'``) or by extrapolation
        (if ``method='linear'``). If False, points outside are set to zero.

        Default is False.


    Returns
    -------
    new_values : ndarray or EMArray
        Values at ``coordinates``.

        If input was a field it returns an EMArray, which is a subclassed
        ndarray with ``.pha`` and ``.amp`` attributes.

        If input was an entire Field instance, output is a tuple (fx, fy, fz).


    See Also
    --------
    grid2grid : Interpolation of model parameters or fields to a new grid.

    """
    # If values is a Field instance, call it recursively for each field.
    if hasattr(values, 'field') and values.field.ndim == 1:
        fx = get_receiver(grid, values.fx, coordinates, method, extrapolate)
        fy = get_receiver(grid, values.fy, coordinates, method, extrapolate)
        fz = get_receiver(grid, values.fz, coordinates, method, extrapolate)
        return fx, fy, fz

    if len(coordinates) != 3:
        print("* ERROR   :: Coordinates  needs to be in the form (x, y, z).")
        print(f"             Length of provided coord.: {len(coordinates)}.")
        raise ValueError("Coordinates error")

    # Get the vectors corresponding to input data. Dimensions:
    #
    #         E-field          H-field      |  Model Parameter
    #  x: [nCx, nNy, nNz]  [nNx, nCy, nCz]  |
    #  y: [nNx, nCy, nNz]  [nCx, nNy, nCz]  |  [nCx, nCy, nCz]
    #  z: [nNx, nNy, nCz]  [nCx, nCy, nNz]  |
    #
    points = tuple()
    for i, coord in enumerate(['x', 'y', 'z']):
        if values.shape[i] == getattr(grid, 'nN'+coord):
            pts = (getattr(grid, 'vectorN'+coord), )
        else:
            pts = (getattr(grid, 'vectorCC'+coord), )

        # Add to points.
        points += pts

    if extrapolate:
        fill_value = None
        mode = 'nearest'
    else:
        fill_value = 0.0
        mode = 'constant'
    out = _interp3d(points, values, coordinates, method, fill_value, mode)

    # Return an EMArray if input is a field, else simply the values.
    if values.size == grid.nC:
        return out
    else:
        return empymod.utils.EMArray(out)


def get_h_field(grid, model, field):
    r"""Return magnetic field corresponding to provided electric field.

    Retrieve the magnetic field :math:`\mathbf{H}` from the electric field
    :math:`\mathbf{E}` using Farady's law, given by

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
    if model._mu_r is not None:

        # Get volume-averaged values.
        vmodel = VolumeModel(grid, model, field)

        # Plus and minus indices.
        ixm = np.r_[0, np.arange(grid.nCx)]
        ixp = np.r_[np.arange(grid.nCx), grid.nCx-1]
        iym = np.r_[0, np.arange(grid.nCy)]
        iyp = np.r_[np.arange(grid.nCy), grid.nCy-1]
        izm = np.r_[0, np.arange(grid.nCz)]
        izp = np.r_[np.arange(grid.nCz), grid.nCz-1]

        # Average mu_r for dual-grid.
        zeta_x = (vmodel.zeta[ixm, :, :] + vmodel.zeta[ixp, :, :])/2.
        zeta_y = (vmodel.zeta[:, iym, :] + vmodel.zeta[:, iyp, :])/2.
        zeta_z = (vmodel.zeta[:, :, izm] + vmodel.zeta[:, :, izp])/2.

        hvx = grid.hx[:, None, None]
        hvy = grid.hy[None, :, None]
        hvz = grid.hz[None, None, :]

        # Define the widths of the dual grid.
        dx = (np.r_[0., grid.hx] + np.r_[grid.hx, 0.])/2.
        dy = (np.r_[0., grid.hy] + np.r_[grid.hy, 0.])/2.
        dz = (np.r_[0., grid.hz] + np.r_[grid.hz, 0.])/2.

        # Multiply fields by mu_r.
        e3d_hx *= zeta_x/(dx[:, None, None]*hvy*hvz)
        e3d_hy *= zeta_y/(hvx*dy[None, :, None]*hvz)
        e3d_hz *= zeta_z/(hvx*hvy*dz[None, None, :])

    # Create a Field-instance and divide by s*mu_0 and return.
    return -Field(e3d_hx, e3d_hy, e3d_hz)/field.smu0


# MODEL
class Model:
    r"""Create a model instance.

    Class to provide model parameters (x-, y-, and z-directed resistivities,
    electric permittivity and magnetic permeability) to the solver. Relative
    magnetic permeability :math:`\mu_\mathrm{r}` is by default set to one and
    electric permittivity :math:`\varepsilon_\mathrm{r}` is by default set to
    zero, but they can also be provided (isotropically). Keep in mind that the
    multigrid method as implemented in ``emg3d`` only works for the diffusive
    approximation. As soon as the displacement-part in the Maxwell's equations
    becomes too dominant it will fail (high frequencies or very high electric
    permittivity).


    Parameters
    ----------
    grid : TensorMesh
        Grid on which to apply model.

    res_x, res_y, res_z : float or ndarray; default to 1.
        Resistivity in x-, y-, and z-directions. If ndarray, they must have the
        shape of grid.vnC (F-ordered) or grid.nC. Resistivities have to be
        bigger than zero and smaller than infinity.

    mu_r : None, float, or ndarray
        Relative magnetic permeability (isotropic). If ndarray it must have the
        shape of grid.vnC (F-ordered) or grid.nC. Default is None, which
        corresponds to 1., but avoids the calculation of zeta. Magnetic
        permeability has to be bigger than zero and smaller than infinity.

    epsilon_r : None, float, or ndarray
       Relative electric permittivity (isotropic). If ndarray it must have the
       shape of grid.vnC (F-ordered) or grid.nC. The displacement part is
       completely neglected (diffusive approximation) if set to None, which is
       the default. Electric permittivity has to be bigger than zero and
       smaller than infinity.

    """

    def __init__(self, grid, res_x=1., res_y=None, res_z=None, freq=None,
                 mu_r=None, epsilon_r=None):
        """Initiate a new model."""

        # Issue warning for backwards compatibility.
        if freq is not None:
            print("\n    ``Model`` does not take frequency ``freq`` any "
                  "longer;\n    providing it will break in the future.")

        # Store required info from grid.
        self.nC = grid.nC
        self.vnC = grid.vnC
        self._vol = grid.vol.reshape(self.vnC, order='F')

        # Check case.
        self.case_names = ['isotropic', 'HTI', 'VTI', 'tri-axial']
        if res_y is None and res_z is None:  # 0: Isotropic.
            self.case = 0
        elif res_z is None:                  # 1: HTI.
            self.case = 1
        elif res_y is None:                  # 2: VTI.
            self.case = 2
        else:                                # 3: Tri-axial anisotropy.
            self.case = 3

        # Initiate all parameters.
        self._res_x = self._check_parameter(res_x, 'res_x')
        self._res_y = self._check_parameter(res_y, 'res_y')
        self._res_z = self._check_parameter(res_z, 'res_z')
        self._mu_r = self._check_parameter(mu_r, 'mu_r')
        self._epsilon_r = self._check_parameter(epsilon_r, 'epsilon_r')

    def __repr__(self):
        """Simple representation."""
        return (f"Model; {self.case_names[self.case]} resistivities"
                f"{'' if self.mu_r is None else '; mu_r'}"
                f"{'' if self.epsilon_r is None else '; epsilon_r'}")

    # RESISTIVITIES
    @property
    def res_x(self):
        r"""Resistivity in x-direction."""
        return self._return_parameter(self._res_x)

    @res_x.setter
    def res_x(self, res):
        r"""Update resistivity in x-direction."""
        self._res_x = self._check_parameter(res, 'res_x')

    @property
    def res_y(self):
        r"""Resistivity in y-direction."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            return self._return_parameter(self._res_y)
        else:                    # Return res_x.
            return self._return_parameter(self._res_x)

    @res_y.setter
    def res_y(self, res):
        r"""Update resistivity in y-direction."""

        # Adjust case in case res_z was not set so far.
        if self.case == 0:  # If it was isotropic, it is HTI now.
            self.case = 1
        elif self.case == 2:  # If it was VTI, it is tri-axial now.
            self.case = 3

        # Update it.
        self._res_y = self._check_parameter(res, 'res_y')

    @property
    def res_z(self):
        r"""Resistivity in z-direction."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            return self._return_parameter(self._res_z)
        else:                    # Return res_x.
            return self._return_parameter(self._res_x)

    @res_z.setter
    def res_z(self, res):
        r"""Update resistivity in z-direction."""

        # Adjust case in case res_z was not set so far.
        if self.case == 0:  # If it was isotropic, it is VTI now.
            self.case = 2
        elif self.case == 1:  # If it was HTI, it is tri-axial now.
            self.case = 3

        # Update it.
        self._res_z = self._check_parameter(res, 'res_z')

    # MAGNETIC PERMEABILITIES
    @property
    def mu_r(self):
        r"""Magnetic permeability."""
        return self._return_parameter(self._mu_r)

    @mu_r.setter
    def mu_r(self, mu_r):
        r"""Update magnetic permeability."""
        self._mu_r = self._check_parameter(mu_r, 'mu_r')

    # ELECTRIC PERMITTIVITIES
    @property
    def epsilon_r(self):
        r"""Electric permittivity."""
        # Get epsilon.
        return self._return_parameter(self._epsilon_r)

    @epsilon_r.setter
    def epsilon_r(self, epsilon_r):
        r"""Update electric permittivity."""
        self._epsilon_r = self._check_parameter(epsilon_r, 'epsilon_r')

    # INTERNAL UTILITIES
    def _check_parameter(self, var, name):
        """Check parameter.

        - Shape must be (), (1,), nC, or vnC.
        - Value(s) must be 0 < var < inf.
        """

        # If None, exit.
        if var is None:
            return None

        # Cast it to floats, ravel.
        var = np.asarray(var, dtype=float).ravel('F')

        # Check for wrong size.
        if var.size not in [1, self.nC]:
            print(f"* ERROR   :: Shape of {name} must be (), {self.vnC}, or "
                  f"{self.nC}.\n             Provided: {var.shape}.")
            raise ValueError("Wrong Shape")

        # Check 0 < val or 0 <= val.
        if not np.all(var > 0):
            print(f"* ERROR   :: ``{name}`` must be all `0 < var`.")
            raise ValueError("Parameter error")

        # Check val < inf.
        if not np.all(var < np.inf):
            print(f"* ERROR   :: ``{name}`` must be all `var < inf`.")
            raise ValueError("Parameter error")

        return var

    def _return_parameter(self, var):
        """Return parameter as float or shape vnC."""

        # Return depending on value and size.
        if var is None:      # Because of mu_r, epsilon_r.
            return None
        elif var.size == 1:  # In case of float.
            return var
        else:                # Else, has shape vnC.
            return var.reshape(self.vnC, order='F')


class VolumeModel:
    r"""Return a volume-averaged version of provided model.

    Takes a Model instance and returns the volume averaged values. This is used
    by the solver internally.

    .. math::

        \eta_{\{x,y,z\}} = -V\mathrm{i}\omega\mu_0
              \left(\rho^{-1}_{\{x,y,z\}} + \mathrm{i}\omega\varepsilon\right)

    .. math::

        \zeta = V\mu_\mathrm{r}^{-1}


    Parameters
    ----------
    grid : TensorMesh
        Grid on which to apply model.

    model : Model
        Model to transform to volume-averaged values.

    sfield : SourceField
       A VolumeModel is frequency-dependent. The frequency-information is taken
       from the provided source filed.

    """

    def __init__(self, grid, model, sfield):
        """Initiate a new model with volume-averaged properties."""

        # Store case, for restriction.
        self.case = model.case

        # eta_x
        self._eta_x = self.calculate_eta('res_x', grid, model, sfield)

        # eta_y
        if model.case in [1, 3]:  # HTI or tri-axial.
            self._eta_y = self.calculate_eta('res_y', grid, model, sfield)

        # eta_z
        if self.case in [2, 3]:  # VTI or tri-axial.
            self._eta_z = self.calculate_eta('res_z', grid, model, sfield)

        # zeta
        self._zeta = self.calculate_zeta('mu_r', grid, model)

    # ETA's
    @property
    def eta_x(self):
        r"""eta in x-direction."""
        return self._eta_x

    @property
    def eta_y(self):
        r"""eta in y-direction."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            return self._eta_y
        else:                    # Return eta_x.
            return self._eta_x

    @property
    def eta_z(self):
        r"""eta in z-direction."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            return self._eta_z
        else:                    # Return eta_x.
            return self._eta_x

    @property
    def zeta(self):
        r"""zeta."""
        return self._zeta

    @staticmethod
    def calculate_eta(name, grid, model, field):
        r"""eta: volume divided by resistivity."""

        # Initiate eta
        eta = field.smu0*grid.vol.reshape(grid.vnC, order='F')

        # Calculate eta depending on epsilon.
        if model.epsilon_r is None:  # Diffusive approximation.
            eta /= getattr(model, name)

        else:
            eps_term = field.sval*epsilon_0*model.epsilon_r
            sig_term = 1./getattr(model, name)
            eta *= sig_term - eps_term

        return eta

    @staticmethod
    def calculate_zeta(name, grid, model):
        r"""zeta: volume divided by mu_r."""

        if getattr(model, name, None) is None:
            return grid.vol.reshape(grid.vnC, order='F')

        else:
            return grid.vol.reshape(grid.vnC, order='F')/getattr(model, name)


# INTERPOLATION
def grid2grid(grid, values, new_grid, method='linear', extrapolate=True):
    """Interpolate ``values`` located on ``grid`` to ``new_grid``.

    The linear method is the fastest, and the volume-averaging method is the
    slowest. For big grids (millions of cells), the difference in runtime can
    be substantial.


    Parameters
    ----------
    grid, new_grid : TensorMesh
        Input and output model grids; ``TensorMesh``-instances.

    values : ndarray
        Model parameters; Field instance, or a particular field (e.g.
        field.fx). For fields the method cannot be 'volume'.

    method : {<'volume'>, 'linear', 'cubic'}, optional
        The method of interpolation to perform. The volume averaging method
        ensures that the total sum of the property stays constant. Default is
        'volume'. The method 'cubic' requires at least three points in any
        direction, otherwise it will fall back to 'linear'.

        Volume averaging is only implemented for model parameters, not for
        fields.

    extrapolate : bool
        If True, points on ``new_grid`` which are outside of ``grid`` are
        filled by the nearest value (if ``method='cubic'``) or by extrapolation
        (if ``method='linear'``). If False, points outside are set to zero.

        For ``method='volume'`` it always uses the nearest value for points
        outside of ``grid``.

        Default is True.


    Returns
    -------
    new_values : ndarray
        Values corresponding to ``new_grid``.


    See Also
    --------
    get_receiver : Interpolation of model parameters or fields to (x, y, z).

    """

    # If values is a Field instance, call it recursively for each field.
    if hasattr(values, 'field') and values.field.ndim == 1:
        fx = grid2grid(grid, np.asarray(values.fx), new_grid, method)
        fy = grid2grid(grid, np.asarray(values.fy), new_grid, method)
        fz = grid2grid(grid, np.asarray(values.fz), new_grid, method)

        # Return a field instance.
        return Field(fx, fy, fz)

    # If values is a particular field, ensure method is not 'volume'.
    if not np.all(grid.vnC == values.shape) and method == 'volume':
        print("* ERROR   :: ``method='volume'`` not implemented for fields.")
        raise ValueError("Method not implemented.")

    if method == 'volume':
        points = (grid.vectorNx, grid.vectorNy, grid.vectorNz)
        new_points = (new_grid.vectorNx, new_grid.vectorNy, new_grid.vectorNz)
        new_values = np.zeros(new_grid.vnC, dtype=values.dtype)
        vol = new_grid.vol.reshape(new_grid.vnC, order='F')

        # Get values from `volume_average`.
        njitted.volume_average(*points, values, *new_points, new_values, vol)

    else:
        # Get the vectors corresponding to input data.
        points = tuple()
        new_points = tuple()
        shape = tuple()
        for i, coord in enumerate(['x', 'y', 'z']):
            if values.shape[i] == getattr(grid, 'nN'+coord):
                pts = getattr(grid, 'vectorN'+coord)
                new_pts = getattr(new_grid, 'vectorN'+coord)
            else:
                pts = getattr(grid, 'vectorCC'+coord)
                new_pts = getattr(new_grid, 'vectorCC'+coord)

            # Add to points.
            points += (pts, )
            new_points += (new_pts, )
            shape += (len(new_pts), )

        # Format the output points.
        xx, yy, zz = np.broadcast_arrays(
                new_points[0][:, None, None],
                new_points[1][:, None],
                new_points[2])
        new_points = np.r_[xx.ravel('F'), yy.ravel('F'), zz.ravel('F')]
        new_points = new_points.reshape(-1, 3, order='F')

        # Get values from `_interp3d`.
        if extrapolate:
            fill_value = None
            mode = 'nearest'
        else:
            fill_value = 0.0
            mode = 'constant'
        new_values = _interp3d(
                points, values, new_points, method, fill_value, mode)

        new_values = new_values.reshape(shape, order='F')

    return new_values


def _interp3d(points, values, new_points, method, fill_value, mode):
    """Interpolate values in 3D either linearly or with a cubic spline.

    Return ``values`` corresponding to a regular 3D grid defined by ``points``
    on ``new_points``.

    This is a modified version of :func:`scipy.interpolate.interpn`, using
    :class:`scipy.interpolate.RegularGridInterpolator` if ``method='linear'``
    and a custom-wrapped version of :func:`scipy.ndimage.map_coordinates` if
    ``method='cubic'``. If speed is important then choose 'linear', as it can
    be significantly faster.


    Parameters
    ----------
    points : tuple of ndarray of float, with shapes ((nx, ), (ny, ) (nz, ))
        The points defining the regular grid in three dimensions.

    values : array_like, shape (nx, ny, nz)
        The data on the regular grid in three dimensions.

    new_points : tuple (rec_x, rec_y, rec_z)
        Coordinates (x, y, z) of new points.

    method : {'cubic', 'linear'}, optional
        The method of interpolation to perform, 'linear' or 'cubic'. Default is
        'cubic' (forced to 'linear' if there are less than 3 points in any
        direction).

    fill_value : float or None
        Passed to ``interpolate.RegularGridInterpolator`` if
        ``method='linear'``: The value to use for points outside of the
        interpolation domain. If None, values outside the domain are
        extrapolated.

    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}
        Passed to ``ndimage.map_coordinates`` if ``method='cubic'``: Determines
        how the input array is extended beyond its boundaries.


    Returns
    -------
    new_values : ndarray
        Values corresponding to ``new_points``.

    """

    # We need at least 3 points in each direction for cubic spline. This should
    # never be an issue for a realistic 3D model.
    for pts in points:
        if len(pts) < 4:
            method = 'linear'

    # Interpolation.
    if method == "linear":
        ifn = interpolate.RegularGridInterpolator(
                points=points, values=values, method="linear",
                bounds_error=False, fill_value=fill_value)

        new_values = ifn(xi=new_points)

    else:

        # Replicate the same expansion of xi as used in
        # RegularGridInterpolator, so the input xi can be quite flexible.
        xi = interpolate.interpnd._ndim_coords_from_arrays(new_points, ndim=3)
        xi_shape = xi.shape
        xi = xi.reshape(-1, 3)

        # map_coordinates uses the indices of the input data (values in this
        # case) as coordinates. We have therefore to transform our desired
        # output coordinates to this artificial coordinate system too.
        coords = np.empty(xi.T.shape)
        for i in range(3):
            coords[i] = interpolate.interp1d(
                    points[i], np.arange(len(points[i])), kind='cubic',
                    bounds_error=False, fill_value='extrapolate',)(xi[:, i])

        # map_coordinates only works for real data; split it up if complex.
        params3d = {'order': 3, 'mode': mode, 'cval': 0.0}
        if 'complex' in values.dtype.name:
            real = ndimage.map_coordinates(values.real, coords, **params3d)
            imag = ndimage.map_coordinates(values.imag, coords, **params3d)
            result = real + 1j*imag
        else:
            result = ndimage.map_coordinates(values, coords, **params3d)

        new_values = result.reshape(xi_shape[:-1])

    return new_values


# MESH
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

    def __repr__(self):
        """Simple representation."""
        return f"TensorMesh: {self.nCx} x {self.nCy} x {self.nCz} ({self.nC})"

    @property
    def vol(self):
        """Construct cell volumes of the 3D model as 1D array."""
        if getattr(self, '_vol', None) is None:
            vol = np.outer(np.outer(self.hx, self.hy).ravel('F'), self.hz)
            self._vol = vol.ravel('F')
        return self._vol


def get_hx_h0(freq, res, domain, fixed=0., possible_nx=None, min_width=None,
              pps=3, alpha=None, raise_error=True, verb=1, return_info=False):
    r"""Return cell widths and origin for given parameters.

    Returns cell widths for the provided frequency, resistivity, domain extent,
    and other parameters using a flexible amount of cells. See input parameters
    for more details. A maximum of three hard/fixed boundaries can be provided
    (one of which is the grid center).

    The minimum cell width is calculated through :math:`\delta/\rm{pps}`, where
    the skin depth is given by :math:`\delta = 503.3 \sqrt{\rho/f}`, and
    the parameter ``pps`` stands for 'points-per-skindepth'. The minimum cell
    width can be restricted with the parameter ``min_width``.

    The actual calculation domain adds a buffer zone around the (survey)
    domain. The thickness of the buffer is six times the skin depth. The field
    is basically zero after two wavelengths. A wavelength is
    :math:`2\pi\delta`, hence roughly 6 times the skin depth. Taking a factor 6
    gives therefore almost two wavelengths, as the field travels to the
    boundary and back. The actual buffer thickness can be steered with the
    ``res`` parameter.

    One has to take into account that the air is very resistive, which has to
    be considered not just in the vertical direction, but also in the
    horizontal directions, as the airwave will bounce back from the sides
    otherwise. In the marine case this issue reduces with increasing water
    depth.


    See Also
    --------
    get_stretched_h : Get ``hx`` for a fixed number ``nx`` and within a fixed
                      domain.


    Parameters
    ----------

    freq : float
        Frequency (Hz) to calculate the skin depth. The skin depth is a concept
        defined in the frequency domain. If a negative frequency is provided,
        it is assumed that the calculation is carried out in the Laplace
        domain. To calculate the skin depth, the value of ``freq`` is then
        multiplied by :math:`-2\pi`, to simulate the closest
        frequency-equivalent.

    res : float or list
        Resistivity (Ohm m) to calculate the skin depth. The skin depth is
        used to calculate the minimum cell width and the boundary thicknesses.
        Up to three resistivities can be provided:

        - float: Same resistivity for everything;
        - [min_width, boundaries];
        - [min_width, left boundary, right boundary].

    domain : list
        Contains the survey-domain limits [min, max]. The actual calculation
        domain consists of this domain plus a buffer zone around it, which
        depends on frequency and resistivity.

    fixed : list, optional
        Fixed boundaries, one, two, or maximum three values. The grid is
        centered around the first value. Hence it is the center location with
        the smallest cell. Two more fixed boundaries can be added, at most one
        on each side of the first one.
        Default is 0.

    possible_nx : list, optional
        List of possible numbers of cells. See :func:`get_cell_numbers`.
        Default is ``get_cell_numbers(500, 5, 3)``, which corresponds to
        [16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384].

    min_width : float, list or None, optional
        Minimum cell width restriction:

        - None : No restriction;
        - float : Fixed to this value, ignoring skin depth and ``pps``.
        - list [min, max] : Lower and upper bounds.

        Default is None.

    pps : int, optional
        Points per skindepth; minimum cell width is calculated via
        `dmin = skindepth/pps`.
        Default = 3.

    alpha : list, optional
        Maximum alpha and step size to find a good alpha. The first value is
        the maximum alpha of the survey domain, the second value is the maximum
        alpha for the buffer zone, and the third value is the step size.
        Default = [1, 1.5, .01], hence no stretching within the survey domain
        and a maximum stretching of 1.5 in the buffer zone; step size is 0.01.

    raise_error : bool, optional
        If True, an error is raised if no suitable grid is found. Otherwise it
        just prints a message and returns None's.
        Default is True.

    verb : int, optional
        Verbosity, 0 or 1.
        Default = 1.

    return_info : bool
        If True, a dictionary is returned with some grid info (min and max
        cell width and alpha).


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    x0 : float
        Origin of the mesh.

    info : dict
        Dictionary with mesh info; only if ``return_info=True``.

        Keys:

        - ``dmin``: Minimum cell width;
        - ``dmax``: Maximum cell width;
        - ``amin``: Minimum alpha;
        - ``amax``: Maximum alpha.

    """
    # Get variables with default lists:
    if alpha is None:
        alpha = [1, 1.5, 0.01]
    if possible_nx is None:
        possible_nx = get_cell_numbers(500, 5, 3)

    # Cast resistivity value(s).
    res = np.array(res, ndmin=1)
    if res.size == 1:
        res_arr = np.array([res[0], res[0], res[0]])
    elif res.size == 2:
        res_arr = np.array([res[0], res[1], res[1]])
    else:
        res_arr = np.array([res[0], res[1], res[2]])

    # Cast and check fixed.
    fixed = np.array(fixed, ndmin=1)
    if fixed.size > 2:

        # Check length.
        if fixed.size > 3:
            print("\n* ERROR   :: Maximum three fixed boundaries permitted.\n"
                  f"             Provided: {fixed.size}.")
            raise ValueError("Wrong input for fixed")

        # Sort second and third, so it doesn't matter how it was provided.
        fixed = np.array([fixed[0], max(fixed[1:]), min(fixed[1:])])

        # Check side.
        if np.sign(np.diff(fixed[:2])) == np.sign(np.diff(fixed[::2])):
            print("\n* ERROR   :: 2nd and 3rd fixed boundaries have to be "
                  "left and right of the first one.\n             "
                  f"Provided: [{fixed[0]}, {fixed[1]}, {fixed[2]}]")
            raise ValueError("Wrong input for fixed")

    # Calculate skin depth.
    skind = 503.3*np.sqrt(res_arr/abs(freq))
    if freq < 0:  # For Laplace-domain calculations.
        skind /= np.sqrt(2*np.pi)

    # Minimum cell width.
    dmin = skind[0]/pps
    if min_width is not None:  # Respect user input.
        min_width = np.array(min_width, ndmin=1)
        if min_width.size == 1:
            dmin = min_width
        else:
            dmin = np.clip(dmin, *min_width)

    # Survey domain.
    domain = np.array(domain, dtype=float)

    # Calculation domain.
    calc_domain = skind[1:]*np.array([6., 6.])  # 6 x sd => buffer zone.
    calc_domain[0] = domain[0] - calc_domain[0]
    calc_domain[1] = domain[1] + calc_domain[1]

    # Initiate flag if terminated.
    finished = False

    # Initiate alpha variables for survey and calculation domains.
    sa, ca = 1.0, 1.0

    # Loop over possible cell numbers from small to big.
    for nx in np.unique(possible_nx):

        # Loop over possible alphas for domain.
        for sa in np.arange(1.0, alpha[0]+alpha[2]/2, alpha[2]):

            # Get current stretched grid cell sizes.
            thxl = dmin*sa**np.arange(nx)  # Left of origin.
            thxr = dmin*sa**np.arange(nx)  # Right of origin.

            # 0. Adjust stretching for fixed boundaries.
            if fixed.size > 1:  # Move mesh to first fixed boundary.
                t_nx = np.r_[fixed[0], fixed[0]+np.cumsum(thxr)]
                ii = np.argmin(abs(t_nx-fixed[1]))
                thxr *= abs(fixed[1]-fixed[0])/np.sum(thxr[:ii])

            if fixed.size > 2:  # Move mesh to second fixed boundary.
                t_nx = np.r_[fixed[0], fixed[0]-np.cumsum(thxl)]
                ii = np.argmin(abs(t_nx-fixed[2]))
                thxl *= abs(fixed[2]-fixed[0])/np.sum(thxl[:ii])

            # 1. Fill from center to left domain.
            nl = np.sum((fixed[0]-np.cumsum(thxl)) > domain[0])+1

            # 2. Fill from center to right domain.
            nr = np.sum((fixed[0]+np.cumsum(thxr)) < domain[1])+1

            # 3. Get remaining number of cells and check termination criteria.
            nsdc = nl+nr  # Number of domain cells.
            nx_remain = nx-nsdc

            # Not good, try next.
            if nx_remain <= 0:
                continue

            # Create the current hx-array.
            hx = np.r_[thxl[:nl][::-1], thxr[:nr]]
            hxo = np.r_[thxl[:nl][::-1], thxr[:nr]]

            # Get actual domain:
            asurv_domain = [fixed[0]-np.sum(thxl[:nl]),
                            fixed[0]+np.sum(thxr[:nr])]
            x0 = float(fixed[0]-np.sum(thxl[:nl]))

            # Get actual stretching (differs in case of fixed layers).
            sa_adj = np.max([hx[1:]/hx[:-1], hx[:-1]/hx[1:]])

            # Loop over possible alphas for calc_domain.
            for ca in np.arange(sa, alpha[1]+alpha[2]/2, alpha[2]):

                # 4. Fill to left calc_domain.
                thxl = hx[0]*ca**np.arange(1, nx_remain+1)
                nl = np.sum((asurv_domain[0]-np.cumsum(thxl)) >
                            calc_domain[0])+1

                # 5. Fill to right calc_domain.
                thxr = hx[-1]*ca**np.arange(1, nx_remain+1)
                nr = np.sum((asurv_domain[1]+np.cumsum(thxr)) <
                            calc_domain[1])+1

                # 6. Get remaining number of cells and check termination
                # criteria.
                ncdc = nl+nr  # Number of calc_domain cells.
                nx_remain2 = nx-nsdc-ncdc

                if nx_remain2 < 0:  # Not good, try next.
                    continue

                # Create hx-array.
                nl += int(np.floor(nx_remain2/2))  # If uneven, add one cell
                nr += int(np.ceil(nx_remain2/2))   # more on the right.
                hx = np.r_[thxl[:nl][::-1], hx, thxr[:nr]]

                # Calculate origin.
                x0 = float(asurv_domain[0]-np.sum(thxl[:nl]))

                # Mark it as finished and break out of the loop.
                finished = True
                break

            if finished:
                break

        if finished:
            break

    # Check finished and print info about found grid.
    if not finished:
        # Throw message if no solution was found.
        print("\n* ERROR   :: No suitable grid found; relax your criteria.\n")
        if raise_error:
            raise ArithmeticError("No grid found!")
        else:
            hx, x0 = None, None

    elif verb > 0:
        print(f"   Skin depth ", end="")
        if res.size == 1:
            print(f"         [m] : {skind[0]:.0f}")
        elif res.size == 2:
            print(f"(m/l-r)  [m] : {skind[0]:.0f} / {skind[1]:.0f}")
        else:
            print(f"(m/l/r)  [m] : {skind[0]:.0f} / {skind[1]:.0f} / "
                  f"{skind[2]:.0f}")
        print(f"   Survey domain       [m] : {domain[0]:.0f} - "
              f"{domain[1]:.0f}")
        print(f"   Calculation domain  [m] : {calc_domain[0]:.0f} - "
              f"{calc_domain[1]:.0f}")
        print(f"   Final extent        [m] : {x0:.0f} - "
              f"{x0+np.sum(hx):.0f}")
        extstr = f"   Min/max cell width  [m] : {min(hx):.0f} / "
        alstr = f"   Alpha survey"
        nrstr = "   Number of cells "
        if not np.isclose(sa, sa_adj):
            sastr = f"{sa:.3f} ({sa_adj:.3f})"
        else:
            sastr = f"{sa:.3f}"
        print(extstr+f"{max(hxo):.0f} / {max(hx):.0f}")
        print(alstr+f"/calc       : {sastr} / {ca:.3f}")
        print(nrstr+f"(s/c/r) : {nx} ({nsdc}/{ncdc}/{nx_remain2})")
        print()

    if return_info:
        if not fixed.size > 1:
            sa_adj = sa

        info = {'dmin': dmin,
                'dmax': np.nanmax(hx),
                'amin': np.nanmin([ca, sa, sa_adj]),
                'amax': np.nanmax([ca, sa, sa_adj])}

        return hx, x0, info
    else:
        return hx, x0


def get_cell_numbers(max_nr, max_prime=5, min_div=3):
    r"""Returns 'good' cell numbers for the multigrid method.

    'Good' cell numbers are numbers which can be divided by 2 as many times as
    possible. At the end there will be a low prime number.

    The function adds all numbers :math:`p 2^n \leq M` for :math:`p={2, 3, ...,
    p_\text{max}}` and :math:`n={n_\text{min}, n_\text{min}+1, ..., \infty}`;
    :math:`M, p_\text{max}, n_\text{min}` correspond to ``max_nr``,
    ``max_prime``, and ``min_div``, respectively.


    Parameters
    ----------
    max_nr : int
        Maximum number of cells.

    max_prime : int
        Highest permitted prime number p for p*2^n. {2, 3, 5, 7} are good upper
        limits in order to avoid too big lowest grids in the multigrid method.
        Default is 5.

    min_div : int
        Minimum times the number can be divided by two.
        Default is 3.


    Returns
    -------
    numbers : array
        Array containing all possible cell numbers from lowest to highest.

    """
    # Primes till 20.
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19])

    # Sanity check; 19 is already ridiculously high.
    if max_prime > primes[-1]:
        print(f"* ERROR   :: Highest prime is {max_prime}, "
              "please use a value < 20.")
        raise ValueError("Highest prime too high")

    # Restrict to max_prime.
    primes = primes[primes <= max_prime]

    # Get possible values.
    # Currently restricted to prime*2**30 (for prime=2 => 1,073,741,824 cells).
    numbers = primes[:, None]*2**np.arange(min_div, 30)

    # Get unique values.
    numbers = np.unique(numbers)

    # Restrict to max_nr and return.
    return numbers[numbers <= max_nr]


def get_stretched_h(min_width, domain, nx, x0=0, x1=None, resp_domain=False):
    """Return cell widths for a stretched grid within the domain.

    Returns ``nx`` cell widths within ``domain``, where the minimum cell width
    is ``min_width``. The cells are not stretched within ``x0`` and ``x1``, and
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


    See Also
    --------
    get_hx_x0 : Get ``hx`` and ``x0`` for a flexible number of ``nx`` with
                given bounds.


    Parameters
    ----------

    min_width : float
        Minimum cell width. If x1 is provided, the actual minimum cell width
        might be smaller than min_width.

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
        location at or after ``x1``. ``x1`` is restricted to ``domain``. This
        will min_width so that an integer number of cells fit within x0 and x1.

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
    min_width = np.array(min_width, dtype=float)
    if x1 is not None:
        x1 = np.array(x1, dtype=float)
        x1 = np.clip(x1, *domain)  # Restrict to model domain

    # If x1 is provided (a part is not stretched)
    if x1 is not None:

        # Store original values
        xlim_orig = domain.copy()
        nx_orig = int(nx)
        x0_orig = x0.copy()
        h_min_orig = min_width.copy()

        # Get number of non-stretched cells
        n_nos = int(np.ceil((x1-x0)/min_width))

        # Re-calculate min_width to fit with x0-x1-limits:
        min_width = (x1-x0)/n_nos

        # Subtract one cell, because the standard scheme provides one
        # min_width-cell.
        n_nos -= 1

        # Reset x0, because the first min_width comes from normal scheme
        x0 += min_width

        # Reset xmax for normal scheme
        domain[1] -= n_nos*min_width

        # Reset nx for normal scheme
        nx -= n_nos

        # If there are not enough points reset to standard procedure. The limit
        # of five is arbitrary. However, nx should be much bigger than five
        # anyways, otherwise stretched grid doesn't make sense.
        if nx <= 5:
            print("Warning :: Not enough points for non-stretched part,"
                  "ignoring therefore `x1`.")
            domain = xlim_orig
            nx = nx_orig
            x0 = x0_orig
            x1 = None
            min_width = h_min_orig

    # Get stretching factor (a = 1+alpha).
    if min_width == 0 or min_width > np.diff(domain)/nx:
        # If min_width is bigger than the domain-extent divided by nx, no
        # stretching is required at all.
        alpha = 0
    else:

        # Wrap _get_dx into a minimization function to call with fsolve.
        def find_alpha(alpha, min_width, args):
            """Find alpha such that min(hx) = min_width."""
            return min(get_hx(alpha, *args))/min_width-1

        # Search for best alpha, must be at least 0
        args = (domain, nx, x0)
        alpha = max(0, optimize.fsolve(find_alpha, 0.02, (min_width, args)))

    # With alpha get actual cell spacing with `resp_domain` to respect the
    # users decision.
    hx = get_hx(alpha, domain, nx, x0, resp_domain)

    # Add the non-stretched center if x1 is provided
    if x1 is not None:
        hx = np.r_[hx[: np.argmin(hx)], np.ones(n_nos)*min_width,
                   hx[np.argmin(hx):]]

    # Print warning min_width could not be respected.
    if abs(hx.min() - min_width) > 0.1:
        print(f"Warning :: Minimum cell width ({np.round(hx.min(), 2)} m) is "
              "below `min_width`, because `nx` is too big for `domain`.")

    return hx


def get_domain(x0=0, freq=1, res=0.3, limits=None, min_width=None,
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


    Parameters
    ----------

    x0 : float
        Center of the calculation domain. Normally the source location.
        Default is 0.

    freq : float
        Frequency (Hz) to calculate the skin depth. The skin depth is a concept
        defined in the frequency domain. If a negative frequency is provided,
        it is assumed that the calculation is carried out in the Laplace
        domain. To calculate the skin depth, the value of ``freq`` is then
        multiplied by :math:`-2\pi`, to simulate the closest
        frequency-equivalent.

        Default is 1 Hz.

    res : float, optional
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
    skind = 503.3*np.sqrt(res/abs(freq))
    if freq < 0:  # For Laplace-domain calculations.
        skind /= np.sqrt(2*np.pi)

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


def get_hx(alpha, domain, nx, x0, resp_domain=True):
    r"""Return cell widths for given input.

    Find the number of cells left and right of ``x0``, ``nl`` and ``nr``
    respectively, for the provided alpha. For this, we solve

    .. math::   \frac{x_\text{max}-x_0}{x_0-x_\text{min}} =
                \frac{a^{nr}-1}{a^{nl}-1}

    where :math:`a = 1+\alpha`.


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
        if np.isclose(x0, domain[0]) or np.isclose(x0, domain[1]):
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
                # h = [(min_width, nl-1, -a), (min_width, n_nos+1),
                #      (min_width, nr, a)]

    return hx


# TIME DOMAIN
class Fourier:
    r"""Time-domain CSEM calculation.

    Class to carry out time-domain modelling with the frequency-domain code
    ``emg3d``. Instances of the class take care of calculating the required
    frequencies, the interpolation from coarse, limited-band frequencies to the
    required frequencies, and carrying out the actual transform.

    Everything related to the Fourier transform is done by utilising the
    capabilities of the 1D modeller :mod:`empymod`. The input parameters
    ``time``, ``signal``, ``ft``, and ``ftarg`` are passed to the function
    :func:`empymod.utils.check_time` to obtain the required frequencies. The
    actual transform is subsequently carried out by calling
    :func:`empymod.model.tem`. See these functions for more details about the
    exact implementations of the Fourier transforms and its parameters.
    Note that also the ``verb``-argument follows the definition in ``empymod``.

    The mapping from calculated frequencies to the frequencies required for the
    Fourier transform is done in three steps:

    - Data for :math:`f>f_\mathrm{max}` is set to 0+0j.
    - Data for :math:`f<f_\mathrm{min}` is interpolated by adding an additional
      data point at a frequency of 1e-100 Hz. The data for this point is
      ``data.real[0]+0j``, hence the real part of the lowest calculated
      frequency and zero imaginary part. Interpolation is carried out using
      PCHIP :func:`scipy.interpolate.pchip_interpolate`.
    - Data for :math:`f_\mathrm{min}\le f \le f_\mathrm{max}` is calculated
      with cubic spline interpolation (on a log-scale)
      :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

    Note that ``fmin`` and ``fmax`` should be chosen wide enough such that the
    mapping for :math:`f>f_\mathrm{max}` :math:`f<f_\mathrm{min}` does not
    matter that much.


    Parameters
    ----------

    time : ndarray
        Desired times (s).

    fmin, fmax : float
        Minimum and maximum frequencies (Hz) to calculate:

          - Data for freq > fmax is set to 0+0j.
          - Data for freq < fmin is interpolated, using an extra data-point at
            f = 1e-100 Hz, with value data.real[0]+0j. (Hence zero imaginary
            part, and the lowest calculated real value.)

    signal : {0, 1, -1}, optional
        Source signal, default is 0:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    ft : {'sin', 'cos', 'fftlog'}, optional
        Flag to choose either the Digital Linear Filter method (Sine- or
        Cosine-Filter) or the FFTLog for the Fourier transform.
        Defaults to 'sin'.

    ftarg : dict, optional
        Depends on the value for ``ft``:
            - If ``ft`` = 'sin' or 'cos':

                - fftfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (Default: ``empymod.filters.key_201_CosSin_2012()``)
                - pts_per_dec: points per decade; (default: -1)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF

            - If ``ft`` = 'fftlog':

                - pts_per_dec: sampels per decade (default: 10)
                - add_dec: additional decades [left, right] (default: [-2, 1])
                - q: exponent of power law bias (default: 0); -1 <= q <= 1

    freq_inp : array
        Frequencies to use for calculation. Mutually exclusive with
        `every_x_freq`.

    every_x_freq : int
        Every `every_x_freq`-th frequency of the required frequency-range is
        used for calculation. Mutually exclusive with `freq_calc`.


    """

    def __init__(self, time, fmin, fmax, signal=0, ft='sin', ftarg=None,
                 **kwargs):
        """Initialize a Fourier instance."""

        # Store the input parameters.
        self._time = time
        self._fmin = fmin
        self._fmax = fmax
        self._signal = signal
        self._ft = ft
        self._ftarg = ftarg

        # Get kwargs.
        self._freq_inp = kwargs.pop('freq_inp', None)
        self._every_x_freq = kwargs.pop('every_x_freq', None)
        self.verb = kwargs.pop('verb', 3)

        # Ensure no kwargs left.
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        # Ensure freq_inp and every_x_freq are not both set.
        self._check_coarse_inputs(keep_freq_inp=True)

        # Get required frequencies.
        self._check_time()

    def __repr__(self):
        """Simple representation."""
        return (f"Fourier: {self._ft}; {self.time.min()}-{self.time.max()} s; "
                f"{self.fmin}-{self.fmax} Hz")

    # PURE PROPERTIES
    @property
    def freq_req(self):
        """Frequencies required to carry out the Fourier transform."""
        return self._freq_req

    @property
    def freq_coarse(self):
        """Coarse frequency range, can be different from ``freq_req``."""
        if self.every_x_freq is None and self.freq_inp is None:
            # If none of {every_x_freq, freq_inp} given, then
            # freq_coarse = freq_req.
            return self.freq_req

        elif self.every_x_freq is None:
            # If freq_inp given, then freq_coarse = freq_inp.
            return self.freq_inp

        else:
            # If every_x_freq given, get subset of freq_req.
            return self.freq_req[::self.every_x_freq]

    @property
    def freq_calc_i(self):
        """Indices of ``freq_coarse`` which have to be calculated."""
        ind = (self.freq_coarse >= self.fmin) & (self.freq_coarse <= self.fmax)
        return ind

    @property
    def freq_calc(self):
        """Frequencies at which the model has to be calculated."""
        return self.freq_coarse[self.freq_calc_i]

    @property
    def freq_extrapolate_i(self):
        """Indices of the frequencies to extrapolate."""
        return self.freq_req < self.fmin

    @property
    def freq_extrapolate(self):
        """These are the frequencies to extrapolate.

        In fact, it is dow via interpolation, using an extra data-point at f =
        1e-100 Hz, with value data.real[0]+0j. (Hence zero imaginary part, and
        the lowest calculated real value.)
        """
        return self.freq_req[self.freq_extrapolate_i]

    @property
    def freq_interpolate_i(self):
        """Indices of the frequencies to interpolate.

        If freq_req is equal freq_coarse, then this is eual to freq_calc_i.
        """
        return (self.freq_req >= self.fmin) & (self.freq_req <= self.fmax)

    @property
    def freq_interpolate(self):
        """These are the frequencies to interpolate.

        If freq_req is equal freq_coarse, then this is eual to freq_calc.
        """
        return self.freq_req[self.freq_interpolate_i]

    @property
    def ft(self):
        """Type of Fourier transform.
        Set via ``fourier_arguments(ft, ftarg)``.
        """
        return self._ft

    @property
    def ftarg(self):
        """Fourier transform arguments.
        Set via ``fourier_arguments(ft, ftarg)``.
        """
        return self._ftarg

    # PROPERTIES WITH SETTERS
    @property
    def time(self):
        """Desired times (s)."""
        return self._time

    @time.setter
    def time(self, time):
        """Update desired times (s)."""
        self._time = time
        self._check_time()

    @property
    def fmax(self):
        """Maximum frequency (Hz) to calculate."""
        return self._fmax

    @fmax.setter
    def fmax(self, fmax):
        """Update maximum frequency (Hz) to calculate."""
        self._fmax = fmax
        self._print_freq_calc()

    @property
    def fmin(self):
        """Minimum frequency (Hz) to calculate."""
        return self._fmin

    @fmin.setter
    def fmin(self, fmin):
        """Update minimum frequency (Hz) to calculate."""
        self._fmin = fmin
        self._print_freq_calc()

    @property
    def signal(self):
        """Signal in time domain {0, 1, -1}."""
        return self._signal

    @signal.setter
    def signal(self, signal):
        """Update signal in time domain {0, 1, -1}."""
        self._signal = signal

    @property
    def freq_inp(self):
        """If set, freq_coarse is set to freq_inp."""
        return self._freq_inp

    @freq_inp.setter
    def freq_inp(self, freq_inp):
        """Update freq_inp. Erases every_x_freq if set."""
        self._freq_inp = freq_inp
        self._check_coarse_inputs(keep_freq_inp=True)

    @property
    def every_x_freq(self):
        """If set, freq_coarse is every_x_freq-frequency of freq_req."""
        return self._every_x_freq

    @every_x_freq.setter
    def every_x_freq(self, every_x_freq):
        """Update every_x_freq. Erases freq_inp if set."""
        self._every_x_freq = every_x_freq
        self._check_coarse_inputs(keep_freq_inp=False)

    # OTHER STUFF
    def fourier_arguments(self, ft, ftarg):
        """Set Fourier type and its arguments."""
        self._ft = ft
        self._ftarg = ftarg
        self._check_time()

    def interpolate(self, fdata):
        """Interpolate from calculated data to required data.

        Parameters
        ----------

        fdata : ndarray
            Frequency-domain data corresponding to `freq_calc`.

        Returns
        -------
        full_data : ndarray
            Frequency-domain data corresponding to `freq_req`.

        """

        # Pre-allocate result.
        out = np.zeros(self.freq_req.size, dtype=complex)

        # 1. Interpolate between fmin and fmax.

        # If freq_coarse is not exactly freq_req, we use cubic spline to
        # interpolate from fmin to fmax.
        if self.freq_coarse.size != self.freq_req.size:

            int_real = Spline(np.log(self.freq_calc),
                              fdata.real)(np.log(self.freq_interpolate))
            int_imag = Spline(np.log(self.freq_calc),
                              fdata.imag)(np.log(self.freq_interpolate))

            out[self.freq_interpolate_i] = int_real + 1j*int_imag

        else:  # If they are the same, just fill in the data.
            out[self.freq_interpolate_i] = fdata

        # 2. Extrapolate from freq_req.min to fmin using PCHIP.

        # 2.a Extend freq_req/data by adding a point at 1e-100 Hz with
        # - same real part as lowest calculated frequency and
        # - zero imaginary part.
        freq_ext = np.r_[1e-100, self.freq_calc]
        data_ext = np.r_[fdata[0].real+0.0j, fdata]

        # 2.b Actual 'extrapolation' (now an interpolation).
        ext_real = Pchip(freq_ext, data_ext.real)(self.freq_extrapolate)
        ext_imag = Pchip(freq_ext, data_ext.imag)(self.freq_extrapolate)

        out[self.freq_extrapolate_i] = ext_real + 1j*ext_imag

        return out

    def freq2time(self, fdata, off):
        """Calculate corresponding time-domain signal.

        Carry out the actual Fourier transform.

        Parameters
        ----------

        fdata : ndarray
            Frequency-domain data corresponding to `freq_calc`.

        off : float
            Corresponding offset (m).

        Returns
        -------
        tdata : ndarray
            Time-domain data corresponding to Fourier.time.

        """
        # Interpolate the calculated data at the required frequencies.
        inp_data = self.interpolate(fdata)

        # Carry out the Fourier transform.
        tdata, _ = empymod.model.tem(
                inp_data[:, None], np.array(off), freq=self.freq_req,
                time=self.time, signal=self.signal, ft=self.ft,
                ftarg=self.ftarg)

        return np.squeeze(tdata)

    # PRIVATE ROUTINES
    def _check_time(self):
        """Get required frequencies for given times and ft/ftarg."""

        # Get freq via empymod.
        _, freq, ft, ftarg = empymod.utils.check_time(
            self.time, self.signal, self.ft, self.ftarg, self.verb)

        # Store required frequencies and check ft, ftarg.
        self._freq_req = freq
        self._ft = ft
        self._ftarg = ftarg

        # Print frequency information (if verbose).
        if self.verb > 2:
            self._print_freq_ftarg()
            self._print_freq_calc()

    def _check_coarse_inputs(self, keep_freq_inp=True):
        """Parameters `freq_inp` and `every_x_freq` are mutually exclusive."""

        # If they are both set, reset one depending on `keep_freq_inp`.
        if self._freq_inp is not None and self._every_x_freq is not None:
            print("\n* WARNING :: `freq_inp` and `every_x_freq` are mutually "
                  "exclusive.\n             Re-setting ", end="")

            if keep_freq_inp:  # Keep freq_inp.
                print("`every_x_freq=None`.\n")
                self._every_x_freq = None

            else:              # Keep every_x_freq.
                print("`freq_inp=None`.\n")
                self._freq_inp = None

    # PRINTING ROUTINES
    def _print_freq_ftarg(self):
        """Print required frequency range."""
        if self.verb > 2:
            empymod.utils._prnt_min_max_val(
                    self.freq_req, "   Req. freq  [Hz] : ", self.verb)

    def _print_freq_calc(self):
        """Print actually calculated frequency range."""
        if self.verb > 2:
            empymod.utils._prnt_min_max_val(
                    self.freq_calc, "   Calc. freq [Hz] : ", self.verb)


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

            # If the parameter is a TensorMesh-instance, we set the volume
            # None. This saves space, and it will simply be reconstructed if
            # required.
            if type(values[i]).__name__ == 'TensorMesh':
                if hasattr(values[i], '_vol'):
                    delattr(values[i], '_vol')

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


# TIMING AND REPORTING
class Time:
    """Class for timing (now; runtime)."""

    def __init__(self):
        """Initialize time zero (t0) with current time stamp."""
        self._t0 = default_timer()

    def __repr__(self):
        """Simple representation."""
        return f"Runtime : {self.runtime}"

    @property
    def t0(self):
        """Return time zero of this class instance."""
        return self._t0

    @property
    def now(self):
        """Return string of current time."""
        return datetime.now().strftime("%H:%M:%S")

    @property
    def runtime(self):
        """Return string of runtime since time zero."""
        return str(timedelta(seconds=np.round(self.elapsed)))

    @property
    def elapsed(self):
        """Return runtime in seconds since time zero."""
        return default_timer() - self._t0


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
