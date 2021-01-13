"""
Everything related to the multigrid solver that is a field: source field,
electric and magnetic fields, and fields at receivers.
"""
# Copyright 2018-2021 The emg3d Developers.
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

from copy import deepcopy

import numpy as np
from scipy.constants import mu_0
from scipy.special import sindg, cosdg

from emg3d import maps, models, utils

__all__ = ['Field', 'SourceField', 'get_source_field', 'get_receiver',
           'get_receiver_response', 'get_h_field']


class Field(np.ndarray):
    r"""Create a Field instance with x-, y-, and z-views of the field.

    A `Field` is an `ndarray` with additional views of the x-, y-, and
    z-directed fields as attributes, stored as `fx`, `fy`, and `fz`. The
    default array contains the whole field, which can be the electric field,
    the source field, or the residual field, in a 1D array. A `Field` instance
    has additionally the property `ensure_pec` which, if called, ensures
    Perfect Electric Conductor (PEC) boundary condition. It also has the two
    attributes `amp` and `pha` for the amplitude and phase, as common in
    frequency-domain CSEM.

    A `Field` can be initiated in three ways:

    1. ``Field(grid, dtype=np.complex128)``:
       Calling it with a :class:`emg3d.meshes.TensorMesh` instance returns a
       `Field` instance of correct dimensions initiated with zeroes of data
       type `dtype`.
    2. ``Field(grid, field)``:
       Calling it with a :class:`emg3d.meshes.TensorMesh` instance and an
       `ndarray` returns a `Field` instance of the provided `ndarray`, of same
       data type.
    3. ``Field(fx, fy, fz)``:
       Calling it with three `ndarray`'s which represent the field in x-, y-,
       and z-direction returns a `Field` instance with these views, of same
       data type.

    Sort-order is 'F'.


    Parameters
    ----------

    fx_or_grid : :class:`emg3d.meshes.TensorMesh` or ndarray
        Either a TensorMesh instance or an ndarray of shape grid.nEx or
        grid.vnEx. See explanations above. Only mandatory parameter; if the
        only one provided, it will initiate a zero-field of `dtype`.

    fy_or_field : :class:`Field` or ndarray, optional
        Either a Field instance or an ndarray of shape grid.nEy or grid.vnEy.
        See explanations above.

    fz : ndarray, optional
        An ndarray of shape grid.nEz or grid.vnEz. See explanations above.

    dtype : dtype, optional
        Only used if ``fy_or_field=None`` and ``fz=None``; the initiated
        zero-field for the provided TensorMesh has data type `dtype`.
        Default: complex.

    freq : float, optional
        Source frequency (Hz), used to compute the Laplace parameter `s`.
        Either positive or negative:

        - `freq` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - `freq` < 0: Laplace domain, hence
          :math:`s = f` (real).

        Just added as info if provided.

    """

    def __new__(cls, fx_or_grid, fy_or_field=None, fz=None,
                dtype=np.complex128, freq=None):
        """Initiate a new Field instance."""

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

            # Ensure the grid has three dimensions.
            # (Can happend with a 1D or 2D discretize mesh.)
            if None in [obj.nEx, obj.nEy, obj.nEz]:
                raise ValueError("Provided grid must be a 3D grid.")

        # Store frequency
        if freq is None and hasattr(fy_or_field, 'freq'):
            freq = fy_or_field._freq
        obj._freq = freq
        if freq == 0.0:
            raise ValueError(
                    "`freq` must be >0 (frequency domain) "
                    "or <0 (Laplace domain).\n"
                    f"Provided frequency: {freq} Hz.")

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

    def __reduce__(self):
        """Customize __reduce__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Get the parent's __reduce__ tuple.
        pickled_state = super().__reduce__()

        # Create our own tuple to pass to __setstate__.
        new_state = pickled_state[2]
        attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz', '_freq']
        for attr in attr_list:
            new_state += (getattr(self, attr),)

        # Return tuple that replaces parent's __setstate__ tuple with our own.
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Customize __setstate__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Set the necessary attributes (in reverse order).
        attr_list = ['nEx', 'nEy', 'nEz', 'vnEx', 'vnEy', 'vnEz', '_freq']
        attr_list.reverse()
        for i, name in enumerate(attr_list):
            i += 1  # We need it 1..#attr instead of 0..#attr-1.
            setattr(self, name, state[-i])

        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-i])

    def copy(self):
        """Return a copy of the Field."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Field in a dict."""
        out = {'field': np.array(self.field), 'freq': self._freq,
               'vnEx': self.vnEx, 'vnEy': self.vnEy, 'vnEz': self.vnEz,
               '__class__': self.__class__.__name__}
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Field` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Field.to_dict`.
            The dictionary needs the keys `field`, `freq`, `vnEx`, `vnEy`, and
            `vnEz`.

        Returns
        -------
        obj : :class:`Field` instance

        """

        # Create a dummy with the required attributes for the field instance.
        class Grid:
            pass

        grid = Grid()

        # Check and get the required keys from the input.
        try:
            field = inp['field']
            freq = inp['freq']
            grid.vnEx = inp['vnEx']
            grid.vnEy = inp['vnEy']
            grid.vnEz = inp['vnEz']
        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.") from e

        # Compute missing info.
        grid.nEx = np.prod(grid.vnEx)
        grid.nEy = np.prod(grid.vnEy)
        grid.nEz = np.prod(grid.vnEz)
        grid.nE = grid.nEx + grid.nEy + grid.nEz

        # Return Field instance.
        return cls(fx_or_grid=grid, fy_or_field=field, freq=freq)

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

    def amp(self):
        """Amplitude of the electromagnetic field."""
        return utils.EMArray(self.view()).amp()

    def pha(self, deg=False, unwrap=True, lag=True):
        """Phase of the electromagnetic field.

        Parameters
        ----------
        deg : bool
            If True the returned phase is in degrees, else in radians.
            Default is False (radians).

        unwrap : bool
            If True the returned phase is unwrapped.
            Default is True (unwrapped).

        lag : bool
            If True the returned phase is lag, else lead defined.
            Default is True (lag defined).

        """
        return utils.EMArray(self.view()).pha(deg, unwrap, lag)

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
        if getattr(self, '_smu0', None) is None:
            if self.sval is not None:
                self._smu0 = self.sval*mu_0
            else:
                self._smu0 = None

        return self._smu0

    @property
    def sval(self):
        """Return s; s=iw in frequency domain; s=freq in Laplace domain."""

        if getattr(self, '_sval', None) is None:
            if self._freq is not None:
                if self._freq < 0:  # Laplace domain; s.
                    self._sval = np.array(self._freq)
                else:  # Frequency domain; s = iw = 2i*pi*f.
                    self._sval = np.array(-2j*np.pi*self._freq)
            else:
                self._sval = None

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
    source vector (`vector`, `vx`, `vy`, `vz`), which sum is always one. For a
    `SourceField` frequency is a mandatory  parameter, unlike for a `Field`
    (recommended also for `Field` though),

    Parameters
    ----------

    fx_or_grid : :class:`emg3d.meshes.TensorMesh` or ndarray
        Either a TensorMesh instance or an ndarray of shape grid.nEx or
        grid.vnEx. See explanations above. Only mandatory parameter; if the
        only one provided, it will initiate a zero-field of `dtype`.

    fy_or_field : :class:`Field` or ndarray, optional
        Either a Field instance or an ndarray of shape grid.nEy or grid.vnEy.
        See explanations above.

    fz : ndarray, optional
        An ndarray of shape grid.nEz or grid.vnEz. See explanations above.

    dtype : dtype, optional
        Only used if ``fy_or_field=None`` and ``fz=None``; the initiated
        zero-field for the provided TensorMesh has data type `dtype`.
        Default: complex.

    freq : float
        Source frequency (Hz), used to compute the Laplace parameter `s`.
        Either positive or negative:

        - `freq` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - `freq` < 0: Laplace domain, hence
          :math:`s = f` (real).

        In difference to `Field`, the frequency has to be provided for
        a `SourceField`.

    """

    def __new__(cls, fx_or_grid, fy_or_field=None, fz=None,
                dtype=np.complex128, freq=None):
        """Initiate a new Source Field."""
        # Ensure frequency is provided.
        if freq is None:
            raise ValueError("SourceField requires the frequency.")

        if freq > 0:
            dtype = complex
        else:
            dtype = float

        return super().__new__(cls, fx_or_grid, fy_or_field=fy_or_field,
                               fz=fz, dtype=dtype, freq=freq)

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


def get_source_field(grid, src, freq, strength=0, msrc=False, length=1.0,
                     decimals=6):
    r"""Return the source field.

    The source field is given in Equation 2 in [Muld06]_,

    .. math::

        s \mu_0 \mathbf{J}_\mathrm{s} ,

    where :math:`s = \mathrm{i} \omega`. Either finite length dipoles,
    infinitesimal small point dipoles, or arbitrarily shaped segments can be
    defined, whereas the returned source field corresponds to a normalized
    (1 Am) source distributed within the cell(s) it resides (can be changed
    with the `strength`-parameter).

    The adjoint of the trilinear interpolation is used to distribute the
    point(s) to the grid edges, which corresponds to the discretization of a
    Dirac ([PlDM07]_).


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a :class:`emg3d.meshes.TensorMesh` instance.

    src : list of floats
        Source coordinates (m). There are three formats:

          - Finite length dipole: ``[x0, x1, y0, y1, z0, z1]``.
          - Point dipole: ``[x, y, z, azimuth, dip]``.
          - Arbitrarily shaped source: ``[[x-coo], [y-coo], [z-coo]]``.

        Point dipoles will be converted internally to finite length dipoles
        of ``length``. In the case of a point dipole one can set ``msrc=True``,
        which will create a square loop of ``length``x``length`` perpendicular
        to the dipole.

    freq : float
        Source frequency (Hz), used to compute the Laplace parameter `s`.
        Either positive or negative:

        - `freq` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - `freq` < 0: Laplace domain, hence
          :math:`s = f` (real).

    strength : float or complex, optional
        Source strength (A):

          - If 0, output is normalized to a source of 1 m length, and source
            strength of 1 A.
          - If != 0, output is returned for given source length and strength.

        Default is 0.

    msrc : bool, optional
        Shortcut to create a magnetic source. If True, the format of ``src``
        must be that of a point dipole: ``[x, y, z, azimuth, dip]`` (for the
        other formats setting ``msrc`` has no effect). It then creates a square
        loop perpendicular to this dipole, with side-length 1.

    length : float, optional
        Length (m) of the point dipole when converted to a finite length
        dipole, or edge-length (m) of the square loop if ``msrc=True``.
        Default is 1.0.

    decimals: int, optional
        Grid nodes and source coordinates are rounded to given number of
        decimals. Default is 6 (micrometer).


    Returns
    -------
    sfield : :func:`SourceField` instance
        Source field, normalized to 1 A m.

    """
    # Cast some parameters.
    if not np.allclose(np.size(src[0]), [np.size(c) for c in src]):
        raise ValueError(
                "All source coordinates must have the same dimension."
                f"Provided source: {src}.")

    src = np.asarray(src, dtype=np.float64)
    strength = np.asarray(strength)

    # Convert arbitrary shapes and point dipoles to finite length dipoles.
    if len(src) == 5 and not msrc:  # Point dipole: convert to finite length.

        src = _finite_dipole_from_point_dipole(src, length)

    elif len(src) in [3, 5]:  # Arbitrary shapes.

        # Get points of square loop perp. to dipole in case of msrc.
        if len(src) == 5:
            src = _square_loop_from_point_dipole(src, length)

        # Get arbitrarily shaped dipole source using recursion.
        sx, sy, sz = src

        # Get normalized segment lengths.
        lengths = np.sqrt(np.sum((src[:, :-1] - src[:, 1:])**2, axis=0))
        if strength == 0:
            lengths /= lengths.sum()
        else:
            lengths *= strength

        # Initiate a zero-valued source field and loop over segments.
        sfield = SourceField(grid, freq=freq)
        sfield.src = src
        sfield.strength = strength
        sfield.moment = np.array([0., 0, 0])

        # Loop over elements.
        for i in range(sx.size-1):
            segment = (sx[i], sx[i+1], sy[i], sy[i+1], sz[i], sz[i+1])
            seg_field = get_source_field(grid, segment, freq, lengths[i])
            sfield += seg_field
            sfield.moment += seg_field.moment

        return sfield

    # From here onwards `src` has to be a finite length dipole  of format
    # [x1, x2, y1, y2, z1, z2]. Ensure that:
    if src.shape != (6, ):
        raise ValueError(
                "Source is wrong defined. It must be either\n- a point, "
                "[x, y, z, azimuth, dip],\n- a finite dipole, "
                "[x1, x2, y1, y2, z1, z2], or\n- an arbitrarily shaped "
                "dipole, [[x-coo], [y-coo], [z-coo]].\n"
                f"Provided source: {src}.")

    # Get length in each direction.
    length = src[1::2]-src[::2]

    # Ensure finite length dipole is not a point dipole.
    if np.allclose(length, 0, atol=1e-15):
        raise ValueError("Provided finite dipole has no length; use "
                         "the format [x, y, z, azimuth, dip] instead.")

    # Get source moment (individually for x, y, z).
    if strength == 0:  # 1 A m
        length /= np.linalg.norm(length)
        moment = length
    else:              # Multiply source length with source strength
        moment = strength*length

    # Initiate zero source field.
    sfield = SourceField(grid, freq=freq)

    # Return source-field for each direction.
    for xyz, sf in enumerate([sfield.fx, sfield.fy, sfield.fz]):

        # Get source field for this direction.
        _finite_source_xyz(grid, src, sf, xyz, decimals)

        # Multiply by moment*s*mu
        sf *= moment[xyz]*sfield.smu0

    # Add src and moment information.
    sfield.src = src
    sfield.strength = strength
    sfield.moment = moment

    return sfield


def get_receiver(grid, values, coordinates, method='cubic', extrapolate=False):
    """Return values corresponding to grid at coordinates.

    Works for electric fields as well as magnetic fields obtained with
    :func:`get_h_field`, and for model parameters.

    Note that in order to avoid boundary effects the first and last value in
    each direction is neglected. Values for coordinates outside of the grid
    are set to NaN's, unless extrapolate is set.


    Parameters
    ----------
    grid : :class:`emg3d.meshes.TensorMesh`
        The model grid.

    values : ndarray
        Field instance, or a particular field (e.g. field.fx); Model
        parameters.

    coordinates : tuple (x, y, z)
        Coordinates (x, y, z) where to interpolate `values`; e.g. receiver
        locations.

    method : str, optional
        The method of interpolation to perform, 'linear' or 'cubic'.
        Default is 'cubic' (forced to 'linear' if there are less than 3 points
        in any direction).

    extrapolate : bool
        If True, points on `new_grid` which are outside of `grid` are
        filled by the nearest value (if ``method='cubic'``) or by extrapolation
        (if ``method='linear'``). If False, points outside are set to NaN's.

        Default is False.


    Returns
    -------
    new_values : ndarray or :class:`utils.EMArray`
        Values at `coordinates`.

        If input was a field it returns an EMArray, which is a subclassed
        ndarray with ``.pha`` and ``.amp`` attributes.

        If input was an entire Field instance, output is a tuple (fx, fy, fz).


    See Also
    --------
    grid2grid : Interpolation of model parameters or fields to a new grid.
    get_receiver_response : Get response for arbitrarily rotated receivers.

    """
    # If values is a Field instance, call it recursively for each field.
    if hasattr(values, 'field') and values.field.ndim == 1:
        fx = get_receiver(grid, values.fx, coordinates, method, extrapolate)
        fy = get_receiver(grid, values.fy, coordinates, method, extrapolate)
        fz = get_receiver(grid, values.fz, coordinates, method, extrapolate)
        return fx, fy, fz

    if len(coordinates) != 3:
        raise ValueError(
                "Coordinates needs to be in the form (x, y, z).\n"
                f"Length of provided coord.: {len(coordinates)}.")

    # Get the vectors corresponding to input data. Dimensions:
    #
    #         E-field          H-field      |  Model Parameter
    #  x: [nCx, nNy, nNz]  [nNx, nCy, nCz]  |
    #  y: [nNx, nCy, nNz]  [nCx, nNy, nCz]  |  [nCx, nCy, nCz]
    #  z: [nNx, nNy, nCz]  [nCx, nCy, nNz]  |
    #
    points = tuple()
    for i, coord in enumerate(['x', 'y', 'z']):
        if values.shape[i] == getattr(grid, 'shape_nodes')[i]:
            pts = (getattr(grid, 'nodes_'+coord)[1:-1], )
        else:
            pts = (getattr(grid, 'cell_centers_'+coord)[1:-1], )

        # Add to points.
        points += pts

    if extrapolate:
        fill_value = None
        mode = 'nearest'
    else:
        fill_value = np.array(0, values.dtype)*np.nan
        mode = 'constant'
    out = maps.interp3d(points, values[1:-1, 1:-1, 1:-1], coordinates,
                        method, fill_value, mode, cval=np.nan)

    # Return an EMArray if input is a field, else simply the values.
    if values.size == grid.nC:
        return out
    else:
        return utils.EMArray(out)


def get_receiver_response(grid, field, rec):
    """Return the field (response) at receiver coordinates.

    Note that in order to avoid boundary effects the first and last value in
    each direction is neglected. Field values for coordinates outside of the
    grid are set to NaN's.


    Parameters
    ----------
    grid : :class:`emg3d.meshes.TensorMesh`
        The model grid.

    field : :class:`Field`
        The electric or magnetic field.

    rec : tuple (x, y, z, azimuth, dip)
        Receiver coordinates and angles (m, °).

        All values can either be a scalar or having the same length as number
        of receivers.

        Angles:

        - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
        - dip (°): vertical deviation from xy-plane up-wards.


    Returns
    -------
    responses : :class:`utils.EMArray`
        Responses at receiver.


    .. note::

        Currently only implemented for point receivers, not for finite length
        dipoles.


    See Also
    --------
    get_receiver : Get values at coordinates (fields and models).

    """

    # Check receiver dimension.
    if len(rec) != 5:
        raise ValueError(
                "`rec` needs to be in the form (x, y, z, azimuth, dip).\n"
                f"Length of provided `rec`: {len(rec)}.")

    # Check field dimension to ensure it is not a particular field.
    if field.field.ndim == 3:
        raise ValueError("`field` must be a `Field`-instance, not a\n"
                         "particular field such as `field.fx`.")

    # Get the vectors corresponding to input data.
    if field.is_electric:
        points = ((grid.cell_centers_x, grid.nodes_y, grid.nodes_z),
                  (grid.nodes_x, grid.cell_centers_y, grid.nodes_z),
                  (grid.nodes_x, grid.nodes_y, grid.cell_centers_z))
    else:
        points = ((grid.nodes_x, grid.cell_centers_y, grid.cell_centers_z),
                  (grid.cell_centers_x, grid.nodes_y, grid.cell_centers_z),
                  (grid.cell_centers_x, grid.cell_centers_y, grid.nodes_z))

    # Remove first and last value in each direction.
    points = tuple([tuple([p[1:-1] for p in pp]) for pp in points])

    # Pre-allocate the response.
    resp = np.zeros(max([np.atleast_1d(x).size for x in rec]),
                    dtype=field.dtype)

    # Add the required responses.
    factors = _rotation(*rec[3:])  # Geometrical weights from angles.
    inp = {'method': 'cubic', 'fill_value': 0.0, 'mode': 'constant',
           'cval': np.nan}
    for i, ff in enumerate((field.fx, field.fy, field.fz)):
        if np.any(abs(factors[i]) > 1e-10):
            resp += factors[i]*maps.interp3d(
                       points[i], ff[1:-1, 1:-1, 1:-1], rec[:3], **inp)

    # Return response.
    return utils.EMArray(resp)


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

       E-field:  x: [grid.cell_centers_x,  grid.nodes_y,  grid.nodes_z]
                 y: [ grid.nodes_x, grid.cell_centers_y,  grid.nodes_z]
                 z: [ grid.nodes_x,  grid.nodes_y, grid.cell_centers_z]

       H-field:  x: [ grid.nodes_x, grid.cell_centers_y, grid.cell_centers_z]
                 y: [grid.cell_centers_x,  grid.nodes_y, grid.cell_centers_z]
                 z: [grid.cell_centers_x, grid.cell_centers_y,  grid.nodes_z]


    Parameters
    ----------
    grid : TensorMesh
        Model grid; :class:`TensorMesh` instance.

    model : Model
        Model; :class:`Model` instance.

    field : Field
        Electric field; :class:`Field` instance.


    Returns
    -------
    hfield : Field
        Magnetic field; :class:`Field` instance.

    """

    # Carry out the curl (^ corresponds to differentiation axis):
    # H_x = (E_z^1 - E_y^2)
    e3d_hx = (np.diff(field.fz, axis=1)/grid.h[1][None, :, None] -
              np.diff(field.fy, axis=2)/grid.h[2][None, None, :])

    # H_y = (E_x^2 - E_z^0)
    e3d_hy = (np.diff(field.fx, axis=2)/grid.h[2][None, None, :] -
              np.diff(field.fz, axis=0)/grid.h[0][:, None, None])

    # H_z = (E_y^0 - E_x^1)
    e3d_hz = (np.diff(field.fy, axis=0)/grid.h[0][:, None, None] -
              np.diff(field.fx, axis=1)/grid.h[1][None, :, None])

    # If relative magnetic permeability is not one, we have to take the volume
    # into account, as mu_r is volume-averaged.
    if model._mu_r is not None:

        # Get volume-averaged values.
        vmodel = models.VolumeModel(grid, model, field)

        # Plus and minus indices.
        ixm = np.r_[0, np.arange(grid.vnC[0])]
        ixp = np.r_[np.arange(grid.vnC[0]), grid.vnC[0]-1]
        iym = np.r_[0, np.arange(grid.vnC[1])]
        iyp = np.r_[np.arange(grid.vnC[1]), grid.vnC[1]-1]
        izm = np.r_[0, np.arange(grid.vnC[2])]
        izp = np.r_[np.arange(grid.vnC[2]), grid.vnC[2]-1]

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

    # Create a Field instance and divide by s*mu_0 and return.
    return -Field(e3d_hx, e3d_hy, e3d_hz)/field.smu0


def _finite_source_xyz(grid, src, s, xyz, decimals):
    """Set finite dipole source using the adjoint interpolation method.

    See :func:`get_source_field` for further details.

    """
    # Round nodes and src coordinates (to avoid floating point issues etc).
    nodes_x = np.round(grid.nodes_x, decimals)
    nodes_y = np.round(grid.nodes_y, decimals)
    nodes_z = np.round(grid.nodes_z, decimals)
    src = np.round(src, decimals)

    # Ensure source is within nodes.
    outside = (src[0] < nodes_x[0] or src[1] > nodes_x[-1] or
               src[2] < nodes_y[0] or src[3] > nodes_y[-1] or
               src[4] < nodes_z[0] or src[5] > nodes_z[-1])
    if outside:
        raise ValueError(f"Provided source outside grid: {src}.")

    # Source lengths in x-, y-, and z-directions.
    d_xyz = src[1::2]-src[::2]

    # Inverse source lengths.
    id_xyz = d_xyz.copy()
    id_xyz[id_xyz != 0] = 1/id_xyz[id_xyz != 0]

    # Cell fractions.
    a1 = (nodes_x-src[0])*id_xyz[0]
    a2 = (nodes_y-src[2])*id_xyz[1]
    a3 = (nodes_z-src[4])*id_xyz[2]

    # Get range of indices of cells in which source resides.
    def min_max_ind(vector, i):
        """Return [min, max]-index of cells in which source resides."""
        vmin = min(src[2*i:2*i+2])
        vmax = max(src[2*i:2*i+2])
        return [max(0, np.where(vmin < np.r_[vector, np.infty])[0][0]-1),
                max(0, np.where(vmax < np.r_[vector, np.infty])[0][0]-1)]

    rix = min_max_ind(nodes_x, 0)
    riy = min_max_ind(nodes_y, 1)
    riz = min_max_ind(nodes_z, 2)

    # Loop over these indices.
    for iz in range(riz[0], min(riz[1]+1, a3.size-1)):
        for iy in range(riy[0], min(riy[1]+1, a2.size-1)):
            for ix in range(rix[0], min(rix[1]+1, a1.size-1)):

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

                # Contribution to edge (coordinate xyz)
                rx = (x_c[0]-nodes_x[ix])/grid.h[0][ix]
                ex = 1-rx
                ry = (x_c[1]-nodes_y[iy])/grid.h[1][iy]
                ey = 1-ry
                rz = (x_c[2]-nodes_z[iz])/grid.h[2][iz]
                ez = 1-rz

                # Add to field (only if segment inside cell).
                if min(rx, ry, rz) >= 0 and np.max(np.abs(ar-al)) > 0:

                    if xyz == 0:
                        s[ix, iy, iz] += ey*ez*x_len
                        s[ix, iy+1, iz] += ry*ez*x_len
                        s[ix, iy, iz+1] += ey*rz*x_len
                        s[ix, iy+1, iz+1] += ry*rz*x_len
                    if xyz == 1:
                        s[ix, iy, iz] += ex*ez*x_len
                        s[ix+1, iy, iz] += rx*ez*x_len
                        s[ix, iy, iz+1] += ex*rz*x_len
                        s[ix+1, iy, iz+1] += rx*rz*x_len
                    if xyz == 2:
                        s[ix, iy, iz] += ex*ey*x_len
                        s[ix+1, iy, iz] += rx*ey*x_len
                        s[ix, iy+1, iz] += ex*ry*x_len
                        s[ix+1, iy+1, iz] += rx*ry*x_len

    # Ensure unity (should not be necessary).
    sum_s = abs(s.sum())
    if abs(sum_s-1) > 1e-6:
        print(f"* WARNING :: Normalizing Source: {sum_s:.10f}.")
        s /= sum_s


def _rotation(azm, dip):
    """Rotation factors for RHS with positive z upwards.

    Easting is x, Northing is y, and positive upwards is z. All functions
    should use this rotation to ensure they use all the same definition.

    Parameters
    ----------
    azm : float
        Azimuth (°): horizontal deviation from x-axis, anti-clockwise.

    dip: float
        Dip (°): vertical deviation from xy-plane up-wards.


    Returns
    -------
    rot : ndarray (3,)
        Rotation factors (x, y, z).

    """
    return np.array([cosdg(azm)*cosdg(dip), sindg(azm)*cosdg(dip), sindg(dip)])


def _finite_dipole_from_point_dipole(src, length):
    """Return finite dipole of length given a point dipole."""
    factors = _rotation(*src[3:])*length/2
    return np.ravel(src[:3] + np.stack([-factors, factors]), 'F')


def _square_loop_from_point_dipole(src, length):
    """Return points of a square loop of length x length m perp. to dipole."""
    half_diagonal = np.sqrt(2)*length/2
    rot_hor = _rotation(src[3]+90, 0)*half_diagonal
    rot_ver = _rotation(src[3], src[4]+90)*half_diagonal
    points = src[:3]+np.stack([rot_hor, rot_ver, -rot_hor, -rot_ver, rot_hor])
    return points.T
