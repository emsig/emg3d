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

import warnings
from copy import deepcopy

import numpy as np
from scipy.constants import mu_0
from scipy.special import sindg, cosdg

from emg3d import maps, meshes, models, utils

__all__ = ['Field', 'SourceField', 'get_source_field', 'get_receiver',
           'get_h_field']


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

    Sort-order is 'F'.


    Parameters
    ----------

    grid : :class:`emg3d.meshes.TensorMesh` or ndarray
        Either a TensorMesh instance or an ndarray of shape grid.n_edges_x or
        grid.shape_edges_x. See explanations above. Only mandatory parameter;
        if the only one provided, it will initiate a zero-field of `dtype`.

    field : :class:`Field` or ndarray, optional
        Either a Field instance or an ndarray of shape grid.n_edges_y or
        grid.shape_edges_y. See explanations above.

    dtype : dtype, optional
        Only used if ``field=None``; the initiated zero-field for the provided
        TensorMesh has data type `dtype`. Default: complex.

    freq : float, optional
        Source frequency (Hz), used to compute the Laplace parameter `s`.
        Either positive or negative:

        - `freq` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - `freq` < 0: Laplace domain, hence
          :math:`s = f` (real).

        Just added as info if provided.

    """

    def __new__(cls, grid, field=None, dtype=np.complex128, freq=None):
        """Initiate a new Field instance."""

        if len(grid.shape_cells) != 3:
            raise ValueError("Provided grid must be a 3D grid.")

        # Collect field
        if field is None:
            nc = grid.n_edges_x + grid.n_edges_y + grid.n_edges_z
            field = np.zeros(nc, dtype=dtype)

        # Store the field as object
        obj = np.asarray(field).view(cls)

        # Store grid
        obj.grid = grid

        # Store frequency
        if freq is None and hasattr(field, 'freq'):
            freq = field._freq
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

        self.grid = getattr(obj, 'grid', None)
        self._freq = getattr(obj, '_freq', None)

    def __reduce__(self):
        """Customize __reduce__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Get the parent's __reduce__ tuple.
        pickled_state = super().__reduce__()

        # Create our own tuple to pass to __setstate__.
        new_state = pickled_state[2]
        new_state += (self.grid.h[0], self.grid.h[1], self.grid.h[2],
                      self.grid.origin, self._freq)

        # Return tuple that replaces parent's __setstate__ tuple with our own.
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Customize __setstate__ to make `Field` work with pickle.
        => https://stackoverflow.com/a/26599346
        """
        # Set the necessary attributes (in reverse order).
        self._freq = state[-1]
        origin = state[-2]
        hz = state[-3]
        hy = state[-4]
        hx = state[-5]
        self.grid = meshes.TensorMesh([hx, hy, hz], origin)

        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-5])

    def copy(self):
        """Return a copy of the Field."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Field in a dict."""
        out = {'field': np.array(self.field), 'freq': self._freq,
               '__class__': self.__class__.__name__}
        out['grid'] = {'hx': self.grid.h[0], 'hy': self.grid.h[1],
                       'hz': self.grid.h[2], 'origin': self.grid.origin}
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
            The dictionary needs the keys `field`, `freq`, `shape_edges_x`,
            `shape_edges_y`, and `shape_edges_z`.

        Returns
        -------
        obj : :class:`Field` instance

        """

        # Check and get the required keys from the input.
        try:
            grid = meshes.TensorMesh.from_dict(inp.pop('grid'))
            return cls(grid=grid, field=inp['field'], freq=inp['freq'])
        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.") from e

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
        return self.view()[:self.grid.n_edges_x].reshape(
                self.grid.shape_edges_x, order='F')

    @fx.setter
    def fx(self, fx):
        """Update field in x-direction."""
        self.view()[:self.grid.n_edges_x] = fx.ravel('F')

    @property
    def fy(self):
        """View of the field in the y-direction (nNx, nCy, nNz)."""
        return self.view()[self.grid.n_edges_x:-self.grid.n_edges_z].reshape(
                self.grid.shape_edges_y, order='F')

    @fy.setter
    def fy(self, fy):
        """Update field in y-direction."""
        self.view()[self.grid.n_edges_x:-self.grid.n_edges_z] = fy.ravel('F')

    @property
    def fz(self):
        """View of the field in the z-direction (nNx, nNy, nCz)."""
        return self.view()[-self.grid.n_edges_z:].reshape(
                self.grid.shape_edges_z, order='F')

    @fz.setter
    def fz(self, fz):
        """Update electric field in z-direction."""
        self.view()[-self.grid.n_edges_z:] = fz.ravel('F')

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

    # INTERPOLATION
    def interpolate_to_grid(self, grid, **interpolate_opts):
        """Interpolate the field to a new grid.


        Parameters
        ----------
        grid : TensorMesh
            Grid of the new model; a :class:`emg3d.meshes.TensorMesh` instance.

        interpolate_opts : dict
            Passed through to :func:`emg3d.maps.interpolate`. Defaults are
            ``method='cubic'``, ``log=True``, and ``extrapolate=False``.


        Returns
        -------
        obj : Field
            A new :class:`emg3d.fields.Field` instance on ``grid``.

        """

        # Get solver options, set to defaults if not provided.
        g2g_inp = {
            'method': 'cubic',
            'extrapolate': False,
            'log': True,
            **({} if interpolate_opts is None else interpolate_opts),
            'grid': self.grid,
            'new_grid': grid,
        }

        # Interpolate f{x;y;z} add to dict.
        field = np.r_[maps.interpolate(values=self.fx, **g2g_inp).ravel('F'),
                      maps.interpolate(values=self.fy, **g2g_inp).ravel('F'),
                      maps.interpolate(values=self.fz, **g2g_inp).ravel('F')]

        # Assemble new field.
        return Field(grid, field, freq=self._freq)


class SourceField(Field):
    r"""Create a Source-Field instance with x-, y-, and z-views of the field.

    A subclass of :class:`Field`. Additional properties are the real-valued
    source vector (`vector`, `vx`, `vy`, `vz`), which sum is always one. For a
    `SourceField` frequency is a mandatory  parameter, unlike for a `Field`
    (recommended also for `Field` though),

    Parameters
    ----------

    grid : :class:`emg3d.meshes.TensorMesh` or ndarray
        Either a TensorMesh instance or an ndarray of shape grid.n_edges_x or
        grid.shape_edges_x. See explanations above. Only mandatory parameter;
        if the only one provided, it will initiate a zero-field of `dtype`.

    field : :class:`Field` or ndarray, optional
        Either a Field instance or an ndarray of shape grid.n_edges_y or
        grid.shape_edges_y. See explanations above.

    dtype : dtype, optional
        Only used if ``field=None``; the initiated zero-field for the provided
        TensorMesh has data type `dtype`. Default: complex.

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

    def __new__(cls, grid, field=None, dtype=np.complex128, freq=None):
        """Initiate a new Source Field."""
        # Ensure frequency is provided.
        if freq is None:
            raise ValueError("SourceField requires the frequency.")

        if freq > 0:
            dtype = complex
        else:
            dtype = float

        return super().__new__(cls, grid, field=field, dtype=dtype, freq=freq)

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


def get_source_field(grid, src, freq, strength=0, electric=True, length=1.0,
                     decimals=6, **kwargs):
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
        of ``length``. In the case of a point dipole one can set
        ``electric=False``, which will create a square loop of
        ``length``x``length`` perpendicular to the dipole.

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

    electric : bool, optional
        Shortcut to create a magnetic source. If False, the format of ``src``
        must be that of a point dipole: ``[x, y, z, azimuth, dip]`` (for the
        other formats setting ``electric`` has no effect). It then creates a
        square loop perpendicular to this dipole, with side-length 1.
        Default is True, meaning an electric source.

    length : float, optional
        Length (m) of the point dipole when converted to a finite length
        dipole, or edge-length (m) of the square loop if ``electric=False``.
        Default is 1.0.

    decimals: int, optional
        Grid nodes and source coordinates are rounded to given number of
        decimals. Default is 6 (micrometer).


    Returns
    -------
    sfield : :func:`SourceField` instance
        Source field, normalized to 1 A m.

    """

    # Ensure no kwargs left.
    if kwargs:
        raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

    # Cast some parameters.
    if not np.allclose(np.size(src[0]), [np.size(c) for c in src]):
        raise ValueError(
                "All source coordinates must have the same dimension."
                f"Provided source: {src}.")

    src = np.asarray(src, dtype=np.float64)
    strength = np.asarray(strength)

    # Convert point dipole sources to finite dipoles or loops (electric).
    if src.shape == (5, ):  # Point dipole

        if not electric:  # Magnetic: convert to square loop perp. to dipole.
            src = _square_loop_from_point_dipole(src, length)
            # src.shape = (3, 5)

        else:  # Electric: convert to finite length.
            src = _finite_dipole_from_point_dipole(src, length)
            # src.shape = (6, )

    # Get arbitrary shaped sources recursively.
    if src.shape[0] == 3 and src.ndim > 1:

        # Get arbitrarily shaped dipole source using recursion.
        sx, sy, sz = src

        # Get normalized segment lengths.
        lengths = np.sqrt(np.sum((src[:, :-1] - src[:, 1:])**2, axis=0))
        if strength == 0:
            lengths /= lengths.sum()
        else:  # (Not in-place multiplication, as strength can be complex.)
            lengths = lengths*strength

        # Initiate a zero-valued source field and loop over segments.
        sfield = SourceField(grid, freq=freq)
        sfield.src = src
        sfield.strength = strength
        sfield.moment = np.array([0., 0, 0], dtype=lengths.dtype)

        # Loop over elements.
        for i in range(sx.size-1):
            segment = (sx[i], sx[i+1], sy[i], sy[i+1], sz[i], sz[i+1])
            seg_field = get_source_field(grid, segment, freq, lengths[i])
            sfield += seg_field
            sfield.moment += seg_field.moment

        # Check this with iw/-iw; source definition etc.
        if not electric:
            sfield *= -1

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


def get_receiver(field, rec):
    """Return the field (response) at receiver coordinates.

    - TODO :: check, simplify, document
    - TODO :: incorporate into Field
    - TODO :: Improve tests!

    Note that in order to avoid boundary effects the first and last value in
    each direction is neglected. Field values for coordinates outside of the
    grid are set to NaN's. However, all receivers should be much further away
    from the boundary.


    Parameters
    ----------
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
    grid = field.grid
    points = ((grid.cell_centers_x, grid.nodes_y, grid.nodes_z),
              (grid.nodes_x, grid.cell_centers_y, grid.nodes_z),
              (grid.nodes_x, grid.nodes_y, grid.cell_centers_z))

    # Remove first and last value in each direction.
    points = tuple([tuple([p[1:-1] for p in pp]) for pp in points])

    # Pre-allocate the response.
    _, xi, shape = maps._points_from_grids(
            field.grid, field.fx, rec[:3], 'cubic')
    resp = np.zeros(xi.shape[0], dtype=field.dtype)

    # Add the required responses.
    factors = _rotation(*rec[3:])  # Geometrical weights from angles.
    for i, ff in enumerate((field.fx, field.fy, field.fz)):
        if np.any(abs(factors[i]) > 1e-10):
            resp += factors[i]*maps.interp_spline_3d(
                        points[i], ff[1:-1, 1:-1, 1:-1], xi,
                        mode='constant', cval=np.nan)

    # Return response.
    return utils.EMArray(resp.reshape(shape, order='F'))


def get_h_field(model, field):
    r"""Return magnetic field corresponding to provided electric field.

    - TODO REWORK THIS!

    - TODO Decide if to remove outermost layer or not

    - TODO :: incorporate into Field

    Retrieve the magnetic field :math:`\mathbf{H}` from the electric field
    :math:`\mathbf{E}` using Farady's law, given by

    .. math::

        \nabla \times \mathbf{E} = \rm{i}\omega\mu\mathbf{H} .

    Note that the magnetic field in x-direction is defined in the center of the
    face defined by the electric field in y- and z-directions, and similar for
    the other field directions. This means that the provided electric field and
    the returned magnetic field have different dimensions::

       E-field:  x: [grid.cell_centers_x, grid.nodes_y, grid.nodes_z]
                 y: [grid.nodes_x, grid.cell_centers_y, grid.nodes_z]
                 z: [grid.nodes_x, grid.nodes_y, grid.cell_centers_z]

       H-field:  x: [grid.nodes_x, grid.cell_centers_y, grid.cell_centers_z]
                 y: [grid.cell_centers_x, grid.nodes_y, grid.cell_centers_z]
                 z: [grid.cell_centers_x, grid.cell_centers_y, grid.nodes_z]


    Parameters
    ----------
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
    e3d_hx = (np.diff(field.fz, axis=1)/field.grid.h[1][None, :, None] -
              np.diff(field.fy, axis=2)/field.grid.h[2][None, None, :])

    # H_y = (E_x^2 - E_z^0)
    e3d_hy = (np.diff(field.fx, axis=2)/field.grid.h[2][None, None, :] -
              np.diff(field.fz, axis=0)/field.grid.h[0][:, None, None])

    # H_z = (E_y^0 - E_x^1)
    e3d_hz = (np.diff(field.fy, axis=0)/field.grid.h[0][:, None, None] -
              np.diff(field.fx, axis=1)/field.grid.h[1][None, :, None])

    # If relative magnetic permeability is not one, we have to take the volume
    # into account, as mu_r is volume-averaged.
    if model.mu_r is not None:

        # Get volume-averaged values.
        vmodel = models.VolumeModel(model, field)

        # Plus and minus indices.
        ixm = np.r_[0, np.arange(model.shape[0])]
        ixp = np.r_[np.arange(model.shape[0]), model.shape[0]-1]
        iym = np.r_[0, np.arange(model.shape[1])]
        iyp = np.r_[np.arange(model.shape[1]), model.shape[1]-1]
        izm = np.r_[0, np.arange(model.shape[2])]
        izp = np.r_[np.arange(model.shape[2]), model.shape[2]-1]

        # Average mu_r for dual-grid.
        zeta_x = (vmodel.zeta[ixm, :, :] + vmodel.zeta[ixp, :, :])/2.
        zeta_y = (vmodel.zeta[:, iym, :] + vmodel.zeta[:, iyp, :])/2.
        zeta_z = (vmodel.zeta[:, :, izm] + vmodel.zeta[:, :, izp])/2.

        hvx = field.grid.h[0][:, None, None]
        hvy = field.grid.h[1][None, :, None]
        hvz = field.grid.h[2][None, None, :]

        # Define the widths of the dual grid.
        dx = (np.r_[0., field.grid.h[0]] + np.r_[field.grid.h[0], 0.])/2.
        dy = (np.r_[0., field.grid.h[1]] + np.r_[field.grid.h[1], 0.])/2.
        dz = (np.r_[0., field.grid.h[2]] + np.r_[field.grid.h[2], 0.])/2.

        # Multiply fields by mu_r.
        e3d_hx *= zeta_x/(dx[:, None, None]*hvy*hvz)
        e3d_hy *= zeta_y/(hvx*dy[None, :, None]*hvz)
        e3d_hz *= zeta_z/(hvx*hvy*dz[None, None, :])

    # TODO change to Magnetic Field

    # Create magnetic grid.
    hx = np.diff((field.grid.nodes_x[:-1] + field.grid.nodes_x[1:])/2)
    hx = np.r_[field.grid.h[0][0], hx, field.grid.h[0][-1]]
    hy = np.diff((field.grid.nodes_y[:-1] + field.grid.nodes_y[1:])/2)
    hy = np.r_[field.grid.h[1][0], hy, field.grid.h[1][-1]]
    hz = np.diff((field.grid.nodes_z[:-1] + field.grid.nodes_z[1:])/2)
    hz = np.r_[field.grid.h[2][0], hz, field.grid.h[2][-1]]
    origin = (field.grid.origin[0] - field.grid.h[0][0]/2,
              field.grid.origin[1] - field.grid.h[1][0]/2,
              field.grid.origin[2] - field.grid.h[2][0]/2)
    grid = meshes.TensorMesh([hx, hy, hz], origin)

    n_e3d_hx = np.zeros(grid.shape_edges_x, dtype=e3d_hx.dtype)
    n_e3d_hx[:, 1:-1, 1:-1] = e3d_hx
    e3d_hx = n_e3d_hx

    n_e3d_hy = np.zeros(grid.shape_edges_y, dtype=e3d_hy.dtype)
    n_e3d_hy[1:-1, :, 1:-1] = e3d_hy
    e3d_hy = n_e3d_hy

    n_e3d_hz = np.zeros(grid.shape_edges_z, dtype=e3d_hz.dtype)
    n_e3d_hz[1:-1, 1:-1, :] = e3d_hz
    e3d_hz = n_e3d_hz

    # Create a Field instance and divide by s*mu_0 and return.
    new = np.r_[e3d_hx.ravel('F'), e3d_hy.ravel('F'), e3d_hz.ravel('F')]
    return -Field(grid, new, freq=field._freq)/field.smu0


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
        # Print is always shown and simpler, warn for the CLI logs.
        msg = f"Normalizing Source: {sum_s:.10f}."
        print(f"* WARNING :: {msg}")
        warnings.warn(msg, UserWarning)
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
