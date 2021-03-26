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

from emg3d import maps, meshes, models, utils, electrodes

__all__ = ['Field', 'get_source_field', 'get_receiver', 'get_magnetic_field']


class Field:
    r"""A Field contains the x-, y-, and z- directed electromagnetic fields.

    A Field is a simple container that has a 1D array ``Field.field``
    containing the x-, y-, and z-directed fields one after the other.
    The field can be any field, such as an electric field, a magnetic field,
    or a source field (which is an electric field).

    The particular fields can be accessed via the ``Field.f{x;y;z}``
    attributes, which are 3D arrays corresponding to the shape of the edges
    in this direction; sort-order is Fortran-like ('F').


    Parameters
    ----------

    grid : TensorMesh
        The grid; a :class:`emg3d.meshes.TensorMesh` instance.

    data : ndarray, default: None
        The actual data, a ``ndarray`` of size ``grid.n_edges``. If ``None``,
        it is initiated with zeros.

    frequency : float, default: None
        Field frequency (Hz), used to compute the Laplace parameter ``s``.
        Either positive or negative:

        - ``frequency > 0``: Frequency domain, hence
          :math:`s = \mathrm{i}\omega = 2\mathrm{i}\pi f` (complex);
        - ``frequency < 0``: Laplace domain, hence
          :math:`s = f` (real).

    dtype : dtype, default: complex
        Data type of the initiated field; only used if both ``frequency`` and
        ``data`` are None.

    """

    def __init__(self, grid, data=None, frequency=None, dtype=None):
        """Initiate a new Field instance."""

        # Get dtype.
        if frequency is not None:  # Frequency is top priority.
            if frequency > 0:
                dtype = np.complex128
            elif frequency < 0:
                dtype = np.float64
            else:
                raise ValueError(
                    "`frequency` must be f>0 (frequency domain) or f<0 "
                    "(Laplace domain). Provided: {frequency} Hz."
                )
        elif data is not None:  # Data is second priority.
            dtype = data.dtype

        elif dtype is None:  # Default.
            dtype = np.complex128

        # Store field.
        if data is None:
            self._field = np.zeros(grid.n_edges, dtype=dtype)
        else:
            self._field = np.asarray(data, dtype=dtype)

        # Store grid and frequency.
        self.grid = grid
        self._frequency = frequency

    def __repr__(self):
        """Simple representation."""
        return (f"{self.__class__.__name__}: {self.grid.shape_cells[0]} x "
                f"{self.grid.shape_cells[1]} x {self.grid.shape_cells[2]}; "
                f"{self.field.size:,}")

    def __eq__(self, field):
        """Compare two fields."""
        equal = self.__class__.__name__ == field.__class__.__name__
        equal *= self.grid == field.grid
        equal *= self._frequency == field._frequency
        equal *= np.allclose(self._field, field._field, atol=0, rtol=1e-10)
        return bool(equal)

    def copy(self):
        """Return a copy of the Field."""
        return self.from_dict(self.to_dict(copy=True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Field in a dict.

        Parameters
        ----------
        copy : bool, default: False
            If True, returns a deep copy of the dict.


        Returns
        -------
        out : dict
            Dictionary containing all information to re-create the Field.

        """
        out = {
            '__class__': self.__class__.__name__,  # v ensure emg3d-TensorMesh
            'grid': meshes.TensorMesh(self.grid.h, self.grid.origin).to_dict(),
            'data': self._field,
            'frequency': self._frequency,
        }
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`emg3d.fields.Field` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`emg3d.fields.Field.to_dict`. The
            dictionary needs the keys ``field``, ``frequency``, and ``grid``;
            ``grid`` itself is also a dict which needs the keys ``hx``, ``hy``,
            ``hz``, and ``origin``.

        Returns
        -------
        field : Field
            A :class:`emg3d.fields.Field` instance.

        """
        inp.pop('__class__', None)
        MeshClass = getattr(meshes, inp['grid']['__class__'])
        return cls(grid=MeshClass.from_dict(inp.pop('grid')), **inp)

    @property
    def field(self):
        """Entire field as 1D array [fx, fy, fz]."""
        return self._field

    @field.setter
    def field(self, field):
        """Update field as 1D array [fx, fy, fz]."""
        self._field[:] = field

    @property
    def fx(self):
        """Field in x direction; shape: (cell_centers_x, nodes_y, nodes_z)."""
        ix = self.grid.n_edges_x
        shape = self.grid.shape_edges_x
        return utils.EMArray(self._field[:ix]).reshape(shape, order='F')

    @fx.setter
    def fx(self, fx):
        """Update field in x-direction."""
        self._field[:self.grid.n_edges_x] = fx.ravel('F')

    @property
    def fy(self):
        """Field in y direction; shape: (nodes_x, cell_centers_y, nodes_z)."""
        i0, i1 = self.grid.n_edges_x, self.grid.n_edges_z
        shape = self.grid.shape_edges_y
        return utils.EMArray(self._field[i0:-i1]).reshape(shape, order='F')

    @fy.setter
    def fy(self, fy):
        """Update field in y-direction."""
        self._field[self.grid.n_edges_x:-self.grid.n_edges_z] = fy.ravel('F')

    @property
    def fz(self):
        """Field in z direction; shape: (nodes_x, nodes_y, cell_centers_z)."""
        i0, shape = self.grid.n_edges_z, self.grid.shape_edges_z
        return utils.EMArray(self._field[-i0:].reshape(shape, order='F'))

    @fz.setter
    def fz(self, fz):
        """Update electric field in z-direction."""
        self._field[-self.grid.n_edges_z:] = fz.ravel('F')

    @property
    def frequency(self):
        """Return frequency (Hz)."""
        if self._frequency is None:
            return None
        else:
            return abs(self._frequency)

    @property
    def smu0(self):
        """Return s*mu_0; mu_0 = magn permeability of free space [H/m]."""
        if getattr(self, '_smu0', None) is None:
            if self.sval is not None:
                self._smu0 = self.sval*mu_0
            else:
                self._smu0 = None

        return self._smu0

    @property
    def sval(self):
        """Return s=iw in frequency domain and s=f in Laplace domain."""

        if getattr(self, '_sval', None) is None:
            if self._frequency is not None:
                if self._frequency < 0:  # Laplace domain; s.
                    self._sval = np.array(self._frequency)
                else:  # Frequency domain; s = iw = 2i*pi*f.
                    self._sval = np.array(-2j*np.pi*self._frequency)
            else:
                self._sval = None

        return self._sval

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
        field : Field
            A new :class:`emg3d.fields.Field` instance on ``grid``.

        """

        # Get solver options, set to defaults if not provided.
        g2g_inp = {
            'method': 'cubic',
            'extrapolate': False,
            'log': True,
            **({} if interpolate_opts is None else interpolate_opts),
            'grid': self.grid,
            'xi': grid,
        }

        # Interpolate f{x;y;z}.
        field = np.r_[maps.interpolate(values=self.fx, **g2g_inp).ravel('F'),
                      maps.interpolate(values=self.fy, **g2g_inp).ravel('F'),
                      maps.interpolate(values=self.fz, **g2g_inp).ravel('F')]

        # Assemble and return new field.
        return Field(grid, field, frequency=self._frequency)

    def get_receiver(self, receiver):
        """Return the field at receiver locations.

        Parameters
        ----------
        receiver : tuple
            Receiver coordinates (m) and angles (°) in the format
            ``(x, y, z, azimuth, dip)``.

            All values can either be a scalar or having the same length as
            number of receivers.

            Angles:

            - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
            - dip (°): vertical deviation from xy-plane up-wards.


        Returns
        -------
        responses : EMArray
            Responses at receiver locations.

        """
        return get_receiver(self, receiver)


def get_source_field(grid, source, frequency, strength=0, electric=True,
                     length=1.0, decimals=6, **kwargs):
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

    source : list of floats
        Source coordinates (m). There are three formats:

          - Finite length dipole: ``[x0, x1, y0, y1, z0, z1]``.
          - Point dipole: ``[x, y, z, azimuth, dip]``.
          - Arbitrarily shaped source: ``[[x-coo], [y-coo], [z-coo]]``.

        Point dipoles will be converted internally to finite length dipoles
        of ``length``. In the case of a point dipole one can set
        ``electric=False``, which will create a square loop of
        ``length``x``length`` perpendicular to the dipole.

    frequency : float
        Source frequency (Hz), used to compute the Laplace parameter `s`.
        Either positive or negative:

        - `frequency` > 0: Frequency domain, hence
          :math:`s = -\mathrm{i}\omega = -2\mathrm{i}\pi f` (complex);
        - `frequency` < 0: Laplace domain, hence
          :math:`s = f` (real).

    strength : float or complex, optional
        Source strength (A):

          - If 0, output is normalized to a source of 1 m length, and source
            strength of 1 A.
          - If != 0, output is returned for given source length and strength.

        Default is 0.

    electric : bool, optional
        Shortcut to create a magnetic source. If False, the format of
        ``source`` must be that of a point dipole: ``[x, y, z, azimuth, dip]``
        (for the other formats setting ``electric`` has no effect). It then
        creates a square loop perpendicular to this dipole, with side-length 1.
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
    sfield : :func:`Field` instance
        Source field, normalized to 1 A m.

    """

    # Ensure no kwargs left.
    if kwargs:
        raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

    # Cast some parameters.
    if not np.allclose(np.size(source[0]), [np.size(c) for c in source]):
        raise ValueError(
            "All source coordinates must have the same dimension."
            f"Provided source: {source}."
        )

    source = np.asarray(source, dtype=np.float64)
    strength = np.asarray(strength)

    # Convert point dipole sources to finite dipoles or loops (electric).
    if source.shape == (5, ):  # Point dipole

        if not electric:  # Magnetic: convert to square loop perp. to dipole.
            source = electrodes._point_to_square_loop(source, length)
            # source.shape = (3, 5)

        else:  # Electric: convert to finite length.
            source = electrodes._point_to_dipole(
                    source, length).ravel('F')
            # source.shape = (6, )

    # Get arbitrary shaped sources recursively.
    if source.shape[0] == 3 and source.ndim > 1:

        # Get arbitrarily shaped dipole source using recursion.
        sx, sy, sz = source

        # Get normalized segment lengths.
        lengths = np.sqrt(np.sum((source[:, :-1] - source[:, 1:])**2, axis=0))
        if strength == 0:
            lengths /= lengths.sum()
        else:  # (Not in-place multiplication, as strength can be complex.)
            lengths = lengths*strength

        # Initiate a zero-valued source field and loop over segments.
        sfield = Field(grid, frequency=frequency)
        sfield.source = source
        sfield.strength = strength
        sfield.moment = np.array([0., 0, 0], dtype=lengths.dtype)

        # Loop over elements.
        for i in range(sx.size-1):
            segment = (sx[i], sx[i+1], sy[i], sy[i+1], sz[i], sz[i+1])
            seg_field = get_source_field(grid, segment, frequency, lengths[i])
            sfield.field += seg_field.field
            sfield.moment += seg_field.moment

        # Check this with iw/-iw; source definition etc.
        if not electric:
            sfield.field *= -1

        return sfield

    # From here onwards `source` has to be a finite length dipole  of format
    # [x1, x2, y1, y2, z1, z2]. Ensure that:
    if source.shape != (6, ):
        raise ValueError(
            "Source is wrong defined. It must be either (1) a point, "
            "[x, y, z, azimuth, dip], (2) a finite dipole, "
            "[x1, x2, y1, y2, z1, z2], or (3) an arbitrarily shaped "
            f"dipole, [[x-coo], [y-coo], [z-coo]]. Provided: {source}."
        )

    # Get length in each direction.
    length = source[1::2]-source[::2]

    # Ensure finite length dipole is not a point dipole.
    if np.allclose(length, 0, atol=1e-15):
        raise ValueError(
            "Provided finite dipole has no length; use "
            "the format [x, y, z, azimuth, dip] instead."
        )

    # Get source moment (individually for x, y, z).
    if strength == 0:  # 1 A m
        length /= np.linalg.norm(length)
        moment = length
    else:              # Multiply source length with source strength
        moment = strength*length

    # Initiate zero source field.
    sfield = Field(grid, frequency=frequency)

    # Return source-field for each direction.
    for xyz, field in enumerate([sfield.fx, sfield.fy, sfield.fz]):

        # Get source field for this direction.
        _finite_source_xyz(grid, source, field, decimals)

        # Multiply by moment*s*mu
        field *= moment[xyz]*sfield.smu0

    # Add source and moment information.
    sfield.source = source
    sfield.strength = strength
    sfield.moment = moment

    return sfield


def get_receiver(field, receiver):
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

    receiver : tuple (x, y, z, azimuth, dip)
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
    if len(receiver) != 5:
        raise ValueError(
            "`receiver` needs to be in the form (x, y, z, azimuth, dip). "
            f"Length of provided `receiver`: {len(receiver)}."
        )

    # Check field dimension to ensure it is not a particular field.
    if not hasattr(field, 'field'):
        raise ValueError(
            "`field` must be a `Field`-instance, not a "
            "particular field such as `field.fx`."
        )

    # Get the vectors corresponding to input data.
    grid = field.grid
    points = ((grid.cell_centers_x, grid.nodes_y, grid.nodes_z),
              (grid.nodes_x, grid.cell_centers_y, grid.nodes_z),
              (grid.nodes_x, grid.nodes_y, grid.cell_centers_z))

    # Remove first and last value in each direction.
    points = tuple([tuple([p[1:-1] for p in pp]) for pp in points])

    # Pre-allocate the response.
    _, xi, shape = maps._points_from_grids(
            field.grid, field.fx, receiver[:3], 'cubic')
    resp = np.zeros(xi.shape[0], dtype=field.field.dtype)

    # Add the required responses.
    factors = electrodes._rotation(*receiver[3:])
    for i, ff in enumerate((field.fx, field.fy, field.fz)):
        if np.any(abs(factors[i]) > 1e-10):
            resp += factors[i]*maps.interp_spline_3d(
                        points[i], ff[1:-1, 1:-1, 1:-1], xi,
                        mode='constant', cval=np.nan)

    # Return response.
    return utils.EMArray(resp.reshape(shape, order='F'))


def get_magnetic_field(model, field):
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
    return Field(grid, -new/field.smu0, frequency=field._frequency)


def _finite_source_xyz(grid, source, field, decimals):
    """Set finite dipole source using the adjoint interpolation method.

    The result is placed directly in the provided ``field``-component.


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a :class:`emg3d.meshes.TensorMesh` instance.

    source : ndarray
        Source coordinates in the form of (x0, x1, y0, y1, z0, z1) (m).

    field : ndarray
        A particular component of the source field, one of ``field.f{x;y;z}``.

    decimals: int, optional
        Grid nodes and source coordinates are rounded to given number of
        decimals. Default is 6 (micrometer).

    """
    # Round nodes and source coordinates (to avoid floating point issues etc).
    nodes_x = np.round(grid.nodes_x, decimals)
    nodes_y = np.round(grid.nodes_y, decimals)
    nodes_z = np.round(grid.nodes_z, decimals)
    source = np.round(source, decimals)

    # Ensure source is within nodes.
    outside = (source[0] < nodes_x[0] or source[1] > nodes_x[-1] or
               source[2] < nodes_y[0] or source[3] > nodes_y[-1] or
               source[4] < nodes_z[0] or source[5] > nodes_z[-1])
    if outside:
        raise ValueError(f"Provided source outside grid: {source}.")

    # Source lengths in x-, y-, and z-directions.
    d_xyz = source[1::2]-source[::2]

    # Inverse source lengths.
    id_xyz = d_xyz.copy()
    id_xyz[id_xyz != 0] = 1/id_xyz[id_xyz != 0]

    # Cell fractions.
    a1 = (nodes_x-source[0])*id_xyz[0]
    a2 = (nodes_y-source[2])*id_xyz[1]
    a3 = (nodes_z-source[4])*id_xyz[2]

    # Get range of indices of cells in which source resides.
    def min_max_ind(vector, i):
        """Return [min, max]-index of cells in which source resides."""
        vmin = min(source[2*i:2*i+2])
        vmax = max(source[2*i:2*i+2])
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
                xmin = source[::2]+al*d_xyz
                xmax = source[::2]+ar*d_xyz
                x_c = (xmin+xmax)/2.0
                slen = np.linalg.norm(source[1::2]-source[::2])
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

                    if field.shape == grid.shape_edges_x:
                        field[ix, iy, iz] += ey*ez*x_len
                        field[ix, iy+1, iz] += ry*ez*x_len
                        field[ix, iy, iz+1] += ey*rz*x_len
                        field[ix, iy+1, iz+1] += ry*rz*x_len
                    elif field.shape == grid.shape_edges_y:
                        field[ix, iy, iz] += ex*ez*x_len
                        field[ix+1, iy, iz] += rx*ez*x_len
                        field[ix, iy, iz+1] += ex*rz*x_len
                        field[ix+1, iy, iz+1] += rx*rz*x_len
                    else:
                        field[ix, iy, iz] += ex*ey*x_len
                        field[ix+1, iy, iz] += rx*ey*x_len
                        field[ix, iy+1, iz] += ex*ry*x_len
                        field[ix+1, iy+1, iz] += rx*ry*x_len

    # Ensure unity (should not be necessary).
    sum_s = abs(field.sum())
    if abs(sum_s-1) > 1e-6:
        # Print is always shown and simpler, warn for the CLI logs.
        msg = f"Normalizing Source: {sum_s:.10f}."
        print(f"* WARNING :: {msg}")
        warnings.warn(msg, UserWarning)
        field /= sum_s
