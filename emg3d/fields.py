"""
Everything that is related to fields: storing the electric field, obtaining the
magnetic field, generating the source field; obtaining the fields at receiver
locations.
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

import warnings
from copy import deepcopy

import numba as nb
import numpy as np
from scipy.constants import mu_0

from emg3d.core import _numba_setting
from emg3d import maps, meshes, models, utils, electrodes

__all__ = ['Field', 'get_source_field', 'get_receiver', 'get_magnetic_field']


@utils._known_class
class Field:
    r"""A Field contains the x-, y-, and z- directed electromagnetic fields.

    A Field is a simple container that has a 1D array ``Field.field``
    containing the x-, y-, and z-directed fields one after the other.
    The field can be any field, such as an electric field, a magnetic field,
    or a source field (which is an electric field).

    The particular fields can be accessed via the ``Field.f{x;y;z}``
    attributes, which are 3D arrays corresponding to the shape of the edges
    (electric fields) or the faces (magnetic fields) in this direction;
    sort-order is Fortran-like ('F').


    Parameters
    ----------

    grid : TensorMesh
        The grid; a :class:`emg3d.meshes.TensorMesh` instance.

    data : ndarray, default: None
        The actual data, a ``ndarray`` of size ``grid.n_edges`` (electric
        fields) or ``grid.n_faces`` (magnetic fields). If ``None``, it is
        initiated with zeros.

    frequency : float, default: None
        Field frequency (Hz), used to compute the Laplace parameter ``s``.
        Either positive or negative:

        - ``frequency > 0``: Frequency domain, hence
          :math:`s = \mathrm{i}\omega = 2\mathrm{i}\pi f` (complex);
        - ``frequency < 0``: Laplace domain, hence
          :math:`s = f` (real).

        This is primarily important for source fields. However, frequency
        information is also required to obtain the magnetic field from an
        electric field.

    dtype : dtype, default: complex
        Data type of the initiated field; only used if both ``frequency`` and
        ``data`` are None.

    electric : bool, default: True
        If electric, the properties live on the edges of the grid, if magnetic,
        they live on the faces.

    """

    def __init__(self, grid, data=None, frequency=None, dtype=None,
                 electric=True):
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

        # Store grid, frequency, and electric.
        self.grid = grid
        self._frequency = frequency
        self.electric = electric

        # Store field.
        if data is None:
            field = np.zeros(self._get_prop('n'), dtype=dtype, order='F')
        else:
            field = np.asarray(data, dtype=dtype)
        self._field = utils.EMArray(field)

    def __repr__(self):
        """Simple representation."""
        return (f"{self.__class__.__name__}: "
                f"{['magnetic', 'electric'][self.electric]}; "
                f"{self.grid.shape_cells[0]} x {self.grid.shape_cells[1]} x "
                f"{self.grid.shape_cells[2]}; {self.field.size:,}")

    def __eq__(self, field):
        """Compare two fields."""
        equal = self.__class__.__name__ == field.__class__.__name__
        equal *= self.grid == field.grid
        equal *= self._frequency == field._frequency
        equal *= self.electric == field.electric
        if equal:
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
            'electric': self.electric,
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
            dictionary needs the keys ``field``, ``frequency``, ``grid``, and
            ``electric``; ``grid`` itself is also a dict which needs the keys
            ``hx``, ``hy``, ``hz``, and ``origin``.

        Returns
        -------
        field : Field
            A :class:`emg3d.fields.Field` instance.

        """
        inp = {k: v for k, v in inp.items() if k != '__class__'}
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
        """Field in x direction.

        Shape:

        - electric: (grid.cell_centers_x, grid.nodes_y, grid.nodes_z)
        - magnetic: (grid.nodes_x, grid.cell_centers_y, grid.cell_centers_z)

        """
        i1 = self._get_prop('n', 'x')
        shape = self._get_prop('shape', 'x')
        return self._field[:i1].reshape(shape, order='F')

    @fx.setter
    def fx(self, fx):
        """Update field in x-direction."""
        i1 = self._get_prop('n', 'x')
        self._field[:i1] = fx.ravel('F')

    @property
    def fy(self):
        """Field in y direction.

        Shape:

        - electric: (grid.nodes_x, grid.cell_centers_y, grid.nodes_z)
        - magnetic: (grid.cell_centers_x, grid.nodes_y, grid.cell_centers_z)

        """
        i0, i1 = self._get_prop('n', 'x'), self._get_prop('n', 'z')
        shape = self._get_prop('shape', 'y')
        return self._field[i0:-i1].reshape(shape, order='F')

    @fy.setter
    def fy(self, fy):
        """Update field in y-direction."""
        i0, i1 = self._get_prop('n', 'x'), self._get_prop('n', 'z')
        self._field[i0:-i1] = fy.ravel('F')

    @property
    def fz(self):
        """Field in z direction.

        Shape:

        - electric: (grid.nodes_x, grid.nodes_y, grid.cell_centers_z)
        - magnetic: (grid.cell_centers_x, grid.cell_centers_y, grid.nodes_z)

        """
        i0 = self._get_prop('n', 'z')
        shape = self._get_prop('shape', 'z')
        return self._field[-i0:].reshape(shape, order='F')

    @fz.setter
    def fz(self, fz):
        """Update electric field in z-direction."""
        i0 = self._get_prop('n', 'z')
        self._field[-i0:] = fz.ravel('F')

    @property
    def frequency(self):
        """Frequency (Hz)."""
        if self._frequency is None:
            return None
        else:
            return abs(self._frequency)

    @property
    def smu0(self):
        """s*mu_0; mu_0 = magn permeability of free space [H/m]."""
        if getattr(self, '_smu0', None) is None:
            if self.sval is not None:
                self._smu0 = self.sval*mu_0
            else:
                self._smu0 = None

        return self._smu0

    @property
    def sval(self):
        """Laplace parameter; s=iw in f-domain and s=f in Laplace-domain."""

        if getattr(self, '_sval', None) is None:
            if self._frequency is not None:
                if self._frequency < 0:  # Laplace domain; s.
                    self._sval = np.array(-self._frequency)
                else:  # Frequency domain; s = iw = 2i*pi*f.
                    self._sval = np.array(2j*np.pi*self._frequency)
            else:
                self._sval = None

        return self._sval

    def _get_prop(self, pre=None, post=None):
        """Returns `edges` or `faces` property depending on `electric`."""
        name = '' if pre is None else pre + '_'
        name += 'edges' if self.electric else 'faces'
        name += '' if post is None else '_' + post
        return getattr(self.grid, name)

    # INTERPOLATION
    def interpolate_to_grid(self, grid, **interpolate_opts):
        """Interpolate the field to a new grid.


        Parameters
        ----------
        grid : TensorMesh
            New grid; a :class:`emg3d.meshes.TensorMesh` instance.

        interpolate_opts : dict
            Passed through to :func:`emg3d.maps.interpolate`. Defaults are
            ``method='cubic'``, ``log=False``, and ``extrapolate=False``.


        Returns
        -------
        field : Field
            A new :class:`emg3d.fields.Field` instance on ``grid``.

        """

        # Get solver options, set to defaults if not provided.
        g2g_inp = {
            'method': 'cubic',
            'extrapolate': False,
            'log': False,
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
        """Return the field (response) at receiver coordinates.

        Note that in order to avoid boundary effects from the PEC boundary the
        outermost cells are neglected. Field values for coordinates outside of
        the grid are set to NaN's. However, take into account that for good
        results all receivers should be far away from the boundary.

        Parameters
        ----------
        receiver : {Rx*, list, tuple}
            Receiver coordinates. The following formats are accepted:

            - ``Rx*`` instance, any receiver object from
              :mod:`emg3d.electrodes`.
            - ``list``: A list of ``Rx*`` instances.
            - ``tuple``: ``(x, y, z, azimuth, elevation)``; receiver
              coordinates and angles (m, °). All values can either be a scalar
              or having the same length as number of receivers.

            Note that the actual receiver type of the ``Rx*`` instances has no
            effect here, it just takes the coordinates from the receiver
            instances.


        Returns
        -------
        responses : EMArray
            Responses at receiver.

        """
        return get_receiver(self, receiver)


def get_source_field(grid, source, frequency, **kwargs):
    r"""Return source field for provided source and frequency.

    The source term is given in Equation 2 of [Muld06]_,

    .. math::

        -\mathrm{i} \omega \mu_0 \mathbf{J}_\mathrm{s} \, .

    The adjoint of the trilinear interpolation is used to distribute the points
    to the grid edges, which corresponds to the discretization of a Dirac
    ([PlDM07]_).


    Parameters
    ----------
    grid : TensorMesh
        Model grid; a :class:`emg3d.meshes.TensorMesh` instance.

    source : {Tx*, tuple, list, ndarray)
        Any source object from :mod:`emg3d.electrodes` (recommended usage).

        If it is a list, tuple, or ndarray it is in the case of a dipole put
        through to :class:`emg3d.electrodes.TxElectricDipole` or, if
        ``electric=False``, to :class:`emg3d.electrodes.TxMagneticDipole`. If
        it has more than two points it is put through to
        :class:`emg3d.electrodes.TxElectricWire`. Consult the documentation of
        the respective classes for the format, but it is recommended to provide
        directly the classes.

    frequency : float, default: None
        Source frequency (Hz), used to compute the Laplace parameter ``s``.
        Either positive or negative:

        - ``frequency > 0``: Frequency domain, hence
          :math:`s = \mathrm{i}\omega = 2\mathrm{i}\pi f` (complex);
        - ``frequency < 0``: Laplace domain, hence
          :math:`s = f` (real).

    strength : {float, complex}, default: 1.0
        Source strength (A), put through to
        :class:`emg3d.electrodes.TxElectricDipole` or, if ``electric=False``,
        to :class:`emg3d.electrodes.TxMagneticDipole`.

        | *Only used if the provided source is not a source instance.*

    length : float, default: 1.0
        Dipole length (m), put through to
        :class:`emg3d.electrodes.TxElectricDipole` or, if ``electric=False``,
        to :class:`emg3d.electrodes.TxMagneticDipole`.

        | *Only used if the provided source is not a source instance.*

    electric : bool, default: True
        If True, :class:`emg3d.electrodes.TxElectricDipole` is used to get the
        source instance, else :class:`emg3d.electrodes.TxMagneticDipole`.

        | *Only used if the provided source is not a source instance.*


    Returns
    -------
    sfield : Field
        Source field, a :class:`emg3d.fields.Field` instance.


    Examples
    --------

    .. ipython::

       In [1]: import emg3d
          ...: import numpy as np

       In [2]: # Create a simple grid, 8 cells of length 100 m in each
          ...: # direction, centered around the origin.
          ...: hx = np.ones(8)*100
          ...: grid = emg3d.TensorMesh([hx, hx, hx], origin=(-400, -400, -400))
          ...: grid  # For QC

       In [3]: # Create an electric dipole source from
          ...: # x1=y1=z1=0 to x2=100, y2=z2=0; strength=100 A.
          ...: source = emg3d.TxElectricDipole(
          ...:             [[0, 0, 0], [100, 0, 0]], strength=100)
          ...: source  # For QC

       In [4]: # Get the corresponding source field for f=0.5 Hz.
          ...: sfield = emg3d.get_source_field(grid, source, frequency=0.5)
          ...: sfield  # For QC

    """

    # Convert tuples, lists, and ndarrays to Source instances.
    if isinstance(source, (tuple, list, np.ndarray)):

        # Get optional kwargs and cast source.
        inp = {'strength': kwargs.get('strength', 1.0)}
        source = np.asarray(source)
        if source.size == 5:
            inp['length'] = kwargs.get('length', 1.0)

        # Call Tx*-class depending on provided source shape and `electric`.
        if source.size > 6:
            source = electrodes.TxElectricWire(source, **inp)
        elif kwargs.get('electric', True):
            source = electrodes.TxElectricDipole(source, **inp)
        else:
            source = electrodes.TxMagneticDipole(source, **inp)

    # Get total vector field by looping over segments.
    vfield = np.zeros(grid.n_edges, order='F')
    for p0, p1 in zip(source.points[:-1, :], source.points[1:, :]):

        # Add this segments' vector field to total vector field.
        vfield += _dipole_vector(grid, source=np.r_[[p0, p1]]).field

    # Initiate field with the total vector field.
    sfield = Field(grid, data=vfield, frequency=frequency)

    # Multiply by source strength
    sfield.field *= source.strength

    # Multiply by -i*w*mu_0 to obtain the full source term.
    sfield.field *= -sfield.smu0

    return sfield


def get_receiver(field, receiver):
    """Return the field (response) at receiver coordinates.

    Note that in order to avoid boundary effects from the PEC boundary the
    outermost cells are neglected. Field values for coordinates outside of the
    grid are set to NaN's. However, take into account that for good results all
    receivers should be far away from the boundary.


    Parameters
    ----------
    field : Field
        The electric or magnetic field; a :class:`emg3d.fields.Field` instance.

    receiver : {Rx*, list, tuple}
        Receiver coordinates. The following formats are accepted:

        - ``Rx*`` instance, any receiver object from :mod:`emg3d.electrodes`.
        - ``list``: A list of ``Rx*`` instances.
        - ``tuple``: ``(x, y, z, azimuth, elevation)``; receiver coordinates
          and angles (m, °). All values can either be a scalar or having the
          same length as number of receivers.

        Note that the actual receiver type of the ``Rx*`` instances has no
        effect here, it just takes the coordinates from the receiver instances.


    Returns
    -------
    responses : EMArray
        Responses at receiver.

    """

    # Rx* instance.
    if hasattr(receiver, 'coordinates'):
        coordinates = receiver.coordinates

    # List of Rx* instances.
    elif hasattr(tuple(receiver)[0], 'coordinates'):
        nrec = len(receiver)
        coordinates = np.zeros((nrec, 5))
        for i, r in enumerate(receiver):
            coordinates[i, :] = r.coordinates
        coordinates = tuple(coordinates.T)

    # Tuple of coordinates.
    else:
        coordinates = receiver

        # Check receiver dimension.
        if len(coordinates) != 5:
            raise ValueError(
                "`receiver` needs to be in the form "
                "(x, y, z, azimuth, elevation). "
                f"Length of provided `receiver`: {len(coordinates)}."
            )

    # Grid.
    grid = field.grid

    # Pre-allocate the response.
    _, xi, shape = maps._points_from_grids(
            grid, field.fx, coordinates[:3], 'cubic')
    resp = np.zeros(xi.shape[0], dtype=field.field.dtype)

    # Get weighting factors per direction.
    factors = electrodes.rotation(*coordinates[3:])

    # Add the required responses.
    opts = {'method': 'cubic', 'extrapolate': False, 'log': False, 'mode':
            'constant', 'cval': np.nan}
    for i, ff in enumerate((field.fx, field.fy, field.fz)):
        if np.any(abs(factors[i]) > 1e-10):
            resp += factors[i]*maps.interpolate(grid, ff, xi, **opts)

    # PEC: If receivers are in the outermost cell, set them to NaN.
    # Note: Receivers should be MUCH further away from the boundary.
    ind = ((xi[:, 0] < grid.nodes_x[1]) | (xi[:, 0] > grid.nodes_x[-2]) |
           (xi[:, 1] < grid.nodes_y[1]) | (xi[:, 1] > grid.nodes_y[-2]) |
           (xi[:, 2] < grid.nodes_z[1]) | (xi[:, 2] > grid.nodes_z[-2]))
    resp[ind] = np.nan

    # Return response.
    return utils.EMArray(resp.reshape(shape, order='F'))


def get_magnetic_field(model, efield):
    r"""Return magnetic field corresponding to provided electric field.

    Retrieve the magnetic field :math:`\mathbf{H}` from the electric field
    :math:`\mathbf{E}` using Farady's law, given by

    .. math::

        \nabla \times \mathbf{E} = \rm{i}\omega\mu\mathbf{H} .

    Note that the magnetic field is defined on the faces of the grid, unlike
    the electric field which is defined on the edges.


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

    # Initiate magnetic field with zeros.
    hfield = Field(efield.grid, frequency=efield._frequency, electric=False)

    # Get volume-averaged mu_r divided by s*mu_0.
    vmodel = models.VolumeModel(model, efield)
    zeta = vmodel.zeta / efield.smu0

    # Compute magnetic field.
    _edge_curl_factor(
            hfield.fx[1:-1, :, :], hfield.fy[:, 1:-1, :],
            hfield.fz[:, :, 1:-1], efield.fx, efield.fy, efield.fz,
            efield.grid.h[0], efield.grid.h[1], efield.grid.h[2], zeta)

    return hfield


def _dipole_vector(grid, source, decimals=9):
    """Get unit dipole source vector using the adjoint interpolation method.


    Parameters
    ----------
    grid : TensorMesh
        The grid; a :class:`emg3d.meshes.TensorMesh` instance.

    source : ndarray
        Source coordinates of shape (2, 3): [[x1, y1, z1], [x2, y2, z2]] (m).
        Source field, a :class:`emg3d.fields.Field` instance; created if not
        provided.

    decimals : int, default: 9
        Grid nodes and source coordinates are rounded to given number of
        decimals. Default is nanometers.


    Returns
    -------
    vfield : Field
        Source field, a :class:`emg3d.fields.Field` instance.

    """
    vfield = Field(grid, dtype=float)

    # Round nodes and source coordinates (to avoid floating point issues etc).
    nodes_x = np.round(grid.nodes_x, decimals)
    nodes_y = np.round(grid.nodes_y, decimals)
    nodes_z = np.round(grid.nodes_z, decimals)
    source = np.round(np.asarray(source, dtype=float), decimals)

    # Ensure source is within nodes.
    outside = (
        min(source[:, 0]) < nodes_x[0] or max(source[:, 0]) > nodes_x[-1] or
        min(source[:, 1]) < nodes_y[0] or max(source[:, 1]) > nodes_y[-1] or
        min(source[:, 2]) < nodes_z[0] or max(source[:, 2]) > nodes_z[-1]
    )
    if outside:
        raise ValueError(f"Provided source outside grid: {source}.")

    # Dipole lengths in x-, y-, and z-directions, and overall.
    dxdydz = source[1, :] - source[0, :]
    length = np.linalg.norm(dxdydz)

    # Ensure finite length dipole is not a point dipole.
    if length < 1e-15:
        raise ValueError(f"Provided finite dipole has no length: {source}.")

    # Inverse source lengths.
    id_xyz = dxdydz.copy()
    id_xyz[id_xyz != 0] = 1/id_xyz[id_xyz != 0]

    # Cell fractions.
    a1 = (nodes_x - source[0, 0]) * id_xyz[0]
    a2 = (nodes_y - source[0, 1]) * id_xyz[1]
    a3 = (nodes_z - source[0, 2]) * id_xyz[2]

    # Get range of indices of cells in which source resides.
    def min_max_ind(vector, i):
        """Return [min, max]-index of cells in which source resides."""
        vmin = min(source[:, i])
        vmax = max(source[:, i])
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
                aa = np.sort(aa[dxdydz != 0, :], 1)
                al = max(0, aa[:, 0].max())  # Left and right
                ar = min(1, aa[:, 1].min())  # elements.

                # Characteristics of this cell.
                xmin = source[0, :] + al*dxdydz
                xmax = source[0, :] + ar*dxdydz
                x_c = (xmin + xmax) / 2.0
                x_len = np.linalg.norm(xmax - xmin) / length

                # Contribution to edge (coordinate xyz)
                rx = (x_c[0] - nodes_x[ix]) / grid.h[0][ix]
                ex = 1 - rx
                ry = (x_c[1] - nodes_y[iy]) / grid.h[1][iy]
                ey = 1 - ry
                rz = (x_c[2] - nodes_z[iz]) / grid.h[2][iz]
                ez = 1 - rz

                # Add to field (only if segment inside cell).
                if min(rx, ry, rz) >= 0 and np.max(np.abs(ar-al)) > 0:

                    vfield.fx[ix, iy, iz] += ey*ez*x_len
                    vfield.fx[ix, iy+1, iz] += ry*ez*x_len
                    vfield.fx[ix, iy, iz+1] += ey*rz*x_len
                    vfield.fx[ix, iy+1, iz+1] += ry*rz*x_len

                    vfield.fy[ix, iy, iz] += ex*ez*x_len
                    vfield.fy[ix+1, iy, iz] += rx*ez*x_len
                    vfield.fy[ix, iy, iz+1] += ex*rz*x_len
                    vfield.fy[ix+1, iy, iz+1] += rx*rz*x_len

                    vfield.fz[ix, iy, iz] += ex*ey*x_len
                    vfield.fz[ix+1, iy, iz] += rx*ey*x_len
                    vfield.fz[ix, iy+1, iz] += ex*ry*x_len
                    vfield.fz[ix+1, iy+1, iz] += rx*ry*x_len

    # Ensure unity (should not be necessary).
    for field in [vfield.fx, vfield.fy, vfield.fz]:
        sum_s = abs(field.sum())
        if abs(sum_s-1) > 1e-6:
            # Print is always shown and simpler, warn for the CLI logs.
            msg = f"Normalizing Source: {sum_s:.10f}."
            print(f"* WARNING :: {msg}")
            warnings.warn(msg, UserWarning)
            field /= sum_s

    # Multiply by distance in each direction to obtain length.
    vfield.fx *= dxdydz[0]
    vfield.fy *= dxdydz[1]
    vfield.fz *= dxdydz[2]

    return vfield


@nb.njit(**_numba_setting)
def _edge_curl_factor(mx, my, mz, ex, ey, ez, hx, hy, hz, zeta):
    r"""Curl of values living on edges yielding values on faces.

    Used by :func:`emg3d.fields.get_magnetic_field` to compute the magnetic
    field from the electric field. The result is put into ``{mx;my;mz}``.


    Parameters
    ----------
    mx, my, mz : ndarray
        Pre-allocated zero magnetic field in x-, y-, and z-directions
        (:class:`emg3d.fields.Field`). The magnetic field grid has one cell
        less in each direction than the electric field grid.

    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions
        (:class:`emg3d.fields.Field`).

    hx, hy, hz : ndarray
        Cell widths in x-, y-, and z-directions
        (:class:`emg3d.meshes.TensorMesh`).

    zeta : ndarray
        Factor by which nabla x E will be divided. Shape of
        ``efield.grid.shape_cells``. In the case of Farady's law this is
        (smu)^-1, volume-averaged.

    """

    # Get dimensions
    nx = len(hx)
    ny = len(hy)
    nz = len(hz)

    # Loop over dimensions; x-fastest, then y, z
    for iz in range(nz):
        izm = max(0, iz-1)
        izp = iz+1
        for iy in range(ny):
            iym = max(0, iy-1)
            iyp = iy+1
            for ix in range(nx):
                ixm = max(0, ix-1)
                ixp = ix+1

                # Nabla x E.
                fx = ((ez[ix, iyp, iz] - ez[ix, iy, iz])/hy[iy] -
                      (ey[ix, iy, izp] - ey[ix, iy, iz])/hz[iz])
                fy = ((ex[ix, iy, izp] - ex[ix, iy, iz])/hz[iz] -
                      (ez[ixp, iy, iz] - ez[ix, iy, iz])/hx[ix])
                fz = ((ey[ixp, iy, iz] - ey[ix, iy, iz])/hx[ix] -
                      (ex[ix, iyp, iz] - ex[ix, iy, iz])/hy[iy])

                # Average zeta for dual-grid.
                dx = (hx[ixm] + hx[ix])/2
                dy = (hy[iym] + hy[iy])/2
                dz = (hz[izm] + hz[iz])/2
                zeta_x = (zeta[ixm, iy, iz] + zeta[ix, iy, iz])/2
                zeta_y = (zeta[ix, iym, iz] + zeta[ix, iy, iz])/2
                zeta_z = (zeta[ix, iy, izm] + zeta[ix, iy, iz])/2

                # Divide by zeta (averaged over the two cells) and store.
                if ix != 0:
                    mx[ixm, iy, iz] = fx * zeta_x / (dx * hy[iy] * hz[iz])
                if iy != 0:
                    my[ix, iym, iz] = fy * zeta_y / (hx[ix] * dy * hz[iz])
                if iz != 0:
                    mz[ix, iy, izm] = fz * zeta_z / (hx[ix] * hy[iy] * dz)
