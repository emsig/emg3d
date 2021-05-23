"""
Mapping routines to map to and from linear conductivities (what is used
internally) to other representations such as resistivities or logarithms
thereof.

Interpolation routines mapping values between different grids.
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

import numba as nb
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator, interpnd, interp1d

from emg3d.core import _numba_setting

__all__ = ['BaseMap', 'MapConductivity', 'MapLgConductivity',
           'MapLnConductivity', 'MapResistivity', 'MapLgResistivity',
           'MapLnResistivity', 'interpolate', 'interp_spline_3d',
           'interp_volume_average', 'interp_edges_to_vol_averages']


class BaseMap:
    """Maps variable `x` to computational variable `σ` (conductivity).

    Subclass this BaseMap to create new maps. A map class must start with
    ``Map`` followed by a name, e.g., ``MapProperty``.

    To be able to load custom maps using ``emg3d.io.load`` define the map
    before you load the files, and register the map by putting the decorator
    ``emg3d.maps.register_map``. This will enable the I/O to properly
    instantiate your custom maps.

    .. code-block:: python

        @emg3d.maps.register_map
        class MapProperty(emg3d.maps.BaseMap):
            '''Description'''
            def __init__(self):
                super().__init__('property')

            def forward(self, conductivity):
                return # Mapping from your property to conductivity.

            def backward(self, mapped):
                return # Mapping from conductivity to your property.

            def derivative_chain(self, gradient, mapped):
                gradient *= # Chain rule of your backward mapping.

    """

    def __init__(self, description):
        """Initiate the map."""
        self.name = self.__class__.__name__[3:]  # Class name without `Map`
        self.description = description

    def __repr__(self):
        return (f"{self.__class__.__name__}: {self.description}\n"
                "    Maps investigation variable `x` to\n"
                "    computational variable `σ` (conductivity).")

    def forward(self, conductivity):
        """Conductivity to mapping."""
        raise NotImplementedError("Forward map not implemented.")

    def backward(self, mapped):
        """Mapping to conductivity."""
        raise NotImplementedError("Backward map not implemented.")

    def derivative_chain(self, gradient, mapped):
        """Chain rule to map gradient from conductivity to mapping space."""
        raise NotImplementedError("Derivative chain not implemented.")


class MapConductivity(BaseMap):
    """Maps `σ` to computational variable `σ` (conductivity).

    - forward: x = σ
    - backward: σ = x

    """

    def __init__(self):
        super().__init__('conductivity')

    def forward(self, conductivity):
        return conductivity

    def backward(self, mapped):
        return mapped

    def derivative_chain(self, gradient, mapped):
        pass


class MapLgConductivity(BaseMap):
    """Maps `log_10(σ)` to computational variable `σ` (conductivity).

    - forward: x = log_10(σ)
    - backward: σ = 10^x

    """

    def __init__(self):
        super().__init__('log_10(conductivity)')

    def forward(self, conductivity):
        return np.log10(conductivity)

    def backward(self, mapped):
        return 10**mapped

    def derivative_chain(self, gradient, mapped):
        gradient *= self.backward(mapped)*np.log(10)


class MapLnConductivity(BaseMap):
    """Maps `log_e(σ)` to computational variable `σ` (conductivity).

    - forward: x = log_e(σ)
    - backward: σ = exp(x)

    """

    def __init__(self):
        super().__init__('log_e(conductivity)')

    def forward(self, conductivity):
        return np.log(conductivity)

    def backward(self, mapped):
        return np.exp(mapped)

    def derivative_chain(self, gradient, mapped):
        gradient *= self.backward(mapped)


class MapResistivity(BaseMap):
    """Maps `ρ` to computational variable `σ` (conductivity).

    - forward: x = ρ = σ^-1
    - backward: σ = ρ^-1 = x^-1

    """

    def __init__(self):
        super().__init__('resistivity')

    def forward(self, conductivity):
        return 1.0/conductivity

    def backward(self, mapped):
        return 1.0/mapped

    def derivative_chain(self, gradient, mapped):
        gradient *= -self.backward(mapped)**2


class MapLgResistivity(BaseMap):
    """Maps `log_10(ρ)` to computational variable `σ` (conductivity).

    - forward: x = log_10(ρ) = log_10(σ^-1)
    - backward: σ = ρ^-1 = 10^-x

    """

    def __init__(self):
        super().__init__('log_10(resistivity)')

    def forward(self, conductivity):
        return np.log10(1.0/conductivity)

    def backward(self, mapped):
        return 10**-mapped

    def derivative_chain(self, gradient, mapped):
        gradient *= -self.backward(mapped)*np.log(10)


class MapLnResistivity(BaseMap):
    """Maps `log_e(ρ)` to computational variable `σ` (conductivity).

    - forward: x = log_e(ρ) = log_e(σ^-1)
    - backward: σ = ρ^-1 = exp(-x)

    """

    def __init__(self):
        super().__init__('log_e(resistivity)')

    def forward(self, conductivity):
        return np.log(1.0/conductivity)

    def backward(self, mapped):
        return np.exp(-mapped)

    def derivative_chain(self, gradient, mapped):
        gradient *= -self.backward(mapped)


# INTERPOLATIONS
def interpolate(grid, values, xi, method='linear', extrapolate=True,
                log=False, **kwargs):
    """Interpolate values from one grid to another grid or to points.


    Parameters
    ----------
    grid : TensorMesh
        Input grid; a :class:`emg3d.meshes.TensorMesh` instance.

    values : ndarray
        A model property such as ``Model.property_x``, or a field such as
        ``Field.fx`` (``ndim=3``; the dimension in each direction must either
        correspond to the number of nodes or cell centers in the corresponding
        direction).

    xi : {TensorMesh, tuple, ndarray}
        Output coordinates; possibilities:

        - A grid (:class:`emg3d.meshes.TensorMesh`): interpolation from one
          grid to another.
        - A tuple (array_like, array_like, array_like) containing x-, y-, and
          z-coordinates. The length of each can be either one or the number of
          coordinates, the size-one elements will be expanded internally to the
          length of the coordinates. E.g., ``(x, [y0, y1, y2], z)`` will be
          expanded to ``([x, x, x], [y0, y1, y2], [z, z, z])``.
        - Arbitrary point coordinates as ``ndarray`` of shape ``(..., 3)``,
          e.g., ``array([[x0, y0, z0], ..., [xN, yN, zN]))``.

    method : {'nearest', 'linear', 'volume', 'cubic'}, default: 'linear'
        The method of interpolation to perform.

        - ``'nearest', 'linear'``: Fastest methods; work for model properties
          and fields living on edges or faces. Carried out with
          :class:`scipy.interpolate.RegularGridInterpolator`.

        - ``'cubic'``: Cubic spline interpolation using
          :func:`emg3d.maps.interp_spline_3d`.

        - ``'volume'``: Volume average interpolation using
          :func:`emg3d.maps.interp_volume_average`.

          Volume average interpolation ensures that the total sum of the
          interpolated quantity stays constant. The result can be quite
          different if you provide resistivity, conductivity, or the logarithm
          of any of the two. The recommended way is to use ``log=True``, in
          which case the output is the same for conductivities and
          resistivities.

          This method is only implemented for quantities living on cell
          centers, not on edges/faces (hence not for fields); and only for
          grids as input to ``xi``.

    extrapolate : bool, default: True
        This parameter controls the default parameters provided to the
        interpolation routines.

        - ``'nearest', 'linear'``: If True, values outside of the domain are
          extrapolated (``bounds_error=False, fill_value=None``); if False,
          values outside are set to 0.0 (``bounds_error=False,
          fill_value=0.0``)

        - ``'cubic'``: If True, values outside of the domain are extrapolated
          using nearest interpolation (``mode='nearest'``); if False, values
          outside are set to 0.0 (``mode='constant', cval=0.0``).

        - ``'volume'``: Always uses nearest interpolation for points outside of
          the provided grid, independent of the choice of ``extrapolate``.

    log : bool, default: False
        If True, the interpolation is carried out on a log10-scale; this
        corresponds to ``10**interpolate(grid, np.log10(values), ...)``.

    kwargs : dict, optional
        Will be forwarded to the corresponding interpolation algorithm, if they
        accept additional keywords. This can be used, e.g., to change the
        behaviour outlined in the parameter ``extrapolate``.


    Returns
    -------
    values_x : ndarray
        Values corresponding to the new grid.

    """

    # Take log10 if set.
    if log:
        values = np.log10(values)

    # Get points in the right shape.
    points, new_points, shape = _points_from_grids(grid, values, xi, method)

    # Carry out the actual interpolation.
    if method == 'volume':

        # Pre-allocate output.
        values_x = np.zeros(shape, order='F', dtype=values.dtype)

        interp_volume_average(
                nodes_x=points[0], nodes_y=points[1], nodes_z=points[2],
                values=values, new_nodes_x=new_points[0],
                new_nodes_y=new_points[1], new_nodes_z=new_points[2],
                new_values=values_x,
                new_vol=xi.cell_volumes.reshape(shape, order='F'))

    elif method == 'cubic':

        opts = {
            'mode': 'nearest' if extrapolate else 'constant',
            **({} if kwargs is None else kwargs),
        }

        values_x = interp_spline_3d(
                points=points, values=values, xi=new_points, **opts)

    else:  # 'nearest'/'linear' (will raise ValueError if unknown method).

        opts = {
            'bounds_error': False,
            'fill_value': None if extrapolate else 0.0,
            **({} if kwargs is None else kwargs),
        }

        values_x = RegularGridInterpolator(
                points=points, values=values, method=method,
                **opts)(xi=new_points)

    # Return to linear if log10 was applied.
    if log:
        values_x = 10**values_x

    # Reshape and return.
    return values_x.reshape(shape, order='F')


def _points_from_grids(grid, values, xi, method):
    """Return `points` and `new_points` from original grid and new grid/points.

    Returns ``points``, ``new_points``, and ``shape`` to use with
    :func:`emg3d.maps.interp_volume_average`,
    :func:`emg3d.maps.interp_spline_3d`, and
    :class:`scipy.interpolate.RegularGridInterpolator`.


    For the input parameters, see :func:`emg3d.maps.interpolate`.

    Returns
    -------
    points : (ndarray, ndarray, ndarray)
        Tuple containing the x-, y-, and z-coordinates of the input values.

    new_points : {(ndarray, ndarray, ndarray); ndarray}
        Depends on the ``method``:

        - If ``method='volume'``: (ndarray, ndarray, ndarray)

          Tuple containing the x-, y-, and z-coordinates of the output values.

        - Else: ndarray

          Coordinates in an ndarray of shape (..., 3):
          ``array([[x1, y1, z1], ..., [xn, yn, zn]])``.

    shape : tuple
        Final shape of the output values.

    """

    # Specific checks for method='volume'.
    if method == 'volume':
        msg = "``method='volume'`` is only implemented for "

        # 'xi' must be a TensorMesh.
        if not hasattr(xi, 'nodes_x'):
            msg += "TensorMesh instances as input for ``xi``."
            raise ValueError(msg)

        # Shape of the values must correspond to shape of cells.
        if grid.shape_cells != values.shape:
            msg += "cell-centered properties; required shape = "
            raise ValueError(msg + f"{grid.shape_cells}.")

    # General dimensionality check.
    else:
        electric = [grid.shape_edges_x, grid.shape_faces_y, grid.shape_edges_z]
        magnetic = [grid.shape_faces_x, grid.shape_edges_y, grid.shape_faces_z]
        centered = [grid.shape_cells, ]
        if values.shape not in np.r_[electric, magnetic, centered]:
            msg = ("``values`` must be a 3D ndarray living on cell centers, "
                   "edges, or faces of the ``grid``.")
            raise ValueError(msg)

    # Get electric flag (living on edges vs living on faces).
    electric = values.shape not in [grid.shape_faces_x, grid.shape_edges_y,
                                    grid.shape_faces_z]

    # Check if 'xi' is a TensorMesh.
    xi_is_grid = hasattr(xi, 'nodes_x')

    # # Get points from input # #

    # 1. Get required tuples from input grids.
    points = tuple()
    if xi_is_grid:
        new_points = tuple()
        shape = tuple()

    # Loop over dimensions to get the vectors corresponding to input data.
    for i, coord in enumerate(['x', 'y', 'z']):

        # Cell nodes.
        comp_shape = [grid.shape_cells[i], grid.shape_nodes[i]][electric]
        if method == 'volume' or values.shape[i] == comp_shape:
            prop = ['cell_centers_', 'nodes_'][electric]
            pts = getattr(grid, prop + coord)
            if xi_is_grid:
                new_pts = getattr(xi, prop + coord)

        # Cell centers.
        else:
            prop = ['nodes_', 'cell_centers_'][electric]
            pts = getattr(grid, prop + coord)
            if xi_is_grid:
                new_pts = getattr(xi, prop + coord)

        # Add to points.
        points += (pts, )
        if xi_is_grid:
            new_points += (new_pts, )
            shape += (len(new_pts), )

    # After this step the points/new_points are:
    # points: (x-points, y-points, z-points)
    # new_points: (new-x-points, new-y-points, new-z-points)  # if xi_is_grid

    # 'volume' takes new_points as tuples. However, the other methods take an
    # (..., 3) ndarray of the coordinates.
    if method != 'volume':

        # # Convert points to correct format # #
        if xi_is_grid:
            xx, yy, zz = np.broadcast_arrays(
                    new_points[0][:, None, None],
                    new_points[1][:, None],
                    new_points[2])
            new_points = np.r_[xx.ravel('F'), yy.ravel('F'), zz.ravel('F')]
            new_points = new_points.reshape(-1, 3, order='F')

        else:
            # Replicate the same expansion of xi as used in
            # RegularGridInterpolator, so the input xi can be quite flexible.
            new_points = interpnd._ndim_coords_from_arrays(
                    xi, ndim=3)
            shape = new_points.shape[:-1]
            new_points = new_points.reshape(-1, 3, order='F')

        # After this step the new_points are:
        # new_points: array([[x1, y1, z1], ..., [xn, yn, zn]])

    else:
        shape = xi.shape_cells

    return points, new_points, shape


def interp_spline_3d(points, values, xi, **kwargs):
    """Interpolate values in 3D with a cubic spline.

    This functionality is best accessed through :func:`emg3d.maps.interpolate`
    by setting ``method='cubic'``.

    3D cubic spline interpolation is achieved by mapping the ``points`` to
    regular indices and interpolate with cubic splines
    (:class:`scipy.interpolate.interp1d`) the ``xi`` to this artificial
    coordinate system. The ``values`` can then be interpolated from ``points``
    to ``xi`` on this transformed coordinate system using cubic spline
    interpolation through :func:`scipy.ndimage.map_coordinates`.


    Parameters
    ----------
    points : (ndarray, ndarray, ndarray)
        The points defining the regular grid in (x, y, z) direction.

    values : ndarray
        The data on the regular grid in three dimensions (nx, ny, nz).

    xi : ndarray
        Coordinates (x, y, z) of new points, shape ``(..., 3)``.

    kwargs : dict, optional
        Passed through to :func:`scipy.ndimage.map_coordinates`.
        Potentially valuable keywords to pass are

        - ``order``: which has to be in the range of 0-5, default: 3;
        - ``mode``: default is ``'constant'``, options include ``'nearest'``;
        - ``cval``: the value to fill past edges if ``mode='constant'``,
          default is 0.0.


    Returns
    -------
    values_x : ndarray
        Values corresponding to ``xi``.

    """

    # `map_coordinates` uses the indices of the input data (our values) as
    # coordinates. We have therefore to transform our desired output
    # coordinates to this artificial coordinate system too.
    coords = np.empty(xi.T.shape)
    for i in range(3):
        coords[i] = interp1d(points[i], np.arange(len(points[i])),
                             kind='cubic', bounds_error=False,
                             fill_value='extrapolate')(xi[:, i])

    # `map_coordinates` only works for real data; split it up if complex.
    # Note: SciPy 1.6 (12/2020) introduced complex-valued
    #       ndimage.map_coordinates; replace eventually.
    values_x = map_coordinates(values.real, coords, **kwargs)
    if 'complex' in values.dtype.name:
        imag = map_coordinates(values.imag, coords, **kwargs)
        values_x = values_x + 1j*imag

    return values_x


@nb.njit(**_numba_setting)
def interp_volume_average(
        nodes_x, nodes_y, nodes_z, values, new_nodes_x, new_nodes_y,
        new_nodes_z, new_values, new_vol):
    """Interpolate properties from `grid` to `new_grid` using volume averages.

    This functionality is best accessed through :func:`emg3d.maps.interpolate`
    by setting ``method='volume'``.

    Interpolation using the volume averaging technique. The original
    implementation (see ``emg3d v0.7.1``) followed [PlDM07]_. Joseph Capriotti
    took that algorithm and made it much faster for implementation in
    *discretize*. The current implementation is a translation of that from
    Cython to Numba, heavily simplified for the 3D use case in *emg3d*.

    The result is added to ``new_values``.


    Parameters
    ----------
    nodes_{x;y;z} : ndarray
        The nodes in x-, y-, and z-directions for the original grid,
        ``grid.nodes_{x;y;z}``, from a :func:`emg3d.meshes.TensorMesh`
        instance.

    values : ndarray
        Values corresponding to original grid (of shape ``grid.shape_cells``).

    new_nodes_{x;y;z} : ndarray
        The nodes in x-, y-, and z-directions for the new grids,
        ``new_grid.nodes_{x;y;z}``, from a :func:`emg3d.meshes.TensorMesh`
        instance.

    new_values : ndarray
        Array where values corresponding to the new grid will be added (of
        shape ``new_grid.shape_cells``).

    new_vol : ndarray
        The cell volumes of the new grid (``new_grid.cell_volumes``).

    """

    # Get the weights and indices for each direction.
    wx, ix_in, ix_out = _volume_average_weights(nodes_x, new_nodes_x)
    wy, iy_in, iy_out = _volume_average_weights(nodes_y, new_nodes_y)
    wz, iz_in, iz_out = _volume_average_weights(nodes_z, new_nodes_z)

    # Loop over the elements and sum up the contributions.
    for iz, w_z in enumerate(wz):
        izi = iz_in[iz]
        izo = iz_out[iz]
        for iy, w_y in enumerate(wy):
            iyi = iy_in[iy]
            iyo = iy_out[iy]
            w_zy = w_z*w_y
            for ix, w_x in enumerate(wx):
                ixi = ix_in[ix]
                ixo = ix_out[ix]
                new_values[ixo, iyo, izo] += w_zy*w_x*values[ixi, iyi, izi]

    # Normalize by new volume.
    new_values /= new_vol


@nb.njit(**_numba_setting)
def _volume_average_weights(x_i, x_o):
    """Return weights for volume averaging technique.


    Parameters
    ----------
    x_i, x_o : ndarray
        The nodes in x-, y-, or z-directions for the input (x_i) and output
        (x_o) grids.


    Returns
    -------
    hs : ndarray
        Weights for the mapping of x_i to x_o.

    ix_i, ix_o : ndarray
        Indices to map x_i to x_o.

    """
    # Get unique nodes.
    xs = np.unique(np.concatenate((x_i, x_o)))
    n1, n2, nh = len(x_i), len(x_o), len(xs)-1

    # Get weights and indices for the two arrays.
    # - wx corresponds to np.diff(xs) where x_i and x_o overlap; zero outside.
    # - x_i[ix_i] can be mapped to x_o[ix_o] with the corresponding weight.
    wx = np.empty(nh)                   # Pre-allocate weights.
    ix_i = np.zeros(nh, dtype=np.int32)  # Pre-allocate indices for x_i.
    ix_o = np.zeros(nh, dtype=np.int32)  # Pre-allocate indices for x_o.
    center = 0.0
    i1, i2, i, ii = 0, 0, 0, 0
    for i in range(nh):
        center = 0.5*(xs[i]+xs[i+1])
        if x_o[0] <= center and center <= x_o[n2-1]:
            wx[ii] = xs[i+1]-xs[i]
            while i1 < n1-1 and center >= x_i[i1]:
                i1 += 1
            while i2 < n2-1 and center >= x_o[i2]:
                i2 += 1
            ix_i[ii] = min(max(i1-1, 0), n1-1)
            ix_o[ii] = min(max(i2-1, 0), n2-1)
            ii += 1

    return wx[:ii], ix_i[:ii], ix_o[:ii]


@nb.njit(**_numba_setting)
def interp_edges_to_vol_averages(ex, ey, ez, volumes, ox, oy, oz):
    r"""Interpolate fields defined on edges to volume-averaged cell values.

    Parameters
    ----------
    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions from a
        :func:`emg3d.fields.Field` instance (``field.f{x;y;z}``).

    volumes : ndarray
        Cell volumes of the corresponding grid (``field.grid.cell_volumes``).

    ox, oy, oz : ndarray
        Output arrays where the results are placed (of shape
        ``field.grid.shape_cells``).

    """

    # Get dimensions
    nx, ny, nz = volumes.shape

    # Loop over dimensions.
    for iz in range(nz+1):
        izm = max(0, iz-1)
        izp = min(nz-1, iz)

        for iy in range(ny+1):
            iym = max(0, iy-1)
            iyp = min(ny-1, iy)

            for ix in range(nx+1):
                ixm = max(0, ix-1)
                ixp = min(nx-1, ix)

                # Multiply field by volume/4.
                if ix < nx:
                    ox[ix, iym, izm] += volumes[ix, iym, izm]*ex[ix, iy, iz]/4
                    ox[ix, iyp, izm] += volumes[ix, iyp, izm]*ex[ix, iy, iz]/4
                    ox[ix, iym, izp] += volumes[ix, iym, izp]*ex[ix, iy, iz]/4
                    ox[ix, iyp, izp] += volumes[ix, iyp, izp]*ex[ix, iy, iz]/4

                if iy < ny:
                    oy[ixm, iy, izm] += volumes[ixm, iy, izm]*ey[ix, iy, iz]/4
                    oy[ixp, iy, izm] += volumes[ixp, iy, izm]*ey[ix, iy, iz]/4
                    oy[ixm, iy, izp] += volumes[ixm, iy, izp]*ey[ix, iy, iz]/4
                    oy[ixp, iy, izp] += volumes[ixp, iy, izp]*ey[ix, iy, iz]/4

                if iz < nz:
                    oz[ixm, iym, iz] += volumes[ixm, iym, iz]*ez[ix, iy, iz]/4
                    oz[ixp, iym, iz] += volumes[ixp, iym, iz]*ez[ix, iy, iz]/4
                    oz[ixm, iyp, iz] += volumes[ixm, iyp, iz]*ez[ix, iy, iz]/4
                    oz[ixp, iyp, iz] += volumes[ixp, iyp, iz]*ez[ix, iy, iz]/4
