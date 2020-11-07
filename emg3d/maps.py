"""
Interpolation routines mapping grids to grids, grids to fields, and fields to
grids.
"""
# Copyright 2018-2020 The emg3d Developers.
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
from scipy import interpolate, ndimage

__all__ = ['grid2grid', 'interp3d', 'MapConductivity', 'MapLgConductivity',
           'MapLnConductivity', 'MapResistivity', 'MapLgResistivity',
           'MapLnResistivity', 'edges2cellaverages']

# Numba-settings
_numba_setting = {'nogil': True, 'fastmath': True, 'cache': True}


# INTERPOLATIONS
def grid2grid(grid, values, new_grid, method='linear', extrapolate=True,
              log=False):
    """Interpolate `values` located on `grid` to `new_grid`.

    **Note 1:**
    The default method is 'linear', because it works with fields and model
    parameters. However, recommended are 'volume' for model parameters and
    'cubic' for fields.

    **Note 2:**
    For model parameters with `method='volume'` the result is quite different
    if you provide resistivity, conductivity, or the logarithm of any of the
    two. The recommended way is to provide the logarithm of resistivity or
    conductivity, in which case the output of one is indeed the inverse of the
    output of the other.


    Parameters
    ----------
    grid, new_grid : TensorMesh
        Input and output model grids;
        :class:`TensorMesh` instances.

    values : ndarray
        Model parameters; :class:`emg3d.fields.Field` instance, or a particular
        field (e.g. field.fx). For fields the method cannot be 'volume'.

    method : {<'linear'>, 'volume', 'cubic'}, optional
        The method of interpolation to perform. The volume averaging method
        ensures that the total sum of the property stays constant.

        Volume averaging is only implemented for model parameters, not for
        fields. The method 'cubic' requires at least three points in any
        direction, otherwise it will fall back to 'linear'.

        Default is 'linear', because it works with fields and model parameters.
        However, recommended are 'volume' for model parameters and 'cubic' for
        fields.

    extrapolate : bool
        If True, points on `new_grid` which are outside of `grid` are filled by
        the nearest value (if ``method='cubic'``) or by extrapolation (if
        ``method='linear'``). If False, points outside are set to zero.

        For ``method='volume'`` it always uses the nearest value for points
        outside of `grid`.

        Default is True.

    log : bool
        If True, the interpolation is carried out on a log10-scale; hence the
        same as ``10**grid2grid(grid, np.log10(values), ...)``.
        Default is False.


    Returns
    -------
    new_values : ndarray
        Values corresponding to `new_grid`.


    See Also
    --------
    get_receiver : Interpolation of model parameters or fields to (x, y, z).

    """

    # If values is a Field instance, call it recursively for each field.
    if hasattr(values, 'field') and values.field.ndim == 1:
        fx = grid2grid(grid, np.asarray(values.fx), new_grid, method,
                       extrapolate, log)
        fy = grid2grid(grid, np.asarray(values.fy), new_grid, method,
                       extrapolate, log)
        fz = grid2grid(grid, np.asarray(values.fz), new_grid, method,
                       extrapolate, log)

        # Return a field instance.
        return values.__class__(fx, fy, fz)

    # If values is a particular field, ensure method is not 'volume'.
    if not np.all(grid.vnC == values.shape) and method == 'volume':
        raise ValueError("``method='volume'`` not implemented for fields.")

    if method == 'volume':
        points = (grid.vectorNx, grid.vectorNy, grid.vectorNz)
        new_points = (new_grid.vectorNx, new_grid.vectorNy, new_grid.vectorNz)
        new_values = np.zeros(new_grid.vnC, dtype=values.dtype)
        vol = new_grid.vol.reshape(new_grid.vnC, order='F')

        # Get values from `volume_average`.
        if log:
            volume_average(*points, np.log10(values), *new_points,
                           new_values, vol)
        else:
            volume_average(*points, values, *new_points, new_values, vol)

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

        # Get values from `interp3d`.
        if extrapolate:
            fill_value = None
            mode = 'nearest'
        else:
            fill_value = 0.0
            mode = 'constant'

        if log:
            new_values = interp3d(points, np.log10(values), new_points,
                                  method, fill_value, mode)
        else:
            new_values = interp3d(points, values, new_points, method,
                                  fill_value, mode)

        new_values = new_values.reshape(shape, order='F')

    if log:
        return 10**new_values
    else:
        return new_values


def interp3d(points, values, new_points, method, fill_value, mode):
    """Interpolate values in 3D either linearly or with a cubic spline.

    Return `values` corresponding to a regular 3D grid defined by `points` on
    `new_points`.

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
        Passed to :class:`scipy.interpolate.RegularGridInterpolator` if
        ``method='linear'``: The value to use for points outside of the
        interpolation domain. If None, values outside the domain are
        extrapolated.

    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}
        Passed to :func:`scipy.ndimage.map_coordinates` if ``method='cubic'``:
        Determines how the input array is extended beyond its boundaries.


    Returns
    -------
    new_values : ndarray
        Values corresponding to `new_points`.

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


# MAPS
MAPLIST = {}


def register_map(func):
    MAPLIST[func.__name__] = func
    return func


class _Map:
    """Maps variable `x` to computational variable `σ` (conductivity)."""

    def __init__(self, description):
        """Initiate the map."""
        self.name = self.__class__.__name__[3:]
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

    def to_dict(self):
        """Store the map name in a dict for serialization."""
        return {'name': self.name, '__class__': '_Map'}

    @classmethod
    def from_dict(cls, inp):
        """Get :class:`_Map` instance from name in dict."""
        return MAPLIST['Map'+inp['name']]()


@register_map
class MapConductivity(_Map):
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


@register_map
class MapLgConductivity(_Map):
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


@register_map
class MapLnConductivity(_Map):
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


@register_map
class MapResistivity(_Map):
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


@register_map
class MapLgResistivity(_Map):
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


@register_map
class MapLnResistivity(_Map):
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


# VOLUME AVERAGING
@nb.njit(**_numba_setting)
def volume_average(edges_x, edges_y, edges_z, values,
                   new_edges_x, new_edges_y, new_edges_z, new_values, new_vol):
    """Interpolation using the volume averaging technique.

    The result is added to new_values.

    The original implementation (see ``emg3d v0.7.1``) followed [PlDM07]_. Joe
    Capriot took that algorithm and made it much faster for implementation in
    `discretize`. The current implementation is a simplified Numba-version of
    his Cython version (the `discretize` version works for 1D, 2D, and 3D
    meshes and can also return a sparse matrix representing the operation).


    Parameters
    ----------
    edges_[x, y, z] : ndarray
        The edges in x-, y-, and z-directions for the original grid.

    values : ndarray
        Values corresponding to `grid`.

    new_edges_[x, y, z] : ndarray
        The edges in x-, y-, and z-directions for the new grid.

    new_values : ndarray
        Array where values corresponding to `new_grid` will be added.

    new_vol : ndarray
        The volumes of the `new_grid`-cells.

    """

    # Get the weights and indices for each direction.
    wx, ix_in, ix_out = _volume_average_weights(edges_x, new_edges_x)
    wy, iy_in, iy_out = _volume_average_weights(edges_y, new_edges_y)
    wz, iz_in, iz_out = _volume_average_weights(edges_z, new_edges_z)

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
def _volume_average_weights(x1, x2):
    """Returaveragethe weights for the volume averaging technique.


    Parameters
    ----------
    x1, x2 : ndarray
        The edges in x-, y-, or z-directions for the original (x1) and the new
        (x2) grids.


    Returns
    -------
    hs : ndarray
        Weights for the mapping of x1 to x2.

    ix1, ix2 : ndarray
        Indices to map x1 to x2.

    """
    # Fill xs with uniques and truncate.
    # Corresponds to np.unique(np.concatenate([x1, x2])).
    n1, n2 = len(x1), len(x2)
    xs = np.empty(n1 + n2)  # Pre-allocate array containing all edges.
    i1, i2, i = 0, 0, 0
    while i1 < n1 or i2 < n2:
        if i1 < n1 and i2 < n2:
            if x1[i1] < x2[i2]:
                xs[i] = x1[i1]
                i1 += 1
            elif x1[i1] > x2[i2]:
                xs[i] = x2[i2]
                i2 += 1
            else:
                xs[i] = x1[i1]
                i1 += 1
                i2 += 1
        elif i1 < n1 and i2 == n2:
            xs[i] = x1[i1]
            i1 += 1
        elif i2 < n2 and i1 == n1:
            xs[i] = x2[i2]
            i2 += 1
        i += 1

    # Get weights and indices for the two arrays.
    # - hs corresponds to np.diff(xs) where x1 and x2 overlap; zero outside.
    # - x1[ix1] can be mapped to x2[ix2] with the corresponding weight.
    nh = i-1
    hs = np.empty(nh)                   # Pre-allocate weights.
    ix1 = np.zeros(nh, dtype=np.int32)  # Pre-allocate indices for x1.
    ix2 = np.zeros(nh, dtype=np.int32)  # Pre-allocate indices for x2.
    center = 0.0
    i1, i2, i, ii = 0, 0, 0, 0
    for i in range(nh):
        center = 0.5*(xs[i]+xs[i+1])
        if x2[0] <= center and center <= x2[n2-1]:
            hs[ii] = xs[i+1]-xs[i]
            while i1 < n1-1 and center >= x1[i1]:
                i1 += 1
            while i2 < n2-1 and center >= x2[i2]:
                i2 += 1
            ix1[ii] = min(max(i1-1, 0), n1-1)
            ix2[ii] = min(max(i2-1, 0), n2-1)
            ii += 1

    return hs[:ii], ix1[:ii], ix2[:ii]


# EDGES => CENTERS
@nb.njit(**_numba_setting)
def edges2cellaverages(ex, ey, ez, vol, out_x, out_y, out_z):
    r"""Interpolate fields defined on edges to volume-averaged cell values.


    Parameters
    ----------
    ex, ey, ez : ndarray
        Electric fields in x-, y-, and z-directions, as obtained from
        :class:`emg3d.fields.Field`.

    vol : ndarray
        Volumes of the grid, as obtained from :class:`emg3d.meshes.TensorMesh`.

    out_x, out_y, out_z : ndarray
        Arrays where the results are placed (per direction).

    """

    # Get dimensions
    nx, ny, nz = vol.shape

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
                    out_x[ix, iym, izm] += vol[ix, iym, izm]*ex[ix, iy, iz]/4
                    out_x[ix, iyp, izm] += vol[ix, iyp, izm]*ex[ix, iy, iz]/4
                    out_x[ix, iym, izp] += vol[ix, iym, izp]*ex[ix, iy, iz]/4
                    out_x[ix, iyp, izp] += vol[ix, iyp, izp]*ex[ix, iy, iz]/4

                if iy < ny:
                    out_y[ixm, iy, izm] += vol[ixm, iy, izm]*ey[ix, iy, iz]/4
                    out_y[ixp, iy, izm] += vol[ixp, iy, izm]*ey[ix, iy, iz]/4
                    out_y[ixm, iy, izp] += vol[ixm, iy, izp]*ey[ix, iy, iz]/4
                    out_y[ixp, iy, izp] += vol[ixp, iy, izp]*ey[ix, iy, iz]/4

                if iz < nz:
                    out_z[ixm, iym, iz] += vol[ixm, iym, iz]*ez[ix, iy, iz]/4
                    out_z[ixp, iym, iz] += vol[ixp, iym, iz]*ez[ix, iy, iz]/4
                    out_z[ixm, iyp, iz] += vol[ixm, iyp, iz]*ez[ix, iy, iz]/4
                    out_z[ixp, iyp, iz] += vol[ixp, iyp, iz]*ez[ix, iy, iz]/4
