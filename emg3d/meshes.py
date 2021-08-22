"""
Everything related to meshes appropriate for the multigrid solver.
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

import numpy as np
from scipy.constants import mu_0

from emg3d import maps, utils

try:
    import discretize
except ImportError:
    discretize = None

__all__ = ['TensorMesh', 'BaseMesh', 'construct_mesh', 'origin_and_widths',
           'good_mg_cell_nr', 'skin_depth', 'wavelength', 'cell_width',
           'check_mesh', 'estimate_gridding_opts']


class BaseMesh:
    """Minimal TensorMesh for internal multigrid computation.

    The base mesh has everything that is needed within
    :func:`emg3d.solver.solve`, but nothing more.

    Parameters
    ----------
    h : [array_like, array_like, array_like]
        Cell widths in x, y, and z directions.

    origin : array_like
        Origin (x, y, z).


    Examples
    --------

    .. ipython::

       In [1]: import emg3d
          ...: import numpy as np

       In [2]: # Create a simple grid, 8 cells of length 100 m in each
          ...: # direction, centered around the origin.
          ...: hx = np.ones(8)*100
          ...: grid = emg3d.meshes.BaseMesh(
          ...:            [hx, hx, hx], origin=(-400, -400, -400))
          ...: grid  # QC grid

    """

    def __init__(self, h, origin, **kwargs):
        """Initialize the mesh."""

        # Store origin.
        self.origin = np.array(origin)

        # Width of cells, cast to arrays.
        self.h = [np.array(h[0]), np.array(h[1]), np.array(h[2])]

        # Node related properties.
        shape_nodes = (self.h[0].size+1, self.h[1].size+1, self.h[2].size+1)
        self.shape_nodes = shape_nodes
        self.nodes_x = np.r_[0., self.h[0].cumsum()] + self.origin[0]
        self.nodes_y = np.r_[0., self.h[1].cumsum()] + self.origin[1]
        self.nodes_z = np.r_[0., self.h[2].cumsum()] + self.origin[2]

        # Cell related properties.
        shape_cells = (self.h[0].size, self.h[1].size, self.h[2].size)
        self.shape_cells = shape_cells
        self.n_cells = np.prod(shape_cells)
        self.cell_centers_x = (self.nodes_x[1:] + self.nodes_x[:-1])/2
        self.cell_centers_y = (self.nodes_y[1:] + self.nodes_y[:-1])/2
        self.cell_centers_z = (self.nodes_z[1:] + self.nodes_z[:-1])/2

        # Edge related properties.
        self.shape_edges_x = (shape_cells[0], shape_nodes[1], shape_nodes[2])
        self.shape_edges_y = (shape_nodes[0], shape_cells[1], shape_nodes[2])
        self.shape_edges_z = (shape_nodes[0], shape_nodes[1], shape_cells[2])
        self.n_edges_x = np.prod(self.shape_edges_x)
        self.n_edges_y = np.prod(self.shape_edges_y)
        self.n_edges_z = np.prod(self.shape_edges_z)
        self.n_edges = self.n_edges_x + self.n_edges_y + self.n_edges_z

        # Face related properties.
        self.shape_faces_x = (shape_nodes[0], shape_cells[1], shape_cells[2])
        self.shape_faces_y = (shape_cells[0], shape_nodes[1], shape_cells[2])
        self.shape_faces_z = (shape_cells[0], shape_cells[1], shape_nodes[2])
        self.n_faces_x = np.prod(self.shape_faces_x)
        self.n_faces_y = np.prod(self.shape_faces_y)
        self.n_faces_z = np.prod(self.shape_faces_z)
        self.n_faces = self.n_faces_x + self.n_faces_y + self.n_faces_z

    def __repr__(self):
        """Simple representation."""
        return (f"TensorMesh: {self.shape_cells[0]} x {self.shape_cells[1]} x "
                f"{self.shape_cells[2]} ({self.n_cells:,})")

    @property
    def cell_volumes(self):
        """Construct cell volumes of the 3D model as 1D array."""
        if getattr(self, '_cell_volumes', None) is None:
            self._cell_volumes = (
                    self.h[0][None, None, :]*self.h[1][None, :, None] *
                    self.h[2][:, None, None]).ravel()
        return self._cell_volumes


@utils._known_class
class TensorMesh(discretize.TensorMesh if discretize else BaseMesh):
    """A slightly modified version of :class:`discretize.TensorMesh`.

    Adds a few custom attributes (``__eq__``, ``copy``, and ``{to;from}_dict``)
    to :class:`discretize.TensorMesh`.

    It falls back to a minimal :class:`emg3d.meshes.BaseMesh` if discretize is
    not installed. Nothing fancy is possible with the minimal TensorMesh,
    particularly *no* plotting.


    Parameters
    ----------
    h : [array_like, array_like, array_like]
        Cell widths in x, y, and z directions.

    origin : array_like
        Origin (x, y, z).

    """

    def __eq__(self, mesh):
        """Compare two meshes.

        The provided ``mesh`` can be either an emg3d or a discretize
        TensorMesh instance.

        """

        # Check if mesh is of the same instance.
        equal = mesh.__class__.__name__ == self.__class__.__name__

        # Check dimensions.
        if equal:
            equal *= len(mesh.shape_cells) == len(self.shape_cells)

        # Check shape.
        if equal:
            equal *= np.all(self.shape_cells == mesh.shape_cells)

        # Check distances and origin.
        if equal:
            equal *= np.allclose(self.h[0], mesh.h[0], atol=0)
            equal *= np.allclose(self.h[1], mesh.h[1], atol=0)
            equal *= np.allclose(self.h[2], mesh.h[2], atol=0)
            equal *= np.allclose(self.origin, mesh.origin, atol=0)

        return bool(equal)

    def copy(self):
        """Return a copy of the TensorMesh."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information in a dict for serialization.

        Parameters
        ----------
        copy : bool, default: False
            If True, returns a deep copy of the dict.


        Returns
        -------
        out : dict
            Dictionary containing all information to re-create the TensorMesh.

        """
        out = {
            'hx': self.h[0],
            'hy': self.h[1],
            'hz': self.h[2],
            'origin': self.origin,
            '__class__': self.__class__.__name__
        }
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`emg3d.meshes.TensorMesh` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from
            :func:`emg3d.meshes.TensorMesh.to_dict`. The dictionary needs the
            keys ``hx``, ``hy``, ``hz``, and ``origin``.

        Returns
        -------
        mesh : TensorMesh
            A :class:`emg3d.meshes.TensorMesh` instance.

        """
        inp = {k: v for k, v in inp.items() if k != '__class__'}
        return cls(h=[inp.pop('hx'), inp.pop('hy'), inp.pop('hz')], **inp)


def construct_mesh(frequency, properties, center, domain=None, vector=None,
                   seasurface=None, **kwargs):
    r"""Return a TensorMesh for given parameters.

    Designing an appropriate grid is the most time-consuming part of any 3D
    modelling:

    - Cell sizes should be small enough to represent changes in the model as
      well as to minimize interpolation errors in the fields.
    - The computational domain has to be big enough to avoid effects from the
      boundary condition.
    - The total number of cells should be small to speed up computation.

    These are in itself contradictory requirements, and they additionally
    depend all on the subsurface properties, the frequency under consideration,
    and the survey type.

    This function is a helper routine to construct an appropriate grid.
    However, there is no guarantee that it is the best or even a good grid. The
    constructed grid is frequency- and property-dependent. Some details are
    explained in other functions:

    - The minimum cell width :math:`\Delta_\text{min}` is a function of
      ``frequency``, ``properties[0]``, ``min_width_pps``, and
      ``min_width_limits``, see Equation :eq:`mincellwidth`.
    - The skin depth :math:`\delta` is a function of ``frequency`` and
      ``properties``, see Equation :eq:`skindepth`.
    - The wavelength :math:`\lambda` is a function of ``frequency`` and
      ``properties``, see Equation :eq:`wavelength`.

    The relation of the survey domain, computational domain, and buffer zone is
    shown in  :numref:`Figure %s <AutoGrid>` for a x-z-section; the y-direction
    behaves the same as the x-direction (the figures are only visible in the
    web version of the API docs on https://emg3d.emsig.xyz).

    .. figure:: ../_static/construct_mesh.png
       :align: center
       :alt: Sketch for automatic gridding.
       :name: AutoGrid

       Relation between survey domain (Ds, ``domain``), computational domain
       (Dc), and buffer zone. The survey domain should contain all sources (red
       stars) and receivers (blue triangles) as well as any important feature
       that should be represented in the data. The buffer zone is calculated as
       a function of wavelength with the provided property in the given
       direction.

    The buffer zone around the survey domain is by default one wavelength. This
    means that the signal has to travel two wavelengths to get from the end of
    the survey domain to the end of the computational domain and back. This
    approach is quite conservative and on the safe side. You can reduce the
    buffer thickness if you know what you are doing. There are three parameters
    which influence the thickness of the buffer for a given frequency:
    ``properties``, which is used to calculate the skin depth and the
    wavelength, ``lambda_factor`` (default is 1) which sets how many times the
    wavelength is the thickness of the buffer (relative factor), and
    ``max_buffer``, which is an absolute maximum for the buffer thickness. A
    graphical illustration is given in :numref:`Figure %s <Buffer>`.

    .. figure:: ../_static/construct_mesh2.png
       :align: center
       :alt: Sketch for the buffer zone.
       :name: Buffer

       The thickness of the buffer zone (B) for (I)
       ``lambda_from_center=False`` (default) and for (II)
       ``lambda_from_center=True``. The ``lambda_factor``
       (:math:`\lambda_{fact}`) is a simple scaling factor for the wavelength
       :math:`\lambda`. The ``max_buffer`` is an absolute limitation.


    Parameters
    ----------

    frequency : float
        Frequency (Hz) to calculate skin depth; both the minimum cell width and
        the extent of the buffer zone, and therefore of the computational
        domain, are a function of skin depth.

    properties : {float, array_like}
        Properties to calculate the skin depths, which in turn are used to
        calculate the minimum cell width (usually at source location) and the
        extent of the buffer around the survey domain. The properties can be
        either resistivities, conductivities, or the logarithm (natural or base
        10) thereof. By default it assumes resistivities, but it can be changed
        with the parameter ``mapping``.

        Five formats are recognized for properties: it can either be a float or
        a list of 2, 3, 4, or 7 floats. Depending on the format these
        properties are used to calculate the following parameters:

        - ``p``: min_width and buffer in all directions;

        - ``[p1, p2]``:

          - ``p1`` : min_width,
          - ``p2`` : buffer in all directions;

        - ``[p1, p2, p3]``:

          - ``p1`` : min_width,
          - ``p2`` : buffer in negative z-direction,
          - ``p3`` : buffer in all other directions;

        - ``[p1, p2, p3, p4]``:

          - ``p1`` : min_width,
          - ``p2`` : buffer in horizontal directions,
          - ``p3``; ``p4`` : buffer in negative; positive z-direction.

        - ``[p1, p2, p3, p4, p5, p6, p7]``:

          - ``p1`` : min_width,
          - ``p2``; ``p3`` : buffer in negative; positive x-direction,
          - ``p4``; ``p5`` : buffer in negative; positive y-direction,
          - ``p6``; ``p7`` : buffer in negative; positive z-direction.

    center : array_like
        Center coordinates (x, y, z). The mesh is centered around this point,
        which means that here is the smallest cell. Usually this is the source
        location.

    domain : {tuple, list, dict, None}, optional
        Contains the survey-domain limits. This domain should include all
        source and receiver positions as well as any important feature of the
        model. Format: ``([xmin, xmax], [ymin, ymax], [zmin, zmax])`` or
        ``{'x': [xmin, xmax], 'y': [ymin, ymax], 'z': [zmin, zmax]}``.

        It can be None, or individual lists can be None (e.g., ``(None, None,
        [zmin, zmax])``), in which case you have to provide either the
        corresponding ``distance`` or ``vector``, which is then assumed to span
        exactly the domain. If only one list is provided it is applied to all
        dimensions.

    distance : {tuple, list, dict, None}, optional
        An alternative to ``domain``: Instead of defining the domain in
        absolute values, they are defined here as distance from the center.
        Format:
        ``([xl, xr], [yl, yr], [zd, zu])`` or
        ``{'x': [xl, xr], 'y': [yl, yr], 'z': [zd, zu]}``.
        From this the domain is given as
        ``([cx-xl, cx+xr], [cy-yl, cy+yr], [cz-zd, cz+zu])``,
        where ``center=(cx, cy, cz)``.

    vector : {tuple, ndarray, dict, None}, optional
        Contains vectors of mesh-edges that should be used. If provided, the
        vector *must* at least include all of the survey domain. If ``domain``
        is not provided, it is defined as the minimum/maximum of the provided
        vector. Format: ``(xvector, yvector, zvector)`` or ``{'x': xvector,
        'y': yvector, 'z': zvector}``.

        It can be None, or individual ndarrays can be None (e.g., ``(xvector,
        yvector, None)``), in which case you have to provide a ``domain`` or
        ``distance``. If only one ndarray is provided it is applied to all
        dimensions.

    seasurface : float, default: None
        Air-sea interface. This has only to be set in the marine case, when the
        mesh in z-direction is sought for (and the interface is not contained
        in ``vector``). If set, it will ensure that at the sea surface is an
        actual boundary. It has to be bigger than the lower limit of the survey
        domain.

    stretching : {tuple, list, dict}, default: [1.0, 1.5]
        Maximum stretching factors in the form of ``[max Ds, max Dc]``: the
        first value is the maximum stretching for the survey domain (default is
        1.0), the second value is the maximum stretching for the buffer zone
        (default is 1.5). If a list is provided the same is used for all three
        dimension. Alternatively a tuple of three lists can be provided, ``(x,
        y, z)`` or ``{'x': x, 'y': y, 'z': z}``. Note that the first value has
        no influence on dimensions where a ``vector`` is provided.

    min_width_limits : {float, list, tuple, dict, None}, default: None
        Passed through to :func:`cell_width` as ``limits``. A tuple of three
        or a dict with ``x;y;z`` can be provided for direction dependent
        values. Note that this value has no influence on dimensions where a
        ``vector`` is provided.

    min_width_pps : {float, tuple, dict}, default: 3.0
        Passed through to :func:`cell_width` as ``pps``. A tuple of three or a
        dict with ``x;y;z`` can be provided for direction dependent values.
        Note that this value has no influence on dimensions where a ``vector``
        is provided.

    lambda_factor : float, default: 1.0
        The buffer is taken as one wavelength from the survey domain. This can
        be regarded as quite conservative (but safe). The parameter
        ``lambda_factor`` can be used to reduce (or increase) this factor.

    max_buffer : float, default: 100_000
        Maximum thickness of the buffer zone around survey domain. If
        ``lambda_from_center=True``, this is the maximum distance from the
        center to the end of the computational domain.

    lambda_from_center : bool, default: False
        Flag how to compute the extent of the computational mesh as a function
        of wavelength:

        - False (default): The distance from the edge of the survey domain to
          the edge of the computational domain is one wavelength.
        - True: The distance from the center to the edge of the computational
          domain and back to the end of the survey domain is two wavelengths.

    mapping : {str, Map}, default: 'Resistivity'
        Defines what type the input ``property_{x;y;z}``-values correspond to.
        By default, they represent resistivities (Ohm.m). The implemented
        mappings are:

        - ``'Resistivity'``; ρ (Ω m);
        - ``'Conductivity'``; σ (S/m);
        - ``'LgResistivity'``; log_10(ρ);
        - ``'LgConductivity'``; log_10(σ);
        - ``'LnResistivity'``; log_e(ρ);
        - ``'LnConductivity'``; log_e(σ).

    cell_numbers : array_like, optional
        List of possible numbers of cells. See :func:`good_mg_cell_nr`.
        Default is ``good_mg_cell_nr(1024, 5, 3)``, which corresponds to
        numbers 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384,
        512, 640, 768, 1024.

    verb : int, default: 0
        If 1 verbose, if 0 silent. The info is added either way to the returned
        mesh as ``mesh.construct_mesh_info``.


    Returns
    -------
    grid : TensorMesh
        Resulting mesh, a :class:`emg3d.meshes.TensorMesh` instance.

    """
    kwargs = deepcopy(kwargs)  # To not change a provided dict.
    verb = kwargs.get('verb', 0)
    distance = kwargs.pop('distance', None)

    # Initiate direction-specific dicts, add unambiguous args.
    kwargs['frequency'] = frequency
    kwargs['verb'] = -1            # Run x/y/z first, print/collect
    kwargs['raise_error'] = False  # info, then raise if necessary.
    xparams = {'center': center[0]}
    yparams = {'center': center[1]}
    zparams = {'center': center[2], 'seasurface': seasurface}

    # Add properties.
    if isinstance(properties, (int, float)):
        properties = np.array([properties])
    if len(properties) == 3:
        xparams['properties'] = [properties[0], properties[2], properties[2]]
        yparams['properties'] = [properties[0], properties[2], properties[2]]
        zparams['properties'] = [properties[0], properties[1], properties[2]]
    elif len(properties) == 4:
        xparams['properties'] = [properties[0], properties[1], properties[1]]
        yparams['properties'] = [properties[0], properties[1], properties[1]]
        zparams['properties'] = [properties[0], properties[2], properties[3]]
    elif len(properties) == 7:
        xparams['properties'] = [properties[0], properties[1], properties[2]]
        yparams['properties'] = [properties[0], properties[3], properties[4]]
        zparams['properties'] = [properties[0], properties[5], properties[6]]
    else:
        kwargs['properties'] = properties

    def _put_in_dicts(dicts, value, name):
        """Loop over dicts and put corresponding values."""
        for i, data in enumerate(dicts):
            if value[i] is not None:
                data[name] = value[i]

    # Add optionally direction specific args.
    for name, value in zip(['domain', 'vector', 'distance'],
                           [domain, vector, distance]):
        if value is None or isinstance(value, np.ndarray):
            kwargs[name] = value
        elif isinstance(value, dict):
            _put_in_dicts([xparams, yparams, zparams],
                          (value['x'], value['y'], value['z']), name)
        elif len(value) == 3:
            _put_in_dicts([xparams, yparams, zparams], value, name)
        else:
            kwargs[name] = value

    # Add optionally direction specific kwargs.
    for name in ['stretching', 'min_width_limits', 'min_width_pps']:
        value = kwargs.pop(name, None)
        if value is not None:
            if isinstance(value, (int, float)):
                kwargs[name] = np.array([value])
            elif isinstance(value, dict):
                _put_in_dicts([xparams, yparams, zparams],
                              (value['x'], value['y'], value['z']), name)
            elif len(value) == 3:
                _put_in_dicts([xparams, yparams, zparams], value, name)
            else:
                kwargs[name] = value

    # Get origins and widths in all directions.
    x0, hx, xinfo = origin_and_widths(**kwargs, **xparams)
    y0, hy, yinfo = origin_and_widths(**kwargs, **yparams)
    z0, hz, zinfo = origin_and_widths(**kwargs, **zparams)

    # Throw message if no solution was found.
    if any([out is None for out in [x0, y0, z0]]):
        raise RuntimeError("No suitable grid found; relax your criteria.")

    # Create mesh.
    mesh = TensorMesh(h=[hx, hy, hz], origin=np.array([x0, y0, z0]))

    # Collect info.
    info = (f"\n         == GRIDDING IN X ==\n{xinfo}\n"
            f"\n         == GRIDDING IN Y ==\n{yinfo}\n"
            f"\n         == GRIDDING IN Z ==\n{zinfo}\n")
    mesh.construct_mesh_info = info
    if verb > 0:
        print(info)

    return mesh


def origin_and_widths(frequency, properties, center, domain=None, vector=None,
                      seasurface=None, **kwargs):
    r"""Return origin and cell widths for given parameters.

    Calculate and return the starting point (origin) and cell widths given
    the input parameters. The output can be used as input (for one dimension)
    to create a :class:`emg3d.meshes.TensorMesh`. The function is used by
    :func:`construct_mesh` once in each direction.

    .. note::

        The parameters are described in :func:`construct_mesh`. Described here
        are only the differences. It is recommended to use directly
        :func:`construct_mesh`.


    Parameters
    ----------

    verb : int, default: 0
        If 1 verbose, if 0 silent, if -1 it returns the info as string instead
        of printing it.

    raise_error : bool, default: True
        If True, an error is raised if no suitable grid is found. Otherwise it
        just returns None's (used by ``construct_mesh``).

    Returns
    -------
    origin : float
        Origin of the mesh.

    widths : ndarray
        Cell widths of mesh.

    info : str, returned if verb<0
        Info.

    """
    # Get all kwargs.
    distance = kwargs.pop('distance', None)
    stretching = kwargs.pop('stretching', [1.0, 1.5])
    min_width_limits = kwargs.pop('min_width_limits', None)
    min_width_pps = kwargs.pop('min_width_pps', 3)
    lambda_factor = kwargs.pop('lambda_factor', 1.0)
    max_buffer = kwargs.pop('max_buffer', 100000)
    lambda_from_center = kwargs.pop('lambda_from_center', False)
    pmap = kwargs.pop('mapping', 'Resistivity')
    cell_numbers = kwargs.pop('cell_numbers', good_mg_cell_nr())
    raise_error = kwargs.pop('raise_error', True)
    verb = kwargs.pop('verb', 0)

    # Ensure no kwargs left.
    if kwargs:
        raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

    # Get property map from string.
    if isinstance(pmap, str):
        pmap = getattr(maps, 'Map'+pmap)()

    # Properties.
    cond = pmap.backward(np.array(properties, ndmin=1, dtype=float))
    cond_arr = np.array([
        cond[0], cond[min(cond.size-1, 1)], cond[min(cond.size-1, 2)]])

    # Get skin depth.
    skind = skin_depth(frequency, cond_arr)

    # Minimum cell width.
    dmin = cell_width(skind[0], min_width_pps, min_width_limits)

    # Survey domain: if not provided get from vector or distance.
    # Priority: domain > vector > distance.
    if domain is None and vector is None and distance is None:
        raise ValueError(
            "At least one of `domain`, `distance`, "
            "and `vector` must be provided."
        )
    elif domain is None:
        if vector is None:
            domain = np.array([center-abs(distance[0]),
                               center+abs(distance[1])])
        else:
            domain = np.array([vector.min(), vector.max()], dtype=float)
    else:
        domain = np.array(domain, dtype=np.float64)
        if vector is not None:
            if domain[0] < vector.min() or domain[1] > vector.max():
                raise ValueError(
                    "Provided vector MUST at least include "
                    "all of the survey domain."
                )

    # Seasurface related checks.
    if seasurface is not None:

        # Check that seasurface > center.
        if seasurface <= center:
            raise ValueError("The `seasurface` must be bigger than `center`.")

        # If center is close to seasurface, set it to seasurface.
        if abs(seasurface - center) < dmin:
            center = seasurface

    # Computation domain; big enough to avoid boundary effects.
    # To avoid boundary effects we want the signal to travel two wavelengths
    # from the source to the boundary and back to the receiver.
    # => 2*pi*sd ~ 6.3*sd = one wavelength => signal is ~ 0.2 %.
    # Two wavelengths we can safely assume it is zero.
    wlength = lambda_factor*wavelength(skind[1:])

    if lambda_from_center:
        # Center to edges of domain.
        in_domain = abs(domain - center)

        # Required buffer, additional to domain.
        d_buff = np.max([np.zeros(2), (2*wlength - in_domain)/2], axis=0)

        # Add buffer to domain.
        comp_domain = np.array([domain[0]-d_buff[0], domain[1]+d_buff[1]])

        # Restrict total domain to max_buffer.
        comp_domain[0] = max(comp_domain[0], center-max_buffer)
        comp_domain[1] = min(comp_domain[1], center+max_buffer)

    else:
        dbuffer = np.min([wlength, np.ones(2)*max_buffer], axis=0)
        comp_domain = np.array([domain[0]-dbuffer[0], domain[1]+dbuffer[1]])

    # Initiate flag if terminated.
    finished = False

    # Initiate alpha variables for survey and computation domains.
    sa, ca = 1.0, 1.0

    # Loop over possible cell numbers from small to big.
    for nx in np.unique(cell_numbers):

        # Loop over possible alphas for domain.
        nsa = max(1, min(100, int((stretching[0] - 1) / 0.001)))
        for sa in np.linspace(1.0, stretching[0], nsa):

            if vector is None:

                # Get current stretched grid cell sizes.
                thxl = dmin*sa**np.arange(nx)  # Left of origin.
                thxr = dmin*sa**np.arange(nx)  # Right of origin.

                # Adjust stretching for seasurface if required.
                if seasurface is not None and seasurface > center:
                    t_nx = np.r_[center, center+np.cumsum(thxr)]
                    ii = np.argmin(abs(t_nx-seasurface))
                    thxr[:ii] *= abs(seasurface-center)/np.sum(thxr[:ii])

                # Fill from center to left and right domain.
                nl = np.sum((center-np.cumsum(thxl)) > domain[0]) + 1
                nr = np.sum((center+np.cumsum(thxr)) < domain[1]) + 1

                # Create the current hx-array.
                hx = np.r_[thxl[:nl][::-1], thxr[:nr]]

                # Get actual domain:
                asurv_domain = [center - np.sum(thxl[:nl]),
                                center + np.sum(thxr[:nr])]

            else:
                # Store actual domain, current hx-array, and number of cells.
                asurv_domain = [vector[0], vector[-1]]
                hx = np.diff(vector)

            # Expand for seasurface if necessary.
            if seasurface is not None and seasurface > asurv_domain[-1]:
                thxr = hx[-1]*sa**np.arange(nx)
                sdepth = seasurface - asurv_domain[-1]

                # Get number of element, round down, and stretch.
                ii = np.argmax(np.cumsum(thxr) > sdepth)
                thxr = thxr[:ii]  # Restrict.
                thxr *= abs(seasurface-asurv_domain[-1])/np.sum(thxr)

                # Adjust actual domain, hx, and count.
                asurv_domain[1] += sum(thxr)
                hx = np.r_[hx, thxr]

            # Remaining number of cells
            nx_remain = nx - hx.size

            # Not good, try next.
            if nx_remain <= 0:
                continue

            # Store for verbosity
            hxo = hx

            # Loop over possible alphas for buffer.
            nca = max(1, min(100, int((stretching[1] - sa) / 0.001)))
            for ca in np.linspace(sa, stretching[1], nca):

                # Get current stretched grid cell sizes.
                thxl = hx[0]*ca**np.arange(1, nx_remain+1)   # Left of survey.
                thxr = hx[-1]*ca**np.arange(1, nx_remain+1)  # Right of survey.

                # Fill from survey to left and right domain.
                nl = np.sum((asurv_domain[0] - np.cumsum(thxl)) >
                            comp_domain[0]) + 1
                nr = np.sum((asurv_domain[1] + np.cumsum(thxr)) <
                            comp_domain[1]) + 1

                # Get remaining number of cells.
                nx_remain2 = nx_remain - nl - nr

                if nx_remain2 < 0:  # Not good, try next.
                    continue

                # Create hx-array.
                nl += int(np.floor(nx_remain2/2))  # If uneven, add one cell
                nr += int(np.ceil(nx_remain2/2))   # more on the right.
                hx = np.r_[thxl[:nl][::-1], hx, thxr[:nr]]

                # Compute origin.
                x0 = float(asurv_domain[0]-np.sum(thxl[:nl]))

                # Mark it as finished and break out of the loop.
                finished = True
                break

            if finished:
                break

        if finished:
            break

    # Raise Error or return Nones if not finish.
    if not finished:
        msg = "No suitable grid found; relax your criteria."
        # Throw message if no solution was found.
        if raise_error:
            raise RuntimeError(msg)
        else:
            x0, hx, info = None, None, msg

    else:  # Collect info about final grid.

        # Check max stretching.
        sa_adj = np.max([hxo[1:]/hxo[:-1], hxo[:-1]/hxo[1:]])
        sa_limit = min(1.5, stretching[0]+0.25)

        # Display precision.
        prec = int(np.ceil(max(0, -np.log10(min(hx))+1)))

        info = f"Skin depth     [m] : {skind[0]:.{prec}f}"
        if cond.size > 1:
            info += f" / {skind[1]:.{prec}f}"
        if cond.size > 2:
            info += f" / {skind[2]:.{prec}f}"
        info += "  [corr. to `properties`]\n"

        info += (
            f"Survey dom. DS [m] : "
            f"{domain[0]:.{prec}f} - {domain[1]:.{prec}f}\n"
            #
            f"Comp. dom. DC  [m] : {comp_domain[0]:.{prec}f} - "
            f"{comp_domain[1]:.{prec}f}\n"
            #
            f"Final extent   [m] : {x0:.{prec}f} - {x0+np.sum(hx):.{prec}f}\n"
            #
            f"Cell widths    [m] : {min(hxo):.{prec}f} / {max(hxo):.{prec}f} "
            f"/ {max(hx):.{prec}f}  [min(DS) / max(DS) / max(DC)]\n"
            #
            f"Number of cells    : {nx} ({hxo.size} / "
            f"{nx-hxo.size-nx_remain2} / {nx_remain2})  "
            f"[Total (DS/DC/remain)]\n"
            #
            f"Max stretching     : {sa:.3f} ({sa_adj:.3f}) / {ca:.3f}"
            "  [DS (seasurface) / DC]")

        if sa_adj > sa_limit:
            info += (f"\nNote: Stretching in DS >> {sa}.\nThe reason "
                     "is usually the interplay of center/domain/seasurface.")

    if verb > 0:
        print(info)

    # Return required info.
    if verb < 0:
        return x0, hx, info
    else:
        return x0, hx


def good_mg_cell_nr(max_nr=1024, max_lowest=5, min_div=3):
    r"""Return "good" cell numbers for the multigrid method.

    "Good" cell numbers are numbers which can be divided by two as many times
    as possible. At the end there should be 2 or a small odd number.

    The function adds all numbers

    .. math::

        p\ 2^n &\leq M \ , \text{ for}

        p &= {2, 3, ..., p_\text{max}} \ ;

        n &= {n_\text{min}, n_\text{min}+1, ..., \infty} \ ,

    where :math:`M, p_\text{max}, n_\text{min}` correspond to ``max_nr``,
    ``max_lowest``, and ``min_div``, respectively.


    Parameters
    ----------
    max_nr : int, default: 1024
        Maximum number of cells.

    max_lowest : int, default: 5
        Maximum permitted lowest number p for p*2^n. {2, 3, 5, 7} are good
        upper limits in order to avoid too big lowest grids in the multigrid
        method.

    min_div : int, default: 3
        Minimum times the number can be divided by two.


    Returns
    -------
    numbers : ndarray
        Array containing all possible cell numbers from lowest to highest.

    """
    # 2 + odd numbers till 20.
    lowest = np.array([2, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=np.int64)

    # Sanity check; 19 is already ridiculously high.
    if max_lowest > lowest[-1]:
        raise ValueError(
            f"Maximum lowest is {max_lowest}, please use a value < 20."
        )

    # Restrict to max_lowest.
    lowest = lowest[lowest <= max_lowest]

    # Get possible values.
    # Currently restr. to lowest*2**29 (for lowest=2 => 1,073,741,824 cells).
    numbers = lowest[:, None]*2**np.arange(min_div, 30)

    # Get unique values.
    numbers = np.unique(numbers)

    # Restrict to max_nr and return.
    return numbers[numbers <= max_nr]


def skin_depth(frequency, conductivity, mu_r=1.0):
    r"""Return skin depth as a function of frequency and conductivity.

    The skin depth :math:`\delta` (m) is given by

    .. math::
        :label: skindepth

        \delta = \sqrt{\frac{2}{\omega\sigma\mu}}\ ,

    where :math:`\omega=2\pi f` is angular frequency of frequency :math:`f`
    (Hz), :math:`\sigma` is conductivity (S/m), and :math:`\mu=\mu_\rm{r}\mu_0`
    is magnetic permeability (H/m).


    Parameters
    ----------
    frequency : float
        Frequency (Hz).

    conductivity : float
        Conductivity (S/m).

    mu_r : float, default: 1.0
        Relative magnetic permeability (-).


    Returns
    -------
    skindepth : float
        Skin depth (m).

    """
    skindepth = 1/np.sqrt(np.pi*abs(frequency)*conductivity*mu_r*mu_0)

    # For Laplace-domain computations.
    if frequency < 0:
        skindepth /= np.sqrt(2*np.pi)

    return skindepth


def wavelength(skin_depth):
    r"""Return wavelength as a function of skin depth.

    The wavelength :math:`\lambda` (m) is given by

    .. math::
        :label: wavelength

        \lambda = 2\pi\delta\ ,

    where the skin depth :math:`\delta` is a function of frequency and
    conductivity and is given by :func:`skin_depth`, Equation :eq:`skindepth`.


    Parameters
    ----------
    skin_depth : {float, ndarray}
        Skin depth (m).


    Returns
    -------
    wavelength : {float, ndarray}
        Wavelength (m).

    """
    return 2*np.pi*skin_depth


def cell_width(skin_depth, pps=3, limits=None):
    r"""Return cell width as function of points per skin depth.

    The cell width :math:`\Delta` is defined by the desired points per skin
    depth,

    .. math::
        :label: mincellwidth

        \Delta = \Delta_\text{min} \le \frac{\delta}{\text{pps}} \le
                 \Delta_\text{max} \ ,

    to ensure that there are ``pps`` cells per skin depth (unless restricted by
    ``limits``).

    The skin depth :math:`\delta` is a function of frequency and conductivity
    and is given by :func:`skin_depth`, Equation :eq:`skindepth`.


    Parameters
    ----------
    skin_depth : float
        Skin depth (m).

    pps : float
        Points per skin depth.

    limits : {None, float, array_like}, default: None
        Limits on cell width:

        - ``None``: No limits.
        - ``float``: Simply returns the ``limits`` as cell width.
        - ``[min, max]``: Cell width is limited to this range.


    Returns
    -------
    cell_width : float
        Cell width (m).

    """
    # Calculate cell width.
    cell_width = skin_depth/pps

    # Respect user limits.
    if limits is not None:

        limits = np.array(limits, ndmin=1)

        if limits.size == 1:
            cell_width = limits  # Ignores skin depth and pps.

        else:
            cell_width = np.clip(cell_width, *limits)  # Restrict.

    return cell_width


def check_mesh(mesh):
    """Check provided mesh and throw a warning if it is not good for multigrid.

    Parameters
    ----------
    mesh : TensorMesh
        A :class:`emg3d.meshes.TensorMesh` instance.

    """

    # Extreme values.
    good = good_mg_cell_nr(max_nr=50000, max_lowest=5, min_div=0)

    # Ensure mesh is a TensorMesh.
    if not mesh.__class__.__name__ == 'TensorMesh':
        raise TypeError("Mesh must be a TensorMesh.")

    # Ensure it is a 3D mesh.
    if len(mesh.origin) != 3:
        raise TypeError('Mesh must be a 3D mesh.')

    # Check mesh dimensions, warn if not optimal.
    if any(n_cells not in good for n_cells in mesh.shape_cells):
        msg = (
            f"emg3d: Mesh dimension {(mesh.shape_cells)} is not optimal for "
            f"MG solver. Good numbers are:\n{good_mg_cell_nr(max_nr=5000)}"
        )
        warnings.warn(msg, UserWarning)


def estimate_gridding_opts(gridding_opts, model, survey, input_sc2=None):
    """Estimate parameters for automatic gridding.

    Automatically determines the required gridding options from the provided
    model, and survey, if they are not provided in ``gridding_opts``.

    The dict ``gridding_opts`` can contain any input parameter taken by
    :func:`emg3d.meshes.construct_mesh`, see the corresponding documentation
    for more details with regards to the possibilities.

    Different keys of ``gridding_opts`` are treated differently:

    - The following parameters are estimated from the ``model`` if not
      provided:

      - ``properties``: lowest conductivity / highest resistivity in the
        outermost layer in a given direction. This is usually air in x/y and
        positive z. Note: This is very conservative. If you go into deeper
        water you could provide less conservative values.
      - ``mapping``: taken from model.

    - The following parameters are estimated from the ``survey`` if not
      provided:

      - ``frequency``: average (on log10-scale) of all frequencies.
      - ``center``: center of all sources.
      - ``domain``: from ``vector`` or ``distance``, if provided, or

        - in x/y-directions: extent of sources and receivers plus 10% on each
          side, ensuring ratio of 3.
        - in z-direction: extent of sources and receivers, ensuring ratio of 2
          to horizontal dimension; 1/10 tenth up, 9/10 down.

        The ratio means that it is enforced that the survey dimension in x or
        y-direction is not smaller than a third of the survey dimension in the
        other direction. If not, the smaller dimension is expanded
        symmetrically. Similarly in the vertical direction, which must be at
        least half the dimension of the maximum horizontal dimension or 5 km,
        whatever is smaller. Otherwise it is expanded in a ratio of 9 parts
        downwards, one part upwards.

    - The following parameter is taken from the ``grid`` if provided as a
      string:

      - ``vector``: This is the only real "difference" to the inputs of
        :func:`emg3d.meshes.construct_mesh`. The normal input is accepted, but
        it can also be a string containing any combination of ``'x'``, ``'y'``,
        and ``'z'``. All directions contained in this string are then taken
        from the provided grid. E.g., if ``gridding_opts['vector']='xz'`` it
        will take the x- and z-directed vectors from the grid.

    - The following parameters are simply passed along if they are provided,
      nothing is done otherwise:

      - ``vector``
      - ``distance``
      - ``stretching``
      - ``seasurface``
      - ``cell_numbers``
      - ``lambda_factor``
      - ``lambda_from_center``
      - ``max_buffer``
      - ``min_width_limits``
      - ``min_width_pps``
      - ``verb``


    Parameters
    ----------
    gridding_opts : dict
        Containing input parameters to provide to
        :func:`emg3d.meshes.construct_mesh`. See the corresponding
        documentation and the explanations above.

    model : Model
        The model; a :class:`emg3d.models.Model` instance.

    survey : Survey
        The survey; a :class:`emg3d.surveys.Survey` instance.

    input_sc2 : int, default: None
        If :func:`emg3d.models.expand_grid_model` was used, ``input_sc2``
        corresponds to the original ``grid.shape_cells[2]``.


    Returns
    -------
    gridding_opts : dict
        Dict to provide to :func:`emg3d.meshes.construct_mesh`.

    """
    # Initiate new gridding_opts.
    gopts = {}
    grid = model.grid

    # Optional values that we only include if provided.
    for name in ['seasurface', 'cell_numbers', 'lambda_factor',
                 'lambda_from_center', 'max_buffer', 'verb']:
        if name in gridding_opts.keys():
            gopts[name] = gridding_opts.pop(name)
    for name in ['stretching', 'min_width_limits', 'min_width_pps']:
        if name in gridding_opts.keys():
            value = gridding_opts.pop(name)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                value = {'x': value[0], 'y': value[1], 'z': value[2]}
            gopts[name] = value

    # Mapping defaults to model map.
    gopts['mapping'] = gridding_opts.pop('mapping', model.map)
    if not isinstance(gopts['mapping'], str):
        gopts['mapping'] = gopts['mapping'].name

    # Frequency defaults to average frequency (log10).
    frequency = 10**np.mean(np.log10([v for v in survey.frequencies.values()]))
    gopts['frequency'] = gridding_opts.pop('frequency', frequency)

    # Center defaults to center of all sources.
    center = np.array([s.center for s in survey.sources.values()]).mean(0)
    gopts['center'] = gridding_opts.pop('center', center)

    # Vector.
    vector = gridding_opts.pop('vector', None)
    if isinstance(vector, str):
        # If vector is a string we take the corresponding vectors from grid.
        vector = (
                grid.nodes_x if 'x' in vector.lower() else None,
                grid.nodes_y if 'y' in vector.lower() else None,
                grid.nodes_z[:input_sc2] if 'z' in vector.lower() else None,
        )
    gopts['vector'] = vector
    if isinstance(vector, dict):
        vector = (vector['x'], vector['y'], vector['z'])
    elif vector is not None and len(vector) == 3:
        gopts['vector'] = {'x': vector[0], 'y': vector[1], 'z': vector[2]}

    # Distance.
    distance = gridding_opts.pop('distance', None)
    gopts['distance'] = distance
    if isinstance(distance, dict):
        distance = (distance['x'], distance['y'], distance['z'])
    elif distance is not None and len(distance) == 3:
        gopts['distance'] = {'x': distance[0], 'y': distance[1],
                             'z': distance[2]}

    # Properties defaults to lowest conductivities (AFTER model expansion).
    properties = gridding_opts.pop('properties', None)
    if properties is None:

        # Get map (in principle the map in gridding_opts could be different
        # from the map in the model).
        m = gopts['mapping']
        if isinstance(m, str):
            m = getattr(maps, 'Map'+m)()

        # Minimum conductivity of all values (x, y, z).
        def get_min(ix, iy, iz):
            """Get minimum: very conservative/costly, but avoiding problems."""

            # Collect all x (y, z) values.
            data = np.array([])
            for p in ['x', 'y', 'z']:
                prop = getattr(model, 'property_'+p)
                if prop is not None:
                    prop = model.map.backward(prop[ix, iy, iz])
                    data = np.r_[data, np.min(prop)]

            # Return minimum conductivity (on mapping).
            return m.forward(min(data))

        # Buffer properties.
        xneg = get_min(0, slice(None), slice(None))
        xpos = get_min(-1, slice(None), slice(None))
        yneg = get_min(slice(None), 0, slice(None))
        ypos = get_min(slice(None), -1, slice(None))
        zneg = get_min(slice(None), slice(None), 0)
        zpos = get_min(slice(None), slice(None), -1)

        # Source property.
        ix = np.argmin(abs(grid.nodes_x - gopts['center'][0]))
        iy = np.argmin(abs(grid.nodes_y - gopts['center'][1]))
        iz = np.argmin(abs(grid.nodes_z - gopts['center'][2]))
        source = get_min(ix, iy, iz)

        properties = [source, xneg, xpos, yneg, ypos, zneg, zpos]

    gopts['properties'] = properties

    # Domain; default taken from survey.
    domain = gridding_opts.pop('domain', None)
    if isinstance(domain, dict):
        domain = (domain['x'], domain['y'], domain['z'])

    def get_dim_diff(i):
        """Return ([min, max], dim) of inp.

        Take it from domain if provided, else from vector if provided, else
        from survey, adding 10% on each side).
        """
        if domain is not None and domain[i] is not None:
            # domain is provided.
            dim = domain[i]
            diff = np.diff(dim)[0]
            get_it = False

        elif vector is not None and vector[i] is not None:
            # vector is provided.
            dim = [np.min(vector[i]), np.max(vector[i])]
            diff = np.diff(dim)[0]
            get_it = False

        elif distance is not None and distance[i] is not None:
            # distance is provided.
            dim = None
            diff = abs(distance[i][0]) + abs(distance[i][1])
            get_it = False

        else:
            # Get it from survey, add 5 % on each side.
            inp = np.array([s.center[i] for s in survey.sources.values()])
            for s in survey.sources.values():
                inp = np.r_[inp, [r.center_abs(s)[i]
                                  for r in survey.receivers.values()]]
            dim = [min(inp), max(inp)]
            diff = np.diff(dim)[0]
            dim = [min(inp)-diff/10, max(inp)+diff/10]
            diff = np.diff(dim)[0]
            get_it = True

        diff = np.where(diff > 1e-9, diff, 1e-9)  # Avoid division by 0 later
        return dim, diff, get_it

    xdim, xdiff, get_x = get_dim_diff(0)
    ydim, ydiff, get_y = get_dim_diff(1)
    zdim, zdiff, get_z = get_dim_diff(2)

    # Ensure the ratio xdim:ydim is at most 3.
    if get_y and xdiff/ydiff > 3:
        diff = round((xdiff/3.0 - ydiff)/2.0)
        ydim = [ydim[0]-diff, ydim[1]+diff]
    elif get_x and ydiff/xdiff > 3:
        diff = round((ydiff/3.0 - xdiff)/2.0)
        xdim = [xdim[0]-diff, xdim[1]+diff]

    # Ensure the ratio zdim:horizontal is at most 2.
    hdist = min(10000, max(xdiff, ydiff))
    if get_z and hdist/zdiff > 2:
        diff = round((hdist/2.0 - zdiff)/10.0)
        zdim = [zdim[0]-9*diff, zdim[1]+diff]

    # Collect
    gopts['domain'] = {'x': xdim, 'y': ydim, 'z': zdim}

    # Ensure no gridding_opts left.
    if gridding_opts:
        raise TypeError(
            f"Unexpected gridding_opts: {list(gridding_opts.keys())}."
        )

    # Return gridding_opts.
    return gopts
