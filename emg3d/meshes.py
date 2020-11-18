"""
Everything related to meshes appropriate for the multigrid solver.
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

import warnings
from copy import deepcopy

import numpy as np
from scipy import optimize
from scipy.constants import mu_0

from emg3d import maps

try:
    import discretize.TensorMesh as dTensorMesh
except ImportError:
    class dTensorMesh:
        pass

__all__ = ['TensorMesh', 'construct_mesh', 'get_origin_widths', 'skin_depth',
           'wavelength', 'good_mg_cell_nr', 'min_cell_width',
           # Deprecated ones:
           'get_hx_h0', 'get_cell_numbers', 'get_stretched_h', 'get_domain',
           'get_hx']

MESHWARNING = (
    "\n    `get_hx_h0`, `get_stretched_h`, `get_domain`, and `get_hx` are"
    "\n    deprecated and will be removed. Use `construct_mesh`` instead."
)
CELLWARNING = (
    "\n    `get_cell_numbers` is deprecated and will be removed."
    "\n    Use `good_mg_cell_nr` instead."
)


class _TensorMesh:
    """Minimal TensorMesh for internal multigrid computation.


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
        self.hx = np.array(h[0])
        self.hy = np.array(h[1])
        self.hz = np.array(h[2])

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
        self.vnN = np.array([self.nNx, self.nNy, self.nNz], dtype=np.int_)
        self.nN = int(self.vnN.prod())
        self.vectorNx = np.r_[0., self.hx.cumsum()] + self.x0[0]
        self.vectorNy = np.r_[0., self.hy.cumsum()] + self.x0[1]
        self.vectorNz = np.r_[0., self.hz.cumsum()] + self.x0[2]

        # Edge related properties.
        self.vnEx = np.array([self.nCx, self.nNy, self.nNz], dtype=np.int_)
        self.vnEy = np.array([self.nNx, self.nCy, self.nNz], dtype=np.int_)
        self.vnEz = np.array([self.nNx, self.nNy, self.nCz], dtype=np.int_)
        self.nEx = int(self.vnEx.prod())
        self.nEy = int(self.vnEy.prod())
        self.nEz = int(self.vnEz.prod())
        self.vnE = np.array([self.nEx, self.nEy, self.nEz], dtype=int)
        self.nE = int(self.vnE.sum())

    def __repr__(self):
        """Simple representation."""
        return (f"TensorMesh: {self.nCx} x {self.nCy} x {self.nCz} "
                f"({self.nC:,})")

    @property
    def vol(self):
        """Construct cell volumes of the 3D model as 1D array."""
        if getattr(self, '_vol', None) is None:
            self._vol = (self.hx[None, None, :]*self.hy[None, :, None] *
                         self.hz[:, None, None]).ravel()
        return self._vol


class TensorMesh(dTensorMesh, _TensorMesh):
    """A slightly modified :class:`discretize.TensorMesh`.

    Adds a few attributes (`__eq__`, `copy`, `to_dict`, and `from_dict`) to the
    :class:`discretize.TensorMesh`.

    Falls back to a minimal TensorMesh if `discretize` is not installed.
    Nothing fancy is possible with the minimal TensorMesh, particularly NO
    plotting nor nice repr-functions.


    Parameters
    ----------
    h : list of three ndarrays
        Cell widths in [x, y, z] directions.

    x0 : ndarray of dimension (3, )
        Origin (x, y, z).

    """

    def __init__(self, h, x0):
        """Initiate TensorMesh."""
        # Cast `h` to list, as `discretize.TensorMesh`
        # fails if `h` is an # ndarray.
        super().__init__(h=list(h), x0=x0)

    def __eq__(self, mesh):
        """Compare two meshes.

        The provided `mesh` can be either a `emg3d` or a `discretize`
        TensorMesh.

        """

        # Check if mesh is of the same instance.
        equal = mesh.__class__.__name__ == self.__class__.__name__

        # Check dimensions.
        if equal:
            equal *= mesh.vnC.size == self.vnC.size

        # Check shape.
        if equal:
            equal *= np.all(self.vnC == mesh.vnC)

        # Check distances and origin.
        if equal:
            equal *= np.allclose(self.hx, mesh.hx, atol=0)
            equal *= np.allclose(self.hy, mesh.hy, atol=0)
            equal *= np.allclose(self.hz, mesh.hz, atol=0)
            equal *= np.allclose(self.x0, mesh.x0, atol=0)

        return bool(equal)

    def copy(self):
        """Return a copy of the TensorMesh."""
        return TensorMesh.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the TensorMesh in a dict."""
        out = {'hx': self.hx, 'hy': self.hy, 'hz': self.hz, 'x0': self.x0,
               '__class__': self.__class__.__name__}
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`TensorMesh` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`TensorMesh.to_dict`.
            The dictionary needs the keys `hx`, `hy`, `hz`, and `x0`.

        Returns
        -------
        obj : :class:`TensorMesh` instance

        """
        try:
            return cls(h=[inp['hx'], inp['hy'], inp['hz']], x0=inp['x0'])
        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.") from e


def construct_mesh(frequency, properties, center, domain=None, vector=None,
                   seasurface=None, **kwargs):
    r"""Return a TensorMesh for given parameters.

    The constructed mesh is frequency- and conductivity-dependent, where
    ``properties`` are turned into conductivities through the provided
    ``mapping``, which is ``'Resistivity'`` by default. Some details are
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
    web version on https://emg3d.rtfd.io).

    .. figure:: ../_static/construct_mesh.png
       :align: center
       :alt: Sketch for automatic gridding.
       :name: AutoGrid

       Relation between survey domain (Ds, ``domain``), computational domain
       (Dc), and buffer zone. The survey domain should contain all sources and
       receivers as well as any important feature that should be represented in
       the data. The buffer zone is calculated as a function of wavelength with
       the provided property in the given direction.

    By default, the buffer zone around the survey domain is one wavelength.
    This means that the signal has to travel two wavelengths to get from the
    end of the survey domain to the end of the computational domain and back.
    This approach is quite conservative and on the safe side. You can reduce
    the buffer thickness if you know what you are doing. There are three
    parameters which influence the thickness of the buffer for a given
    frequency: ``properties``, which is used to calculate the skin depth and
    the wavelength, ``lambda_factor`` (default is 1) which sets how many times
    the wavelength is the thickness of the buffer (relative factor), and
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

    properties : float or list
        Properties to calculate the skin depths. The properties can be either
        resistivities, conductivities, or the logarithm (natural or base 10)
        thereof. By default it assumes resistivities, but it can be changed
        with the parameter ``mapping``.

        Four formats are recognized:

        - 1: Same property for everything;
        - 2: [min_width, buffer (+/-)] for all directions;
        - 4: [min_width, xy-buffer (+/-), z-, z+];
        - 7: [min_width, x-, x+, y-, y+, z-, z+].

        The property ``min_width`` is usually the property at the center,
        hence at the source location. The other properties are used to define
        the extent of the buffer zone around the survey domain in the
        respective directions.

    center : tuple
        Tuple (or list, ndarray) of three floats for (x, y, z). The mesh is
        centered around this point, which means that here is the smallest cell.
        Usually this is the source location.

    domain : tuple of lists, list, or None, optional
        Contains the survey-domain limits. This domain should include all
        source and receiver positions as well as any important feature of the
        model. Format: ``([xmin, xmax], [ymin, ymax], [zmin, zmax])``.

        It can be None, or individual lists can be None (e.g., ``(None, None,
        [zmin, zmax])``), in which case you have to provide a ``vector``, which
        is then assumed to span exactly the domain. If only one list is
        provided it is applied to all dimensions.

    vector : tuple of three ndarrays, ndarray, or None, optional
        Contains vectors of mesh-edges that should be used. If provided, the
        vector MUST at least include all of the survey domain. If ``domain``
        is not provided, it is defined as the minimum/maximum of the provided
        vector. Format: ``(xvector, yvector, zvector)``.

        It can be None, or individual ndarrays can be None (e.g., ``(xvector,
        yvector, None)``), in which case you have to provide a ``domain``. If
        only one ndarray is provided it is applied to all dimensions.

    seasurface : float, optional
        Air-sea interface. This has only to be set in the marine case, when the
        mesh in z-direction is sought for (and the interface is not contained
        in ``vector``). If set, it will ensure that at the sea surface is an
        actual boundary. It has to be bigger then the lower limit of the survey
        domain.
        Default is None.

    stretching : list or tuple of lists, optional
        Maximum stretching factors in the form of ``[max Ds, max Dc]``: the
        first value is the maximum stretching for the survey domain (default is
        1.0), the second value is the maximum stretching for the buffer zone
        (default is 1.5). If a list is provided the same is used for all three
        dimension. Alternatively a tuple of three lists can be provided, ``(x,
        y, z)``. Note that the first value has no influence on dimensions where
        a ``vector`` is provided.

    min_width_limits : float, list or None, optional
        Passed through :func:`min_cell_width` as ``limits``.
        A tuple of three can be provided for direction dependent values.
        Note that this value has no influence on dimensions where a ``vector``
        is provided.

        Default is None.

    min_width_pps : float or int, optional
        Passed through :func:`min_cell_width` as ``pps``.
        A tuple of three can be provided for direction dependent values.
        Note that this value has no influence on dimensions where a ``vector``
        is provided.

        Default is 3.

    lambda_factor : float, optional
        The buffer is taken as one wavelength from the survey domain. This can
        be regarded as quite conservative (but safe). The parameter
        ``lambda_factor`` can be used to reduce (or increase) this factor.
        Default is 1.0.

    max_buffer : float, optional
        Maximum thickness of the buffer zone around survey domain. If
        ``lambda_from_center=True``, this is the maximum distance from the
        center to the end of the computational domain.
        Default is 100,000 (100 km).

    lambda_from_center : bool, optional
        Flag how to compute the extent of the computational mesh as a function
        of wavelength:

        - False (default): The distance from the edge of the survey domain to
          the edge of the computational domain is one wavelength.
        - True: The distance from the center to the edge of the computational
          domain and back to the end of the survey domain is two wavelengths.

    mapping : str or map, optional
        Defines what type the input ``property_{x;y;z}``-values correspond to.
        By default, they represent resistivities (Ohm.m). The implemented
        mappings are:

        - 'Conductivity'; σ (S/m),
        - 'LgConductivity'; log_10(σ),
        - 'LnConductivity'; log_e(σ),
        - 'Resistivity'; ρ (Ohm.m); Default,
        - 'LgResistivity'; log_10(ρ),
        - 'LnResistivity'; log_e(ρ).

    cell_numbers : list, optional
        List of possible numbers of cells. See :func:`good_mg_cell_nr`.
        Default is ``good_mg_cell_nr(1024, 5, 3)``, which corresponds to
        numbers 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384,
        512, 640, 768, 1024.

    verb : int, optional
        Verbosity, -1 (error); 0 (warning), 1 (info), 2 (verbose).
        Default = 0 (Warnings only).


    Returns
    -------
    origin : float
        Origin of the mesh.

    widths : ndarray
        Cell widths of mesh.


    """
    verb = kwargs.get('verb', 0)

    # Initiate direction-specific dicts, add unambiguous args.
    kwargs['frequency'] = frequency
    xparams = {'center': center[0]}
    yparams = {'center': center[1]}
    zparams = {'center': center[2], 'seasurface': seasurface}

    # Add properties.
    if isinstance(properties, (int, float)):
        properties = np.array([properties])
    if len(properties) == 4:
        xparams['properties'] = [properties[0], properties[1], properties[1]]
        yparams['properties'] = [properties[0], properties[1], properties[1]]
        zparams['properties'] = [properties[0], properties[2], properties[3]]
    elif len(properties) == 7:
        xparams['properties'] = [properties[0], properties[1], properties[2]]
        yparams['properties'] = [properties[0], properties[3], properties[4]]
        zparams['properties'] = [properties[0], properties[5], properties[6]]
    else:
        kwargs['properties'] = properties

    # Add optionally direction specific args.
    for name, value in zip(['domain', 'vector'], [domain, vector]):
        if (value is not None and len(value) == 3 and not
                isinstance(value, np.ndarray)):
            if value[0] is not None:
                xparams[name] = value[0]
            if value[1] is not None:
                yparams[name] = value[1]
            if value[2] is not None:
                zparams[name] = value[2]
        else:
            kwargs[name] = value

    # Add optionally direction specific kwargs.
    for name in ['stretching', 'min_width_limits', 'min_width_pps']:
        value = kwargs.pop(name, None)
        if value is not None:
            if isinstance(value, (int, float)):
                kwargs[name] = np.array([value])
            elif len(value) == 3:
                if value[0] is not None:
                    xparams[name] = value[0]
                if value[1] is not None:
                    yparams[name] = value[1]
                if value[2] is not None:
                    zparams[name] = value[2]
            else:
                kwargs[name] = value

    # Get origins and widths in all directions.
    if verb > 0:
        print("         == GRIDDING IN X ==")
    x0, hx = get_origin_widths(**kwargs, **xparams)
    if verb > 0:
        print("\n         == GRIDDING IN Y ==")
    y0, hy = get_origin_widths(**kwargs, **yparams)
    if verb > 0:
        print("\n         == GRIDDING IN Z ==")
    z0, hz = get_origin_widths(**kwargs, **zparams)

    # Return mesh.
    return TensorMesh([hx, hy, hz], x0=np.array([x0, y0, z0]))


def get_origin_widths(frequency, properties, center, domain=None, vector=None,
                      seasurface=None, **kwargs):
    r"""Return origin and cell widths for given parameters.

    This function works in one dimension only, and is used by
    :func:`construct_mesh` once in each direction. It is recommended to use
    directly function :func:`construct_mesh`, which returns a
    :class:`TensorMesh`.

    All the parameters are described in :func:`construct_mesh`. Described here
    are only the differences.


    Parameters
    ----------

    All : description
        All parameters are described in :func:`construct_mesh`. The only
        difference is that here only variables for one direction are accepted.

    raise_error : bool, optional
        If True, an error is raised if no suitable grid is found. Otherwise it
        just prints a message and returns None's.
        Default is True.

    Returns
    -------
    origin : float
        Origin of the mesh.

    widths : ndarray
        Cell widths of mesh.

    """
    # Get all kwargs.
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
        raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

    # Get property map from string.
    if isinstance(pmap, str):
        pmap = getattr(maps, 'Map'+pmap)()

    # Properties.
    cond = pmap.backward(np.array(properties, ndmin=1, dtype=float))
    cond_arr = np.array([
        cond[0], cond[min(cond.size-1, 1)], cond[min(cond.size-1, 2)]])

    # Get skin depth.
    skind = skin_depth(frequency, cond_arr, precision=0)

    # Minimum cell width.
    dmin = min_cell_width(skind[0], min_width_pps, min_width_limits)

    # Survey domain: if not provided get from vector.
    if domain is None and vector is None:
        raise ValueError(
                "At least one of `domain` and `vector` must be provided.")
    elif domain is None:
        domain = np.array([vector.min(), vector.max()], dtype=float)
    else:
        domain = np.array(domain, dtype=np.float64)
        if vector is not None:
            if domain[0] < vector.min() or domain[1] > vector.max():
                raise ValueError(
                        "Provided vector MUST at least include all of the "
                        "survey domain.")

    # Seasurface related checks.
    if seasurface is not None:

        # Check that seasurface > center.
        if seasurface <= center:
            raise ValueError(
                    "The `seasurface` but be bigger then `center`.")

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
        for sa in np.arange(1.0, stretching[0]+0.005, 0.01):

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
            for ca in np.arange(sa, stretching[1]+0.005, 0.01):

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
            print(f"* ERROR   :: {msg}")
            return None, None

    # Print info about final grid.
    if verb > 0:

        # Check max stretching.
        sa_adj = np.max([hxo[1:]/hxo[:-1], hxo[:-1]/hxo[1:]])
        sa_limit = min(1.5, stretching[0]+0.25)

        if verb > 1:
            end = ""
        else:
            end = "\n"

        print(f"Skin depth          [m] : {skind[0]:.0f}", end="")
        if cond.size == 2:
            print(f" / {skind[1]:.0f}", end=end)
        elif cond.size == 3:
            print(f" / {skind[1]:.0f} / {skind[2]:.0f}", end=end)
        else:
            print(end=end)
        if verb > 1:
            print("  [corresponding to `properties`]")

        print(f"Survey domain DS    [m] : {domain[0]:.0f} - {domain[1]:.0f}")

        print(f"Comp. domain DC     [m] : {comp_domain[0]:.0f} - "
              f"{comp_domain[1]:.0f}")

        print(f"Final extent        [m] : {x0:.0f} - {x0+np.sum(hx):.0f}")

        print(f"Cell widths         [m] : "
              f"{min(hxo):.0f} / {max(hxo):.0f} / {max(hx):.0f}", end=end)
        if verb > 1:
            print("  [min(DS) / max(DS) / max(DC)]")

        print(f"Number of cells         : {nx} ({hxo.size} / "
              f"{nx-hxo.size-nx_remain2} / {nx_remain2})", end=end)
        if verb > 1:
            print("  [Total (DS/DC/remain)]")

        print(f"Max stretching          : {sa:.3f} ({sa_adj:.3f}) / {ca:.3f}",
              end=end)
        if verb > 1:
            print("  [DS (seasurface) / DC]")
        if sa_adj > sa_limit:
            print(f"Note: Stretching in DS >> {sa}.\nThe reason "
                  "is usually the interplay of center/domain/seasurface.")

    return x0, hx


def good_mg_cell_nr(max_nr=1024, max_prime=5, min_div=3):
    r"""Returns 'good' cell numbers for the multigrid method.

    'Good' cell numbers are numbers which can be divided by 2 as many times as
    possible. At the end there will be a low prime number.

    The function adds all numbers :math:`p 2^n \leq M` for :math:`p={2, 3, ...,
    p_\text{max}}` and :math:`n={n_\text{min}, n_\text{min}+1, ..., \infty}`;
    :math:`M, p_\text{max}, n_\text{min}` correspond to `max_nr`, `max_prime`,
    and `min_div`, respectively.


    Parameters
    ----------
    max_nr : int, optional
        Maximum number of cells.
        Default is 1024.

    max_prime : int, optional
        Highest permitted prime number p for p*2^n. {2, 3, 5, 7} are good upper
        limits in order to avoid too big lowest grids in the multigrid method.
        Default is 5.

    min_div : int, optional
        Minimum times the number can be divided by two.
        Default is 3.


    Returns
    -------
    numbers : array
        Array containing all possible cell numbers from lowest to highest.

    """
    # Primes till 20.
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.int64)

    # Sanity check; 19 is already ridiculously high.
    if max_prime > primes[-1]:
        raise ValueError(
                f"Highest prime is {max_prime}, please use a value < 20.")

    # Restrict to max_prime.
    primes = primes[primes <= max_prime]

    # Get possible values.
    # Currently restricted to prime*2**30 (for prime=2 => 1,073,741,824 cells).
    numbers = primes[:, None]*2**np.arange(min_div, 30)

    # Get unique values.
    numbers = np.unique(numbers)

    # Restrict to max_nr and return.
    return numbers[numbers <= max_nr]


def skin_depth(frequency, conductivity, mu=mu_0, precision=0):
    r"""Return skin depth for provided frequency and conductivity.

    The skin depth :math:`\delta` (m) is given by

    .. math::
        :label: skindepth

        \delta = \sqrt{\frac{2}{\omega\sigma\mu}}\ ,

    where :math:`\omega=2\pi f` is angular frequency of frequency :math:`f`
    (Hz), :math:`\sigma` is conductivity (S/m), and :math:`\mu` is magnetic
    permeability (H/m).


    Parameters
    ----------
    frequency : float
        Frequency (Hz).

    conductivity : float
        Conductivity (S/m).

    mu : float, optional
        Magnetic permeability (H/m); default is :math:`\mu_0`.

    precision : int, optional
        Precision of the return skin depth.
        Default is 0, hence meters.


    Returns
    -------
    skindepth : float
        Skin depth (m).

    """
    skind = 1/np.sqrt(np.pi*abs(frequency)*conductivity*mu)
    if frequency < 0:  # For Laplace-domain computations.
        skind /= np.sqrt(2*np.pi)
    return np.round(skind, precision)


def wavelength(skin_depth, precision=0):
    r"""Return the wavelength.

    The wavelength :math:`\lambda` (m) is given by

    .. math::
        :label: wavelength

        \lambda = 2\pi\delta\ .

    The skin depth :math:`\delta` is a function of frequency and conductivity
    and is given by :func:`skin_depth`, Equation :eq:`skindepth`.


    Parameters
    ----------
    skin_depth : float or ndarray.
        Skin depth (m).

    precision : int, optional
        Precision of the returned wave length.
        Default is 0, hence meters.


    Returns
    -------
    wavelength : float or ndarray
        Wavelength (m).

    """
    return np.round(2*np.pi*skin_depth, precision)


def min_cell_width(skin_depth, pps=3, limits=None, precision=0):
    r"""Return the minimum cell width.

    The minimum cell width is defined by the desired points per skin depth,

    .. math::
        :label: mincellwidth

        \Delta_\text{min} =
        \text{limits[0]} \le \frac{\delta}{\text{pps}} \le \text{limits[1]} \ .

    The skin depth :math:`\delta` is a function of frequency and conductivity
    and is given by :func:`skin_depth`, Equation :eq:`skindepth`.


    Parameters
    ----------
    skin_depth : float
        Skin depth (m).

    pps : int
        Points per skin depth.

    limits : None, float, or list of two floats
        Limits on minimum width:

        - None: No limits.
        - float: Returns limits as dmin.
        - [min, max]: dmin is limited to this range.

    precision : int, optional
        Precision of the cell width. Provided limits are not rounded.
        Default is 0, hence meters.


    Returns
    -------
    dmin : float
        Minimum cell width (m).

    """
    # Calculate min cell width.
    dmin = np.round(skin_depth/pps, precision)

    # Respect user limits.
    if limits is not None:

        limits = np.array(limits, ndmin=1)

        if limits.size == 1:
            dmin = limits  # Ignores skin depth and pps.
        else:
            dmin = np.clip(dmin, *limits)  # Restrict.

    return dmin


# # # DEPRECATED FUNCTIONS # # #
def get_hx_h0(freq, res, domain, fixed=0., possible_nx=None, min_width=None,
              pps=3, alpha=None, max_domain=100000., raise_error=True, verb=1,
              return_info=False):
    r"""Return cell widths and origin for given parameters.

    Returns cell widths for the provided frequency, resistivity, domain extent,
    and other parameters using a flexible amount of cells. See input parameters
    for more details. A maximum of three hard/fixed boundaries can be provided
    (one of which is the grid center).

    The minimum cell width is computed through :math:`\delta/\rm{pps}`, where
    the skin depth is given by :math:`\delta = 503.3 \sqrt{\rho/f}`, and the
    parameter `pps` stands for 'points-per-skindepth'. The minimum cell width
    can be restricted with the parameter `min_width`.

    The actual computation domain adds a buffer zone around the (survey)
    domain. The thickness of the buffer is six times the skin depth. The field
    is basically zero after two wavelengths. A wavelength is
    :math:`2\pi\delta`, hence roughly 6 times the skin depth. Taking a factor 6
    gives therefore almost two wavelengths, as the field travels to the
    boundary and back. The actual buffer thickness can be steered with the
    `res` parameter.

    One has to take into account that the air is very resistive, which has to
    be considered not just in the vertical direction, but also in the
    horizontal directions, as the airwave will bounce back from the sides
    otherwise. In the marine case this issue reduces with increasing water
    depth.


    See Also
    --------
    get_stretched_h : Get `hx` for a fixed number `nx` and within a fixed
                      domain.


    Parameters
    ----------

    freq : float
        Frequency (Hz) to compute the skin depth. The skin depth is a concept
        defined in the frequency domain. If a negative frequency is provided,
        it is assumed that the computation is carried out in the Laplace
        domain. To compute the skin depth, the value of `freq` is then
        multiplied by :math:`-2\pi`, to simulate the closest
        frequency-equivalent.

    res : float or list
        Resistivity (Ohm m) to compute the skin depth. The skin depth is
        used to compute the minimum cell width and the boundary thicknesses.
        Up to three resistivities can be provided:

        - float: Same resistivity for everything;
        - [min_width, boundaries];
        - [min_width, left boundary, right boundary].

    domain : list
        Contains the survey-domain limits [min, max]. The actual computation
        domain consists of this domain plus a buffer zone around it, which
        depends on frequency and resistivity.

    fixed : list, optional
        Fixed boundaries, one, two, or maximum three values. The grid is
        centered around the first value. Hence it is the center location with
        the smallest cell. Two more fixed boundaries can be added, at most one
        on each side of the first one.
        Default is 0.

    possible_nx : list, optional
        List of possible numbers of cells. See :func:`good_mg_cell_nr`.
        Default is ``good_mg_cell_nr(1024, 5, 3)``, which corresponds to
        numbers 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384,
        512, 640, 768, 1024.

    min_width : float, list or None, optional
        Minimum cell width restriction:

        - None : No restriction;
        - float : Fixed to this value, ignoring skin depth and `pps`.
        - list [min, max] : Lower and upper bounds.

        Default is None.

    pps : int, optional
        Points per skindepth; minimum cell width is computed via
        `dmin = skindepth/pps`.
        Default = 3.

    alpha : list, optional
        Maximum alpha and step size to find a good alpha. The first value is
        the maximum alpha of the survey domain, the second value is the maximum
        alpha for the buffer zone, and the third value is the step size.
        Default = [1, 1.5, .01], hence no stretching within the survey domain
        and a maximum stretching of 1.5 in the buffer zone; step size is 0.01.

    max_domain : float, optional
        Maximum computation domain from fixed[0] (usually source position).
        Default is 100,000.

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

        - `dmin`: Minimum cell width;
        - `dmax`: Maximum cell width;
        - `amin`: Minimum alpha;
        - `amax`: Maximum alpha.

    """
    warnings.warn(MESHWARNING, DeprecationWarning)

    # Get variables with default lists:
    if alpha is None:
        alpha = [1, 1.5, 0.01]
    if possible_nx is None:
        possible_nx = good_mg_cell_nr()

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
            raise ValueError("Maximum three fixed boundaries permitted. "
                             f"Provided: {fixed.size}.")

        # Sort second and third, so it doesn't matter how it was provided.
        fixed = np.array([fixed[0], max(fixed[1:]), min(fixed[1:])])

        # Check side.
        if np.sign(np.diff(fixed[:2])) == np.sign(np.diff(fixed[::2])):
            raise ValueError(
                    "2nd and 3rd fixed boundaries have to be left and right "
                    "of the first one.\n"
                    f"Provided: [{fixed[0]}, {fixed[1]}, {fixed[2]}]")

    # Get skin depth.
    skind = skin_depth(freq, 1/res_arr, precision=3)

    # Minimum cell width.
    dmin = min_cell_width(skind[0], pps, min_width)

    # Survey domain; contains all sources and receivers.
    domain = np.array(domain, dtype=np.float64)

    # Computation domain; big enough to avoid boundary effects.
    # To avoid boundary effects we want the signal to travel two wavelengths
    # from the source to the boundary and back to the receiver.
    # => 2*pi*sd ~ 6.3*sd = one wavelength => signal is ~ 0.2 %.
    # Two wavelengths we can safely assume it is zero.
    #
    # The air does not follow the concept of skin depth, as it is a wave rather
    # than diffusion. For this is the factor `max_domain`, which restricts
    # the domain in each direction to this value from the center.

    # (a) Source to edges of domain.
    dist_in_domain = abs(domain - fixed[0])

    # (b) Two wavelengths.
    two_lambda = 2*wavelength(skind[1:], precision=3)

    # (c) Required buffer, additional to domain.
    dist_buff = np.max([np.zeros(2), (two_lambda - dist_in_domain)/2], axis=0)

    # (d) Add buffer to domain.
    comp_domain = np.array([domain[0]-dist_buff[0], domain[1]+dist_buff[1]])

    # (e) Restrict total domain to max_domain.
    comp_domain[0] = max(comp_domain[0], fixed[0]-max_domain)
    comp_domain[1] = min(comp_domain[1], fixed[0]+max_domain)

    # Initiate flag if terminated.
    finished = False

    # Initiate alpha variables for survey and computation domains.
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

            # Loop over possible alphas for comp_domain.
            for ca in np.arange(sa, alpha[1]+alpha[2]/2, alpha[2]):

                # 4. Fill to left comp_domain.
                thxl = hx[0]*ca**np.arange(1, nx_remain+1)
                nl = np.sum((asurv_domain[0]-np.cumsum(thxl)) >
                            comp_domain[0])+1

                # 5. Fill to right comp_domain.
                thxr = hx[-1]*ca**np.arange(1, nx_remain+1)
                nr = np.sum((asurv_domain[1]+np.cumsum(thxr)) <
                            comp_domain[1])+1

                # 6. Get remaining number of cells and check termination
                # criteria.
                ncdc = nl+nr  # Number of comp_domain cells.
                nx_remain2 = nx-nsdc-ncdc

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

    # Check finished and print info about found grid.
    if not finished:
        # Throw message if no solution was found.
        if raise_error:
            raise RuntimeError("No suitable grid found; relax your criteria.")
        else:
            print("* ERROR   :: No suitable grid found; relax your criteria.")
            hx, x0 = None, None

    elif verb > 0:
        print("   Skin depth ", end="")
        if res.size == 1:
            print(f"         [m] : {skind[0]:.0f}")
        elif res.size == 2:
            print(f"(m/l-r)  [m] : {skind[0]:.0f} / {skind[1]:.0f}")
        else:
            print(f"(m/l/r)  [m] : {skind[0]:.0f} / {skind[1]:.0f} / "
                  f"{skind[2]:.0f}")
        print(f"   Survey domain       [m] : {domain[0]:.0f} - "
              f"{domain[1]:.0f}")
        print(f"   Computation domain  [m] : {comp_domain[0]:.0f} - "
              f"{comp_domain[1]:.0f}")
        print(f"   Final extent        [m] : {x0:.0f} - "
              f"{x0+np.sum(hx):.0f}")
        extstr = f"   Min/max cell width  [m] : {min(hx):.0f} / "
        alstr = "   Alpha survey"
        nrstr = "   Number of cells "
        if not np.isclose(sa, sa_adj):
            sastr = f"{sa:.3f} ({sa_adj:.3f})"
        else:
            sastr = f"{sa:.3f}"
        print(extstr+f"{max(hxo):.0f} / {max(hx):.0f}")
        print(alstr+f"/comp       : {sastr} / {ca:.3f}")
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
    warnings.warn(CELLWARNING, DeprecationWarning)
    return good_mg_cell_nr(max_nr, max_prime, min_div)


def get_stretched_h(min_width, domain, nx, x0=0, x1=None, resp_domain=False):
    """Return cell widths for a stretched grid within the domain.

    Returns `nx` cell widths within `domain`, where the minimum cell width is
    `min_width`. The cells are not stretched within `x0` and `x1`, and outside
    uses a power-law stretching. The actual stretching factor and the number of
    cells left and right of `x0` and `x1` are found in a minimization process.

    The domain is not completely respected. The starting point of the domain
    is, but the endpoint of the domain might slightly shift (this is more
    likely the case for small `nx`, for big `nx` the shift should be small).
    The new endpoint can be obtained with ``domain[0]+np.sum(hx)``. If you want
    the domain to be respected absolutely, set ``resp_domain=True``. However,
    be aware that this will introduce one stretch-factor which is different
    from the other stretch factors, to accommodate the restriction. This
    one-off factor is between the left- and right-side of `x0`, or, if `x1` is
    provided, just after `x1`.


    See Also
    --------
    get_hx_x0 : Get `hx` and `x0` for a flexible number of `nx` with
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
        Center of the grid. `x0` is restricted to `domain`.
        Default is 0.

    x1 : float
        If provided, then no stretching is applied between `x0` and `x1`. The
        non-stretched part starts at `x0` and stops at the first possible
        location at or after `x1`. `x1` is restricted to `domain`. This will
        min_width so that an integer number of cells fit within x0 and x1.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of `x0`, or, if `x1` is provided, just after `x1`.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """
    warnings.warn(MESHWARNING, DeprecationWarning)

    # Cast to arrays
    domain = np.array(domain, dtype=np.float64)
    x0 = np.array(x0, dtype=np.float64)
    x0 = np.clip(x0, *domain)  # Restrict to model domain
    min_width = np.array(min_width, dtype=np.float64)
    if x1 is not None:
        x1 = np.array(x1, dtype=np.float64)
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

        # Re-compute min_width to fit with x0-x1-limits:
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

    Returns the extent of the computation domain and the minimum cell width as
    a multiple of the skin depth, with possible user restrictions on minimum
    computation domain and range of possible minimum cell widths.

    .. math::

            \delta &= 503.3 \sqrt{\frac{\rho}{f}} , \\
            x_\text{start} &= x_0-k_\text{neg}\delta , \\
            x_\text{end} &= x_0+k_\text{pos}\delta , \\
            h_\text{min} &= k_\text{min} \delta .


    Parameters
    ----------

    x0 : float
        Center of the computation domain. Normally the source location.
        Default is 0.

    freq : float
        Frequency (Hz) to compute the skin depth. The skin depth is a concept
        defined in the frequency domain. If a negative frequency is provided,
        it is assumed that the computation is carried out in the Laplace
        domain. To compute the skin depth, the value of `freq` is then
        multiplied by :math:`-2\pi`, to simulate the closest
        frequency-equivalent.

        Default is 1 Hz.

    res : float, optional
        Resistivity (Ohm m) to compute skin depth.
        Default is 0.3 Ohm m (sea water).

    limits : None or list
        [start, end] of model domain. This extent represents the minimum extent
        of the domain. The domain is therefore only adjusted if it has to reach
        outside of [start, end].
        Default is None.

    min_width : None, float, or list of two floats
        Minimum cell width is computed as a function of skin depth:
        fact_min*sd. If `min_width` is a float, this is used. If a list of
        two values [min, max] are provided, they are used to restrain
        min_width. Default is None.

    fact_min, fact_neg, fact_pos : floats
        The skin depth is multiplied with these factors to estimate:

            - Minimum cell width (`fact_min`, default 0.2)
            - Domain-start (`fact_neg`, default 5), and
            - Domain-end (`fact_pos`, defaults to `fact_neg`).


    Returns
    -------

    h_min : float
        Minimum cell width.

    domain : list
        Start- and end-points of computation domain.

    """
    warnings.warn(MESHWARNING, DeprecationWarning)

    # Set fact_pos to fact_neg if not provided.
    if fact_pos is None:
        fact_pos = fact_neg

    # Get skin depth.
    skind = skin_depth(freq, 1/res, precision=3)

    # Estimate minimum cell width.
    h_min = fact_min*skind
    if min_width is not None:  # Respect user input.
        if np.array(min_width).size == 1:
            h_min = min_width
        else:
            h_min = np.clip(h_min, *min_width)

    # Estimate computation domain.
    domain = [x0-fact_neg*skind, x0+fact_pos*skind]
    if limits is not None:  # Respect user input.
        domain = [min(limits[0], domain[0]), max(limits[1], domain[1])]

    return h_min, domain


def get_hx(alpha, domain, nx, x0, resp_domain=True):
    r"""Return cell widths for given input.

    Find the number of cells left (``nl``) and right (``nr``) of the center
    ``x0`` for the provided alpha. For this, we solve

    .. math::   \frac{x_\text{max}-x_0}{x_0-x_\text{min}} =
                \frac{a^\text{nr}-1}{a^\text{nl}-1}

    where :math:`a = 1+\alpha`.


    Parameters
    ----------

    alpha : float
        Stretching factor `a` is given by ``a=1+alpha``.

    domain : list
        ``[x_min, x_max]`` of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. ``x0`` is restricted to ``domain``.

    resp_domain : bool
        If False, then the domain-end might shift slightly to assure that the
        same stretching factor is applied throughout. If set to True (default),
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of `x0`, or, if `x1` is provided, just after `x1`.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh. All points are given by
        ``np.r_[xmin, xmin+np.cumsum(hx)]``

    """
    warnings.warn(MESHWARNING, DeprecationWarning)

    if alpha <= 0.:  # If alpha <= 0: equal spacing (no stretching at all)
        hx = np.ones(nx)*np.diff(np.squeeze(domain))/nx

    else:            # Get stretched hx
        a = alpha+1

        # Get hx depending if x0 is on the domain boundary or not.
        if np.isclose(x0, domain[0]) or np.isclose(x0, domain[1]):
            # Get all a's
            alr = np.diff(domain)*alpha/(a**nx-1)*a**np.arange(nx)
            if x0 == domain[1]:
                alr = alr[::-1]

            # Compute differences
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

            # Compute differences
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
