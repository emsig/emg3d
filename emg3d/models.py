"""
Everything to store electromagnetic material properties for the solver.
"""
# Copyright 2018 The emsig community.
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
import scipy as sp

from emg3d import maps, meshes, utils

__all__ = ['Model', 'VolumeModel', 'expand_grid_model']


def __dir__():
    return __all__


# MODEL
@utils._known_class
class Model:
    r"""A model containing the electromagnetic properties of the Earth.

    A model provides the required model parameters to the solver. The x-, y-,
    and z-directed electrical properties are by default resistivities. However,
    they can be defined as resistivities, :math:`\rho\ (\Omega\,\mathrm{m})`,
    or conductivities, :math:`\sigma\ (\mathrm{S/m})`, either on a linear or on
    a logarithmic scale (log or ln), by choosing the appropriate ``mapping``.
    Relative magnetic permeability :math:`\mu_\mathrm{r}` is by default set to
    one and electric permittivity :math:`\varepsilon_\mathrm{r}` is by default
    set to zero, but they can also be provided (isotropically). Keep in mind
    that the multigrid method as implemented in emg3d works for the diffusive
    approximation. When the displacement part in Maxwell's equations becomes
    too dominant it will start to fail (high frequencies or very high electric
    permittivity).


    Parameters
    ----------
    grid : TensorMesh
        The grid; a :class:`emg3d.meshes.TensorMesh` instance.

    property_{x;y;z} : {None, array_like}, default: 1 (x), None (y, z)
        Electrical material property in x-, y-, and z-directions. The
        properties are stored as Fortran-ordered arrays with the shape given by
        ``grid.shape``. The provided value must be broadcastable to that shape.

        By default, property refers to electrical resistivity. However, this
        can be changed with an appropriate ``mapping`` (the internals of
        emg3d work, irrelevant of the map, with electrical conductivities).

        The properties have to be of finite value, bigger than zero (on linear
        scale). The four supported anisotropy cases are:

        - ``x;y=None;z=None``: isotropic (``y=z=x``);
        - ``x;y=None;z``: vertical transverse isotropy VTI (``y=x``);
        - ``x;y;z=None``: horizontal transverse isotropy HTI (``z=x``);
        - ``x;y;z``: triaxial  anisotropy.

        If a property is not initiated it cannot be set later on (e.g., if a
        VTI model is created, it is not possible to set ``property_y`` later
        on, instead, a new model has to be initiated).

    mu_r, epsilon_r : {None, array_like}, default: None
        Relative magnetic permeability (-) and relative electric permittivity
        (-), respectively, both isotropic. The properties are stored as
        Fortran-ordered arrays with the shape given by ``grid.shape``. The
        provided value must be broadcastable to that shape.

        The properties have to be of finite value, bigger than zero.

        The relative magnetic permeability is assumed to be 1 if not provided.

        The relative electric permittivity is assumed to be 0 if not provided,
        which ignores the displacement part completely (diffusive
        approximation)

    mapping : str, default: 'Resistivity'
        Defines what type the electrical input ``property_{x;y;z}``-values
        correspond to. The implemented types are:

        - ``'Resistivity'``; ρ (Ω m);
        - ``'Conductivity'``; σ (S/m);
        - ``'LgResistivity'``; log_10(ρ);
        - ``'LgConductivity'``; log_10(σ);
        - ``'LnResistivity'``; log_e(ρ);
        - ``'LnConductivity'``; log_e(σ).

    """

    def __init__(self, grid, property_x=1., property_y=None, property_z=None,
                 mu_r=None, epsilon_r=None, mapping='Resistivity'):
        """Initiate a new model."""

        # Store grid.
        self.grid = grid

        # Alias shape_cells and n_cells to shape and size.
        self.shape = self.grid.shape_cells
        self.size = self.grid.n_cells

        # Get and store map.
        if isinstance(mapping, maps.BaseMap):
            # It can be an already instantiated map (not in docstring).
            self.map = mapping
        else:
            self.map = getattr(maps, 'Map'+mapping)()

        # Initiate and store all parameters.
        self._property_x = self._init_parameter(property_x, 'property_x')
        self._property_y = self._init_parameter(property_y, 'property_y')
        self._property_z = self._init_parameter(property_z, 'property_z')
        self._mu_r = self._init_parameter(mu_r, 'mu_r',)
        self._epsilon_r = self._init_parameter(epsilon_r, 'epsilon_r')
        self._properties = ['property_x', 'property_y', 'property_z',
                            'mu_r', 'epsilon_r']

        # Store case.
        if self._property_y is None and self._property_z is None:
            self.case = 'isotropic'
        elif self._property_z is None:
            self.case = 'HTI'
        elif self._property_y is None:
            self.case = 'VTI'
        else:
            self.case = 'triaxial'

    def __repr__(self):
        """Simple representation."""
        return (f"{self.__class__.__name__}: {self.map.description}; "
                f"{self.case}{'' if self.mu_r is None else '; mu_r'}"
                f"{'' if self.epsilon_r is None else '; epsilon_r'}"
                f"; {self.shape[0]} x {self.shape[1]} x {self.shape[2]} "
                f"({self.size:,})")

    def __add__(self, model):
        """Add two models."""

        # Ensure model is a Model instance.
        if model.__class__.__name__ != 'Model':
            return NotImplemented

        # Check input.
        self._operator_test(model)

        # Apply operator.
        kwargs = self._apply_operator(model, np.add)

        # Return new Model instance.
        return Model(grid=self.grid, mapping=self.map.name, **kwargs)

    def __sub__(self, model):
        """Subtract two models."""

        # Ensure model is a Model instance.
        if model.__class__.__name__ != 'Model':
            return NotImplemented

        # Check input.
        self._operator_test(model)

        # Apply operator.
        kwargs = self._apply_operator(model, np.subtract)

        # Return new Model instance.
        return Model(grid=self.grid, mapping=self.map.name, **kwargs)

    def __eq__(self, model):
        """Compare two models."""

        # Check if model is a Model instance.
        equal = model.__class__.__name__ == 'Model'

        # Check input.
        if equal:
            try:
                self._operator_test(model)
            except ValueError:
                equal = False

        # Compare values if not None.
        if equal:
            for prop in self._def_properties:
                equal *= np.allclose(getattr(self, prop), getattr(model, prop))

        return bool(equal)

    def copy(self):
        """Return a copy of the Model."""
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
            Dictionary containing all information to re-create the Model.

        """
        out = {
            '__class__': self.__class__.__name__,  # v ensure emg3d-TensorMesh
            'grid': meshes.TensorMesh(self.grid.h, self.grid.origin).to_dict(),
            **{prop: getattr(self, prop) for prop in self._properties},
            'mapping': self.map.name,
        }
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`emg3d.models.Model` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`emg3d.models.Model.to_dict`. The
            dictionary needs the keys ``property_x``, ``property_y``,
            ``property_z``, ``mu_r``, ``epsilon_r``, ``grid``, and ``mapping``;
            ``grid`` itself is also a dict which needs the keys ``hx``, ``hy``,
            ``hz``, and ``origin``.

        Returns
        -------
        model : Model
            A :class:`emg3d.models.Model` instance.

        """
        inp = {k: v for k, v in inp.items() if k != '__class__'}
        MeshClass = getattr(meshes, inp['grid']['__class__'])
        return cls(grid=MeshClass.from_dict(inp.pop('grid')), **inp)

    # ELECTRICAL PROPERTIES
    @property
    def property_x(self):
        r"""Electrical property in x-direction."""
        return self._property_x

    @property_x.setter
    def property_x(self, property_x):
        r"""Update electrical property in x-direction."""
        self._check_positive_finite(property_x, 'property_x')
        self._property_x[:] = np.asfortranarray(property_x, dtype=np.float64)

    @property
    def property_y(self):
        r"""Electrical property in y-direction."""
        return self._property_y

    @property_y.setter
    def property_y(self, property_y):
        r"""Update electrical property in y-direction."""
        self._check_positive_finite(property_y, 'property_y')
        self._property_y[:] = np.asfortranarray(property_y, dtype=np.float64)

    @property
    def property_z(self):
        r"""Electrical property in z-direction."""
        return self._property_z

    @property_z.setter
    def property_z(self, property_z):
        r"""Update electrical property in z-direction."""
        self._check_positive_finite(property_z, 'property_z')
        self._property_z[:] = np.asfortranarray(property_z, dtype=np.float64)

    @property
    def mu_r(self):
        r"""Relative magnetic permeability."""
        return self._mu_r

    @mu_r.setter
    def mu_r(self, mu_r):
        r"""Update relative magnetic permeability."""
        self._check_positive_finite(mu_r, 'mu_r')
        self._mu_r[:] = np.asfortranarray(mu_r, dtype=np.float64)

    @property
    def epsilon_r(self):
        r"""Relative electric permittivity."""
        return self._epsilon_r

    @epsilon_r.setter
    def epsilon_r(self, epsilon_r):
        r"""Update relative electric permittivity."""
        self._check_positive_finite(epsilon_r, 'epsilon_r')
        self._epsilon_r[:] = np.asfortranarray(epsilon_r, dtype=np.float64)

    @property
    def _def_properties(self):
        """Returns a list of the defined (not-None) properties."""
        if not hasattr(self, '__def_properties'):
            self.__def_properties = [
                k for k in self._properties if getattr(self, k) is not None
            ]
        return self.__def_properties

    # INTERPOLATION
    def interpolate_to_grid(self, grid, **interpolate_opts):
        """Interpolate the model to a new grid.

        If the provided grid is identical to the grid of the model, it returns
        the actual model (not a copy).


        Parameters
        ----------
        grid : TensorMesh
            Grid of the new model; a :class:`emg3d.meshes.TensorMesh` instance.

        interpolate_opts : dict
            Passed through to :func:`emg3d.maps.interpolate`. Defaults are
            ``method='volume'``, ``log=True``, and ``extrapolate=True``.


        Returns
        -------
        obj : Model
            A new :class:`emg3d.models.Model` instance on ``grid``.

        """
        # If grids are identical, return model.
        if grid == self.grid:
            return self

        # Get solver options, set to defaults if not provided.
        g2g_inp = {
            'method': 'volume',
            'extrapolate': True,
            'log': not self.map.name.startswith('L'),
            **({} if interpolate_opts is None else interpolate_opts),
            'grid': self.grid,
            'xi': grid,
        }

        # Interpolate property_{x;y;z}; mu_r; and epsilon_r; add to dict.
        model_inp = {}
        for prop in self._def_properties:
            var = getattr(self, prop)
            model_inp[prop] = maps.interpolate(values=var, **g2g_inp)

        # Assemble new model.
        return Model(grid, mapping=self.map.name, **model_inp)

    def extract_1d(self, method, p0, p1=None, ellipse=None, merge=False,
                   return_imat=False):
        """Return a layered (1D) model.

        The returned model has shape (1, 1, nz), where nz is the original
        number of cells in vertical direction (except if ``merge=True``). How
        this 1D model is obtained depends on the input parameters.

        .. note::

            If ``method`` is either ``'cylinder'`` or ``'prism'``, an
            ``ellipse``-dict has to be provided with at least the key-value
            pair for ``'radius'``. E.g., ``ellipse={'radius': 1000}``.


        Parameters
        ----------
        method : str
            Method how to obtain the layered model. Implemented are:

            - ``'midpoint'``: Returns the model at the midpoint between ``p0``
              and ``p1``. If ``p1`` is not provided, the model at ``p0`` is
              returned. The x- and y-width of the returned model corresponds to
              the selected cell.
            - ``'cylinder'``: Volume-averages the values of each layer within a
              right elliptic cylinder, using the function
              :func:`emg3d.maps.ellipse_indices` to obtain the ellipse. This
              method requires the ``ellipse``-parameter. The x- and y-width of
              the returned model corresponds to the x- and y-width of the
              ellipse.
            - ``'prism'``: Same as ``'cylinder'``, but an enveloping right
              rectangular prism is fit around the cylinder.

        p0, p1 : array_like
            (x, y)-coordinates of two points. If ``p1`` is not provided, it is
            set to ``p0``.

        ellipse : dict, default: None
            Options-dict passed through to :func:`emg3d.maps.ellipse_indices`.
            This is only used in the methods ``'cylinder'`` and ``'prism'``,
            for which it is a required parameter.

        merge : bool, default: False
            If True, adjacent layers of identical properties are combined. That
            implies that the returned model might have less (but larger) cells
            in z-direction than the original model.

        return_imat : bool, default: False
            If True, the interpolation matrix is returned in addition to the
            model. This matrix can be used, for instance, to inspect the chosen
            selection.

        Returns
        -------
        layered : Model
            The layered Model; a :class:`emg3d.models.Model` instance.

        imat : ndarray, returned if return_imat=True
            Interpolation matrix of shape (nx, ny) of the original model.

        """
        # Get ellipse dict.
        ellipse = {} if ellipse is None else ellipse

        # Check if input is consistent with method.
        methods = ['midpoint', 'cylinder', 'prism']
        if method not in methods:
            raise ValueError(
                f"Unknown method '{method}'; implemented: {methods}."
            )
        if method != 'midpoint' and 'radius' not in ellipse.keys():
            raise TypeError(
                f"Method '{method}' requires the dict 'ellipse' "
                "containing at least the parameter 'radius'."
            )

        # Flag if midpoint is used or another method.
        midpoint = method == 'midpoint'

        # If p1 is not given, we set it to p0.
        if p1 is None:
            p1 = p0

        # Get relevant indices and the used-cells matrix.
        if not midpoint:

            # Indices of used cells.
            coo = (self.grid.cell_centers_x, self.grid.cell_centers_y)
            use = maps.ellipse_indices(coo=coo, p0=p0, p1=p1, **ellipse)

            # Start and end indices.
            ix, iy = use.nonzero()
            if not ix.size:  # Switch to 'midpoint' if selection is empty.
                midpoint = True
            else:
                six, eix = ix.min(), ix.max()
                siy, eiy = iy.min(), iy.max()

        if midpoint:

            # Find corresponding indices, limit to grid.
            def index(nodes, coo):
                """Return index for interval btw nodes in which coo resides."""
                x = np.asarray(coo < np.r_[nodes, np.inf]).nonzero()[0][0]-1
                return np.clip(x, 0, nodes.size-2)

            # Start and end indices.
            six = eix = index(self.grid.nodes_x, (p0[0] + p1[0])/2)
            siy = eiy = index(self.grid.nodes_y, (p0[1] + p1[1])/2)

        # Create interpolation matrix.
        imat = np.zeros(self.shape[:2])
        if not midpoint:
            pp = np.outer(self.grid.h[0][six:eix+1], self.grid.h[1][siy:eiy+1])
            if method == "cylinder":
                pp *= use[six:eix+1, siy:eiy+1]
            pp /= pp.sum()
        else:
            pp = 1.0
        imat[six:eix+1, siy:eiy+1] = pp

        # Interpolate property_{x;y;z}; mu_r; and epsilon_r.
        props = {}
        for prop in self._def_properties:
            values = getattr(self, prop)

            if not midpoint:
                if not self.map.name.startswith('L'):
                    values = np.log10(values)
                val = np.einsum('ij,ijk->k', imat, values)
                if not self.map.name.startswith('L'):
                    val = 10**val
            else:
                val = values[six, siy, :]

            props[prop] = val

        # If merge, combine adjacent layers of identical properties.
        if merge:

            # Get indices of non-merge sequences.
            diff = np.zeros(self.shape[2])
            for k, v in props.items():
                diff += abs(np.diff(np.r_[-1, v]))
            ind = diff.nonzero()[0]

            # Merge.
            props = {k: v[ind] for k, v in props.items()}

            # Get cell sizes in z.
            hz = np.diff(np.r_[self.grid.nodes_z[ind], self.grid.nodes_z[-1]])

        else:
            hz = self.grid.h[2]

        # Create (1, 1, nz) grid.
        grid_out = meshes.TensorMesh(
            h=(
                [self.grid.nodes_x[eix+1] - self.grid.nodes_x[six], ],
                [self.grid.nodes_y[eiy+1] - self.grid.nodes_y[siy], ],
                hz
            ),
            origin=(
                self.grid.nodes_x[six],
                self.grid.nodes_y[siy],
                self.grid.origin[2]
            ),
        )

        # Get layered model on grid_out.
        layered = Model(grid=grid_out, **props, mapping=self.map)

        # Return "1D" model (only 1 cell in x/y).
        if return_imat:
            return layered, imat
        else:
            return layered

    # INTERNAL UTILITIES
    def _init_parameter(self, values, name):
        """Initiate parameter by casting and broadcasting."""

        # If None, exit.
        if values is None:
            return None

        # Cast it to an array of floats, in Fortran order.
        values = np.asfortranarray(values, dtype=np.float64)

        # If 1D array of self.size, reshape it.
        if values.size == self.size:
            values = values.reshape(self.shape, order='F')

        # If not of shape self.shape, broadcast it.
        elif values.shape != self.shape:
            values = np.ones(self.shape, order='F')*values

        # Check >0 and finite.
        self._check_positive_finite(values, name)

        return values

    def _check_positive_finite(self, values, name):
        """Check parameter values are positive (on linear scale) and finite."""

        # If it is None, it cannot be set.
        if hasattr(self, '_'+name) and getattr(self, '_'+name) is None:
            raise ValueError(
                f"Model was initiated without `{name}`; cannot set values."
            )

        # Get mapped values; checks are carried out on conductivities.
        if 'property_' in name:
            mapped = self.map.backward(np.asarray(values))
        else:
            mapped = values

        # Check they are positive.
        if not np.all(np.real(mapped) > 0.0):
            raise ValueError(f"`{name}` must be all bigger than zero.")

        # Check |val| < inf.
        if not np.all(np.isfinite(mapped)):
            raise ValueError(f"`{name}` must be all finite.")

    def _operator_test(self, model):
        """Check if ``self`` and ``model`` are consistent for operations."""

        # Ensure the two instances have the same grid.
        if self.grid != model.grid:
            raise ValueError("Models have different grids.")

        # Ensure the two instances have the same case.
        if self.case != model.case:
            raise ValueError("Models have different anisotropy.")

        # Ensure both or none has mu_r:
        if (self.mu_r is None) != (model.mu_r is None):
            raise ValueError("One model has mu_r, the other not.")

        # Ensure both or none has epsilon_r:
        if (self.epsilon_r is None) != (model.epsilon_r is None):
            raise ValueError("One model has epsilon_r, the other not.")

        # Ensure the two instances have the same mapping:
        if self.map.name != model.map.name:
            raise ValueError("Models have different mappings.")

    def _apply_operator(self, model, operator):
        """Apply the provided operator to self and model."""

        # Apply operator to property_{x;y;z}; mu_r; and epsilon_r; add to dict.
        kwargs = {}
        for prop in self._def_properties:
            kwargs[prop] = operator(getattr(self, prop), getattr(model, prop))

        return kwargs


class VolumeModel:
    r"""Return simplified model with volume-averaged eta_{x;y;z}; zeta.

    Takes a model and a source field and returns the volume-averaged eta and
    zeta values. This is used internally by the solver.

    .. math::

        \eta_{\{x,y,z\}} = -V\mathrm{i}\omega\mu_0
              \left(\sigma_{\{x,y,z\}} + \mathrm{i}\omega\varepsilon\right)

    .. math::

        \zeta = V\mu_\mathrm{r}^{-1}


    Parameters
    ----------
    model : Model
        Model to transform to volume-averaged values.

    sfield : Field
       A VolumeModel is frequency-dependent. The frequency-information is taken
       from the provided source field.

    """

    def __init__(self, model, sfield):
        """Initiate a new model with volume-averaged properties."""

        # Store case and minimal TensorMesh.
        self.case = model.case
        self.grid = meshes.BaseMesh(model.grid.h, model.grid.origin)

        # Get volume for volume-averaged values.
        vol = self.grid.cell_volumes.reshape(model.shape, order='F')

        # Compute and store eta.
        for name in model._properties[:3]:

            prop = getattr(model, name)

            if prop is None:
                eta = None

            else:
                # Get conductivities.
                cond = model.map.backward(prop)

                # Diffusive approximation.
                if model.epsilon_r is None:
                    eta = -sfield.smu0*vol*cond

                # Complete version.
                else:
                    smu = sfield.sval*sp.constants.epsilon_0*model.epsilon_r
                    eta = -sfield.smu0*vol*(cond + smu)

            setattr(self, '_eta_' + name[-1], eta)

        # Compute and store zeta.
        zeta = vol
        if model.mu_r is not None:
            zeta /= model.mu_r
        self._zeta = zeta

    @property
    def eta_x(self):
        r"""Volume-averaged eta in x-direction."""
        return self._eta_x

    @property
    def eta_y(self):
        r"""Volume-averaged eta in y-direction."""
        if self.case in ['HTI', 'triaxial']:
            return self._eta_y
        else:
            return self._eta_x

    @property
    def eta_z(self):
        r"""Volume-averaged eta in z-direction."""
        if self.case in ['VTI', 'triaxial']:
            return self._eta_z
        else:
            return self._eta_x

    @property
    def zeta(self):
        r"""Volume-averaged, isotropic zeta."""
        return self._zeta


def expand_grid_model(model, expand, interface):
    """Expand model and grid according to provided parameters.

    Expand the grid and corresponding model in positive z-direction from the
    edge of the grid to the interface with property ``expand[0]``, and a 100 m
    thick layer above the interface with property ``expand[1]``.

    The provided properties are taken as isotropic (as is the case in water and
    air); ``mu_r`` and ``epsilon_r`` are expanded with ones, if necessary.

    The ``interface`` is usually the sea-surface, and ``expand`` is therefore
    ``[property_sea, property_air]``.

    Parameters
    ----------
    model : Model
        The model; a :class:`emg3d.models.Model` instance.

    expand : list
        The two properties below and above the interface:
        ``[below_interface, above_interface]``.

    interface : float
        Interface between the two properties in ``expand``.


    Returns
    -------
    exp_grid : TensorMesh
        Expanded grid; a :class:`emg3d.meshes.TensorMesh` instance.

    exp_model : Model
        The expanded model; a :class:`emg3d.models.Model` instance.

    """
    grid = model.grid

    def extend_property(prop, add_values, nadd):
        """Expand property `model.prop`, IF it is not None."""

        if getattr(model, prop) is None:
            prop_ext = None

        else:
            prop_ext = np.zeros((grid.shape_cells[0], grid.shape_cells[1],
                                 grid.shape_cells[2]+nadd))
            prop_ext[:, :, :-nadd] = getattr(model, prop)
            if nadd == 2:
                prop_ext[:, :, -2] = add_values[0]
            prop_ext[:, :, -1] = add_values[1]

        return prop_ext

    # Initiate.
    nzadd = 0
    hz_ext = grid.h[2]

    # Fill-up property_below.
    if grid.nodes_z[-1] < interface-0.05:  # At least 5 cm.
        hz_ext = np.r_[hz_ext, interface-grid.nodes_z[-1]]
        nzadd += 1

    # Add 100 m of property_above.
    if grid.nodes_z[-1] <= interface+0.001:  # +1mm
        hz_ext = np.r_[hz_ext, 100]
        nzadd += 1

    if nzadd > 0:
        # Extend properties.
        property_x = extend_property('property_x', expand, nzadd)
        property_y = extend_property('property_y', expand, nzadd)
        property_z = extend_property('property_z', expand, nzadd)
        mu_r = extend_property('mu_r', [1, 1], nzadd)
        epsilon_r = extend_property('epsilon_r', [1, 1], nzadd)

        # Create extended grid and model.
        grid = meshes.TensorMesh(
                [grid.h[0], grid.h[1], hz_ext], origin=grid.origin)
        model = Model(grid, property_x, property_y, property_z, mu_r,
                      epsilon_r, mapping=model.map.name)

    return model
