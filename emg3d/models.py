"""
Everything to create model-properties for the multigrid solver.
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
from scipy.constants import epsilon_0

from emg3d import maps

__all__ = ['Model', 'VolumeModel']


# MODEL
class Model:
    r"""Create a model instance.

    Class to provide model parameters (x-, y-, and z-directed properties
    [resistivity or conductivity; linear or on log_10/log_e-scale], electric
    permittivity and magnetic permeability) to the solver. Relative magnetic
    permeability :math:`\mu_\mathrm{r}` is by default set to one and electric
    permittivity :math:`\varepsilon_\mathrm{r}` is by default set to zero, but
    they can also be provided (isotropically). Keep in mind that the multigrid
    method as implemented in `emg3d` only works for the diffusive
    approximation. As soon as the displacement-part in the Maxwell's equations
    becomes too dominant it will fail (high frequencies or very high electric
    permittivity).


    Parameters
    ----------
    grid : TensorMesh
        Grid on which to apply model.

    property_{x;y;z} : float or ndarray; default to 1.
        Material property in x-, y-, and z-directions. If ndarray, they must
        have the shape of grid.vnC (F-ordered) or grid.nC.

        By default, property refers to electrical resistivity. However, this
        can be changed with an appropriate map. For more info, see the
        description of the parameter `mapping`. The internals of `emg3d` work,
        irrelevant of the map, with electrical conductivities.

        Resistivities and conductivities have to be bigger than zero and
        smaller than infinity (if provided on a linear scale; not on
        logarithmic scales).

    mu_r : None, float, or ndarray
        Relative magnetic permeability (isotropic). If ndarray it must have the
        shape of grid.vnC (F-ordered) or grid.nC. Default is None, which
        corresponds to 1., but avoids the computation of zeta. Magnetic
        permeability has to be bigger than zero and smaller than infinity.

    epsilon_r : None, float, or ndarray
        Relative electric permittivity (isotropic). If ndarray it must have the
        shape of grid.vnC (F-ordered) or grid.nC. The displacement part is
        completely neglected (diffusive approximation) if set to None, which is
        the default. Electric permittivity has to be bigger than zero and
        smaller than infinity.

    mapping : str
        Defines what type the input `property_{x;y;z}`-values correspond to. By
        default, they represent resistivities (Ohm.m). The implemented types
        are:

        - 'Conductivity'; σ (S/m),
        - 'LgConductivity'; log_10(σ),
        - 'LnConductivity'; log_e(σ),
        - 'Resistivity'; ρ (Ohm.m); Default,
        - 'LgResistivity'; log_10(ρ),
        - 'LnResistivity'; log_e(ρ).

    """

    def __init__(self, grid, property_x=1., property_y=None, property_z=None,
                 mu_r=None, epsilon_r=None, mapping='Resistivity', **kwargs):
        """Initiate a new model."""

        # Ensure no kwargs left.
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

        # Store required info from grid.
        if hasattr(grid, 'shape'):
            # This is an alternative possibility. Instead of the grid, we only
            # need the model.vnC. Mainly used internally to construct new
            # models.
            self.vnC = grid
            self.nC = np.prod(grid)
        else:
            self.nC = grid.nC
            self.vnC = grid.vnC

        # Copies of vnC and nC, but more widely used/known
        # (vnC and nC are the discretize attributes).
        self.shape = tuple(self.vnC)
        self.size = self.nC

        # Check case.
        self.case_names = ['isotropic', 'HTI', 'VTI', 'tri-axial']
        if property_y is None and property_z is None:
            # 0: Isotropic.
            self.case = 0
        elif property_z is None:
            # 1: HTI.
            self.case = 1
        elif property_y is None:
            # 2: VTI.
            self.case = 2
        else:
            # 3: Tri-axial anisotropy.
            self.case = 3

        # Get map.
        self.map = getattr(maps, 'Map'+mapping)()

        # Initiate all parameters.
        self._property_x = self._check_parameter(
                property_x, 'property_x', True)
        self._property_y = self._check_parameter(
                property_y, 'property_y', True)
        self._property_z = self._check_parameter(
                property_z, 'property_z', True)
        self._mu_r = self._check_parameter(mu_r, 'mu_r')
        self._epsilon_r = self._check_parameter(epsilon_r, 'epsilon_r')

    def __repr__(self):
        """Simple representation."""
        return (f"Model [{self.map.description}]; {self.case_names[self.case]}"
                f"{'' if self.mu_r is None else '; mu_r'}"
                f"{'' if self.epsilon_r is None else '; epsilon_r'}"
                f"; {self.vnC[0]} x {self.vnC[1]} x {self.vnC[2]} "
                f"({self.nC:,})")

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
        return Model(grid=np.array(self.vnC), **kwargs)

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
        return Model(grid=np.array(self.vnC), **kwargs)

    def __eq__(self, model):
        """Compare two models.

        Note: Shape of parameters can be different, e.g. float, nC, or vnC. As
              long as all values agree it returns True.

        """

        # Check if model is a Model instance.
        equal = model.__class__.__name__ == 'Model'

        # Check input.
        if equal:
            try:
                _ = self._operator_test(model)
            except ValueError:
                equal = False

        # Compare values.
        if equal:
            equal *= np.allclose(self.property_x, model.property_x)
            equal *= np.allclose(self.property_y, model.property_y)
            equal *= np.allclose(self.property_z, model.property_z)
            # operator_test ensures mu_r are both or neither None.
            if self.mu_r is not None:
                equal *= np.allclose(self.mu_r, model.mu_r)
            # operator_test ensures epsilon_r are both or neither None.
            if self.epsilon_r is not None:
                equal *= np.allclose(self.epsilon_r, model.epsilon_r)

        return bool(equal)

    def copy(self):
        """Return a copy of the Model."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Model in a dict."""
        # Initiate dict.
        out = {}

        # Properties.
        out['property_x'] = self.property_x
        if self.case in [1, 3]:
            out['property_y'] = self.property_y
        else:
            out['property_y'] = None
        if self.case in [2, 3]:
            out['property_z'] = self.property_z
        else:
            out['property_z'] = None

        # mu_r.
        if self.mu_r is not None:
            out['mu_r'] = self.mu_r
        else:
            out['mu_r'] = None

        # epsilon_r.
        if self.epsilon_r is not None:
            out['epsilon_r'] = self.epsilon_r
        else:
            out['epsilon_r'] = None

        # vnC.
        out['vnC'] = np.array(self.vnC)

        # Map.
        out['mapping'] = self.map.name

        # Name
        out['__class__'] = self.__class__.__name__

        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert the dictionary into a Model instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Model.to_dict`.
            The dictionary needs the keys `property_x`, `property_y`,
            `property_z`, `mu_r`, `epsilon_r`, `vnC`, and `mapping`.

        Returns
        -------
        obj : :class:`Model` instance

        """
        try:
            return cls(grid=inp['vnC'], property_x=inp['property_x'],
                       property_y=inp['property_y'],
                       property_z=inp['property_z'], mu_r=inp['mu_r'],
                       epsilon_r=inp['epsilon_r'], mapping=inp['mapping'])
        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.") from e

    # PROPERTY
    @property
    def property_x(self):
        r"""Property in x-direction."""
        return self._return_parameter(self._property_x)

    @property_x.setter
    def property_x(self, property_x):
        r"""Update property in x-direction."""
        self._property_x = self._check_parameter(
                property_x, 'property_x', True)

    @property
    def property_y(self):
        r"""Property in y-direction."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            return self._return_parameter(self._property_y)
        else:                    # Return property_x.
            return self._return_parameter(self._property_x)

    @property_y.setter
    def property_y(self, property_y):
        r"""Update property in y-direction."""

        # Adjust case in case property_z was not set so far.
        if self.case == 0:  # If it was isotropic, it is HTI now.
            self.case = 1
        elif self.case == 2:  # If it was VTI, it is tri-axial now.
            self.case = 3

        # Update it.
        self._property_y = self._check_parameter(
                property_y, 'property_y', True)

    @property
    def property_z(self):
        r"""Property in z-direction."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            return self._return_parameter(self._property_z)
        else:                    # Return property_x.
            return self._return_parameter(self._property_x)

    @property_z.setter
    def property_z(self, property_z):
        r"""Update property in z-direction."""

        # Adjust case in case property_z was not set so far.
        if self.case == 0:  # If it was isotropic, it is VTI now.
            self.case = 2
        elif self.case == 1:  # If it was HTI, it is tri-axial now.
            self.case = 3

        # Update it.
        self._property_z = self._check_parameter(
                property_z, 'property_z', True)

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

    def interpolate2grid(self, grid, new_grid, **grid2grid_opts):
        """Interpolate `Model` located on `grid` to `new_grid`.


        Parameters
        ----------
        grid, new_grid : TensorMesh
            Input and output model grids;
            :class:`emg3d.meshes.TensorMesh` instances.

        grid2grid_opts : dict
            Passed through to :func:`maps.grid2grid`. Defaults are
            `method='volume'`, `log=True`, and `extrapolate=True`.


        Returns
        -------
        NewModel : Model
            New :class:`Model` instance on `new_grid`.

        """

        # Get solver options, set to defaults if not provided.
        inp = {
                'method': 'volume',
                'extrapolate': True,
                'log': not self.map.name.startswith('L'),
                **(grid2grid_opts if grid2grid_opts is not None else {}),
                'grid': grid,
                'new_grid': new_grid
        }

        def ensure_vnc(prop):
            """Expand float-properties to shape vnC."""
            return prop*np.ones(grid.vnC) if prop.size == 1 else prop

        # property_x (always).
        property_x = maps.grid2grid(values=ensure_vnc(self.property_x), **inp)

        # property_y.
        if self.case in [1, 3]:
            property_y = maps.grid2grid(
                    values=ensure_vnc(self.property_y), **inp)
        else:
            property_y = None

        # property_z.
        if self.case in [2, 3]:
            property_z = maps.grid2grid(
                    values=ensure_vnc(self.property_z), **inp)
        else:
            property_z = None

        # mu_r.
        if self._mu_r is not None:
            mu_r = maps.grid2grid(values=ensure_vnc(self.mu_r), **inp)
        else:
            mu_r = None

        # epsilon_r.
        if self._epsilon_r is not None:
            epsilon_r = maps.grid2grid(
                values=ensure_vnc(self.epsilon_r), **inp)
        else:
            epsilon_r = None

        # Assemble coarse model.
        return Model(new_grid, property_x=property_x, property_y=property_y,
                     property_z=property_z, mu_r=mu_r, epsilon_r=epsilon_r,
                     mapping=self.map.name)

    # INTERNAL UTILITIES
    def _check_parameter(self, var, name, mapped=False):
        """Check parameter.

        - Shape must be (), (1,), nC, or vnC.
        - Value(s) must be 0 < var < inf.
        """

        # If None, exit.
        if var is None:
            return None

        # Cast it to floats, ravel.
        var = np.asarray(var, dtype=np.float64).ravel('F')

        # Check for wrong size.
        if var.size not in [1, self.nC]:
            raise ValueError(
                    f"Shape of {name} must be (), {self.vnC}, or "
                    f"{self.nC}.\nProvided: {var.shape}.")

        # Check they are positive.
        if not mapped or (mapped and not self.map.name.startswith('L')):
            if not np.all(var > 0):
                raise ValueError(f"`{name}` must be all `0 < var`.")

        # Check |val| < inf.
        if not np.all(np.abs(var) < np.inf):
            raise ValueError(f"`{name}` must be all `var < inf`.")

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

    def _operator_test(self, model):
        """Check if `self` and `model` are consistent for operations.

        Note: {hx; hy; hz} is not checked. As long as the models have the
              same shape, isotropy, and mapping the operation will be carried
              out.

        """

        # Ensure the two instances have the same dimension.
        if np.any(self.shape != model.shape):
            msg = (f"Models could not be broadcast together with shapes "
                   f"{self.shape} and {model.shape}.")
            raise ValueError(msg)

        # Ensure the two instances have the same case.
        if self.case != model.case:
            msg = ("Models must be of the same isotropy type but have types"
                   f" '{self.case_names[self.case]}' and"
                   f" '{model.case_names[model.case]}'.")
            raise ValueError(msg)

        # Ensure both or none has mu_r:
        if hasattr(self.mu_r, 'dtype') != hasattr(model.mu_r, 'dtype'):
            msg = ("Either both or none of the models must have `mu_r` "
                   f"defined; provided: '{hasattr(self.mu_r, 'dtype')}' "
                   f"and '{hasattr(model.mu_r, 'dtype')}'.")
            raise ValueError(msg)

        # Ensure both or none has epsilon_r:
        if (hasattr(self.epsilon_r, 'dtype') !=
                hasattr(model.epsilon_r, 'dtype')):
            msg = ("Either both or none of the models must have `epsilon_r` "
                   f"defined; provided: '{hasattr(self.epsilon_r, 'dtype')}' "
                   f"and '{hasattr(model.epsilon_r, 'dtype')}'.")
            raise ValueError(msg)

        # Ensure the two instances have the same mapping:
        if self.map.name != model.map.name:
            msg = ("Models must have the same mapping but have mappings"
                   f" '{self.map.name}' and '{model.map.name}'.")
            raise ValueError(msg)

    def _apply_operator(self, model, operator):
        """Apply the provided operator to self and model."""

        kwargs = {}

        # Subtract properties.
        kwargs['property_x'] = operator(self.property_x, model.property_x)
        if self.case in [1, 3]:
            kwargs['property_y'] = operator(self.property_y, model.property_y)
        else:
            kwargs['property_y'] = None
        if self.case in [2, 3]:
            kwargs['property_z'] = operator(self.property_z, model.property_z)
        else:
            kwargs['property_z'] = None

        # Subtract mu_r.
        if self.mu_r is not None:
            kwargs['mu_r'] = operator(self.mu_r, model.mu_r)
        else:
            kwargs['mu_r'] = None

        # Subtract epsilon_r.
        if self.epsilon_r is not None:
            kwargs['epsilon_r'] = operator(self.epsilon_r, model.epsilon_r)
        else:
            kwargs['epsilon_r'] = None

        kwargs['mapping'] = self.map.name

        return kwargs


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
        self._eta_x = self.calculate_eta('property_x', grid, model, sfield)

        # eta_y
        if model.case in [1, 3]:  # HTI or tri-axial.
            self._eta_y = self.calculate_eta('property_y', grid, model, sfield)

        # eta_z
        if self.case in [2, 3]:  # VTI or tri-axial.
            self._eta_z = self.calculate_eta('property_z', grid, model, sfield)

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
        r"""eta: volume multiplied with conductivity."""

        # Initiate eta
        eta = field.smu0*grid.cell_volumes.reshape(grid.vnC, order='F')

        # Compute eta depending on epsilon.
        if model.epsilon_r is None:  # Diffusive approximation.
            eta *= model.map.backward(getattr(model, name))

        else:
            eps_term = field.sval*epsilon_0*model.epsilon_r
            sig_term = model.map.backward(getattr(model, name))
            eta *= sig_term - eps_term

        return eta

    @staticmethod
    def calculate_zeta(name, grid, model):
        r"""zeta: volume divided by mu_r."""

        zeta = grid.cell_volumes.reshape(grid.vnC, order='F')
        if getattr(model, name, None) is None:
            return zeta

        else:
            return zeta/getattr(model, name)
