"""

:mod:`models` -- Earth properties
=================================

Everything to create model-properties for the multigrid solver.

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


import numpy as np
from copy import deepcopy
from scipy.constants import epsilon_0

__all__ = ['Model', 'VolumeModel']


# MODEL
class Model:
    r"""Create a model instance.

    Class to provide model parameters (x-, y-, and z-directed resistivities,
    electric permittivity and magnetic permeability) to the solver. Relative
    magnetic permeability :math:`\mu_\mathrm{r}` is by default set to one and
    electric permittivity :math:`\varepsilon_\mathrm{r}` is by default set to
    zero, but they can also be provided (isotropically). Keep in mind that the
    multigrid method as implemented in `emg3d` only works for the diffusive
    approximation. As soon as the displacement-part in the Maxwell's equations
    becomes too dominant it will fail (high frequencies or very high electric
    permittivity).


    Parameters
    ----------
    grid : TensorMesh
        Grid on which to apply model.

    res_x, res_y, res_z : float or ndarray; default to 1.
        Resistivity in x-, y-, and z-directions. If ndarray, they must have the
        shape of grid.vnC (F-ordered) or grid.nC. Resistivities have to be
        bigger than zero and smaller than infinity.

    mu_r : None, float, or ndarray
        Relative magnetic permeability (isotropic). If ndarray it must have the
        shape of grid.vnC (F-ordered) or grid.nC. Default is None, which
        corresponds to 1., but avoids the calculation of zeta. Magnetic
        permeability has to be bigger than zero and smaller than infinity.

    epsilon_r : None, float, or ndarray
       Relative electric permittivity (isotropic). If ndarray it must have the
       shape of grid.vnC (F-ordered) or grid.nC. The displacement part is
       completely neglected (diffusive approximation) if set to None, which is
       the default. Electric permittivity has to be bigger than zero and
       smaller than infinity.

    """

    def __init__(self, grid, res_x=1., res_y=None, res_z=None, mu_r=None,
                 epsilon_r=None):
        """Initiate a new model."""

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
        if res_y is None and res_z is None:  # 0: Isotropic.
            self.case = 0
        elif res_z is None:                  # 1: HTI.
            self.case = 1
        elif res_y is None:                  # 2: VTI.
            self.case = 2
        else:                                # 3: Tri-axial anisotropy.
            self.case = 3

        # Initiate all parameters.
        self._res_x = self._check_parameter(res_x, 'res_x')
        self._res_y = self._check_parameter(res_y, 'res_y')
        self._res_z = self._check_parameter(res_z, 'res_z')
        self._mu_r = self._check_parameter(mu_r, 'mu_r')
        self._epsilon_r = self._check_parameter(epsilon_r, 'epsilon_r')

    def __repr__(self):
        """Simple representation."""
        return (f"Model; {self.case_names[self.case]} resistivities"
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
        return Model(grid=self.vnC, **kwargs)

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
        return Model(grid=self.vnC, **kwargs)

    def __eq__(self, model):
        """Compare two models.

        Note: Shape of parameters can be different, e.g. float, nC, or vnC. As
              long as all values agree it returns True.

        """

        # Ensure model is a Model instance.
        if model.__class__.__name__ != 'Model':
            return NotImplemented

        # Check input.
        try:
            _ = self._operator_test(model)
            equal = True
        except ValueError:
            equal = False

        # Compare resistivities.
        equal *= np.all(self.res_x == model.res_x)
        equal *= np.all(self.res_y == model.res_y)
        equal *= np.all(self.res_z == model.res_z)
        equal *= np.all(self.mu_r == model.mu_r)
        equal *= np.all(self.epsilon_r == model.epsilon_r)

        return equal

    def copy(self):
        """Return a copy of the Model."""
        return Model.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Model in a dict."""
        # Initiate dict.
        out = {}

        # resistivities.
        out['res_x'] = self.res_x
        if self.case in [1, 3]:
            out['res_y'] = self.res_y
        else:
            out['res_y'] = None
        if self.case in [2, 3]:
            out['res_z'] = self.res_z
        else:
            out['res_z'] = None

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
        out['vnC'] = self.vnC

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
            The dictionary needs the keys `res_x`, `res_y`, `res_z`, `mu_r`,
            `epsilon_r`, and `vnC`.

        Returns
        -------
        obj : :class:`Model` instance

        """
        try:
            return cls(grid=inp['vnC'], res_x=inp['res_x'], res_y=inp['res_y'],
                       res_z=inp['res_z'], mu_r=inp['mu_r'],
                       epsilon_r=inp['epsilon_r'])
        except KeyError as e:
            print(f"* ERROR   :: Variable {e} missing in `inp`.")
            raise

    # RESISTIVITIES
    @property
    def res_x(self):
        r"""Resistivity in x-direction."""
        return self._return_parameter(self._res_x)

    @res_x.setter
    def res_x(self, res):
        r"""Update resistivity in x-direction."""
        self._res_x = self._check_parameter(res, 'res_x')

    @property
    def res_y(self):
        r"""Resistivity in y-direction."""
        if self.case in [1, 3]:  # HTI or tri-axial.
            return self._return_parameter(self._res_y)
        else:                    # Return res_x.
            return self._return_parameter(self._res_x)

    @res_y.setter
    def res_y(self, res):
        r"""Update resistivity in y-direction."""

        # Adjust case in case res_z was not set so far.
        if self.case == 0:  # If it was isotropic, it is HTI now.
            self.case = 1
        elif self.case == 2:  # If it was VTI, it is tri-axial now.
            self.case = 3

        # Update it.
        self._res_y = self._check_parameter(res, 'res_y')

    @property
    def res_z(self):
        r"""Resistivity in z-direction."""
        if self.case in [2, 3]:  # VTI or tri-axial.
            return self._return_parameter(self._res_z)
        else:                    # Return res_x.
            return self._return_parameter(self._res_x)

    @res_z.setter
    def res_z(self, res):
        r"""Update resistivity in z-direction."""

        # Adjust case in case res_z was not set so far.
        if self.case == 0:  # If it was isotropic, it is VTI now.
            self.case = 2
        elif self.case == 1:  # If it was HTI, it is tri-axial now.
            self.case = 3

        # Update it.
        self._res_z = self._check_parameter(res, 'res_z')

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

    # INTERNAL UTILITIES
    def _check_parameter(self, var, name):
        """Check parameter.

        - Shape must be (), (1,), nC, or vnC.
        - Value(s) must be 0 < var < inf.
        """

        # If None, exit.
        if var is None:
            return None

        # Cast it to floats, ravel.
        var = np.asarray(var, dtype=float).ravel('F')

        # Check for wrong size.
        if var.size not in [1, self.nC]:
            print(f"* ERROR   :: Shape of {name} must be (), {self.vnC}, or "
                  f"{self.nC}.\n             Provided: {var.shape}.")
            raise ValueError("Wrong Shape")

        # Check 0 < val or 0 <= val.
        if not np.all(var > 0):
            print(f"* ERROR   :: `{name}` must be all `0 < var`.")
            raise ValueError("Parameter error")

        # Check val < inf.
        if not np.all(var < np.inf):
            print(f"* ERROR   :: `{name}` must be all `var < inf`.")
            raise ValueError("Parameter error")

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
              same shape and resistivity type the operation will be carried
              out.

        """

        # Ensure the two instances have the same dimension.
        if np.any(self.shape != model.shape):
            msg = (f"Models could not be broadcast together with shapes "
                   f"{self.shape} and {model.shape}.")
            raise ValueError(msg)

        # Ensure the two instances have the same case.
        if self.case != model.case:
            msg = ("Models must be of the same resistivity type but have types"
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

    def _apply_operator(self, model, operator):
        """Apply the provided operator to self and model."""

        kwargs = {}

        # Subtract resistivities.
        kwargs['res_x'] = operator(self.res_x, model.res_x)
        if self.case in [1, 3]:
            kwargs['res_y'] = operator(self.res_y, model.res_y)
        else:
            kwargs['res_y'] = None
        if self.case in [2, 3]:
            kwargs['res_z'] = operator(self.res_z, model.res_z)
        else:
            kwargs['res_z'] = None

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
        self._eta_x = self.calculate_eta('res_x', grid, model, sfield)

        # eta_y
        if model.case in [1, 3]:  # HTI or tri-axial.
            self._eta_y = self.calculate_eta('res_y', grid, model, sfield)

        # eta_z
        if self.case in [2, 3]:  # VTI or tri-axial.
            self._eta_z = self.calculate_eta('res_z', grid, model, sfield)

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
        r"""eta: volume divided by resistivity."""

        # Initiate eta
        eta = field.smu0*grid.vol.reshape(grid.vnC, order='F')

        # Calculate eta depending on epsilon.
        if model.epsilon_r is None:  # Diffusive approximation.
            eta /= getattr(model, name)

        else:
            eps_term = field.sval*epsilon_0*model.epsilon_r
            sig_term = 1./getattr(model, name)
            eta *= sig_term - eps_term

        return eta

    @staticmethod
    def calculate_zeta(name, grid, model):
        r"""zeta: volume divided by mu_r."""

        if getattr(model, name, None) is None:
            return grid.vol.reshape(grid.vnC, order='F')

        else:
            return grid.vol.reshape(grid.vnC, order='F')/getattr(model, name)
