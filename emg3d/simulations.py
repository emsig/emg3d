"""

:mod:`simulation` -- Model a survey
===================================

A simulation is the computation (modelling) of electromagnetic responses of a
resistivity model for a given survey.

In its heart, `emg3d` is a multigrid solver for 3D electromagnetic diffusion
with tri-axial electrical anisotropy. However, it contains most functionalities
to also act as a modeller. The simulation module combines all these things
by combining surveys with computational meshes and fields and providing
high-level, specialised modelling routines.

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


# import numpy as np

from emg3d import fields, solver

__all__ = ['Simulation']


class Simulation():
    r"""Create a simulation for a given survey on a given model.

    The computational mesh(es) can be either the same as the model mesh, or
    they can be provided, or automatic gridding can be applied.

    .. todo::

        - Take care if `'return_info': True` for `solver_opts`.
        - gridding options
        - min_amp to consider
        - min_offset, max_offset
        - NOTHING with inversion
        - make synthetic data
        - dpred
        - dobs
        - compute E-Source (H-source)  [forward] => synthetic
        - get E-field (H-field)        [interpolation]
        - gradient, residual, misfit


    Parameters
    ----------
    survey : :class:`emg3d.survey.Survey`
        The survey layout, containing sources, receivers, and frequencies.

    grid : :class:`emg3d.meshes.TensorMesh`
        The grid. See :class:`emg3d.meshes.TensorMesh`.

    model : :class:`emg3d.models.Model`
        The model. See :class:`emg3d.models.Model`.

    solver_opts : dict
        Passed through to :func:`emg3d.solver.solve`. The dict can contain any
        parameter that is accepted by the solver except for `grid`, `model`,
        `sfield`, and `efield`.

    comp_grids : str, dict, or  :class:`emg3d.meshes.TensorMesh`
        Computational grids. The possibilities are:

        - A string:

            - 'same': Same grid as for model.
            - 'single': A single grid for all sources and frequencies.
            - 'frequency': Frequency-dependent grids.
            - 'source': Source-dependent grids.
            - 'both': Frequency- and source-dependent grids.

            Except for 'same', the grids are created using the adaptive
            gridding routine :func:`emg3d.meshes.csem_model`.

        - A dict: If a dict is provided the keys must be the source names
          and/or the frequencies, and the values are
          :class:`emg3d.meshes.TensorMesh` instances. The structure of the
          dict can be:

            - `dict[freq]`: corresponds to 'frequency'.
            - `dict[source]`: corresponds to 'source'.
            - `dict[source][freq]`: corresponds to 'both'.

        - A :class:`emg3d.meshes.TensorMesh` instance. This is the same as
          'single', but the provided grid is used instead of the adaptive
          gridding routine.

        Default is 'same', hence the modelling grid is used for computation.

    """

    def __init__(self, survey, grid, model, solver_opts=None,
                 comp_grids='same', **kwargs):
        """Initiate a new Simulation instance."""

        # Store inputs (should these be copied?).
        self.survey = survey
        self.grid = grid
        self.model = model

        if solver_opts is None:
            solver_opts = {
                    'sslsolver': True,
                    'semicoarsening': True,
                    'linerelaxation': True,
                    'verb': -1,
                    }
        self.solver_opts = solver_opts

        # Ensure no kwargs left (currently kwargs is not used).
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

        # Initiate comp_grids-, model-, sfield-, and efield-dicts.
        self._comp_grids = {src: {freq: None for freq in survey.frequencies}
                            for src in survey.sources.keys()}
        self._comp_models = {src: {freq: None for freq in survey.frequencies}
                             for src in survey.sources.keys()}
        self._sfields = {src: {freq: None for freq in survey.frequencies}
                         for src in survey.sources.keys()}
        self._efields = {src: {freq: None for freq in survey.frequencies}
                         for src in survey.sources.keys()}

        # Take care of `comp_grids`; for consistency, it is always a dict with
        # the structure dict[source][freq]; no copies are made of same meshes,
        # just pointers.
        if isinstance(comp_grids, str):

            # Store comp-type.
            self._comp_grids_type = comp_grids

            # Act depending on string.
            if comp_grids == 'same':  # Store same grid for all cases.
                for src, val in self._comp_grids.items():
                    for freq in self._comp_grids[src].keys():
                        self._comp_grids[src][freq] = self.grid

            else:
                # Need to implement adaptive gridding for:
                # 'single', 'frequency', 'source', 'both'
                raise NotImplementedError(f"`comp_dicts` {comp_grids}")
        else:
            # Need to implement: dict, TensorMesh
            raise NotImplementedError(f"`comp_dicts` {type(grid)}")

    def copy(self):
        """Return a copy of the Simulation."""
        return self.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the Simulation in a dict."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Simulation` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Simulation.to_dict`.
            The dictionary needs the keys TODO.

        Returns
        -------
        obj : :class:`Simulation` instance

        """
        raise NotImplementedError

    def comp_grids(self, source, frequency):
        """Return computational grid of the given source and frequency."""
        return self._comp_grids[source][float(frequency)]

    def comp_models(self, source, frequency):
        """Return model on the grid of the given source and frequency."""

        # If model is not stored yet, get it.
        if self._comp_models[source][float(frequency)] is None:
            if self._comp_grids_type == 'same':
                self._comp_models[source][float(frequency)] = self.model
            else:
                raise NotImplementedError

        return self._comp_models[source][float(frequency)]

    def sfields(self, source, frequency):
        """Return source field for given source and frequency."""

        # If source field is not stored yet, get it.
        if self._sfields[source][float(frequency)] is None:
            sfield = fields.get_source_field(
                    self.comp_grids(source, frequency),
                    self.survey.sources[source].coordinates,
                    frequency, strength=0)
            self._sfields[source][float(frequency)] = sfield

        return self._sfields[source][float(frequency)]

    def efields(self, source, frequency, **kwargs):
        """Return electric field for given source and frequency.

        The efield is computed if it is not stored already, except if
        `recalc=True` is in `kwargs`. All other `kwargs` are passed to solver,
        overwriting `self.solver_opts`.

        """
        recalc = kwargs.pop('recalc', False)

        # If electric field not computed yet compute it.
        if self._efields[source][float(frequency)] is None or recalc:

            # Get solver options and update with kwargs.
            solver_opts = {**self.solver_opts, **kwargs}

            # Compute electric field.
            efield = solver.solve(
                    self.comp_grids(source, frequency),
                    self.comp_models(source, frequency),
                    self.sfields(source, frequency),
                    **solver_opts)

            # Store electric field.
            self._efields[source][float(frequency)] = efield

        return self._efields[source][float(frequency)]


def model_marine_csem():
    # => MOVE TO :mod:`emg3d.meshes`
    # JUST adaptive gridding, modelling is done by simulation class.
    # takes a model; fills up water if req., adds air
    # takes a survey -> deduces computational domain from that
    # takes gridding parameters
    pass
