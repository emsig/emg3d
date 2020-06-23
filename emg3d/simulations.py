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


import numpy as np
from concurrent import futures

from emg3d import fields, solver

__all__ = ['Simulation']


class Simulation():
    r"""Create a simulation for a given survey on a given model.

    The computational mesh(es) can be either the same as the provided model
    mesh, or automatic gridding can be used.

    .. todo::

        - Properly test.
        - Adaptive gridding.
        - Make synthetic data; dpred; dobs.
        - `to_dict`, `from_dict`.
        - Include logging/verbosity; check with CLI.
        - Check what not implemented:

          - Finite length dipoles for psolve.
          - H-fields for psolve.

        - NOTHING with inversion.

          - Gradient, residual, misfit.
          - `min_amp`, `min_offset`, `max_offset`.


    Parameters
    ----------
    survey : :class:`emg3d.survey.Survey`
        The survey layout, containing sources, receivers, and frequencies.

    grid : :class:`emg3d.meshes.TensorMesh`
        The grid. See :class:`emg3d.meshes.TensorMesh`.

    model : :class:`emg3d.models.Model`
        The model. See :class:`emg3d.models.Model`.

    solver_opts : dict
        Passed through to :func:`emg3d.solve`. The dict can contain any
        parameter that is accepted by the :func:`emg3d.solve` except for
        `grid`, `model`, `sfield`, and `efield`.
        If not provided the following defaults are used:
        `sslsolver = semicoarsening = linerelaxation = True`, `verb = -1`.

    comp_grids : str
        Computational grids; default is currently 'same', but will change to
        'single'.

        - 'same': Same grid as for model.
        - 'single': A single grid for all sources and frequencies.
        - 'frequency': Frequency-dependent grids.
        - 'source': Source-dependent grids.
        - 'both': Frequency- and source-dependent grids.

        Except for 'same', the grids are created using the adaptive gridding
        routine :func:`emg3d.meshes.csem_model`.

        Not implemented yet is the possibility to provide a single TensorMesh
        instance or a dict of TensorMesh instances for user-provided meshes.
        You can still do this by setting `simulation._comp_meshes` after
        instantiation.

    """

    def __init__(self, survey, grid, model, solver_opts=None,
                 comp_grids='same', **kwargs):
        """Initiate a new Simulation instance."""

        # Store inputs (should these be copied to avoid altering them?).
        self.survey = survey
        self.grid = grid
        self.model = model
        self._grids_type = comp_grids

        # Get solver options, set to defaults if not provided.
        self.solver_opts = {
                # Defaults; overwritten by inputs v.
                'sslsolver': True,
                'semicoarsening': True,
                'linerelaxation': True,
                'verb': -1,
                **(solver_opts if solver_opts is not None else {}),
                }

        # Ensure no kwargs left (currently kwargs is not used).
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

        # Initiate comp_{grids;models}-dicts and {s;e;h}fields-dicts.
        self._comp_grids = self._initiate_none_dict()
        self._comp_models = self._initiate_none_dict()
        self._sfields = self._initiate_none_dict()
        self._efields = self._initiate_none_dict()
        self._hfields = self._initiate_none_dict()
        self._solver_info = self._initiate_none_dict()

        # Initialize synthetic data.
        self.survey._ds['synthetic'] = self.survey.data*np.nan

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
        freq = float(frequency)

        # Get grid if it is not stored already.
        if self._comp_grids[source][freq] is None:

            # Act depending on grids_type:
            if self._grids_type == 'same':  # Same grid as for provided model.

                # Store link to grid.
                self._comp_grids[source][freq] = self.grid

            elif self._grids_type == 'frequency':  # Frequency-dependent grids.

                # Initiate dict.
                if not hasattr(self, '_frequency_grid'):
                    self._frequency_grid = {}

                # Get grid for this frequency if not yet computed.
                if freq not in self._frequency_grid.keys():
                    self._frequency_grid[freq] = None  # TODO adaptive grid
                    raise NotImplementedError("Adaptive gridding")

                # Store link to grid.
                self._comp_grids[source][freq] = self._frequency_grid[freq]

            elif self._grids_type == 'source':  # Source-dependent grids.

                # Initiate dict.
                if not hasattr(self, '_source_grid'):
                    self._source_grid = {}

                # Get grid for this source if not yet computed.
                if source not in self._source_grid.keys():
                    self._source_grid[source] = None  # TODO adaptive grid
                    raise NotImplementedError("Adaptive gridding")

                # Store link to grid.
                self._comp_grids[source][freq] = self._source_grid[source]

            elif self._grids_type == 'both':  # Src- & freq-dependent grids.

                # Get grid and store it.
                self._comp_grids[source][freq] = None  # TODO adaptive grid
                raise NotImplementedError("Adaptive gridding")

            else:  # Use a single grid for all sources and receivers.
                # Default case; catches 'single' but also anything else.

                # Get grid if not yet computed.
                if not hasattr(self, '_single_grid'):
                    self._single_grid = None  # TODO adaptive grid
                    raise NotImplementedError("Adaptive gridding")

                # Store link to grid.
                self._comp_grids[source][freq] = self._single_grid

        return self._comp_grids[source][freq]

    def comp_models(self, source, frequency):
        """Return model on the grid of the given source and frequency."""
        freq = float(frequency)

        # If model is not stored yet, get it.
        if self._comp_models[source][freq] is None:

            # Act depending on grids_type:
            if self._grids_type == 'same':  # Same grid as for provided model.

                # Store link to model.
                self._comp_models[source][freq] = self.model

            elif self._grids_type == 'frequency':  # Frequency-dependent grids.

                # Initiate dict.
                if not hasattr(self, '_frequency_model'):
                    self._frequency_model = {}

                # Get model for this frequency if not yet computed.
                if freq not in self._frequency_model.keys():
                    self._frequency_model[freq] = self.model.interpolate2grid(
                            self.grid, self.comp_grids(source, freq))

                # Store link to model.
                self._comp_models[source][freq] = self._frequency_model[freq]

            elif self._grids_type == 'source':  # Source-dependent grids.

                # Initiate dict.
                if not hasattr(self, '_source_model'):
                    self._source_model = {}

                # Get model for this source if not yet computed.
                if source not in self._source_model.keys():
                    self._source_model[freq] = self.model.interpolate2grid(
                            self.grid, self.comp_grids(source, freq))

                # Store link to model.
                self._comp_models[source][freq] = self._source_model[source]

            elif self._grids_type == 'both':  # Src- & freq-dependent grids.

                # Get model and store it.
                self._comp_models[source][freq] = self.model.interpolate2grid(
                            self.grid, self.comp_grids(source, freq))

            else:  # Use a single grid for all sources and receivers.
                # Default case; catches 'single' but also anything else.

                # Get model if not yet computed.
                if not hasattr(self, '_single_model'):
                    self._single_model = self.model.interpolate2grid(
                            self.grid, self.comp_grids(source, freq))

                # Store link to model.
                self._comp_models[source][freq] = self._single_model

        return self._comp_models[source][freq]

    def sfields(self, source, frequency):
        """Return source field for given source and frequency."""
        freq = float(frequency)

        # If source field is not stored yet, get it.
        if self._sfields[source][freq] is None:
            sfield = fields.get_source_field(
                    self.comp_grids(source, frequency),
                    self.survey.sources[source].coordinates,
                    frequency, strength=0)
            self._sfields[source][freq] = sfield

        return self._sfields[source][freq]

    def efields(self, source, frequency, **kwargs):
        """Return electric field for given source and frequency.

        The efield is only computed if it is not stored already, except if
        `recalc=True` is in `kwargs`. All other `kwargs` are passed to
        :func:`emg3d.solve`, overwriting `self.solver_opts`.

        Parameters
        ----------
        source : str
            Source name.

        frequency : float
            Frequency

        kwargs : dict
            Passed to :func:`solver.solve`.


        Returns
        -------
        efield : :class:`emg3d.fields.Field`
            Resulting electric field.

        info_dict : dict
            Dictionary with runtime info; only if ``return_info=True`` was
            provided in `kwargs`.

        """
        freq = float(frequency)
        recalc = kwargs.pop('recalc', False)
        return_info = kwargs.get('return_info', False)

        # If electric field not computed yet compute it.
        if self._efields[source][freq] is None or recalc:

            # Get solver options and update with kwargs.
            solver_opts = {**self.solver_opts, **kwargs}

            # Compute electric field.
            efield = solver.solve(
                    self.comp_grids(source, frequency),
                    self.comp_models(source, frequency),
                    self.sfields(source, frequency),
                    **solver_opts)

            # Store electric field.
            if return_info:
                self._efields[source][freq] = efield[0]
                self._solver_info[source][freq] = efield[1]
            else:
                self._efields[source][freq] = efield

            # Clean corresponding hfield, so it will be recalculated.
            del self._hfields[source][freq]
            self._hfields[source][freq] = None

        if return_info:
            return (self._efields[source][freq],
                    self._solver_info[source][freq])
        else:
            return self._efields[source][freq]

    def hfields(self, source, frequency, **kwargs):
        """Return magnetic field for given source and frequency.

        The hfield is only computed from the efield if it is not stored
        already, and so is the efield, except if `recalc=True` is in `kwargs`.
        All other `kwargs` are passed to :func:`emg3d.solve`, overwriting
        `self.solver_opts`.


        Parameters
        ----------
        source : str
            Source name.

        frequency : float
            Frequency

        kwargs : dict
            Passed to :func:`solver.solve`.


        Returns
        -------
        hfield : Field
            Magnetic field; :class:`Field` instance.

        """
        freq = float(frequency)
        recalc = kwargs.get('recalc', False)
        return_info = kwargs.get('return_info', False)

        # If electric field not computed yet compute it.
        if self._efields[source][freq] is None or recalc:
            _ = self.efields(source, frequency, **kwargs)

        # If magnetic field not computed yet compute it.
        if self._hfields[source][freq] is None or recalc:
            hfield = fields.get_h_field(
                    self.comp_grids(source, frequency),
                    self.comp_models(source, frequency),
                    self.efields(source, frequency))
            self._hfields[source][freq] = hfield

        if return_info:
            return (self._hfields[source][freq],
                    self._solver_info[source][freq])
        else:
            return self._hfields[source][freq]

    @property
    def synthetic(self):
        """Synthetic data, an :class:`xarray.DataArray` instance.."""
        return self.survey.ds.synthetic

    def psolve(self, nproc=4, **kwargs):
        """Compute efields asynchronously for all sources and frequencies.

        Parameters
        ----------
        nproc : int
            Maximum number of processes that can be used. Default is four.

        kwargs : dict
            Passed to :func:`emg3d.solver.solve`; can contain any of the
            arguments of the solver except `grid`, `model`, `sfield`, and
            `efield`.


        Returns
        -------
        info : dict
            Dictionary of the form `dict_name[source][frequency]`, containing
            the solver-info; only returned if `kwargs['return_info']=True`.

        """

        # Initiate futures-dict to store output.
        out = self._initiate_none_dict()

        # Context manager for futures.
        with futures.ProcessPoolExecutor(nproc) as executor:

            # Loop over source positions.
            for src in out.keys():

                # Loop over frequencies.
                for freq in out[src].keys():

                    # Call emg3d
                    out[src][freq] = executor.submit(
                        solver.solve,
                        grid=self.comp_grids(src, freq),
                        model=self.comp_models(src, freq),
                        sfield=self.sfields(src, freq),
                        **kwargs,
                    )

        # Clean hfields, so they will be recalculated.
        del self._hfields
        self._hfields = self._initiate_none_dict()

        # Extract the result(s).
        if kwargs.get('return_info', False):
            self._efields = {f: {k: v.result()[0] for k, v in s.items()}
                             for f, s in out.items()}

            # Return info.
            self._solver_info = {f: {k: v.result()[1] for k, v in s.items()}
                                 for f, s in out.items()}
        else:
            self._efields = {f: {k: v.result() for k, v in s.items()}
                             for f, s in out.items()}

        # Extract data at receivers.
        if self.survey.fixed:
            all_rec_coords = self.survey.rec_coords
        else:
            rec_coords = self.survey.rec_coords
        # Loop over sources and frequencies.
        for src in self.survey.sources.keys():
            if self.survey.fixed:
                rec_coords = all_rec_coords[src]
            for freq in self.survey.frequencies:
                resp = fields.get_receiver_response(
                        grid=self._comp_grids[src][freq],
                        field=self._efields[src][freq],
                        rec=rec_coords
                )
                self.survey._ds['synthetic'].loc[src, :, freq] = resp

    def clean(self):
        """Remove computed fields and corresponding data."""
        for name in ['efields', 'hfields', 'solver_info']:
            delattr(self, '_'+name)
            setattr(self, '_'+name, self._initiate_none_dict())

    def _initiate_none_dict(self):
        """Returns a dict of the structure `dict[source][freq]=None`."""
        return {src: {freq: None for freq in self.survey.frequencies}
                for src in self.survey.sources.keys()}
