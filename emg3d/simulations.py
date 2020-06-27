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

from emg3d import fields, solver, models, meshes

__all__ = ['Simulation']


class Simulation():
    r"""Create a simulation for a given survey on a given model.

    The computational grid(s) can be either the same as the provided model
    grid, or automatic gridding can be used.


    Parameters
    ----------
    survey : :class:`emg3d.survey.Survey`
        The survey layout, containing sources, receivers, and frequencies.

    grid : :class:`meshes.TensorMesh`
        The grid. See :class:`meshes.TensorMesh`.

    model : :class:`emg3d.models.Model`
        The model. See :class:`emg3d.models.Model`.

    solver_opts : dict
        Passed through to :func:`emg3d.solve`. The dict can contain any
        parameter that is accepted by the :func:`emg3d.solve` except for
        `grid`, `model`, `sfield`, and `efield`.
        If not provided the following defaults are used:
        `sslsolver = semicoarsening = linerelaxation = True`, `verb = -1`.

    comp_grids : str
        Computational grids; default is 'single'.

        - 'same': Same grid as for model.
        - 'single': A single grid for all sources and frequencies.
        - 'frequency': Frequency-dependent grids.
        - 'source': Source-dependent grids.
        - 'both': Frequency- and source-dependent grids.

        Except for 'same', the grids are created using the adaptive gridding
        routine :func:`emg3d.meshes.csem_model`.

        Not implemented yet is the possibility to provide a single TensorMesh
        instance or a dict of TensorMesh instances for user-provided meshes.
        You can still do this by setting `simulation._comp_grids` after
        instantiation.

    """

    def __init__(self, survey, grid, model, solver_opts=None,
                 comp_grids='single', **kwargs):
        """Initiate a new Simulation instance."""

        # Store inputs (should these be copied to avoid altering them?).
        self.survey = survey

        # Magnetic dipoles are not yet implemented in simulation.
        if sum([not r.electric for r in survey.receivers.values()]) > 0:
            raise TypeError(
                    "Simulation not yet implemented for magnetic dipoles.")

        # Get gridding options, set to defaults if not provided.
        grid_opts = kwargs.pop('grid_opts', {})
        self.grid_opts = {
                # Defaults; overwritten by inputs v.
                'type': 'marine',
                'air_resistivity': 1e8,
                'res': np.array([0.3, 1, 1e5]),
                'min_width': np.array([200., 200., 100]),
                'zval': np.array([-1000, -2000, 0]),
                'verb': 1,
                **grid_opts,
                }
        self._initiate_model_grid(model, grid)
        self._grids_type = comp_grids
        self._grids_type_descr = {
                'same': 'Same grid as for model',
                'single': 'A single grid for all sources and frequencies',
                'frequency': 'Frequency-dependent grids',
                'source': 'Source-dependent grids',
                'both': 'Frequency- and source-dependent grids',
                }

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
        self.survey._data['synthetic'] = self.survey.data.observed*np.nan

    def __repr__(self):
        return (f"*{self.__class__.__name__}* of «{self.survey.name}»\n\n"
                f"- {self.survey.__class__.__name__}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies\n"
                f"- {self.model.__repr__()}\n"
                f"- Gridding: {self._grids_type_descr[self._grids_type]}")

    def _repr_html_(self):
        return (f"<h3>{self.__class__.__name__}</h3>"
                f"of «{self.survey.name}»<ul>"
                f"<li>{self.survey.__class__.__name__}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies</li>"
                f"<li>{self.model.__repr__()}</li>"
                f"<li>Gridding: "
                f"{self._grids_type_descr[self._grids_type]}</li>"
                f"</ul>")

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

                    # Get grid for this frequency.
                    # TODO does not work for fixed surveys
                    coords = [np.r_[self.survey.src_coords[i],
                              self.survey.rec_coords[i]] for i in range(3)]

                    # Get grid and store it.
                    self._frequency_grid[freq] = meshes.marine_csem_mesh(
                            coords, freq=freq, **self.grid_opts)

                # Store link to grid.
                self._comp_grids[source][freq] = self._frequency_grid[freq]

            elif self._grids_type == 'source':  # Source-dependent grids.

                # Initiate dict.
                if not hasattr(self, '_source_grid'):
                    self._source_grid = {}

                # Get grid for this source if not yet computed.
                if source not in self._source_grid.keys():

                    # Get grid for this frequency.
                    # TODO does not work for fixed surveys
                    coords = [np.r_[self.survey.sources[source].coordinates[i],
                              self.survey.rec_coords[i]] for i in range(3)]

                    # Use average frequency (log10).
                    mfreq = 10**np.mean(np.log10(self.survey.frequencies))

                    # Get grid and store it.
                    self._source_grid[source] = meshes.marine_csem_mesh(
                            coords, freq=mfreq, **self.grid_opts)

                # Store link to grid.
                self._comp_grids[source][freq] = self._source_grid[source]

            elif self._grids_type == 'both':  # Src- & freq-dependent grids.

                # Get grid for this frequency.
                # TODO does not work for fixed surveys
                coords = [np.r_[self.survey.sources[source].coordinates[i],
                          self.survey.rec_coords[i]] for i in range(3)]

                # Get grid and store it.
                self._comp_grids[source][freq] = meshes.marine_csem_mesh(
                        coords, freq=freq, **self.grid_opts)

            else:  # Use a single grid for all sources and receivers.
                # Default case; catches 'single' but also anything else.

                # Get grid if not yet computed.
                if not hasattr(self, '_single_grid'):

                    # Get grid for this frequency.
                    # TODO does not work for fixed surveys
                    coords = [np.r_[self.survey.src_coords[i],
                              self.survey.rec_coords[i]] for i in range(3)]

                    # Use average frequency (log10).
                    mfreq = 10**np.mean(np.log10(self.survey.frequencies))

                    # Get grid and store it.
                    self._single_grid = meshes.marine_csem_mesh(
                            coords, freq=mfreq, **self.grid_opts)

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
        `recomp=True` is in `kwargs`. All other `kwargs` are passed to
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

        """
        freq = float(frequency)
        recomp = kwargs.pop('recomp', False)
        call_from_psolve = kwargs.pop('call_from_psolve', False)

        # If electric field not computed yet compute it.
        if self._efields[source][freq] is None or recomp:

            # Get solver options and update with kwargs.
            solver_opts = {**self.solver_opts, **kwargs}
            solver_opts['return_info'] = True  # Always return solver info.

            # Compute electric field.
            efield, info = solver.solve(
                    self.comp_grids(source, freq),
                    self.comp_models(source, freq),
                    self.sfields(source, freq),
                    **solver_opts)

            # Store electric field and info.
            self._efields[source][freq] = efield
            self._solver_info[source][freq] = info

            # Clean corresponding hfield, so it will be recomputed.
            del self._hfields[source][freq]
            self._hfields[source][freq] = None

            # Extract data at receivers.
            if self.survey.fixed:
                rec_coords = self.survey.rec_coords[source]
            else:
                rec_coords = self.survey.rec_coords

            # Loop over sources and frequencies.
            resp = fields.get_receiver_response(
                    grid=self._comp_grids[source][freq],
                    field=self._efields[source][freq],
                    rec=rec_coords
            )
            self.data.synthetic.loc[source, :, freq] = resp

        # Return electric field.
        if call_from_psolve:
            return (self._efields[source][freq],
                    self._solver_info[source][freq],
                    self.data.synthetic.loc[source, :, freq].data)
        else:
            return self._efields[source][freq]

    def hfields(self, source, frequency, **kwargs):
        """Return magnetic field for given source and frequency.

        The hfield is only computed from the efield if it is not stored
        already, and so is the efield, except if `recomp=True` is in `kwargs`.
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
        recomp = kwargs.get('recomp', False)

        # If magnetic field not computed yet compute it.
        if self._hfields[source][freq] is None or recomp:
            self._hfields[source][freq] = fields.get_h_field(
                    self.comp_grids(source, freq),
                    self.comp_models(source, freq),
                    self.efields(source, freq, **kwargs))

        # Return magnetic field.
        return self._hfields[source][freq]

    def solver_info(self, source, frequency):
        """Return the solver information of the corresponding computation.

        Parameters
        ----------
        source : str
            Source name.

        frequency : float
            Frequency


        Returns
        -------
        info_dict : dict or None
            Dictionary with runtime info if corresponding efield as calculated,
            else None.

        """
        return self._solver_info[source][float(frequency)]

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

        """

        # Initiate futures-dict to store output.
        out = self._initiate_none_dict()

        # Context manager for futures.
        with futures.ProcessPoolExecutor(nproc) as executor:

            # Loop over source positions.
            for src in out.keys():

                # Loop over frequencies.
                for freq in out[src].keys():

                    # Call `self.efields`.
                    out[src][freq] = executor.submit(
                        self.efields,
                        source=src,
                        frequency=freq,
                        call_from_psolve=True,
                        **kwargs,
                    )

        # Clean hfields, so they will be recomputed.
        del self._hfields
        self._hfields = self._initiate_none_dict()

        # Extract and store the electric fields.
        self._efields = {f: {k: v.result()[0] for k, v in s.items()}
                         for f, s in out.items()}

        # Extract and store the solver info.
        self._solver_info = {f: {k: v.result()[1] for k, v in s.items()}
                             for f, s in out.items()}

        # Extract and store responses at receiver locations.
        for src in self.survey.sources.keys():
            for freq in self.survey.frequencies:
                resp = out[src][freq].result()[2]
                self.data['synthetic'].loc[src, :, freq] = resp

    def clean(self):
        """Remove computed fields and corresponding data."""

        # Clean efield, hfield, and solver info.
        for name in ['efields', 'hfields', 'solver_info']:
            delattr(self, '_'+name)
            setattr(self, '_'+name, self._initiate_none_dict())

        # Set synthetic data to nan's.
        self.data['synthetic'] = self.data.observed*np.nan

    def _initiate_none_dict(self):
        """Returns a dict of the structure `dict[source][freq]=None`."""
        return {src: {freq: None for freq in self.survey.frequencies}
                for src in self.survey.sources.keys()}

    def _initiate_model_grid(self, model, grid):
        # In the marine case we assume that the sea-surface is at z=0.
        #
        # If the provided model does not reach the surface, we fill it up
        # to the surface with the resistivities of the last provided layer,
        # and add a 100 m thick air layer.
        #
        # So the model MUST contain at least one layer of sea water, from
        # where the sea resistivity is deduced.
        res_air = self.grid_opts.pop('air_resistivity')
        grid_type = self.grid_opts.pop('type')

        def extend_property(prop, check, add_values, nadd):

            if check is None:
                prop_ext = None

            else:
                prop_ext = np.zeros((grid.nCx, grid.nCy, grid.nCz+nadd))
                prop_ext[:, :, :-nadd] = prop
                if nadd == 2:
                    prop_ext[:, :, -nadd] = prop[:, :, -1]
                prop_ext[:, :, -1] = add_values

            return prop_ext

        if grid_type == 'marine' and grid.vectorNz[-1] < -0.01:
            # Fill water and add air if highest depth is less then -1 cm.

            # Extend hz.
            hz_ext = np.r_[grid.hz, -max(grid.vectorNz), 100]

            # Extend properties.
            res_x = extend_property(model.res_x, model._res_x, res_air, 2)
            res_y = extend_property(model.res_y, model._res_y, res_air, 2)
            res_z = extend_property(model.res_z, model._res_z, res_air, 2)
            mu_r = extend_property(model.mu_r, model._mu_r, 1, 2)
            epsilon_r = extend_property(
                    model.epsilon_r, model._epsilon_r, 1, 2)

            # Store grid and model.
            self.grid = meshes.TensorMesh(
                    [grid.hx, grid.hy, hz_ext], x0=grid.x0)
            self.model = models.Model(
                    self.grid, res_x, res_y, res_z, mu_r, epsilon_r)

        elif grid_type == 'marine' and abs(grid.vectorNz[-1]) < 0.01:
            # Add air if highest depth is less then 1 cm.

            # Extend hz.
            hz_ext = np.r_[grid.hz, 100]

            # Extend properties.
            res_x = extend_property(model.res_x, model._res_x, res_air, 1)
            res_y = extend_property(model.res_y, model._res_y, res_air, 1)
            res_z = extend_property(model.res_z, model._res_z, res_air, 1)
            mu_r = extend_property(model.mu_r, model._mu_r, 1, 1)
            epsilon_r = extend_property(
                    model.epsilon_r, model._epsilon_r, 1, 1)

            # Store grid and model.
            self.grid = meshes.TensorMesh(
                    [grid.hx, grid.hy, hz_ext], x0=grid.x0)
            self.model = models.Model(
                    self.grid, res_x, res_y, res_z, mu_r, epsilon_r)

        else:
            # Just store provided grid and model.
            self.grid = grid
            self.model = model

    @property
    def data(self):
        """Shortcut to survey.data."""
        return self.survey.data

#     def gradient(self, solver_opts=None, nproc=4, wdepth=None, wdata=None,
#                  regularization=None, verb=2):
#         """Computes the gradient using the adjoint-state method.
#
#         Following [PlMu06]_.
#
#         .. todo::
#
#             - Several parts only work for Ex at the moment, no azimuth, no
#               dip, no magnetic fields; generalize.
#             - Regularization is not implemented.
#
#
#         Parameters
#         ----------
#         solver_opts : dict
#             Parameter-dicts passed to the solver and the automatic gridding,
#             respectively.
#
#         nproc : int
#             Maximum number of processes that can be used. Default is four.
#
#         wdepth, wdata, regularization : dict or None
#             Parameter-dict for depth- and data-weighting and regularization,
#             respectively. Regularization is NOT implemented yet.
#
#         verb : int; optional
#             Level of verbosity (as used in emg3d)
#
#
#         Returns
#         -------
#         grad : ndarray
#             Current gradient; has shape mesh.vnC (same as model properties).
#
#         misfit : float
#             Current misfit; as regularization is not implemented this
#             corresponds to the data misfit.
#
#         """
#         def cprint(str, verb=verb):
#             """Conditional print."""
#             if verb > 1:
#                 print(str)
#
#         # Get forwards electric fields (parallel).
#         cprint("\n** Forward fields **\n")
#         # TODO Adjust; Used to be ffields
#
#         self.psolve(nproc=4,
#                     **(solver_opts if solver_opts is not None else {}),
#                     )
#
#         # Get residual fields (parallel).
#         cprint("\n** Residual fields **\n")
#         # TODO Implement resfield- [sfield/rfield]
#         resfields, data_misfit = resfield_parallel(
#                 data, grids, ffields, wdata, nproc)
#         cprint(f"   Data misfit: {data_misfit:.2e}")
#
#         # Get backwards electric fields (parallel).
#         cprint("\n** Backward fields **\n")
#         # TODO Implement resfield- [efield/bfield]
#         bfields = self.psolve(grids, models, resfields, solver_args, nproc)
#
#         # Get gradient.
#         cprint("\n** Gradient **\n")
#         # TODO implement
#         grad = compute_gradient(ffields, bfields, grids, mesh, wdepth)
#
#         return grad, data_misfit
