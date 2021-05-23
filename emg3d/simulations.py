"""
A simulation is the computation (modelling) of the electromagnetic responses
due to a given model and survey.

The simulation module combines the different pieces of ``emg3d`` providing
a high-level, specialised modelling tool for the end user.
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

import itertools
from copy import deepcopy

import numpy as np

from emg3d import (electrodes, fields, io, maps, meshes, models,
                   optimize, solver, surveys, utils)

__all__ = ['Simulation', 'expand_grid_model', 'estimate_gridding_opts']


@utils._known_class
class Simulation:
    """Create a simulation for a given survey on a given model.

    A simulation can be used to compute responses for an entire survey, hence
    for an arbitrary amount of sources, receivers, and frequencies. The
    responses are computed in parallel over sources and frequencies. It can
    also be used to compute the misfit with the data and to compute the
    gradient of the misfit function.

    The computational grid(s) can either be provided, or automatic gridding can
    be used; see the description of the parameters ``gridding`` and
    ``gridding_opts`` for more details.

    .. note::

        The automatic gridding does its best to generate meshes that are
        suitable for the provided model and survey. However, CSEM spans a wide
        range of acquisition layouts, and both frequencies and conductivities
        or resistivities span many orders of magnitude. This makes it hard to
        have a function that fits all purposes. Check the meshes with your
        expert knowledge. Also, the automatic gridding is conservative in its
        estimate, trying to be on the save side (correct results over speed).
        This means, however, that often smaller grids could be used by
        providing the appropriate options in ``gridding_opts`` or directly
        providing your own computational grids.

    .. note::

        The package ``xarray`` has to be installed in order to use
        ``Simulation``:
        ``pip install xarray`` or ``conda install -c conda-forge xarray``.


    Parameters
    ----------
    survey : Survey
        The survey; a :class:`emg3d.surveys.Survey` instance. The survey
        contains sources, receivers, frequencies, and optionally data.

        The survey-data will be modified in place. Provide ``survey.copy()`` if
        you want to avoid this.

    model : Model
        The model; a :class:`emg3d.models.Model` instance.

    max_workers : int, default: 4
        The maximum number of processes that can be used to execute the
        given calls.

    gridding : str, default: 'single'
        Method to create the computational grids.

        The different methods are:

        - ``'same'``: Same grid as for the input model.
        - ``'single'``: A single grid for all sources and frequencies.
        - ``'frequency'``: Frequency-dependent grids.
        - ``'source'``: Source-dependent grids.
        - ``'both'``: Frequency- and source-dependent grids.
        - ``'input'``: Same as ``'single'``, but the grid has to be provided in
          ``gridding_opts`` instead of being automatically created.
        - ``'dict'``: Same as ``'both'``, but the grids have to be provided in
          ``gridding_opts`` in the form of ``dict[source][frequency]`` instead
          of being automatically created.

        See the parameter ``gridding_opts`` for more details.

    gridding_opts : {dict, TensorMesh}, default: {}
        Input format depends on ``gridding``:

        - ``'same'``: Nothing, ``gridding_opts`` is not permitted.
        - ``'single'``, ``'frequency'``, ``'source'``, ``'both'``: Described
          below.
        - ``'input'``: A :class:`emg3d.meshes.TensorMesh`.
        - ``'dict'``: Dictionary of the format ``dict[source][frequency]``
          containing a :class:`emg3d.meshes.TensorMesh` for each
          source-frequency pair.

        The dict in the case of ``'single'``, ``'frequency'``, ``'source'``,
        ``'both``' is passed to :func:`emg3d.meshes.construct_mesh`; consult
        the corresponding documentation for more information. Parameters that
        are not provided are estimated from the provided model, grid, and
        survey using :func:`emg3d.simulations.estimate_gridding_opts`, which
        documentation contains more information too.

        There are two notably differences to the parameters described in
        :func:`emg3d.meshes.construct_mesh`:

        - ``vector``: besides the normal possibility it can also be a string
          containing one or several of ``'x'``, ``'y'``, and ``'z'``. In these
          cases the corresponding dimension of the input mesh is provided as
          vector. See :func:`emg3d.simulations.estimate_gridding_opts`.
        - ``expand``: in the format of ``[property_sea, property_air]``; if
          provided, the input model is expanded up to the seasurface with sea
          water, and an air layer is added. The actual height of the seasurface
          can be defined with the key ``seasurface``. See
          :func:`emg3d.simulations.expand_grid_model`.

    solver_opts : dict, default: {'verb': 2'}
        Passed through to :func:`emg3d.solver.solve`. The dict can contain any
        parameter that is accepted by the :func:`emg3d.solver.solve` except for
        ``model``, ``sfield``, ``efield``, ``return_info``, and ``log``.
        Default verbosity is ``verb=2``.

    verb : int, default: 0
        Level of verbosity. Possible options:

        - -1: Errors.
        - 0: Warnings.
        - 1: Info.

    name : str, default: None
        Name of the simulation.

    info : str, default: None
        Simulation info or any other info (e.g., what was the purpose of this
        simulation).

    """

    # Gridding descriptions (for repr's).
    _gridding_descr = {
            'same': 'Same grid as for model',
            'single': 'Single grid for all sources and frequencies',
            'frequency': 'Frequency-dependent grids',
            'source': 'Source-dependent grids',
            'both': 'Frequency- and source-dependent grids',
            'input': 'Provided grid, same for all sources/frequencies',
            'dict': 'Provided grids, frequency-/source-dependent',
            }

    def __init__(self, survey, model, max_workers=4, gridding='single',
                 **kwargs):
        """Initiate a new Simulation instance."""

        # Store some inputs as is, optional ones with defaults.
        self.survey = survey
        self.max_workers = max_workers
        self.gridding = gridding
        self.verb = kwargs.pop('verb', 0)
        self.name = kwargs.pop('name', None)
        self.info = kwargs.pop('info', None)

        # Assemble solver_opts.
        self.solver_opts = {
                'verb': 2,  # Default verbosity, can be overwritten.
                **kwargs.pop('solver_opts', {}),  # User setting.
                'return_info': True,  # return_info=True is forced.
                'log': -1             # log=-1 is forced.
        }

        # Initiate dictionaries and other values with None's.
        self._dict_grid = self._dict_initiate
        self._dict_model = self._dict_initiate
        self._dict_efield = self._dict_initiate
        self._dict_hfield = self._dict_initiate
        self._dict_efield_info = self._dict_initiate
        self._gradient = None
        self._misfit = None

        # Get model taking gridding_opts into account.
        # Sets self.model and self.gridding_opts.
        self._set_model(model, kwargs)

        # Initiate synthetic data with NaN's if they don't exist.
        if 'synthetic' not in self.survey.data.keys():
            self.survey._data['synthetic'] = self.data.observed.copy(
                    data=np.full(self.survey.shape, np.nan+1j*np.nan))

        # `tqdm`-options; undocumented.
        # Can be used to, e.g., disable tqdm completely via:
        # > emg3d.Simulation(args, kwargs, tqdm_opts={'disable': True})
        self._tqdm_opts = kwargs.pop(
                'tqdm_opts',
                {'bar_format': '{desc}: {bar}{n_fmt}/{total_fmt}  [{elapsed}]'}
        )

        # Ensure no kwargs left.
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

    def __repr__(self):
        """Simple representation."""
        name = f" «{self.name}»" if self.name else ""
        info = f"{self.info}\n" if self.info else ""
        survey = f" «{self.survey.name}»" if self.survey.name else ""
        return (f":: {self.__class__.__name__}{name} ::\n{info}\n"
                f"- {self.survey.__class__.__name__}{survey}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies\n"
                f"- {self.model.__repr__()}\n"
                f"- Gridding: {self._gridding_descr[self.gridding]}; "
                f"{self._info_grids}")

    def _repr_html_(self):
        """HTML representation."""
        name = f" «{self.name}»" if self.name else ""
        info = f"{self.info}<br>" if self.info else ""
        survey = f" «{self.survey.name}»" if self.survey.name else ""
        return (f"<h3>{self.__class__.__name__}{name}</h3>{info}"
                f"<ul>"
                f"  <li>{self.survey.__class__.__name__}{survey}:"
                f"    {self.survey.shape[0]} sources;"
                f"    {self.survey.shape[1]} receivers;"
                f"    {self.survey.shape[2]} frequencies</li>"
                f"  <li>{self.model.__repr__()}</li>"
                f"  <li>Gridding: {self._gridding_descr[self.gridding]}; "
                f"    {self._info_grids}</li>"
                f"</ul>")

    def clean(self, what='computed'):
        """Clean part of the data base.

        Parameters
        ----------
        what : str, default: 'computed'
            What to clean. Possibilities:

            - ``'computed'``:
              Removes all computed properties: electric and magnetic fields and
              responses at receiver locations.
            - ``'keepresults'``:
              Removes everything  except for the responses at receiver
              locations.
            - ``'all'``:
              Removes everything (leaves it plain as initiated).

        """

        if what not in ['computed', 'keepresults', 'all']:
            raise TypeError(f"Unrecognized `what`: {what}.")

        # Clean grid/model-dicts.
        if what in ['keepresults', 'all']:

            # These exist always and have to be initiated.
            for name in ['_dict_grid', '_dict_model']:
                delattr(self, name)
                setattr(self, name, self._dict_initiate)

        # Clean field-dicts.
        if what in ['computed', 'keepresults', 'all']:

            # These exist always and have to be initiated.
            for name in ['_dict_efield', '_dict_efield_info', '_dict_hfield']:
                delattr(self, name)
                setattr(self, name, self._dict_initiate)

            # These only exist with gradient; don't initiate them.
            for name in ['_dict_bfield', '_dict_bfield_info']:
                if hasattr(self, name):
                    delattr(self, name)

        # Clean data.
        if what in ['computed', 'all']:
            for key in ['residual', 'weight']:
                if key in self.data.keys():
                    del self.data[key]
            self.data['synthetic'] = self.data.observed.copy(
                    data=np.full(self.survey.shape, np.nan+1j*np.nan))
            for name in ['_gradient', '_misfit']:
                delattr(self, name)
                setattr(self, name, None)

    def copy(self, what='computed'):
        """Return a copy of the Simulation.

        See ``to_file`` for more information regarding ``what``.

        """
        return self.from_dict(self.to_dict(what, True))

    def to_dict(self, what='computed', copy=False):
        """Store the necessary information of the Simulation in a dict.

        See `to_file` for more information regarding `what`.

        """

        # If to_dict is called from to_file, it has a _what_to_file attribute.
        if hasattr(self, '_what_to_file'):
            what = self._what_to_file
            delattr(self, '_what_to_file')

        if what not in ['computed', 'results', 'all', 'plain']:
            raise TypeError(f"Unrecognized `what`: {what}.")

        # Initiate dict with input parameters.
        out = {
            '__class__': self.__class__.__name__,
            'survey': self.survey.to_dict(),
            'model': self.model.to_dict(),
            'max_workers': self.max_workers,
            'gridding': self.gridding,
            'gridding_opts': self.gridding_opts,
            'solver_opts': self.solver_opts,
            'verb': self.verb,
            'name': self.name,
            'info': self.info,
            '_input_sc2': self._input_sc2,
        }

        # Clean unwanted data if plain.
        if what == 'plain':
            for key in ['synthetic', 'residual', 'weights']:
                if key in out['survey']['data'].keys():
                    del out['survey']['data'][key]

        # Store wanted dicts.
        if what in ['computed', 'all']:
            for name in ['_dict_efield', '_dict_efield_info', '_dict_hfield',
                         '_dict_bfield', '_dict_bfield_info']:
                if hasattr(self, name):
                    out[name] = getattr(self, name)

            if what == 'all':
                for name in ['_dict_grid', '_dict_model']:
                    if hasattr(self, name):
                        out[name] = getattr(self, name)

        # Store gradient and misfit.
        if what in ['computed', 'results', 'all']:
            out['gradient'] = self._gradient
            out['misfit'] = self._misfit

        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dict into :class:`emg3d.simulations.Simulation` instance.


        Parameters
        ----------
        inp : dict
            Dictionary as obtained from
            :func:`emg3d.simulations.Simulation.to_dict`.

        Returns
        -------
        simulation : Simulation
            A :class:`emg3d.simulations.Simulation` instance.

        """
        inp = {k: v for k, v in inp.items() if k != '__class__'}

        # Get all class-inputs.
        inp_names = ['survey', 'model', 'max_workers', 'gridding',
                     'solver_opts', 'verb', 'name', 'info']
        cls_inp = {k: inp.pop(k) for k in inp_names}
        cls_inp['gridding_opts'] = inp.pop('gridding_opts', {})
        cls_inp['survey'] = surveys.Survey.from_dict(cls_inp['survey'])
        cls_inp['model'] = models.Model.from_dict(cls_inp['model'])
        input_sc2 = inp.pop('_input_sc2', False)
        if input_sc2:
            cls_inp['_input_sc2'] = input_sc2

        # Instantiate the class.
        out = cls(**cls_inp)

        # Add existing derived/computed properties.
        data = ['_dict_grid', '_dict_model',
                '_dict_hfield', '_dict_efield', '_dict_efield_info',
                '_dict_bfield', '_dict_bfield_info']
        for name in data:
            if name in inp.keys():
                values = inp.pop(name)

                # De-serialize Model, Field, and TensorMesh instances.
                io._dict_deserialize(values)

                setattr(out, name, values)

        # Add gradient and misfit.
        data = ['gradient', 'misfit']
        for name in data:
            if name in inp.keys():
                setattr(out, '_'+name, inp.pop(name))

        return out

    def to_file(self, fname, what='computed', name='simulation', **kwargs):
        """Store Simulation to a file.

        Parameters
        ----------
        fname : str
            Absolute or relative file name including ending, which defines the
            used data format. See :func:`emg3d.io.save` for the options.

        what : str, default: 'computed'
            What to store. Possibilities:

            - ``'computed'``:
              Stores all computed properties: electric fields and responses at
              receiver locations.
            - '``results'``:
              Stores only the response at receiver locations.
            - ``'all'``:
              Stores everything.
            - ``'plain'``:
              Only stores the plain Simulation (as initiated).

        name : str, default: 'simulation'
            Name with which the simulation is stored in the file.

        kwargs : Keyword arguments, optional
            Passed through to :func:`emg3d.io.save`.

        """
        # Add what to self, will be removed in to_dict.
        self._what_to_file = what

        # Add simulation to dict.
        kwargs[name] = self

        # If verb is not defined, use verbosity of simulation.
        kwargs['verb'] = kwargs.get('verb', self.verb)

        return io.save(fname, **kwargs)

    @classmethod
    def from_file(cls, fname, name='simulation', **kwargs):
        """Load Simulation from a file.

        Parameters
        ----------
        fname : str
            Absolute or relative file name including extension.

        name : str, default: 'simulation'
            Name under which the simulation is stored within the file.

        kwargs : Keyword arguments, optional
            Passed through to :func:`io.load`.


        Returns
        -------
        simulation : Simulation
            A :class:`emg3d.simulations.Simulation` instance.

        info : str, returned if verb<0
            Info-string.

        """
        out = io.load(fname, **kwargs)
        if 'verb' in kwargs and kwargs['verb'] < 0:
            return out[0][name], out[1]
        else:
            return out[name]

    # GET FUNCTIONS
    @property
    def data(self):
        """Shortcut to survey.data."""
        return self.survey.data

    def get_grid(self, source, frequency):
        """Return computational grid of the given source and frequency."""
        freq = self._freq_inp2key(frequency)

        # Return grid if it exists already.
        if self._dict_grid[source][freq] is not None:
            return self._dict_grid[source][freq]

        # Same grid as for provided model.
        if self.gridding == 'same':

            # Store link to grid.
            self._dict_grid[source][freq] = self.model.grid

        # Frequency-dependent grids.
        elif self.gridding == 'frequency':

            # Initiate dict.
            if not hasattr(self, '_grid_frequency'):
                self._grid_frequency = {}

            # Get grid for this frequency if not yet computed.
            if freq not in self._grid_frequency.keys():

                # Get grid and store it.
                inp = {**self.gridding_opts, 'frequency':
                       self.survey.frequencies[freq]}
                self._grid_frequency[freq] = meshes.construct_mesh(**inp)

            # Store link to grid.
            self._dict_grid[source][freq] = self._grid_frequency[freq]

        # Source-dependent grids.
        elif self.gridding == 'source':

            # Initiate dict.
            if not hasattr(self, '_grid_source'):
                self._grid_source = {}

            # Get grid for this source if not yet computed.
            if source not in self._grid_source.keys():

                # Get grid and store it.
                center = self.survey.sources[source].center
                inp = {**self.gridding_opts, 'center': center}
                self._grid_source[source] = meshes.construct_mesh(**inp)

            # Store link to grid.
            self._dict_grid[source][freq] = self._grid_source[source]

        # Source- and frequency-dependent grids.
        elif self.gridding == 'both':

            # Get grid and store it.
            center = self.survey.sources[source].center
            inp = {**self.gridding_opts, 'frequency':
                   self.survey.frequencies[freq], 'center': center}
            self._dict_grid[source][freq] = meshes.construct_mesh(**inp)

        # Use a single grid for all sources and receivers.
        # Default case; catches 'single' but also anything else.
        else:

            # Get grid if not yet computed.
            if not hasattr(self, '_grid_single'):

                # Get grid and store it.
                self._grid_single = meshes.construct_mesh(**self.gridding_opts)

            # Store link to grid.
            self._dict_grid[source][freq] = self._grid_single

        # Use recursion to return grid.
        return self.get_grid(source, frequency)

    def get_model(self, source, frequency):
        """Return model on the grid of the given source and frequency."""
        freq = self._freq_inp2key(frequency)

        # Return model if it exists already.
        if self._dict_model[source][freq] is not None:
            return self._dict_model[source][freq]

        # Same grid as for provided model.
        if self.gridding == 'same':

            # Store link to model.
            self._dict_model[source][freq] = self.model

        # Frequency-dependent grids.
        elif self.gridding == 'frequency':

            # Initiate dict.
            if not hasattr(self, '_model_frequency'):
                self._model_frequency = {}

            # Get model for this frequency if not yet computed.
            if freq not in self._model_frequency.keys():
                self._model_frequency[freq] = self.model.interpolate_to_grid(
                        self.get_grid(source, freq))

            # Store link to model.
            self._dict_model[source][freq] = self._model_frequency[freq]

        # Source-dependent grids.
        elif self.gridding == 'source':

            # Initiate dict.
            if not hasattr(self, '_model_source'):
                self._model_source = {}

            # Get model for this source if not yet computed.
            if source not in self._model_source.keys():
                self._model_source[source] = self.model.interpolate_to_grid(
                        self.get_grid(source, freq))

            # Store link to model.
            self._dict_model[source][freq] = self._model_source[source]

        # Source- and frequency-dependent grids.
        elif self.gridding == 'both':

            # Get model and store it.
            self._dict_model[source][freq] = self.model.interpolate_to_grid(
                        self.get_grid(source, freq))

        # Use a single grid for all sources and receivers.
        # Default case; catches 'single' but also anything else.
        else:

            # Get model if not yet computed.
            if not hasattr(self, '_model_single'):
                self._model_single = self.model.interpolate_to_grid(
                        self.get_grid(source, freq))

            # Store link to model.
            self._dict_model[source][freq] = self._model_single

        # Use recursion to return model.
        return self.get_model(source, frequency)

    def get_efield(self, source, frequency, **kwargs):
        """Return electric field for given source and frequency."""
        freq = self._freq_inp2key(frequency)

        # Get call_from_compute and ensure no kwargs are left.
        call_from_compute = kwargs.pop('call_from_compute', False)
        call_from_hfield = kwargs.pop('call_from_hfield', False)
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

        # Compute electric field if it is not stored yet.
        if self._dict_efield[source][freq] is None:

            # Input parameters.
            solver_input = {
                **self.solver_opts,
                'model': self.get_model(source, freq),
                'sfield': fields.get_source_field(
                    self.get_grid(source, freq),
                    self.survey.sources[source],
                    self.survey.frequencies[freq]),
            }

            # Compute electric field.
            efield, info = solver.solve(**solver_input)

            # Store electric field and info.
            self._dict_efield[source][freq] = efield
            self._dict_efield_info[source][freq] = info

            if not call_from_hfield:

                # Clean corresponding hfield, so it will be recomputed.
                del self._dict_hfield[source][freq]
                self._dict_hfield[source][freq] = None

                # Store electric and magnetic responses at receiver locations.
                self._store_responses(source, freq)

        # Return electric field.
        if call_from_compute:
            return (self._dict_efield[source][freq],
                    self._dict_efield_info[source][freq],
                    self._dict_hfield[source][freq],
                    self.data.synthetic.loc[source, :, freq].data)
        else:
            return self._dict_efield[source][freq]

    def get_hfield(self, source, frequency, **kwargs):
        """Return magnetic field for given source and frequency."""
        freq = self._freq_inp2key(frequency)

        # If magnetic field not computed yet compute it.
        if self._dict_hfield[source][freq] is None:

            self._dict_hfield[source][freq] = fields.get_magnetic_field(
                    self.get_model(source, freq),
                    self.get_efield(source, freq,
                                    call_from_hfield=True, **kwargs))

            # Store electric and magnetic responses at receiver locations.
            self._store_responses(source, frequency)

        # Return magnetic field.
        return self._dict_hfield[source][freq]

    def get_efield_info(self, source, frequency):
        """Return the solver information of the corresponding computation."""
        return self._dict_efield_info[source][self._freq_inp2key(frequency)]

    def _store_responses(self, source, frequency):
        """Return electric and magnetic fields at receiver locations."""
        freq = self._freq_inp2key(frequency)

        # Get receiver types.
        rec_types = tuple([r.xtype == 'electric'
                           for r in self.survey.receivers.values()])

        # Get absolute coordinates as fct of source.
        # (Only relevant in case of "relative" receivers.)
        rl = list(self.survey.receivers.values())

        def rec_coord_tuple(rec_list):
            """Return abs. coordinates for as a fct of source."""
            return tuple(np.array(
                [rl[i].coordinates_abs(self.survey.sources[source])
                 for i in rec_list]
            ).T)

        # Store electric receivers.
        if rec_types.count(True):

            # Extract data at receivers.
            erec = np.nonzero(rec_types)[0]
            resp = self.get_efield(source, freq).get_receiver(
                    receiver=rec_coord_tuple(erec)
            )

            # Store the receiver response.
            self.data.synthetic.loc[source, :, freq][erec] = resp

        # Store magnetic receivers.
        if rec_types.count(False):

            # Extract data at receivers.
            mrec = np.nonzero(np.logical_not(rec_types))[0]
            resp = self.get_hfield(source, freq).get_receiver(
                    receiver=rec_coord_tuple(mrec)
            )

            # Store the receiver response.
            self.data.synthetic.loc[source, :, freq][mrec] = resp

    # ASYNCHRONOUS COMPUTATION
    def _get_efield(self, inp):
        """Wrapper of `get_efield` for `concurrent.futures`."""
        return self.get_efield(*inp, call_from_compute=True)

    def compute(self, observed=False, **kwargs):
        """Compute efields asynchronously for all sources and frequencies.

        Parameters
        ----------
        observed : bool, default: False
            If True, it stores the current result also as observed model. This
            is usually done for pure forward modelling (not inversion). It will
            as such be stored within the survey. If the survey has either
            ``relative_error`` or ``noise_floor``, random Gaussian noise of
            standard deviation will be added to the ``data.observed`` (not to
            ``data.synthetic``). Also, data below the noise floor will be set
            to NaN.

        min_offset : float, default: 0.0
            Data points in ``data.observed`` where the offset < min_offset are
            set to NaN.

        """
        srcfreq = self._srcfreq.copy()

        # We remove the ones that were already computed.
        remove = []
        for src, freq in srcfreq:
            if self._dict_efield[src][freq] is not None:
                remove += [(src, freq)]
        for src, freq in remove:
            srcfreq.remove((src, freq))

        # Ensure grids, models, and source fields are computed.
        #
        # => This could be done within the field computation. But then it might
        #    have to be done multiple times even if 'single' or 'same' grid.
        #    Something to keep in mind.
        #    For `gridding='same'` it does not really matter.
        for src, freq in srcfreq:
            _ = self.get_grid(src, freq)
            _ = self.get_model(src, freq)

        # Initiate futures-dict to store output.
        out = utils._process_map(
                self._get_efield,
                srcfreq,
                max_workers=self.max_workers,
                **{'desc': 'Compute efields', **self._tqdm_opts},
        )

        # Loop over src-freq combinations to extract and store.
        for i, (src, freq) in enumerate(srcfreq):

            # Store efield and solver info.
            self._dict_efield[src][freq] = out[i][0]
            self._dict_efield_info[src][freq] = out[i][1]
            self._dict_hfield[src][freq] = out[i][2]

            # Store responses at receivers.
            self.data['synthetic'].loc[src, :, freq] = out[i][3]

        # Print solver info.
        self.print_solver_info('efield', verb=self.verb)

        # If it shall be used as observed data save a copy.
        if observed:

            self.data['observed'] = self.data['synthetic'].copy()

            # Add noise if noise_floor and/or relative_error given.
            if self.survey.standard_deviation is not None:

                # Create noise.
                std = self.survey.standard_deviation
                random = np.random.randn(self.survey.count*2)
                noise_re = std*random[::2].reshape(self.survey.shape)
                noise_im = std*random[1::2].reshape(self.survey.shape)

                # Add noise to observed data.
                self.data['observed'].data += noise_re + 1j*noise_im

            # Set data below the noise floor to NaN.
            if self.survey.noise_floor is not None:
                noise_floor = self.survey.noise_floor
                min_amp = abs(self.data.synthetic.data) < noise_floor
                self.data['observed'].data[min_amp] = np.nan + 1j*np.nan

            # Set near-offsets to NaN.
            min_off = kwargs.get('min_offset', 0.0)
            nan = np.nan + 1j*np.nan
            for ks, s in self.survey.sources.items():
                for kr, r in self.survey.receivers.items():
                    if np.linalg.norm(r.center_abs(s) - s.center) < min_off:
                        self.data['observed'].loc[ks, kr, :] = nan

    # OPTIMIZATION
    @property
    def gradient(self):
        """Return the gradient of the misfit function.

        See :func:`emg3d.optimize.gradient`.

        """
        if self._gradient is None:
            self._gradient = optimize.gradient(self)
        return self._gradient[:, :, :self._input_sc2]

    @property
    def misfit(self):
        """Return the misfit function.

        See :func:`emg3d.optimize.misfit`.

        """
        if self._misfit is None:
            self._misfit = optimize.misfit(self)
        return self._misfit

    def _get_bfields(self, inp):
        """Return back-propagated electric field for given inp (src, freq)."""

        # Input parameters.
        solver_input = {
            **self.solver_opts,
            'model': self.get_model(*inp),
            'sfield': self._get_rfield(*inp),  # Residual field.
        }

        # Compute and return back-propagated electric field.
        return solver.solve(**solver_input)

    def _bcompute(self):
        """Compute bfields asynchronously for all sources and frequencies."""

        # Initiate futures-dict to store output.
        out = utils._process_map(
                self._get_bfields,
                self._srcfreq,
                max_workers=self.max_workers,
                **{'desc': 'Back-propagate ', **self._tqdm_opts},
        )

        # Store back-propagated electric field and info.
        if not hasattr(self, '_dict_bfield'):
            self._dict_bfield = self._dict_initiate
            self._dict_bfield_info = self._dict_initiate

        # Loop over src-freq combinations to extract and store.
        for i, (src, freq) in enumerate(self._srcfreq):

            # Store bfield and solver info.
            self._dict_bfield[src][freq] = out[i][0]
            self._dict_bfield_info[src][freq] = out[i][1]

        # Print solver info.
        self.print_solver_info('bfield', verb=self.verb)

    def _get_rfield(self, source, frequency):
        """Return residual source field for given source and frequency."""

        freq = self.survey.frequencies[frequency]
        grid = self.get_grid(source, frequency)

        # Initiate empty residual source field
        rfield = fields.Field(grid, frequency=freq)

        # Loop over receivers, input as source.
        for name, rec in self.survey.receivers.items():

            # Get residual of this receiver.
            residual = self.data.residual.loc[source, name, frequency].data
            if np.isnan(residual):
                continue

            # Residual source strength: Weighted residual, normalized by -smu0.
            weight = self.data.weights.loc[source, name, frequency].data
            strength = np.conj(residual * weight / -rfield.smu0)

            # Create source.
            if rec.xtype == 'magnetic':
                src_fct = electrodes.TxMagneticDipole

                # If the data is from a magnetic point we have to undo another
                # factor smu0 here, as the source will be a loop.
                strength /= rfield.smu0

            else:
                src_fct = electrodes.TxElectricDipole

            # Get absolute coordinates as fct of source.
            # (Only relevant in case of "relative" receivers.)
            coords = rec.coordinates_abs(self.survey.sources[source])

            # Get residual field and add it to the total field.
            rfield.field += fields.get_source_field(
                    grid=grid,
                    source=src_fct(coords, strength=strength),
                    frequency=freq,
            ).field

        return rfield

    # UTILS
    @property
    def _dict_initiate(self):
        """Return a dict of the structure `dict[source][freq]=None`."""
        return {src: {freq: None for freq in self.survey.frequencies}
                for src in self.survey.sources.keys()}

    @property
    def _srcfreq(self):
        """Return list of all source-frequency pairs."""

        if getattr(self, '__srcfreq', None) is None:
            self.__srcfreq = list(
                itertools.product(
                    self.survey.sources.keys(), self.survey.frequencies.keys()
                )
            )

        return self.__srcfreq

    def _freq_inp2key(self, frequency):
        """Return key of frequency entry given its key or its value. """
        if not isinstance(frequency, str):
            if not hasattr(self, '__freq_inp2key'):
                self.__freq_inp2key = {
                    float(v): k for k, v in self.survey.frequencies.items()
                }
            frequency = self.__freq_inp2key[frequency]

        return frequency

    @property
    def _info_grids(self):
        """Return a string with "min {- max}" grid size."""

        # Single grid for all sources and receivers.
        if self.gridding in ['same', 'single', 'input']:
            grid = self.get_grid(*self._srcfreq[0])
            min_nc = grid.n_cells
            min_vc = grid.shape_cells
            has_minmax = False

        # Source- and/or frequency-dependent grids.
        else:
            min_nc = 1e100
            max_nc = 0
            # Loop over all grids and get smallest/biggest values.
            for src, freq in self._srcfreq:
                grid = self.get_grid(src, freq)
                if grid.n_cells > max_nc:
                    max_nc = grid.n_cells
                    max_vc = grid.shape_cells
                if grid.n_cells < min_nc:
                    min_nc = grid.n_cells
                    min_vc = grid.shape_cells
            has_minmax = min_nc != max_nc

        # Assemble info.
        info = f"{min_vc[0]} x {min_vc[1]} x {min_vc[2]} ({min_nc:,})"
        if has_minmax:
            info += f" - {max_vc[0]} x {max_vc[1]} x {max_vc[2]} ({max_nc:,})"

        return info

    def print_grid_info(self, verb=1, return_info=False):
        """Print info for all generated grids."""

        def get_grid_info(src, freq):
            """Return grid info for given source and frequency."""
            grid = self.get_grid(src, freq)
            out = ''
            if verb != 0 and hasattr(grid, 'construct_mesh_info'):
                out += grid.construct_mesh_info
            out += grid.__repr__()
            return out

        # Act depending on gridding:
        out = ""

        # Frequency-dependent.
        if self.gridding == 'frequency':
            for freq in self.survey.frequencies.values():
                out += f"= Source: all; Frequency: {freq} Hz =\n"
                out += get_grid_info(self._srcfreq[0][0], freq)

        # Source-dependent.
        elif self.gridding == 'source':
            for src in self.survey.sources.keys():
                out += f"= Source: {src}; Frequency: all =\n"
                out += get_grid_info(src, self._srcfreq[0][1])

        # Source- and frequency-dependent.
        elif self.gridding == 'both':
            for src, freq in self._srcfreq:
                out += f"= Source: {src}; Frequency: "
                out += f"{self.survey.frequencies[freq]} Hz =\n"
                out += get_grid_info(src, freq)

        # Single grid.
        else:
            out += "= Source: all; Frequency: all =\n"
            out += get_grid_info(self._srcfreq[0][0], self._srcfreq[0][1])

        if return_info:
            return out
        elif out:
            print(out)

    def print_solver_info(self, field='efield', verb=1, return_info=False):
        """Print solver info."""

        # Get info dict.
        info = getattr(self, f"_dict_{field}_info", {})
        out = ""

        if verb < 0:
            return

        # Loop over sources and frequencies.
        for src, freq in self._srcfreq:
            cinfo = info[src][freq]

            # Print if verbose or not converged.
            if cinfo is not None and (verb > 0 or cinfo['exit'] != 0):

                # Initial message.
                if not out:
                    out += "\n"
                    if verb > 0:
                        out += f"    - SOLVER INFO <{field}> -\n\n"

                # Source and frequency info.
                out += f"= Source {src}; Frequency "
                out += f"{self.survey.frequencies[freq]} Hz ="

                # Print log depending on solver and simulation verbosities.
                if verb == 0 or self.solver_opts['verb'] not in [1, 2]:
                    out += f" {cinfo['exit_message']}\n"

                if verb > 0 and self.solver_opts['verb'] > 2:
                    out += f"\n{cinfo['log']}\n"

                if verb > 0 and self.solver_opts['verb'] in [1, 2]:
                    out += f" {cinfo['log'][12:]}"

        if return_info:
            return out
        elif out:
            print(out)

    def _set_model(self, model, kwargs):
        """Set self.model and self.gridding_opts."""

        # Store original input_sc2. Undocumented.
        # This should eventually be replaced by an `active_cells` mask.
        self._input_sc2 = kwargs.pop('_input_sc2', model.grid.shape_cells[2])

        # Get gridding_opts from kwargs.
        gridding_opts = kwargs.pop('gridding_opts', {})

        # If 'dict', entire dict should be provided.
        if self.gridding == 'dict':
            self._dict_grid = gridding_opts

        # If 'input', TensorMesh should be provided
        elif self.gridding == 'input':
            self._grid_single = gridding_opts

        # If 'same', there shouldn't be any options.
        elif self.gridding == 'same':
            if gridding_opts:
                msg = "`gridding_opts` is not permitted if `gridding='same'`."
                raise TypeError(msg)

        # If 'source', 'frequency', 'both', 'single' => automatic gridding.
        else:
            g_opts = gridding_opts.copy()

            # Expand model by water and air if required.
            expand = g_opts.pop('expand', None)
            if expand is not None:
                try:
                    interface = g_opts['seasurface']
                except KeyError as e:
                    msg = ("`g_opts['seasurface']` is required if "
                           "`g_opts['expand']` is provided.")
                    raise KeyError(msg) from e

                model = expand_grid_model(model, expand, interface)

            # Get automatic gridding input.
            # Estimate the parameters from survey and model if not provided.
            gridding_opts = estimate_gridding_opts(
                    g_opts, model, self.survey, self._input_sc2)

        self.gridding_opts = gridding_opts
        self.model = model


# HELPER FUNCTIONS
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
        model = models.Model(
                grid, property_x, property_y, property_z, mu_r,
                epsilon_r, mapping=model.map.name)

    return model


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
        If :func:`emg3d.simulations.expand_grid_model` was used, ``input_sc2``
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
