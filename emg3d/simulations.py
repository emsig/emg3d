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

import warnings
import itertools
from copy import deepcopy

import numpy as np

from emg3d import (electrodes, fields, io, maps, meshes, models, solver,
                   surveys, utils)

__all__ = ['Simulation', ]


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

        If ``gridding`` is ``'same'`` or ``'input'``, the input grid is checked
        if it is a sensible grid for emg3d; if not, it throws a warning. In the
        other cases the grids are created by emg3d itself, they will be fine.
        (If ``'dict'`` we assume the user knows how to provide good grids.)

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
        survey using :func:`emg3d.meshes.estimate_gridding_opts`, which
        documentation contains more information too.

        There are two notably differences to the parameters described in
        :func:`emg3d.meshes.construct_mesh`:

        - ``vector``: besides the normal possibility it can also be a string
          containing one or several of ``'x'``, ``'y'``, and ``'z'``. In these
          cases the corresponding dimension of the input mesh is provided as
          vector. See :func:`emg3d.meshes.estimate_gridding_opts`.
        - ``expand``: in the format of ``[property_sea, property_air]``; if
          provided, the input model is expanded up to the seasurface with sea
          water, and an air layer is added. The actual height of the seasurface
          can be defined with the key ``seasurface``. See
          :func:`emg3d.models.expand_grid_model`.

    solver_opts : dict, default: {'verb': 1'}
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

    receiver_interpolation : str, default: 'cubic':
        Interpolation method to obtain the response at receiver location;
        'cubic' or 'linear'. Cubic is more precise. However, if you are
        interested in the gradient, you need to choose 'linear' at the moment,
        as there are only linearly interpolated source functions. To be the
        proper adjoint for the gradient the receiver has to be interpolated
        linearly too.

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
        self.receiver_interpolation = kwargs.pop(
                'receiver_interpolation', 'cubic')

        # Assemble solver_opts.
        self.solver_opts = {
                'verb': 1,  # Default verbosity, can be overwritten.
                'log': -1,  # Default only log, can be overwritten.
                **kwargs.pop('solver_opts', {}),  # User setting.
                'return_info': True,  # return_info=True is forced.
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

        # Check the grid if one was explicitly provided.
        if gridding == 'same':
            meshes.check_mesh(self.model.grid)
        elif gridding == 'input':
            meshes.check_mesh(self._grid_single)

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
                    receiver=rec_coord_tuple(erec),
                    method=self.receiver_interpolation,
            )

            # Store the receiver response.
            self.data.synthetic.loc[source, :, freq][erec] = resp

        # Store magnetic receivers.
        if rec_types.count(False):

            # Extract data at receivers.
            mrec = np.nonzero(np.logical_not(rec_types))[0]
            resp = self.get_hfield(source, freq).get_receiver(
                    receiver=rec_coord_tuple(mrec),
                    method=self.receiver_interpolation,
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
            If True, it stores the current `synthetic` responses also as
            `observed` responses.

        add_noise : bool, default: True
            Boolean if to add noise to observed data (if ``observed=True``).
            All remaining ``kwargs`` are forwarded to
            :meth:`emg3d.surveys.Survey.add_noise`.

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

            # Copy synthetic to observed.
            self.data['observed'] = self.data['synthetic'].copy()

            # Add noise.
            if kwargs.get('add_noise', True):
                self.survey.add_noise(**kwargs)

    # OPTIMIZATION
    @property
    def gradient(self):
        r"""Compute the discrete gradient using the adjoint-state method.

        The discrete adjoint-state gradient for a single source at a single
        frequency is given by Equation (10) in [PlMu08]_,

        .. math::

            \nabla_p \phi(\textbf{p}) =
                -&\sum_{k,l,m}\mathbf{\bar{\lambda}}_{x; k+\frac{1}{2}, l, m}
                  \frac{\partial S_{k+\frac{1}{2}, l, m}}{\partial \textbf{p}}
                  \textbf{E}_{x; k+\frac{1}{2}, l, m}\\
                -&\sum_{k,l,m}\mathbf{\bar{\lambda}}_{y; k, l+\frac{1}{2}, m}
                  \frac{\partial S_{k, l+\frac{1}{2}, m}}{\partial \textbf{p}}
                  \textbf{E}_{y; k, l+\frac{1}{2}, m}\\
                -&\sum_{k,l,m}\mathbf{\bar{\lambda}}_{z; k, l, m+\frac{1}{2}}
                  \frac{\partial S_{k, l, m+\frac{1}{2}}}{\partial \textbf{p}}
                  \textbf{E}_{z; k, l, m+\frac{1}{2}}\, ,


        where :math:`\textbf{E}` is the electric (forward) field and
        :math:`\mathbf{\lambda}` is the back-propagated residual field (from
        electric and magnetic receivers); :math:`\bar{~}` denotes conjugate.
        The :math:`\partial S`-part takes care of the volume-averaged model
        parameters.

        .. warning::

            To obtain the proper adjoint-state gradient you have to choose
            linear interpolation for the receiver responses:
            ``emg3d.Simulation(..., receiver_interpolation='linear')``. The
            reason is that the point-source is the adjoint of a tri-linear
            interpolation, so the residual should come from a linear
            interpolation.

            Also, the adjoint test for magnetic receivers does not yet pass.
            Electric receivers are good to go.

        .. note::

            The currently implemented gradient is only for isotropic models
            without relative electric permittivity nor relative magnetic
            permeability.


        Returns
        -------
        grad : ndarray
            Adjoint-state gradient (same shape as ``simulation.model``).

        """
        if self._gradient is None:

            # Warn that cubic is not good for adjoint-state gradient.
            if self.receiver_interpolation == 'cubic':
                msg = (
                    "emg3d: Receiver responses were obtained with cubic "
                    "interpolation. This will not yield the exact gradient. "
                    "Change `receiver_interpolation='linear'` in the call to "
                    "Simulation()."
                )
                warnings.warn(msg, UserWarning)

            # Check limitation 1: So far only isotropic models.
            if self.model.case != 'isotropic':
                raise NotImplementedError(
                    "Gradient only implemented for isotropic models."
                )

            # Check limitation 2: No epsilon_r, mu_r.
            var = (self.model.epsilon_r, self.model.mu_r)
            for v, n in zip(var, ('el. permittivity', 'magn. permeability')):
                if v is not None and not np.allclose(v, 1.0):
                    raise NotImplementedError(
                        f"Gradient not implemented for {n}."
                    )

            # Ensure misfit has been computed
            # (and therefore the electric fields).
            _ = self.misfit

            # Compute back-propagating electric fields.
            self._bcompute()

            # Pre-allocate the gradient on the mesh.
            gradient_model = np.zeros(self.model.grid.shape_cells, order='F')

            # Loop over source-frequency pairs.
            for src, freq in self._srcfreq:

                # Multiply forward field with backward field; take real part.
                # This is the actual Equation (10), with:
                #   del S / del p = iwu0 V sigma / sigma,
                # where lambda and E are already volume averaged.
                efield = self._dict_efield[src][freq]  # Forward electric field
                bfield = self._dict_bfield[src][freq]  # Conj. backprop. field
                gfield = fields.Field(
                    grid=efield.grid,
                    data=-np.real(bfield.field * efield.smu0 * efield.field),
                    dtype=float,
                )

                # Bring the gradient back from the computation grid to the
                # model grid.
                gradient = gfield.interpolate_to_grid(self.model.grid)

                # Pre-allocate the gradient for this src-freq.
                shape = gradient.grid.shape_cells
                grad_x = np.zeros(shape, order='F')
                grad_y = np.zeros(shape, order='F')
                grad_z = np.zeros(shape, order='F')

                # Map the field to cell centers times volume.
                cell_volumes = gradient.grid.cell_volumes.reshape(
                        shape, order='F')
                maps.interp_edges_to_vol_averages(
                        ex=gradient.fx, ey=gradient.fy, ez=gradient.fz,
                        volumes=cell_volumes,
                        ox=grad_x, oy=grad_y, oz=grad_z)
                grad = grad_x + grad_y + grad_z

                # => Frequency-dependent depth-weighting should go here.

                # Add this src-freq gradient to the total gradient.
                gradient_model -= grad

            # => Frequency-independent depth-weighting should go here.

            # Apply derivative-chain of property-map
            # (only relevant if `mapping` is something else than conductivity).
            self.model.map.derivative_chain(
                    gradient_model, self.model.property_x)

            self._gradient = gradient_model

        return self._gradient[:, :, :self._input_sc2]

    @property
    def misfit(self):
        r"""Misfit or cost function.

        The data misfit or weighted least-squares functional using an
        :math:`l_2` norm is given by

        .. math::
            :label: misfit

                \phi = \frac{1}{2} \sum_s\sum_r\sum_f
                    \left\lVert
                        W_{s,r,f} \left(
                        \textbf{d}_{s,r,f}^\text{pred}
                        -\textbf{d}_{s,r,f}^\text{obs}
                        \right) \right\rVert^2 \, ,

        where :math:`s, r, f` stand for source, receiver, and frequency,
        respectively; :math:`\textbf{d}^\text{obs}` are the observed electric
        and magnetic data, and :math:`\textbf{d}^\text{pred}` are the synthetic
        electric and magnetic data. As of now the misfit does not include any
        regularization term.

        The data weight of observation :math:`d_i` is given by :math:`W_i =
        \varsigma^{-1}_i`, where :math:`\varsigma_i` is the standard deviation
        of the observation, see
        :attr:`emg3d.surveys.Survey.standard_deviation`.

        .. note::

            You can easily implement your own misfit function (to include,
            e.g., a regularization term) by monkey patching this misfit
            function with your own::

                @property  # misfit is a property
                def my_misfit_function(self):
                    '''Returns the misfit as a float.'''

                    if self._misfit is None:
                        self.compute()  # Ensures fields are computed.

                        # Computing your misfit...
                        self._misfit = your misfit

                    return self._misfit

                # Monkey patch simulation.misfit:
                emg3d.simulation.Simulation.misfit = my_misfit_function

                # And now all the regular stuff, initiate a Simulation etc
                simulation = emg3d.Simulation(survey, grid, model)
                simulation.misfit
                # => will return your misfit
                #   (will also be used for the adjoint-state gradient).


        Returns
        -------
        misfit : float
            Value of the misfit function.

        """
        if self._misfit is None:

            # Check if electric fields have already been computed.
            test_efield = sum(
                [1 if self._dict_efield[src][freq] is None else 0
                 for src, freq in self._srcfreq]
            )

            if test_efield:
                self.compute()

            # Check if weights are stored already. (weights are currently
            # simply 1/std^2; but might change in the future).
            if 'weights' not in self.data.keys():

                # Get standard deviation, raise warning if not set.
                std = self.survey.standard_deviation
                if std is None:
                    raise ValueError(
                        "Either `noise_floor` or `relative_error` or both "
                        "must be provided (>0) to compute the "
                        "`standard_deviation`. It can also be set directly "
                        "(same shape as data). The standard deviation is "
                        "required to compute the misfit."
                    )

                # Store weights
                self.data['weights'] = std**-2

            # Calculate and store residual.
            residual = self.data.synthetic - self.data.observed
            self.data['residual'] = residual

            # Get weights, calculate misfit.
            weights = self.data['weights']
            self._misfit = np.sum(weights*(residual.conj()*residual)).real/2

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

                # TODO: The adjoint test for magnetic receivers does not pass.
                src_fct = electrodes.TxMagneticDipole

                # If the data is from a magnetic point we have to undo another
                # factor smu0 here, as the source will be a loop.
                strength /= rfield.smu0

            else:
                src_fct = electrodes.TxElectricPoint

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

        # If not verbose, return.
        if verb < 0:
            return

        # Get info dict.
        info = getattr(self, f"_dict_{field}_info", {})
        out = ""

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
                if verb == 0 or self.solver_opts['verb'] != 1:
                    out += f" {cinfo['exit_message']}\n"

                if verb == 1 and self.solver_opts['verb'] == 1:
                    out += f" {cinfo['log'][12:]}"

                if verb == 1 and self.solver_opts['verb'] > 1:
                    out += f"\n{cinfo['log']}\n"

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

                model = models.expand_grid_model(model, expand, interface)

            # Get automatic gridding input.
            # Estimate the parameters from survey and model if not provided.
            gridding_opts = meshes.estimate_gridding_opts(
                    g_opts, model, self.survey, self._input_sc2)

        self.gridding_opts = gridding_opts
        self.model = model


# DEPRECATED, will be removed from v1.4.0 onwards.
def expand_grid_model(model, expand, interface):
    """Deprecated, moved to models: `emg3d.models.expand_grid_model`."""
    msg = ("emg3d: `expand_grid_model` moved from `simulations` to `models`. "
           "Its availability in `simulations` will be removed in v1.4.0.")
    warnings.warn(msg, FutureWarning)
    return models.expand_grid_model(model, expand, interface)


def estimate_gridding_opts(gridding_opts, model, survey, input_sc2=None):
    """Deprecated, moved to meshes: `emg3d.meshes.estimate_gridding_opts`."""
    msg = ("emg3d: `estimate_gridding_opts` moved from `simulations` to "
           "`meshes`. Its availability in `simulations` will be removed in "
           "v1.4.0.")
    warnings.warn(msg, FutureWarning)
    return meshes.estimate_gridding_opts(
                gridding_opts, model, survey, input_sc2)
