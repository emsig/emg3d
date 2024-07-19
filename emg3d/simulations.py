"""
A simulation is the computation (modelling) of the electromagnetic responses
due to a given model and survey.

The simulation module combines the different pieces of ``emg3d`` providing
a high-level, specialised modelling tool for the end user.
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

import os
import warnings
import itertools
from pathlib import Path
from copy import deepcopy

import numpy as np

from emg3d import fields, io, maps, meshes, models, surveys, utils

__all__ = ['Simulation', ]


def __dir__():
    return __all__


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
          NOTE: ``expand`` is deprecated in v1.7.0, and will be removed in
          v1.9.0. A property-complete model has to be provided.

    solver_opts : dict, default: {'verb': 1'}
        Passed through to :func:`emg3d.solver.solve`. The dict can contain any
        parameter that is accepted by the :func:`emg3d.solver.solve` except for
        ``model``, ``sfield``, ``efield``, ``return_info``, and ``log``.

        In addition to the regular parameter ``tol``, one can provide a
        parameter ``tol_gradient``. This tolerance will be used when calling
        ``gradient``/``jtvec`` and ``jvec``. By default, it is set to the same
        value as ``tol``, which is used for ``compute``. However, for
        inversions it is usually possible to relax this tolerance (e.g.,
        ``tol=1e-6``, ``tol_gradient=1e-3``).

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

    file_dir : str, default: None
        Absolute or relative path, where temporary files should be stored. By
        default, everything is done in memory. However, for large models with
        many sources and frequencies this can become memory consuming.
        Providing a ``file_dir`` can help with this. Fields and models are then
        stored to disk, and each process accesses the files it needs. There is
        only a gain if there are more source-frequency pairs than concurrent
        running processes.

        Note that the directory is created if it does not exist. However, the
        parent directory must exist.

        Also note that the files are stored as .h5-files, and you need to have
        ``h5py`` installed to use this feature.

    tqdm_opts : {bool, dict}, default: True
        Boolean if a progress bar should be shown (only if ``tqdm`` is
        installed). If a dict is provided it will be forwarded to ``tqdm``.

    layered : bool, default: False
        If True, the responses are computed with approximated layered (1D)
        models for each source-receiver pair using empymod.

        The computation happens in parallel for each source location. Each
        source-receiver pair is done separately, but all frequencies at once.

        If layered is set, the gradient is computed using the finite-difference
        method, by perturbing each layer slightly.

        Current limitations:

        - Only point and dipole sources.
        - Only isotropic and VTI models.

        Setting this to True also means:

        - There are no {e;h}fields, only the fields at receiver locations.
        - ``gridding``, most of ``gridding_opts``, ``solver_opts``,
          ``receiver_interpolation``, and ``file_dir`` have no effect.
        - The attribute :attr:`emg3d.simulations.Simulation.jvec`` is not
          implemented.

    layered_opts : dict, default: {}
        Options passed to :attr:`emg3d.models.Model.extract_1d`, defining how
        the layered model is obtained from the 3D model. ``p0`` and ``p1``
        are taken to be the source and receiver locations. Consult that
        function for more information. Below are only things described which
        differ.

        The possible methods are: cylinder, prism, midpoint, source, receiver.
        The last two are the same as ``midpoint``, where just both points are
        set either to the source or receiver location, respectively.

        The default method is ``'cylinder'``. If the ellipse parameters are not
        given for the methods cylinder and prism, they are set as follows:

        - factor: 1.2.
        - minor: 0.8.
        - radius: one skin depth using the lowest frequency of the survey and
          the downwards property from the gridding options (or the minimum
          conductivity value in the lowest vertical layer).

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

        # Assemble solver_opts and store the tolerances separately.
        self.solver_opts = {
                'verb': 1,  # Default verbosity, can be overwritten.
                'log': -1,  # Default only log, can be overwritten.
                **kwargs.pop('solver_opts', {}),  # User setting.
                'return_info': True,  # return_info=True is forced.
        }
        self.tol_forward = self.solver_opts.get('tol', 1e-6)
        self.tol_gradient = self.solver_opts.pop(
                'tol_gradient', self.tol_forward)

        # Initiate dictionaries and other values with None's.
        self._dict_grid = self._dict_initiate
        self._dict_efield = self._dict_initiate
        self._dict_efield_info = self._dict_initiate
        self._gradient = None
        self._misfit = None
        self._computed = False

        # Initiate file_dir
        self.file_dir = kwargs.pop('file_dir', None)
        if self.file_dir:
            self.file_dir = os.path.abspath(self.file_dir)
            # Create directory if it doesn't exist yet.
            Path(self.file_dir).mkdir(exist_ok=True)

        # Get model taking gridding_opts into account.
        # Sets self.model and self.gridding_opts.
        self._set_model(model, kwargs)
        self._set_layered_opts(kwargs.pop('layered', False),
                               kwargs.pop('layered_opts', {}))

        # Initiate synthetic data with NaN's if they don't exist.
        if 'synthetic' not in self.survey.data.keys():
            self.survey.data['synthetic'] = self.data.observed.copy(
                    data=np.full(self.survey.shape, np.nan+1j*np.nan))

        # `tqdm`-options.
        tqdm_opts = kwargs.pop('tqdm_opts', {})
        if isinstance(tqdm_opts, bool):
            tqdm_opts = {'disable': not tqdm_opts}
        self._tqdm_opts = {
            **{'bar_format': '{desc} {bar} {n_fmt}/{total_fmt}  [{elapsed}]'},
            **tqdm_opts,
        }

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
                f"- {self._info_grids}")

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
                f"  <li>{self._info_grids}</li>"
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
            for name in ['_dict_grid', ]:
                delattr(self, name)
                setattr(self, name, self._dict_initiate)

        # Clean field-dicts.
        if what in ['computed', 'keepresults', 'all']:

            # These exist always and have to be initiated.
            for name in ['_dict_efield', '_dict_efield_info']:
                delattr(self, name)
                setattr(self, name, self._dict_initiate)

            # These only exist with gradient; don't initiate them.
            for name in ['_dict_bfield', '_dict_bfield_info']:
                if hasattr(self, name):
                    delattr(self, name)

            # Remove files if they exist.
            if self.file_dir:
                for p in Path(self.file_dir).glob('[ebg]field_*.h5'):
                    p.unlink()

        # Clean data.
        if what in ['computed', 'all']:
            self._computed = False
            for key in ['residual', 'weights']:
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
        self.solver_opts['tol'] = self.tol_forward
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
            'tqdm_opts': self._tqdm_opts,
            'layered': self.layered,
            'layered_opts': self.layered_opts,
            'receiver_interpolation': self.receiver_interpolation,
            'tol_gradient': self.tol_gradient,
            'file_dir': self.file_dir,
            '_input_sc2': self._input_sc2,
        }

        # Clean unwanted data if plain.
        if what == 'plain':
            for key in ['synthetic', 'residual', 'weights']:
                if key in out['survey']['data'].keys():
                    del out['survey']['data'][key]

        # Store wanted dicts.
        if what in ['computed', 'all']:
            for name in ['_dict_grid',
                         '_dict_efield', '_dict_efield_info',
                         '_dict_bfield', '_dict_bfield_info']:
                if hasattr(self, name):
                    out[name] = getattr(self, name)

        # Store gradient and misfit.
        if what in ['computed', 'results', 'all']:
            out['gradient'] = self._gradient
            out['misfit'] = self._misfit
            out['computed'] = self._computed

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
        cls_inp['receiver_interpolation'] = inp.pop(
                'receiver_interpolation', 'cubic')
        cls_inp['file_dir'] = inp.pop('file_dir', None)
        cls_inp['tqdm_opts'] = inp.pop('tqdm_opts', {})
        cls_inp['layered'] = inp.pop('layered', False)
        cls_inp['layered_opts'] = inp.pop('layered_opts', {})
        cls_inp['solver_opts'] = cls_inp['solver_opts'].copy()
        cls_inp['solver_opts']['tol_gradient'] = inp.pop(
                'tol_gradient', cls_inp['solver_opts'].get('tol', 1e-6))

        # Instantiate the class.
        out = cls(**cls_inp)

        # Add existing derived/computed properties.
        data = ['_dict_grid',
                '_dict_efield', '_dict_efield_info',
                '_dict_bfield', '_dict_bfield_info']
        for name in data:
            if name in inp.keys():
                values = inp.pop(name)

                # De-serialize Model, Field, and TensorMesh instances.
                io._dict_deserialize(values)

                setattr(out, name, values)

        # Add gradient and misfit.
        data = ['gradient', 'misfit', 'computed']
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

            - ``'computed'``, ``'all'``:
              Stores all computed properties: electric fields and responses at
              receiver locations.
            - '``results'``:
              Stores only the response at receiver locations.
            - ``'plain'``:
              Only stores the plain Simulation (as initiated).

            Note that if ``file_dir`` is set, those files will remain there.

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
        grid = self.get_grid(source, self._freq_inp2key(frequency))
        return self.model.interpolate_to_grid(grid)

    def get_efield(self, source, frequency):
        """Return electric field for given source and frequency."""
        freq = self._freq_inp2key(frequency)

        # If it doesn't exist yet, compute it.
        if self._dict_get('efield', source, freq) is None:
            self.compute(source=source, frequency=freq)

        return self._dict_get('efield', source, freq)

    def get_hfield(self, source, frequency):
        """Return magnetic field for given source and frequency."""
        freq = self._freq_inp2key(frequency)

        # If electric field not computed yet compute it.
        if self._dict_get('efield', source, freq) is None:
            self.compute(source=source, frequency=freq)

        # Return magnetic field.
        return fields.get_magnetic_field(
            self.get_model(source, freq),
            self._dict_get('efield', source, freq),
        )

    def get_efield_info(self, source, frequency):
        """Return the solver information of the corresponding computation."""
        freq = self._freq_inp2key(frequency)
        return self._dict_get('efield_info', source, freq)

    def _dict_get(self, which, source, frequency):
        """Return source-frequency pair from dictionary `which`.

        Thin wrapper for ``self._dict_{which}[{source}][{frequency}]``, that
        works as well for file-based computations.
        """
        value = getattr(self, f"_dict_{which}")[source][frequency]
        return self._load(value, ['efield', 'info']['info' in which])

    def _load(self, value, what):
        """Returns `value` (memory) or loads `value[what]` (files)."""
        if self.file_dir and value is not None:
            return io.load(value, verb=0)[what]
        else:
            return value

    def _data_or_file(self, what, source, frequency, data):
        """Return data or file-name for given what, source, and frequency."""
        if self.file_dir:
            fname = os.path.join(
                self.file_dir, f"{what}_{source}_{frequency}.h5")
            io.save(fname, data=data, verb=0)
            return fname
        else:
            return data

    def _get_responses(self, source, frequency, efield=None):
        """Return electric and magnetic fields at receiver locations."""

        # Get receiver types and their coordinates.
        erec, mrec = self.survey._irec_types
        erec_coord, mrec_coord = self.survey._rec_types_coord(source)

        # Initiate output.
        resp = np.zeros_like(self.data.synthetic.loc[source, :, frequency])

        # efield of this source/frequency if not provided.
        if efield is None:
            efield = self._dict_get('efield', source, frequency)

        if erec.size:

            # Store electric receivers.
            resp[erec] = efield.get_receiver(
                receiver=erec_coord, method=self.receiver_interpolation,
            )

        if mrec.size:

            # Compute magnetic field.
            hfield = fields.get_magnetic_field(
                self.get_model(source, frequency), efield,
            )

            # Store electric receivers.
            resp[mrec] = hfield.get_receiver(
                receiver=mrec_coord, method=self.receiver_interpolation,
            )

        return resp

    # ASYNCHRONOUS COMPUTATION
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
        source = kwargs.pop('source', None)
        frequency = kwargs.pop('frequency', None)
        if self.layered:
            if source or frequency:
                raise NotImplementedError("No fields if `layered` is used.")
            self._compute_1d()
        else:
            # Undocumented (internal):
            # If the call is from `get_efield`, it will have source/frequency.
            # This use is only internal. End users should use `get_efield()`.
            self._compute([(source, frequency), ])

        # If it shall be used as observed data save a copy.
        if observed:

            # Copy synthetic to observed.
            self.data['observed'] = self.data['synthetic'].copy()

            # Add noise.
            if kwargs.pop('add_noise', True):
                self.survey.add_noise(**kwargs)

        elif source is None and frequency is None:
            self._computed = True

    def _compute(self, srcfreq):
        """Compute efields and responses asynchronously using emg3d.solve()."""
        from emg3d import _multiprocessing as _mp

        if not srcfreq[0][0]:
            # "Normal" case: all source-frequency pairs.
            srcfreq = self._srcfreq

        # Create iterable from src/freq-list for parallel computation.
        def collect_efield_inputs(inp):
            """Collect inputs."""
            source, freq = inp

            data = {
                'model': self.model,
                'grid': self.get_grid(source, freq),
                'source': self.survey.sources[source],
                'frequency': self.survey.frequencies[freq],
                # efield is None if not comp. yet; else it is the solution.
                'efield': self._dict_get('efield', source, freq),
                'solver_opts': self.solver_opts,
            }
            data['solver_opts']['tol'] = self.tol_forward
            return self._data_or_file('efield', source, freq, data)

        # Compute fields in parallel.
        out = _mp.process_map(
            _mp.solve,
            list(map(collect_efield_inputs, srcfreq)),
            max_workers=self.max_workers,
            **{'desc': 'Compute efields', **self._tqdm_opts},
        )

        # Loop over src-freq combinations to extract and store.
        for i, (src, freq) in enumerate(srcfreq):

            # Store efield and solver info.
            self._dict_efield[src][freq] = out[i][0]
            self._dict_efield_info[src][freq] = out[i][1]

            # Store responses at receiver locations.
            resp = self._get_responses(src, freq)
            self.data['synthetic'].loc[src, :, freq] = resp

        # Print solver info.
        self.print_solver_info('efield', verb=self.verb)

    def _compute_1d(self, gradient=False):
        """Compute responses asynchronously using empymod.bipole()."""
        from emg3d import _multiprocessing as _mp

        # Check if there are observed data.
        has_data = np.isfinite(self.data.observed.data).sum() > 0

        # Create iterable from src-list for parallel computation.
        def collect_empymod_inputs(source):
            """Collect inputs."""

            data = {
                'model': self.model,
                'src': self.survey.sources[source],
                'receivers': self.survey.receivers,
                'frequencies': self.survey.frequencies,
                'empymod_opts': self._empymod_opts,
                'observed': None,
                'layered_opts': self.layered_opts,
                'gradient': gradient,
            }

            # If there is data, add it.
            if has_data:
                data['observed'] = self.data.observed.loc[source, :, :]

            # For the gradient we also need the residuals and weights.
            if gradient:
                data['residual'] = self.data.residual.loc[source, :, :]
                data['weights'] = self.data.weights.loc[source, :, :]

            return data

        # Compute responses in parallel.
        out = _mp.process_map(
            _mp.layered,
            list(map(collect_empymod_inputs, self.survey.sources.keys())),
            max_workers=self.max_workers,
            **{'desc': 'Compute empymod', **self._tqdm_opts},
        )

        # If gradient, return it.
        if gradient:

            # Pre-allocate the gradient on the mesh.
            grad = np.zeros((3, *self.model.grid.shape_cells), order='F')

            # Sum all sources up and return.
            for val in out:
                grad += val

            return grad

        # If forward, store responses in synthetic.
        else:

            # Loop over sources to extract and store.
            for i, src in enumerate(self.survey.sources.keys()):
                self.data['synthetic'].loc[src, :, :] = out[i]

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

        .. note::

            The currently implemented gradient does only work for models
            without relative electric permittivity nor relative magnetic
            permeability.

        .. note::

            If ``layered=True``, the gradient is computed using finite
            differences.


        Returns
        -------
        grad : ndarray
            Adjoint-state gradient. Shape depends on the anisotropy type:

            - isotropic: (nx, ny, nz)
            - HTI/VTI: (2, nx, ny, nz)
            - triaxial: (3, nx, ny, nz)

        """
        if self._gradient is None:

            # Ensure misfit has been computed
            # (and therefore the electric fields).
            _ = self.misfit

            if self.layered:  # 1D finite-difference gradient.
                gradient = self._compute_1d(gradient=True)

            else:             # 3D adjoint-state gradient

                # Warn that cubic is not good for adjoint-state gradient.
                if self.receiver_interpolation == 'cubic':
                    msg = (
                        "emg3d: Receiver responses were obtained with cubic "
                        "interpolation. This will not yield the exact "
                        "gradient. Change `receiver_interpolation='linear'` "
                        "in the call to Simulation()."
                    )
                    warnings.warn(msg, UserWarning)

                # Check limitation: No epsilon_r, mu_r.
                var = (self.model.epsilon_r, self.model.mu_r)
                nam = ('el. permittivity', 'magn. permeability')
                for v, n in zip(var, nam):
                    if v is not None and not np.allclose(v, 1.0):
                        raise NotImplementedError(
                            f"Gradient not implemented for {n}."
                        )

                # Compute back-propagating electric fields.
                self._bcompute()

                # Pre-allocate the gradient on the mesh.
                gradient = np.zeros((3, *self.model.shape), order='F')

                # Loop over source-frequency pairs.
                for src, freq in self._srcfreq:

                    efield = self._dict_get('efield', src, freq)
                    bfield = self._dict_get('bfield', src, freq)

                    # Multiply forward field with backward; take real part.
                    gfield = fields.Field(
                        grid=efield.grid,
                        data=np.real(bfield.field*efield.smu0*efield.field),
                    )

                    # Pre-allocate the gradient for the computational grid.
                    shape = gfield.grid.shape_cells
                    grad = np.zeros((3, *shape), order='F')

                    # Map the field to cell centers times volume.
                    cell_volumes = gfield.grid.cell_volumes
                    maps.interp_edges_to_vol_averages(
                            ex=gfield.fx, ey=gfield.fy, ez=gfield.fz,
                            volumes=cell_volumes.reshape(shape, order='F'),
                            ox=grad[0, ...], oy=grad[1, ...], oz=grad[2, ...])

                    # Bring gradient back from computation grid to inversion
                    # grid and add it to the total gradient.
                    if self.model.grid != gfield.grid:
                        # Wrapped in own function as it requires discretize.
                        maps._interp_volume_average_adj(
                            oval=gradient, ogrid=self.model.grid,
                            nval=grad, ngrid=gfield.grid,
                        )
                    else:
                        gradient += grad

            # Apply derivative-chain of property-map (only relevant if
            # `mapping` is something else than conductivity) and collect.
            indices = [0, ]
            if self.model.case in ['HTI', 'triaxial']:
                self.model.map.derivative_chain(
                        gradient[1, ...], self.model.property_y)
                indices.append(1)
            else:
                gradient[0, ...] += gradient[1, ...]

            if self.model.case in ['VTI', 'triaxial']:
                self.model.map.derivative_chain(
                        gradient[2, ...], self.model.property_z)
                indices.append(2)
            else:
                gradient[0, ...] += gradient[2, ...]

            self.model.map.derivative_chain(
                    gradient[0, ...], self.model.property_x)

            # Select required directions, excluded "expanded" layers & squeeze.
            self._gradient = gradient[indices, ..., :self._input_sc2].squeeze()

        return self._gradient

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

            # Ensure efields are computed
            if not self._computed:
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

        return self._misfit.data

    def _bcompute(self):
        """Compute bfields asynchronously for all sources and frequencies."""
        from emg3d import _multiprocessing as _mp

        # Initiate back-propagated electric field and info dicts.
        if not hasattr(self, '_dict_bfield'):
            self._dict_bfield = self._dict_initiate
            self._dict_bfield_info = self._dict_initiate

        # Create iterable from src/freq-list for parallel computation.
        def collect_bfield_inputs(inp):
            """Collect inputs."""
            source, freq = inp

            data = {
                'model': self.model,
                'sfield': self._get_rfield(source, freq),
                # bfield is None unless it was explicitly set.
                'efield': self._dict_get('bfield', source, freq),
                'solver_opts': self.solver_opts
            }
            data['solver_opts']['tol'] = self.tol_gradient
            return self._data_or_file('bfield', source, freq, data)

        # Compute fields in parallel.
        out = _mp.process_map(
            _mp.solve,
            list(map(collect_bfield_inputs, self._srcfreq)),
            max_workers=self.max_workers,
            **{'desc': 'Back-propagate', **self._tqdm_opts},
        )

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

        # Get values for this source and frequency.
        grid = self.get_grid(source, frequency)
        residual = self.data.residual.loc[source, :, frequency].data
        weight = self.data.weights.loc[source, :, frequency].data

        # Initiate empty residual source field.
        rfield = fields.Field(grid, frequency=freq)

        # Residual source strength: Weighted residual, normalized by -smu0.
        strength = np.conj(residual * weight / -rfield.smu0)

        # Loop over receivers, input as source.
        for i, rec in enumerate(self.survey.receivers.values()):

            # Skip if no data.
            if np.isnan(residual[i]):
                continue

            # Get absolute coordinates as fct of source.
            # (Only relevant in case of "relative" receivers.)
            coords = rec.coordinates_abs(self.survey.sources[source])

            # Create adjoint source.
            src = rec._adjoint_source(coords, strength=strength[i])

            # Get and add this source field.
            rfield.field += src.get_field(grid=grid, frequency=freq).field

        return rfield

    @utils._requires('discretize')
    def jvec(self, vector):
        r"""Compute the sensitivity times a vector.

        .. math::
            :label: jvec

            J v = P A^{-1} G v \ ,

        where :math:`v` has size of the model.


        Parameters
        ----------
        vector : ndarray
            Vector applied to J. Shape depends on the anisotropy type:

            - isotropic: (nx, ny, nz) or (1, nx, ny, nz)
            - HTI/VTI: (2, nx, ny, nz)
            - triaxial: (3, nx, ny, nz)


        Returns
        -------
        jvec : ndarray
            Shape of the data.

        """
        from emg3d import _multiprocessing as _mp

        if self.layered:
            msg = "`jvec` is not implemented for `layered`."
            raise NotImplementedError(msg)

        # Missing for jvec/jtvec
        # - Refactor `compute/gradient/_bcompute/_get_rfield/jvec/jtvec`.
        # - Document properly jvec and jtvec.
        # - Would gradient (?, nx, ny, nz) be better a Model-like instance?

        # Ensure misfit has been computed (and therefore the electric fields).
        _ = self.misfit

        # Apply derivative-chain of property-map (copy to not overwrite).
        if vector.ndim == 3:
            vector = vector[None, ...].copy()
        else:
            vector = vector.copy()

        self.model.map.derivative_chain(vector[0, ...], self.model.property_x)
        if self.model.case in ['HTI', 'triaxial']:
            self.model.map.derivative_chain(
                    vector[1, ...], self.model.property_y)
        if self.model.case in ['VTI', 'triaxial']:
            n = 1 if self.model.case == 'VTI' else 2
            self.model.map.derivative_chain(
                    vector[n, ...], self.model.property_z)

        # Interpolation options.
        iopts = {'method': 'volume', 'extrapolate': True,
                 'log': False, 'grid': self.model.grid}

        # Create iterable from src/freq-list for parallel computation.
        def collect_gfield_inputs(inp, vector=vector):
            """Collect inputs."""
            source, freq = inp

            # Forward electric field
            efield = self._dict_get('efield', source, freq)

            # Interpolate to computational grid.
            cvector = [
                maps.interpolate(values=v, xi=efield.grid, **iopts).ravel('F')
                for v in vector[:, ...]
            ]

            if self.model.case == 'isotropic':
                ncase = 1
                cvector = cvector[0]
            else:
                ncase = 3
                if self.model.case == 'HTI':
                    cvector = np.r_[cvector[0], cvector[1], cvector[0]]
                elif self.model.case == 'VTI':
                    cvector = np.r_[cvector[0], cvector[0], cvector[1]]
                else:
                    cvector = np.r_[cvector].ravel()

            # Compute gvec = G * vector (using discretize).
            gvec = efield.grid.get_edge_inner_product_deriv(
                np.ones(efield.grid.n_cells*ncase)
            )(efield.field) * cvector

            # Create source field.
            gfield = fields.Field(
                grid=efield.grid,
                data=-efield.smu0*gvec,  # -iwu: To get complete source field.
                frequency=efield.frequency
            )

            data = {
                'model': self.model,
                'sfield': gfield,
                'efield': None,
                'solver_opts': self.solver_opts,
            }
            data['solver_opts']['tol'] = self.tol_gradient
            return self._data_or_file('gfield', source, freq, data)

        # Compute fields (A^-1 * G * vector) in parallel.
        out = _mp.process_map(
            _mp.solve,
            list(map(collect_gfield_inputs, self._srcfreq)),
            max_workers=self.max_workers,
            **{'desc': 'Compute jvec', **self._tqdm_opts},
        )

        # Initiate jvec data with NaN's if it doesn't exist.
        if 'jvec' not in self.data.keys():
            self.data['jvec'] = self.data.observed.copy(
                    data=np.full(self.survey.shape, np.nan+1j*np.nan))

        # Loop over src-freq combinations to extract and store.
        for i, (src, freq) in enumerate(self._srcfreq):
            gfield = self._load(out[i][0], 'efield')
            resp = self._get_responses(src, freq, gfield)
            self.data['jvec'].loc[src, :, freq] = resp

        return self.data['jvec'].data

    def jtvec(self, vector):
        r"""Compute the sensitivity transpose times a vector.

        If ``vector=residual*weights``, ``jtvec`` corresponds to the
        ``gradient``.

        .. math::
            :label: jtvec

            J^H v = G^H A^{-H} P^H v \ ,

        where :math:`v` has the shape of the data.


        Parameters
        ----------
        vector : ndarray, DataArray
            An array with the shape of the data, or directly a DataArray as
            stored in the :class:`emg3d.surveys.Survey`.


        Returns
        -------
        jtvec : ndarray
            Adjoint-state gradient for the provided vector. Shape depends on
            the anisotropy type:

            - isotropic: (nx, ny, nz)
            - HTI/VTI: (2, nx, ny, nz)
            - triaxial: (3, nx, ny, nz)

        """

        # Replace residual by provided vector
        # (division by weight is undone in gradient).
        with np.errstate(invalid='ignore'):  # (For division by cplx-NaN.)
            self.data.residual[...] = vector/self.data.weights.data

        # Reset gradient, so it will be computed.
        self._gradient = None
        for name in ['_dict_bfield', '_dict_bfield_info']:
            if hasattr(self, name):
                delattr(self, name)

        # Return gradient from weighted residual `vector`.
        return self.gradient

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

        info = "Gridding: "

        if self.layered:
            info += "layered computation using method "
            info += f"'{self.layered_opts['method']}'"

            if self.layered_opts['method'] in ['prism', 'cylinder']:
                opts = '; '.join(
                    [f"{k}: {v:.2f}" for k, v in
                     self.layered_opts['ellipse'].items()]
                )
                info += "; "+opts

            return info

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
        info += f"{self._gridding_descr[self.gridding]}; "
        info += f"{min_vc[0]} x {min_vc[1]} x {min_vc[2]} ({min_nc:,})"
        if has_minmax:
            info += f" - {max_vc[0]} x {max_vc[1]} x {max_vc[2]} ({max_nc:,})"

        return info

    def print_grid_info(self, verb=1, return_info=False):
        """Print info for all generated grids."""

        # Act depending on gridding:
        out = ""

        # If layered, return.
        if self.layered:
            return out if return_info else None

        def get_grid_info(src, freq):
            """Return grid info for given source and frequency."""
            grid = self.get_grid(src, freq)
            out = ''
            if verb != 0 and hasattr(grid, 'construct_mesh_info'):
                out += grid.construct_mesh_info
            out += grid.__repr__()
            return out

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
        out = ""

        # If not verbose or layered, return.
        if verb < 0 or self.layered:
            return out if return_info else None

        # Loop over sources and frequencies.
        for src, freq in self._srcfreq:
            cinfo = self._dict_get(f"{field}_info", src, freq)

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
        self._input_sc2 = kwargs.pop('_input_sc2', model.shape[2])

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
                msg = ("emg3d: `expand` is deprecated and will be removed in "
                       "v1.9.0. A property-complete model has to be provided.")
                warnings.warn(msg, FutureWarning)

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

    @property
    def layered(self):
        """If True, use layered computations (empymod)."""
        return self._layered

    @layered.setter
    def layered(self, layered):
        """Update layered and therefore layered_opts."""
        self._set_layered_opts(layered, self.layered_opts)

    def _set_layered_opts(self, layered, layered_opts):
        """Set self.layered and self.layered_opts."""

        # Set layered.
        self._layered = layered

        # If not layered, just store layered_opts and return.
        if not layered:
            self.layered_opts = layered_opts
            return

        # Ensure we can handle the sources and receivers.
        srlist = list(self.survey.sources.values())
        srlist = srlist + list(self.survey.receivers.values())
        for sr in srlist:
            name = sr.__class__.__name__
            if 'Point' not in name and 'Dipole' not in name:
                raise ValueError(
                    "Layered: Only Points and Dipoles supported, "
                    f"provided: {sr}!"
                )

        # Check limitation: Only isotropic and VTI.
        if self.model.case not in ['isotropic', 'VTI']:
            raise NotImplementedError(
                f"Layered compute not implemented for {self.model.case} case."
            )

        # Make a copy of layered to not overwrite.
        layered_opts = deepcopy(layered_opts)

        # Ensure method is defined; default: cylinder
        layered_opts['method'] = layered_opts.get('method', 'cylinder')

        # For cylinder/prism, ensure there is ellipse['radius'].
        if layered_opts['method'] in ['prism', 'cylinder']:

            # Initiate or get ellipse dict.
            ellipse = layered_opts.get('ellipse', {})

            # Try to estimate radius if not given.
            if ellipse.get('radius') is None:

                # Check if negz-cond is in gridding_opts.
                try:
                    prop = self.gridding_opts['properties']
                    prop = np.atleast_1d(prop)
                    m = getattr(maps, 'Map' + self.gridding_opts['mapping'])()
                    # Take the negative z property
                    ind = -1 if prop.size < 3 else -2
                    cond = m.backward(prop[ind])

                # If not, calculate it.
                except (KeyError, TypeError):
                    zneg = self.model.property_x[:, :, 0]
                    cond = np.min(self.model.map.backward(zneg))

                # Lowest frequency.
                freq = min(self.survey.frequencies.values())

                # Set the radius to one skin depth.
                ellipse['radius'] = meshes.skin_depth(freq, cond)

            # Set factor/minor and store back.
            ellipse['factor'] = ellipse.get('factor', 1.2)
            ellipse['minor'] = ellipse.get('minor', 0.8)
            layered_opts['ellipse'] = ellipse

        # Store layered options.
        self.layered_opts = layered_opts
        self._empymod_opts = {'verb': 1}
