"""

:mod:`simulation` -- Model a survey
===================================

A simulation is the computation (modelling) of electromagnetic responses of a
resistivity (conductivity) model for a given survey.

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


import itertools
import numpy as np
from copy import deepcopy

from emg3d import fields, solver, io, surveys, models, meshes, optimize

# Check soft dependencies.
try:
    from tqdm.contrib.concurrent import process_map
except ImportError:
    from concurrent.futures import ProcessPoolExecutor

    def process_map(fn, *iterables, max_workers, **kwargs):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, *iterables))

__all__ = ['Simulation']


class Simulation():
    r"""Create a simulation for a given survey on a given model.

    The computational grid(s) can be either the same as the provided model
    grid, or automatic gridding can be used.

    .. note::

        The Simulation-class has currently a few limitations:

        - `gridding` must be `'same'`;
        - `survey.fixed`: must be `False`;
        - sources and receivers must be electric;
        - sources strength is always normalized to 1 Am.


    Parameters
    ----------
    survey : :class:`emg3d.surveys.Survey`
        The survey layout, containing sources, receivers, frequencies, and
        optionally the measured data.

        The survey-data will be modified in place. Provide survey.copy() if you
        want to avoid this.

    grid : :class:`meshes.TensorMesh`
        The grid. See :class:`meshes.TensorMesh`.

    model : :class:`emg3d.models.Model`
        The model. See :class:`emg3d.models.Model`.

    max_workers : int
        The maximum number of processes that can be used to execute the
        given calls. Default is 4.

    gridding : str
        Method how the computational grids are computed. The default is
        currently 'same', the only implemented method so far. This will change
        in the future, probably to 'single'. The different methods are:

        - 'same': Same grid as for the input model.
        - 'single': A single grid for all sources and frequencies.
        - 'frequency': Frequency-dependent grids.
        - 'source': Source-dependent grids.
        - 'both': Frequency- and source-dependent grids.

        Except for 'same', the grids are created using ...

        Not planned (yet) is the possibility to provide a single TensorMesh
        instance or a dict of TensorMesh instances for user-provided meshes.
        You can still do this by setting `simulation._dict_grid` after
        instantiation.

    solver_opts : dict, optional
        Passed through to :func:`emg3d.solver.solve`. The dict can contain any
        parameter that is accepted by the :func:`emg3d.solver.solve` except for
        `grid`, `model`, `sfield`, and `efield`.
        If not provided the following defaults are used:

        - `sslsolver = True`;
        - `semicoarsening = True`;
        - `linerelaxation = True`;
        - `verb = 0` (yet warnings are capture and shown).

    """

    def __init__(self, name, survey, grid, model, max_workers=4,
                 gridding='same', **kwargs):
        """Initiate a new Simulation instance."""

        # Store inputs.
        self.name = name
        self.survey = survey
        self.grid = grid
        self.model = model
        self.max_workers = max_workers

        self.gridding = gridding
        self._gridding_descr = {
                'same': 'Same grid as for model',
                'single': 'A single grid for all sources and frequencies',
                'frequency': 'Frequency-dependent grids',
                'source': 'Source-dependent grids',
                'both': 'Frequency- and source-dependent grids',
                }

        # Get kwargs.
        solver_opts = kwargs.pop('solver_opts', {})

        # Ensure no kwargs left (currently kwargs is not used).
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

        # Check current limitations.
        if self.gridding != 'same':
            raise NotImplementedError(
                    "Simulation currently only implemented for "
                    "`gridding='same'`.")

        if self.survey.fixed:
            raise NotImplementedError(
                    "Simulation currently only implemented for "
                    "`survey.fixed=False`.")

        # Magnetic sources are not yet implemented in simulation.
        if sum([not s.electric for s in survey.sources.values()]) > 0:
            raise NotImplementedError(
                    "Simulation not yet implemented for magnetic sources.")

        # Magnetic receivers are not yet implemented in simulation.
        if sum([not r.electric for r in survey.receivers.values()]) > 0:
            raise NotImplementedError(
                    "Simulation not yet implemented for magnetic receivers.")

        # Set default solver options if not provided.
        self.solver_opts = {
                'sslsolver': True,
                'semicoarsening': True,
                'linerelaxation': True,
                'verb': 0,
                **solver_opts,  # Overwrites defaults.
                }

        # Initiate dictionaries with None's.
        self._dict_grid = self._dict_initiate
        self._dict_model = self._dict_initiate
        self._dict_sfield = self._dict_initiate
        self._dict_efield = self._dict_initiate
        self._dict_hfield = self._dict_initiate
        self._dict_efield_info = self._dict_initiate

        # Initiate synthetic data with NaN's.
        self.survey._data['synthetic'] = self.survey.data.observed*np.nan

    def __repr__(self):
        return (f"*{self.__class__.__name__}* «{self.name}» "
                f"of {self.survey.__class__.__name__} «{self.survey.name}»\n\n"
                f"- {self.survey.__class__.__name__}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies\n"
                f"- {self.model.__repr__()}\n"
                f"- Gridding: {self._gridding_descr[self.gridding]}")

    def _repr_html_(self):
        return (f"<h3>{self.__class__.__name__} «{self.name}»</h3>"
                f"of {self.survey.__class__.__name__} «{self.survey.name}»<ul>"
                f"<li>{self.survey.__class__.__name__}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies</li>"
                f"<li>{self.model.__repr__()}</li>"
                f"<li>Gridding: "
                f"{self._gridding_descr[self.gridding]}</li>"
                f"</ul>")

    def copy(self, what='computed'):
        """Return a copy of the Simulation.

        See `to_file` for more information regarding `what`.

        """
        return self.from_dict(self.to_dict(what, True))

    def to_dict(self, what='computed', copy=False):
        """Store the necessary information of the Simulation in a dict.

        See `to_file` for more information regarding `what`.

        """

        # TODO not adjusted for optimize                                       #

        if what not in ['computed', 'results', 'all', 'plain']:
            raise TypeError(f"Unrecognized `what`: {what}")

        # If to_dict is called from to_file, it has a _what_to_file attribute.
        if hasattr(self, '_what_to_file'):
            what = self._what_to_file
            delattr(self, '_what_to_file')

        # Initiate dict.
        out = {'name': self.name, '__class__': self.__class__.__name__}

        # Add initiation parameters.
        out['survey'] = self.survey.to_dict()
        out['grid'] = self.grid.to_dict()
        out['model'] = self.model.to_dict()
        out['max_workers'] = self.max_workers
        out['gridding'] = self.gridding
        out['solver_opts'] = self.solver_opts

        # Get required properties.
        store = []

        if what == 'all':
            store += ['_dict_grid', '_dict_model', '_dict_sfield',
                      '_dict_hfield']

        if what in ['computed', 'all']:
            store += ['_dict_efield', '_dict_efield_info']

        # store dicts.
        for name in store:
            out[name] = getattr(self, name)

        # store data.
        if what in ['computed', 'results', 'all']:
            out['synthetic'] = self.data.synthetic

        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`Simulation` instance.


        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`Simulation.to_dict`.

        Returns
        -------
        obj : :class:`Simulation` instance

        """

        # TODO not adjusted for optimize                                       #

        try:
            # Initiate class.
            out = cls(
                    name=inp['name'],
                    survey=surveys.Survey.from_dict(inp['survey']),
                    grid=meshes.TensorMesh.from_dict(inp['grid']),
                    model=models.Model.from_dict(inp['model']),
                    max_workers=inp['max_workers'],
                    gridding=inp['gridding'],
                    solver_opts=inp['solver_opts'],
                    )

            # Add existing derived/computed properties.
            data = ['_dict_grid', '_dict_model', '_dict_sfield',
                    '_dict_hfield', '_dict_efield', '_dict_efield_info']

            for name in data:
                if name in inp.keys():
                    setattr(out, name, inp.get(name))

            if 'synthetic' in inp.keys():
                synthetic = out.data.observed*0+inp['synthetic']
                out.survey._data['synthetic'] = synthetic

            return out

        except KeyError as e:
            raise KeyError(f"Variable {e} missing in `inp`.") from e

    def to_file(self, fname, what='computed', compression="gzip",
                json_indent=2, verb=1):
        """Store Simulation to a file.

        Parameters
        ----------
        fname : str
            File name inclusive ending, which defines the used data format.
            Implemented are currently:

            - `.h5` (default): Uses `h5py` to store inputs to a hierarchical,
              compressed binary hdf5 file. Recommended file format, but
              requires the module `h5py`. Default format if ending is not
              provided or not recognized.
            - `.npz`: Uses `numpy` to store inputs to a flat, compressed binary
              file. Default format if `h5py` is not installed.
            - `.json`: Uses `json` to store inputs to a hierarchical, plain
              text file.

        what : str
            What to store. Currently implemented:

            - 'computed' (default):
              Stores all computed properties: electric fields and responses at
              receiver locations.
            - 'results':
              Stores only the response at receiver locations.
            - 'all':
              Stores everything.
            - 'plain':
              Only stores the plain Simulation (as initiated).

        compression : int or str, optional
            Passed through to h5py, default is 'gzip'.

        json_indent : int or None
            Passed through to json, default is 2.

        verb : int
            Silent if 0, verbose if 1.

        """

        # Add what to self, will be removed in to_dict.
        self._what_to_file = what

        io.save(fname, compression=compression, json_indent=json_indent,
                collect_classes=False, verb=verb, simulation=self)

    @classmethod
    def from_file(cls, fname, verb=1):
        """Load Simulation from a file.

        Parameters
        ----------
        fname : str
            File name including extension. Used backend depends on the file
            extensions:

            - '.npz': numpy-binary
            - '.h5': h5py-binary (needs `h5py`)
            - '.json': json

        verb : int
            Silent if 0, verbose if 1.

        Returns
        -------
        simulation : :class:`Simulation`
            The simulation that was stored in the file.

        """
        return io.load(fname, verb=verb)['simulation']

    # GET FUNCTIONS
    def get_grid(self, source, frequency):
        """Return computational grid of the given source and frequency."""
        freq = float(frequency)

        # Get grid if it is not stored yet.
        if self._dict_grid[source][freq] is None:

            # Act depending on gridding:
            if self.gridding == 'same':  # Same grid as for provided model.

                # Store link to grid.
                self._dict_grid[source][freq] = self.grid

            # Rest is not yet implemented.

        # Return grid.
        return self._dict_grid[source][freq]

    def get_model(self, source, frequency):
        """Return model on the grid of the given source and frequency."""
        freq = float(frequency)

        # Get model if it is not stored yet.
        if self._dict_model[source][freq] is None:

            # Act depending on gridding:
            if self.gridding == 'same':  # Same grid as for provided model.

                # Store link to model.
                self._dict_model[source][freq] = self.model

            # Rest is not yet implemented.

        # Return model.
        return self._dict_model[source][freq]

    def get_sfield(self, source, frequency):
        """Return source field for given source and frequency."""
        freq = float(frequency)

        # Get source field if it is not stored yet.
        if self._dict_sfield[source][freq] is None:

            sfield = fields.get_source_field(
                    grid=self.get_grid(source, frequency),
                    src=self.survey.sources[source].coordinates,
                    freq=frequency,
                    strength=0)

            self._dict_sfield[source][freq] = sfield

        # Return source field.
        return self._dict_sfield[source][freq]

    def get_efield(self, source, frequency, **kwargs):
        """Return electric field for given source and frequency."""
        freq = float(frequency)

        # Get call_from_compute and ensure no kwargs are left.
        call_from_compute = kwargs.pop('call_from_compute', False)
        if kwargs:
            raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

        # Compute electric field if it is not stored yet.
        if self._dict_efield[source][freq] is None:

            # Input parameters.
            solver_input = {
                **self.solver_opts,
                'grid': self.get_grid(source, freq),
                'model': self.get_model(source, freq),
                'sfield': self.get_sfield(source, freq),
                'return_info': True,
            }

            # Compute electric field.
            efield, info = solver.solve(**solver_input)

            # Store electric field and info.
            self._dict_efield[source][freq] = efield
            self._dict_efield_info[source][freq] = info

            # Clean corresponding hfield, so it will be recomputed.
            del self._dict_hfield[source][freq]
            self._dict_hfield[source][freq] = None

            # Get receiver coordinates.
            rec_coords = self.survey.rec_coords
            # For fixed surveys:
            # rec_coords = self.survey.rec_coords[source]

            # Extract data at receivers.
            resp = fields.get_receiver_response(
                    grid=self._dict_grid[source][freq],
                    field=self._dict_efield[source][freq],
                    rec=rec_coords
            )

            # Store the receiver response.
            self.data.synthetic.loc[source, :, freq] = resp

        # Return electric field.
        if call_from_compute:
            return (self._dict_efield[source][freq],
                    self._dict_efield_info[source][freq],
                    self.data.synthetic.loc[source, :, freq].data)
        else:
            return self._dict_efield[source][freq]

    def get_hfield(self, source, frequency, **kwargs):
        """Return magnetic field for given source and frequency."""
        freq = float(frequency)

        # If magnetic field not computed yet compute it.
        if self._dict_hfield[source][freq] is None:

            self._dict_hfield[source][freq] = fields.get_h_field(
                    self.get_grid(source, freq),
                    self.get_model(source, freq),
                    self.get_efield(source, freq, **kwargs))

        # Return magnetic field.
        return self._dict_hfield[source][freq]

    def get_efield_info(self, source, frequency):
        """Return the solver information of the corresponding computation."""
        return self._dict_efield_info[source][float(frequency)]

    # ASYNCHRONOUS COMPUTATION
    def _get_efield(self, inp):
        """Wrapper of `get_efield` for `concurrent.futures`."""
        return self.get_efield(*inp, call_from_compute=True)

    def compute(self, observed=False):
        """Compute efields asynchronously for all sources and frequencies.

        Parameters
        ----------
        observed : bool
            By default, the data at receiver locations is stored in the
            `Survey` as `synthetic`. If `observed=True`, however, it is stored
            in `observed`.

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
            _ = self.get_sfield(src, freq)

        # Initiate futures-dict to store output.
        out = process_map(
                self._get_efield,
                srcfreq,
                max_workers=self.max_workers,
                desc='Compute efields',
                bar_format='{desc}: {bar}{n_fmt}/{total_fmt}  [{elapsed}]',
        )

        # Clean hfields, so they will be recomputed.
        del self._dict_hfield
        self._dict_hfield = self._dict_initiate

        # The store-name is not water-tight if beforehand get_efield was used
        # or similar. It works with a clean simulation.compute.
        store_name = ['synthetic', 'observed'][observed]

        # Loop over src-freq combinations to extract and store.
        warned = False  # Flag for warnings.
        for i, (src, freq) in enumerate(srcfreq):

            # Store efield.
            self._dict_efield[src][freq] = out[i][0]

            # Store solver info.
            info = out[i][1]
            self._dict_efield_info[src][freq] = info
            if info['exit'] != 0:
                if not warned:
                    print("Solver warnings:")
                    warned = True
                print(f"- Src {src}; {freq} Hz : {info['exit_message']}")

            # Store responses at receivers.
            self.data[store_name].loc[src, :, freq] = out[i][2]

    # DATA
    @property
    def data(self):
        """Shortcut to survey.data."""
        return self.survey.data

    # OPTIMIZATION

    # TODO gradient()

    @property
    def data_misfit(self):
        r"""Return the misfit between observed and synthetic data.

        The weighted least-squares functional, as implemented in `emg3d`, is
        given by Equation 1 [PlMu08]_,

        .. math::
            :label:misfit

                J(\textbf{p}) = \frac{1}{2} \sum_f\sum_s\sum_r
                  \left\{
                    \left\lVert
                      W_{s,r,f}^e\left(\textbf{e}_{s,r,f}[\sigma(\textbf{p})]
                      -\textbf{e}_{s,r,f}^\text{obs}\right)
                    \right\rVert^2
                  + \left\lVert
                      W_{s,r,f}^h\left(\textbf{h}_{s,r,f}[\sigma(\textbf{p})]
                      -\textbf{h}_{s,r,f}^\text{obs}\right)
                    \right\rVert^2
                  \right\}
                + R(\textbf{p}) \, .

        """

        # Compute it if not stored already.
        # if getattr(self, '_data_misfit', None) is None:

        # # TODO: - Data weighting;
        # #       - Min_offset;
        # #       - Noise floor.
        # DW = optimize.weights.DataWeighting(**self.data_weight_opts)

        # Store the residual.
        self.data['residual'] = (self.data.synthetic - self.data.observed)

        # Store a copy for the weighted residual.
        self.data['wresidual'] = self.data.residual.copy()


        # Compute the weights.
        for src, freq in self._srcfreq:
            data = self.data.wresidual.loc[src, :, freq].data

            # # TODO: Actual weights.
            # weig = DW.weights(
            #         data,
            #         self.survey.rec_coords,
            #         self.survey.sources[src].coords,
            #         freq)
            # self.survey._data['wresidual'].loc[sname, :, freq] *= weig

        self._data_misfit = np.sum(np.abs(self.data.residual.data.conj() *
                                          self.data.wresidual.data))/2

        return self._data_misfit

    # UTILS
    def clean(self, what='computed'):
        """Clean part of the data base.

        Parameters
        ----------
        what : str
            What to clean. Currently implemented:

            - 'computed' (default):
              Removes all computed properties: electric and magnetic fields and
              responses at receiver locations.
            - 'keepresults':
              Removes everything  except for the responses at receiver
              locations.
            - 'all':
              Removes everything (leaves it plain as initiated).

        """

        # TODO not adjusted for optimize                                       #

        if what not in ['computed', 'keepresults', 'all']:
            raise TypeError(f"Unrecognized `what`: {what}")

        clean = []

        if what in ['keepresults', 'all']:
            clean += ['grid', 'model', 'sfield']

        if what in ['computed', 'keepresults', 'all']:
            clean += ['efield', 'efield_info', 'hfield']

        # Clean dicts.
        for name in clean:
            delattr(self, '_dict_'+name)
            setattr(self, '_dict_'+name, self._dict_initiate)

        # Clean data.
        if what in ['computed', 'all']:
            self.data['synthetic'] = self.data.observed*np.nan

    @property
    def _dict_initiate(self):
        """Returns a dict of the structure `dict[source][freq]=None`."""
        return {src: {freq: None for freq in self.survey.frequencies}
                for src in self.survey.sources.keys()}
    @property
    def _srcfreq(self):
        """Return list of all source-frequency pairs."""

        if getattr(self, '__srcfreq', None) is None:
            self.__srcfreq = list(
                    itertools.product(self.survey.sources.keys(),
                                      self.survey.frequencies))

        return self.__srcfreq
