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

from emg3d import fields, solver, io

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

        - `adaptive` must be `'same'`;
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

    adaptive : str
        Method how the adaptive computational grids are computed. The default
        is currently 'same', the only implemented method so far. This will
        change in the future, probably to 'single'. The different methods are:

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

    def __init__(self, survey, grid, model, max_workers=4, adaptive='same',
                 **kwargs):
        """Initiate a new Simulation instance."""

        # Store inputs.
        self.survey = survey
        self.grid = grid
        self.model = model
        self.max_workers = max_workers

        self.adaptive = adaptive
        self._adaptive_descr = {
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
        if self.adaptive != 'same':
            raise NotImplementedError(
                    "Simulation currently only implemented for "
                    "`adaptive='same'`.")

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
        self._dict_grid = self._dict_initiate()
        self._dict_model = self._dict_initiate()
        self._dict_sfield = self._dict_initiate()
        self._dict_efield = self._dict_initiate()
        self._dict_hfield = self._dict_initiate()
        self._dict_efield_info = self._dict_initiate()

        # Initiate synthetic data with NaN's.
        self.survey._data['synthetic'] = self.survey.data.observed*np.nan

    def __repr__(self):
        return (f"*{self.__class__.__name__}* of «{self.survey.name}»\n\n"
                f"- {self.survey.__class__.__name__}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies\n"
                f"- {self.model.__repr__()}\n"
                f"- Gridding: {self._adaptive_descr[self.adaptive]}")

    def _repr_html_(self):
        return (f"<h3>{self.__class__.__name__}</h3>"
                f"of «{self.survey.name}»<ul>"
                f"<li>{self.survey.__class__.__name__}: "
                f"{self.survey.shape[0]} sources; "
                f"{self.survey.shape[1]} receivers; "
                f"{self.survey.shape[2]} frequencies</li>"
                f"<li>{self.model.__repr__()}</li>"
                f"<li>Gridding: "
                f"{self._adaptive_descr[self.adaptive]}</li>"
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

    def to_file(self, fname, what='results', compression="gzip",
                json_indent=2):
        """Store Simulation to a file.

        Work in progress...

        # Options
        - 'results': A, B      # Input and results
        - 'resultsonly': B     # Just the results (can not be recovered)
        - 'computed': A, B, C  # Input, fields, and results
        - 'all': A, B, C, D    # plus derived meshes/models etc.

        # (A) Input
        - survey
        - grid  # (restore orig using self._input_nCz)
        - model # (restore orig using self._input_nCz)
        - max_workers
        - solver_opts
        - adaptive
        - kwargs

        # (B) Results (if you don't want to store them => self.clean())
        - survey.data.synthetic
        => all variables: list(survey.data.keys())

        # (C) Computed quantities
        - _dict_efield, _dict_efield_info

        # (D) Derived quantities
        - _dict_grid
        - _dict_model
        - _dict_sfield
        - _dict_hfield

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

        compression : int or str, optional
            Passed through to h5py, default is 'gzip'.

        json_indent : int or None
            Passed through to json, default is 2.
        """
        if what == 'resultsonly':
            # Can not be loaded with `from_file`, but with `emg3d.load`.
            io.save(fname, compression=compression, json_indent=json_indent,
                    collect_classes=False,
                    synthetic=self.survey.data.synthetic)
        else:
            raise NotImplementedError
            # io.save(fname, compression=compression, json_indent=json_indent,
            #         collect_classes=False, simulation=self)

    @classmethod
    def from_file(cls, fname):
        """Load Simulation from a file."""
        raise NotImplementedError
        # return io.load(fname)['simulation']

    # GET FUNCTIONS
    def get_grid(self, source, frequency):
        """Return computational grid of the given source and frequency."""
        freq = float(frequency)

        # Get grid if it is not stored yet.
        if self._dict_grid[source][freq] is None:

            # Act depending on adaptive:
            if self.adaptive == 'same':  # Same grid as for provided model.

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

            # Act depending on adaptive:
            if self.adaptive == 'same':  # Same grid as for provided model.

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
    def _call_efields(self, inp):
        return self.get_efield(*inp, call_from_compute=True)

    def compute(self, observed=False, **kwargs):
        """Compute efields asynchronously for all sources and frequencies.

        Parameters
        ----------
        observed : bool
            By default, the data at receiver is stored in the `Survey` as
            `synthetic`. If `observed=True`, however, it is stored in
            `observed`.

        kwargs : dict
            Passed to :func:`emg3d.solver.solve`; can contain any of the
            arguments of the solver except `grid`, `model`, `sfield`, and
            `efield`.

        """

        # Ensure grid, model, and sfield are computed.
        # TODO : This could be done within the field computation. But then
        #        it might have to be done multiple times even if 'single' or
        #        'same' grid.
        for src in self.survey.sources.keys():  # Loop over source positions.
            for freq in self.survey.frequencies:  # Loop over frequencies.
                self.get_grid(src, freq),
                self.get_model(src, freq),
                self.get_sfield(src, freq),

        # Get all source-frequency pairs.
        srcfreq = list(itertools.product(self.survey.sources.keys(),
                                         self.survey.frequencies))

        # Initiate futures-dict to store output.
        disable = self.max_workers >= len(srcfreq)
        out = process_map(self._call_efields, srcfreq,
                          max_workers=self.max_workers,
                          desc='Compute efields',
                          bar_format='{desc}: {bar}{n_fmt}/{total_fmt}',
                          disable=disable)

        # Clean hfields, so they will be recomputed.
        del self._dict_hfield
        self._dict_hfield = self._dict_initiate()

        # Extract and store.
        i = 0
        warned = False
        for src in self.survey.sources.keys():
            for freq in self.survey.frequencies:
                # Store efield.
                self._dict_efield[src][freq] = out[i][0]

                # Store responses at receivers.
                store_name = ['synthetic', 'observed'][observed]
                self.data[store_name].loc[src, :, freq] = out[i][2]

                # Store solver info.
                info = out[i][1]
                self._dict_efield_info[src][freq] = info
                if info['exit'] != 0:
                    if not warned:
                        print("Solver warnings:")
                        warned = True
                    print(f"- Src {src}; {freq} Hz : {info['exit_message']}")

                i += 1

    # UTILS
    def clean(self):
        """Remove computed fields and corresponding data.

        Delete computed by now:
            - survey._data['synthetic']
            - _misfit
            - gradient # (NOT STORED YET!)
            - _dict_efield, _dict_efield_info
            - _bfields, _bfields_info
            - _dict_hfield
            - _rfields # (Never actually stored)

        Do not delete:
            - _dict_grid
            - _dict_model
            - _dict_sfield

        Later, for inversion, we'll need a flag to also delete _dict_model,
        and maybe move/store 'synthetic' & misfit somewhere else.
        """

        # Clean efield, hfield, and solver info.
        for name in ['efield', 'hfield', 'efield_info']:
            delattr(self, '_dict_'+name)
            setattr(self, '_dict_'+name, self._dict_initiate())

        # Set synthetic data to nan's.
        self.data['synthetic'] = self.data.observed*np.nan

    def _dict_initiate(self):
        """Returns a dict of the structure `dict[source][freq]=None`."""
        return {src: {freq: None for freq in self.survey.frequencies}
                for src in self.survey.sources.keys()}

    # DATA
    @property
    def data(self):
        """Shortcut to survey.data."""
        return self.survey.data
