from os.path import join, dirname

from numpy.testing import assert_allclose

import emg3d
from emg3d import _multiprocessing as _mp

try:
    import tqdm
except ImportError:
    tqdm = None

# Data generated with tests/create_data/regression.py
REGRES = emg3d.load(join(dirname(__file__), 'data', 'regression.npz'))


def dummy(inp):
    """Dummy fct to test process_map."""
    return inp


def test_process_map():

    # Parallel
    out = _mp.process_map(dummy, [1, 2], max_workers=4, disable=True)
    assert out == [1, 2]

    # Sequential
    out = _mp.process_map(dummy, [1, 2], max_workers=1, disable=True)
    assert out == [1, 2]

    # If tqdm is installed, run now without.
    if tqdm is not None:
        _mp.tqdm = None

        # Parallel
        out = _mp.process_map(dummy, [1, 2], max_workers=4, disable=True)
        assert out == [1, 2]

        # Sequential
        out = _mp.process_map(dummy, [1, 2], max_workers=1, disable=True)
        assert out == [1, 2]

    _mp.tqdm = tqdm


def test__solve():
    # Has keys [model, sfield, efield, solver_opts]
    dat = REGRES['res']
    inp = {'model': emg3d.Model(**dat['input_model']),
           'sfield': emg3d.get_source_field(**dat['input_source']),
           'efield': None,
           'solver_opts': {'plain': True}}
    efield, info = _mp.solve(inp)
    assert_allclose(dat['Fresult'].field, efield.field)

    # Has keys [model, grid, source, frequency, efield, solver_opts]
    dat = REGRES['res']
    model = model = emg3d.Model(**dat['input_model'])
    inp = {'model': model,
           'grid': model.grid,
           'source': dat['input_source']['source'],
           'frequency': dat['input_source']['frequency'],
           'efield': None,
           'solver_opts': {'plain': True}}
    efield, info = _mp.solve(inp)
    assert_allclose(dat['Fresult'].field, efield.field)
