from os.path import join, dirname

import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import _multiprocessing as _mp

try:
    import tqdm
except ImportError:
    tqdm = None

try:
    import empymod
except ImportError:
    empymod = None

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


@pytest.mark.skipif(empymod is None, reason="empymod not installed.")
def test_empymod_fwd():
    # Simple check for status quo.
    empymod_inp = {
        'src': [0, 0, -150, 40, -20],
        'rec': [4000, 0, -200, 13, 58],
        'depth': [0, -200, -1000],
        'freqtime': [0.01, np.pi, 10],
    }
    res = np.array([2e14, 0.3, 1, 2])
    aniso = np.array([1., 1., np.sqrt(2), 2])
    resp1 = empymod.bipole(res=res, aniso=aniso, **empymod_inp)
    resp2 = _mp._empymod_fwd(1/res, 1/(aniso**2 * res), empymod_inp)

    assert_allclose(resp1, resp2)


def test_get_points():

    class DummySrcRec:
        pass

    src = DummySrcRec()
    src.center = [1, 2]
    rec = DummySrcRec()
    rec.center = [10, 20]

    # Method which is not source/receiver:
    out = _mp._get_points('cylinder', src, rec)
    assert out['method'] == 'cylinder'
    assert out['p0'] == src.center
    assert out['p1'] == rec.center

    # Method source:
    out = _mp._get_points('source', src, rec)
    assert out['method'] == 'midpoint'
    assert out['p0'] == src.center
    assert out['p1'] == src.center

    # Method receiver:
    out = _mp._get_points('receiver', src, rec)
    assert out['method'] == 'midpoint'
    assert out['p0'] == rec.center
    assert out['p1'] == rec.center


# def test_fd_gradient():
#
#     _fd_gradient(gradient, cond_h, cond_v, data, weight, misfit, empymod_inp,
#                  imat, vertical)
