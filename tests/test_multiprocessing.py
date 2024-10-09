from os.path import join, dirname

import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
import empymod
from emg3d import _multiprocessing as _mp

try:
    import tqdm
except ImportError:
    tqdm = None

try:
    import xarray
except ImportError:
    xarray = None

# Data generated with tests/create_data/regression.py
REGRES = emg3d.load(join(dirname(__file__), 'data', 'regression.npz'))


def dummy(inp):
    """Dummy fct to test process_map."""
    return inp


@pytest.mark.filterwarnings("ignore:.*lead to deadlocks*:DeprecationWarning")
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


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_layered():

    src = emg3d.TxElectricDipole((-2000, 0, 200, 20, 5))
    off = np.array([2000, 6000])
    rz = -250
    rec = emg3d.surveys.txrx_coordinates_to_dict(
            emg3d.RxElectricPoint, (off, 0, rz, 0, 0))
    freqs = [1.0, 2.0, 3.0]

    grid = emg3d.TensorMesh(
        h=[[4000, 4000, 4000], [2000, 2000], [100, 250, 100]],
        origin=(-4000, -2000, -350),
    )
    res = [2e14, 0.3, 1.0]
    model = emg3d.Model(grid, property_x=1.0, property_z=1.0)
    model.property_x[:, :, 1] = res[1]
    model.property_z[:, :, 1] = res[1]
    model.property_x[:, :, 2] = res[0]
    model.property_z[:, :, 2] = res[0]

    obs = empymod.bipole(
            src=src.coordinates,
            rec=(off, off*0, rz, 0, 0),
            depth=[0, -250],
            res=[res[0], res[1], 2*res[2]],
            freqtime=freqs,
            verb=1,
    )
    syn = empymod.bipole(
            src=src.coordinates,
            rec=(off, off*0, rz, 0, 0),
            depth=[0, -250],
            res=res,
            freqtime=freqs,
            verb=1,
    )

    survey = emg3d.Survey(src, rec, freqs)
    survey.data.observed[0, ...] = obs.T
    survey.data['synthetic'] = survey.data['observed'].copy()
    survey.data.synthetic[0, ...] = obs.T

    inp = {
        'model': model,
        'src': survey.sources['TxED-1'],
        'receivers': survey.receivers,
        'frequencies': survey.frequencies,
        'empymod_opts': {'verb': 1},
        'observed': None,
        'layered_opts': {'method': 'receiver'},
        'gradient': False,
    }

    # Forward - generate data
    out = _mp.layered(inp)
    assert_allclose(out, syn.T)

    # Gradient without required data.
    inp['gradient'] = True
    out = _mp.layered(inp)
    assert_allclose(out, 0.0)

    # Gradient but all-NaN obs.
    inp['observed'] = survey.data.observed[0, :, :]*np.nan
    inp['weights'] = survey.data['observed'].copy()*0+1
    inp['weights'] = inp['weights'][0, :, :]
    inp['residual'] = survey.data.synthetic - survey.data.observed
    inp['residual'] = inp['residual'][0, :, :]
    out = _mp.layered(inp)
    assert_allclose(out, 0.0)

    # Gradient
    inp['observed'] = survey.data.observed[0, :, :]
    out = _mp.layered(inp)
    # Only checking locations, not actual gradient
    # (that is done in _fd_gradient)
    assert_allclose(out[1, ...], 0.0)
    assert_allclose(out[::2, 0, :, :], 0.0)
    assert_allclose(out[::2, :, 0, :], 0.0)
    assert np.all(out[::2, 1:, 1, :] != 0.0)


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


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_fd_gradient():

    res = np.array([0.9876, ])

    empymod_inp = {
        'src': (0, 0, 0, 20, 5),
        'rec': (100, 0, 0, -5, -5),
        'depth': [],
        'freqtime': 1.0,
        'verb': 1,
    }

    obs = empymod.bipole(res=res, **empymod_inp)

    d = 1/res*0.0001
    dobs_h = empymod.bipole(res=d+res, **empymod_inp)
    dobs_v = empymod.bipole(
            res=res, aniso=np.sqrt((d+res)/res), **empymod_inp)

    weight = np.pi
    misfit = np.e
    inp = {
        'data': obs,
        'weight': weight,
        'misfit': misfit,
        'empymod_inp': empymod_inp,
        'imat': np.array([[0.5, 0.0], [0.0, 0.0], [.1, 0.4]]),
    }

    rh = dobs_h-obs
    gh = (np.real(weight*(rh.conj()*rh)/2) - misfit)/d
    rv = dobs_v-obs
    gv = (np.real(weight*(rv.conj()*rv)/2) - misfit)/d

    cond = 1/res
    ghc = _mp._fd_gradient(cond_h=cond, cond_v=None, vertical=False, **inp)
    gvc = _mp._fd_gradient(cond_h=cond, cond_v=cond, vertical=True, **inp)

    assert_allclose(gh, 2*ghc[0, 0])
    assert_allclose(gv, 10*gvc[2, 0])
    assert_allclose(gv, 2.5*gvc[2, 1])
