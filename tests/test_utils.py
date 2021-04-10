import re
import pytest
import numpy as np
from timeit import default_timer
from numpy.testing import assert_allclose

from emg3d import utils

try:
    import scooby
except ImportError:
    scooby = None

try:
    import tqdm
except ImportError:
    tqdm = None


def test_known_class():
    @utils._known_class
    class Dummy:
        pass

    assert utils._KNOWN_CLASSES['Dummy'] == Dummy


def test_requires(capsys):

    @utils._requires('nowidea', 'whatever')
    def dummy():
        pass

    with pytest.warns(UserWarning, match='This feature of emg3d requires'):
        a = dummy()
    out1, _ = capsys.readouterr()
    assert a is None
    assert "* WARNING :: This feature of emg3d requires" in out1


def dummy(inp):
    """Dummy fct to test process_map."""
    return inp


def test_process_map():

    # Parallel
    out = utils._process_map(dummy, [1, 2], max_workers=4, disable=True)
    assert out == [1, 2]

    # Sequential
    out = utils._process_map(dummy, [1, 2], max_workers=1, disable=True)
    assert out == [1, 2]

    # If tqdm is installed, run now without.
    if tqdm is not None:
        utils.tqdm = None

        # Parallel
        out = utils._process_map(dummy, [1, 2], max_workers=4, disable=True)
        assert out == [1, 2]

        # Sequential
        out = utils._process_map(dummy, [1, 2], max_workers=1, disable=True)
        assert out == [1, 2]

    utils.tqdm = tqdm


@pytest.mark.skipif(scooby is None, reason="scooby not installed.")
def test_Report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    # Reporting is now done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    if scooby:
        out1 = utils.Report()
        out2 = scooby.Report(
                core=['numpy', 'scipy', 'numba', 'emg3d'],
                optional=['empymod', 'xarray', 'discretize', 'h5py',
                          'matplotlib', 'tqdm', 'IPython'],
                ncol=4)

        # Ensure they're the same; exclude time to avoid errors.
        assert out1.__repr__()[115:] == out2.__repr__()[115:]

    else:  # soft dependency
        _ = utils.Report()
        out, _ = capsys.readouterr()  # Empty capsys
        assert 'WARNING :: `emg3d.Report` requires `scooby`' in out


def test_EMArray():
    out = utils.EMArray(3)
    assert out.amp() == 3
    assert out.pha() == 0
    assert out.real == 3
    assert out.imag == 0

    out = utils.EMArray(1+1j)
    assert out.amp() == np.sqrt(2)
    assert_allclose(out.pha(), np.pi/4)
    assert out.real == 1
    assert out.imag == 1

    out = utils.EMArray([1+1j, 0+1j, -1-1j])
    assert_allclose(out.amp(), [np.sqrt(2), 1, np.sqrt(2)])
    assert_allclose(out.pha(unwrap=False), [np.pi/4, np.pi/2, -3*np.pi/4])
    assert_allclose(out.pha(deg=True, unwrap=False), [45., 90., -135.])
    assert_allclose(out.pha(deg=True, unwrap=False, lag=False),
                    [-45., -90., 135.])
    assert_allclose(out.pha(deg=True, lag=False), [-45., -90., -225.])
    assert_allclose(out.real, [1, 0, -1])
    assert_allclose(out.imag, [1, 1, -1])


def test_Timer():
    t0 = default_timer()  # Create almost at the same time a
    time = utils.Timer()   # t0-stamp and a Timer-instance.

    # Ensure they are the same.
    assert_allclose(t0, time.t0, atol=1e-3)

    # Ensure `now` is a string of numbers and :.
    out = time.now
    assert re.match(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]', out)

    # This should have taken less then 1s.
    out = time.runtime
    assert "0:00:00" == str(out)

    # Check representation of Timer.
    assert 'Runtime : 0:00:0' in time.__repr__()
