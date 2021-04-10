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


# EMArray
def test_emarray():
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


# FUNCTIONS RELATED TO TIMING
def test_Time():
    t0 = default_timer()  # Create almost at the same time a
    time = utils.Time()   # t0-stamp and a Time-instance.

    # Ensure they are the same.
    assert_allclose(t0, time.t0, atol=1e-3)

    # Ensure `now` is a string of numbers and :.
    out = time.now
    assert re.match(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]', out)

    # This should have taken less then 1s.
    out = time.runtime
    assert "0:00:00" == str(out)

    # Check representation of Time.
    assert 'Runtime : 0:00:0' in time.__repr__()


# OTHER
@pytest.mark.skipif(scooby is None, reason="scooby not installed.")
def test_report(capsys):
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
