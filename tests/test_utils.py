import re
import pytest
from timeit import default_timer
from numpy.testing import assert_allclose

import scooby
from emg3d import utils
from emg3d.inversion import pygimli as ipygimli
from emg3d.inversion import simpeg as isimpeg


def test_known_class():
    @utils._known_class
    class Dummy:
        pass

    assert utils._KNOWN_CLASSES['Dummy'] == Dummy


def test_requires(capsys):

    @utils._requires('nowidea', 'whatever')
    def dummy():
        pass

    with pytest.warns(UserWarning, match='emg3d: This feature requires'):
        a = dummy()
    out1, _ = capsys.readouterr()
    assert a is None


def test_Report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    add = []

    if ipygimli.pygimli:
        add.extend(['pygimli', 'pgcore'])
    if isimpeg.simpeg:
        add.append('simpeg')

    # Reporting is now done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    out1 = utils.Report()
    out2 = scooby.Report(
            core=['numpy', 'scipy', 'numba', 'emg3d', 'empymod'],
            optional=['empymod', 'xarray', 'discretize', 'h5py',
                      'matplotlib', 'tqdm', 'IPython'] + add,
            ncol=4)

    # Ensure they're the same; exclude time to avoid errors.
    assert out1.__repr__()[115:] == out2.__repr__()[115:]


def test_Timer():
    t0 = default_timer()  # Create almost at the same time a
    time = utils.Timer()   # t0-stamp and a Timer-instance.

    # Ensure they are the same.
    assert_allclose(t0, time.t0, atol=1e-2, rtol=1e-3)

    # Ensure `now` is a string of numbers and :.
    out = time.now
    assert re.match(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]', out)

    # This should have taken less then 1s.
    out = time.runtime
    assert "0:00:00" == str(out)

    # Check representation of Timer.
    assert 'Runtime : 0:00:0' in time.__repr__()


def test_all_dir():
    assert set(utils.__all__) == set(dir(utils))
