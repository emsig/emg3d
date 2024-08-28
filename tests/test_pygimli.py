from emg3d import inversion
from emg3d.inversion import pygimli as ipygimli


def test_all_dir():
    assert set(inversion.__all__) == set(dir(inversion))
    assert set(ipygimli.__all__) == set(dir(ipygimli))
