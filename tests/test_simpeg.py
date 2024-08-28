from emg3d import inversion
from emg3d.inversion import simpeg as isimpeg


def test_all_dir():
    assert set(inversion.__all__) == set(dir(inversion))
    assert set(isimpeg.__all__) == set(dir(isimpeg))
