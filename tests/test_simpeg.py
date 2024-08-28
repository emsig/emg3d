# import numpy as np
# from numpy.testing import assert_allclose

from emg3d import inversion
from emg3d.inversion import simpeg as isimpeg


def test_all_dir():
    assert set(inversion.__all__) == set(dir(inversion))
    assert set(isimpeg.__all__) == set(dir(isimpeg))


# Old stuff from old SimPEG-PR's
#
# import pytest
#
# import SimPEG.electromagnetics.frequency_domain as fdem
#
# # Soft dependencies
# try:
#     import emg3d
# except ImportError:
#     emg3d = None
#
#
# @pytest.mark.skipif(emg3d is None, reason="emg3d not installed.")
# class TestSurvey2SimPEG():
