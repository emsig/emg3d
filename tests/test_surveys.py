import pytest
import dataclasses
# import numpy as np
from numpy.testing import assert_allclose

from emg3d import surveys


class TestSurvey():
    def test_general(self):
        sources = (0, [1000, 2000, 3000, 4000, 5000], -950, 0, 0)
        receivers = ([1000, 2000, 3000, 4000], 2000, -1000, 0, 0)
        frequencies = (1, 0.1, 2, 3)
        srvy = surveys.Survey('Test', sources, receivers, frequencies)

        assert_allclose(frequencies, srvy.frequencies)
        assert isinstance(srvy.sources, dict)
        assert srvy.sources['Tx1'].xco == 0
        assert srvy.size == 0
        assert srvy.shape == (5, 4, 4)

        assert 'Test' in srvy.__repr__()
        assert 'Test' in srvy._repr_html_()
        assert 'Coordinates' in srvy._repr_html_()

    def test_dipole_info_to_dict(self):
        # == 1. List ==
        s_list = [surveys.Dipole('Tx0', (0, 0, 0, 0, 0)),
                  surveys.Dipole('Tx1', (0, 0, 0, 0, 0))]
        r_list = [surveys.Dipole('Rx0', (0, 0, 0, 0, 0)),
                  surveys.Dipole('Rx1', (0, 0, 0, 0, 0))]
        sur_list = surveys.Survey('Test', s_list, r_list, 1)
        assert sur_list.sources['Tx0'] == s_list[0]
        assert sur_list.receivers['Rx1'] == r_list[1]
        # fixed
        fsur_list = surveys.Survey('Test', s_list, r_list, 1, fixed=1)
        assert fsur_list.sources['Tx0'] == s_list[0]
        assert fsur_list.receivers['Off0']['Tx1'] == r_list[1]

        # == 2. Tuple ==
        s_tupl = ([0, 0], 0, 0, 0, 0)
        r_tupl = (0, 0, 0, (0, 0), 0)
        sur_tupl = surveys.Survey('Test', s_tupl, r_tupl, 1)
        assert sur_tupl.sources['Tx0'] == s_list[0]
        assert sur_tupl.receivers['Rx1'] == r_list[1]
        # fixed
        fsur_tupl = surveys.Survey('Test', s_tupl, r_tupl, 1, fixed=1)
        assert fsur_tupl.sources['Tx0'] == s_list[0]
        assert fsur_tupl.receivers['Off0']['Tx1'] == r_list[1]

        # == 3. Dict ==
        s_dict = {k.name: k.to_dict() for k in s_list}
        r_dict = {k.name: k.to_dict() for k in r_list}
        sur_dict = surveys.Survey('Test', s_dict, r_dict, 1)
        assert sur_dict.sources['Tx0'] == s_list[0]
        assert sur_dict.receivers['Rx1'] == r_list[1]
        # fixed
        fr_dict = {'Off0': {'Tx0': r_dict['Rx0'], 'Tx1':  r_dict['Rx1']}}
        fsur_dict = surveys.Survey('Test', s_dict, fr_dict, 1, fixed=1)
        assert fsur_dict.sources['Tx0'] == s_list[0]
        assert fsur_dict.receivers['Off0']['Tx1'] == r_list[1]

        # == 4. Mix and match ==
        # list-tuple
        list_tupl = surveys.Survey('Test', s_list, r_tupl, 1)
        assert list_tupl.sources['Tx0'] == s_list[0]
        assert list_tupl.receivers['Rx1'] == r_list[1]
        # list-dict
        list_dict = surveys.Survey('Test', s_list, r_dict, 1)
        assert list_dict.sources['Tx0'] == s_list[0]
        assert list_dict.receivers['Rx1'] == r_list[1]
        # tuple-dict
        tupl_dict = surveys.Survey('Test', s_tupl, r_dict, 1)
        assert tupl_dict.sources['Tx0'] == s_list[0]
        assert tupl_dict.receivers['Rx1'] == r_list[1]
        # tuple-list
        tupl_list = surveys.Survey('Test', s_tupl, r_list, 1)
        assert tupl_list.sources['Tx0'] == s_list[0]
        assert tupl_list.receivers['Rx1'] == r_list[1]
        # dict-list
        dict_list = surveys.Survey('Test', s_dict, r_list, 1)
        assert dict_list.sources['Tx0'] == s_list[0]
        assert dict_list.receivers['Rx1'] == r_list[1]
        # dict-tuple
        dict_tuple = surveys.Survey('Test', s_dict, r_tupl, 1)
        assert dict_tuple.sources['Tx0'] == s_list[0]
        assert dict_tuple.receivers['Rx1'] == r_list[1]

        # == 5. Other ==
        sources = surveys.Dipole('Tx1', (0, 0, 0, 0, 0))
        # As Dipole it should fail.
        with pytest.raises(ValueError):
            surveys.Survey('T', sources, (1, 0, 0, 0, 0), 1)
        # Cast as list it should work.
        surveys.Survey('T', [sources], (1, 0, 0, 0, 0), 1)
        # Fixed with different sizes have to fail.
        with pytest.raises(ValueError):
            surveys.Survey('Test', s_list, [r_list[0]], 1, fixed=1)
        with pytest.raises(ValueError):
            surveys.Survey('Test', s_tupl,
                           (r_tupl[0], r_tupl[0], r_tupl[0]), 1, fixed=1)
        # Duplicate names should fail.
        with pytest.raises(ValueError):
            surveys.Survey('Test', s_list, [r_list[0], r_list[0]], 1)

    def test_copy(self):
        # This also checks to_dict()/from_dict().
        srvy1 = surveys.Survey('Test', (0, 0, 0, 0, 0),
                               (1000, 0, 0, 0, 0), 1.0, [[[3+3j]]])
        srvy2 = srvy1.copy()
        assert srvy1.sources == srvy2.sources

        cpy = srvy1.to_dict()
        del cpy['sources']
        with pytest.raises(KeyError):
            surveys.Survey.from_dict(cpy)

        srvy3 = surveys.Survey('Test', (0, 0, 0, 0, 0),
                               (1000, 0, 0, 0, 0), 1.0, [[[3+3j]]],
                               fixed=1)
        srvy4 = srvy3.copy()
        assert srvy3.sources == srvy4.sources


def test_PointDipole():
    # Define a few point dipoles
    dip1 = surveys.PointDipole('Tx001', 0, 100, 0, 12, 69)
    dip2 = surveys.PointDipole('Tx932', 0, 1000, 0, 12, 69)
    dip3 = surveys.PointDipole('Tx001', 0, 0, -950.0, 12, 0)
    dip4 = surveys.PointDipole('Tx004', 0, 0, 0, 12, 69)
    dip1copy = dataclasses.copy.deepcopy(dip1)

    # Some checks
    assert dip1.name == 'Tx001'
    assert dip1.xco == 0.0
    assert dip1.yco == 100.0
    assert dip1.zco == 0.0
    assert dip1.azm == 12.0
    assert dip1.dip == 69.0

    # Collect dipoles
    dipoles = [dip1, dip2, dip3, dip4, dip1copy]

    # Check it can be ordered
    dipset = set(dipoles)

    # Assert the duplicate has been removed.
    assert len(dipset) == len(dipoles)-1

    # Check it is hashable
    dipdict = {key: key.name for key in dipoles}
    assert dipdict[dip2] == dip2.name


def test_Dipole(capsys):
    dipcoord = (0.0, 1000.0, -950.0, 0.0, 0.0)
    bipcoord = (-0.5, 0.5, 1000.0, 1000.0, -950.0, -950.0)
    pointdip = surveys.Dipole('dip', dipcoord)
    finitdip = surveys.Dipole('dip', bipcoord)

    # Some checks
    assert pointdip.name == 'dip'
    assert pointdip.xco == 0.0
    assert pointdip.yco == 1000.0
    assert pointdip.zco == -950.0
    assert pointdip.azm == 0.0
    assert pointdip.dip == 0.0
    assert pointdip.is_finite is False
    assert pointdip.length == 1.0
    assert pointdip.electrode1 == (-0.5, 1000.0, -950.0)
    assert pointdip.electrode2 == (0.5, 1000.0, -950.0)
    assert_allclose(dipcoord, pointdip.coordinates)
    assert_allclose(bipcoord, finitdip.coordinates)

    assert pointdip == finitdip

    # More general case
    pointdip2 = surveys.Dipole('1', (0.0, 1000.0, -950.0, 30.0, 70.0))
    finitdip2 = surveys.Dipole(
            '1', (-0.14809906635, 0.14809906635, 999.91449496415,
                  1000.08550503585, -950.4698463104, -949.5301536896))

    assert pointdip2 == finitdip2

    # Check wrong number of points fails.
    with pytest.raises(ValueError):
        surveys.Dipole('dip', (0, 0, 0, 0))
    out, _ = capsys.readouterr()
    assert "* ERROR   :: Dipole coordinates are wrong defined." in out

    # Check that two identical poles fails.
    with pytest.raises(ValueError):
        surveys.Dipole('dip', (0, 0, 0, 0, 0, 0))

    # Check adding various attributs.
    _, _ = capsys.readouterr()
    source = surveys.Dipole(
            'dip', (0.0, 1000.0, -950.0, 0.0, 0.0), strength=75, foo='bar')
    out, _ = capsys.readouterr()

    assert source.strength == 75
    assert source.foo == 'bar'
    assert out == "* WARNING :: Unknown kwargs {foo: bar}\n"

    reprstr = "Dipole(dip, {0.0m; 1,000.0m; -950.0m}, θ=0.0°, φ=0.0°, l=1.0m)"
    assert reprstr in source.__repr__()

    # Copy, to_dict, from_dict
    source2 = source.copy()
    source3 = source.to_dict()
    source4 = source2.to_dict()
    assert source2 == source
    assert source3 == source4
    del source4['coordinates']
    with pytest.raises(KeyError):
        surveys.Dipole.from_dict(source4)
