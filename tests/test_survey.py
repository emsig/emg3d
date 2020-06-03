import pytest
import dataclasses
# import numpy as np
from numpy.testing import assert_allclose

from emg3d import survey


class TestSurvey():
    def test_general(self):
        sources = (0, [1000, 2000, 3000, 4000, 5000], -950, 0, 0)
        receivers = ([1000, 2000, 3000, 4000], 2000, -1000, 0, 0)
        frequencies = (1, 0.1, 2, 3)
        srvy = survey.Survey('Test', sources, receivers, frequencies)

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
        sinp1 = [survey.Dipole('Tx0', (0, 0, 0, 0, 0)),
                 survey.Dipole('Tx1', (0, 0, 0, 0, 0))]
        rinp1 = [survey.Dipole('Rx0', (0, 0, 0, 0, 0)),
                 survey.Dipole('Rx1', (0, 0, 0, 0, 0))]
        srvy1 = survey.Survey('Test', sinp1, rinp1, 1)
        assert srvy1.sources['Tx0'] == sinp1[0]
        assert srvy1.receivers['Rx1'] == rinp1[1]

        # == 2. Tuple ==
        sinp2 = ([0, 0], 0, 0, 0, 0)
        rinp2 = (0, 0, 0, (0, 0), 0)
        srvy2 = survey.Survey('Test', sinp2, rinp2, 1)
        assert srvy2.sources['Tx0'] == sinp1[0]
        assert srvy2.receivers['Rx1'] == rinp1[1]

        # == 3. Dict ==
        sinp3 = {k.name: k.to_dict() for k in sinp1}
        rinp3 = {k.name: k.to_dict() for k in rinp1}
        srvy3 = survey.Survey('Test', sinp3, rinp3, 1)
        assert srvy3.sources['Tx0'] == sinp1[0]
        assert srvy3.receivers['Rx1'] == rinp1[1]

        # == 4. Other ==
        sources = survey.Dipole('Tx1', (0, 0, 0, 0, 0))
        # As Dipole it should fail.
        with pytest.raises(ValueError):
            survey.Survey('T', sources, (1, 0, 0, 0, 0), 1)
        # Cast as list it should work.
        survey.Survey('T', [sources], (1, 0, 0, 0, 0), 1)

    def test_copy(self):
        # This also checks to_dict()/from_dict().
        srvy1 = survey.Survey('Test', (0, 0, 0, 0, 0),
                              (1000, 0, 0, 0, 0), 1.0, [[[3+3j]]])
        srvy2 = srvy1.copy()
        assert srvy1.sources == srvy2.sources

        cpy = srvy1.to_dict()
        del cpy['sources']
        with pytest.raises(KeyError):
            survey.Survey.from_dict(cpy)

        srvy3 = survey.Survey('Test', (0, 0, 0, 0, 0),
                              (1000, 0, 0, 0, 0), 1.0, [[[3+3j]]],
                              fixed=1)
        srvy4 = srvy3.copy()
        assert srvy3.sources == srvy4.sources


def test_PointDipole():
    # Define a few point dipoles
    dip1 = survey.PointDipole('Tx001', 0, 100, 0, 12, 69)
    dip2 = survey.PointDipole('Tx932', 0, 1000, 0, 12, 69)
    dip3 = survey.PointDipole('Tx001', 0, 0, -950.0, 12, 0)
    dip4 = survey.PointDipole('Tx004', 0, 0, 0, 12, 69)
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
    pointdip = survey.Dipole('dip', dipcoord)
    finitdip = survey.Dipole('dip', bipcoord)

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
    pointdip2 = survey.Dipole('1', (0.0, 1000.0, -950.0, 30.0, 70.0))
    finitdip2 = survey.Dipole(
            '1', (-0.14809906635, 0.14809906635, 999.91449496415,
                  1000.08550503585, -950.4698463104, -949.5301536896))

    assert pointdip2 == finitdip2

    # Check wrong number of points fails.
    with pytest.raises(ValueError):
        survey.Dipole('dip', (0, 0, 0, 0))
    out, _ = capsys.readouterr()
    assert "* ERROR   :: Dipole coordinates are wrong defined." in out

    # Check that two identical poles fails.
    with pytest.raises(ValueError):
        survey.Dipole('dip', (0, 0, 0, 0, 0, 0))

    # Check adding various attributs.
    _, _ = capsys.readouterr()
    source = survey.Dipole(
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
        survey.Dipole.from_dict(source4)
