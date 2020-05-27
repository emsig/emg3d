import pytest
import dataclasses
from numpy.testing import assert_allclose

from emg3d import survey


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


def test_DipoleSource(capsys):
    dipole = survey.Dipole('dip', (0.0, 1000.0, -950.0, 0.0, 0.0))
    source = survey.DipoleSource(
            'dip', (0.0, 1000.0, -950.0, 0.0, 0.0), strength=75)

    assert source.strength == 75

    assert dipole.__repr__()[-55:-1] in source.__repr__()

    _ = survey.DipoleSource('dip', (0, 0, 0, 0, 0), bla='foo')
    out, _ = capsys.readouterr()
    assert "* WARNING :: Remaining kwargs: {'bla': 'foo'}" in out


def test_DipoleReceiver(capsys):
    # Nothing else than a Dipole
    dipole = survey.Dipole('dip', (0.0, 1000.0, -950.0, 0.0, 0.0))
    receiver = survey.DipoleReceiver('dip', (0.0, 1000.0, -950.0, 0.0, 0.0))

    assert dipole.__repr__()[-55:] == receiver.__repr__()[-55:]

    _ = survey.DipoleReceiver('dip', (0, 0, 0, 0, 0), bla='foo')
    out, _ = capsys.readouterr()
    assert "* WARNING :: Remaining kwargs: {'bla': 'foo'}" in out
