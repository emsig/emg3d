import pytest
import dataclasses
import numpy as np
from numpy.testing import assert_allclose

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None

from emg3d import surveys


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestSurvey():
    def test_general(self):
        sources = (0, [1000, 2000, 3000, 4000, 5000], -950, 0, 0)
        receivers = ([1000, 2000, 3000, 4000], 2000, -1000, 0, 0)
        frequencies = (1, 0.1, 2, 3)
        srvy = surveys.Survey(sources, receivers, frequencies, name='Test')

        assert_allclose(frequencies, srvy.freq_array)
        assert isinstance(srvy.sources, dict)
        assert srvy.sources['Tx1'].xco == 0
        assert srvy.size == 0
        assert srvy.shape == (5, 4, 4)

        assert 'Test' in srvy.__repr__()
        assert 'Test' in srvy._repr_html_()
        assert 'Coordinates' in srvy._repr_html_()

        with pytest.raises(TypeError, match="Unexpected "):
            surveys.Survey(sources, receivers, frequencies, bla='a')

    def test_standard_deviation(self):
        sources = (0, [1000, 2000, 3000, 4000, 5000], -950, 0, 0)
        receivers = ([1000, 3000, 4000], 2000, -1000, 0, 0)
        frequencies = (1, 0.1, 2, 3)
        srvy0 = surveys.Survey(sources, receivers, frequencies,
                               data=np.ones((5, 3, 4)))
        assert srvy0.standard_deviation is None

        # Test f-dependent noise-floor and source-dependent rel. error.
        nf = np.arange(1, 5)[None, None, :]*1e-15  # No noise floor for f=1.0
        re = np.arange(1, 6)[:, None, None]/100    # No rel. error for Tx1
        srvy = surveys.Survey(sources, receivers, frequencies,
                              relative_error=re, noise_floor=nf,
                              data=np.ones((5, 3, 4)))

        assert srvy.noise_floor == 'data._noise_floor'
        assert srvy.relative_error == 'data._relative_error'
        assert_allclose(srvy.data._noise_floor[0, 0, :], np.squeeze(nf))
        assert_allclose(srvy.data._relative_error[:, 0, 0], np.squeeze(re))
        # As data are ones, we can check standard_deviation without it.
        std = np.sqrt(np.ones(srvy.shape)*nf**2 + np.ones(srvy.shape)*re**2)
        assert_allclose(srvy.standard_deviation.data, std)

        # Set the standard deviations
        test_std = np.arange(1, srvy.size+1).reshape(srvy.shape)
        srvy.standard_deviation = test_std
        assert_allclose(srvy.data._noise_floor[0, 0, :], np.squeeze(nf))
        assert_allclose(srvy.data._relative_error[:, 0, 0], np.squeeze(re))
        assert_allclose(srvy.standard_deviation.data, test_std)
        srvy.standard_deviation = None  # Delete again
        assert_allclose(srvy.standard_deviation.data, std)

        with pytest.raises(ValueError, match='All values of `standard_dev'):
            srvy.standard_deviation = np.zeros(srvy.shape)
        with pytest.raises(ValueError, match='All values of `noise_floor`'):
            srvy.noise_floor = 0.0
        with pytest.raises(ValueError, match='All values of `relative_error'):
            srvy.relative_error = 0.0
        with pytest.raises(ValueError, match='operands could not be '):
            srvy.noise_floor = np.ones(srvy.shape)[:, :2, :]
        with pytest.raises(ValueError, match='operands could not be '):
            srvy.relative_error = np.ones(srvy.shape)[2:6, :, :]

    def test_dipole_info_to_dict(self):
        src1 = surveys.Dipole((0, 0, 0, 0, 0))
        rec1 = surveys.Dipole((0, 0, 0, 0, 0))

        # == 1. Tuple ==
        s_tupl = ([0, 0], 0, 0, 0, 0)
        r_tupl = (0, 0, 0, (0, 0), 0)
        sur_tupl = surveys.Survey(s_tupl, r_tupl, 1)
        assert sur_tupl.sources['Tx0'] == src1
        assert sur_tupl.receivers['Rx1'] == rec1

        # == 2. Dict ==
        s_dict = {f"Tx{i}": k.to_dict() for i, k in enumerate([src1, src1])}
        r_dict = {f"Rx{i}": k.to_dict() for i, k in enumerate([rec1, rec1])}
        sur_dict = surveys.Survey(s_dict, r_dict, 1)
        assert sur_dict.sources['Tx0'] == src1
        assert sur_dict.receivers['Rx1'] == rec1

        # == 3. Mix and match ==
        # tuple-dict
        tupl_dict = surveys.Survey(s_tupl, r_dict, 1)
        assert tupl_dict.sources['Tx0'] == src1
        assert tupl_dict.receivers['Rx1'] == rec1
        # dict-tuple
        dict_tupl = surveys.Survey(s_dict, r_tupl, 1)
        assert dict_tupl.sources['Tx0'] == src1
        assert dict_tupl.receivers['Rx1'] == rec1

        # == 4. Other ==
        sources = surveys.Dipole((0, 0, 0, 0, 0))
        # As Dipole it should fail.
        with pytest.raises(TypeError, match='Input format of <sources>'):
            surveys.Survey('T', sources, (1, 0, 0, 0, 0), 1)

    def test_dipole_info_to_dict_elmag(self):
        # == 1. Tuple ==
        s_tupl = ([0, 0], 0, 0, 0, 0, True)
        r_tupl = (0, 0, 0, (0, 0), 0, (True, False))
        sur_tupl = surveys.Survey(s_tupl, r_tupl, 1)
        assert sur_tupl.sources['Tx0'].electric is True
        assert sur_tupl.sources['Tx1'].electric is True
        assert sur_tupl.receivers['Rx0'].electric is True
        assert sur_tupl.receivers['Rx1'].electric is False
        assert sur_tupl.rec_types == (True, False)

    def test_copy(self, tmpdir):
        # This also checks to_dict()/from_dict().
        srvy1 = surveys.Survey((0, 0, 0, 0, 0), (1000, 0, 0, 0, 0), 1.0)
        # Set observed and standard deviation.
        srvy1.observed = [[[3+3j]]]
        srvy1.standard_deviation = np.array([[[1.1]]])
        srvy2 = srvy1.copy()
        assert srvy1.sources == srvy2.sources

        cpy = srvy1.to_dict()
        del cpy['sources']
        with pytest.raises(KeyError, match="Variable 'sources' missing"):
            surveys.Survey.from_dict(cpy)

        srvy3 = surveys.Survey((0, 0, 0, 0, 0), (1000, 0, 0, 0, 0),
                               1.0, [[[3+3j]]])
        srvy4 = srvy3.copy()
        srvy4.standard_deviation = np.array([[[1.1]]])
        assert srvy3.sources == srvy4.sources

        # Also check to_file()/from_file().
        srvy4.to_file(tmpdir+'/test.npz')
        srvy5 = surveys.Survey.from_file(tmpdir+'/test.npz')
        assert srvy4.name == srvy5.name
        assert srvy4.sources == srvy5.sources
        assert srvy4.receivers == srvy5.receivers
        assert srvy4.frequencies == srvy5.frequencies
        assert_allclose(srvy4.data.observed, srvy5.data.observed)
        assert_allclose(srvy4.standard_deviation, srvy5.standard_deviation)

        srvy7 = surveys.Survey.from_file(tmpdir+'/test.npz', verb=-1)
        assert srvy5.name == srvy7[0].name
        assert 'Data loaded from' in srvy7[1]

    def test_select(self):
        survey = surveys.Survey(
            ((0, 50, 100), 0, 0, 0, 0),
            ((1000, 1100, 1200, 1300, 1400), 0, 0, 0, 0),
            frequencies=(1.0, 2.0, 3.4, 4.0),
            data=np.arange(3*5*4).reshape((3, 5, 4)),
            noise_floor=np.ones((3, 5, 4)),
            relative_error=np.ones((3, 5, 4)),
            name='Test',
        )

        t1 = survey.select('Tx0', ['Rx0', 'Rx4'], 'f1')
        assert t1.shape == (1, 2, 1)

        t2 = survey.select(frequencies=[], receivers='Rx0')
        assert t2.shape == (3, 1, 0)


def test_PointDipole():
    # Define a few point dipoles
    dip1 = surveys.PointDipole(0, 100, 0, 12, 69, True)
    dip2 = surveys.PointDipole(0, 1000, 0, 12, 69, True)
    dip3 = surveys.PointDipole(0, 0, -950.0, 12, 0, True)
    dip4 = surveys.PointDipole(0, 0, 0, 12, 69, False)
    dip1copy = dataclasses.copy.deepcopy(dip1)

    # Some checks
    assert dip1.xco == 0.0
    assert dip1.yco == 100.0
    assert dip1.zco == 0.0
    assert dip1.azm == 12.0
    assert dip1.dip == 69.0
    assert dip1.electric is True
    assert dip4.electric is False

    # Collect dipoles
    dipoles = [dip1, dip2, dip3, dip4, dip1copy]

    # Check it can be ordered
    dipset = set(dipoles)

    # Assert the duplicate has been removed.
    assert len(dipset) == len(dipoles)-1

    # Check it is hashable
    dipdict = {key: key for key in dipoles}
    assert dipdict[dip2] == dip2


def test_Dipole(capsys):
    dipcoord = (0.0, 1000.0, -950.0, 0.0, 0.0)
    bipcoord = (-0.5, 0.5, 1000.0, 1000.0, -950.0, -950.0)
    pointdip = surveys.Dipole(dipcoord)
    finitdip = surveys.Dipole(bipcoord)

    # Some checks
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
    pointdip2 = surveys.Dipole((0.0, 1000.0, -950.0, 30.0, 70.0))
    finitdip2 = surveys.Dipole(
            (-0.14809906635, 0.14809906635, 999.91449496415,
             1000.08550503585, -950.4698463104, -949.5301536896))

    assert_allclose(pointdip2.electrode1, finitdip2.electrode1)
    assert_allclose(pointdip2.electrode2, finitdip2.electrode2)

    # Check wrong number of points fails.
    with pytest.raises(ValueError, match="Dipole coordinates are wrong"):
        surveys.Dipole((0, 0, 0, 0))

    # Check that two identical poles fails.
    with pytest.raises(ValueError, match='The two poles are identical'):
        surveys.Dipole((0, 0, 0, 0, 0, 0))

    # Check adding various attributes.
    _, _ = capsys.readouterr()
    source = surveys.Dipole((0.0, 1000.0, -950.0, 0.0, 0.0), strength=7)
    out, _ = capsys.readouterr()
    with pytest.raises(TypeError, match='Unexpected '):
        source = surveys.Dipole((0, 1, 5, 0, 0), foo='bar')

    reprstr = "pole(E, {0.0m; 1,000.0m; -950.0m}, θ=0.0°, φ=0.0°, l=1.0m)"
    assert reprstr in source.__repr__()

    # Copy, to_dict, from_dict
    source2 = source.copy()
    source3 = source.to_dict()
    source4 = source2.to_dict()
    assert source2 == source
    assert source3 == source4
    del source4['coordinates']
    with pytest.raises(KeyError, match="Variable 'coordinates' missing"):
        surveys.Dipole.from_dict(source4)

    # Magnetic dipole
    mdip = surveys.Dipole(dipcoord, False)
    mdipcopy = mdip.copy()
    assert mdipcopy.electric is False
