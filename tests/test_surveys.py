import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import surveys


# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestSurvey():
    sources = surveys.txrx_coordinates_to_dict(
            emg3d.TxElectricDipole,
            (0, [1000, 2000, 3000, 4000, 5000], -950, 0, 0))
    receivers = surveys.txrx_coordinates_to_dict(
            emg3d.RxElectricPoint,
            ([1000, 2000, 3000, 4000], 2000, -1000, 0, 0))
    frequencies = (1, 0.1, 2, 3)
    shape = (len(sources), len(receivers), len(frequencies))

    def test_defaults(self):
        srvy = surveys.Survey(self.sources, self.receivers, self.frequencies)

        assert isinstance(srvy.sources, dict)
        assert srvy.sources['TxED-1'].center[0] == 0
        assert srvy.count == 0
        assert srvy.size == 80
        assert srvy.shape == self.shape

        assert 'Coordinates' in srvy._repr_html_()
        assert 'Coordinates' in srvy.__repr__()

        # Check defaults
        assert_allclose(srvy.data.observed.data,
                        np.ones(self.shape)*(np.nan+1j*np.nan))
        assert srvy.noise_floor is None
        assert srvy.relative_error is None
        assert srvy.name is None
        assert srvy.date is None
        assert srvy.info is None

        with pytest.raises(TypeError, match="Unexpected "):
            surveys.Survey(self.sources, self.receivers, self.frequencies,
                           bla='a')

    def test_basics(self):
        data = np.arange(np.prod(self.shape)).reshape(self.shape)
        srvy = surveys.Survey(
                self.sources, self.receivers, self.frequencies,
                data={'test': data},
                relative_error=0.05, noise_floor=1e-15,
                name='MySurvey', info='RainyDay', date='today',
                )

        assert isinstance(srvy.sources, dict)
        assert srvy.sources['TxED-1'].center[0] == 0
        assert srvy.count == 0
        assert srvy.size == 80
        assert srvy.shape == self.shape

        assert 'MySurvey' in srvy._repr_html_()
        assert 'MySurvey' in srvy.__repr__()
        assert 'RainyDay' in srvy._repr_html_()
        assert 'RainyDay' in srvy.__repr__()
        assert 'today' in srvy._repr_html_()
        assert 'today' in srvy.__repr__()

        assert_allclose(srvy.data.test.data, data)
        assert_allclose(srvy.data.observed.data,
                        np.ones(self.shape)*(np.nan+1j*np.nan))
        assert srvy.noise_floor == 1e-15
        assert srvy.relative_error == 0.05
        assert srvy.name == 'MySurvey'
        assert srvy.date == 'today'
        assert srvy.info == 'RainyDay'

    def test_no_receiver(self):
        srvy = surveys.Survey(self.sources, None, self.frequencies)

        assert isinstance(srvy.sources, dict)
        assert srvy.sources['TxED-1'].center[0] == 0
        assert srvy.receivers == {}
        assert srvy.shape == (self.shape[0], 0, self.shape[2])

    def test_standard_deviation(self):
        srvy0 = surveys.Survey(self.sources, self.receivers, self.frequencies,
                               data=np.ones(self.shape))
        assert srvy0.standard_deviation is None

        srvy1 = surveys.Survey(self.sources, self.receivers, self.frequencies,
                               relative_error=0.5, noise_floor=0.1,
                               data=np.ones(self.shape))
        std = np.sqrt(np.ones(self.shape)*0.1**2 + np.ones(self.shape)*0.5**2)
        assert_allclose(srvy1.standard_deviation.data, std)

        # Test f-dependent noise-floor and source-dependent rel. error.
        nf = np.arange(1, 5)[None, None, :]*1e-15  # No noise floor for f=1.0
        re = np.arange(1, 6)[:, None, None]/100    # No rel. error for Tx1
        srvy2 = surveys.Survey(self.sources, self.receivers, self.frequencies,
                               relative_error=re, noise_floor=nf,
                               data=np.ones(self.shape))

        assert_allclose(srvy2.noise_floor, np.ones(srvy2.shape)*nf)
        assert_allclose(srvy2.relative_error, np.ones(srvy2.shape)*re)
        assert_allclose(srvy2.data._noise_floor[0, 0, :], np.squeeze(nf))
        assert_allclose(srvy2.data._relative_error[:, 0, 0], np.squeeze(re))
        # As data are ones, we can check standard_deviation without it.
        std = np.sqrt(np.ones(srvy2.shape)*nf**2 + np.ones(srvy2.shape)*re**2)
        assert_allclose(srvy2.standard_deviation.data, std)

        # Set the standard deviations
        test_std = np.arange(1, srvy2.size+1).reshape(srvy2.shape)
        srvy2.standard_deviation = test_std
        assert_allclose(srvy2.data._noise_floor[0, 0, :], np.squeeze(nf))
        assert_allclose(srvy2.data._relative_error[:, 0, 0], np.squeeze(re))
        assert_allclose(srvy2.standard_deviation.data, test_std)
        srvy2.standard_deviation = None  # Delete again
        assert_allclose(srvy2.standard_deviation.data, std)

        with pytest.raises(ValueError, match='All values of `standard_dev'):
            srvy2.standard_deviation = np.zeros(srvy2.shape)
        with pytest.raises(ValueError, match='All values of `noise_floor`'):
            srvy2.noise_floor = 0.0
        with pytest.raises(ValueError, match='All values of `relative_error'):
            srvy2.relative_error = 0.0
        with pytest.raises(ValueError, match='operands could not be '):
            srvy2.noise_floor = np.ones(srvy2.shape)[:, :2, :]
        with pytest.raises(ValueError, match='operands could not be '):
            srvy2.relative_error = np.ones(srvy2.shape)[2:6, :, :]

    def test_copy(self, tmpdir):
        sources = emg3d.TxElectricDipole((0, 0, 0, 0, 0))
        receivers = emg3d.RxElectricPoint((1000, 0, 0, 0, 0))
        # This also checks to_dict()/from_dict().
        srvy1 = surveys.Survey(sources, receivers, 1.0)
        # Set observed and standard deviation.
        srvy1.observed = [[[3+3j]]]
        srvy1.standard_deviation = np.array([[[1.1]]])
        srvy2 = srvy1.copy()
        assert srvy1.sources == srvy2.sources

        srvy3 = surveys.Survey(sources, receivers, 1.0, [[[3+3j]]])
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
        sources = [emg3d.TxElectricDipole((x, 0, 0, 0, 0))
                   for x in [0, 50, 100]]
        receivers = [emg3d.RxElectricPoint((x, 0, 0, 0, 0))
                     for x in [1000, 1100, 1200, 1300, 1400]]
        survey = surveys.Survey(
            sources,
            receivers,
            frequencies=(1.0, 2.0, 3.4, 4.0),
            data=np.arange(3*5*4).reshape((3, 5, 4)),
            noise_floor=np.ones((3, 5, 4)),
            relative_error=np.ones((3, 5, 4)),
            name='Test',
        )

        t1 = survey.select('TxED-1', ['RxEP-1', 'RxEP-5'], 'f-1')
        assert t1.shape == (1, 2, 1)

        t2 = survey.select(frequencies=[], receivers='RxEP-1')
        assert t2.shape == (3, 1, 0)

    def test_src_rec_coordinates(self):
        survey = surveys.Survey(
            sources=[
                emg3d.TxElectricDipole((0, 0, 0, 0, 0)),
                emg3d.TxElectricDipole((100, 200, 300, 400, 500, 600))
            ],
            receivers=[
                emg3d.RxElectricPoint((1000, 0, 2000, 0, 0)),
                emg3d.RxElectricPoint((1000, 0, 2000, 0, 0), relative=True)
            ],
            frequencies=1,
        )

        assert_allclose(survey.source_coordinates(),
                        [[0, 150], [0, 350], [0, 550]])

        assert_allclose(survey.receiver_coordinates(),
                        [[1000, 1000, 1150], [0, 0, 350], [2000, 2000, 2550]])

        assert_allclose(survey.receiver_coordinates('TxED-2'),
                        [[1000, 1150], [0, 350], [2000, 2550]])

    def test_add_noise(self):
        offs = np.linspace(0, 10000, 21)
        rec = surveys.txrx_coordinates_to_dict(
                emg3d.electrodes.RxElectricPoint, (offs, 0, 0, 0, 0)
        )

        data = np.logspace(0, -20, offs.size)+1j*np.logspace(0, -20, offs.size)

        survey = surveys.Survey(
            sources=emg3d.electrodes.TxElectricDipole((0, 0, 0, 0, 0)),
            receivers=rec,
            frequencies=1.0,
            data=data,
            relative_error=0.01,
            noise_floor=1e-15
        )

        # Defined cutting
        survey.add_noise(min_offset=1000, min_amplitude=1e-19, add_to='test1')
        # Ensure short offsets are NaN
        assert np.all(np.isnan(survey.data.test1.data[:, :2, :]))
        # Ensure low amplitudes are NaN
        assert np.all(np.isnan(survey.data.test1.data[:, -1:, :]))
        # Ensure no others are none
        assert np.sum(np.isnan(survey.data.test1.data)) == 3

        # No cutting
        survey.add_noise(min_offset=0, min_amplitude=10e-50, add_to='test2')
        assert np.sum(np.isnan(survey.data.test2.data)) == 0

        # Defaults
        survey.add_noise()
        # Ensure low amplitudes are NaN
        assert np.all(np.isnan(survey.data.observed.data[:, -5:, :]))
        assert np.sum(np.isnan(survey.data.observed.data)) == 5


def test_random_noise():
    std = np.ones((2, 3, 4))
    mean_noise = 0.0

    # Default is white noise
    noise_wn = surveys.random_noise(std, mean_noise)
    assert_allclose(std, abs(noise_wn))  # Constant amplitude!

    nm = np.angle(surveys.random_noise(np.ones(100000), mean_noise=0.0)).mean()
    assert_allclose(nm, 0.0, atol=0.02, rtol=1e-1)
    nm = np.angle(surveys.random_noise(np.ones(100000), mean_noise=0.5)).mean()
    assert_allclose(nm, 0.5, rtol=1e-1)

    noise_gu = surveys.random_noise(std, mean_noise, 'gaussian_uncorrelated')
    # Real and imaginary part have to be different.
    assert not np.allclose(noise_gu.imag, noise_gu.real, 0, 1e-15)
    nn_gu = surveys.random_noise(
                np.ones(100000)*.01, 0.5, 'gaussian_uncorrelated')
    assert_allclose(nn_gu.real.mean(), 0.01*0.5, rtol=1e-1)
    assert_allclose(nn_gu.imag.mean(), 0.01*0.5, rtol=1e-1)

    noise_gc = surveys.random_noise(std, mean_noise, 'gaussian_correlated')
    # Real and imaginary part are the same.
    assert_allclose(noise_gc.imag, noise_gc.real)
    nn_gc = surveys.random_noise(
                np.ones(100000)*.01, 0.5, 'gaussian_correlated')
    assert_allclose(nn_gc.real.mean(), 0.01*0.5, rtol=1e-1)
    assert_allclose(nn_gc.imag.mean(), 0.01*0.5, rtol=1e-1)


def test_txrx_coordinates_to_dict():
    sources = surveys.txrx_coordinates_to_dict(
                    emg3d.TxElectricDipole,
                    ([-1, 1, ], 0, 0, [-10, 10], 0), strength=[100, 1])
    assert sources['TxED-1'].strength == 100
    assert sources['TxED-1'].azimuth == -10
    assert_allclose(sources['TxED-1'].center, (-1, 0, 0))
    assert sources['TxED-2'].strength == 1
    assert sources['TxED-2'].azimuth == 10
    assert_allclose(sources['TxED-2'].center, (1, 0, 0))


def test_txrx_lists_to_dict():
    electric = [emg3d.RxElectricPoint((x, 0, 0, 0, 0))
                for x in [1000, 1100]]
    magnetic = surveys.txrx_coordinates_to_dict(
                    emg3d.RxMagneticPoint,
                    ([950, 1050, 1150], 0, 0, 0, 90))
    streamer = emg3d.RxElectricPoint((5, 0, 0, 0, 0), relative=True)

    # If instance, it should give the instance in a dict.
    rec1 = surveys.txrx_lists_to_dict(streamer)
    assert streamer == rec1['RxEP-1']

    # If dict, it should yield the same.
    rec2 = surveys.txrx_lists_to_dict(magnetic)
    assert magnetic['RxMP-1'] == rec2['RxMP-1']

    rec3 = surveys.txrx_lists_to_dict([streamer, electric, magnetic])
    assert rec3['RxEP-1'] == streamer
    assert rec3['RxEP-2'] == electric[0]
    assert rec3['RxEP-3'] == electric[1]
    assert rec3['RxMP-4'] == magnetic['RxMP-1']
    assert rec3['RxMP-5'] == magnetic['RxMP-2']
    assert rec3['RxMP-6'] == magnetic['RxMP-3']

    rec4 = surveys.txrx_lists_to_dict((streamer, tuple(electric), magnetic))
    assert rec3 == rec4


def test_frequencies_to_dict():
    f1 = {'n': 1.8, 1: True}
    f2 = surveys.frequencies_to_dict(f1)
    assert f1 == f2

    f3 = [1.0, 2]
    f4 = surveys.frequencies_to_dict(f3)
    assert f4['f-1'] == 1
    assert f4['f-2'] == 2

    with pytest.raises(ValueError, match="Contains non-unique frequencies: "):
        surveys.frequencies_to_dict([1, 2, 3, 4, 1])
