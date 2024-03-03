import pytest
import numpy as np
from numpy.testing import assert_allclose

from emg3d import time

try:
    import empymod
except ImportError:
    empymod = None


@pytest.mark.skipif(empymod is None, reason="empymod not installed.")
class TestFourier:
    def test_defaults(self, capsys):
        times = np.logspace(-2, 2)
        fmin = 0.01
        fmax = 100

        Fourier = time.Fourier(times, fmin, fmax)
        out, _ = capsys.readouterr()

        # Check representation of Fourier.
        assert '0.01-100.0 s' in Fourier.__repr__()
        assert '0.01-100 Hz' in Fourier.__repr__()
        assert Fourier.every_x_freq is None
        assert Fourier.fmin == fmin
        assert Fourier.fmax == fmax
        assert 'dlf' in Fourier.__repr__()
        assert Fourier.ft == 'dlf'
        assert Fourier.ftarg['pts_per_dec'] == -1.0   # Convolution DLF
        assert Fourier.ftarg['kind'] == 'sin'  # Sine-DLF is default
        assert Fourier.signal == 0        # Impulse respons
        assert_allclose(times, Fourier.time, 0, 0)
        assert Fourier.verb == 3          # Verbose by default
        assert 'key' in out
        assert 'Req. freq' in out
        assert 'Calc. freq' in out
        assert Fourier.freq_compute.min() >= fmin
        assert Fourier.freq_compute.max() <= fmax

        # Check frequencies to extrapolate.
        assert_allclose(Fourier.freq_extrapolate,
                        Fourier.freq_required[Fourier.freq_required < fmin])

        # If not input_freq nor every_x_freq, interpolate and calc have to be
        # the same.
        assert_allclose(Fourier.freq_interpolate, Fourier.freq_compute)

        # Change time, ensure it changes required frequencies.
        freq_required = Fourier.freq_required
        time2 = np.logspace(-1, 2)
        Fourier.time = time2
        assert freq_required.size != Fourier.freq_required.size

    def test_kwargs(self):
        times = np.logspace(-1, 1)
        fmin = 0.1 - np.finfo(float).eps
        fmax = 10
        input_freq = np.logspace(-1, 1, 11)
        xfreq = 10

        # input_freq; verb=0
        Fourier1 = time.Fourier(times, fmin, fmax, input_freq=input_freq,
                                verb=0, ftarg={'kind': 'sin'})
        assert_allclose(input_freq, Fourier1.freq_compute, 0, 0)

        # input_freq AND every_x_freq => re-sets every_x_freq.
        with pytest.warns(UserWarning, match='Re-setting'):
            Fourier2 = time.Fourier(times, fmin, fmax, every_x_freq=xfreq,
                                    input_freq=input_freq, verb=1)
        assert_allclose(input_freq, Fourier2.freq_compute, 0, 0)
        assert_allclose(Fourier1.freq_compute, Fourier2.freq_compute)

        # Now set every_x_freq again => re-sets input_freq.
        with pytest.warns(UserWarning, match='Re-setting'):
            Fourier2.every_x_freq = xfreq
        assert_allclose(Fourier2.freq_coarse, Fourier2.freq_required[::xfreq])
        assert Fourier2.input_freq is None
        test = Fourier2.freq_required[::xfreq][
                (Fourier2.freq_required[::xfreq] >= fmin) &
                (Fourier2.freq_required[::xfreq] <= fmax)]
        assert_allclose(Fourier2.freq_compute, test)

        # And back
        with pytest.warns(UserWarning, match='Re-setting'):
            Fourier2.input_freq = input_freq
        assert_allclose(Fourier2.freq_compute, input_freq)
        assert Fourier2.every_x_freq is None

        # Unknown argument, must fail with TypeError.
        with pytest.raises(TypeError):
            time.Fourier(times, fmin, fmax, does_not_exist=0)

    def test_setters(self, capsys):
        times = np.logspace(-1.4, 1.4)
        fmin = 0.1
        fmax = 10

        # input_freq; verb=0
        _, _ = capsys.readouterr()
        Fourier1 = time.Fourier(times, fmin=np.pi/10, fmax=np.pi*10)
        Fourier1.fmin = fmin
        Fourier1.fmax = fmax
        Fourier1.signal = -1
        Fourier1.fourier_arguments('fftlog', {'pts_per_dec': 5})
        assert Fourier1.ft == 'fftlog'
        assert Fourier1.ftarg['pts_per_dec'] == 5
        assert Fourier1.ftarg['mu'] == -0.5  # cosine, as signal == -1

    def test_interpolation(self, capsys):
        times = np.logspace(-2, 1, 201)
        model = {'src': [0, 0, 0], 'rec': [900, 0, 0], 'res': 1,
                 'depth': [], 'verb': 1}
        Fourier = time.Fourier(times, 0.005, 10)

        # Calculate data.
        data_true = empymod.dipole(freqtime=Fourier.freq_required, **model)
        data = empymod.dipole(freqtime=Fourier.freq_compute, **model)

        # Interpolate.
        data_int = Fourier.interpolate(data)

        # Compare, extrapolate < 0.05; interpolate equal.
        assert_allclose(data_int[Fourier.ifreq_extrapolate].imag,
                        data_true[Fourier.ifreq_extrapolate].imag, rtol=0.05)
        assert_allclose(data_int[Fourier.ifreq_compute].imag,
                        data_true[Fourier.ifreq_compute].imag)

        # Now set every_x_freq and again.
        Fourier.every_x_freq = 2

        data = empymod.dipole(freqtime=Fourier.freq_compute, **model)

        # Interpolate.
        data_int = Fourier.interpolate(data)

        # Compare, extrapolate < 0.05; interpolate < 0.01.
        assert_allclose(data_int[Fourier.ifreq_extrapolate].imag,
                        data_true[Fourier.ifreq_extrapolate].imag, rtol=0.05)
        assert_allclose(data_int[Fourier.ifreq_interpolate].imag,
                        data_true[Fourier.ifreq_interpolate].imag, rtol=0.01)

    def test_freq2transform(self, capsys):
        times = np.linspace(0.1, 10, 101)
        x = 900
        model = {'src': [0, 0, 0], 'rec': [x, 0, 0], 'res': 1,
                 'depth': [], 'verb': 1}

        # Initiate Fourier instance.
        Fourier = time.Fourier(times, 0.001, 100)

        # Calculate required frequencies.
        data = empymod.dipole(freqtime=Fourier.freq_compute, **model)

        # Transform the data.
        tdata = Fourier.freq2time(data, x)

        # Calculate data in empymod.
        data_true = empymod.dipole(freqtime=times, signal=0, **model)

        # Compare.
        assert_allclose(data_true, tdata, rtol=1e-4)


def test_all_dir():
    assert set(time.__all__) == set(dir(time))
