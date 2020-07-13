import re
import pytest
import numpy as np
from timeit import default_timer
from numpy.testing import assert_allclose

# Soft dependencies
try:
    import scooby
except ImportError:
    scooby = None
try:
    import empymod
except ImportError:
    empymod = None

from emg3d import utils


# TIME DOMAIN
@pytest.mark.skipif(empymod is None, reason="empymod not installed.")
class TestFourier:
    def test_defaults(self, capsys):
        time = np.logspace(-2, 2)
        fmin = 0.01
        fmax = 100

        Fourier = utils.Fourier(time, fmin, fmax)
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
        assert_allclose(time, Fourier.time, 0, 0)
        assert Fourier.verb == 3          # Verbose by default
        assert 'Key 201 CosSin (2012)' in out
        assert 'Req. freq' in out
        assert 'Calc. freq' in out
        assert Fourier.freq_calc.min() >= fmin
        assert Fourier.freq_calc.max() <= fmax

        # Check frequencies to extrapolate.
        assert_allclose(Fourier.freq_extrapolate,
                        Fourier.freq_req[Fourier.freq_req < fmin])

        # If not freq_inp nor every_x_freq, interpolate and calc have to be the
        # same.
        assert_allclose(Fourier.freq_interpolate, Fourier.freq_calc)

        # Change time, ensure it changes required frequencies.
        freq_req = Fourier.freq_req
        time2 = np.logspace(-1, 2)
        Fourier.time = time2
        assert freq_req.size != Fourier.freq_req.size

    def test_kwargs(self, capsys):
        time = np.logspace(-1, 1)
        fmin = 0.1
        fmax = 10
        freq_inp = np.logspace(-1, 1, 11)
        xfreq = 10

        # freq_inp; verb=0
        _, _ = capsys.readouterr()
        Fourier1 = utils.Fourier(time, fmin, fmax, freq_inp=freq_inp, verb=0,
                                 ftarg={'kind': 'sin'})
        out, _ = capsys.readouterr()
        assert '' == out
        assert_allclose(freq_inp, Fourier1.freq_calc, 0, 0)

        # freq_inp AND every_x_freq => re-sets every_x_freq.
        Fourier2 = utils.Fourier(time, fmin, fmax, every_x_freq=xfreq,
                                 freq_inp=freq_inp, verb=1)
        out, _ = capsys.readouterr()
        assert 'Re-setting `every_x_freq=None`' in out
        assert_allclose(freq_inp, Fourier2.freq_calc, 0, 0)
        assert_allclose(Fourier1.freq_calc, Fourier2.freq_calc)

        # Now set every_x_freq again => re-sets freq_inp.
        Fourier2.every_x_freq = xfreq
        out, _ = capsys.readouterr()
        assert 'Re-setting `freq_inp=None`' in out
        assert_allclose(Fourier2.freq_coarse, Fourier2.freq_req[::xfreq])
        assert Fourier2.freq_inp is None
        test = Fourier2.freq_req[::xfreq][
                (Fourier2.freq_req[::xfreq] >= fmin) &
                (Fourier2.freq_req[::xfreq] <= fmax)]
        assert_allclose(Fourier2.freq_calc, test)

        # And back
        Fourier2.freq_inp = freq_inp
        out, _ = capsys.readouterr()
        assert 'Re-setting `every_x_freq=None`' in out
        assert_allclose(Fourier2.freq_calc, freq_inp)
        assert Fourier2.every_x_freq is None

        # Unknown argument, must fail with TypeError.
        with pytest.raises(TypeError):
            utils.Fourier(time, fmin, fmax, does_not_exist=0)

    def test_setters(self, capsys):
        time = np.logspace(-1.4, 1.4)
        fmin = 0.1
        fmax = 10

        # freq_inp; verb=0
        _, _ = capsys.readouterr()
        Fourier1 = utils.Fourier(time, fmin=np.pi/10, fmax=np.pi*10)
        Fourier1.fmin = fmin
        Fourier1.fmax = fmax
        Fourier1.signal = -1
        Fourier1.fourier_arguments('fftlog', {'pts_per_dec': 5})
        assert Fourier1.ft == 'fftlog'
        assert Fourier1.ftarg['pts_per_dec'] == 5
        assert Fourier1.ftarg['mu'] == -0.5  # cosine, as signal == -1

    def test_interpolation(self, capsys):
        time = np.logspace(-2, 1, 201)
        model = {'src': [0, 0, 0], 'rec': [900, 0, 0], 'res': 1,
                 'depth': [], 'verb': 1}
        Fourier = utils.Fourier(time, 0.005, 10)

        # Calculate data.
        data_true = empymod.dipole(freqtime=Fourier.freq_req, **model)
        data = empymod.dipole(freqtime=Fourier.freq_calc, **model)

        # Interpolate.
        data_int = Fourier.interpolate(data)

        # Compare, extrapolate < 0.05; interpolate equal.
        assert_allclose(data_int[Fourier.freq_extrapolate_i].imag,
                        data_true[Fourier.freq_extrapolate_i].imag, rtol=0.05)
        assert_allclose(data_int[Fourier.freq_calc_i].imag,
                        data_true[Fourier.freq_calc_i].imag)

        # Now set every_x_freq and again.
        Fourier.every_x_freq = 2

        data = empymod.dipole(freqtime=Fourier.freq_calc, **model)

        # Interpolate.
        data_int = Fourier.interpolate(data)

        # Compare, extrapolate < 0.05; interpolate < 0.01.
        assert_allclose(data_int[Fourier.freq_extrapolate_i].imag,
                        data_true[Fourier.freq_extrapolate_i].imag, rtol=0.05)
        assert_allclose(data_int[Fourier.freq_interpolate_i].imag,
                        data_true[Fourier.freq_interpolate_i].imag, rtol=0.01)

    def test_freq2transform(self, capsys):
        time = np.linspace(0.1, 10, 101)
        x = 900
        model = {'src': [0, 0, 0], 'rec': [x, 0, 0], 'res': 1,
                 'depth': [], 'verb': 1}

        # Initiate Fourier instance.
        Fourier = utils.Fourier(time, 0.001, 100)

        # Calculate required frequencies.
        data = empymod.dipole(freqtime=Fourier.freq_calc, **model)

        # Transform the data.
        tdata = Fourier.freq2time(data, x)

        # Calculate data in empymod.
        data_true = empymod.dipole(freqtime=time, signal=0, **model)

        # Compare.
        assert_allclose(data_true, tdata, rtol=1e-4)


# FUNCTIONS RELATED TO TIMING
def test_Time():
    t0 = default_timer()  # Create almost at the same time a
    time = utils.Time()   # t0-stamp and a Time-instance.

    # Ensure they are the same.
    assert_allclose(t0, time.t0, atol=1e-3)

    # Ensure `now` is a string of numbers and :.
    out = time.now
    assert re.match(r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]', out)

    # This should have taken less then 1s.
    out = time.runtime
    assert "0:00:00" == str(out)

    # Check representation of Time.
    assert 'Runtime : 0:00:0' in time.__repr__()


# OTHER
@pytest.mark.skipif(scooby is None, reason="scooby not installed.")
def test_report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    # Reporting is now done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    if scooby:
        out1 = utils.Report()
        out2 = scooby.Report(
                core=['numpy', 'scipy', 'numba', 'emg3d'],
                optional=['empymod', 'discretize', 'h5py', 'matplotlib',
                          'IPython'],
                ncol=4)

        # Ensure they're the same; exclude time to avoid errors.
        assert out1.__repr__()[115:] == out2.__repr__()[115:]

    else:  # soft dependency
        _ = utils.Report()
        out, _ = capsys.readouterr()  # Empty capsys
        assert 'WARNING :: `emg3d.Report` requires `scooby`' in out
