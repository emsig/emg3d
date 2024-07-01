import pytest
import logging
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import inversion
from emg3d.inversion import pygimli as ipygimli


# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None
try:
    import discretize
except ImportError:
    discretize = None
try:
    import pygimli
except ImportError:
    pygimli = None


LOGGER = logging.getLogger(__name__)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestPygimli():

    if xarray is not None:
        # Create data
        survey = emg3d.surveys.Survey(
            sources=emg3d.TxElectricDipole((0, 0, -250, 0, 0)),
            receivers=emg3d.RxElectricPoint((0, 0, -1250, 0, 0)),
            frequencies=1.0,
            noise_floor=1e-17,
            relative_error=0.05,
        )

        hx = np.ones(3)*500.0
        grid = emg3d.TensorMesh([hx, hx, hx], [-750, -750, -1500])

        model_start = emg3d.Model(grid, 1.0, mapping='Conductivity')
        model_true = emg3d.Model(grid, 1.0, mapping='Conductivity')
        model_true.property_x[1, 1, 1] = 1/1000

        # Create an emg3d Simulation instance
        sim = emg3d.simulations.Simulation(
            survey=survey.copy(),
            model=model_true,
            gridding='both',
            max_workers=1,
            gridding_opts={'center_on_edge': False},
            receiver_interpolation='linear',
            solver_opts={'tol_gradient': 1e-2},
            tqdm_opts=False,
        )
        sim.compute(observed=True)
        synthetic = sim.survey.data.synthetic.copy()
        sim.clean('computed')

        sim.model = model_start

        sim.compute()
        sim.survey.data['start'] = sim.survey.data.synthetic
        sim.clean('computed')

        markers = np.zeros(sim.model.shape, dtype=int)
        markers[1, 1, 1] = 1
        markers[0, :, :] = 2
        markers[2, :, :] = 3

        def set_regions(self, fop):
            fop.setRegionProperties(1, limits=(0.0001, 2), startModel=1.0)
            fop.setRegionProperties(0, background=True)
            fop.setRegionProperties(2, fix=True, startModel=1)
            fop.setRegionProperties(
                    3, single=True, limits=(0.99999, 1.00001), startModel=1.0)

    @pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
    def test_Kernel_errors(self):

        sim = self.sim.copy()
        sim.model = emg3d.Model(sim.model.grid, mapping='Resistivity')
        with pytest.raises(NotImplementedError, match='for Resistivity'):
            _ = ipygimli.Kernel(simulation=sim)

        sim.model = emg3d.Model(sim.model.grid, 1, 2)
        with pytest.raises(NotImplementedError, match='for HTI'):
            _ = ipygimli.Kernel(simulation=sim)

    @pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
    def test_Kernel_markers(self):

        # No regions
        sim = self.sim.copy()
        fop = ipygimli.Kernel(simulation=sim)
        assert_allclose(fop.markers, np.arange(sim.model.size, dtype=int))
        assert fop.fullmodel is True

        # Regions
        fop = ipygimli.Kernel(simulation=sim, markers=self.markers)
        self.set_regions(fop)
        assert_allclose(fop.markers, self.markers)
        assert fop.fullmodel is False

    @pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
    def test_Kernel_conversions(self):
        sim = self.sim.copy()
        fop1 = ipygimli.Kernel(simulation=sim)
        fop2 = ipygimli.Kernel(simulation=sim, markers=self.markers)
        self.set_regions(fop2)

        for fop in [fop1, fop2]:

            assert_allclose(
                fop.model2emg3d(fop.createStartModel()),
                sim.model.property_x
            )

            assert_allclose(
                sim.data.observed,
                fop.data2emg3d(fop.data2gimli(sim.data.observed.data))
            )

            assert_allclose(
                sim.model.property_x,
                fop.model2emg3d(fop.model2gimli(sim.model.property_x))
            )
            data = sim.survey.standard_deviation.data
            assert fop.data2gimli(data).dtype == np.float64

    @pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
    def test_Kernel_response(self):
        sim = self.sim.copy()
        sim.model = self.model_true
        fop = ipygimli.Kernel(simulation=sim)
        assert_allclose(
            fop.data2emg3d(fop.response(fop.createStartModel())),
            self.synthetic.data
        )

    @pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    def test_Inversion_noregions(self, caplog):
        # Mainly "runtest"
        sim = self.sim.copy()

        fop = ipygimli.Kernel(simulation=sim, pgthreads=1)

        INV = ipygimli.Inversion(fop=fop)
        INV.inv.setCGLSTolerance(10)
        INV.inv.setMaxCGLSIter(30)

        _ = INV.run(maxIter=2, lam=0.1)

        assert 'pyGIMLi(emg3d)' in caplog.text
        assert 'Created startmodel from forward operator: 27' in caplog.text
        assert 'λ = 0.1' in caplog.text

        assert INV.inv.chi2() < 1

    @pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
    @pytest.mark.skipif(discretize is None, reason="discretize not installed.")
    def test_Inversion_regions(self, caplog):
        # Mainly "runtest"
        sim = self.sim.copy()

        fop = ipygimli.Kernel(
                simulation=sim, markers=self.markers, pgthreads=1)

        INV = ipygimli.Inversion(fop=fop)
        INV.inv.setCGLSTolerance(10)
        INV.inv.setMaxCGLSIter(30)

        INV.fop.setRegionProperties(1, limits=(0.0001, 2), startModel=1.0)
        INV.fop.setRegionProperties(0, background=True)
        INV.fop.setRegionProperties(2, fix=True, startModel=1)
        INV.fop.setRegionProperties(
                3, single=True, limits=(0.99999, 1.00001), startModel=1.0)

        _ = INV.run(maxIter=2, lam=1)

        assert 'pyGIMLi(emg3d)' in caplog.text
        assert 'Created startmodel from region infos: 2' in caplog.text
        assert 'λ = 1.0' in caplog.text

        assert INV.inv.chi2() < 1


def test_all_dir():
    assert set(inversion.__all__) == set(dir(inversion))
    assert set(ipygimli.__all__) == set(dir(ipygimli))
