# import os
import pytest
import numpy as np
# from numpy.testing import assert_allclose

import emg3d
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
    import h5py
except ImportError:
    h5py = False
try:
    import pygimli
except ImportError:
    pygimli = False


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
@pytest.mark.skipif(pygimli is None, reason="pygimli not installed.")
class TestPygimli():

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
        solver_opts={'tol_gradient': 1e-3},
        tqdm_opts=False,
    )
    sim.compute(observed=True)
    sim.clean('computed')

    sim.model = model_start

    sim.compute()
    sim.survey.data['start'] = sim.survey.data.synthetic
    sim.clean('computed')

    sim.tol_gradient = 1e-2  # Reduce further
    model_start = sim.model.copy()
    grid = model_true.grid

    markers = np.zeros(model_start.shape, dtype=int)
    markers[1, 1, 1] = 1

    # Without regions
    fop = ipygimli.Kernel(simulation=sim, pgthreads=1)

    INV = ipygimli.Inversion(fop=fop)
    INV.inv.setCGLSTolerance(10)
    INV.inv.setMaxCGLSIter(30)

    errmodel = INV.run(maxIter=2, lam=0.10)

    markers = np.zeros(model_start.shape, dtype=int)
    markers[1, 1, 1] = 1
    markers[0, :, :] = 2
    markers[2, :, :] = 3

    # With regions
    fop = ipygimli.Kernel(simulation=sim, markers=markers, pgthreads=1)

    INV = ipygimli.Inversion(fop=fop)
    INV.inv.setCGLSTolerance(10)
    INV.inv.setMaxCGLSIter(30)

    INV.fop.setRegionProperties(1, limits=(0.0001, 2), startModel=1.0)
    INV.fop.setRegionProperties(0, background=True)
    INV.fop.setRegionProperties(2, fix=True, startModel=1)
    INV.fop.setRegionProperties(
            3, single=True, limits=(0.99999, 1.00001), startModel=1.0)

    errmodel = INV.run(maxIter=2, lam=1)


def test_all_dir():
    assert set(ipygimli.__all__) == set(dir(ipygimli))
