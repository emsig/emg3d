import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import fields


def test_basics_I():

    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = emg3d.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )


def test_basics_I():

    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = emg3d.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )

    model = emg3d.Model(grid, res)
    sfield = fields.get_source_field(grid, src, freq)
    efield = emg3d.solve(model, sfield, semicoarsening=True,
                            sslsolver=True, linerelaxation=True, verb=1)


def test_basics_I():

    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = emg3d.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )

    model = emg3d.Model(grid, res)
    sfield = fields.get_source_field(grid, src, freq)
    efield = emg3d.solve(model, sfield, semicoarsening=True,
                            sslsolver=True, linerelaxation=True, verb=1)

    epm = np.array([-1.27832028e-11+1.21383502e-11j,
                    -1.90064149e-12+7.51937145e-12j,
                    1.09602131e-12+3.33066197e-12j,
                    1.25359248e-12+1.02630145e-12j])
    e3d = fields.get_receiver(efield, rec)

    assert_allclose(epm, e3d, rtol=0.1)


def test_basics_I():

    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = emg3d.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )

    model = emg3d.Model(grid, res)
    sfield = fields.get_source_field(grid, src, freq)
    efield = emg3d.solve(model, sfield, semicoarsening=True,
                            sslsolver=True, linerelaxation=True, verb=1)

    epm = np.array([-1.27832028e-11+1.21383502e-11j,
                    -1.90064149e-12+7.51937145e-12j,
                    1.09602131e-12+3.33066197e-12j,
                    1.25359248e-12+1.02630145e-12j])
    e3d = fields.get_receiver(efield, rec)

    assert_allclose(epm, e3d, rtol=0.1)

    rec_inst = [emg3d.RxElectricPoint((rec[0][i], rec[1][i], *rec[2:]))
                for i in range(rec[0].size)]
    e3d_inst = fields.get_receiver(efield, rec_inst)
    assert_allclose(e3d_inst, e3d)


def test_basics_I():

    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = emg3d.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )

    model = emg3d.Model(grid, res)
    sfield = fields.get_source_field(grid, src, freq)
    efield = emg3d.solve(model, sfield, semicoarsening=True,
                            sslsolver=True, linerelaxation=True, verb=1)

    epm = np.array([-1.27832028e-11+1.21383502e-11j,
                    -1.90064149e-12+7.51937145e-12j,
                    1.09602131e-12+3.33066197e-12j,
                    1.25359248e-12+1.02630145e-12j])
    e3d = fields.get_receiver(efield, rec)

    assert_allclose(epm, e3d, rtol=0.1)

    rec_inst = [emg3d.RxElectricPoint((rec[0][i], rec[1][i], *rec[2:]))
                for i in range(rec[0].size)]
    e3d_inst = fields.get_receiver(efield, rec_inst)
    assert_allclose(e3d_inst, e3d)

    e3d_inst = fields.get_receiver(efield, rec_inst[0])
    assert_allclose(e3d_inst, e3d[0])


def test_basics_I():

    x = np.array([400, 450, 500, 550])
    rec = (x, x*0, 0, 20, 70)
    res = 0.3
    src = (0, 0, 0, 0, 0)
    freq = 10

    grid = emg3d.construct_mesh(
            frequency=freq,
            center=(0, 0, 0),
            properties=res,
            domain=[[0, 1000], [-25, 25], [-25, 25]],
            min_width_limits=20,
    )

    model = emg3d.Model(grid, res)
    sfield = fields.get_source_field(grid, src, freq)
    efield = emg3d.solve(model, sfield, semicoarsening=True,
                            sslsolver=True, linerelaxation=True, verb=1)

    epm = np.array([-1.27832028e-11+1.21383502e-11j,
                    -1.90064149e-12+7.51937145e-12j,
                    1.09602131e-12+3.33066197e-12j,
                    1.25359248e-12+1.02630145e-12j])
    e3d = fields.get_receiver(efield, rec)

    assert_allclose(epm, e3d, rtol=0.1)

    rec_inst = [emg3d.RxElectricPoint((rec[0][i], rec[1][i], *rec[2:]))
                for i in range(rec[0].size)]
    e3d_inst = fields.get_receiver(efield, rec_inst)
    assert_allclose(e3d_inst, e3d)

    e3d_inst = fields.get_receiver(efield, rec_inst[0])
    assert_allclose(e3d_inst, e3d[0])

    h = np.ones(4)*100
    grid = emg3d.TensorMesh([h, h, h], (-200, -200, -200))
    model = emg3d.Model(grid)
    sfield = fields.get_source_field(
            grid=grid, source=[0, 0, 0, 0, 0], frequency=10)
    out = emg3d.solve(model=model, sfield=sfield, plain=True, verb=0)
    off = np.arange(11)*50-250
    resp = out.get_receiver((off, off, off, 0, 0))
    assert_allclose(np.isfinite(resp),
                    np.r_[3*[False, ], 5*[True, ], 3*[False, ]])
