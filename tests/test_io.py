import pytest
import numpy as np
from copy import deepcopy
from numpy.testing import assert_allclose

import emg3d
from emg3d import io

# Soft dependencies
try:
    import h5py
except ImportError:
    h5py = False
try:
    import xarray
except ImportError:
    xarray = None


class TestSaveLoad:

    frequency = 1.0
    grid = emg3d.TensorMesh([[2, 2], [3, 4], [0.5, 2]], (0, 0, 0))
    field = emg3d.Field(grid)
    model = emg3d.Model(grid, 1)

    tx_e_d = emg3d.TxElectricDipole((0, 1000, 0, 0, -900, -950))
    tx_m_d = emg3d.TxMagneticDipole([[0, 0, -900], [1000, 0, -950]])
    tx_e_w = emg3d.TxElectricWire(([[0, 0, 0], [1, 1, 1], [1, 0, 1]]))
    rx_e_p = emg3d.RxElectricPoint((0, 1000, -950, 0, 20))
    rx_m_p = emg3d.RxMagneticPoint((0, 1000, -950, 20, 0))

    a = np.arange(10.)
    b = 1+5j

    data = {
        'Grid': grid,
        'Model': model,
        'Field': field,
        'a': a,
        'b': b,
    }

    if xarray:
        survey = emg3d.Survey(
            emg3d.surveys.txrx_lists_to_dict([tx_e_d, tx_m_d, tx_e_w]),
            emg3d.surveys.txrx_lists_to_dict([rx_e_p, rx_m_p]),
            frequency
        )
        simulation = emg3d.Simulation(survey, model, gridding='same')
        data['Survey'] = survey
        data['Simulation'] = simulation

    def test_npz(self, tmpdir, capsys):
        io.save(tmpdir+'/test.npz', **self.data)
        outstr, _ = capsys.readouterr()
        assert 'Data saved to «' in outstr
        assert emg3d.__version__ in outstr

        # Save it with other verbosity.
        _, _ = capsys.readouterr()
        io.save(tmpdir+'/test.npz', **self.data, verb=0)
        outstr, _ = capsys.readouterr()
        assert outstr == ""
        out = io.save(tmpdir+'/test.npz', **self.data, verb=-1)
        assert 'Data saved to «' in out

        # Load it.
        out_npz = io.load(str(tmpdir+'/test.npz'), allow_pickle=True)
        outstr, _ = capsys.readouterr()
        assert 'Data loaded from «' in outstr
        assert 'test.npz' in outstr
        assert emg3d.__version__ in outstr

        assert out_npz['Model'] == self.model
        assert_allclose(out_npz['a'], self.a)
        assert out_npz['b'] == self.b
        assert_allclose(self.field.fx, out_npz['Field'].fx)
        assert_allclose(self.grid.cell_volumes, out_npz['Grid'].cell_volumes)

        # Load it with other verbosity.
        _, _ = capsys.readouterr()
        out = io.load(tmpdir+'/test.npz', verb=0)
        outstr, _ = capsys.readouterr()
        assert outstr == ""
        out, out_str = io.load(tmpdir+'/test.npz', verb=-1)
        assert 'Data loaded from «' in out_str

    def test_h5(self, tmpdir):
        if h5py:
            io.save(tmpdir+'/test.h5', **self.data)
            out_h5 = io.load(str(tmpdir+'/test.h5'))
            assert out_h5['Model'] == self.model
            assert_allclose(out_h5['a'], self.a)
            assert out_h5['b'] == self.b
            assert_allclose(self.field.fx, out_h5['Field'].fx)
            assert_allclose(self.grid.cell_volumes,
                            out_h5['Grid'].cell_volumes)

        else:
            with pytest.raises(ImportError):
                io.save(tmpdir+'/test.h5', grid=self.grid)
            with pytest.raises(ImportError):
                io.load(str(tmpdir+'/test-h5.h5'))

    def test_json(self, tmpdir):
        io.save(tmpdir+'/test.json', **self.data)
        out_json = io.load(str(tmpdir+'/test.json'))
        assert out_json['Model'] == self.model
        assert_allclose(out_json['a'], self.a)
        assert out_json['b'] == self.b
        assert_allclose(self.field.fx, out_json['Field'].fx)
        assert_allclose(self.grid.cell_volumes, out_json['Grid'].cell_volumes)

    def test_warnings(self, tmpdir, capsys):
        # Check message from loading another file
        data = io._dict_serialize({'meshes': self.grid})
        fdata = io._dict_flatten(data)
        np.savez_compressed(tmpdir+'/test2.npz', **fdata)
        _ = io.load(str(tmpdir+'/test2.npz'), allow_pickle=True)
        outstr, _ = capsys.readouterr()
        assert "[version/format/date unknown; not created by emg" in outstr

        # Unknown keyword.
        with pytest.raises(TypeError, match="Unexpected "):
            io.load('ttt.npz', stupidkeyword='a')

        # Unknown extension.
        with pytest.raises(ValueError, match="Unknown extension '.abc'"):
            io.save(tmpdir+'/testwrongextension.abc', something=1)
        with pytest.raises(ValueError, match="Unknown extension '.abc'"):
            io.load(tmpdir+'/testwrongextension.abc')


def test_dict_serialize_deserialize():
    frequency = 1.0
    grid = emg3d.TensorMesh([[2, 2], [3, 4], [0.5, 2]], (0, 0, 0))
    field = emg3d.Field(grid)
    model = emg3d.Model(grid, 1)

    tx_e_d = emg3d.TxElectricDipole((0, 1000, 0, 0, -900, -950))
    tx_m_d = emg3d.TxMagneticDipole([[0, 0, -900], [1000, 0, -950]])
    tx_e_w = emg3d.TxElectricWire(([[0, 0, 0], [1, 1, 1], [1, 0, 1]]))
    rx_e_p = emg3d.RxElectricPoint((0, 1000, -950, 0, 20))
    rx_m_p = emg3d.RxMagneticPoint((0, 1000, -950, 20, 0))

    data = {
        'Grid': grid,
        'Model': model,
        'Field': field,
    }

    if xarray:
        survey = emg3d.Survey(
            emg3d.surveys.txrx_lists_to_dict([tx_e_d, tx_m_d, tx_e_w]),
            emg3d.surveys.txrx_lists_to_dict([rx_e_p, rx_m_p]),
            frequency
        )
        simulation = emg3d.Simulation(survey, model, gridding='same')
        data['Survey'] = survey
        data['Simulation'] = simulation

    # Get everything into a serialized dict.
    out = io._dict_serialize(data)

    # Get everything back.
    io._nonetype_to_none(out)
    keep = deepcopy(out)
    io._dict_deserialize(out)

    assert data.keys() == out.keys()
    assert out['Field'] == field
    assert out['Grid'] == grid
    assert out['Model'] == model
    if xarray:
        assert out['Survey'].sources == survey.sources
        assert (out['Simulation'].survey.receivers ==
                simulation.survey.receivers)

    del keep['Grid']['hx']
    with pytest.warns(UserWarning, match="Could not de-serialize"):
        io._dict_deserialize(keep)


def test_nonetype_to_none():
    orig = {'1': 'NoneType', '2': np.bool_(1),
            '3': np.array([True, ]),
            '4': {'5': 'NoneType', '6': np.array(False)}}

    io._nonetype_to_none(orig)

    assert orig['1'] is None
    assert orig['2'] is True
    assert orig['3'] is True
    assert orig['4']['5'] is None
    assert orig['4']['6'] is False


def test_dict_flatten_unflatten():
    orig = {'one': 1, 'two': {'three': {'four': 4}},
            'np-string': np.array('test')}  # np-strings are created by npz/h5
    flat = io._dict_flatten(orig)

    assert flat['one'] == 1
    assert flat['two>three>four'] == 4
    assert flat['np-string'].dtype.type == np.str_

    data = io._dict_unflatten(flat)
    assert data['one'] == 1
    assert data['two']['three']['four'] == 4
    assert isinstance(data['np-string'], str)  # A string!
    assert data['np-string'] == 'test'


def test_dict_dearray_decomp_array_comp():
    d1 = np.arange(2*3*4).reshape((2, 3, 4))
    d2 = np.arange(10) + 1j*np.arange(10)[::-1]
    orig = {'d1': d1, 'd3': {'d2': d2}}

    deac = io._dict_dearray_decomp(orig)

    assert isinstance(deac['d1__array-int64'], list)
    assert isinstance(deac['d3']['d2__complex__array-float64'], list)

    data = io._dict_array_comp(deac)

    assert_allclose(d1, data['d1'])
    assert_allclose(d2, data['d3']['d2'])


def test_hdf5_dump_load(tmpdir):
    d1 = np.arange(10)
    d2 = 3
    d3 = True
    d4 = 'test'
    d5 = ['1', '2', '3']

    orig = {'d1': d1, 'd2': d2, 'd3': d3, 'ddict': {'d4': d4}, 'd5': d5}

    io._hdf5_dump(fname=str(tmpdir) + 'test.h5', data=orig, compression='gzip')

    out = io._hdf5_load(fname=str(tmpdir) + 'test.h5')

    assert_allclose(out['d1'], d1)
    assert out['d2'] == 3
    assert out['d3']
    assert out['ddict']['d4'] == 'test'
    assert out['d5'] == ['1', '2', '3']
