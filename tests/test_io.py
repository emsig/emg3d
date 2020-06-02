import os
import pytest
import numpy as np
from copy import deepcopy as dc
from numpy.testing import assert_allclose

from emg3d import meshes, models, fields, utils, io

try:
    import h5py
except ImportError:
    h5py = False


def create_dummy(nx, ny, nz, imag=True):
    """Return complex dummy arrays of shape nx*ny*nz.

    Numbers are from 1..nx*ny*nz for the real part, and 1/100 of it for the
    imaginary part.

    """
    if imag:
        out = np.arange(1, nx*ny*nz+1) + 1j*np.arange(1, nx*ny*nz+1)/100.
    else:
        out = np.arange(1, nx*ny*nz+1)
    return out.reshape(nx, ny, nz)


def test_data_write_read(tmpdir, capsys):
    # Create test data
    grid = utils.TensorMesh(
            [np.array([100, 4]), np.array([100, 8]), np.array([100, 16])],
            np.zeros(3))

    freq = np.pi

    model = utils.Model(grid, res_x=1., res_y=2., res_z=3., mu_r=4.)

    e1 = create_dummy(*grid.vnEx)
    e2 = create_dummy(*grid.vnEy)
    e3 = create_dummy(*grid.vnEz)
    ee = utils.Field(e1, e2, e3, freq=freq)

    # Write and read data, single arguments
    io.data_write('testthis', 'ee', ee, tmpdir, -1)
    ee_out = io.data_read('testthis', 'ee', tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    assert_allclose(ee.smu0, ee_out.smu0)
    assert_allclose(ee.sval, ee_out.sval)
    assert_allclose(ee.freq, ee_out.freq)

    # Write and read data, multi arguments
    args = ('grid', 'ee', 'model')
    io.data_write('testthis', args, (grid, ee, model), tmpdir, -1)
    grid_out, ee_out, model_out = io.data_read('testthis', args, tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(grid_out, attr)

    # Ensure volume averages got deleted or do not exist anyway.
    assert hasattr(grid_out, '_vol') is False
    assert hasattr(model_out, '_eta_x') is False
    assert hasattr(model_out, '_zeta') is False

    # Ensure they can be reconstructed
    assert_allclose(grid.vol, grid_out.vol)

    # Write and read data, None
    io.data_write('testthis', ('grid', 'ee'), (grid, ee), tmpdir, -1)
    out = io.data_read('testthis', path=tmpdir)

    # Compare data
    assert_allclose(ee, ee_out)
    for attr in ['nCx', 'nCy', 'nCz']:
        assert getattr(grid, attr) == getattr(out['grid'], attr)

    # Test exists-argument 0
    _, _ = capsys.readouterr()  # Clean-up
    io.data_write('testthis', 'ee', ee*2, tmpdir, 0)
    out, _ = capsys.readouterr()
    datout = io.data_read('testthis', path=tmpdir)
    assert 'NOT SAVING THE DATA' in out
    assert_allclose(datout['ee'], ee)

    # Test exists-argument 1
    io.data_write('testthis', 'ee2', ee, tmpdir, 1)
    out, _ = capsys.readouterr()
    assert 'appending to it' in out
    io.data_write('testthis', ['ee', 'ee2'], [ee*2, ee], tmpdir, 1)
    out, _ = capsys.readouterr()
    assert "overwriting existing key(s) 'ee', 'ee2'" in out
    datout = io.data_read('testthis', path=tmpdir)
    assert_allclose(datout['ee'], ee*2)
    assert_allclose(datout['ee2'], ee)

    # Check if file is missing.
    os.remove(tmpdir+'/testthis.dat')
    out = io.data_read('testthis', path=tmpdir)
    assert out is None
    out1, out2 = io.data_read('testthis', ['ee', 'ee2'], path=tmpdir)
    assert out1 is None
    assert out2 is None
    io.data_write('testthis', ['ee', 'ee2'], [ee*2, ee], tmpdir, -1)


def test_save_and_load(tmpdir, capsys):

    # Create some dummy data
    grid = meshes.TensorMesh(
            [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
            np.zeros(3))

    # TensorMesh grid to 'simulate' discretize TensorMesh without `to_dict`.
    class TensorMesh:
        pass

    grid2 = TensorMesh()
    grid2.hx = grid.hx
    grid2.hy = grid.hy
    grid2.hz = grid.hz
    grid2.x0 = grid.x0

    grid3 = TensorMesh()  # "Broken" mesh
    grid3.hx = grid.hx
    grid3.x0 = grid.x0

    # Some field.
    field = fields.Field(grid)
    field.field = np.arange(grid.nE)+1j*np.ones(grid.nE)
    field.ensure_pec

    # Some model.
    res_x = create_dummy(*grid.vnC, False)
    res_y = res_x/2.0
    res_z = res_x*1.4
    mu_r = res_x*1.11
    model = models.Model(grid, res_x, res_y, res_z, mu_r=mu_r)

    # Save it.
    io.save(tmpdir+'/test.npz', emg3d=grid, discretize=grid2, model=model,
            broken=grid3, a=None,
            field=field, what={'f': field.fx, 12: 12})
    outstr, _ = capsys.readouterr()
    assert 'WARNING :: Could not serialize <broken>' in outstr

    # Load it.
    out_npz = io.load(str(tmpdir+'/test.npz'), allow_pickle=True)
    outstr, _ = capsys.readouterr()
    assert 'Loaded file' in outstr
    assert 'test.npz' in outstr
    assert utils.__version__ in outstr

    assert out_npz['Model']['model'] == model
    assert_allclose(field.fx, out_npz['Field']['field'].fx)
    assert_allclose(grid.vol, out_npz['TensorMesh']['emg3d'].vol)
    assert_allclose(grid.vol, out_npz['TensorMesh']['discretize'].vol)
    assert_allclose(out_npz['Data']['what']['f'], field.fx)

    # Check message from loading another file

    data = io._dict_serialize({'meshes': grid})
    fdata = io._dict_flatten(data)
    del fdata['TensorMesh>meshes>hx']

    np.savez_compressed(tmpdir+'/test2.npz', **fdata)
    _ = io.load(str(tmpdir+'/test2.npz'), allow_pickle=True)
    outstr, _ = capsys.readouterr()
    assert "was not created by emg3d." in outstr
    assert "WARNING :: Could not de-serialize <meshes>" in outstr

    # Unknown keyword.
    with pytest.raises(TypeError):
        io.load('ttt.npz', stupidkeyword='a')

    # Unknown backend/extension.
    with pytest.raises(NotImplementedError):
        io.save(tmpdir+'/testwrongbackend', something=1, backend='what?')
    io.save(tmpdir+'/testwrongbackend.abc', something=1)
    with pytest.raises(NotImplementedError):
        io.load(tmpdir+'/testwrongbackend.abc')
    if h5py:
        io.load(tmpdir+'/testwrongbackend.abc.h5')
    else:
        io.load(tmpdir+'/testwrongbackend.abc.npz')

    # Ensure deprecated backend/extension still work.
    if h5py:
        io.save(tmpdir+'/ttt', backend='h5py')
    io.save(tmpdir+'/ttt', backend='numpy')

    # Test h5py.
    if h5py:
        io.save(tmpdir+'/test', emg3d=grid, discretize=grid2,
                a=1.0, b=1+1j,
                model=model, field=field, what={'f': field.fx})
        out_h5 = io.load(str(tmpdir+'/test.h5'))
        assert out_h5['Model']['model'] == model
        assert out_h5['Data']['a'] == 1.0
        assert out_h5['Data']['b'] == 1+1j
        assert_allclose(field.fx, out_h5['Field']['field'].fx)
        assert_allclose(grid.vol, out_h5['TensorMesh']['emg3d'].vol)
        assert_allclose(grid.vol, out_h5['TensorMesh']['discretize'].vol)
        assert_allclose(out_h5['Data']['what']['f'], field.fx)

        assert io._compare_dicts(out_h5, out_npz) is True
    else:
        with pytest.raises(ImportError):
            # Ensure deprecated backend/extension still work.
            io.save(tmpdir+'/ttt', backend='h5py')
        with pytest.raises(ImportError):
            io.save(tmpdir+'/test.h5', grid=grid)
        with pytest.raises(ImportError):
            io.load(str(tmpdir+'/test-h5.h5'))

    # Test json.
    io.save(tmpdir+'/test', emg3d=grid, discretize=grid2,
            a=1.0, b=1+1j,
            model=model, field=field, what={'f': field.fx}, backend='json')
    out_json = io.load(str(tmpdir+'/test.json'))
    assert out_json['Model']['model'] == model
    assert out_json['Data']['a'] == 1.0
    assert out_json['Data']['b'] == 1+1j
    assert_allclose(field.fx, out_json['Field']['field'].fx)
    assert_allclose(grid.vol, out_json['TensorMesh']['emg3d'].vol)
    assert_allclose(grid.vol, out_json['TensorMesh']['discretize'].vol)
    assert_allclose(out_json['Data']['what']['f'], field.fx)

    assert io._compare_dicts(out_json, out_npz) is True


def test_compare_dicts(capsys):
    # Create test data
    grid = utils.TensorMesh(
            [np.array([100, 4]), np.array([100, 8]), np.array([100, 16])],
            np.zeros(3))

    model = utils.Model(grid, res_x=1., res_y=2., res_z=3., mu_r=4.)

    e1 = create_dummy(*grid.vnEx)
    e2 = create_dummy(*grid.vnEy)
    e3 = create_dummy(*grid.vnEz)
    ee = utils.Field(e1, e2, e3, freq=.938)

    dict1 = io._dict_serialize(
            {'model': model, 'grid': grid, 'field': ee,
             'a': 1, 'b': None, 'c': 1e-9+1j*1e13,
             'd': {'aa': np.arange(10), 'bb': np.sqrt(1.0),
                   'cc': {'another': 1}, 'dd': None}
             })

    dict2 = dc(dict1)
    out = io._compare_dicts(dict1, dict2)
    assert out is True

    del dict1['Data']['d']['bb']
    del dict2['Field']
    del dict2['Model']['model']['mu_r']
    dict2['TensorMesh']['grid']['hy'] *= 2
    dict2['whatever'] = 'whatever'
    dict2['2onlydict'] = {'booh': 12}

    out = io._compare_dicts(dict1, dict2, True)
    assert out is False
    outstr, _ = capsys.readouterr()
    assert " True  :: Model      > model      > res_x" in outstr
    assert "  {1}  ::                           mu_r" in outstr
    assert " False ::                           hy" in outstr
    assert " True  ::                           cc         > another" in outstr
    assert "  {2}  :: Data       > d          > bb" in outstr
    assert "  {2}  :: 2onlydict" in outstr
