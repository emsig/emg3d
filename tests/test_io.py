import pytest
import numpy as np
from numpy.testing import assert_allclose

from emg3d import meshes, models, fields, misc, io

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
    io.save(tmpdir+'/test', emg3d=grid, discretize=grid2, model=model,
            broken=grid3, a=None,
            field=field, what={'f': field.fx, 12: 12}, backend="numpy")
    outstr, _ = capsys.readouterr()
    assert 'WARNING :: Could not serialize <broken>' in outstr

    # Load it.
    out = io.load(str(tmpdir+'/test.npz'), allow_pickle=True)
    outstr, _ = capsys.readouterr()
    assert 'Loaded file' in outstr
    assert 'test.npz' in outstr
    assert misc.__version__ in outstr

    assert out['Model']['model'] == model
    assert_allclose(field.fx, out['Field']['field'].fx)
    assert_allclose(grid.vol, out['TensorMesh']['emg3d'].vol)
    assert_allclose(grid.vol, out['TensorMesh']['discretize'].vol)
    assert_allclose(out['Data']['what']['f'], field.fx)

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
        io.save('ttt', backend='a')
    with pytest.raises(NotImplementedError):
        io.load('ttt.abc')

    # Test h5py.
    if h5py:
        io.save(tmpdir+'/test', emg3d=grid, discretize=grid2,
                model=model, field=field, what={'f': field.fx})
        out_h5 = io.load(str(tmpdir+'/test.h5'))
        assert out_h5['Model']['model'] == model
        assert_allclose(field.fx, out_h5['Field']['field'].fx)
        assert_allclose(grid.vol, out_h5['TensorMesh']['emg3d'].vol)
        assert_allclose(grid.vol, out_h5['TensorMesh']['discretize'].vol)
        assert_allclose(out_h5['Data']['what']['f'], field.fx)
    else:
        with pytest.raises(ImportError):
            io.save(tmpdir+'/test-h5', grid=grid)
        with pytest.raises(ImportError):
            io.load(str(tmpdir+'/test-h5.h5'))
