import pytest
import numpy as np
from numpy.testing import assert_allclose

from emg3d import fields, meshes, models


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


def test_Model(capsys):
    # Mainly regression tests

    # Create some dummy data
    grid = meshes.TensorMesh(
            [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
            np.zeros(3))

    res_x = create_dummy(*grid.vnC, False)
    res_y = res_x/2.0
    res_z = res_x*1.4
    mu_r = res_x*1.11

    # Check representation of TensorMesh.
    assert 'TensorMesh: 2 x 2 x 2 (8)' in grid.__repr__()

    _, _ = capsys.readouterr()  # Clean-up
    # Using defaults; check backwards compatibility for freq.
    sfield = fields.SourceField(grid, freq=1)
    model1 = models.Model(grid)

    # Check representation of Model.
    assert 'Model; isotropic resistivities' in model1.__repr__()

    model1e = models.Model(grid)
    model1.mu_r = None
    model1.epsilon_r = None
    vmodel1 = models.VolumeModel(grid, model1, sfield)
    assert_allclose(model1.res_x, model1.res_y)
    assert_allclose(model1.nC, grid.nC)
    assert_allclose(model1.vnC, grid.vnC)
    assert_allclose(vmodel1.eta_z, vmodel1.eta_y)
    assert model1.mu_r is None
    assert model1.epsilon_r is None

    # Assert that the cases changes if we assign y- and z- resistivities (I).
    model1.res_x = model1.res_x
    assert model1.case == 0
    model1.res_y = 2*model1.res_x
    assert model1.case == 1
    model1.res_z = 2*model1.res_x
    assert model1.case == 3

    assert model1e.case == 0
    model1e.res_z = 2*model1.res_x
    assert model1e.case == 2
    model1e.res_y = 2*model1.res_x
    assert model1e.case == 3

    model1.epsilon_r = 1  # Just set to 1 CURRENTLY.
    vmodel1b = models.VolumeModel(grid, model1, sfield)
    assert_allclose(vmodel1b.eta_x, vmodel1.eta_x)
    assert np.iscomplex(vmodel1b.eta_x[0, 0, 0])

    # Using ints
    model2 = models.Model(grid, 2., 3., 4.)
    vmodel2 = models.VolumeModel(grid, model2, sfield)
    assert_allclose(model2.res_x*1.5, model2.res_y)
    assert_allclose(model2.res_x*2, model2.res_z)

    # VTI: Setting res_x and res_z, not res_y
    model2b = models.Model(grid, 2., res_z=4.)
    vmodel2b = models.VolumeModel(grid, model2b, sfield)
    assert_allclose(model2b.res_x, model2b.res_y)
    assert_allclose(vmodel2b.eta_y, vmodel2b.eta_x)
    model2b.res_z = model2b.res_x
    model2c = models.Model(grid, 2., res_z=model2b.res_z.copy())
    vmodel2c = models.VolumeModel(grid, model2c, sfield)
    assert_allclose(model2c.res_x, model2c.res_z)
    assert_allclose(vmodel2c.eta_z, vmodel2c.eta_x)

    # HTI: Setting res_x and res_y, not res_z
    model2d = models.Model(grid, 2., 4.)
    vmodel2d = models.VolumeModel(grid, model2d, sfield)
    assert_allclose(model2d.res_x, model2d.res_z)
    assert_allclose(vmodel2d.eta_z, vmodel2d.eta_x)

    # Pure air, epsilon_r init with 0 => should be 1!
    model6 = models.Model(grid, 2e14)
    assert model6.epsilon_r is None

    # Check wrong shape
    with pytest.raises(ValueError):
        models.Model(grid, np.arange(1, 11))
    with pytest.raises(ValueError):
        models.Model(grid, res_y=np.ones((2, 5, 6)))
    with pytest.raises(ValueError):
        models.Model(grid, res_z=np.array([1, 3]))
    with pytest.raises(ValueError):
        models.Model(grid, mu_r=np.array([[1, ], [3, ]]))

    # Check with all inputs
    gridvol = grid.vol.reshape(grid.vnC, order='F')
    model3 = models.Model(grid, res_x, res_y, res_z, mu_r=mu_r)
    vmodel3 = models.VolumeModel(grid, model3, sfield)
    assert_allclose(model3.res_x, model3.res_y*2)
    assert_allclose(model3.res_x.shape, grid.vnC)
    assert_allclose(model3.res_x, model3.res_z/1.4)
    assert_allclose(gridvol/mu_r, vmodel3.zeta)
    # Check with all inputs
    model3b = models.Model(grid, res_x.ravel('F'), res_y.ravel('F'),
                           res_z.ravel('F'), mu_r=mu_r.ravel('F'))
    vmodel3b = models.VolumeModel(grid, model3b, sfield)
    assert_allclose(model3b.res_x, model3b.res_y*2)
    assert_allclose(model3b.res_x.shape, grid.vnC)
    assert_allclose(model3b.res_x, model3b.res_z/1.4)
    assert_allclose(gridvol/mu_r, vmodel3b.zeta)

    # Check setters vnC
    tres = np.ones(grid.vnC)
    model3.res_x[:, :, :] = tres*2.0
    model3.res_y[:, :1, :] = tres[:, :1, :]*3.0
    model3.res_y[:, 1:, :] = tres[:, 1:, :]*3.0
    model3.res_z = tres*4.0
    model3.mu_r = tres*5.0
    model3.epsilon_r = tres*6.0
    assert_allclose(tres*2., model3.res_x)
    assert_allclose(tres*3., model3.res_y)
    assert_allclose(tres*4., model3.res_z)
    assert_allclose(tres*6., model3.epsilon_r)

    # Check eta
    iomep = sfield.sval*models.epsilon_0
    eta_x = sfield.smu0*(1./model3.res_x + iomep)*gridvol
    eta_y = sfield.smu0*(1./model3.res_y + iomep)*gridvol
    eta_z = sfield.smu0*(1./model3.res_z + iomep)*gridvol
    vmodel3 = models.VolumeModel(grid, model3, sfield)
    assert_allclose(vmodel3.eta_x, eta_x)
    assert_allclose(vmodel3.eta_y, eta_y)
    assert_allclose(vmodel3.eta_z, eta_z)

    # Check volume
    assert_allclose(grid.vol.reshape(grid.vnC, order='F'), vmodel2.zeta)
    model4 = models.Model(grid, 1)
    vmodel4 = models.VolumeModel(grid, model4, sfield)
    assert_allclose(vmodel4.zeta, grid.vol.reshape(grid.vnC, order='F'))

    # Check a couple of out-of-range failures
    with pytest.raises(ValueError):
        _ = models.Model(grid, res_x=res_x*0)
    Model = models.Model(grid, res_x=res_x)
    with pytest.raises(ValueError):
        Model._check_parameter(res_x*0, 'res_x')
    with pytest.raises(ValueError):
        Model._check_parameter(-1.0, 'res_x')
    with pytest.raises(ValueError):
        _ = models.Model(grid, res_y=np.inf)
    with pytest.raises(ValueError):
        _ = models.Model(grid, res_z=res_z*np.inf)
    with pytest.raises(ValueError):
        _ = models.Model(grid, mu_r=-1)


class TestModelOperators:

    # Define two different sized meshes.
    mesh_base = meshes.TensorMesh(
            [np.ones(3), np.ones(4), np.ones(5)], x0=np.array([0, 0, 0]))
    mesh_diff = meshes.TensorMesh(
            [np.ones(3), np.ones(4), np.ones(6)], x0=np.array([0, 0, 0]))

    # Define a couple of models.
    model_int = models.Model(mesh_base, 1.)
    model_1_a = models.Model(mesh_base, 1., 2.)
    model_1_b = models.Model(mesh_base, 2., 4.)
    model_2_a = models.Model(mesh_base, 1., res_z=3.)
    model_2_b = models.Model(mesh_base, 2., res_z=6.)
    model_3_a = models.Model(mesh_base, 1., 2., 3.)
    model_3_b = models.Model(mesh_base, 2., 4., 6.)
    model_mu_a = models.Model(mesh_base, 1., mu_r=1.)
    model_mu_b = models.Model(mesh_base, 2., mu_r=2.)
    model_epsilon_a = models.Model(mesh_base, 1., epsilon_r=1.)
    model_epsilon_b = models.Model(mesh_base, 2., epsilon_r=2.)

    model_int_diff = models.Model(mesh_diff, 1.)
    model_nC = models.Model(mesh_base, np.ones(mesh_base.nC)*2.)
    model_vnC = models.Model(mesh_base, np.ones(mesh_base.vnC))

    def test_operator_test(self):
        with pytest.raises(ValueError):
            self.model_int + self.model_1_a
        with pytest.raises(ValueError):
            self.model_int - self.model_2_a
        with pytest.raises(ValueError):
            self.model_int + self.model_3_a
        with pytest.raises(ValueError):
            self.model_int - self.model_mu_a
        with pytest.raises(ValueError):
            self.model_int + self.model_epsilon_a
        with pytest.raises(ValueError):
            self.model_int - self.model_int_diff

    def test_add(self):
        a = self.model_int + self.model_nC
        b = self.model_nC + self.model_vnC
        c = self.model_int + self.model_vnC
        assert a == b
        assert a != c
        assert a.res_x[0, 0, 0] == 3.0
        assert b.res_x[0, 0, 0] == 3.0
        assert c.res_x[0, 0, 0] == 2.0

        # Addition with something else than a model
        with pytest.raises(TypeError):
            self.model_int + self.mesh_base

        # All different cases
        a = self.model_1_a + self.model_1_a
        assert a.res_x == self.model_1_b.res_x
        assert a.res_y == self.model_1_b.res_y
        assert a.res_z.base is a.res_x.base

        a = self.model_2_a + self.model_2_a
        assert a.res_x == self.model_2_b.res_x
        assert a.res_y.base is a.res_x.base
        assert a.res_z == self.model_2_b.res_z

        a = self.model_3_a + self.model_3_a
        assert a.res_x == self.model_3_b.res_x
        assert a.res_y == self.model_3_b.res_y
        assert a.res_z == self.model_3_b.res_z

        # mu_r and epsilon_r
        a = self.model_mu_a + self.model_mu_a
        b = self.model_epsilon_a + self.model_epsilon_a
        assert a.mu_r == self.model_mu_b.mu_r
        assert b.epsilon_r == self.model_epsilon_b.epsilon_r

    def test_sub(self):
        # Addition with something else than a model
        with pytest.raises(TypeError):
            self.model_int - self.mesh_base

        # All different cases
        a = self.model_1_b - self.model_1_a
        assert a.res_x == self.model_1_a.res_x
        assert a.res_y == self.model_1_a.res_y
        assert a.res_z.base is a.res_x.base

        a = self.model_2_b - self.model_2_a
        assert a.res_x == self.model_2_a.res_x
        assert a.res_y.base is a.res_x.base
        assert a.res_z == self.model_2_a.res_z

        a = self.model_3_b - self.model_3_a
        assert a.res_x == self.model_3_a.res_x
        assert a.res_y == self.model_3_a.res_y
        assert a.res_z == self.model_3_a.res_z

        # mu_r and epsilon_r
        a = self.model_mu_b - self.model_mu_a
        b = self.model_epsilon_b - self.model_epsilon_a
        assert a.mu_r == self.model_mu_a.mu_r
        assert b.epsilon_r == self.model_epsilon_a.epsilon_r

    def test_eq(self):

        assert (self.model_int == self.mesh_base) is False

        out = self.model_int == self.model_int_diff
        assert not out

        out = self.model_int == self.model_int
        assert out

    def test_general(self):

        # Check shape and size
        assert_allclose(self.model_int.shape, self.mesh_base.vnC)
        assert self.model_int.size == self.mesh_base.nC

    def test_copy(self):
        model_new1 = self.model_vnC.copy()
        model_new2 = self.model_3_a.copy()
        model_new3 = self.model_mu_a.copy()
        model_new4 = self.model_epsilon_a.copy()

        assert model_new1 == self.model_vnC
        assert model_new2 == self.model_3_a
        assert model_new3 == self.model_mu_a
        assert model_new4 == self.model_epsilon_a

        assert model_new1.res_x.base is not self.model_vnC.res_x.base
        assert model_new2.res_y.base is not self.model_3_a.res_y.base
        assert model_new2.res_z.base is not self.model_3_a.res_z.base
        assert model_new3.mu_r.base is not self.model_mu_a.mu_r.base
        assert (model_new4.epsilon_r.base is not
                self.model_epsilon_a.epsilon_r.base)

    def test_dict(self):
        # dict is already tested via copy. Just the other cases here.
        mdict = self.model_3_b.to_dict()
        keys = ['res_x', 'res_y', 'res_z', 'mu_r', 'epsilon_r', 'vnC']
        for key in keys:
            assert key in mdict.keys()
        for key in keys[:3]:
            val = getattr(self.model_3_b, key)
            assert_allclose(mdict[key], val)

        del mdict['res_x']
        with pytest.raises(KeyError):
            models.Model.from_dict(mdict)