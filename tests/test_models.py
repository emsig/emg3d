import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import models

from . import helpers


# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None


class TestModel:

    def test_regression(self, capsys):
        # Mainly regression tests

        # Create some dummy data
        grid = emg3d.TensorMesh(
                [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
                np.zeros(3))

        property_x = helpers.dummy_field(*grid.shape_cells, False)
        property_y = property_x/2.0
        property_z = property_x*1.4
        mu_r = property_x*1.11

        _, _ = capsys.readouterr()  # Clean-up
        # Using defaults; check backwards compatibility for freq.
        sfield = emg3d.Field(grid, frequency=1)
        model1 = models.Model(grid)

        # Check representation of Model.
        assert 'Model: resistivity; isotropic' in model1.__repr__()

        vmodel1 = models.VolumeModel(model1, sfield)
        assert_allclose(model1.size, grid.n_cells)
        assert_allclose(model1.shape, grid.shape_cells)
        assert_allclose(vmodel1.eta_z, vmodel1.eta_y)
        assert model1.property_y is None
        assert model1.property_z is None
        assert model1.mu_r is None
        assert model1.epsilon_r is None

        # Using ints
        model2 = models.Model(grid, 2., 3., 4.)
        vmodel2 = models.VolumeModel(model2, sfield)
        assert_allclose(model2.property_x*1.5, model2.property_y)
        assert_allclose(model2.property_x*2, model2.property_z)

        # VTI: Setting property_x and property_z, not property_y
        model2b = models.Model(grid, 2., property_z=4., epsilon_r=1)
        vmodel2b = models.VolumeModel(model2b, sfield)
        assert model1.property_y is None
        assert_allclose(vmodel2b.eta_y, vmodel2b.eta_x)
        model2b.property_z = model2b.property_x
        model2c = models.Model(grid, 2., property_z=model2b.property_z.copy())
        vmodel2c = models.VolumeModel(model2c, sfield)
        assert_allclose(model2c.property_x, model2c.property_z)
        assert_allclose(vmodel2c.eta_z, vmodel2c.eta_x)

        # HTI: Setting property_x and property_y, not property_z
        model2d = models.Model(grid, 2., 4.)
        vmodel2d = models.VolumeModel(model2d, sfield)
        assert model1.property_z is None
        assert_allclose(vmodel2d.eta_z, vmodel2d.eta_x)

        # Pure air, epsilon_r init with 0 => should be 1!
        model6 = models.Model(grid, 2e14)
        assert model6.epsilon_r is None

        # Check wrong shape
        with pytest.raises(ValueError, match='could not be broadcast'):
            models.Model(grid, np.arange(1, 11))
        with pytest.raises(ValueError, match='could not be broadcast'):
            models.Model(grid, property_y=np.ones((2, 5, 6)))
        # Actual broadcasting.
        models.Model(grid, property_z=np.array([1, 3]))
        models.Model(grid, mu_r=np.array([[1, ], [3, ]]))

        # Check with all inputs
        gridvol = grid.cell_volumes.reshape(grid.shape_cells, order='F')
        model3 = models.Model(
            grid, property_x, property_y, property_z, mu_r=mu_r)
        vmodel3 = models.VolumeModel(model3, sfield)
        assert_allclose(model3.property_x, model3.property_y*2)
        assert_allclose(model3.property_x.shape, grid.shape_cells)
        assert_allclose(model3.property_x, model3.property_z/1.4)
        assert_allclose(gridvol/mu_r, vmodel3.zeta)
        # Check with all inputs
        model3b = models.Model(
            grid, property_x.ravel('F'), property_y.ravel('F'),
            property_z.ravel('F'), mu_r=mu_r.ravel('F'))
        vmodel3b = models.VolumeModel(model3b, sfield)
        assert_allclose(model3b.property_x, model3b.property_y*2)
        assert_allclose(model3b.property_x.shape, grid.shape_cells)
        assert_allclose(model3b.property_x, model3b.property_z/1.4)
        assert_allclose(gridvol/mu_r, vmodel3b.zeta)

        # Check setters shape_cells
        tres = np.ones(grid.shape_cells)
        model3.property_x[:, :, :] = tres*2.0
        model3.property_y[:, :1, :] = tres[:, :1, :]*3.0
        model3.property_y[:, 1:, :] = tres[:, 1:, :]*3.0
        model3.property_z = tres*4.0
        model3.mu_r = tres*5.0
        assert_allclose(tres*2., model3.property_x)
        assert_allclose(tres*3., model3.property_y)
        assert_allclose(tres*4., model3.property_z)

        # Check eta
        iomep = sfield.sval*models.epsilon_0
        eta_x = -sfield.smu0*(1./model3.property_x - iomep)*gridvol
        eta_y = -sfield.smu0*(1./model3.property_y - iomep)*gridvol
        eta_z = -sfield.smu0*(1./model3.property_z - iomep)*gridvol
        vmodel3 = models.VolumeModel(model3, sfield)
        assert_allclose(vmodel3.eta_x, eta_x)
        assert_allclose(vmodel3.eta_y, eta_y)
        assert_allclose(vmodel3.eta_z, eta_z)

        # Check volume
        assert_allclose(grid.cell_volumes.reshape(grid.shape_cells, order='F'),
                        vmodel2.zeta)
        model4 = models.Model(grid, 1)
        vmodel4 = models.VolumeModel(model4, sfield)
        assert_allclose(vmodel4.zeta,
                        grid.cell_volumes.reshape(grid.shape_cells, order='F'))

        # Check a couple of out-of-range failures
        with pytest.raises(ValueError, match='`property_x` must be all'):
            with np.errstate(all='ignore'):
                _ = models.Model(grid, property_x=property_x*0)
        Model = models.Model(grid, property_x=property_x)
        with pytest.raises(ValueError, match='`property_x` must be all'):
            with np.errstate(all='ignore'):
                Model._check_positive_finite(property_x*0, 'property_x')
        with pytest.raises(ValueError, match='`property_x` must be all'):
            Model._check_positive_finite(-1.0, 'property_x')
        with pytest.raises(ValueError, match='`property_y` must be all'):
            _ = models.Model(grid, property_y=np.inf)
        with pytest.raises(ValueError, match='`property_z` must be all'):
            _ = models.Model(grid, property_z=property_z*np.inf)
        with pytest.raises(ValueError, match='`mu_r` must be all'):
            _ = models.Model(grid, mu_r=-1)

    def test_interpolate(self):

        # Create some dummy data
        grid = emg3d.TensorMesh(
                [np.array([2, 2]), np.array([4, 4]), np.array([5, 5])],
                np.zeros(3))

        grid2 = emg3d.TensorMesh(
                [np.array([2]), np.array([4]), np.array([5])],
                np.array([1, 2, 2.5]))

        property_x = helpers.dummy_field(*grid.shape_cells, False)
        property_y = property_x/2.0
        property_z = property_x*1.4
        mu_r = property_x*1.11
        epsilon_r = property_x*3.33

        model1inp = models.Model(grid, property_x)
        same = model1inp.interpolate_to_grid(model1inp.grid)
        assert model1inp == same

        model1out = model1inp.interpolate_to_grid(grid2)
        assert_allclose(model1out.property_x[0],
                        10**(np.sum(np.log10(model1inp.property_x))/8))
        assert model1out.property_y is None
        assert model1out.property_z is None
        assert model1out.epsilon_r is None
        assert model1out.mu_r is None

        model2inp = models.Model(
                grid, property_x=property_x, property_y=property_y,
                property_z=property_z, mu_r=mu_r, epsilon_r=epsilon_r)

        model2out = model2inp.interpolate_to_grid(grid2)
        assert_allclose(model2out.property_x[0],
                        10**(np.sum(np.log10(model2inp.property_x))/8))
        assert_allclose(model2out.property_y[0],
                        10**(np.sum(np.log10(model2inp.property_y))/8))
        assert_allclose(model2out.property_z[0],
                        10**(np.sum(np.log10(model2inp.property_z))/8))
        assert_allclose(model2out.epsilon_r,
                        10**(np.sum(np.log10(model2inp.epsilon_r))/8))
        assert_allclose(model2out.mu_r,
                        10**(np.sum(np.log10(model2inp.mu_r))/8))

    def test_equal_mapping(self):

        # Create some dummy data
        grid = emg3d.TensorMesh(
                [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
                np.zeros(3))

        model1 = models.Model(grid)
        mapping = emg3d.maps.MapConductivity()
        model2 = models.Model(grid, mapping=mapping)

        check = model1 == model2

        assert check is False

    def test_negative_values(self):
        # Create some dummy data
        grid = emg3d.TensorMesh(
                [np.array([2, 2]), np.array([3, 4]), np.array([0.5, 2])],
                np.zeros(3))

        # Check these fails.
        with pytest.raises(ValueError, match='`property_x` must be '):
            models.Model(grid, property_x=-1, mapping='Conductivity')
        with pytest.raises(ValueError, match='`property_x` must be '):
            models.Model(grid, property_x=-1, mapping='Resistivity')

        # Check these do not fail.
        models.Model(grid, property_x=-1, mapping='LgConductivity')
        models.Model(grid, property_x=-1, mapping='LnConductivity')
        models.Model(grid, property_x=-1, mapping='LgResistivity')
        models.Model(grid, property_x=-1, mapping='LnResistivity')

    def test_broadcasting(self):
        grid = emg3d.TensorMesh(
                [np.ones(4), np.ones(2), np.ones(6)], (0, 0, 0))

        model = models.Model(grid, 1, 1, 1, 1, 1)
        assert_allclose(model.property_x, model.property_y)
        assert_allclose(model.property_x, model.property_z)
        assert_allclose(model.property_x, model.mu_r)
        assert_allclose(model.property_x, model.epsilon_r)

        model.property_x = [[[1.]], [[2.]], [[3.]], [[4.]]]
        assert_allclose(model.property_x[:, 0, 0], [1, 2, 3, 4])

        model.property_y = [[[1, ], [2, ]]]
        assert_allclose(model.property_y[0, :, 0], [1, 2])

        model.property_z = [[[1, 2., 3., 4., 5., 6.]]]
        assert_allclose(model.property_z[0, 0, :], [1, 2, 3, 4, 5, 6])

        model.mu_r = 3.33
        assert_allclose(model.mu_r, 3.33)

        data = np.arange(1, 4*2*6+1).reshape((4, 2, 6), order='F')
        model.epsilon_r = data
        assert_allclose(model.epsilon_r, data)

    def test_change_anisotropy(self):
        hx = [2, 2]
        grid = emg3d.TensorMesh([hx, hx, hx], (0, 0, 0))
        model = models.Model(grid, 1)

        with pytest.raises(ValueError, match='initiated without `property_y`'):
            model.property_y = 3

        with pytest.raises(ValueError, match='initiated without `property_z`'):
            model.property_z = 3

        with pytest.raises(ValueError, match='l was initiated without `mu_r`'):
            model.mu_r = 3

        with pytest.raises(ValueError, match=' initiated without `epsilon_r`'):
            model.epsilon_r = 3


class TestModelOperators:

    # Define two different sized meshes.
    mesh_base = emg3d.TensorMesh(
            [np.ones(3), np.ones(4), np.ones(5)], origin=np.array([0, 0, 0]))
    mesh_diff = emg3d.TensorMesh(
            [np.ones(3), np.ones(4), np.ones(6)], origin=np.array([0, 0, 0]))

    # Define a couple of models.
    mod_iso = models.Model(mesh_base, 1.)
    mod_map = models.Model(mesh_base, 1., mapping='Conductivity')
    mod_hti_a = models.Model(mesh_base, 1., 2.)
    mod_hti_b = models.Model(mesh_base, 2., 4.)
    mod_vti_a = models.Model(mesh_base, 1., property_z=3.)
    mod_vti_b = models.Model(mesh_base, 2., property_z=6.)
    mod_tri_a = models.Model(mesh_base, 1., 2., 3.)
    mod_tri_b = models.Model(mesh_base, 2., 4., 6.)
    mod_mu_a = models.Model(mesh_base, 1., mu_r=1.)
    mod_mu_b = models.Model(mesh_base, 2., mu_r=2.)
    mod_epsilon_a = models.Model(mesh_base, 1., epsilon_r=1.)
    mod_epsilon_b = models.Model(mesh_base, 2., epsilon_r=2.)

    mod_int_diff = models.Model(mesh_diff, 1.)
    mod_size = models.Model(mesh_base, np.ones(mesh_base.n_cells)*2.)
    mod_shape = models.Model(mesh_base, np.ones(mesh_base.shape_cells))

    def test_operator_test(self):
        with pytest.raises(ValueError, match='Models have different anisotro'):
            self.mod_iso + self.mod_hti_a
        with pytest.raises(ValueError, match='Models have different anisotro'):
            self.mod_iso - self.mod_vti_a
        with pytest.raises(ValueError, match='Models have different anisotro'):
            self.mod_iso + self.mod_tri_a
        with pytest.raises(ValueError, match='One model has mu_r, the other '):
            self.mod_iso - self.mod_mu_a
        with pytest.raises(ValueError, match='One model has epsilon_r, the o'):
            self.mod_iso + self.mod_epsilon_a
        with pytest.raises(ValueError, match='Models have different grids.'):
            self.mod_iso - self.mod_int_diff
        with pytest.raises(ValueError, match='Models have different mappings'):
            self.mod_iso - self.mod_map

    def test_add(self):
        a = self.mod_iso + self.mod_size
        b = self.mod_size + self.mod_shape
        c = self.mod_iso + self.mod_shape
        assert a == b
        assert a != c
        assert a.property_x[0, 0, 0] == 3.0
        assert b.property_x[0, 0, 0] == 3.0
        assert c.property_x[0, 0, 0] == 2.0

        # Addition with something else than a model
        with pytest.raises(TypeError):
            self.mod_iso + self.mesh_base

        # All different cases
        a = self.mod_hti_a + self.mod_hti_a
        assert_allclose(a.property_x, self.mod_hti_b.property_x)
        assert_allclose(a.property_y, self.mod_hti_b.property_y)

        a = self.mod_vti_a + self.mod_vti_a
        assert_allclose(a.property_x, self.mod_vti_b.property_x)
        assert_allclose(a.property_z, self.mod_vti_b.property_z)

        a = self.mod_tri_a + self.mod_tri_a
        assert_allclose(a.property_x, self.mod_tri_b.property_x)
        assert_allclose(a.property_y, self.mod_tri_b.property_y)
        assert_allclose(a.property_z, self.mod_tri_b.property_z)

        # mu_r and epsilon_r
        a = self.mod_mu_a + self.mod_mu_a
        b = self.mod_epsilon_a + self.mod_epsilon_a
        assert_allclose(a.mu_r, self.mod_mu_b.mu_r)
        assert_allclose(b.epsilon_r, self.mod_epsilon_b.epsilon_r)

    def test_sub(self):
        # Addition with something else than a model
        with pytest.raises(TypeError):
            self.mod_iso - self.mesh_base

        # All different cases
        a = self.mod_hti_b - self.mod_hti_a
        assert_allclose(a.property_x, self.mod_hti_a.property_x)
        assert_allclose(a.property_y, self.mod_hti_a.property_y)

        a = self.mod_vti_b - self.mod_vti_a
        assert_allclose(a.property_x, self.mod_vti_a.property_x)
        assert_allclose(a.property_z, self.mod_vti_a.property_z)

        a = self.mod_tri_b - self.mod_tri_a
        assert_allclose(a.property_x, self.mod_tri_a.property_x)
        assert_allclose(a.property_y, self.mod_tri_a.property_y)
        assert_allclose(a.property_z, self.mod_tri_a.property_z)

        # mu_r and epsilon_r
        a = self.mod_mu_b - self.mod_mu_a
        b = self.mod_epsilon_b - self.mod_epsilon_a
        assert_allclose(a.mu_r, self.mod_mu_a.mu_r)
        assert_allclose(b.epsilon_r, self.mod_epsilon_a.epsilon_r)

    def test_eq(self):

        assert (self.mod_iso == self.mesh_base) is False

        out = self.mod_iso == self.mod_int_diff
        assert not out

        out = self.mod_iso == self.mod_iso
        assert out

    def test_general(self):

        # Check shape and size
        assert_allclose(self.mod_iso.shape, self.mesh_base.shape_cells)
        assert self.mod_iso.size == self.mesh_base.n_cells

    def test_copy(self):
        model_new1 = self.mod_shape.copy()
        model_new2 = self.mod_tri_a.copy()
        model_new3 = self.mod_mu_a.copy()
        model_new4 = self.mod_epsilon_a.copy()

        assert model_new1 == self.mod_shape
        assert model_new2 == self.mod_tri_a
        assert model_new3 == self.mod_mu_a
        assert model_new4 == self.mod_epsilon_a

        assert not np.may_share_memory(model_new1.property_x,
                                       self.mod_shape.property_x)
        assert not np.may_share_memory(model_new2.property_y,
                                       self.mod_tri_a.property_y)
        assert not np.may_share_memory(model_new2.property_z,
                                       self.mod_tri_a.property_z)
        assert not np.may_share_memory(model_new3.mu_r, self.mod_mu_a.mu_r)
        assert not np.may_share_memory(model_new4.epsilon_r,
                                       self.mod_epsilon_a.epsilon_r)

    def test_dict(self):
        # dict is already tested via copy. Just the other cases here.
        mdict = self.mod_tri_b.to_dict()
        keys = ['property_x', 'property_y', 'property_z', 'mu_r', 'epsilon_r',
                'grid']
        for key in keys:
            assert key in mdict.keys()
        for key in keys[:3]:
            val = getattr(self.mod_tri_b, key)
            assert_allclose(mdict[key], val)

        del mdict['grid']
        with pytest.raises(KeyError, match="'grid'"):
            models.Model.from_dict(mdict)


class TestExtract1D:

    grid = emg3d.TensorMesh(
            [[1, 2, 3, 4], [3, 2, 2, 3, 4], [1, 1, 1, 1, 1]], (0, 0, 0))
    num = np.arange(1, 21).reshape((4, 5), order='F')
    property_x = np.zeros((4, 5, 5), order='F')
    property_x[:, :, 0] = num+100
    property_x[:, :, 1] = num+100
    property_x[:, :, 2] = num
    property_x[:, :, 3] = num+200
    property_x[:, :, 4] = num+200
    property_y = property_x/2.0
    property_z = property_x*1.4
    mu_r = property_x*1.11
    epsilon_r = property_x*2.22

    mlin = models.Model(
        grid, property_x, property_y, property_z, mu_r, epsilon_r
    )
    mlog = models.Model(
        grid, np.log10(property_x), np.log10(property_y), np.log10(property_z),
        np.log10(mu_r), np.log10(epsilon_r), mapping='LgResistivity'
    )

    def test_errors(self):
        # Unknown method
        with pytest.raises(ValueError, match="Unknown method 'unknown'"):
            self.mlin.extract_1d(method='unknown', p0=[0, 0])

        # Missing ellipse
        with pytest.raises(TypeError, match="requires the dict 'ellipse'"):
            self.mlin.extract_1d(method='prism', p0=[0, 0])

        # Ellipse but missing radius
        with pytest.raises(TypeError, match="the parameter 'radius'"):
            self.mlin.extract_1d(
                method='cylinder', p0=[0, 0], ellipse={'factor': 1.2}
            )

    def test_midpoint(self):

        # One point, outside
        layered, imat = self.mlin.extract_1d(
                'midpoint', [-3, -3], return_imat=True)
        assert_allclose(layered.property_x[0, 0, :],
                        [101.0, 101.0, 1.0, 201.0, 201.0])
        assert_allclose(layered.property_y, layered.property_x/2.0)
        assert_allclose(layered.property_z, layered.property_x*1.4)
        assert_allclose(layered.mu_r, layered.property_x*1.11)
        assert_allclose(layered.epsilon_r, layered.property_x*2.22)
        assert imat[0, 0] == 1.0
        assert imat.sum() == 1.0

        # Two points, merge
        layered = self.mlog.extract_1d('midpoint', [2, 4], [7, 8], merge=True)
        assert_allclose(10**layered.property_x[0, 0, :], [111.0, 11.0, 211.0])

    def test_averaging(self):

        # Empty selection => same as midpoint
        mpt = self.mlin.extract_1d('midpoint', [2.1, 7.1])
        cyl = self.mlin.extract_1d(
                'cylinder', [2.1, 7.1], ellipse={'radius': 0.001})
        assert mpt == cyl

        cylin, cmat = self.mlin.extract_1d(
                'cylinder', [2, 4], [7, 8], ellipse={'radius': 1.5},
                return_imat=True)
        cylog, _ = self.mlog.extract_1d(
                'cylinder', [2, 4], [7, 8], ellipse={'radius': 1.5},
                return_imat=True)

        vals = [110.4, 110.4, 9.4, 210.5, 210.5]
        assert_allclose(cylin.property_x[0, 0, :], vals, atol=0.1)
        assert_allclose(10**cylog.property_x[0, 0, :], vals, atol=0.1)

        cylin, pmat = self.mlin.extract_1d(
                'prism', [2, 4], [7, 8], ellipse={'radius': 1.5},
                return_imat=True)

        assert_allclose(pmat[:, :-1] > 0, 1)
        assert_allclose(pmat[:, -1], 0.0)


def test_expand_grid_model():
    grid = emg3d.TensorMesh([[4, 2, 2, 4], [2, 2, 2, 2], [1, 1]], (0, 0, 0))
    model = emg3d.Model(grid, 1, np.ones(grid.shape_cells)*2, mu_r=3,
                        epsilon_r=5)

    o_model = models.expand_grid_model(model, [2, 3], 5)

    # Grid.
    assert_allclose(grid.nodes_z, o_model.grid.nodes_z[:-2])
    assert o_model.grid.nodes_z[-2] == 5
    assert o_model.grid.nodes_z[-1] == 105

    # Property x (from float).
    assert_allclose(o_model.property_x[:, :, :-2], 1)
    assert_allclose(o_model.property_x[:, :, -2], 2)
    assert_allclose(o_model.property_x[:, :, -1], 3)

    # Property y (from shape_cells).
    assert_allclose(o_model.property_y[:, :, :-2], model.property_y)
    assert_allclose(o_model.property_y[:, :, -2], 2)
    assert_allclose(o_model.property_y[:, :, -1], 3)

    # Property z.
    assert o_model.property_z is None

    # Property mu_r (from float).
    assert_allclose(o_model.mu_r[:, :, :-2], 3)
    assert_allclose(o_model.mu_r[:, :, -2], 1)
    assert_allclose(o_model.mu_r[:, :, -1], 1)

    # Property epsilon_r (from float).
    assert_allclose(o_model.epsilon_r[:, :, :-2], 5)
    assert_allclose(o_model.epsilon_r[:, :, -2], 1)
    assert_allclose(o_model.epsilon_r[:, :, -1], 1)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
def test_estimate_layered_opts():

    # Regular survey.
    survey = emg3d.Survey(
        sources=emg3d.TxElectricDipole((0, 0, 0, 0, 0)),
        receivers=emg3d.RxElectricPoint((1000, 500, 100, 0, 0)),
        frequencies=[1.0, 2.0, 3.0],
    )

    # If method not in ['prism', 'cylinder', lopts_in == lopts_out
    lopts = {'method': 'aoeuaoeu', 'whatever_else': 'yes'}
    out = emg3d.models._estimate_layered_opts(lopts, survey, None)
    assert out['method'] == 'aoeuaoeu'
    assert out['whatever_else'] == 'yes'

    # empty -> 'midpoint'
    lopts = {'method': 'prism'}
    out = emg3d.models._estimate_layered_opts(lopts, survey, None)
    assert out['method'] == 'midpoint'

    # prism; radius; no gopts
    lopts = {'method': 'prism', 'ellipse': {'radius': 1000.0}}
    out = emg3d.models._estimate_layered_opts(lopts, survey, {'a': 1})
    assert out['method'] == 'prism'
    assert out['ellipse']['radius'] == 1000.0

    # Check defaults for cylinder.
    gopts = {'mapping': 'Resistivity', 'properties':  [0.3, 1.0]}
    lopts = {'ellipse': {'check_foci': True, 'merge': True}}
    out = emg3d.models._estimate_layered_opts(lopts, survey, gopts)
    assert out['method'] == 'cylinder'
    assert round(out['ellipse']['radius']) == 503  # Skindepth 1 Hz; 1 Ohm.m
    assert out['ellipse']['factor'] == 1.2
    assert out['ellipse']['minor'] == 0.8
    assert out['ellipse']['merge']

    # Ensure it doesn't change.
    gopts = {'mapping': 'Conductivity', 'properties':  [1/0.3, 1.0, 1.0, 1e-8]}
    lopts = {'ellipse': {'factor': 2.0, 'minor': 0.5}}
    out = emg3d.models._estimate_layered_opts(lopts, survey, gopts)
    assert out['method'] == 'cylinder'
    assert round(out['ellipse']['radius']) == 503  # Skindepth 1 Hz; 1 Ohm.m
    assert out['ellipse']['factor'] == 2.0
    assert out['ellipse']['minor'] == 0.5


def test_all_dir():
    assert set(models.__all__) == set(dir(models))
