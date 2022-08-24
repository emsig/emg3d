import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import electrodes


# Import soft dependencies.
try:
    import discretize
except ImportError:
    discretize = None


# Random number generator.
rng = np.random.default_rng()


class TestWire:

    class Magnetic(electrodes.Dipole):
        def __init__(self, coordinates, *args, **kwargs):
            self._coordinates = coordinates
            super().__init__(coordinates, *args, **kwargs)

    class Electric(electrodes.Dipole):
        def __init__(self, coordinates, *args, **kwargs):
            self._coordinates = coordinates
            super().__init__(coordinates, *args, **kwargs)

    def test_basics(self):

        p1 = [[0, 0, 0], [1, 1, 1]]
        p2 = [0.5, 0.5, 0.5]

        e1 = electrodes.Wire(p1)
        e2 = electrodes.Wire(p2)

        assert e1 == e1
        assert e1 != e2
        assert e1._prefix == 'Wi'
        assert_allclose(e1.points, e1.coordinates)
        assert e1.xtype == 'electric'
        assert_allclose(e2.center, p2)
        assert e2.length == 0
        assert e1.segment_n == 1
        assert e2.segment_n == 0
        assert e1.segment_lengths == np.sqrt(3)
        assert np.empty(e2.segment_lengths)

        e3 = e1.copy()
        assert e1 == e3

        de3 = e3.to_dict()
        assert_allclose(de3['coordinates'], p1)

        # Dummy test
        e4 = self.Magnetic(p1)
        assert e4.xtype == 'magnetic'
        assert e4.coordinates.shape != e4.points.shape

        e5 = self.Electric(p1)
        assert e5.xtype == 'electric'
        assert_allclose(e5.coordinates, e5.points)

    def test_basic_repr(self):
        p1 = np.array([[-10, -10, -10], [10, 10, 10]])
        e1 = electrodes.Wire(p1)
        r1 = e1.__repr__()
        assert 'Wire' in r1
        assert 'center={0.0; 0.0; 0.0} m' in r1
        assert 'n=1; l=' in r1

    def test_warnings(self):

        # Too long 1D array.
        with pytest.raises(ValueError, match="`coordinates` must be of shape"):
            electrodes.Wire((0, 1, 2, 3, 4, 5, 6))

        # 3D array.
        with pytest.raises(ValueError, match="`coordinates` must be of shape"):
            electrodes.Wire(np.ones((3, 3, 3)))


def test_point():
    p1 = (100, 0, -20, 15, 70.1234)
    e1 = electrodes.Point(p1)
    assert_allclose(e1.points, [p1[:3]])
    assert_allclose(e1.center, p1[:3])
    assert e1.azimuth == p1[3]
    assert e1.elevation == p1[4]
    assert e1._prefix == 'Po'

    r1 = e1.__repr__()
    assert 'Point' in r1
    assert 'x=100.0 m, y=0.0 m, z=-20.0 m' in r1
    assert '15.0°' in r1
    assert '70.1°' in r1

    with pytest.raises(ValueError, match="Point coordinates are wrong "):
        electrodes.Point((0, 1, 2, 3, 4, 5, 6))


class TestDipole:

    class Magnetic(electrodes.Dipole):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def test_point(self):
        p1 = (100, 2, 6, 0, 90)
        e1 = electrodes.Dipole(p1)
        assert e1.length == 1.0
        assert_allclose(p1, e1.coordinates)
        assert_allclose([p1[0], p1[1], p1[2]-0.5], e1.points[0, :])
        assert e1.azimuth == 0
        assert e1.elevation == 90
        assert e1._prefix == 'Di'

        e2 = electrodes.Dipole(p1, np.pi)
        assert e2.length == np.pi

        # Dummy test
        e3 = self.Magnetic(p1)
        assert e3.xtype == 'magnetic'
        assert e3.points.shape == (5, 3)

    def test_flat(self):
        p1 = (-10, 10, -10, 10, -10, 10)
        e1 = electrodes.Dipole(p1)
        assert_allclose(e1.length, 20*np.sqrt(3))
        assert_allclose(p1, e1.coordinates)
        assert_allclose(p1[::2], e1.points[0, :])
        assert e1.elevation == 35.264389682754654
        assert e1.azimuth == 45

    def test_dipole(self):
        p1 = np.array([[-10, -10, -10], [10, 10, 10]])
        e1 = electrodes.Dipole(p1)
        assert_allclose(e1.length, 20*np.sqrt(3))
        assert_allclose(p1, e1.coordinates)
        assert_allclose(p1, e1.points)
        assert e1.azimuth == 45
        assert e1.elevation == 35.264389682754654

        # Dummy test
        e2 = self.Magnetic(p1)
        assert e2.xtype == 'magnetic'
        assert e2.points.shape == (5, 3)

    def test_to_from_dict(self):
        p1 = np.array([[-10, -10, -10], [10, 10, 10]])
        e1 = electrodes.Dipole(p1)
        e2 = e1.copy()
        assert e1 == e2

    def test_basic_repr(self):
        p1 = (100, 2, 6, 0, 90)
        e1 = electrodes.Dipole(p1)
        r1 = e1.__repr__()
        assert 'Dipole' in r1
        assert 'center={100.0; 2.0; 6.0} m' in r1
        assert '0.0°' in r1
        assert '90.0°' in r1
        assert 'l=1.0 m' in r1

        p2 = np.array([[-10, -10, -10], [10, 10, 10]])
        e2 = electrodes.Dipole(p2)
        r2 = e2.__repr__()
        assert 'Dipole' in r2
        assert 'e1={-10.0; -10.0; -10.0} m' in r2
        assert 'e2={10.0; 10.0; 10.0} m' in r2

    def test_warnings(self):

        with pytest.raises(ValueError, match="The two electrodes are identic"):
            electrodes.Dipole((0, 0, 0, 0, 0, 0))

        with pytest.raises(ValueError, match="The two electrodes are identic"):
            electrodes.Dipole([[0, 0, 0], [0, 0, 0]])

        with pytest.raises(ValueError, match="Coordinates are wrong defined."):
            electrodes.Dipole((0, 1, 2))

        with pytest.raises(ValueError, match="Coordinates are wrong defined."):
            electrodes.Dipole(np.zeros((2, 4)))

        with pytest.raises(ValueError, match="Coordinates are wrong defined."):
            electrodes.Dipole(np.zeros((5, 3)))


def test_source():
    p1 = [[0, 0, 0], [1, 0, 0]]
    strength = np.pi
    freq = 1.234

    s1 = electrodes.Source(strength, coordinates=p1)
    assert s1.strength == strength
    assert s1._prefix == 'So'

    grid = emg3d.TensorMesh([[1, 1], [1, 1], [1, 1]], (0, 0, 0))
    sfield = emg3d.fields.get_source_field(grid, s1, freq)
    assert s1.get_field(grid, freq) == sfield


def test_tx_electric_point():
    coo = (0, 0, 0, 45, 45)
    s1a = electrodes.TxElectricPoint(coo, strength=np.pi)
    assert s1a.xtype == 'electric'
    assert s1a._prefix == 'TxEP'
    s1b = electrodes.TxElectricPoint.from_dict(s1a.to_dict())
    assert s1a == s1b
    assert_allclose(s1b.coordinates, coo)
    assert_allclose(s1b.points, np.atleast_2d(coo[:3]))
    assert_allclose(s1b.strength, np.pi)

    rep = s1a.__repr__()
    assert "3.1 A" in rep
    assert "=0.0 m, θ=45.0°" in rep


def test_tx_electric_dipole():
    s1a = electrodes.TxElectricDipole(
            (0, 0, 0, 45, 45), strength=np.pi, length=4)
    assert s1a.xtype == 'electric'
    assert s1a._prefix == 'TxED'
    s1b = electrodes.TxElectricDipole.from_dict(s1a.to_dict())
    assert s1a == s1b
    s2a = electrodes.TxElectricDipole(
            (-1, 1, -1, 1, -np.sqrt(2), np.sqrt(2)), strength=np.pi)
    s2b = electrodes.TxElectricDipole.from_dict(s2a.to_dict())
    assert s2a == s2b
    s3a = electrodes.TxElectricDipole(
            [[-1, -1, -np.sqrt(2)], [1, 1, np.sqrt(2)]], strength=np.pi)
    s3b = electrodes.TxElectricDipole.from_dict(s3a.to_dict())
    assert s3a == s3b
    assert_allclose(s1b.points, s2b.points)
    assert_allclose(s1b.points, s3b.points)
    assert_allclose(s1b.strength, s2b.strength)
    assert_allclose(s1b.strength, s3b.strength)
    assert_allclose(s1b.length, s2b.length)
    assert_allclose(s1b.length, s3b.length)

    rep = s3b.__repr__()
    assert "3.1 A" in rep
    assert "m; e2={1.0" in rep


def test_tx_magnetic_point():
    coo = (0, 0, 0, 45, 45)
    s1a = electrodes.TxMagneticPoint(coo, strength=np.pi)
    assert s1a.xtype == 'magnetic'
    assert s1a._prefix == 'TxMP'
    s1b = electrodes.TxMagneticPoint.from_dict(s1a.to_dict())
    assert s1a == s1b
    assert_allclose(s1b.coordinates, coo)
    assert_allclose(s1b.points, np.atleast_2d(coo[:3]))
    assert_allclose(s1b.strength, np.pi)

    rep = s1a.__repr__()
    assert "3.1 A" in rep
    assert "=0.0 m, θ=45.0°" in rep


def test_tx_magnetic_dipole():
    s4a = electrodes.TxMagneticDipole(
            (0, 0, 0, 45, 45), strength=np.pi, length=4)
    assert s4a.xtype == 'magnetic'
    assert s4a._prefix == 'TxMD'
    s4b = electrodes.TxMagneticDipole.from_dict(s4a.to_dict())
    assert s4a == s4b
    s5a = electrodes.TxMagneticDipole(
            (-1, 1, -1, 1, -np.sqrt(2), np.sqrt(2)), strength=np.pi)
    s5b = electrodes.TxMagneticDipole.from_dict(s5a.to_dict())
    assert s5a == s5b
    s6a = electrodes.TxMagneticDipole(
            [[-1, -1, -np.sqrt(2)], [1, 1, np.sqrt(2)]], strength=np.pi)
    s6b = electrodes.TxMagneticDipole.from_dict(s6a.to_dict())
    assert s6a == s6b
    assert_allclose(s4b.points, s5b.points)
    assert_allclose(s4b.points, s6b.points)
    assert_allclose(s4b.strength, s5b.strength)
    assert_allclose(s4b.strength, s6b.strength)
    assert_allclose(s4b.length*2, s5b.length)
    assert_allclose(s4b.length*2, s6b.length)

    rep = s6b.__repr__()
    assert "3.1 A" in rep
    assert "m; e2={-0.7" in rep


def test_tx_electric_wire():
    n = np.sqrt(2)/2
    coo = np.array([[0, -n, 0], [n, 0, 0], [0, n, 0],
                    [-n, 0, n], [0, 0, -n]])
    s7a = electrodes.TxElectricWire(coo, np.pi)
    assert s7a._prefix == 'TxEW'
    assert s7a.xtype == 'electric'
    s7b = electrodes.TxElectricWire.from_dict(s7a.to_dict())
    assert s7a == s7b
    assert_allclose(s7a.center, 0)

    rep = s7b.__repr__()
    assert "3.1 A" in rep
    assert "n=4; l=4.8" in rep


def test_receiver():
    rcoo = [1000, -200, 0]
    scoo = [50, 50, 50, 0, 0]

    ra = electrodes.Receiver(False, coordinates=rcoo, data_type='complex')
    rr = electrodes.Receiver(True, coordinates=rcoo, data_type='complex')
    s1 = electrodes.TxElectricDipole(coordinates=scoo)

    assert ra.relative is False
    assert ra._prefix == 'Re'
    assert rr.relative is True

    assert_allclose(ra.center_abs(s1), ra.center)
    assert_allclose(ra.coordinates_abs(s1), ra.center)

    assert_allclose(rr.center_abs(s1), [1050, -150, 50])
    assert_allclose(rr.coordinates_abs(s1), [1050, -150, 50])


def test_rx_electric_point():
    r1a = electrodes.RxElectricPoint((1200, -56, 23.214, 368, 15), True)

    assert r1a._adjoint_source == electrodes.TxElectricPoint

    assert r1a.xtype == 'electric'
    assert r1a._prefix == 'RxEP'
    assert r1a.relative is True
    r1b = electrodes.RxElectricPoint.from_dict(r1a.to_dict())
    assert r1a == r1b

    s1 = electrodes.TxElectricDipole((10000, 500, 50, 10, 60))
    assert_allclose(r1a.center_abs(s1), (11200, 444, 73.214))
    assert_allclose(r1a.coordinates_abs(s1), (11200, 444, 73.214, 368, 15))

    r2a = electrodes.RxElectricPoint((1200, -56, 23.214, 368, 15))
    assert r2a.xtype == 'electric'
    assert r2a.relative is False
    r2b = electrodes.RxElectricPoint.from_dict(r2a.to_dict())
    assert r2a == r2b

    rep = r2b.__repr__()
    assert "absolute" in rep
    assert "θ=368.0°, φ=15.0" in rep
    rep = r1a.__repr__()
    assert "relative" in rep

    with pytest.raises(ValueError, match="Unknown data type 'bla'"):
        electrodes.RxElectricPoint((0, 0, 0, 0, 0), data_type='bla')


def test_rx_magnetic_point():
    r1a = electrodes.RxMagneticPoint((-1200, 56, -23.214, 0, 90))

    assert r1a._adjoint_source == electrodes.TxMagneticPoint

    assert r1a.xtype == 'magnetic'
    assert r1a._prefix == 'RxMP'
    assert r1a.relative is False
    r1b = electrodes.RxMagneticPoint.from_dict(r1a.to_dict())
    assert r1a == r1b

    r2a = electrodes.RxMagneticPoint((-1200, 56, -23.214, 0, 90), True)
    assert r2a.xtype == 'magnetic'
    assert r2a.relative is True
    r2b = electrodes.RxMagneticPoint.from_dict(r2a.to_dict())
    assert r2a == r2b

    rep = r2b.__repr__()
    assert "relative" in rep
    assert "x=-1,200.0 m, y=56.0 m" in rep


def test_point_to_dipole():
    source = (10, 100, -1000, 0, 0)
    length = 111.0
    out = electrodes.point_to_dipole(source, length)
    assert out.shape == (2, 3)
    assert out[0, 0] == source[0]-length/2
    assert out[1, 0] == source[0]+length/2
    assert out[0, 1] == source[1]
    assert out[1, 1] == source[1]
    assert out[0, 2] == source[2]
    assert out[1, 2] == source[2]

    source = (10, 100, -1000, 30, 60)
    length = 2.0
    out = electrodes.point_to_dipole(source, length)
    assert_allclose(
        out,
        [[9.5669873, 99.75, -1000.8660254], [10.4330127, 100.25, -999.1339746]]
    )


class TestDipoleToPoint:
    def test_axes_faces_quadrants(self):
        pi_1 = np.pi
        pi_2 = np.pi/2
        pi_4 = np.pi/4
        pi_34 = 3*np.pi/4

        # We test here an extensive list of simple cases:
        # - dipoles along the principal axes
        # - dipoles in the middle between two axes (faces)
        # - dipoles into the middle of three axes (quadrants)

        # Format: (x, y, z): azm_deg, elv_deg, azm_rad, elv_rad
        data = {
            # 6 axes
            (+1, +0, +0): [0, 0, 0, 0],
            (+0, +1, +0): [90, 0, pi_2, 0],
            (-1, +0, +0): [180, 0, pi_1, 0],
            (+0, -1, +0): [-90, 0, -pi_2, 0],
            (+0, +0, +1): [0, 90, 0, pi_2],
            (+0, +0, -1): [0, -90, 0, -pi_2],
            # 12 faces
            (+1, +1, +0): [45, 0, pi_4, 0],
            (-1, +1, +0): [135, 0, pi_34, 0],
            (-1, -1, +0): [-135, 0, -pi_34, 0],
            (+1, -1, +0): [-45, 0, -pi_4, 0],
            # --
            (+1, +0, +1): [0, 45, 0, pi_4],
            (-1, +0, +1): [180, 45, pi_1, pi_4],
            (-1, +0, -1): [180, -45, pi_1, -pi_4],
            (+1, +0, -1): [0, -45, 0, -pi_4],
            # --
            (+0, +1, +1): [90, 45, pi_2, pi_4],
            (+0, -1, +1): [-90, 45, -pi_2, pi_4],
            (+0, -1, -1): [-90, -45, -pi_2, -pi_4],
            (+0, +1, -1): [90, -45, pi_2, -pi_4],
            # 8 quadrants
            (+1, +1, +np.sqrt(2)): [45, 45, pi_4, pi_4],
            (-1, +1, +np.sqrt(2)): [135, 45, pi_34, pi_4],
            (-1, -1, +np.sqrt(2)): [-135, 45, -pi_34, pi_4],
            (+1, -1, +np.sqrt(2)): [-45, 45, -pi_4, pi_4],
            # --
            (+1, +1, -np.sqrt(2)): [45, -45, pi_4, -pi_4],
            (-1, +1, -np.sqrt(2)): [135, -45, pi_34, -pi_4],
            (-1, -1, -np.sqrt(2)): [-135, -45, -pi_34, -pi_4],
            (+1, -1, -np.sqrt(2)): [-45, -45, -pi_4, -pi_4],
        }

        for points, values in data.items():

            # 1. Check get_angles_from_dipole
            coo = np.array([[0, 0, 0], points])

            # 1.a Check angles degree
            azm, elv, rad = electrodes.dipole_to_point(coo)
            assert_allclose(azm, values[0])
            assert_allclose(elv, values[1])
            assert_allclose(rad, np.linalg.norm(points))

            # 1.b Check angles radians
            azm, elv, _ = electrodes.dipole_to_point(coo, deg=False)
            assert_allclose(azm, values[2])
            assert_allclose(elv, values[3])

            # 2. Check get_dipole_from_point
            # Center (0, 0, 0); 2*length, so we can compare 2nd point.
            length = 2*np.linalg.norm(points)

            # 2.a Check points degree
            coo = electrodes.point_to_dipole(
                    (0, 0, 0, values[0], values[1]), length)
            assert_allclose(coo[1, :], points)

            # 2.b Check points radians
            coo = electrodes.point_to_dipole(
                    (0, 0, 0, values[2], values[3]), length, deg=False)
            assert_allclose(coo[1, :], points, atol=1e-15)

    def test_arbitrary(self):

        i_azm1, i_elv1, i_rad1 = -25, 88, 1e6
        coords1 = electrodes.point_to_dipole(
                (1e6, 1e-6, 10, i_azm1, i_elv1), i_rad1)
        o_azm1, o_elv1, o_rad1 = electrodes.dipole_to_point(coords1)
        assert_allclose(i_azm1, o_azm1)
        assert_allclose(i_elv1, o_elv1)
        assert_allclose(i_rad1, o_rad1)

        i_azm2, i_elv2, i_rad2 = -33.3, 90, 2  # <= azm != 0.0
        coords2 = electrodes.point_to_dipole(
                (1e6, -50, 3.33, i_azm2, i_elv2), i_rad2)
        o_azm2, o_elv2, o_rad2 = electrodes.dipole_to_point(coords2)
        assert_allclose(0.0, o_azm2)  # <= azm == 0.0
        assert_allclose(i_elv2, o_elv2)
        assert_allclose(i_rad2, o_rad2)


def test_point_to_square_loop():
    source = (10, 100, -1000, 45, 90)
    length = 4
    out = electrodes.point_to_square_loop(source, length)
    assert out.shape == (5, 3)
    assert_allclose(out[:, 0], [9, 9, 11, 11, 9])
    assert_allclose(out[:, 1], [101, 99, 99, 101, 101])
    assert_allclose(out[:, 2], source[2])

    source = (0, 0, 0, 0, 0)
    length = 8
    out = electrodes.point_to_square_loop(source, length)
    assert out.shape == (5, 3)
    assert_allclose(out[:, 0], 0)
    assert_allclose(out[:, 1], [2, 0, -2, 0, 2])
    assert_allclose(out[:, 2], [0, 2, 0, -2, 0])

    source = (10, 100, -1000, 30, 60)
    length = 4
    out = electrodes.point_to_square_loop(source, length)
    assert_allclose(out[:, 0],
                    [9.292893, 8.93934, 10.707107, 11.06066, 9.292893])
    assert_allclose(out[:, 2],
                    [-1000., -999.292893, -1000., -1000.707107, -1000.])
    assert_allclose(out[0, :], out[-1, :])  # first and last point identical


def test_rotation():
    assert_allclose(electrodes.rotation(0, 0), [1, 0, 0])
    assert_allclose(electrodes.rotation(180, 0), [-1, 0, 0])
    assert_allclose(electrodes.rotation(90, 0), [0, 1, 0])
    assert_allclose(electrodes.rotation(-90, 0), [0, -1, 0])
    assert_allclose(electrodes.rotation(0, 90), [0, 0, 1])
    assert_allclose(electrodes.rotation(0, -90), [0, 0, -1])

    dazm, delv = 30, 60
    razm, relv = np.deg2rad(dazm), np.deg2rad(delv)
    assert_allclose(
        electrodes.rotation(dazm, delv),
        [np.cos(razm)*np.cos(relv), np.sin(razm)*np.cos(relv), np.sin(relv)])

    dazm, delv = -45, 180
    razm, relv = np.deg2rad(dazm), np.deg2rad(delv)
    assert_allclose(
        electrodes.rotation(dazm, delv),
        [np.cos(razm)*np.cos(relv), np.sin(razm)*np.cos(relv), np.sin(relv)],
        atol=1e-14)

    # Radians
    azm, elv = np.pi/3, np.pi/4
    rot1 = electrodes.rotation(azm, elv, deg=False)
    rot2 = electrodes.rotation(np.rad2deg(azm), np.rad2deg(elv))
    assert_allclose(rot1, rot2)


@pytest.mark.skipif(discretize is None, reason="discretize not installed.")
def test_adjoint():
    # Simple stretched mesh.
    hx = [200, 100, 50, 100, 200]
    mesh = emg3d.TensorMesh([hx, hx, hx], 'CCC')

    # Randomly located & rotated point receivers in the middle cell.
    rx, ry, rz = rng.uniform(-25, 25, 3)
    azm = rng.uniform(-180, 180, 1)[0]
    ele = rng.uniform(-90, 90, 1)[0]

    # The two currently implemented receiver types.
    RxEP = emg3d.RxElectricPoint((rx, ry, rz, azm, ele))
    RxMP = emg3d.RxMagneticPoint((rx, ry, rz, azm, ele))

    # Frequency
    frequency = 0.7
    omega = 2*np.pi*frequency
    sval = 1j*omega
    mu0 = 4e-7*np.pi
    smu0 = sval*mu0

    # Loop over receivers and check.
    for rec in [RxEP, RxMP]:

        def fwd(u):
            """What does `Fields`."""
            # Cast input vector to field instance
            field = emg3d.Field(mesh, u.copy(), frequency)  # ¡ COPY !

            # Get magnetic field if rec type is magnetic.
            if rec.xtype == 'magnetic':
                model = emg3d.Model(mesh)
                field = emg3d.fields.get_magnetic_field(model, field)

                # _point_vector_magnetic has a minus factor.
                # Most likely correct, but verif with jvec/jtvec test ?
                field.field *= -1

            return field.get_receiver(rec.coordinates, method='linear')

        def adj(v):
            """What does `Simulation._get_rfield()`."""
            # The strength in _get_rfield is conj(residual*weight), normalized
            strength = complex(v/-smu0)
            src = rec._adjoint_source(rec.coordinates, strength=strength)
            return src.get_field(mesh, frequency).field

        discretize.tests.assert_isadjoint(
            forward=fwd, adjoint=adj,
            shape_u=mesh.n_edges, shape_v=1,
            complex_u=rng.choice([True, False]),  # Random float or complex.
            complex_v=rng.choice([True, False]),  # Random float or complex.
        )


def test_all_dir():
    assert set(electrodes.__all__) == set(dir(electrodes))
