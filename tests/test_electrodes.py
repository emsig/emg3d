# import pytest
import numpy as np
from numpy.testing import assert_allclose

from emg3d import electrodes


def test_point_to_dipole():
    source = (10, 100, -1000, 0, 0)
    length = 111.0
    out = electrodes._point_to_dipole(source, length)
    assert out.shape == (2, 3)
    assert out[0, 0] == source[0]-length/2
    assert out[1, 0] == source[0]+length/2
    assert out[0, 1] == source[1]
    assert out[1, 1] == source[1]
    assert out[0, 2] == source[2]
    assert out[1, 2] == source[2]

    source = (10, 100, -1000, 30, 60)
    length = 2.0
    out = electrodes._point_to_dipole(source, length)
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
            azm, elv, rad = electrodes._dipole_to_point(coo)
            assert_allclose(azm, values[0])
            assert_allclose(elv, values[1])
            assert_allclose(rad, np.linalg.norm(points))

            # 1.b Check angles radians
            azm, elv, _ = electrodes._dipole_to_point(coo, deg=False)
            assert_allclose(azm, values[2])
            assert_allclose(elv, values[3])

            # 2. Check get_dipole_from_point
            # Center (0, 0, 0); 2*length, so we can compare 2nd point.
            length = 2*np.linalg.norm(points)

            # 2.a Check points degree
            coo = electrodes._point_to_dipole(
                    (0, 0, 0, values[0], values[1]), length)
            assert_allclose(coo[1, :], points)

            # 2.b Check points radians
            coo = electrodes._point_to_dipole(
                    (0, 0, 0, values[2], values[3]), length, deg=False)
            assert_allclose(coo[1, :], points, atol=1e-15)

    def test_arbitrary(self):

        i_azm1, i_elv1, i_rad1 = -25, 88, 1e6
        coords1 = electrodes._point_to_dipole(
                (1e6, 1e-6, 10, i_azm1, i_elv1), i_rad1)
        o_azm1, o_elv1, o_rad1 = electrodes._dipole_to_point(coords1)
        assert_allclose(i_azm1, o_azm1)
        assert_allclose(i_elv1, o_elv1)
        assert_allclose(i_rad1, o_rad1)

        i_azm2, i_elv2, i_rad2 = -33.3, 90, 2  # <= azm != 0.0
        coords2 = electrodes._point_to_dipole(
                (1e6, -50, 3.33, i_azm2, i_elv2), i_rad2)
        o_azm2, o_elv2, o_rad2 = electrodes._dipole_to_point(coords2)
        assert_allclose(0.0, o_azm2)  # <= azm == 0.0
        assert_allclose(i_elv2, o_elv2)
        assert_allclose(i_rad2, o_rad2)


def test_point_to_square_loop():
    source = (10, 100, -1000, 0, 0)
    length = np.sqrt(2)
    out = electrodes._point_to_square_loop(source, length)
    assert out.shape == (3, 5)
    assert_allclose(out[0, :], source[0])  # x-directed, all x the same
    assert_allclose(out[1, :], [101, 100, 99, 100, 101])
    assert_allclose(out[2, :], [-1000, -999, -1000, -1001, -1000])

    source = (10, 100, -1000, 30, 60)
    length = np.sqrt(2)
    out = electrodes._point_to_square_loop(source, length)
    assert_allclose(out[0, :], [9.5, 9.25, 10.5, 10.75, 9.5])
    assert_allclose(out[2, :], [-1000, -999.5, -1000, -1000.5, -1000])
    assert_allclose(out[:, 0], out[:, -1])  # first and last point identical


def test_rotation():
    assert_allclose(electrodes._rotation(0, 0), [1, 0, 0])
    assert_allclose(electrodes._rotation(180, 0), [-1, 0, 0])
    assert_allclose(electrodes._rotation(90, 0), [0, 1, 0])
    assert_allclose(electrodes._rotation(-90, 0), [0, -1, 0])
    assert_allclose(electrodes._rotation(0, 90), [0, 0, 1])
    assert_allclose(electrodes._rotation(0, -90), [0, 0, -1])

    dazm, delv = 30, 60
    razm, relv = np.deg2rad(dazm), np.deg2rad(delv)
    assert_allclose(
        electrodes._rotation(dazm, delv),
        [np.cos(razm)*np.cos(relv), np.sin(razm)*np.cos(relv), np.sin(relv)])

    dazm, delv = -45, 180
    razm, relv = np.deg2rad(dazm), np.deg2rad(delv)
    assert_allclose(
        electrodes._rotation(dazm, delv),
        [np.cos(razm)*np.cos(relv), np.sin(razm)*np.cos(relv), np.sin(relv)],
        atol=1e-14)

    # Radians
    azm, elv = np.pi/3, np.pi/4
    rot1 = electrodes._rotation(azm, elv, deg=False)
    rot2 = electrodes._rotation(np.rad2deg(azm), np.rad2deg(elv))
    assert_allclose(rot1, rot2)
