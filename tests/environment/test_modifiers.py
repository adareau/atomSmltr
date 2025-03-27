import pytest
import numpy as np
from atomsmltr.utils.misc import (
    check_scalar_field_value_function,
    check_vector_field_value_function,
)


def _generic_modifier_tests(modifier, *args, **kwargs):
    from atomsmltr.environment import MagneticGradient, GaussianLaserBeam, Limits

    # - init objects for test
    mag_field = MagneticGradient((0, 0, 0), 4, (1, 2, 3), (0, 1, -1))
    laser_beam = GaussianLaserBeam()
    zone = Limits(0, 1, 0)

    # - modify
    for obj in [mag_field, laser_beam, zone]:
        modifier(obj, *args, **kwargs)
        obj.print_info()
        if obj.vector:
            check_vector_field_value_function(obj.get_value)
        else:
            check_scalar_field_value_function(obj.get_value)


def test_rotation_functions():
    from atomsmltr.environment.modifiers import (
        rotation_matrix,
        rotate_position_vector,
        rotate_position_vector_alt,
    )

    # -- testing rmat
    # rotation along x
    u = (1, 0, 0)
    for theta in np.linspace(-5, 5, 10):
        rmat = rotation_matrix(u, theta)
        rmat_th = [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
        assert np.allclose(rmat, rmat_th)

    # rotation along y
    u = (0, 1, 0)
    for theta in np.linspace(-5, 5, 10):
        rmat = rotation_matrix(u, theta)
        rmat_th = [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
        assert np.allclose(rmat, rmat_th)

    # rotation along z
    u = (0, 0, 1)
    for theta in np.linspace(-5, 5, 10):
        rmat = rotation_matrix(u, theta)
        rmat_th = [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
        assert np.allclose(rmat, rmat_th)

    # invert
    u = (1, 2, -3)
    for theta in np.linspace(-5, 5, 10):
        rmat = rotation_matrix(u, theta)
        irmat = rotation_matrix(u, -theta)
        assert np.allclose(np.matmul(rmat, irmat), np.eye(3))

    # -- testing coordinates rotators

    # - element-wise test
    # settings
    u = np.random.uniform(-1, 1, (3,))
    theta = np.random.uniform(-6, 6)
    pos = np.random.uniform(-5, 5, (10, 8, 9, 3))

    # compute
    rm = rotation_matrix(u, theta)
    irm = np.linalg.inv(rm)
    rotpos = rotate_position_vector(pos, u, theta)
    rotpos_alt = rotate_position_vector_alt(pos, u, theta)
    irotpos = rotate_position_vector(pos, u, -theta)

    # check
    for i in range(10):
        for j in range(8):
            for k in range(9):
                r = pos[i, j, k, :]
                rot = rotpos[i, j, k, :]
                rotman = rotpos_alt[i, j, k, :]
                irot = irotpos[i, j, k, :]
                assert np.allclose(rot, np.matmul(rm, r))
                assert np.allclose(rotman, np.matmul(rm, r))
                assert np.allclose(irot, np.matmul(irm, r))

    # - forth and back tests
    u = np.random.uniform(-1, 1, (3,))
    theta = np.random.uniform(-6, 6)
    pos = np.random.uniform(-5, 5, (5, 4, 3))

    # normal
    rpos = rotate_position_vector(pos, u, theta)
    irpos = rotate_position_vector(rpos, u, -theta)
    assert np.allclose(irpos, pos)

    # alt
    rpos = rotate_position_vector_alt(pos, u, theta)
    irpos = rotate_position_vector_alt(rpos, u, -theta)
    assert np.allclose(irpos, pos)


def test_rotation_modifier():
    from atomsmltr.environment.modifiers import rotate
    from atomsmltr.environment import MagneticGradient

    # - generic tests
    _generic_modifier_tests(rotate, u=(1, 2, 3), theta=0.5)

    # - specific tests
    # magnetic field gradient, gradient along x, field along y
    settings = {
        "origin": (0, 0, 0),
        "slope": 1,
        "gradient_direction": (1, 0, 0),
        "field_direction": (0, 1, 0),
    }
    # 0) define a comparison field
    mag_gradient_cmp = MagneticGradient(**settings)
    # 1) rotation of pi/2 along z
    # this should make gradient direction along y
    # and field pointing along -x
    # 1.a) prepare comparison gradient
    mag_gradient_cmp.gradient_direction = (0, 1, 0)
    mag_gradient_cmp.field_direction = (-1, 0, 0)
    # 1.b) create gradient and rotate it
    mag_gradient = MagneticGradient(**settings)
    rotate(mag_gradient, (0, 0, 1), np.pi / 2)
    # 1.c) compare
    pos = np.mgrid[-10:10:10j, -10:10:10j, -10:10:10j].T
    assert np.allclose(mag_gradient.get_value(pos), mag_gradient_cmp.get_value(pos))

    # 2) rotation of pi/2 along x
    # this should make gradient direction along x
    # and field pointing along z
    # 1.a) prepare comparison gradient
    mag_gradient_cmp.gradient_direction = (1, 0, 0)
    mag_gradient_cmp.field_direction = (0, 0, 1)
    # 1.b) create gradient and rotate it
    mag_gradient = MagneticGradient(**settings)
    rotate(mag_gradient, (1, 0, 0), np.pi / 2)
    # 1.c) compare
    pos = np.mgrid[-10:10:10j, -10:10:10j, -10:10:10j].T
    assert np.allclose(mag_gradient.get_value(pos), mag_gradient_cmp.get_value(pos))


def test_shift_modifier():
    from atomsmltr.environment.modifiers import shift
    from atomsmltr.environment import MagneticQuadrupoleX, GaussianLaserBeam

    # - generic tests
    _generic_modifier_tests(shift, dr=(5.0, -8.0, 3.0))

    # - specific tests
    # magfield
    dr = (-5.0, 0.45, 8.4)
    mag_field = MagneticQuadrupoleX((0, 0, 0), 1)
    mag_field_cmp = MagneticQuadrupoleX(dr, 1)
    shift(mag_field, dr)
    pos = np.mgrid[-10:10:10j, -10:10:10j, -10:10:10j].T
    assert np.allclose(mag_field.get_value(pos), mag_field_cmp.get_value(pos))

    # gaussian beam
    dr = (1e-3, -0.5e-3, 0)
    laser = GaussianLaserBeam(direction=(1, 1, 0), waist_position=(0, 0, 0))
    laser_cmp = GaussianLaserBeam(direction=(1, 1, 0), waist_position=dr)
    shift(laser, dr)
    pos = np.mgrid[-1e-3:1e-3:10j, -1e-3:1e-3:10j, 0:0:1j].T
    assert np.allclose(laser.get_value(pos), laser_cmp.get_value(pos))


if __name__ == "__main__":
    # test_rotation_functions()
    # test_rotation_modifier()
    test_shift_modifier()
