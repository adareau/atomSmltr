import pytest
import numpy as np
from atomsmltr.utils.misc import (
    check_scalar_field_value_function,
    check_vector_field_value_function,
)


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


if __name__ == "__main__":
    test_rotation_functions()
