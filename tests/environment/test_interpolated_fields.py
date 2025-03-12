import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from atomsmltr.utils.misc import check_vector_field_value_function

# % GENERIC TESTS


def _check_position_exceptions(func):
    # - 0 check exceptions
    with pytest.raises(ValueError) as excinfo:
        func(0)
    with pytest.raises(ValueError) as excinfo:
        func((0, 0))
    with pytest.raises(ValueError) as excinfo:
        func(np.linspace(0, 1, 20))
    with pytest.raises(ValueError) as excinfo:
        func(np.mgrid[0:1:10j, 0:1:10j, 0:1:10j])


def test_interpolated_1D_1D():
    from atomsmltr.environment.fields.magnetic import InterpMag1D1D

    # -- load data
    root = os.path.dirname(__file__)
    data_file = Path(root) / "data" / "field.dat"
    data = np.genfromtxt(data_file)
    data_x, data_y = data.T

    # -- interpolate
    mag_field = InterpMag1D1D(data_x, data_y, field_direction=(1, 1, 0))
    mag_field.print_info()
    _check_position_exceptions(mag_field.value)
    check_vector_field_value_function(mag_field.value)

    # -- check values
    # -
    mag_field.field_direction = (1, 2, 3)
    mag_field.position_direction = (1, 0, 0)
    mag_field.scale = 2.0
    position = np.array([(x, 0, 0) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)
    # -
    mag_field.field_direction = (-1, 0, -1)
    mag_field.position_direction = (-1, 0, 0)
    mag_field.scale = -1.5
    position = np.array([(-x, 0, 0) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)
    # -
    mag_field.field_direction = (-1, 0, -1)
    mag_field.position_direction = (0, 1, 0)
    mag_field.scale = -1.5
    position = np.array([(0, x, 0) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)
    # -
    mag_field.field_direction = (-1, 0, -1)
    mag_field.position_direction = (0, 1, 1)
    mag_field.scale = -1.5
    position = np.array([(0, x / np.sqrt(2), x / np.sqrt(2)) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)


def test_interpolated_3D_3D():
    from atomsmltr.environment import InterpMag3D3D

    # -- create a model field
    def B_th(position):
        position = np.asanyarray(position)
        X, Y, Z = position.T
        Bx = X + 2 * Y - Z
        By = Y - 5 * Z
        Bz = 2 * Z - X
        B = np.array([Bx, By, Bz]).T
        return B

    check_vector_field_value_function(B_th)

    x = np.linspace(-1, 1, 20)
    y = np.linspace(-2, 2, 21)
    z = np.linspace(-3, 3, 22)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    position = np.array([X.T, Y.T, Z.T]).T

    B_interp = B_th(position)

    # -- init an interpolated field
    mag_field = InterpMag3D3D(data_position=(x, y, z), data_field=B_interp)
    mag_field.print_info()
    _check_position_exceptions(mag_field.value)
    check_vector_field_value_function(mag_field.value)

    # -- check values at grid position
    p_flat = position.reshape((-1, 3))
    B_flat = B_interp.reshape((-1, 3))
    for p, B in zip(p_flat[::10], B_flat[::10]):
        assert np.allclose(B, B_th(p))
        assert np.allclose(B, mag_field.value(p))

    # -- check interpolation ?
    for p in [(0.54, 0.004, np.pi / 2), (-0.45, 0.1896, 2.8956)]:
        assert np.allclose(B_th(p), mag_field.value(p))


if __name__ == "__main__":
    # test_interpolated_1D_1D()
    test_interpolated_3D_3D()
