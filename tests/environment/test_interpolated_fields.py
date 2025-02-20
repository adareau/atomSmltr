import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from test_magnetic_field import _check_vector_field_value_function


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
    _check_vector_field_value_function(mag_field.value)

    # -- check values
    # -
    mag_field.field_direction = (1, 2, 3)
    mag_field.x_direction = (1, 0, 0)
    mag_field.scale = 2.0
    position = np.array([(x, 0, 0) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)
    # -
    mag_field.field_direction = (-1, 0, -1)
    mag_field.x_direction = (-1, 0, 0)
    mag_field.scale = -1.5
    position = np.array([(-x, 0, 0) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)
    # -
    mag_field.field_direction = (-1, 0, -1)
    mag_field.x_direction = (0, 1, 0)
    mag_field.scale = -1.5
    position = np.array([(0, x, 0) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)
    # -
    mag_field.field_direction = (-1, 0, -1)
    mag_field.x_direction = (0, 1, 1)
    mag_field.scale = -1.5
    position = np.array([(0, x / np.sqrt(2), x / np.sqrt(2)) for x in data_x])
    B = mag_field.value(position)
    B_proj = np.dot(B, mag_field.field_direction)
    assert np.allclose(data_y * mag_field.scale, B_proj)


if __name__ == "__main__":
    test_interpolated_1D_1D()
