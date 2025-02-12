import pytest
import numpy as np


def test_magnetic_import():
    from atomsmltr.environment.fields import magnetic


def test_magnetic_offset():
    from atomsmltr.environment.fields.magnetic import MagneticOffset

    # -- exceptions
    with pytest.raises(ValueError) as excinfo:
        mag_field = MagneticOffset(0.5)
    with pytest.raises(TypeError) as excinfo:
        mag_field = MagneticOffset(("5", 1, 1))

    # -- good usage
    offset = (1, 0, 0)
    mag_field = MagneticOffset(offset=offset)
    mag_field.print_info()
    assert np.allclose(mag_field.value(1, 4, 8), offset)

    new_offset = (4.5, 2.8, np.sqrt(2))
    mag_field.offset = new_offset
    mag_field.print_info()

    assert np.allclose(mag_field.value(1, 4, 8), new_offset)

    # plot
    limits = (-10, 10, -10, 10, -10, 10)
    Npoints = 3
    mag_field.plot3D(limits=limits, Npoints=Npoints, show=False, color="C1")


def test_magnetic_gradient():
    from atomsmltr.environment.fields.magnetic import MagneticGradient

    # -- generate a good field

    mag_field = MagneticGradient(
        origin=(0, 0, 0),
        slope=0,
        gradient_direction=(1, 0, 0),
        field_direction=(0, 1, 1),
        offset=0,
    )
    mag_field.print_info()

    # gradient along x, field at pi/4 wrt. y
    mag_field.slope = -2.0
    mag_field.offset = 0.25
    uv = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])  # unit vector
    for y, z in zip([1, -8, 4, 5], [-9, 7, 5, 4]):
        for x in np.linspace(-10, 10, 100):
            exp_value = (mag_field.offset + x * mag_field.slope) * uv
            field_value = mag_field.value(x, y, z)
            msg = f"{x=}, {y=}, {z=} \n"
            msg += f"{exp_value=} \n"
            msg += f"{field_value=} \n"
            assert np.allclose(field_value, exp_value), msg
            assert np.allclose(
                np.linalg.norm(field_value),
                np.abs((mag_field.offset + x * mag_field.slope)),
            )

    # plot
    limits = (-10, 10, -10, 10, -10, 10)
    Npoints = (10, 3, 3)
    mag_field.plot3D(
        limits=limits, Npoints=Npoints, show=False, color="C2", normalize=True, scale=5
    )

    # gradient along z, field along x
    mag_field.slope = 7.0
    mag_field.offset = -8.0
    mag_field.gradient_direction = (0, 0, 1)
    mag_field.field_direction = (1, 0, 0)
    mag_field.print_info()
    uv = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])  # unit vector

    for y, z in zip([1, -8, 4, 5], [-9, 7, 5, 4]):
        for x in np.linspace(-10, 10, 100):
            exp_value = (z * mag_field.slope + mag_field.offset, 0, 0)
            field_value = mag_field.value(x, y, z)
            msg = f"{x=}, {y=}, {z=} \n"
            msg += f"{exp_value=} \n"
            msg += f"{field_value=} \n"
            assert np.allclose(field_value, exp_value), msg
            assert np.allclose(
                np.linalg.norm(field_value),
                np.abs((mag_field.offset + z * mag_field.slope)),
            )
    # -- check exceptions
    with pytest.raises(ValueError) as excinfo:
        mag_field.gradient_direction = 5.0
    with pytest.raises(TypeError) as excinfo:
        mag_field.gradient_direction = ("5", 1, 1)
    with pytest.raises(ValueError) as excinfo:
        mag_field.origin = 5.0
    with pytest.raises(TypeError) as excinfo:
        mag_field.origin = ("5", 1, 1)
    with pytest.raises(ValueError) as excinfo:
        mag_field.field_direction = 5.0
    with pytest.raises(TypeError) as excinfo:
        mag_field.field_direction = ("5", 1, 1)
    with pytest.raises(ValueError) as excinfo:
        mag_field.slope = [0, 1]
    with pytest.raises(TypeError) as excinfo:
        mag_field.offset = "5"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_magnetic_import()
    test_magnetic_offset()
    test_magnetic_gradient()

    plt.show()
