import pytest
import numpy as np

# % GENERIC TESTS


def _check_vector_field_value_function(func):
    """Checks that a function yielding values of a 3D field
    field behaves correctly with numpy arrays.
    """

    # - 0 check exceptions
    with pytest.raises(ValueError) as excinfo:
        func(0)
    with pytest.raises(ValueError) as excinfo:
        func((0, 0))
    with pytest.raises(ValueError) as excinfo:
        func(np.linspace(0, 1, 20))
    with pytest.raises(ValueError) as excinfo:
        func(np.mgrid[0:1:10j, 0:1:10j, 0:1:10j])

    # - 1 check that it works with a single position
    position = (0, 0, 0)
    value = func(position)
    assert value.shape == (3,)

    # - 2 with arrays
    # -
    position = np.mgrid[0:1:8j, 0:5:10j, 0:0:1j].T
    value = func(position)
    assert value.shape == position.shape
    # -
    position = position[0]
    value = func(position)
    assert value.shape == position.shape


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
    # - check init and info print
    offset = (1, 0, 0)
    mag_field = MagneticOffset(offset=offset)
    mag_field.print_info()

    # - check value function behaviour
    _check_vector_field_value_function(mag_field.value)

    # - check values
    # -
    assert np.allclose(mag_field.value((1, 4, 8)), offset)
    # -
    new_offset = (4.5, 2.8, np.sqrt(2))
    mag_field.offset = new_offset
    mag_field.print_info()
    assert np.allclose(mag_field.value((1, 4, 8)), new_offset)


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

    # -- check value function behaviour
    _check_vector_field_value_function(mag_field.value)

    # -- check values
    # - (1) gradient along x, field at pi/4 wrt. y
    # settings
    mag_field.slope = -2.0
    mag_field.offset = 0.25
    mag_field.gradient_direction = (1, 0, 0)
    mag_field.field_direction = (0, 1, 1)

    # compute
    position = np.mgrid[-10:10:15j, -5:5:20j, -1:1:6j].T
    value = mag_field.value(position)
    # check >> we make a loop, not efficient but this way we know what to expect
    X, Y, Z = position.T
    Bx, By, Bz = value.T
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    Bx, By, Bz = Bx.ravel(), By.ravel(), Bz.ravel()
    uf = mag_field.field_direction / np.linalg.norm(mag_field.field_direction)
    for x, y, z, bx, by, bz in zip(X, Y, Z, Bx, By, Bz):
        norm = x * mag_field.slope + mag_field.offset
        bx_exp, by_exp, bz_exp = uf * norm
        assert np.allclose((bx_exp, by_exp, bz_exp), (bx, by, bz))

    # check

    # gradient along z, field along x
    mag_field.slope = 7.0
    mag_field.offset = -8.0
    mag_field.gradient_direction = (0, 0, 1)
    mag_field.field_direction = (1, 0, 0)
    # compute
    position = np.mgrid[-10:10:15j, -5:5:20j, -1:1:6j].T
    value = mag_field.value(position)
    # check >> we make a loop, not efficient but this way we know what to expect
    X, Y, Z = position.T
    Bx, By, Bz = value.T
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    Bx, By, Bz = Bx.ravel(), By.ravel(), Bz.ravel()
    uf = mag_field.field_direction / np.linalg.norm(mag_field.field_direction)
    for x, y, z, bx, by, bz in zip(X, Y, Z, Bx, By, Bz):
        norm = z * mag_field.slope + mag_field.offset
        bx_exp, by_exp, bz_exp = uf * norm
        assert np.allclose((bx_exp, by_exp, bz_exp), (bx, by, bz))

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


def test_magpy_integration():
    import magpylib as magpy
    from atomsmltr.environment.fields.magnetic import MagpylibWrapper

    loop = magpy.current.Circle(current=1, diameter=1)
    mag_field = MagpylibWrapper(loop)
    mag_field.print_info()
    mag_field.value([0, 0, 0])

    _check_vector_field_value_function(mag_field.value)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test_magnetic_import()
    # test_magnetic_offset()
    # test_magnetic_gradient()
    test_magpy_integration()

    # plt.show()
