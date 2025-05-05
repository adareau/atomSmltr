import pytest
import numpy as np

from atomsmltr.utils.misc import (
    check_scalar_field_value_function,
    check_vector_field_value_function,
)

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


def _generic_magfield_test(mag_field):
    new_tag = "super magnet"
    mag_field.tag = new_tag
    assert mag_field.tag == new_tag

    # - check value function behaviour
    _check_position_exceptions(mag_field.get_value)
    check_vector_field_value_function(mag_field.get_value)
    check_scalar_field_value_function(mag_field.get_norm)


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
    mag_field = MagneticOffset(field_value=offset, tag="offset")
    mag_field.print_info()
    assert mag_field.tag == "offset"
    _generic_magfield_test(mag_field)

    # - check values
    # -
    assert np.allclose(mag_field.get_value((1, 4, 8)), offset)
    # -
    new_offset = (4.5, 2.8, np.sqrt(2))
    mag_field.field_value = new_offset
    mag_field.print_info()
    assert np.allclose(mag_field.get_value((1, 4, 8)), new_offset)

    # - checking copy
    # init mag_field
    mag_field.tag = "old mag_field"
    mag_field.field_value = (0, 0, 0)
    # init copy
    mag_field_copy = mag_field.copy()
    # check copied properties
    assert mag_field_copy.tag != "old mag_field"
    assert np.allclose(mag_field_copy.field_value, (0, 0, 0))
    # update and check old not affected
    mag_field_copy.field_value = (1, 2, 3)
    assert np.allclose(mag_field_copy.field_value, (1, 2, 3))
    assert np.allclose(mag_field.field_value, (0, 0, 0))
    mag_field_copy = mag_field.copy(new_tag="new mag_field")
    assert mag_field_copy.tag == "new mag_field"


def test_magnetic_gradient():
    from atomsmltr.environment.fields.magnetic import MagneticGradient

    # -- generate a good field

    mag_field = MagneticGradient(
        origin=(0, 0, 0),
        slope=0,
        gradient_direction=(1, 0, 0),
        field_direction=(0, 1, 1),
        offset=0,
        tag="gradient",
    )
    mag_field.print_info()
    assert mag_field.tag == "gradient"
    _generic_magfield_test(mag_field)

    # -- check values
    # - (1) gradient along x, field at pi/4 wrt. y
    # settings
    mag_field.slope = -2.0
    mag_field.offset = 0.25
    mag_field.gradient_direction = (1, 0, 0)
    mag_field.field_direction = (0, 1, 1)

    # compute
    position = np.mgrid[-10:10:15j, -5:5:20j, -1:1:6j].T
    value = mag_field.get_value(position)
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
    value = mag_field.get_value(position)
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
    from atomsmltr.environment.fields.magnetic.magpylib import MagpylibWrapper

    loop = magpy.current.Circle(current=1, diameter=1)
    mag_field = MagpylibWrapper(loop, tag="wrapped")
    mag_field.print_info()
    assert mag_field.tag == "wrapped"
    _generic_magfield_test(mag_field)

    mag_field.get_value([0, 0, 0])


def test_magnetic_quadrupole():
    from atomsmltr.environment.fields.magnetic import (
        MagneticQuadrupoleX,
        MagneticQuadrupoleZ,
        MagneticQuadrupoleY,
    )

    # -- Quadrupole X
    # init
    mag_field = MagneticQuadrupoleX(
        origin=(0, 0, 0),
        slope=0.5,
    )
    mag_field.print_info()
    # basic tests
    _generic_magfield_test(mag_field)
    # value test
    slope = mag_field.slope
    for x in [-5, 8, -9, 7]:
        for y in [0, 1, 2, 3, -9]:
            for z in [-8, 8, 6, 10]:
                B = mag_field.get_value((x, y, z))
                B_exp = (-2 * slope * x, slope * y, slope * z)
                assert np.allclose(B, B_exp)

    # -- Quadrupole Y
    # init
    mag_field = MagneticQuadrupoleY(
        origin=(0, 0, 0),
        slope=0.5,
    )
    mag_field.print_info()
    # basic tests
    _generic_magfield_test(mag_field)
    _check_position_exceptions(mag_field.get_value)
    check_vector_field_value_function(mag_field.get_value)
    # value test
    slope = mag_field.slope
    for x in [-5, 8, -9, 7]:
        for y in [0, 1, 2, 3, -9]:
            for z in [-8, 8, 6, 10]:
                B = mag_field.get_value((x, y, z))
                B_exp = (slope * x, -2 * slope * y, slope * z)
                assert np.allclose(B, B_exp)

    # -- Quadrupole Z
    # init
    mag_field = MagneticQuadrupoleZ(
        origin=(0, 0, 0),
        slope=0.66,
    )
    mag_field.print_info()
    # basic tests
    _generic_magfield_test(mag_field)
    _check_position_exceptions(mag_field.get_value)
    check_vector_field_value_function(mag_field.get_value)
    # value test
    slope = mag_field.slope
    for x in [-5, 8, -9, 7]:
        for y in [0, 1, 2, 3, -9]:
            for z in [-8, 8, 6, 10]:
                B = mag_field.get_value((x, y, z))
                B_exp = (slope * x, slope * y, -2 * slope * z)
                assert np.allclose(B, B_exp)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test_magnetic_import()
    test_magnetic_offset()
    # test_magnetic_gradient()
    # test_magpy_integration()
    # test_magnetic_quadrupole()

    # plt.show()
