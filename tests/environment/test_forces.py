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


def _generic_force_test(force):
    new_tag = "may the force be with you"
    force.tag = new_tag
    assert force.tag == new_tag

    # - check value function behaviour
    _check_position_exceptions(force.value)
    check_vector_field_value_function(force.value)
    check_scalar_field_value_function(force.norm)


def test_constant_force():
    from atomsmltr.environment.fields.force import ConstantForce

    # -- exceptions
    with pytest.raises(ValueError) as excinfo:
        force = ConstantForce(0.5)
    with pytest.raises(TypeError) as excinfo:
        force = ConstantForce(("5", 1, 1))

    # -- good usage
    # - check init and info print
    offset = (1, 0, 0)
    force = ConstantForce(offset=offset, tag="offset")
    force.print_info()
    assert force.tag == "offset"
    _generic_force_test(force)

    # - check values
    # -
    assert np.allclose(force.value((1, 4, 8)), offset)
    # -
    new_offset = (4.5, 2.8, np.sqrt(2))
    force.offset = new_offset
    force.print_info()
    assert np.allclose(force.value((1, 4, 8)), new_offset)

    # - checking copy
    # init force
    force.tag = "old force"
    force.offset = (0, 0, 0)
    # init copy
    force_copy = force.copy()
    # check copied properties
    assert force_copy.tag != "old force"
    assert np.allclose(force_copy.offset, (0, 0, 0))
    # update and check old not affected
    force_copy.offset = (1, 2, 3)
    assert np.allclose(force_copy.offset, (1, 2, 3))
    assert np.allclose(force.offset, (0, 0, 0))
    force_copy = force.copy(new_tag="new force")
    assert force_copy.tag == "new force"


def test_force_gradient():
    from atomsmltr.environment.fields.force import GradientForce

    # -- generate a good field

    force = GradientForce(
        origin=(0, 0, 0),
        slope=0,
        gradient_direction=(1, 0, 0),
        field_direction=(0, 1, 1),
        offset=0,
        tag="gradient",
    )
    force.print_info()
    assert force.tag == "gradient"
    _generic_force_test(force)

    # -- check values
    # - (1) gradient along x, field at pi/4 wrt. y
    # settings
    force.slope = -2.0
    force.offset = 0.25
    force.gradient_direction = (1, 0, 0)
    force.field_direction = (0, 1, 1)

    # compute
    position = np.mgrid[-10:10:15j, -5:5:20j, -1:1:6j].T
    value = force.value(position)
    # check >> we make a loop, not efficient but this way we know what to expect
    X, Y, Z = position.T
    Bx, By, Bz = value.T
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    Bx, By, Bz = Bx.ravel(), By.ravel(), Bz.ravel()
    uf = force.field_direction / np.linalg.norm(force.field_direction)
    for x, y, z, bx, by, bz in zip(X, Y, Z, Bx, By, Bz):
        norm = x * force.slope + force.offset
        bx_exp, by_exp, bz_exp = uf * norm
        assert np.allclose((bx_exp, by_exp, bz_exp), (bx, by, bz))

    # check

    # gradient along z, field along x
    force.slope = 7.0
    force.offset = -8.0
    force.gradient_direction = (0, 0, 1)
    force.field_direction = (1, 0, 0)
    # compute
    position = np.mgrid[-10:10:15j, -5:5:20j, -1:1:6j].T
    value = force.value(position)
    # check >> we make a loop, not efficient but this way we know what to expect
    X, Y, Z = position.T
    Bx, By, Bz = value.T
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    Bx, By, Bz = Bx.ravel(), By.ravel(), Bz.ravel()
    uf = force.field_direction / np.linalg.norm(force.field_direction)
    for x, y, z, bx, by, bz in zip(X, Y, Z, Bx, By, Bz):
        norm = z * force.slope + force.offset
        bx_exp, by_exp, bz_exp = uf * norm
        assert np.allclose((bx_exp, by_exp, bz_exp), (bx, by, bz))

    # -- check exceptions
    with pytest.raises(ValueError) as excinfo:
        force.gradient_direction = 5.0
    with pytest.raises(TypeError) as excinfo:
        force.gradient_direction = ("5", 1, 1)
    with pytest.raises(ValueError) as excinfo:
        force.origin = 5.0
    with pytest.raises(TypeError) as excinfo:
        force.origin = ("5", 1, 1)
    with pytest.raises(ValueError) as excinfo:
        force.field_direction = 5.0
    with pytest.raises(TypeError) as excinfo:
        force.field_direction = ("5", 1, 1)
    with pytest.raises(ValueError) as excinfo:
        force.slope = [0, 1]
    with pytest.raises(TypeError) as excinfo:
        force.offset = "5"


if __name__ == "__main__":

    test_force_gradient()
    test_constant_force()
