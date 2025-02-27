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


def _generic_zones_test(zone):
    # - tags
    new_tag = "super zone"
    zone.tag = new_tag
    assert zone.tag == new_tag
    zone.print_info()
    # - invert
    zone.inverted = True
    assert zone.inverted
    zone.invert()
    assert not zone.inverted
    # - check value function behaviour
    _check_position_exceptions(zone.in_zone)
    check_scalar_field_value_function(zone.in_zone)


def test_1D_zones():
    from atomsmltr.environment.zones import LowerLimit, UpperLimit

    # -- upper limit
    # init
    upper = UpperLimit(5, 0)
    _generic_zones_test(upper)

    # few tests
    upper.value = 6.0
    assert upper.value == 6.0


if __name__ == "__main__":
    test_1D_zones()
