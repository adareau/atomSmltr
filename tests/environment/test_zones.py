import pytest
import numpy as np
import matplotlib.pyplot as plt

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
    # - check invert
    grid = np.mgrid[-100:100:11, -100:100:12, -100:100:13]
    position = grid.T
    res = zone.in_zone(position)
    zone.invert()
    res_inv = zone.in_zone(position)
    assert np.all(np.logical_xor(res, res_inv))
    zone.inverted = False


def test_zones_collections():
    from atomsmltr.environment.zones import LowerLimit, UpperLimit

    low_x = LowerLimit(-5, 0, tag="lowx")
    up_x = UpperLimit(5, 0, tag="upx")
    low_y = LowerLimit(0, 1, tag="lowy")
    up_y = UpperLimit(10, 1, tag="upy")

    coll_1 = low_x + up_x
    _generic_zones_test(coll_1)

    coll_2 = coll_1 + low_y
    coll_2 += up_y
    # - plot
    if True:
        grid = np.mgrid[-10:15:100j, -20:20:101j, 0:0:1j]
        position = grid.T
        X, Y, _ = grid
        X = X.T
        Y = Y.T
        c1 = coll_1.in_zone(position)
        c2 = coll_2.in_zone(position)
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        c1 = np.squeeze(c1)
        c2 = np.squeeze(c2)
        fig, axes = plt.subplots(1, 2)
        axes[0].pcolormesh(X, Y, c1)
        axes[1].pcolormesh(X, Y, c2)
        plt.show()


def test_1D_zones():
    from atomsmltr.environment.zones import LowerLimit, UpperLimit

    # -- upper limit
    # init
    upper = UpperLimit(5, 0, tag="up")
    _generic_zones_test(upper)

    # few tests
    upper.value = 6.0
    assert upper.value == 6.0
    assert upper.in_zone((0, 0, 7))
    assert not upper.in_zone((7, 0, 5))
    assert upper.in_zone((4, -5, 7))
    assert not upper.in_zone((8, -8, 5))

    # -- lower limit
    # init
    lower = LowerLimit(5, 2)
    _generic_zones_test(lower)

    # few tests
    lower.value = 6.0
    assert lower.value == 6.0
    assert lower.in_zone((0, 0, 7))
    assert not lower.in_zone((0, 0, 5))
    assert lower.in_zone((4, -5, 7))
    assert not lower.in_zone((8, -8, 5))

    # check copy
    invlow = lower.inverted_copy()
    invlow.value = 0
    assert not lower.inverted
    assert invlow.inverted
    assert invlow.value == 0
    assert lower.value == 6.0


if __name__ == "__main__":
    test_1D_zones()
    test_zones_collections()
