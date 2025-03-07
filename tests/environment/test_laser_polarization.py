import pytest
import numpy as np
from numpy import pi

# -- shorthands

S2 = np.sqrt(2)
S2INV = 1 / S2

# -- common testing functions


def _check_projections(pol):
    for target in ["V", "H", "R", "L"]:
        norm = pol.get_polarization_vector_projection_norm(target)
        proj = pol.get_polarization_vector_projection(target)
        proj_norm = np.linalg.norm(proj) ** 2
        try:
            assert np.allclose(norm, proj_norm)
        except AssertionError as e:
            print(f"Error for : {target=}")
            print(f"{pol.vector=}")
            print(f"{pol.get_polarization_vector_angles()=}")
            print(f"{norm=}, {proj=}, {proj_norm=}")
            raise e


def test_special_polarizations():
    from atomsmltr.environment.lasers.polarization import (
        Vertical,
        Horizontal,
        CircularLeft,
        CircularRight,
    )

    # - VERTICAL
    pol = Vertical()
    assert np.allclose(pol.vector, (1, 0, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, 0))
    _check_projections(pol)
    pol.print_info()

    # - HORIZONTAL
    pol = Horizontal()
    assert np.allclose(pol.vector, (0, 1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 2))
    _check_projections(pol)
    pol.print_info()

    # - CIRCULAR RIGHT
    pol = CircularRight()
    assert np.allclose(pol.vector, (0, 0, 1))
    assert np.allclose(pol.get_polarization_vector_angles(), (0, 0))
    _check_projections(pol)
    pol.print_info()

    # - CIRCULAR LEFT
    pol = CircularLeft()
    assert np.allclose(pol.vector, (0, 0, -1))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi, 0))
    _check_projections(pol)
    pol.print_info()


def test_linear_polarization():
    from atomsmltr.environment.lasers.polarization import Linear

    # -- Check init settings
    pol = Linear(pi / 4)
    assert np.allclose(pol.vector, (S2INV, S2INV, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 4))
    _check_projections(pol)
    # -
    pol = Linear(pi / 2)
    assert np.allclose(pol.vector, (0, 1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 2))
    _check_projections(pol)
    # -
    pol = Linear(-pi / 2)
    assert np.allclose(pol.vector, (0, -1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, -pi / 2))
    _check_projections(pol)
    pol.print_info()

    # -- Check angle setter
    pol = Linear(pi / 2)
    pol.angle = 0
    assert np.allclose(pol.vector, (1, 0, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, 0))
    _check_projections(pol)

    # -- Check exception
    with pytest.raises(ValueError):
        pol = Linear(None)
    # -
    pol = Linear(0)
    with pytest.raises(ValueError):
        pol.angle = (2,)


def test_vector_polarization():
    from atomsmltr.environment.lasers.polarization import Vector

    # -- Check init settings
    pol = Vector((1, 1, 1))
    _check_projections(pol)
    pol.print_info()

    # - Check setter
    pol = Vector((1, 0, 0))
    pol.vector = (1, 1, 0)
    pol.print_info()
    assert np.allclose(pol.vector, (S2INV, S2INV, 0))
    assert np.allclose(pol.vector, (S2INV, S2INV, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 4))
    _check_projections(pol)

    # -- Check exception
    with pytest.raises(ValueError):
        pol = Vector(None)
    # -
    pol = Vector((1, 0, 0))
    with pytest.raises(ValueError):
        pol.vector = (1, 0)
    with pytest.raises(ValueError):
        pol.vector = (0, 0, 0)


if __name__ == "__main__":
    test_special_polarizations()
    test_linear_polarization()
    test_vector_polarization()
