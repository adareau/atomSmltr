import pytest
import numpy as np
from numpy import pi


def _check_projections(pol):
    for target in ["V", "H", "R", "L"]:
        norm = pol.get_polarization_vector_projection_norm(target)
        proj = pol.get_polarization_vector_projection(target)
        proj_norm = np.linalg.norm(proj) ** 2
        try:
            assert np.allclose(norm, proj_norm)
        except AssertionError as e:
            print(f"Error for : {target=}")
            print(f"{pol.get_polarization_vector()=}")
            print(f"{pol.get_polarization_vector_angles()=}")
            print(f"{norm=}, {proj=}, {proj_norm=}")
            raise e


def test_polarization_properties():
    from atomsmltr.environment.lasers.polarization import Polarization

    # -- shorthands
    s2 = np.sqrt(2)
    s2inv = 1 / s2

    # -- check for different polarization states
    # -
    pol = Polarization("v")
    assert pol.type == "VERTICAL"
    assert np.allclose(pol.get_polarization_vector(), (1, 0, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, 0))
    _check_projections(pol)
    pol.display_info_string()
    # -
    pol = Polarization("x")
    assert pol.type == "VERTICAL"
    assert np.allclose(pol.get_polarization_vector(), (1, 0, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, 0))
    _check_projections(pol)
    # -
    pol = Polarization("y")
    assert pol.type == "HORIZONTAL"
    assert np.allclose(pol.get_polarization_vector(), (0, 1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 2))
    _check_projections(pol)
    # -
    pol = Polarization("h")
    assert pol.type == "HORIZONTAL"
    assert np.allclose(pol.get_polarization_vector(), (0, 1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 2))
    _check_projections(pol)
    # -
    pol = Polarization("R")
    assert pol.type == "CIRCULAR RIGHT"
    assert np.allclose(pol.get_polarization_vector(), (0, 0, 1))
    assert np.allclose(pol.get_polarization_vector_angles(), (0, 0))
    _check_projections(pol)
    # -
    pol = Polarization("L")
    assert pol.type == "CIRCULAR LEFT"
    assert np.allclose(pol.get_polarization_vector(), (0, 0, -1))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi, 0))
    _check_projections(pol)
    # -
    pol = Polarization("vec", vec=(1, 1, 1))
    _check_projections(pol)
    pol.display_info_string()
    # -
    pol = Polarization("vec", vec=(1, 1, 0))
    assert np.allclose(pol._vec, (s2inv, s2inv, 0))
    assert np.allclose(pol.get_polarization_vector(), (s2inv, s2inv, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 4))
    _check_projections(pol)
    # -
    pol = Polarization("lin", angle=pi / 4)
    assert np.allclose(pol.get_polarization_vector(), (s2inv, s2inv, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 4))
    _check_projections(pol)
    # -
    pol = Polarization("lin", angle=pi / 2)
    assert np.allclose(pol.get_polarization_vector(), (0, 1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, pi / 2))
    _check_projections(pol)
    # -
    pol = Polarization("lin", angle=-pi / 2)
    assert np.allclose(pol.get_polarization_vector(), (0, -1, 0))
    assert np.allclose(pol.get_polarization_vector_angles(), (pi / 2, -pi / 2))
    _check_projections(pol)
    pol.display_info_string()

    # -- check polarization setting exception
    with pytest.raises(ValueError) as excinfo:
        pol.type = "hum"
    # -
    with pytest.raises(ValueError) as excinfo:
        pol.type = 1
    # -
    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("linear")
    # -
    with pytest.raises(Warning) as excinfo:
        pol = Polarization("circular left", angle=5)
    # -
    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("circular left", angle="hum")
    # -
    with pytest.raises(Warning) as excinfo:
        pol = Polarization("circular left", vec=5)
    # -
    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("vector", vec=5)
    # -
    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("vector", vec=(1, 1))
    # -
    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("vector", vec=(0, 0, 0))
    # -
    pol = Polarization("H")
    pol.type = "vector"
    with pytest.raises(ValueError) as excinfo:
        pol.get_polarization_vector()
    # -
    pol = Polarization("H")
    pol.type = "linear"
    with pytest.raises(ValueError) as excinfo:
        pol.get_polarization_vector()


if __name__ == "__main__":
    test_polarization_properties()
