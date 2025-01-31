import pytest
import numpy as np


def test_polarization_properties():
    from atomsmltr.environment.lasers.polarization import Polarization

    # -- check some initializations
    # a
    pol = Polarization("v")
    assert pol.type == "VERTICAL"
    # b
    pol = Polarization("x")
    assert pol.type == "VERTICAL"
    # c
    pol = Polarization("y")
    assert pol.type == "HORIZONTAL"
    # d
    pol = Polarization("R")
    assert pol.type == "CIRCULAR RIGHT"
    # e
    pol = Polarization("L")
    assert pol.type == "CIRCULAR LEFT"
    # f
    pol = Polarization("vec", vec=(1, 1, 0))
    assert np.allclose(pol._vec, (1 / np.sqrt(2), 1 / np.sqrt(2), 0))

    # -- check polarization setting exception
    with pytest.raises(ValueError) as excinfo:
        pol.type = "hum"
    with pytest.raises(ValueError) as excinfo:
        pol.type = 1

    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("linear")

    with pytest.raises(Warning) as excinfo:
        pol = Polarization("circular left", angle=5)

    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("circular left", angle="hum")

    with pytest.raises(Warning) as excinfo:
        pol = Polarization("circular left", vec=5)

    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("vector", vec=5)

    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("vector", vec=(1, 1))

    with pytest.raises(ValueError) as excinfo:
        pol = Polarization("vector", vec=(0, 0, 0))


if __name__ == "__main__":
    test_polarization_properties()
