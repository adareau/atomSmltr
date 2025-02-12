import pytest
import numpy as np


def test_magnetic_import():
    from atomsmltr.environment.fields import magnetic


def test_magnetic_offset():
    from atomsmltr.environment.fields.magnetic import MagneticGradient, MagneticOffset

    # -- exceptions
    with pytest.raises(ValueError) as excinfo:
        Bfield = MagneticOffset(0.5)
    with pytest.raises(TypeError) as excinfo:
        Bfield = MagneticOffset(("5", 1, 1))

    # -- good usage
    offset = (1, 0, 0)
    Bfield = MagneticOffset(offset=offset)
    Bfield.print_info()
    assert np.allclose(Bfield.value(1, 4, 8), offset)

    new_offset = (4.5, 2.8, np.sqrt(2))
    Bfield.offset = new_offset
    Bfield.print_info()
    assert np.allclose(Bfield.value(1, 4, 8), new_offset)


if __name__ == "__main__":
    test_magnetic_import()
    test_magnetic_offset()
