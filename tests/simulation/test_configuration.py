import pytest


def test_configuration():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium

    # -- init config
    config = Configuration()

    # -- set atom
    config.atom = Ytterbium()


def test_configuration_exceptions():
    from atomsmltr.simulation import Configuration

    config = Configuration()

    with pytest.raises(TypeError) as excinfo:
        config.atom = "ytterbium"


if __name__ == "__main__":
    test_configuration()
    test_configuration_exceptions()
