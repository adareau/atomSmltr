import pytest


def test_import_example_module():

    # Check that the module imports correctly
    import atomsmltr.examples.feng2024 as feng2024

    assert hasattr(feng2024, "config")


def test_config_atom():

    from atomsmltr.examples.feng2024 import config

    # Check that the atom is strontium
    assert config.atom.name.lower() == "strontium"


def test_config_contains_lasers_and_magnets():

    from atomsmltr.examples.feng2024 import config

    # Get the objects in the configuration
    objs = config.objects

    # Count the lasers in the configuration
    num_lasers = len(objs["laser"])
    num_magnetic_fields = len(objs["magnetic field"])

    # Check that there are exactly 6 lasers and 1 magnetic field
    assert num_lasers == 6
    assert num_magnetic_fields == 1


def test_config_has_limits():
    from atomsmltr.examples.feng2024 import config

    # Get the objects in the configuration
    objs = config.objects

    # Count the different zones in the configuration
    num_zones = len(objs["zone"])

    # Check that there is exactly one zone
    assert num_zones == 1
