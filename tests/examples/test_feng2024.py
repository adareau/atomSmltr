import pytest


def test_import_example_module():

    # Check that the module imports correctly
    import atomsmltr.examples.feng2024 as feng2024

    assert hasattr(feng2024, "config_symmetric_field")
    assert hasattr(feng2024, "config_asymmetric_field_1")
    assert hasattr(feng2024, "config_asymmetric_field_2")


def test_config_atom():

    from atomsmltr.examples.feng2024 import config_symmetric_field

    # Check that the atom is strontium
    assert config_symmetric_field.atom.name.lower() == "strontium"


def test_config_contains_lasers_and_magnets():

    from atomsmltr.examples.feng2024 import config_symmetric_field

    # Get the objects in the configuration
    objs = config_symmetric_field.objects

    # Count the lasers in the configuration
    num_lasers = len(objs["laser"])
    num_magnetic_fields = len(objs["magnetic field"])

    # Check that there are exactly 6 lasers and 1 magnetic field
    assert num_lasers == 5
    assert num_magnetic_fields == 1
