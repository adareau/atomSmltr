import pytest


def test_import_example_module():
    import atomsmltr.examples.chen2021 as chen2021

    # Check imports
    assert hasattr(chen2021, "config_1D_MOT")
    assert hasattr(chen2021, "config_1D_molasses")


def test_config_1D_MOT_structure():
    from atomsmltr.examples.chen2021 import config_1D_MOT

    # Check if the atom is strontium
    assert config_1D_MOT.atom.name.lower() == "strontium"

    # Get the objects in the configuration
    objs = config_1D_MOT.objects

    # Count the lasers and the magnetic fields in the configuration
    num_lasers = len(objs["laser"])
    num_mags = len(objs["magnetic field"])

    # Check that there are exactly 2 lasers and 1 magnetic field
    assert num_lasers == 2, f"Il devrait y avoir 2 lasers, trouvé {num_lasers}"
    assert num_mags == 1, f"Il devrait y avoir 1 champ magnétique, trouvé {num_mags}"


def test_config_1D_molasses_structure():
    from atomsmltr.examples.chen2021 import config_1D_molasses

    # Check if the atom is rubidium
    assert config_1D_molasses.atom.name.lower() == "rubidium"

    # Get the objects in the configuration
    objs = config_1D_molasses.objects

    # Count the lasers in the configuration
    num_lasers = len(objs["laser"])

    # Check that there are exactly 2 laser
    assert num_lasers == 2, f"Il devrait y avoir 2 lasers, trouvé {num_lasers}"


def test_config_3D_MOT_structure():
    from atomsmltr.examples.chen2021 import config_3D_MOT

    # Check if the atom is rubidium
    assert config_3D_MOT.atom.name.lower() == "rubidium"

    # Get the objects in the configuration
    objs = config_3D_MOT.objects

    # Count the lasers and the magnetic fields in the configuration
    num_lasers = len(objs["laser"])
    num_mags = len(objs["magnetic field"])

    # Check that there are exactly 6 lasers and 1 magnetic field
    assert num_lasers == 6, f"Il devrait y avoir 6 lasers, trouvé {num_lasers}"
    assert num_mags == 1, f"Il devrait y avoir 1 champ magnétique, trouvé {num_mags}"
