import pytest


def test_import_example_module():
    # Check imports
    import atomsmltr.examples.chen2021 as chen2021

    assert hasattr(chen2021, "config_1D_MOT")
    assert hasattr(chen2021, "config_1D_molasses")


def test_config_1D_MOT_structure():
    from atomsmltr.examples.chen2021 import config_1D_MOT

    # Check if the atom is strontium
    assert config_1D_MOT.atom.name.lower() == "strontium"

    # Vérifie qu'il y a au moins un laser et un champ magnétique
    objs = config_1D_MOT.objects

    # compter les lasers
    num_lasers = len(objs["laser"])
    num_mags = len(objs["magnetic field"])

    assert num_lasers == 2, f"Il devrait y avoir 2 lasers, trouvé {num_lasers}"
    assert num_mags == 1, f"Il devrait y avoir 1 champ magnétique, trouvé {num_mags}"


def test_config_1D_molasses_structure():
    from atomsmltr.examples.chen2021 import config_1D_molasses

    # Vérifie que l'atome est un Rubidium
    assert config_1D_molasses.atom.name.lower() == "rubidium"

    # Vérifie qu'il y a au moins deux lasers
    objs = config_1D_molasses.objects
    num_lasers = len(objs["laser"])
    assert num_lasers == 2, f"Il devrait y avoir 2 lasers, trouvé {num_lasers}"
