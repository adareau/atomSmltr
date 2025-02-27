import pytest
import numpy as np

from atomsmltr.utils.misc import (
    check_scalar_field_value_function,
    check_vector_field_value_function,
)


def _get_env_objects():
    from atomsmltr.environment.fields.magnetic import MagneticOffset
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    mag_field_1 = MagneticOffset((0, 1, 1), tag="offset1")
    mag_field_2 = MagneticOffset((0, 0, 2), tag="offset2")
    laser_1 = GaussianLaserBeam(
        780e-9, 20e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="laser1"
    )
    laser_2 = GaussianLaserBeam(
        780e-9, 20e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="laser2"
    )
    return mag_field_1, mag_field_2, laser_1, laser_2


def test_configuration_collection_management():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium

    # -- init config
    config = Configuration()

    # -- set atom
    config.atom = Ytterbium()

    # -- set environment
    mag1, mag2, las1, las2 = _get_env_objects()
    config.add_objects(mag1, verbose=True)
    config.add_objects(las1, verbose=True)
    config.add_objects([mag2, las2], verbose=True)

    # -- getting a copy of all objects
    obj = config.objects

    # -- listing and removing
    for tag in config.list_lasers():
        config.rm_object("laser", tag)
    assert len(config.list_lasers()) == 0

    for tag in config.list_magnetic_fields():
        config.rm_object("magnetic field", tag)
    assert len(config.list_magnetic_fields()) == 0

    config.add_objects(mag1)
    config.rm_magnetic_field("offset1")
    assert len(config.list_magnetic_fields()) == 0

    config.add_objects(las1)
    config.rm_laser("laser1")
    assert len(config.list_lasers()) == 0

    config.add_objects([mag1, mag2, las1, las2])
    config.rm_all_objects()
    assert len(config.list_magnetic_fields()) == 0
    assert len(config.list_lasers()) == 0

    config.add_objects([mag1, mag2, las1, las2])
    config.rm_all_magnetic_fields()
    assert len(config.list_magnetic_fields()) == 0
    assert len(config.list_lasers()) == 2
    config.add_objects([mag1, mag2])
    config.rm_all_lasers()
    assert len(config.list_magnetic_fields()) == 2
    assert len(config.list_lasers()) == 0

    # -- updating
    config.rm_all_objects()
    las1.direction = [1, 0, 0]
    mag1.offset = [1, 1, 1]
    config.add_objects([las1, mag1])
    config.print_magnetic_field_info("offset1")
    mag1.offset = [2, 1, 1]
    config.update_objects(mag1, verbose=True)
    mag3 = config.get_magnetic_field_copy("offset1")
    assert np.allclose(mag3.offset, mag1.offset)

    # check that a copy is indeed given
    mag3.offset = [0, 0, 1]
    offset = config.get_magnetic_field_copy("offset1").offset
    assert np.allclose(offset, mag1.offset)

    # check copy is also given in entry
    config.rm_all_objects()
    las1.direction = [1, 0, 0]
    config.add_objects(las1)
    las1.direction = [0, 0, 1]
    config.print_laser_info("laser1")
    direction = config.get_laser_copy("laser1").direction
    assert np.allclose(direction, [1, 0, 0])

    # check warning behaviour
    config.rm_all_objects()
    las1.tag = "las1"
    las2.tag = "las2"
    las1.direction = [1, 0, 0]
    config.add_objects(las1)
    las1.direction = [0, 1, 0]
    config.update_objects(
        [las2, las1], verbose=True, error_on_fail=False
    )  # should issue a warning but go on

    assert np.allclose(config.get_laser_copy("las1").direction, [0, 1, 0])

    with pytest.raises(KeyError) as excinfo:
        config.update_objects(
            [las2, las1], verbose=False, error_on_fail=True
        )  # should issue a warning but go on


def test_configuration_atom_light():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    # -- init config
    config = Configuration()

    # -- set atom
    config.atom = Ytterbium()
    config.atom.print_info()

    # -- lasers
    laser399_1 = GaussianLaserBeam(
        399e-9, 100e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="399-1"
    )
    laser399_2 = GaussianLaserBeam(
        399e-9, 100e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="399-2"
    )
    laser556 = GaussianLaserBeam(556e-9, 100e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="556")

    # -- Config tests
    # - init
    config.add_objects([laser399_1, laser399_2, laser556])
    # - key errors
    with pytest.raises(KeyError) as excinfo:
        config.add_atomlight_coupling("557", "intercombination", 0, True)
    with pytest.raises(KeyError) as excinfo:
        config.add_atomlight_coupling("556", "D1", 0, True)

    # - add
    config.add_atomlight_coupling("556", "intercombination", 0, True)
    config.add_atomlight_coupling(laser399_1, "main", 0, True)
    with pytest.raises(KeyError) as excinfo:
        config.add_atomlight_coupling(laser399_1, "main", 0, True)
    config.add_atomlight_coupling(laser399_1, "main", -2, override=True, verbose=True)
    config.print_atomlight_info()

    # - remove
    config.rm_atomlight_coupling("556", "intercombination")
    config.print_atomlight_info()

    with pytest.raises(KeyError) as excinfo:
        config.rm_atomlight_coupling("556", "intercombination")
    config.reset_atomlight_coupling()
    config.print_atomlight_info()


def test_configuration_exceptions():
    from atomsmltr.simulation import Configuration

    # -- init
    config = Configuration()

    # - check atom exception
    with pytest.raises(TypeError) as excinfo:
        config.atom = "ytterbium"

    # - check environement
    mag1, mag2, las1, las2 = _get_env_objects()
    # adding wrong types
    with pytest.raises(TypeError) as excinfo:
        config.add_objects("laser")
    with pytest.raises(TypeError) as excinfo:
        config.add_objects([mag1, mag2, "laser"])
    config.add_objects(mag1)
    # try to add object with already existing tag
    with pytest.raises(ValueError) as excinfo:
        config.add_objects(mag1)
    # try to remove object from wrong collection
    with pytest.raises(ValueError) as excinfo:
        config.rm_object("MAgnetic fields", "mag1")
    # try to add object with already existing tag
    with pytest.raises(KeyError) as excinfo:
        config.rm_laser("las3")


def test_configuration_print_info():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam
    from atomsmltr.environment.fields.magnetic import MagneticGradient, MagneticOffset

    # init config
    config = Configuration()

    # set atom
    config.atom = Ytterbium()

    # set lasers
    laser399_1 = GaussianLaserBeam(
        399e-9, 100e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="399-1"
    )
    laser399_2 = GaussianLaserBeam(
        399e-9, 100e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="399-2"
    )
    laser556 = GaussianLaserBeam(556e-9, 100e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="556")

    # set magnetic fields
    mag_offset = MagneticOffset([0.1, 0, 0.4], tag="compensation")
    mag_gradient = MagneticGradient(
        [0, 0, 0], 1.0, [0, 0, 1], [0, 1, 0], tag="gradient"
    )

    # add
    config.add_objects([laser399_1, laser399_2, laser556, mag_offset, mag_gradient])

    # setup atom-light
    config.add_atomlight_coupling("399-1", "main", detuning=0)
    config.add_atomlight_coupling("399-2", "main", detuning=-4)
    config.add_atomlight_coupling("556", "intercombination", 0)

    # print info
    config.print_info()


def test_configuration_methods():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium, Strontium

    # -- init
    mag_field_1, mag_field_2, laser_1, laser_2 = _get_env_objects()
    conf = Configuration(atom=Ytterbium())
    conf.add_objects([mag_field_1, laser_1, mag_field_2, laser_2])

    check_scalar_field_value_function(conf.getBnorm)
    check_vector_field_value_function(conf.getB)


def test_configuration_operators():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium, Strontium

    # -- init
    mag_field_1, mag_field_2, laser_1, laser_2 = _get_env_objects()
    conf = Configuration(atom=Ytterbium())
    conf.add_objects([mag_field_1, laser_1])
    # -- addition
    # perform operations
    conf2 = conf + laser_2
    conf2.atom = Strontium()
    conf3 = conf + [laser_2, mag_field_2]
    # check
    assert isinstance(conf.atom, Ytterbium)
    assert isinstance(conf2.atom, Strontium)
    assert isinstance(conf3.atom, Ytterbium)
    assert "laser2" not in conf.list_lasers()
    assert "laser2" in conf2.list_lasers()
    assert "laser1" in conf2.list_lasers()
    assert "offset2" not in conf2.list_magnetic_fields()
    assert "laser2" in conf3.list_lasers()
    assert "laser1" in conf3.list_lasers()
    assert "offset2" in conf3.list_magnetic_fields()

    # -- increment
    conf += laser_2
    assert "laser2" in conf.list_lasers()
    conf.rm_all_lasers()
    conf += laser_1, laser_2
    assert "laser1" in conf.list_lasers()
    assert "laser2" in conf.list_lasers()


if __name__ == "__main__":
    # test_configuration_collection_management()
    # test_configuration_exceptions()
    # test_configuration_atom_light()
    # test_configuration_print_info()
    test_configuration_operators()
    test_configuration_methods()
