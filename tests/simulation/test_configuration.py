import pytest
import numpy as np

from atomsmltr.utils.misc import (
    check_scalar_field_value_function,
    check_vector_field_value_function,
)


def _get_env_objects():
    from atomsmltr.environment.fields.magnetic import MagneticOffset
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam
    from atomsmltr.environment.zones import UpperLimit, LowerLimit
    from atomsmltr.environment.fields.force import ConstantForce

    mag_field_1 = MagneticOffset((0, 1, 1), tag="offset1")
    mag_field_2 = MagneticOffset((0, 0, 2), tag="offset2")
    laser_1 = GaussianLaserBeam(
        780e-9, 20e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="laser1"
    )
    laser_2 = GaussianLaserBeam(
        780e-9, 20e-6, 10e-3, (0, 0, 0), (0, 0, 1), tag="laser2"
    )
    lim_up = UpperLimit(0.5, axis=0, target="position", action="stop", tag="x_max")
    lim_low = LowerLimit(0, axis=0, target="position", action="stop", tag="x_min")
    force = ConstantForce((0, 0, 1), tag="force")
    return mag_field_1, mag_field_2, laser_1, laser_2, lim_up, lim_low, force


def test_configuration_collection_management():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium

    # -- init config
    config = Configuration()

    # -- set atom
    config.atom = Ytterbium()

    # -- set environment
    mag1, mag2, las1, las2, lim1, lim2, force = _get_env_objects()
    config.add_objects(mag1, verbose=True)
    config.add_objects(las1, verbose=True)
    config.add_objects([mag2, las2], verbose=True)
    config += lim1, lim2, force

    # -- getting a copy of all objects
    obj = config.objects

    # -- listing and removing
    for tag in config.list_lasers():
        config.rm_object("laser", tag)
    assert len(config.list_lasers()) == 0

    for tag in config.list_magnetic_fields():
        config.rm_object("magnetic field", tag)
    assert len(config.list_magnetic_fields()) == 0

    for tag in config.list_zones():
        config.rm_object("zone", tag)
    assert len(config.list_zones()) == 0

    assert len(config.list_forces()) == 1
    for tag in config.list_forces():
        config.rm_object("force", tag)
    assert len(config.list_forces()) == 0

    config.add_objects(mag1)
    config.rm_magnetic_field("offset1")
    assert len(config.list_magnetic_fields()) == 0

    config.add_objects(las1)
    config.rm_laser("laser1")
    assert len(config.list_lasers()) == 0

    config.add_objects([mag1, mag2, las1, las2, lim1, force])
    config.rm_all_objects()
    assert len(config.list_magnetic_fields()) == 0
    assert len(config.list_lasers()) == 0
    assert len(config.list_zones()) == 0
    assert len(config.list_forces()) == 0

    config.add_objects([mag1, mag2, las1, las2, lim1, lim2, force])
    config.rm_all_magnetic_fields()
    assert len(config.list_magnetic_fields()) == 0
    assert len(config.list_lasers()) == 2
    assert len(config.list_zones()) == 2
    assert len(config.list_forces()) == 1
    config.add_objects([mag1, mag2])
    config.rm_all_lasers()
    config.rm_all_zones()
    assert len(config.list_magnetic_fields()) == 2
    assert len(config.list_lasers()) == 0
    assert len(config.list_zones()) == 0
    assert len(config.list_forces()) == 1

    # -- updating
    config.rm_all_objects()
    las1.direction = [1, 0, 0]
    mag1.field_value = [1, 1, 1]
    force.field_value = [0, 0, 0]
    config.add_objects([las1, mag1, force])
    config.print_magnetic_field_info("offset1")
    mag1.field_value = [2, 1, 1]
    force.field_value = [1, 1, 1]
    config.update_objects([mag1, force], verbose=True)
    mag3 = config.get_magnetic_field_copy("offset1")
    force2 = config.get_force_copy("force")
    assert np.allclose(mag3.field_value, mag1.field_value)
    assert np.allclose(force2.field_value, force.field_value)

    # check that a copy is indeed given
    mag3.field_value = [0, 0, 1]
    offset = config.get_magnetic_field_copy("offset1").field_value
    assert np.allclose(offset, mag1.field_value)

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
    mag1, mag2, las1, las2, lim1, lim2, force = _get_env_objects()
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
    from atomsmltr.environment.zones import LowerLimit

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

    # set zone
    zone = LowerLimit(0, axis=1, tag="y_min", target="position")
    # add
    config.add_objects(
        [laser399_1, laser399_2, laser556, mag_offset, mag_gradient, zone]
    )

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
    mag_field_1, mag_field_2, laser_1, laser_2, lim_up, lim_low, force = (
        _get_env_objects()
    )
    lim_up.target = "position"
    lim_low.target = "speed"

    conf = Configuration(atom=Ytterbium())
    conf.add_objects([mag_field_1, laser_1, mag_field_2, laser_2, lim_low, lim_up])
    check_scalar_field_value_function(conf.getBnorm)
    check_vector_field_value_function(conf.getB)

    # check stop zones getter
    pos, speed = conf.get_stop_zones()
    assert len(pos) == len(speed) == 1
    assert pos[0].tag == "x_max"
    assert speed[0].tag == "x_min"


def test_configuration_operators():
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium, Strontium

    # -- init
    mag_field_1, mag_field_2, laser_1, laser_2, lim_up, lim_low, force = (
        _get_env_objects()
    )
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
    conf += lim_up, lim_low
    assert "x_min" in conf.list_zones()
    assert "x_max" in conf.list_zones()


def test_configuration_inzone():
    from atomsmltr.simulation import Configuration
    from atomsmltr.environment import Limits

    # create limits
    x_lim = Limits(-1, 1, axis=0, target="position", tag="xlim")
    y_lim = Limits(-10, 10, axis=1, target="position", tag="ylim")
    vx_lim = Limits(0, 100, axis=0, target="speed", tag="vxlim")

    # init config
    config = Configuration()
    config += x_lim, y_lim, vx_lim

    # test
    assert config.get_value((0, 0, 0, 1, 0, 0))
    assert config.get_value((0, 0, 90, 1, 0, 0))
    assert config.get_value((0, 0, 90, 1, 6, 8))
    assert config.get_value((0.5, 5, 90, 50, 6, 8))
    assert not config.get_value((2, 5, 90, 50, 6, 8))
    assert not config.get_value((0, 5, 2, -10, 6, 8))
    assert not config.get_value((0, 11, 0, 5, 6, 8))


if __name__ == "__main__":
    # test_configuration_collection_management()
    # test_configuration_exceptions()
    # test_configuration_atom_light()
    # test_configuration_print_info()
    # test_configuration_operators()
    test_configuration_methods()
    # test_configuration_inzone()
    pass
