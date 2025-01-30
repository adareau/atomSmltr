import pytest
import numpy as np


# % GENERIC TESTS


def _LaserBeam_classes_generic_properties_test(LaserBeamClass):

    # - testing initialization
    beam = LaserBeamClass(
        wavelength=780e-9,
        waist=20e-6,
        power=50e-3,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
    )

    # - testing setters and getters
    # wavelength
    new_wavelength = 781e-9
    beam.wavelength = new_wavelength
    assert beam.wavelength == new_wavelength

    # waist
    new_waist = 30e-6
    beam.waist = new_waist
    assert beam.waist == new_waist

    # power
    new_power = 60e-3
    beam.power = new_power
    assert beam.power == new_power

    # waist_position > array
    new_waist_position = np.array([1.0, 1.0, 1.0])
    beam.waist_position = new_waist_position
    assert np.array_equal(beam.waist_position, new_waist_position)

    # waist_position > tuple
    new_waist_position = (1.0, 1.0, 1.0)
    beam.waist_position = new_waist_position
    assert np.array_equal(beam.waist_position, new_waist_position)

    # direction_type
    new_direction_type = "thetaphi"
    beam.direction_type = new_direction_type
    assert beam.direction_type == new_direction_type

    # direction > thetaphi
    beam.direction_type = "thetaphi"
    new_direction = [0.5, 3]
    beam.direction = new_direction
    assert np.array_equal(beam.direction, new_direction)
    # check that unit vector is well normalized
    assert np.linalg.norm(beam._unit_vector) == 1.0

    # direction > vector
    beam.direction_type = "vector"
    new_direction = [0.1, 0.1, 0.1]
    beam.direction = new_direction
    assert np.array_equal(beam.direction, new_direction)
    # check that unit vector is well normalized
    assert np.linalg.norm(beam._unit_vector) == 1.0
    # check that unit vector is colinear to new_direction
    assert np.allclose(np.cross(new_direction, beam._unit_vector), np.zeros(3))

    # check internal conversion thetaphi > vector
    beam.direction_type = "thetaphi"
    # 1
    new_direction = [0.25 * np.pi, 0]
    expected_unit_vector = [np.sqrt(0.5), 0, np.sqrt(0.5)]
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector, expected_unit_vector)
    # 2
    new_direction = [-0.25 * np.pi, 0]
    expected_unit_vector = [-np.sqrt(0.5), 0, np.sqrt(0.5)]
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector, expected_unit_vector)
    # 3
    new_direction = [0.25 * np.pi, 0.25 * np.pi]
    expected_unit_vector = [np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.5)]
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector, expected_unit_vector)

    # check internal conversion vector > thetaphi
    beam.direction_type = "vector"
    # 1
    new_direction = [1, 0, 0]
    expected_theta = 0.5 * np.pi
    expected_phi = 0
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector_phi, expected_phi)
    assert np.allclose(beam._unit_vector_theta, expected_theta)
    # 2
    new_direction = [1, 1, 0]
    expected_theta = 0.5 * np.pi
    expected_phi = 0.25 * np.pi
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector_phi, expected_phi)
    assert np.allclose(beam._unit_vector_theta, expected_theta)
    # 3
    new_direction = [1, -1, 0]
    expected_theta = 0.5 * np.pi
    expected_phi = -0.25 * np.pi
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector_phi, expected_phi)
    assert np.allclose(beam._unit_vector_theta, expected_theta)
    # 4
    new_direction = [np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.5)]
    expected_theta = 0.25 * np.pi
    expected_phi = 0.25 * np.pi
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector_phi, expected_phi)
    assert np.allclose(beam._unit_vector_theta, expected_theta)
    # 5
    new_direction = [np.sqrt(0.25), np.sqrt(0.25), -np.sqrt(0.5)]
    expected_theta = 0.75 * np.pi
    expected_phi = 0.25 * np.pi
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector_phi, expected_phi)
    assert np.allclose(beam._unit_vector_theta, expected_theta)
    # 6
    new_direction = [-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.5)]
    expected_theta = 0.75 * np.pi
    expected_phi = -0.75 * np.pi
    beam.direction = new_direction
    assert np.allclose(beam._unit_vector_phi, expected_phi)
    assert np.allclose(beam._unit_vector_theta, expected_theta)


def _LaserBeam_classes_generic_exception_test(LaserBeamClass):
    # - init
    beam = LaserBeamClass(
        wavelength=780e-9,
        waist=20e-6,
        power=50.0,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
    )

    # - wrong values for setters
    # wavelength
    with pytest.raises(ValueError) as excinfo:
        beam.wavelength = "451"
    with pytest.raises(ValueError) as excinfo:
        beam.wavelength = {}
    with pytest.raises(ValueError) as excinfo:
        beam.wavelength = None
    with pytest.raises(ValueError) as excinfo:
        beam.wavelength = [
            1.5,
        ]
    with pytest.raises(ValueError) as excinfo:
        beam.wavelength = -45.4
    with pytest.raises(Warning) as excinfo:
        beam.wavelength = 5.0

    # waist
    with pytest.raises(ValueError) as excinfo:
        beam.waist = "451"
    with pytest.raises(ValueError) as excinfo:
        beam.waist = {}
    with pytest.raises(ValueError) as excinfo:
        beam.waist = None
    with pytest.raises(ValueError) as excinfo:
        beam.waist = [
            1.5,
        ]
    with pytest.raises(ValueError) as excinfo:
        beam.waist = -45.4

    # power
    with pytest.raises(ValueError) as excinfo:
        beam.power = "451"
    with pytest.raises(ValueError) as excinfo:
        beam.power = {}
    with pytest.raises(ValueError) as excinfo:
        beam.power = None
    with pytest.raises(ValueError) as excinfo:
        beam.power = [
            1.5,
        ]
    with pytest.raises(ValueError) as excinfo:
        beam.power = -45.4

    # waist position
    with pytest.raises(ValueError) as excinfo:
        beam.waist_position = "451"
    with pytest.raises(ValueError) as excinfo:
        beam.waist_position = 0.5
    with pytest.raises(ValueError) as excinfo:
        beam.waist_position = [0.5, 0.5]

    # direction type
    with pytest.raises(ValueError) as excinfo:
        beam.direction_type = "abc"

    # direction type
    beam.direction_type = "vector"
    with pytest.raises(ValueError) as excinfo:
        beam.direction_type = [1, 1]
    beam.direction_type = "thetaphi"
    with pytest.raises(ValueError) as excinfo:
        beam.direction_type = [1, 1, 1]
    with pytest.raises(ValueError) as excinfo:
        beam.direction_type = [0, 0, 0]


def _laserBeam_classes_generic_methods_test(LaserBeamClass):
    # - init
    beam = LaserBeamClass(
        wavelength=780e-9,
        waist=20e-6,
        power=50.0,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
    )

    # - coordinate conversions
    beam.waist_position = (1.0, -5.4, 0.5)
    beam.direction = (1, 1, 1)

    # 1 - sanity check > waist position is origin of new frame
    x, y, z = beam.waist_position
    expected_res = (0, 0, 0, 0, 0)
    res = beam._convert_coordinates_to_laser_frame(x, y, z)
    assert np.allclose(expected_res, res)

    # 2 - testing that it works with arrays
    # 2.a
    x = 1.5
    y = np.linspace(0, 5, 100)
    z = 0
    res = beam._convert_coordinates_to_laser_frame(x, y, z)
    for r in res:
        assert r.shape == y.shape

    # 2.b
    x = 1.5
    y = np.linspace(0, 5, 100)
    z = np.linspace(0, 5, 100)
    res = beam._convert_coordinates_to_laser_frame(x, y, z)
    for r in res:
        assert r.shape == y.shape

    # 2.c
    y = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    z = np.linspace(0, 5, 100)
    res = beam._convert_coordinates_to_laser_frame(x, y, z)
    for r in res:
        assert r.shape == y.shape

    # 2.d
    y = np.linspace(0, 5, 20)
    y = np.linspace(0, 5, 10)
    z = np.linspace(0, 5, 5)
    xxx, yyy, zzz = np.meshgrid(x, y, z)
    res = beam._convert_coordinates_to_laser_frame(xxx, yyy, zzz)
    for r in res:
        assert r.shape == xxx.shape


# % ACTUAL IMPLEMENTATION

# -- Abstract class


def test_Abstract_laser_beam_exception():
    from atomsmltr.environment.lasers.beams import AbstractLaserBeam

    _LaserBeam_classes_generic_exception_test(AbstractLaserBeam)


def test_Abstract_laser_beam_properties_setters_and_getters():
    from atomsmltr.environment.lasers.beams import AbstractLaserBeam

    _LaserBeam_classes_generic_properties_test(AbstractLaserBeam)


def test_Abstract_laser_beam_methods():
    from atomsmltr.environment.lasers.beams import AbstractLaserBeam

    _laserBeam_classes_generic_methods_test(AbstractLaserBeam)


# -- Gaussian beams


def test_Gaussian_laser_beam_exception():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    _LaserBeam_classes_generic_exception_test(GaussianLaserBeam)


def test_Gaussian_laser_beam_properties_setters_and_getters():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    _LaserBeam_classes_generic_properties_test(GaussianLaserBeam)


if __name__ == "__main__":
    test_Gaussian_laser_beam_properties_setters_and_getters()
    test_Gaussian_laser_beam_exception()
    test_Abstract_laser_beam_properties_setters_and_getters()
    test_Abstract_laser_beam_exception()
    test_Abstract_laser_beam_methods()
