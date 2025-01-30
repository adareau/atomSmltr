import pytest
import numpy as np


def test_Gaussian_laser_beam_properties_setters_and_getters():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    # - testing initialization
    gaussian_beam = GaussianLaserBeam(
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
    gaussian_beam.wavelength = new_wavelength
    assert gaussian_beam.wavelength == new_wavelength

    # waist
    new_waist = 30e-6
    gaussian_beam.waist = new_waist
    assert gaussian_beam.waist == new_waist

    # power
    new_power = 60e-3
    gaussian_beam.power = new_power
    assert gaussian_beam.power == new_power

    # waist_position > array
    new_waist_position = np.array([1.0, 1.0, 1.0])
    gaussian_beam.waist_position = new_waist_position
    assert np.array_equal(gaussian_beam.waist_position, new_waist_position)

    # waist_position > tuple
    new_waist_position = (1.0, 1.0, 1.0)
    gaussian_beam.waist_position = new_waist_position
    assert np.array_equal(gaussian_beam.waist_position, new_waist_position)

    # direction_type
    new_direction_type = "thetaphi"
    gaussian_beam.direction_type = new_direction_type
    assert gaussian_beam.direction_type == new_direction_type

    # direction > thetaphi
    gaussian_beam.direction_type = "thetaphi"
    new_direction = [0.5, 3]
    gaussian_beam.direction = new_direction
    assert np.array_equal(gaussian_beam.direction, new_direction)
    # check that unit vector is well normalized
    assert np.linalg.norm(gaussian_beam._unit_vector) == 1.0

    # direction > vector
    gaussian_beam.direction_type = "vector"
    new_direction = [0.1, 0.1, 0.1]
    gaussian_beam.direction = new_direction
    assert np.array_equal(gaussian_beam.direction, new_direction)
    # check that unit vector is well normalized
    assert np.linalg.norm(gaussian_beam._unit_vector) == 1.0
    # check that unit vector is colinear to new_direction
    assert np.allclose(np.cross(new_direction, gaussian_beam._unit_vector), np.zeros(3))

    # check internal conversion thetaphi > vector
    gaussian_beam.direction_type = "thetaphi"
    # 1
    new_direction = [0.25 * np.pi, 0]
    expected_unit_vector = [np.sqrt(0.5), 0, np.sqrt(0.5)]
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector, expected_unit_vector)
    # 2
    new_direction = [-0.25 * np.pi, 0]
    expected_unit_vector = [-np.sqrt(0.5), 0, np.sqrt(0.5)]
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector, expected_unit_vector)
    # 3
    new_direction = [0.25 * np.pi, 0.25 * np.pi]
    expected_unit_vector = [np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.5)]
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector, expected_unit_vector)

    # check internal conversion vector > thetaphi
    gaussian_beam.direction_type = "vector"
    # 1
    new_direction = [1, 0, 0]
    expected_theta = 0.5 * np.pi
    expected_phi = 0
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector_phi, expected_phi)
    assert np.allclose(gaussian_beam._unit_vector_theta, expected_theta)
    # 2
    new_direction = [1, 1, 0]
    expected_theta = 0.5 * np.pi
    expected_phi = 0.25 * np.pi
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector_phi, expected_phi)
    assert np.allclose(gaussian_beam._unit_vector_theta, expected_theta)
    # 3
    new_direction = [1, -1, 0]
    expected_theta = 0.5 * np.pi
    expected_phi = -0.25 * np.pi
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector_phi, expected_phi)
    assert np.allclose(gaussian_beam._unit_vector_theta, expected_theta)
    # 4
    new_direction = [np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.5)]
    expected_theta = 0.25 * np.pi
    expected_phi = 0.25 * np.pi
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector_phi, expected_phi)
    assert np.allclose(gaussian_beam._unit_vector_theta, expected_theta)
    # 5
    new_direction = [np.sqrt(0.25), np.sqrt(0.25), -np.sqrt(0.5)]
    expected_theta = 0.75 * np.pi
    expected_phi = 0.25 * np.pi
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector_phi, expected_phi)
    assert np.allclose(gaussian_beam._unit_vector_theta, expected_theta)
    # 6
    new_direction = [-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.5)]
    expected_theta = 0.75 * np.pi
    expected_phi = -0.75 * np.pi
    gaussian_beam.direction = new_direction
    assert np.allclose(gaussian_beam._unit_vector_phi, expected_phi)
    assert np.allclose(gaussian_beam._unit_vector_theta, expected_theta)


def test_Gaussian_laser_beam_exception():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    # - init
    gaussian_beam = GaussianLaserBeam(
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
        gaussian_beam.wavelength = "451"
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.wavelength = {}
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.wavelength = None
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.wavelength = [
            1.5,
        ]
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.wavelength = -45.4
    with pytest.raises(Warning) as excinfo:
        gaussian_beam.wavelength = 5.0

    # waist
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist = "451"
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist = {}
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist = None
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist = [
            1.5,
        ]
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist = -45.4

    # power
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.power = "451"
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.power = {}
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.power = None
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.power = [
            1.5,
        ]
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.power = -45.4

    # waist position
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist_position = "451"
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist_position = 0.5
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.waist_position = [0.5, 0.5]

    # direction type
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.direction_type = "abc"

    # direction type
    gaussian_beam.direction_type = "vector"
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.direction_type = [1, 1]
    gaussian_beam.direction_type = "thetaphi"
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.direction_type = [1, 1, 1]
    with pytest.raises(ValueError) as excinfo:
        gaussian_beam.direction_type = [0, 0, 0]


if __name__ == "__main__":
    test_Gaussian_laser_beam_properties()
    test_Gaussian_laser_beam_exception()
