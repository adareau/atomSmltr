import pytest
import numpy as np


# % GENERIC TESTS


def _check_scalar_field_value_function(func):
    """Checks that a function yielding values of a 3D field
    field behaves correctly with numpy arrays. Typically used to
    check intensities.
    """

    # - 0 check exceptions
    with pytest.raises(ValueError) as excinfo:
        func(0)
    with pytest.raises(ValueError) as excinfo:
        func((0, 0))
    with pytest.raises(ValueError) as excinfo:
        func(np.linspace(0, 1, 20))
    with pytest.raises(ValueError) as excinfo:
        func(np.mgrid[0:1:10j, 0:1:10j, 0:1:10j])

    # - 1 check that it works with a single position
    position = (0, 0, 0)
    value = func(position)
    assert value.ndim == 0

    # - 2 with arrays
    # -
    position = np.mgrid[0:1:8j, 0:5:10j, 0:0:1j].T
    X, _, _ = position.T
    value = func(position)
    assert value.shape == X.shape
    # -
    position = position[0]
    X, _, _ = position.T
    value = func(position)
    assert value.shape == X.shape


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

    # - info
    beam.print_info()

    # - testing setters and getters
    # tag
    new_tag = "deathstar laser"
    beam.tag = new_tag
    assert beam.tag == new_tag

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


def _LaserBeam_classes_generic_polarization_test(LaserBeamClass):
    import atomsmltr.environment.lasers.polarization as pol

    # - init
    beam = LaserBeamClass(
        wavelength=780e-9,
        waist=20e-6,
        power=50.0,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
    )
    # default = Vertical
    assert beam.polarization.type == "Vertical"

    # test setter # 1
    beam.polarization = pol.CircularLeft()
    assert beam.polarization.type == "Circular Left"

    # test setter # 2
    beam.polarization = pol.Vector((0, 0, 1))
    assert np.allclose(beam.polarization.get_polarization_vector(), (0, 0, 1))

    # - testing polarization vector conversions (lab frame) <> (laser frame)
    # vertical (x) polarization, propagation along -z
    beam.polarization = pol.Vertical()
    beam.direction = (0, 0, -1)
    assert np.allclose(beam.get_polarization_vector_in_laser_frame(), (1, 0, 0))
    assert np.allclose(beam.get_polarization_vector_in_lab_frame(), (-1, 0, 0))

    # circular left polarization, propagation along -z
    beam.polarization = pol.CircularLeft()
    beam.direction = (0, 0, -1)
    assert np.allclose(beam.get_polarization_vector_in_laser_frame(), (0, 0, -1))
    assert np.allclose(beam.get_polarization_vector_in_lab_frame(), (0, 0, 1))

    # - testing mag field projections
    # usefull subfunction
    def _unpack(proj):
        return (proj["sigma+"], proj["sigma-"], proj["pi"])

    # 1) laser colinear with B, and same direction
    # 1.a) Circular Right > sigma +
    beam.direction = (-1, 0, 1)  # take an arbitrary direction for the test
    mag_field = beam.direction
    beam.polarization = pol.CircularRight()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (1, 0, 0))
    # 1.b) Circular Left > sigma -
    beam.direction = (0, 1, 2)  # take an arbitrary direction for the test
    mag_field = beam.direction
    beam.polarization = pol.CircularLeft()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0, 1, 0))
    # 1.c) Horizontal > sigma
    beam.direction = (4, -3, 2)  # take an arbitrary direction for the test
    mag_field = beam.direction
    beam.polarization = pol.Horizontal()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.5, 0.5, 0))
    # 1.d) Vertical > sigma
    beam.direction = (0, -1, 0)  # take an arbitrary direction for the test
    mag_field = beam.direction
    beam.polarization = pol.Vertical()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.5, 0.5, 0))
    # 1.e) Any linear > sigma
    beam.direction = (0, -1, 0)  # take an arbitrary direction for the test
    mag_field = beam.direction
    beam.polarization = pol.Linear(angle=np.pi / 8)
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.5, 0.5, 0))

    # 2) laser colinear with B, and same direction
    # 2.a) Circular Right > sigma -
    beam.direction = (-1, 0, 1)  # take an arbitrary direction for the test
    mag_field = -beam.direction
    beam.polarization = pol.CircularRight()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0, 1, 0))
    # 2.b) Circular Left > sigma -
    beam.direction = (0, 1, 2)  # take an arbitrary direction for the test
    mag_field = -beam.direction
    beam.polarization = pol.CircularLeft()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (1, 0, 0))
    # 2.cd) Any linear > sigma
    beam.direction = (0, -1, 0)  # take an arbitrary direction for the test
    mag_field = -beam.direction
    beam.polarization = pol.Linear(angle=np.pi / 8)
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.5, 0.5, 0))

    # 3) laser orthogonal to B
    # 3.a) circular right > sigma + pi
    beam.direction = (0, 0, 1)
    mag_field = (1, 0, 0)
    beam.polarization = pol.CircularRight()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.25, 0.25, 0.5))
    # 3.b) circular left > sigma + pi
    beam.direction = (0, 1, 1)
    mag_field = (0, 1, -1)
    beam.polarization = pol.CircularLeft()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.25, 0.25, 0.5))
    # 3.c) B aligned with polarization > pi
    beam.direction = (0, 0, 1)
    mag_field = (1, 0, 0)
    beam.polarization = pol.Vertical()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0, 0, 1))
    # 3.d) B aligned with polarization > pi
    beam.direction = (0, 0, 1)
    mag_field = (0, 1, 0)
    beam.polarization = pol.Horizontal()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0, 0, 1))
    # 3.e) B aligned with polarization > pi
    beam.direction = (0, 0, 1)
    mag_field = (1, 1, 0)
    beam.polarization = pol.Linear(angle=np.pi / 4)
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0, 0, 1))
    # 3.e) B aligned with polarization > pi
    beam.direction = (1, 1, 1)
    mag_field = beam._convert_vector_to_lab_frame((1, 0, 0))
    beam.polarization = pol.Vertical()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0, 0, 1))
    # 3.f) B orthogonal to polarization > sigma
    beam.direction = (0, 0, 1)
    mag_field = (0, 1, 0)
    beam.polarization = pol.Vertical()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.5, 0.5, 0))
    # 3.g) B orthogonal to polarization > sigma
    beam.direction = (0, 0, 1)
    mag_field = (1, 0, 0)
    beam.polarization = pol.Horizontal()
    proj = beam.get_polarization_magnetic_projection_norm(mag_field)
    assert np.allclose(_unpack(proj), (0.5, 0.5, 0))


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

    # polarization
    with pytest.raises(TypeError) as excinfo:
        beam.polarization = 0

    # tag
    with pytest.raises(TypeError) as excinfo:
        beam.tag = 0


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

    # -- coordinate conversions
    beam.waist_position = (1.0, -5.4, 0.5)
    beam.direction = (1, 1, 1)

    # 1 - sanity check > waist position is origin of new frame
    x, y, z = beam.waist_position
    expected_res = (0, 0, 0, 0, 0)
    res = beam._convert_coordinates_to_laser_frame(x, y, z)
    assert np.allclose(expected_res, res)

    # -- vector rotations
    # - stupid check : in the laser frame, the beam_direction is aligned with z
    # and its norm should be conserved...
    for beam_direction in [
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, -1),
        (-1, 0, 1),
        (1, 1, 1),
        (0, -1, 0),
        (5, 8, 7),
    ]:
        # align laser with vector
        beam.direction = beam_direction
        # compute vector coordinate in laser frame
        dir_in_laser_frame = beam._convert_vector_to_laser_frame(beam_direction)
        # it should be aligned with z (0,0,1) with the same norm
        expected_direction = np.array([0, 0, 1]) * np.linalg.norm(beam_direction)
        # so we test it !
        assert np.allclose(dir_in_laser_frame, expected_direction)

    # - a few simple checks
    check_list = [  # a list of (laser_direction, lab_frame_vec, laser_frame_vec)
        ((0, 0, 1), (1, 2, 3), (1, 2, 3)),  # no rotation
        ((0, 0, -1), (1, 2, 3), (-1, 2, -3)),  # propagation // -z
        ((1, 0, 0), (1, 2, 3), (-3, 2, 1)),  # propagation // x
        ((-1, 0, 0), (1, 2, 3), (-3, -2, -1)),  # propagation // -x
        ((0, 1, 0), (1, 2, 3), (-3, -1, 2)),  # propagation // y
        ((0, -1, 0), (1, 2, 3), (-3, 1, -2)),  # propagation // -y
    ]

    for to_check in check_list:
        direction, vec_lab, vec_las_exp = to_check
        beam.direction = direction
        vec_las = beam._convert_vector_to_laser_frame(vec_lab)
        assert np.allclose(vec_las_exp, vec_las), f"{vec_las=}, {vec_las_exp=}"

    # - check reverse rotation
    # prepare list of directions
    v = np.array([-1, 0, 1])
    X, Y, Z = np.meshgrid(v, v, v)
    # shorthands
    fwd = beam._convert_vector_to_laser_frame  # forward rotation
    bwd = beam._convert_vector_to_lab_frame  # backward rotation
    # scan
    for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel()):
        if x**2 + y**2 + z**2 > 0:
            beam.direction = (x, y, z)
            for xvec, yvec, zvec in zip(X.ravel(), Y.ravel(), Z.ravel()):
                vec = (xvec, yvec, zvec)
                assert np.allclose(vec, bwd(fwd(vec)))
                assert np.allclose(vec, fwd(bwd(vec)))

    # -- intensity function
    _check_scalar_field_value_function(beam.get_intensity)


# % ACTUAL IMPLEMENTATION

# -- Gaussian beams


def test_Gaussian_laser_beam_exception():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    _LaserBeam_classes_generic_exception_test(GaussianLaserBeam)


def test_Gaussian_laser_beam_properties_setters_and_getters():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    _LaserBeam_classes_generic_properties_test(GaussianLaserBeam)


def test_Gaussian_laser_beam_methods():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    _laserBeam_classes_generic_methods_test(GaussianLaserBeam)


def test_Gaussian_laser_beam_polarization():
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam

    _LaserBeam_classes_generic_polarization_test(GaussianLaserBeam)


if __name__ == "__main__":
    test_Gaussian_laser_beam_properties_setters_and_getters()
    test_Gaussian_laser_beam_exception()
    test_Gaussian_laser_beam_methods()
    test_Gaussian_laser_beam_polarization()
