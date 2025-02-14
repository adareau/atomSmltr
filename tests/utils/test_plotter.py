import pytest
import numpy as np
import matplotlib.pyplot as plt


def test_plotter_import():
    from atomsmltr.utils.plotter import Axes3D
    from atomsmltr.utils import plotter


def test_laserbeam_plotter_3D():
    from atomsmltr.environment.lasers import GaussianLaserBeam
    from atomsmltr.utils.plotter import Axes3D
    from atomsmltr.environment.lasers.polarization import (
        CircularLeft,
        CircularRight,
        Vertical,
        Vector,
    )

    # test 1
    beam = GaussianLaserBeam(
        wavelength=780e-9,
        waist=20e-6,
        power=50e-3,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
        polarization=Vertical(),
    )

    beam.polarization = CircularLeft()
    ax = beam.plot3D(show=False, vscale=1)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    # test 2
    beam1 = GaussianLaserBeam(
        wavelength=780e-9,
        waist=20e-6,
        power=50e-3,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
        polarization=CircularLeft(),
    )

    beam2 = GaussianLaserBeam(
        wavelength=780e-9,
        waist=20e-6,
        power=50e-3,
        waist_position=[0, 0, 0],
        direction=[0, 1, 0],
        direction_type="vector",
        polarization=CircularRight(),
    )

    beam3 = GaussianLaserBeam(
        wavelength=780e-9,
        waist=20e-6,
        power=50e-3,
        waist_position=[0, 0, 0],
        direction=[1, 1, 1],
        direction_type="vector",
        polarization=Vector((1, 1, 1)),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    beam1.plot3D(ax, color="red")
    beam2.plot3D(ax, color="blue")
    beam3.plot3D(ax, color="green")


def test_laserbeam_plotter_2D():
    from atomsmltr.environment.lasers import GaussianLaserBeam

    # Define the beam
    beam = GaussianLaserBeam(
        wavelength=780e-9,
        waist=20e-6,
        power=50e-3,
        waist_position=[0, 0, 0],
        direction=[0, 0, 1],
        direction_type="vector",
    )

    # Check errors
    X, Y = np.meshgrid([1, 2, 3], [1, 2, 3])
    lim = (-1, 1, -1, 1)

    with pytest.raises(AssertionError):
        beam.plot2D(limits=(-5, 5, -5), Npoints=45)
    # -
    with pytest.raises(AssertionError):
        beam.plot2D(limits=lim, Npoints=15.0)
    # -
    with pytest.raises(AssertionError):
        beam.plot2D(limits=lim, Npoints=(5, 4, 6))

    # Plot
    # - 1
    x = 100e-6
    z = 10e-3
    Npoints = 500
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    beam.plot2D((-x, x, -x, x), Npoints, ax=ax[0], plane="XY", space_scale=1e3)
    beam.plot2D((-x, x, -z, z), Npoints, ax=ax[1], plane="YZ", space_scale=1e3)
    beam.plot2D((-z, z, -x, x), Npoints, ax=ax[2], plane="ZX", space_scale=1e3)
    plt.title("Example 1")
    # - 2
    beam.direction = (1, 2, 3)
    limits = (-100e-6, 100e-6, -100e-6, 100e-6)
    Npoints = 500
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    beam.plot2D(limits, Npoints, ax=ax[0], plane="XY", space_scale=1e6)
    beam.plot2D(limits, Npoints, ax=ax[1], plane="YZ", space_scale=1e6)
    beam.plot2D(limits, Npoints, ax=ax[2], plane="ZX", space_scale=1e6)
    plt.title("Example 1")

    # - 3
    beam.direction = (0.1, 1, 0)
    beam.plot2D(
        (-5e-3, 5e-3, -100e-6, 100e-6),
        (500, 100),
        cut=-100e-6,
        plane="YZ",
        space_scale=1e3,
    )

    # - 4
    beam.direction = (0, 1, 2e-3)
    limits = (-100e-6, 100e-6, -100e-6, 100e-6)
    Npoints = 500
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    beam.plot2D(limits, Npoints, ax=ax[0], cut=0, plane="ZX", space_scale=1e6)
    beam.plot2D(limits, Npoints, ax=ax[1], cut=-5e-3, plane="ZX", space_scale=1e6)
    beam.plot2D(limits, Npoints, ax=ax[2], cut=20e-3, plane="ZX", space_scale=1e6)


def test_mag_field_plotter():
    from atomsmltr.environment.fields.magnetic import MagneticGradient, MagneticOffset

    # -- Constant
    mag_field = MagneticOffset([0, 0, 1])
    mag_field.plot3D((-10, 10, -10, 10, -10, 10), (5, 10, 10))
    # -- Gradient

    mag_field = MagneticGradient(
        origin=(0, 0, 0),
        slope=-2,
        gradient_direction=(1, 0, 0),
        field_direction=(0, 1, 1),
        offset=10,
    )
    # plot 3Dc
    limits = (-10, 10, -10, 10, -10, 10)
    Npoints = (10, 4, 4)
    mag_field.plot3D(
        limits=limits, Npoints=Npoints, show=False, color="C2", normalize=True, scale=5
    )

    # plot 2D
    mag_field.offset = 0
    mag_field.plot2D(plane="XY", limits=(-10, 10, -10, 10), Npoints=10)
    mag_field.plot2D(plane="YZ", limits=(-10, 10, -10, 10), Npoints=10)
    mag_field.plot2D(plane="ZX", limits=(-10, 10, -10, 10), Npoints=10, cmap="jet")


def test_mag_field_plotter_for_magpylib():
    import magpylib as magpy
    from atomsmltr.environment.fields.magnetic import MagpylibWrapper

    loop = magpy.current.Circle(current=1, diameter=1)
    mag_field = MagpylibWrapper(loop)

    # plot 3Dc
    limits = (-10, 10, -10, 10, -10, 10)
    Npoints = (10, 4, 4)
    mag_field.plot3D(
        limits=limits, Npoints=Npoints, show=False, color="C2", normalize=True, scale=5
    )

    # plot 2D

    cyl = magpy.magnet.Cylinder(polarization=(0.5, 0.5, 0), dimension=(40, 20))
    mag_field = MagpylibWrapper(cyl)
    limits = (-50, 50, -50, 50)
    Npoints = 100
    mag_field.plot2D(plane="XY", limits=limits, Npoints=Npoints)
    mag_field.plot2D(plane="YZ", limits=limits, Npoints=Npoints)
    mag_field.plot2D(plane="ZX", limits=limits, Npoints=Npoints)


if __name__ == "__main__":
    test_laserbeam_plotter_2D()
    test_mag_field_plotter()
    test_mag_field_plotter_for_magpylib()
    plt.show()
