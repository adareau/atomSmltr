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
    # -
    with pytest.raises(ValueError):
        beam.plot2D(plane="yx")
    # -
    with pytest.raises(ValueError):
        beam.plot2D(Npoints=15)
    # -
    with pytest.raises(ValueError):
        beam.plot2D(limits=lim)
    # -
    with pytest.raises(ValueError):
        beam.plot2D(X=X)
    with pytest.raises(ValueError):
        beam.plot2D(Y=X)
    # -
    with pytest.raises(ValueError):
        beam.plot2D(limits=lim, Npoints=15, X=0)
    # -
    with pytest.raises(AssertionError):
        beam.plot2D(limits=(-5, 5, -5), Npoints=45)
    # -
    with pytest.raises(AssertionError):
        beam.plot2D(limits=lim, Npoints=15.0)
    # -
    with pytest.raises(AssertionError):
        beam.plot2D(limits=lim, Npoints=(5, 4, 6))
    # -
    with pytest.raises(AssertionError):
        beam.plot2D(X=X, Y=X[:2, :2])
    # -
    with pytest.raises(AssertionError):
        beam.plot2D("XY")

    # Plot
    # - 1
    x = np.linspace(-100e-6, 100e-6, 500)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    beam.plot2D(ax[0], "XY", X=X, Y=Y, space_scale=1e6)
    beam.plot2D(ax[1], "YZ", X=X, Y=Y, space_scale=1e6)
    beam.plot2D(ax[2], "ZX", X=X, Y=Y, space_scale=1e6)
    plt.title("Example 1")

    # - 2
    limits = (-100e-6, 100e-6, -100e-6, 100e-6)
    Npoints = 500
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    beam.plot2D(ax[0], "XY", limits=limits, Npoints=Npoints, space_scale=1e6)
    beam.plot2D(ax[1], "YZ", limits=limits, Npoints=Npoints, space_scale=1e6)
    beam.plot2D(ax[2], "ZX", limits=limits, Npoints=Npoints, space_scale=1e6)
    plt.title("Example 2")

    # - 23
    limits = (-100e-6, 100e-6, -10e-3, 10e-3)
    Npoints = (500, 500)
    beam.plot2D(plane="YZ", limits=limits, Npoints=Npoints)
    plt.title("Example 3")


if __name__ == "__main__":
    test_laserbeam_plotter_2D()
    plt.show()
