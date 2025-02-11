import pytest
import numpy as np
import matplotlib.pyplot as plt


def test_plotter_import():
    from atomsmltr.utils.plotter import Axes3D
    from atomsmltr.utils import plotter


def test_laserbeam_plotter():
    from atomsmltr.environment.lasers import GaussianLaserBeam
    from atomsmltr.utils.plotter import Axes3D
    from atomsmltr.environment.lasers.polarization import (
        CircularLeft,
        CircularRight,
        Horizontal,
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
    plt.show()


if __name__ == "__main__":
    test_laserbeam_plotter()
