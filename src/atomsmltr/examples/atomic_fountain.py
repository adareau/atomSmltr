"""
Examples : Configuration (1,1,1)
=======================

This example provides the configuration for the (1,1,1) 3D MOT as described
in http://dx.doi.org/10.1103/PhysRevLett.100.050801
"""

# % EXPERIENCE DESCRIPTION
description = """

Atomic fountain with launched 87Rb atoms in a (1,1,1) MOT configuration:

Ref: http://dx.doi.org/10.1103/PhysRevLett.100.050801

In a 3D magneto-optical trap (MOT), the standard setup uses three orthogonal pairs of
counter-propagating laser beams along the x, y, and z axis. If we picture the trapping zone as a cube, then
the (1,1,1) configuration is the same as a classical MOT, only rotated such that the summit initially
on (1,1,1) and the one on (0,0,0) are now both located on the z-axis.

To do so, we first rotate the lasers by φ = -π/4 (around the z-axis), thus the summit initially in (1,1,1) is
now in (sqrt(2), 0, 1).
Then, we rotate them by an angle θ around the y-axix, such that the the summit initially in (sqrt(2), 0, 1)
is now in (0, 0, sqrt(3)) (because the rotations are orthogonal, the norm of the vector remains the same),
and we solve for θ using the rotation matrix.

You can refer to the github in "" for exact calculations.

This simulation models an atomic fountain using cold 87Rb atoms initially trapped in a 3D
magneto-optical trap (MOT) in a configuration (1,1,1).
After cooling, the magnetic field is suppressed and the atomic ensemble is launched upwards along
the vertical axis using a moving optical molasses, which relies on a detuning (epsilon)
between the upward and the downward-propagating laser beams.

Physical configuration:
- Atom species : 87Rb (D2 cooling transition at 780 nm)
- Polarization : 2 counter-propagating σ- beams + 4 counter-propating σ+ beams
- Laser power : 16.7 mW
- Beam waist (1/e radius) : 15.5 mm
- MOT detuning : -3 Γ
- Detuning between upwards and donwards-propagating lasers : 1 Mhz

"""

# % IMPORTS

import numpy as np

# % LOCAL IMPORTS

from atomsmltr.environment import GaussianLaserBeam
from atomsmltr.simulation import Configuration
from atomsmltr.atoms import Rubidium
from atomsmltr.environment.lasers import CircularLeft, CircularRight
from atomsmltr.environment import ConstantForce

# --------------------------------------------------------------------------------------------------------


# % GENERATE CONFIGURATION of the (1,1,1) 3D MOT


# -- init config with rubidium atom
atom = Rubidium()

# -- get rubidium main (D2) transition information
main = atom.trans["main"]
gamma = main.Gamma


# -- setup magnetic field
# no magnetic field here ¯\_(ツ)_/¯

# -- add the relevant constant forces

m = Rubidium().mass  # kg
g = 9.81  # m/s^2
direction = np.array([0, 0, -1])  # along -z
grav_force = m * g * direction
gravity = ConstantForce(field_value=grav_force, tag="gravity")


# -- setup lasers of the 1D MOT
# cf. config from 'insert ref here'

laser_1 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction_type="thetaphi",
    direction=(0.95500000000000, 3.141592653589793),
    polarization=CircularLeft(),
    tag="las1",
)

laser_2 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction_type="thetaphi",
    direction=(2.186592653589793, 0),
    polarization=CircularLeft(),
    tag="las2",
)

laser_3 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction_type="thetaphi",
    direction=(0.9554749537638146, -1.047003706391126),
    polarization=CircularRight(),
    tag="las3",
)

laser_4 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction_type="thetaphi",
    direction=(2.186117699825979, 2.0945889471986674),
    polarization=CircularRight(),
    tag="las4",
)

laser_5 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction_type="thetaphi",
    direction=(0.9554749537638146, 1.047003706391126),
    polarization=CircularRight(),
    tag="las5",
)

laser_6 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction_type="thetaphi",
    direction=(2.186117699825979, -2.0945889471986674),
    polarization=CircularRight(),
    tag="las6",
)


# -- add everything to the configuration
config = Configuration()
config.atom = atom
config.add_objects([gravity])
config += laser_1, laser_2, laser_3, laser_4, laser_5, laser_6


# -- setup lasers detuning parameters of the 3D MOT
epsilon = 2 * np.pi * 1e6
detuning = -3 * gamma


# -- add the detuning and the right epsilon to the atomlight coupling

list_lasers = config.list_lasers()
for laser_name in list_lasers:
    laser = config.get_laser_copy(laser_name)
    direction = laser.direction
    if direction[0] < np.pi / 2:
        config.add_atomlight_coupling(laser_name, "main", detuning + epsilon)
    else:
        config.add_atomlight_coupling(laser_name, "main", detuning - epsilon)
