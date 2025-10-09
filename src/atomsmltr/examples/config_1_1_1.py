"""
Examples : Configuration (1,1,1)
=======================

This example provides the configuration for the 3D MOT 
in 'insert exact ref here'
"""

# % IMPORTS

import numpy as np
import matplotlib.pyplot as plt

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

# -- get Strontium main transition information
main = atom.trans["main"]
gamma = main.Gamma


# -- setup magnetic field: perfect quadrupole with a strong Z-axis in Chen 2021
# Define magnet properties


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
    direction=(0, 0, 1),
    polarization=CircularLeft(),
    tag="las1",
)

laser_2 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction=(0, 0, -1),
    polarization=CircularLeft(),
    tag="las2",
)

laser_3 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction=(1, 0, 0),
    polarization=CircularRight(),
    tag="las3",
)

laser_4 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction=(-1, 0, 0),
    polarization=CircularRight(),
    tag="las4",
)

laser_5 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction=(0, 1, 0),
    polarization=CircularRight(),
    tag="las5",
)

laser_6 = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=22e-3,
    power=100e-3 / 6,
    waist_position=(0, 0, 0),
    direction=(0, -1, 0),
    polarization=CircularRight(),
    tag="las6",
)


# -- add everything to the configuration
config = Configuration()
config.atom = atom
config.add_objects([gravity])
config += laser_1, laser_2, laser_3, laser_4, laser_5, laser_6


# -- rotate the config to a (1,1,1)
config.config_to_1_1_1()


# -- setup lasers detuning parameters of the 3D MOT
epsilon = 2 * np.pi * 1e6
detuning = -3 * gamma


# -- add the detuning and the right epsilon to the atomlight coupling

list_lasers = config.list_lasers()
for laser_name in list_lasers:
    laser = config.get_laser_copy(laser_name)
    direction = laser.direction
    if direction[2] > 0:
        config.add_atomlight_coupling(laser_name, "main", detuning + epsilon)
    else:
        config.add_atomlight_coupling(laser_name, "main", detuning - epsilon)
