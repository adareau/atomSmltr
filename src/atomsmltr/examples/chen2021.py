"""
Examples : Chen 2021
=======================

This example provides the configuration for the 1D MOT et the 1 molasses
in Chen 2021 'insert exact ref here'
"""

# % IMPORTS

import numpy as np
import matplotlib.pyplot as plt

# % LOCAL IMPORTS
from atomsmltr.environment import GaussianLaserBeam
from atomsmltr.atoms import Strontium, Rubidium
from atomsmltr.simulation import Configuration
from atomsmltr.environment.lasers import CircularLeft, CircularRight
from atomsmltr.environment.fields.magnetic import MagneticQuadrupoleZ

# % GENERATE CONFIGURATION of the 1D MOT


# -- init config with strontium atom
atom_strontium = Strontium()

# -- get Strontium main transition information
main = atom_strontium.trans["main"]

# -- setup magnetic field: perfect quadrupole with a strong Z-axis in Chen 2021
# Define magnet properties
origin = np.array((0, 0, 0))
gradient = 0.15  # T/m
mag_field = MagneticQuadrupoleZ(origin=origin, slope=gradient, tag="mag_field_1")

# -- setup lasers of the 1D MOT
# cf. config from Chen 2021
laser_1_MOT = GaussianLaserBeam(
    wavelength=460.862e-9,
    waist=(1e-2) * np.sqrt(2),
    power=3e-2,
    waist_position=(0, 0, 0),
    direction=(0, 0, 1),
    polarization=CircularLeft(),
    tag="las1mot",
)

laser_2_MOT = GaussianLaserBeam(
    wavelength=460.862e-9,
    waist=(1e-2) * np.sqrt(2),
    power=3e-2,
    waist_position=(0, 0, 0),
    direction=(0, 0, -1),
    polarization=CircularLeft(),
    tag="las2mot",
)


# -- add everything to the configuration

config_1D_MOT = Configuration()
config_1D_MOT.atom = atom_strontium

# add objects
config_1D_MOT += laser_1_MOT, laser_2_MOT, mag_field

# setup atomlight
config_1D_MOT.add_atomlight_coupling("las1", "main", -2 * np.pi * 12e6)
config_1D_MOT.add_atomlight_coupling("las2", "main", -2 * np.pi * 12e6)


# % GENERATE CONFIGURATION of the 1D molasses


# -- init config with strontium atom
atom_rubidium = Rubidium()

# -- get Strontium main transition information
main = atom_rubidium.trans["main"]


# -- setup lasers of the 1D molasses
# cf. config from Chen 2021
laser_1_molasses = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=(1e-2) * np.sqrt(2),
    power=1e-2,
    waist_position=(0, 0, 0),
    direction=(0, 0, 1),
    polarization=CircularLeft(),
    tag="las1molasses",
)

laser_2_molasses = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=(1e-2) * np.sqrt(2),
    power=1e-2,
    waist_position=(0, 0, 0),
    direction=(0, 0, -1),
    polarization=CircularLeft(),
    tag="las2molasses",
)


# -- add everything to the configuration

config_1D_molasses = Configuration()
config_1D_molasses.atom = atom_rubidium

# add objects
config_1D_molasses += laser_1_molasses, laser_2_molasses

# setup atomlight
config_1D_molasses.add_atomlight_coupling("las1molasses", "main", -2 * np.pi * 6e6)
config_1D_molasses.add_atomlight_coupling("las2molasses", "main", -2 * np.pi * 6e6)
