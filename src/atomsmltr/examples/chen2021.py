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

# --------------------------------------------------------------------------------------------------------

# % GENERATE CONFIGURATION of the 1D MOT


# -- init config with strontium atom
atom_strontium = Strontium()

# -- get Strontium main transition information
main = atom_strontium.trans["main"]

# -- setup magnetic field: perfect quadrupole with a strong Z-axis in Chen 2021
# Define magnet properties
origin_1D = np.array((0, 0, 0))
gradient_1D = 0.15  # T/m
mag_field_1D = MagneticQuadrupoleZ(
    origin=origin_1D, slope=gradient_1D, tag="mag_field_1D"
)

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
config_1D_MOT += laser_1_MOT, laser_2_MOT, mag_field_1D

# setup atomlight
config_1D_MOT.add_atomlight_coupling("las1mot", "main", -2 * np.pi * 12e6)
config_1D_MOT.add_atomlight_coupling("las2mot", "main", -2 * np.pi * 12e6)


# --------------------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------------------

# % GENERATE CONFIGURATION of the 3D MOT


# -- init config with strontium atom
atom_rubidium = Rubidium()

# -- get Strontium main transition information
main = atom_rubidium.trans["main"]

# -- setup magnetic field: perfect quadrupole with a strong Z-axis in Chen 2021
# Define magnet properties
origin_3D = np.array((0, 0, 0))
gradient_3D = 0.3  # T/m
mag_field_3D = MagneticQuadrupoleZ(
    origin=origin_3D, slope=gradient_3D, tag="mag_field_3D"
)

# -- setup lasers of the 1D MOT
# cf. config from Chen 2021
laser_1_3D_MOT = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=66.7e-3,
    power=0.02,
    waist_position=(0, 0, 0),
    direction=(0, 0, 1),
    polarization=CircularLeft(),
    tag="las1mot3D",
)

laser_2_3D_MOT = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=66.7e-3,
    power=0.02,
    waist_position=(0, 0, 0),
    direction=(0, 0, -1),
    polarization=CircularLeft(),
    tag="las2mot3D",
)

laser_3_3D_MOT = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=66.7e-3,
    power=0.02,
    waist_position=(0, 0, 0),
    direction=(1, 0, 0),
    polarization=CircularRight(),
    tag="las3mot3D",
)

laser_4_3D_MOT = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=66.7e-3,
    power=0.02,
    waist_position=(0, 0, 0),
    direction=(-1, 0, 0),
    polarization=CircularRight(),
    tag="las4mot3D",
)

laser_5_3D_MOT = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=66.7e-3,
    power=0.02,
    waist_position=(0, 0, 0),
    direction=(0, 1, 0),
    polarization=CircularRight(),
    tag="las5mot3D",
)
laser_6_3D_MOT = GaussianLaserBeam(
    wavelength=780.241e-9,
    waist=66.7e-3,
    power=0.02,
    waist_position=(0, 0, 0),
    direction=(0, -1, 0),
    polarization=CircularRight(),
    tag="las6mot3D",
)


# -- add everything to the configuration

config_3D_MOT = Configuration()
config_3D_MOT.atom = atom_rubidium

# add objects
config_3D_MOT += (
    laser_1_3D_MOT,
    laser_2_3D_MOT,
    laser_3_3D_MOT,
    laser_4_3D_MOT,
    laser_5_3D_MOT,
    laser_6_3D_MOT,
    mag_field_3D,
)

# setup atomlight
config_3D_MOT.add_atomlight_coupling("las1mot3D", "main", -2 * np.pi * 3e6)
config_3D_MOT.add_atomlight_coupling("las2mot3D", "main", -2 * np.pi * 3e6)
config_3D_MOT.add_atomlight_coupling("las3mot3D", "main", -2 * np.pi * 3e6)
config_3D_MOT.add_atomlight_coupling("las4mot3D", "main", -2 * np.pi * 3e6)
config_3D_MOT.add_atomlight_coupling("las5mot3D", "main", -2 * np.pi * 3e6)
config_3D_MOT.add_atomlight_coupling("las6mot3D", "main", -2 * np.pi * 3e6)
