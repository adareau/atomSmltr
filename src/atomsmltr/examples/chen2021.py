"""
Examples : Chen 2021
=======================

This file provides the configuration for the 1D MOT, 1D molasses and 3D MOT configurations
in Chen 2021  : arXiv:2105.06447

"""

# % EXPERIENCE DESCRIPTION
description = """

This code includes three example simulations from the AtomECS paper (Chen et al., arXiv:2105.06447), demonstrating
atomic motion under different laser cooling configurations.

1 ) One-dimensionnal molasses :

>> from atomsmltr.examples.chen2021 import config_1D_molasses

This simulation models a one-dimensional optical molasses for 87Rb atoms, following the setup
described in the AtomECS paper. The system consists of two counter-propagating, red-detuned laser beams
along the z-axis, forming an optical molasses that slows atomic motion through Doppler cooling.

Physical configuration:
- Atom species : 87Rb (D2 cooling transition at 780 nm)
- Laser power : 10 mW per beam
- Beam waist (1/e radius) : 1 cm
- Detuning : -12 MHz from resonance
- Polarization : σ-

2 ) One-dimensional Magneto-Optical Trap (1D MOT):

>> from atomsmltr.examples.chen2021 import config_1D_MOT

This simulation models a one-dimensional magneto-optical trap for 88Sr atoms, extending the
optical molasses setup by adding a magnetic quadrupole field along the z-axis. The combination
of red-detuned counter-propagating laser beams and the position-dependent magnetic field
allows atoms with velocities below the capture velocity to be trapped and cooled.

Physical configuration:
- Atom species : 88Sr (main cooling transition at 461 nm)
- Laser power : 30 mW per beam
- Beam waist (1/e radius): 1 cm
- Detuning : -12 MHz from resonance
- Polarization : σ-
- Magnetic field type : Perfect quadrupole
- Magnetic field direction : (0, 0, 1)
- Magnetic field origin : (0, 0, 0)
- Magnetic field gradient : 0.15 T/m


3 ) Three-dimensional MOT :

>> from atomsmltr.examples.chen2021 import config_3D_MOT

This simulation models a three-dimensional MOT for 87Rb atoms using six counter-propagating
laser beams along the Cartesian axes, combined with a 3D quadrupole magnetic field with strong axis along the z-axis.

Physical configuration:
- Atom species : 87Rb (D2 cooling transition at 461 nm)
- Laser power : 20 mW per beam
- Beam waist (1/e radius) : 1 cm
- Detuning : -12 MHz from resonance
- Polarization for lasers along the z-axis : σ-
- Polarization for lasers along the other axis : σ+
- Magnetic field type : Perfect quadrupole
- Magnetic field direction : (0, 0, 1)
- Magnetic field origin : (0, 0, 0)
- Magnetic field gradient : 0.3 T/m

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
