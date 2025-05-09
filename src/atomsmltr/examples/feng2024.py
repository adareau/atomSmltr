"""
Examples : Feng 2024
=======================

This example provides the configuration for the Strontium source described
in Feng et al 2024 Quantum Sci. Technol. 9 025017 (DOI 10.1088/2058-9565/ad310b).

This example requires magpylib.
"""

# % IMPORTS

import magpylib as magpy
import numpy as np

# % LOCAL IMPORTS
from ..atoms import Strontium
from ..environment.lasers import GaussianLaserBeam
from ..environment.lasers.polarization import CircularLeft, CircularRight
from ..environment.fields.magnetic.magpylib import MagpylibWrapper
from ..environment.zones import Limits
from ..simulation import Configuration

# % GENERATE CONFIGURATION

# -- init config with strontium atom
config = Configuration(atom=Strontium())

# -- get Strontium main transition information
main = config.atom.trans["main"]

# -- setup magnetic field: symmetric configuration of permanent magnets in Feng et a 2024
# Define magnet properties
magnetization = (-8.7e5, 0, 0)
dimension = (0.003, 0.010, 0.025)

# X positions for the cuboids (same as in your original code)
x_positions = np.linspace(-0.012, 0.012, 9)

# Create cube1 and cube2 (lists of magnets)
cube1 = [
    magpy.magnet.Cuboid(magnetization=magnetization, dimension=dimension)
    for _ in x_positions
]
cube2 = [
    magpy.magnet.Cuboid(magnetization=magnetization, dimension=dimension)
    for _ in x_positions
]

# Assign positions to cube1 and cube2
for i, x in enumerate(x_positions):
    cube1[i].position = (x, 0.039, -0.050)
    cube2[i].position = (x, -0.039, -0.050)

# Create cube3 and cube4 as copies of cube1 and cube2
cube3 = [c.copy() for c in cube1]
cube4 = [c.copy() for c in cube2]

# Rotate cube3 and cube4 by 180 degrees along the y-axis
for c in cube3 + cube4:
    c.rotate_from_angax(180, "y", anchor=0)

# Combine all cubes into magnetSet as a single MagnetSet
magnetSet = magpy.Collection(cube1 + cube2 + cube3 + cube4)

# wrap it up
mag_field = MagpylibWrapper(magnetSet)
mag_field.tag = "Symmetric Permanent Magnet Configuration"

# -- setup lasers
# cf. config from Feng et al. 2024
l461_ZS = {
    "wavelength": 461e-9,
    "waist": 6.7e-3,
    "power": 45e-3,
    "waist_position": (0, 0, 0),
}
l461_ZS2 = {
    "wavelength": 461e-9,
    "waist": 7.2e-3,
    "power": 85e-3,
    "waist_position": (0, 0, 0),
}
l461_2DMOT = {
    "wavelength": 461e-9,
    "waist": 12e-3,
    "power": 80e-3,
    "waist_position": (0, 0, 0),
}

lasers = {}


# Updated function to allow specifying direction
def create_laser(tag, lasers_dict, params, direction, polarization=None):
    laser = GaussianLaserBeam(**params)
    laser.direction = np.array(direction)
    laser.tag = tag
    if polarization is not None:
        laser.polarization = polarization
    lasers_dict[tag] = laser


# Create Zeeman slower lasers
create_laser("ZS+", lasers, l461_ZS, [0, 0, -1], CircularRight())
create_laser("ZS-", lasers, l461_ZS, [0, 0, -1], CircularLeft())

# Create Zeeman2 slower lasers
create_laser("ZS2+", lasers, l461_ZS2, [0, 0, -1], CircularRight())
create_laser("ZS2-", lasers, l461_ZS2, [0, 0, -1], CircularLeft())

# Define directions for 2D MOT
d1_dir = np.array([1, 0, 1]) / np.sqrt(2)
d2_dir = np.array([1, 0, -1]) / np.sqrt(2)

# Create 2D MOT lasers
create_laser("d1<", lasers, l461_2DMOT, -d1_dir, CircularLeft())
create_laser("d1>", lasers, l461_2DMOT, d1_dir, CircularLeft())
create_laser("d2<", lasers, l461_2DMOT, -d2_dir, CircularRight())
create_laser("d2>", lasers, l461_2DMOT, d2_dir, CircularRight())

# -- add everything to the configuration

xlim = Limits(-0.15, 0.35, axis=2, target="position", action="stop", tag="xlim")
# add objects
config += (
    xlim,
    lasers["d1>"],
    lasers["d1<"],
    lasers["d2>"],
    lasers["d2<"],
    lasers["ZS+"],
    lasers["ZS-"],
    mag_field,
)

# setup atomlight
for laser in config.list_lasers():
    if (laser == "ZS+") or (laser == "ZS-"):
        config.add_atomlight_coupling(
            laser=laser, transition="main", detuning=-13 * main.Gamma
        )
    if (laser == "d1>") or (laser == "d1<") or (laser == "d2>") or (laser == "d2<"):
        config.add_atomlight_coupling(
            laser=laser, transition="main", detuning=-1.15 * main.Gamma
        )
