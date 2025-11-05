"""
Examples: atomSmltr 2025
=======================

This example provides the configurations used to generate the figures of the atomSmltr
presentation paper from 2025. See the paper for more information.

---------------------------------------------------------------
>> from atomsmltr.examples.atomsmltr import config_1D_MOT_Yb

1D MOT for Ytterbium, used to benchmark the damped oscillation of an atom in a MOT
using analytical formulas for the fluid friction and harmonic oscillator stiffness.

---------------------------------------------------------------
>> from atomsmltr.examples.atomsmltr import config_Doppler_limit

Configuration to test the Doppler limit, using a 3D optical molasses setup with
ytterbium atoms.

---------------------------------------------------------------
>> from atomsmltr.examples.atomsmltr import config_3D_MOT_Yb

3D MOT for Ytterbium, used to benchmark the performances of the various
integration models.

"""

# % EXPERIENCE DESCRIPTION
description = """
Examples from the atomSmltr presentation paper (2025).

---------------------------------------------------------------
>> from atomsmltr.examples.atomsmltr import config_1D_MOT_Yb

1D MOT for Ytterbium, used to benchmark the damped oscillation of an atom in a MOT
using analytical formulas for the fluid friction and harmonic oscillator stiffness.

---------------------------------------------------------------
>> from atomsmltr.examples.atomsmltr import config_Doppler_limit

Configuration to test the Doppler limit, using a 3D optical molasses setup with
ytterbium atoms.

---------------------------------------------------------------
>> from atomsmltr.examples.atomsmltr import config_3D_MOT_Yb

3D MOT for Ytterbium, used to benchmark the performances of the various
integration models.
"""

# % IMPORTS

import numpy as np

# % LOCAL IMPORTS

from ..atoms import Ytterbium
from ..environment.lasers import PlaneWaveLaserBeam, GaussianLaserBeam
from ..environment.fields import MagneticGradient, MagneticQuadrupoleZ
from ..environment.lasers import CircularRight, CircularLeft
from ..simulation import Configuration

# ------------------------------------------------------------------
# % EXAMPLE 1 - 1D MOT SPRING MODEL
# ------------------------------------------------------------------

# - parameters
atom = Ytterbium()
transition = atom.trans["main"]
transition.print_info()
s = 0.02
b = 0.1  # T/m
delta = -0.5 * transition.Gamma

# lasers
laser_1 = PlaneWaveLaserBeam(
    wavelength=transition.wavelength,
    direction=(1, 0, 0),
    tag="399+",
    polarization=CircularRight(),
)
laser_2 = PlaneWaveLaserBeam(
    wavelength=transition.wavelength,
    direction=(-1, 0, 0),
    tag="399-",
    polarization=CircularRight(),
)
laser_1.set_power_from_I(s * transition.Isat)
laser_2.set_power_from_I(s * transition.Isat)

# magnetic field
mag_gradient = MagneticGradient(
    origin=(0, 0, 0),
    slope=b,
    gradient_direction=(1, 0, 0),
    field_direction=(1, 0, 0),
    tag="mag_gradient",
)
# config
config_1D_MOT_Yb = Configuration(atom=atom) + laser_1 + laser_2 + mag_gradient
config_1D_MOT_Yb.add_atomlight_coupling("399+", "main", detuning=delta)
config_1D_MOT_Yb.add_atomlight_coupling("399-", "main", detuning=delta)
config_1D_MOT_Yb.delta = delta
config_1D_MOT_Yb.s = s
config_1D_MOT_Yb.b = b

# ------------------------------------------------------------------
# % EXAMPLE 2 - Optical molasses Doppler limit
# ------------------------------------------------------------------

# set laser saturation parameter
s0 = 0.05  # I/I_sat
main = Ytterbium().trans["main"]  # 399nm transition of Yb

# create laser list
lasers = {}
for axis, direction in zip(["x", "y", "z"], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
    for head, mult in zip([">", "<"], [1, -1]):
        dir = np.array(direction) * mult
        tag = axis + head
        laser = PlaneWaveLaserBeam(wavelength=main.wavelength)
        laser.direction = dir
        laser.tag = tag
        laser.set_power_from_I(s0 * main.Isat)
        lasers[tag] = laser

# generate config
config_Doppler_limit = Configuration(atom=Ytterbium())
config_Doppler_limit += [*lasers.values()]


# ------------------------------------------------------------------
# % EXAMPLE 3 - 3D MOT for performances benchmark
# ------------------------------------------------------------------

# setup magnetic field
B_grad_G_per_cm = 30
B_grad = B_grad_G_per_cm * 1e-4 / 1e-2
mag_quad = MagneticQuadrupoleZ(origin=(0, 0, 0), slope=B_grad, tag="MOT field")

# setup lasers
# cf. config from Letellier et al. 2023
l399 = {
    "wavelength": 399e-9,
    "waist": 22e-3,
    "power": 100e-3 / 6,
    "waist_position": (0, 0, 0),
}
lasers = {}
for axis, direction in zip(["x", "y", "z"], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
    for head, mult in zip([">", "<"], [1, -1]):
        dir = np.array(direction) * mult
        tag = axis + head
        laser = GaussianLaserBeam(**l399)
        laser.direction = dir
        laser.tag = tag
        lasers[tag] = laser

# set laser polarization
# NB : the strong axis of the quadrupole is along z
lasers["x>"].polarization = CircularRight()
lasers["x<"].polarization = CircularRight()
lasers["y>"].polarization = CircularRight()
lasers["y<"].polarization = CircularRight()
lasers["z>"].polarization = CircularLeft()
lasers["z<"].polarization = CircularLeft()

# config
config_3D_MOT_Yb = Configuration(atom=Ytterbium())
config_3D_MOT_Yb += mag_quad
config_3D_MOT_Yb += [*lasers.values()]

# atom-light
main = config_3D_MOT_Yb.atom.trans["main"]
for laser in config_3D_MOT_Yb.list_lasers():
    config_3D_MOT_Yb.add_atomlight_coupling(
        laser=laser, transition="main", detuning=-main.Gamma
    )
