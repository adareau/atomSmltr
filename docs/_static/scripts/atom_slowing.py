"""Atom Slowing

* author  : A. DAREAU
* created : 07/03/2025

* description

    A small script that can be used as a minimal example for testing
    the atomsmtrl package

"""

import numpy as np
import matplotlib.pyplot as plt

from atomsmltr.environment import PlaneWaveLaserBeam
from atomsmltr.atoms import Ytterbium
from atomsmltr.simulation import Configuration, ScipyIVP_3D

# - setup atom
atom = Ytterbium()
main = atom.trans["main"]  # get transition, to help setting up lasers

# - setup laser
laser_1 = PlaneWaveLaserBeam()
laser_1.direction = (0, 0, 1)
laser_1.set_power_from_I(main.Isat)  # set power to reach Isat
laser_1.tag = "las1"

laser_2 = laser_1.copy()  # create a copy
laser_2.direction = (0, 0, -1)  # propagating in opposite direction
laser_2.tag = "las2"

# - config
config = Configuration()
config.atom = atom
config += laser_1, laser_2
config.add_atomlight_coupling("las1", "main", -0.5 * main.Gamma)
config.add_atomlight_coupling("las2", "main", -0.5 * main.Gamma)

# - simulation
sim = ScipyIVP_3D(config=config)
t = np.linspace(0, 0.1, 1000)  # timesteps for integration
u0 = (0, 0, 0, 0, 0, 100)  # atom starts with vz=100m/s
res = sim.integrate(u0, t)

# plot
fix, axes = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
axes[0].plot(res.t * 1e3, res.y[2])
axes[0].set_ylabel("z (m)")
axes[1].plot(res.t * 1e3, res.y[5])
axes[1].set_ylabel("vz (m/s)")
for ax in axes:
    ax.set_xlabel("t (ms)")
    ax.grid()
plt.show()
