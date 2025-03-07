
# atomSmltr - atomSimulator

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A user-friendly python package to perform semi-classical atomic physics simulations.

**Attention:** this package is still under active development.

## 🚀 Installation notes

### install latest stable release
🚨 **Not implemented yet > when we go public** 🚨

just pip it
```
pip install atomsmtlr
```

### install current version
In our git development workflow, we have three main branches: `main` for stable relases, `testing` for development versions that should work most of the time and `devel` for the implementation of new, more experimental features. If you want to benefit from the latest features, we encourage you to use the `testing` branch.

First, clone the repository

```
git clone https://github.com/adareau/atomSmltr.git
cd atomSmltr
git checkout testing
```

We strongly encourage you to use a virtual environment

```
python3 -m venv __venv__
source ./__venv__/bin/activate
```

Then you can use pip

```
pip install .
```

That's it, you should be good to go !!


## ✨ Try `atomsmtlr`

Below is a minimal example. You should also check the examples in the provided documentations, that contains a collection of jupyter notebooks that will guide you into using `atomsmtlr`.

⏩ checkout [`./docs/_notebook_examples`](./docs/_notebook_examples/)

```python
""" Minimal example for testing atomstmltr"""

import numpy as np
import matplotlib.pyplot as plt

from atomsmltr.environment import PlaneWaveLaserBeam
from atomsmltr.atoms import Ytterbium
from atomsmltr.simulation import Configuration, ScipyIVP_3D

# - setup atom
atom = Ytterbium()
main = atom.trans["main"] # get transition, to help setting up lasers

# - setup laser
laser_1 = PlaneWaveLaserBeam()
laser_1.direction = (0, 0, 1)
laser_1.set_power_from_I(main.Isat) # set power to reach Isat
laser_1.tag = "las1"

laser_2 = laser_1.copy() # create a copy
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

```


## 🐍 How to contribute
TODO: add information on tools we use (poetry, documentation, etc..)
