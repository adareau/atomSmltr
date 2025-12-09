
# atomSmltr ⚛️ simulating laser cooling & trapping


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Read the Docs](https://img.shields.io/readthedocs/atomsmltr)](https://atomsmltr.readthedocs.io)
[![PyPI - Version](https://img.shields.io/pypi/v/atomsmltr)](https://pypi.org/project/atomsmltr/)
-----------

<a href="https://github.com/adareau/atomSmltr"><img src="https://raw.githubusercontent.com/adareau/atomSmltr/main/docs/_static/images/atomsmltr_logo_name.svg" alt="atomsmtlr logo" height="60"></a>

**atomSmltr** is a _user-friendly_ and _modular_ python package designed to **simulate laser cooling in complex magnetic field and laser
geometries**.

---

🚨 **Disclaimer:** this package is still under active development.

---
[**📃 Full documentation**](https://atomsmltr.readthedocs.io)  | [**🐍 PyPi**](https://pypi.org/project/atomsmltr/)




## 🚀 Installation notes

### install latest stable release

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

``atomsmltr`` is still under active development, and we would be more than happy to welcome contributions!

We encourage anyone willing to contribute to first have a look at the global architecture underlying ``atomsmtlr``. Note that it is coded in a modular way.

The easiest way to contribute would then be to expand the collection of Environment objects (magnetic field profiles, laser beam types, etc.) or to add new types of integrators in ``simulation.simulator``.

Contributions to the core structure of ``atomsmltr`` would also be welcome, but should be discussed with the main development team to ensure a good coordination of efforts.

In all cases, please make sure that you comply with our coding standards:

+ comment your code: always include docstrings, add comments to tricky code parts, etc.
+ add unit tests: whenever adding a new feature, please make sure that you write a corresponding unit test in the ``tests`` folder.
+ use black for code formatting.
+ use poetry for dependency management.
