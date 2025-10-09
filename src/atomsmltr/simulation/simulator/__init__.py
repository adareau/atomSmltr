"""simulator
==================

Here we implement the ``Simulation`` class, that allows to run simulations
on a configuration defined via the ``Configuration`` class.

Examples
--------------------

Run a simulation with one initial condition vector

.. code-block:: python

        # ... init a config object with the `Configuration` class

        # - import a simulation class
        from atomsmltr.simulation import ScipyIVP_3D

        # - init and setup
        sim = ScipyIVP_3D(method="Radau")
        sim.config = config

        # - parameters
        u0 = (0, 0, -0.15, 0, 0, 200)
        t = np.linspace(0, 0.05, 1000)

        # - integrate
        res = sim.integrate(u0, t)

Run a batch of simulations

.. code-block:: python

        # ... init a config object with the `Configuration` class

        # - import a simulation class
        from atomsmltr.simulation import ScipyIVP_3D

        # - init and setup
        sim = ScipyIVP_3D(method="Radau")
        sim.config = config

        # - parameters
        # initial conditions
        vz_list = np.linspace(10, 300, 40)
        u0_list = [(0, 0, -0.15, 0, 0, v) for v in vz_list]
        sim.u0_list = u0_list
        # time
        t = np.linspace(0, 0.05, 1000)

        # - run a batch in parallel
        res_list = sim.run(t, npools=5, verbose=True)

"""

__all__ = [
    "Simulation",
    "ScipyIVP_3D",
    "RK4",
    "RK4St",
    "Euler",
    "EulerSt",
    "VelocityVerlet",
]

from .simbase import Simulation
from .scipy import ScipyIVP_3D
from .deterministic import RK4, Euler, VelocityVerlet
from .stochastic import RK4St, EulerSt
