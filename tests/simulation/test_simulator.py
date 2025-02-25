import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _init_config():
    # -- IMPORTS
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Strontium
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam
    from atomsmltr.environment.lasers.polarization import Linear
    from atomsmltr.environment.fields.magnetic import MagneticOffset, InterpMag1D1D

    # -- CONFIG
    # - init
    config = Configuration(atom=Strontium())

    # -- mag fields
    root = os.path.dirname(__file__)
    file = Path(root) / "data" / "field.dat"
    data = np.genfromtxt(file)
    data_x = data[:, 0] * 1e-2  # cm to m
    data_y = data[:, 1] * 1e-3  # mT to T
    mag_zeeman = InterpMag1D1D(
        data_x=data_x,
        data_y=data_y,
        field_direction=(1, 0, 0),
        x_direction=(0, 0, 1),
        scale=-9.0 / 11,
        tag="Zeeman",
    )
    mag_offset = MagneticOffset((0, 1e-9, 0), tag="offset")
    config.add_objects([mag_zeeman, mag_offset])

    # -- laser
    laser461 = GaussianLaserBeam(
        wavelength=461e-9,
        waist=5e-3,
        power=25e-3,
        waist_position=(0, 0, 0),
        direction=(0, 0, -1),
        direction_type="vector",
        polarization=Linear(0),
        tag="461",
    )
    config.add_objects(laser461)

    # -- ATOM-LIGHT
    main = config.atom.trans["main"]
    detuning_461 = -14 * main.Gamma
    config.add_atomlight_coupling("461", "main", detuning_461)

    return config


def test_ScipyIVP_3D_integrator():
    from atomsmltr.simulation import ScipyIVP_3D

    # - init simulation object
    sim = ScipyIVP_3D(method="Radau")
    config = _init_config()
    sim.config = config
    sim.solve_ivp_args["vectorized"] = True
    sim.solve_ivp_args["rtol"] = 1e-2

    # - parameters
    u0 = (0, 0, -0.15, 0, 0, 200)
    t = np.linspace(0, 0.05, 1000)

    # - integrate
    res = sim.integrate(u0, t)

    return res


if __name__ == "__main__":
    res = test_ScipyIVP_3D_integrator()
    x, y, z, vx, vy, vz = res.y
    plt.figure()
    plt.plot(z, vz)
    plt.show()
