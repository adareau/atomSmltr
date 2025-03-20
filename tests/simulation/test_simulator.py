import pytest
import os
import time
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
    from atomsmltr.environment.zones import LowerLimit, UpperLimit

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
        data_position=data_x,
        data_field=data_y,
        field_direction=(1, 0, 0),
        position_direction=(0, 0, 1),
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

    # -- zones
    z_max = UpperLimit(
        0.3,
        axis=2,
        target="position",
        action="stop",
        out_tag="position out",
    )
    vz_min = LowerLimit(
        0,
        axis=2,
        target="speed",
        out_tag="speed out",
    )
    config.add_objects(z_max, vz_min)

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


def test_ScipyIVP_3D_batch():
    from atomsmltr.simulation import ScipyIVP_3D

    # - init simulation object
    sim = ScipyIVP_3D(method="Radau")
    config = _init_config()
    sim.config = config

    # - batch preparation
    # checking errors
    with pytest.raises(ValueError) as excinfo:
        sim.u0_list = 0
    with pytest.raises(ValueError) as excinfo:
        sim.u0_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 1)]
    # good usage
    vz_list = np.linspace(10, 300, 40)
    u0_list = [(0, 0, -0.15, 0, 0, v) for v in vz_list]
    sim.u0_list = u0_list

    # - run batch
    t = np.linspace(0, 0.05, 1000)
    # 1) no pool, no verbose
    start_time = time.time()
    res_list = sim.run(t, npools=0, verbose=False)
    print(f"--- {time.time() - start_time:.2g} seconds ---")
    # 2) no pool, verbose
    start_time = time.time()
    res_list = sim.run(t, npools=0, verbose=True)
    print(f"--- {time.time() - start_time:.2g} seconds ---")
    # 3) pool
    start_time = time.time()
    res_list = sim.run(t, npools=3, verbose=False)
    print(f"--- {time.time() - start_time:.2g} seconds ---")
    # 3) pool / verbose
    start_time = time.time()
    res_list = sim.run(t, npools=3, verbose=True)
    print(f"--- {time.time() - start_time:.2g} seconds ---")
    # 3) larger pool
    start_time = time.time()
    res_list = sim.run(t, npools=10, verbose=False)
    print(f"--- {time.time() - start_time:.2g} seconds ---")

    return res_list


def test_RK4_integrator():
    from atomsmltr.simulation import RK4

    # - init simulation object
    sim = RK4()
    config = _init_config()
    sim.config = config

    # - parameters
    u0 = (0, 0, -0.15, 0, 0, 200)
    t = np.linspace(0, 0.05, 1000)

    # - integrate
    res = sim.integrate(u0, t)

    return res


def test_zone_tags():
    from atomsmltr.simulation import RK4, ScipyIVP_3D, Configuration
    from atomsmltr.environment import UpperLimit, LowerLimit, Limits
    from atomsmltr.atoms import Ytterbium

    # - init config
    config = Configuration(atom=Ytterbium())

    # - limits
    for axis, name in zip([0, 1, 2], ["x", "y", "z"]):
        for add, target in zip(["", "v"], ["position", "speed"]):
            min = LowerLimit(
                -1,
                axis=axis,
                target=target,
                action="ignore",
                in_tag=None,
                out_tag=f"{add}{name}<",
                tag=f"{add}{name}_min",
            )
            max = UpperLimit(
                1,
                axis=axis,
                target=target,
                action="ignore",
                in_tag=None,
                out_tag=f"{add}{name}>",
                tag=f"{add}{name}_max",
            )
            lims = Limits(
                -1,
                1,
                axis=axis,
                target=target,
                action="ignore",
                in_tag=f"{add}{name}_in",
                out_tag=None,
                tag=f"{add}{name}_lims",
            )
            config += min, max, lims

    # - test with single shots
    t = np.linspace(0, 1, 100)
    tags = {-2: "<", 0: "_in", 2: ">"}
    for SimModel in [RK4, ScipyIVP_3D]:
        for vx in [-2, 0, 2]:
            for vy in [-2, 0, 2]:
                for vz in [-2, 0, 2]:
                    u0 = (0, 0, 0, vx, vy, vz)
                    sim = SimModel(config)
                    res = sim.integrate(u0, t)
                    for v, axis in zip([vx, vy, vz], ["x", "y", "z"]):
                        tag = axis + tags[v]
                        assert tag in res.tags
                        assert "v" + tag in res.tags


if __name__ == "__main__":
    # res = test_ScipyIVP_3D_integrator()
    # res_coll = test_ScipyIVP_3D_batch()
    # res = test_RK4_integrator()
    test_zone_tags()
