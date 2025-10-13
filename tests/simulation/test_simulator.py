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


def _test_homemade_integrator(Simulator):
    # - init simulation object
    sim = Simulator()
    config = _init_config()
    sim.config = config

    # - parameters
    u0 = (0, 0, -0.15, 0, 0, 200)
    t = np.linspace(0, 0.05, 1000)

    # - integrate
    res = sim.integrate(u0, t)

    # - check vectorization
    grid = np.mgrid[0:0:1j, 0:0:1j, -0.15:-0.05:10j, 0:0:1j, 0:0:1j, 10:100:10]
    u0 = np.squeeze(grid.T)
    res = sim.integrate(u0, t)

    return res


def test_RK4_integrator():
    from atomsmltr.simulation import RK4

    res = _test_homemade_integrator(RK4)
    return res


def test_RK4St_integrator():
    from atomsmltr.simulation import RK4St

    res = _test_homemade_integrator(RK4St)
    return res


def test_EulerSt_integrator():
    from atomsmltr.simulation import EulerSt

    res = _test_homemade_integrator(EulerSt)
    return res


def test_VelocityVerlet_integrator():
    from atomsmltr.simulation import VelocityVerlet

    res = _test_homemade_integrator(VelocityVerlet)
    return res


def test_Euler_integrator():
    from atomsmltr.simulation import Euler

    res = _test_homemade_integrator(Euler)
    return res


def test_force_integration():
    from atomsmltr.simulation import (
        RK4,
        Euler,
        VelocityVerlet,
        ScipyIVP_3D,
        Configuration,
    )
    from atomsmltr.environment import ConstantForce
    from atomsmltr.atoms import Strontium
    import numpy as np

    # - init config
    config = Configuration(atom=Strontium())
    # include gravity
    g = 9.81
    m = Strontium().mass
    g_force = (0, 0, -m * g)  # along -z
    gravity = ConstantForce(field_value=g_force, tag="gravity")
    config += gravity

    # - simulate
    t = np.linspace(0, 1, 1000)
    u0 = np.zeros((6,))

    for Sim in [RK4, ScipyIVP_3D, VelocityVerlet, Euler]:
        res = Sim(config).integrate(u0, t)
        z = res.y[2, :]
        z_th = -0.5 * g * t**2
        error = np.std(z - z_th)
        print(f"'{Sim.__name__} >> {error:.2e}'")
        assert error < 1e-8, f"Error with simulator '{Sim.__name__}'"


def test_zone_tags():
    from atomsmltr.simulation import RK4, ScipyIVP_3D, Configuration
    from atomsmltr.environment import UpperLimit, LowerLimit, Limits
    from atomsmltr.atoms import Ytterbium

    # - init config
    config = Configuration(atom=Ytterbium())

    # 1) TEST WITHOUT STOPPING
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

    # - test with vectors
    grid = np.mgrid[0:0:1j, 0:0:1j, 0:0:1j, -2:2:3j, -2:2:3j, -2:2:3j]
    u0 = np.squeeze(grid.T)
    for SimModel in [RK4]:
        sim = SimModel(config)
        res = sim.integrate(u0, t)
        assert res.tags.shape == u0.shape[:-1]
        tags_flat = res.tags.reshape((-1))
        u0_flat = u0.reshape((-1, 6))
        for u, tg in zip(u0_flat, tags_flat):
            _, _, _, vx, vy, vz = u
            for v, axis in zip([vx, vy, vz], ["x", "y", "z"]):
                tag = axis + tags[v]
                assert tag in tg
                assert "v" + tag in tg

    # 1) TEST WITH STOPPING
    # - limits
    xlim = Limits(
        -1,
        1,
        axis=0,
        target="position",
        action="stop",
        tag="xlim",
        in_tag="x_in",
        out_tag="x_out",
    )
    ylim = Limits(
        -1,
        1,
        axis=1,
        target="position",
        action="stop",
        tag="ylim",
        in_tag="y_in",
        out_tag="y_out",
    )
    config.rm_all_zones()
    config += xlim, ylim
    # - test with vectors, with different stop times
    grid = np.mgrid[0:0:1j, 0:0:1j, 0:0:1j, -2:2:2j, -3:3:5j, -1:1:4j]
    u0_grid = np.squeeze(grid.T)
    t = np.linspace(0, 1, 100)
    for SimModel in [RK4]:
        sim = SimModel(config)
        res = sim.integrate(u0_grid, t)
        assert res.tags.shape == u0_grid.shape[:-1]
        # - reshape
        # trajectories vector
        y = res.y.T
        y = y.reshape((len(res.t), 6, -1))
        y = y.T
        # last vector
        y_last = res.y_last
        y_last = y_last.T
        y_last = y_last.reshape(6, -1)
        y_last = y_last.T
        # last tags
        tags = res.tags
        tags = tags.T
        tags = tags.reshape(-1)
        tags = tags.T
        # scan over all trajectories
        for u, uf, tg in zip(y, y_last, tags):
            # check that last value corresponds to the one
            # given by the simulation (res.y_last)
            (i,) = np.where(np.isnan(u[0, :]))
            if len(i):
                i = np.min(i)
                last = u[:, i - 1]
            else:
                last = u[:, -1]
            assert np.allclose(last, uf)

            # check that the tags are fine
            u0 = u[:, 0]
            _, _, _, vx, vy, _ = u0
            if np.abs(vx) > np.abs(vy):
                assert "x_out" in tg
                assert "x_in" not in tg
                assert "y_in" in tg
                assert "y_out" not in tg
            else:
                assert "x_out" not in tg
                assert "x_in" in tg
                assert "y_in" not in tg
                assert "y_out" in tg


if __name__ == "__main__":
    # res = test_ScipyIVP_3D_integrator()
    # res_coll = test_ScipyIVP_3D_batch()
    # res = test_RK4_integrator()
    # res = test_RK4St_integrator()
    res = test_EulerSt_integrator()
    # test_zone_tags()
    # test_force_integration()
