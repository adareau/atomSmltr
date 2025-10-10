def _test_Dopper_temp(simulator, plot=False):
    # -- IMPORTS
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import constants as csts
    from atomsmltr.simulation import Configuration
    from atomsmltr.atoms.collection import Ytterbium
    from atomsmltr.environment import PlaneWaveLaserBeam

    # -- CONFIG
    # - settings
    main = Ytterbium().trans["main"]
    s0 = 0.05  # I/I_sat
    delta = -0.5 * main.Gamma
    l399 = {
        "wavelength": 399e-9,
        "waist": 22e-3,
        "power": 100e-3 / 6,
        "waist_position": (0, 0, 0),
    }
    # - prepare laser list
    lasers = {}
    for axis, direction in zip(["x", "y", "z"], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        for head, mult in zip([">", "<"], [1, -1]):
            dir = np.array(direction) * mult
            tag = axis + head
            laser = PlaneWaveLaserBeam(**l399)
            laser.direction = dir
            laser.tag = tag
            laser.set_power_from_I(s0 * main.Isat)
            lasers[tag] = laser
    # - add to config
    config = Configuration(atom=Ytterbium())
    config += [*lasers.values()]
    for laser in config.list_lasers():
        config.add_atomlight_coupling(laser=laser, transition="main", detuning=delta)

    # -- SIMULATION
    sim = simulator(config)
    u0 = np.zeros(shape=(2_000, 6))
    t = np.linspace(0, 0.5e-3, 500)
    res = sim.integrate(u0, t)
    v = res.y[:, 3:6, :]
    v_norm = np.linalg.norm(v, axis=1)
    T = Ytterbium().mass * np.mean(v_norm**2, axis=0) / csts.k / 3
    T_Doppler = main.get_Doppler_temperature(delta)
    Tf = T[-1]
    error = np.abs((Tf - T_Doppler) / T_Doppler)
    if plot:
        plt.figure()
        plt.plot(res.t, T * 1e6)
        plt.plot(res.t, res.t * 0 + T_Doppler * 1e6)
        plt.show()

    assert error < 0.05


def test_Doppler_temp_EulerSt(plot=False):
    from atomsmltr.simulation import EulerSt

    _test_Dopper_temp(EulerSt, plot=plot)


def test_Doppler_temp_RK4St(plot=False):
    from atomsmltr.simulation import RK4St

    _test_Dopper_temp(RK4St, plot=plot)


if __name__ == "__main__":
    # test_Doppler_temp_EulerSt(plot=True)
    test_Doppler_temp_RK4St(plot=True)
