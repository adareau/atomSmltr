def test_multifreq_laser_scattering_rate():
    from atomsmltr.simulation import Configuration, RK4
    from atomsmltr.atoms.collection import Ytterbium
    from atomsmltr.environment.lasers.beams import GaussianLaserBeam
    import numpy as np

    # -- init config
    config_1 = Configuration()
    config_2 = Configuration()
    # -- set atom
    for c in [config_1, config_2]:
        c.atom = Ytterbium()
    main = config_1.atom.trans["main"]
    # -- init a laser
    power = 10e-3
    laser399 = GaussianLaserBeam(399e-9, 100e-6, power, (0, 0, 0), (0, 0, 1), tag="399")

    # -- prepare a list of detunings
    detuning_list = [(-0.5 * main.Gamma, 0.5), (-main.Gamma, 1.0), (0, 5.0)]

    # -- setup configs
    # config 1 : with multifreq
    config_1 += laser399
    config_1.add_atomlight_coupling("399", "main", detuning_list)

    # config 2 : add laser manually
    for i, cpl in enumerate(detuning_list):
        detuning, weight = cpl
        laser399.tag = f"399-{i}"
        laser399.power = power * weight
        config_2 += laser399
        config_2.add_atomlight_coupling(laser399, "main", detuning)

    # -- prepare simulation objects
    sim1 = RK4(config_1)
    sim2 = RK4(config_2)

    # -- compare forces
    # check at center
    u = np.array([0, 0, 0, 0, 0, 0])
    F1 = sim1.get_force(u)
    F2 = sim2.get_force(u)
    assert np.allclose(F1, F2)

    # check at random positions
    u = np.random.uniform(-0.1, 0.1, (5, 5, 5, 5, 6))
    F1 = sim1.get_force(u)
    F2 = sim2.get_force(u)
    assert np.allclose(F1, F2)


if __name__ == "__main__":
    test_multifreq_laser_scattering_rate()
