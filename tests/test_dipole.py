# tests/test_dipole.py
import numpy as np
import scipy.constants as csts
from atomsmltr.environment import GaussianLaserBeam
from atomsmltr.atoms import Ytterbium
from atomsmltr.simulation import Configuration
from atomsmltr.simulation.simulator.simbase import get_force_vec


def make_config():
    atom = Ytterbium()
    laser = GaussianLaserBeam(tag="odt")
    laser.direction = (0, 0, 1)
    laser.waist = 50e-6
    laser.set_power_from_I(5)
    config = Configuration()
    config.atom = atom
    config += laser
    omega_laser = 2 * np.pi * csts.c / 1064e-9
    config.add_dipole_coupling("odt", omega_laser)
    return config


def test_force_zero_on_axis():
    """Dipole force must be zero on the beam axis (intensity maximum, zero gradient)"""
    config = make_config()
    # point on axis: x=0, y=0, z=0
    u = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    force = get_force_vec(u, config)
    Fx = force[0, 0]
    Fy = force[0, 1]
    assert abs(Fx) < 1e-30, f"Fx on axis should be ~0, got {Fx}"
    assert abs(Fy) < 1e-30, f"Fy on axis should be ~0, got {Fy}"


def test_force_restoring_red_detuned():
    """For a red-detuned beam, force should point toward axis (restoring)"""
    config = make_config()
    # point off axis: x = +20um
    u_pos = np.array([[20e-6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    u_neg = np.array([[-20e-6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    F_pos = get_force_vec(u_pos, config)[0, 0]
    F_neg = get_force_vec(u_neg, config)[0, 0]
    assert F_pos < 0, f"Force at x>0 should be negative (restoring), got {F_pos}"
    assert F_neg > 0, f"Force at x<0 should be positive (restoring), got {F_neg}"


def test_force_antisymmetric():
    """Force profile must be antisymmetric: F(x) = -F(-x)"""
    config = make_config()
    x = np.linspace(-100e-6, 100e-6, 101)
    pos_speed = np.zeros((101, 6))
    pos_speed[:, 0] = x
    force = get_force_vec(pos_speed, config)
    Fx = force[:, 0]
    assert np.allclose(Fx, -Fx[::-1], rtol=1e-5), "Force should be antisymmetric"
