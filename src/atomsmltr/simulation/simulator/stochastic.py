"""Home-made stochastic integrators
=========================================

Implements homemade integrators for stochastic systems, that is, taking into account
fluctuations due to photon scattering
"""

# % IMPORTS
import numpy as np
import scipy.constants as csts
from functools import partial

# % LOCAL IMPORTS
from .simbase import Simulation, SimRes, get_force_vec
from .deterministic import CustomSimulationBase
from ..configurator import Configuration


# % USEFUL FUNCTIONS
def random_unit_vector(shape=(1,)):
    """Generates a random unit vector

    Parameters
    ----------
    shape : tuple, optional
        shape of the output will be (**shape, 3), by default (1,)

    Returns
    -------
    vec : array
        the random unit vector
    """
    # - get random phi and costheta
    rng = np.random.default_rng()
    phi = rng.uniform(low=0, high=2 * np.pi, size=shape)
    costheta = rng.uniform(low=-1, high=1, size=shape)
    sintheta = np.sqrt(1 - costheta**2)
    # - compute x, y, z
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = costheta
    # - combine into an array of good shape
    vec = np.array([x.T, y.T, z.T]).T
    return vec


# % HOME-MADE SIMULATORS


class RK4_spontem(CustomSimulationBase):
    """A homemade simulator based on fourth order Runge-Kutta method, taking into
    account spontaneous emission.

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation

    References
    ----------
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    """

    def __init__(
        self,
        config: Configuration = None,
    ):
        super(RK4_spontem, self).__init__(config)

    def dudt_fluct(self, t, u, dt):
        _, scatt_list = get_force_vec(u, self.config, return_list=True)
        F = np.zeros_like(u[..., :3])
        rng = np.random.default_rng()
        for scatt in scatt_list:
            rate = scatt["rate"]  # scattering rate
            k = scatt["k"]  # laser wavenumber
            Ni = rate * dt  # number of scattered photons
            sigma_F = (
                np.sqrt(Ni / 3.0) * csts.hbar * k / dt
            )  # std deviation of random force
            dF = np.asanyarray(rng.normal(loc=0, scale=sigma_F))
            direction = random_unit_vector(shape=u.shape[:-1])
            F = F + dF[..., np.newaxis] * direction
        dx, dy, dz = np.zeros_like(F.T)
        dvx, dvy, dvz = F.T / self.config.atom.mass
        res = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return res

    def _iterate(self, t, u, dt):
        """returns the evolution du of u between t and t+dt
        Here we use the fourth order Runge-Kutta method
        """
        # perform step
        # 1) deterministic part
        k1 = self.dudt(t, u)
        k2 = self.dudt(t + 0.5 * dt, u + 0.5 * k1 * dt)
        k3 = self.dudt(t + 0.5 * dt, u + 0.5 * k2 * dt)
        k4 = self.dudt(t + dt, u + k3 * dt)
        du = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 2) fluctating part
        du_fluct = dt * self.dudt_fluct(t, u, dt)
        return du + du_fluct
