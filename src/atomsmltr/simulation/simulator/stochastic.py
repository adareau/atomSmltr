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


class RK4St(CustomSimulationBase):
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
        super(RK4St, self).__init__(config)
        self.rng = np.random.default_rng()

    def du_fluct(self, t, u, dt):
        _, scatt_list = get_force_vec(u, self.config, return_list=True)
        dv_tot = np.zeros_like(u[..., :3])
        for scatt in scatt_list:
            # 0 - get scattering rate, laser wavenumber and unit vector for each laser
            rate = scatt["rate"]  # scattering rate
            k = scatt["k"]  # laser wavenumber
            u = scatt["unit_vector"]  # laser unit vector
            Ni = rate * dt  # number of scattered photons
            m = self.config.atom.mass
            # 1 - absorption fluctuation
            # large number of photon approximation
            #   > fluctuation are Gaussian with std = np.sqrt(Ni)
            # note that dN has the same shape as Ni, and can be an array !!
            dN = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni)))
            dv_abs = (csts.hbar * k / m) * dN[..., np.newaxis] * u
            dv_tot = dv_tot + dv_abs
            # 2 - emission fluctuation
            # Gaussian approx for random walk, with std = sqrt(Ni/3) for x, y, z
            dNx = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni / 3)))
            dNy = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni / 3)))
            dNz = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni / 3)))
            dN = np.array([dNx.T, dNy.T, dNz.T]).T
            dv_em = (csts.hbar * k / m) * dN
            dv_tot = dv_tot + dv_em

        dx, dy, dz = np.zeros_like(dv_tot.T)
        dvx, dvy, dvz = dv_tot.T
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
        du_fluct = self.du_fluct(t, u, dt)
        return du + du_fluct


class EulerSt(CustomSimulationBase):
    """A homemade simulator based on simple Euler integration method, taking into
    account spontaneous emission.

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation

    enable_fluct : Boolean, optional (default=True)
        if set to True, fluctations are enabled. Otherwise, the simulator boils
        down to a deterministic Euler simulator

    References
    ----------
    TODO: put here

    """

    def __init__(
        self,
        config: Configuration = None,
        enable_fluct: bool = True,
    ):
        super(EulerSt, self).__init__(config)
        self.enable_fluct = enable_fluct
        self.rng = np.random.default_rng()

    def du_fluct(self, t, u, dt):
        _, scatt_list = get_force_vec(u, self.config, return_list=True)
        dv_tot = np.zeros_like(u[..., :3])
        for scatt in scatt_list:
            # 0 - get scattering rate, laser wavenumber and unit vector for each laser
            rate = scatt["rate"]  # scattering rate
            k = scatt["k"]  # laser wavenumber
            u = scatt["unit_vector"]  # laser unit vector
            Ni = rate * dt  # number of scattered photons
            m = self.config.atom.mass
            # 1 - absorption fluctuation
            # large number of photon approximation
            #   > fluctuation are Gaussian with std = np.sqrt(Ni)
            # note that dN has the same shape as Ni, and can be an array !!
            dN = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni)))
            dv_abs = (csts.hbar * k / m) * dN[..., np.newaxis] * u
            dv_tot = dv_tot + dv_abs
            # 2 - emission fluctuation
            # Gaussian approx for random walk, with std = sqrt(Ni/3) for x, y, z
            dNx = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni / 3)))
            dNy = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni / 3)))
            dNz = np.asanyarray(self.rng.normal(loc=0, scale=np.sqrt(Ni / 3)))
            dN = np.array([dNx.T, dNy.T, dNz.T]).T
            dv_em = (csts.hbar * k / m) * dN
            dv_tot = dv_tot + dv_em

        dx, dy, dz = np.zeros_like(dv_tot.T)
        dvx, dvy, dvz = dv_tot.T
        res = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return res

    def _iterate(self, t, u, dt):
        """returns the evolution du of u between t and t+dt
        Here we use the fourth order Runge-Kutta method
        """
        # perform step
        # 1) deterministic part
        F = self.get_force(u)
        _, _, _, vx, vy, vz = u.T
        dvx, dvy, dvz = F.T / self.config.atom.mass * dt
        dx = vx * dt + 0.5 * dvx * dt
        dy = vy * dt + 0.5 * dvy * dt
        dz = vz * dt + 0.5 * dvz * dt
        du = np.array([dx, dy, dz, dvx, dvy, dvz]).T

        # 2) fluctating part
        if self.enable_fluct:
            du += self.du_fluct(t, u, dt)
        return du
