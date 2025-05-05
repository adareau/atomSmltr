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
from .deterministic import stop_position_event, stop_speed_event
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


class RK4_spontem(Simulation):
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

    def get_force(self, u):
        # NOTE : here we only keep the mean force, i.e. not the stochastic part
        # this is because this function is also used to plot the force field map
        force = get_force_vec(u, self.config)
        return force

    def dudt(self, t, u):
        F = self.get_force(u)
        _, _, _, vx, vy, vz = u.T
        dx, dy, dz = vx, vy, vz
        dvx, dvy, dvz = F.T / self.config.atom.mass
        res = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return res

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

    def _integrate(self, u0, t):
        # - u0 to array
        u = np.asanyarray(u0)
        # - get stop events
        events = []
        stop_position, stop_speed = self.config.get_stop_zones()
        if stop_position:
            stop_pos = partial(stop_position_event, stop_position=stop_position)
            events.append(stop_pos)
        if stop_speed:
            stop_sp = partial(stop_speed_event, stop_speed=stop_speed)
            events.append(stop_sp)
        # - time
        # TODO : add checks on time
        t = np.asanyarray(t)
        t = np.sort(t)
        dt = np.diff(t)
        # - initialize
        y = np.empty((*u.shape, len(t)))
        y[..., 0] = u
        stop = False
        # - integrate
        i = 1
        u_none = np.full((6,), np.nan)
        for i, (tt, h) in enumerate(zip(t[1:], dt)):
            # check events
            if events:
                for ev in events:
                    test = ev(u)
                    u[np.logical_not(test), :] = u_none
                    if not np.any(test):
                        stop = True
            if stop:
                break

            # perform step
            # 1) deterministic part
            k1 = self.dudt(tt, u)
            k2 = self.dudt(tt + 0.5 * h, u + 0.5 * k1 * h)
            k3 = self.dudt(tt + 0.5 * h, u + 0.5 * k2 * h)
            k4 = self.dudt(tt + h, u + k3 * h)
            u_new = u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 2) stochastic part
            # TODO here
            du = self.dudt_fluct(tt, u, h)
            u_new += du * h

            # store and iterate
            u = u_new
            y[..., i + 1] = u_new

        if stop:
            y = y[..., : i + 1]
            t = t[: i + 1]

        res = SimRes(t=t, y=y)

        return res

    def _u0_list_checker(self, value):
        if not hasattr(value, "__iter__"):
            raise ValueError("'u0_list' should be an iterable object")
        if value:
            for u0 in value:
                if np.asanyarray(u0).shape != (6,):
                    raise ValueError("'u0_list' should be a list of arrays of size 6")
        return value
